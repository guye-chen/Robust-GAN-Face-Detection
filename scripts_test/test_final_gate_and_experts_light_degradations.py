"""
Light degradation test for the final weak/strong gate and its two experts.

This script does not train anything. It evaluates:
  1. Weak expert: Stability-Reg
  2. Strong expert: High-Noise Bias-Calib w/o high-noise gate
  3. Final model: Frequency-Reliability Weak-Strong Gate v2

The degradation settings are intentionally matched to
test_light_degradation_selected_models.py:
  clean, jpeg90, jpeg70, blur1, blur2, downsample0.75, downsample0.5
"""

import os
import sys
import csv
import hashlib
import argparse
from io import BytesIO
from collections import Counter, OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from PIL import Image, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader.ffhq_gan_dataset import FFHQGANData, pil_loader
from model.frequency_reliability_weak_strong_gate import (
    FEATURE_NAMES,
    frequency_reliability_weak_strong_gate,
    compute_frequency_reliability_features,
)
from model.inceptionv3_nmfgf_noise_ra_v1_legacy import inceptionv3_nmfgf_noise_ra_v1
from model.inceptionv3_high_noise_bias_calib import inceptionv3_high_noise_bias_calib


REAL_ROOT = r"E:\FFHQ_off"
FAKE_ROOT = r"E:\archive\GAN_Generated_Fake_Images\fake"
OUT_DIR = (
    r"E:\cbam_ftt_fusion\CBAM.PyTorch-master\weights"
    r"\final_gate_and_experts_light_degradations"
)

GATE_CHECKPOINT = (
    r"E:\cbam_ftt_fusion\CBAM.PyTorch-master\weights"
    r"\noise-aware-weak-strong-gate\best_weak_strong_gate.pth"
)
WEAK_EXPERT_CHECKPOINT = (
    r"E:\cbam_ftt_fusion\CBAM.PyTorch-master\weights"
    r"\inceptionv3-nmfgf-noise-ra-stability-reg\best_model.pth"
)
STRONG_EXPERT_CHECKPOINT = (
    r"E:\cbam_ftt_fusion\CBAM.PyTorch-master\weights"
    r"\inceptionv3-high-noise-bias-calib-no-high-noise-gate\best_model.pth"
)

# ==============================
# Click-run default configuration
# ==============================
CLICK_RUN_BATCH_SIZE = 1
CLICK_RUN_NUM_WORKERS = 0
CLICK_RUN_GPU = "0"


DEGRADATION_MODES = [
    ("clean", None, None),
    ("jpeg90", "jpeg", 90),
    ("jpeg70", "jpeg", 70),
    ("blur1", "blur", 1.0),
    ("blur2", "blur", 2.0),
    ("downsample0.75", "downsample", 0.75),
    ("downsample0.5", "downsample", 0.5),
]

EXPERT_NAMES = ["stability_reg", "high_noise_bias_calib_no_high_noise_gate"]


def jpeg_degrade(img, quality):
    buffer = BytesIO()
    img = img.convert("RGB")
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def blur_degrade(img, sigma):
    img = img.convert("RGB")
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))


def downsample_degrade(img, scale):
    img = img.convert("RGB")
    w, h = img.size
    new_w, new_h = int(w * scale), int(h * scale)
    img_small = img.resize((new_w, new_h), Image.BILINEAR)
    return img_small.resize((w, h), Image.BILINEAR)


class LightDegradationDataset:
    """Apply the same PIL-level light degradations used in prior robustness tests."""

    def __init__(self, dataset, base_transform, degrade_type, degrade_param, base_seed=42):
        self.dataset = dataset
        self.base_transform = base_transform
        self.degrade_type = degrade_type
        self.degrade_param = degrade_param
        self.base_seed = int(base_seed)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        path, label, _ = self.dataset.samples[index]
        img = pil_loader(path)

        if self.degrade_type == "jpeg":
            img = jpeg_degrade(img, self.degrade_param)
        elif self.degrade_type == "blur":
            img = blur_degrade(img, self.degrade_param)
        elif self.degrade_type == "downsample":
            img = downsample_degrade(img, self.degrade_param)
        elif self.degrade_type == "noise":
            key = "{}|{}|{}|{}".format(path, self.degrade_type, self.degrade_param, self.base_seed)
            seed = int(hashlib.md5(key.encode()).digest()[:4].hex(), 16)
            rng = np.random.default_rng(seed)
            arr = np.array(img.convert("RGB")).astype(np.float32)
            noise = rng.normal(0, float(self.degrade_param), arr.shape).astype(np.float32)
            img = Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))

        return self.base_transform(img), int(label)


class StabilityRegInferenceWrapper(nn.Module):
    """Use the Stability-Reg checkpoint with cls_logits as the decision output."""

    def __init__(self, base_model):
        super(StabilityRegInferenceWrapper, self).__init__()
        self.base_model = base_model

    def forward(self, x):
        _, cls_logits, _, _ = self.base_model(x, return_aux=False)
        return cls_logits


class FinalWeakStrongGateWrapper(nn.Module):
    """Final gate v2: route each image to weak or strong expert by frequency features."""

    def __init__(self, gate, weak_expert, strong_expert):
        super(FinalWeakStrongGateWrapper, self).__init__()
        self.gate = gate
        self.weak_expert = weak_expert
        self.strong_expert = strong_expert
        self.last_selection_counts = Counter()
        self.last_p_strong = []

    def reset_diagnostics(self):
        self.last_selection_counts = Counter()
        self.last_p_strong = []

    def forward(self, x):
        gate_features = compute_frequency_reliability_features(x)
        gate_logits = self.gate(gate_features)
        gate_probs = F.softmax(gate_logits, dim=1)
        gate_pred = torch.argmax(gate_logits, dim=1)

        weak_logits = extract_logits(self.weak_expert(x))
        strong_logits = extract_logits(self.strong_expert(x))
        stacked = torch.stack([weak_logits, strong_logits], dim=1)

        batch_idx = torch.arange(x.size(0), device=x.device)
        final_logits = stacked[batch_idx, gate_pred]

        for expert_id in gate_pred.detach().cpu().tolist():
            self.last_selection_counts[EXPERT_NAMES[int(expert_id)]] += 1
        self.last_p_strong.extend(gate_probs[:, 1].detach().cpu().tolist())
        return final_logits


def checkpoint_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"], "model_state_dict"
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"], "state_dict"
    if isinstance(checkpoint, dict):
        return checkpoint, "raw state_dict"
    raise RuntimeError("Unsupported checkpoint format: {}".format(type(checkpoint)))


def freeze_eval(model):
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def extract_logits(outputs):
    """Return the classification logits from model outputs used in this project."""
    if isinstance(outputs, tuple):
        return outputs[0]
    if hasattr(outputs, "logits"):
        return outputs.logits
    return outputs


def load_strict(model, checkpoint_path, device, name, load_target=None):
    target = load_target if load_target is not None else model
    report = {
        "name": name,
        "checkpoint": checkpoint_path,
        "exists": os.path.exists(checkpoint_path),
        "strict": False,
        "loaded_state_dict_keys": 0,
        "missing_keys_count": None,
        "unexpected_keys_count": None,
        "format": "",
    }
    print("Loading {} checkpoint: {}".format(name, checkpoint_path))
    print("  exists: {}".format(report["exists"]))
    if not report["exists"]:
        raise RuntimeError("{} checkpoint not found: {}".format(name, checkpoint_path))

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict, fmt = checkpoint_state_dict(checkpoint)
    report["format"] = fmt

    clean_state = OrderedDict()
    for key, value in state_dict.items():
        clean_key = key[7:] if key.startswith("module.") else key
        clean_state[clean_key] = value

    target.load_state_dict(clean_state, strict=True)
    report["strict"] = True
    report["loaded_state_dict_keys"] = len(clean_state)
    report["missing_keys_count"] = 0
    report["unexpected_keys_count"] = 0
    print("  strict=True: OK")
    print("  loaded state_dict keys: {}".format(len(clean_state)))
    return report


def build_weak_expert(device):
    base_model = inceptionv3_nmfgf_noise_ra_v1(
        num_classes=2,
        pretrained_backbone=False,
        alpha_init=0.02,
        ra_loss_weight=0.005,
        dense_dim=16,
        mlp_hidden=64,
        sigma_train=[5.0, 10.0, 15.0, 25.0],
        sigma_eval=10.0,
        transform_input=True,
    ).to(device)
    wrapper = StabilityRegInferenceWrapper(base_model).to(device)
    report = load_strict(
        wrapper,
        WEAK_EXPERT_CHECKPOINT,
        device,
        "stability_reg",
        load_target=base_model,
    )
    return freeze_eval(wrapper), report


def build_strong_expert(device):
    model = inceptionv3_high_noise_bias_calib(
        num_classes=2,
        pretrained_backbone=False,
        alpha_init=0.02,
        ra_loss_weight=0.005,
        dense_dim=16,
        mlp_hidden=64,
        sigma_train=[25.0, 35.0, 50.0],
        sigma_eval=50.0,
        transform_input=True,
        gamma_init=0.02,
        gamma_max=0.20,
        gate_tau=0.55,
        gate_k=10.0,
        use_high_noise_gate=False,
    ).to(device)
    report = load_strict(
        model,
        STRONG_EXPERT_CHECKPOINT,
        device,
        "high_noise_bias_calib_no_high_noise_gate",
    )
    report["use_high_noise_gate"] = False
    return freeze_eval(model), report


def build_gate(device):
    gate = frequency_reliability_weak_strong_gate(
        num_features=len(FEATURE_NAMES),
        hidden_dim=32,
    ).to(device)
    report = load_strict(gate, GATE_CHECKPOINT, device, "frequency_reliability_weak_strong_gate")
    return freeze_eval(gate), report


def build_models(device):
    weak_expert, weak_report = build_weak_expert(device)
    strong_expert, strong_report = build_strong_expert(device)
    gate, gate_report = build_gate(device)
    final_gate = FinalWeakStrongGateWrapper(gate, weak_expert, strong_expert).to(device)
    return [
        {
            "name": "weak_expert_stability_reg",
            "checkpoint": WEAK_EXPERT_CHECKPOINT,
            "model": weak_expert,
            "report": weak_report,
        },
        {
            "name": "strong_expert_bias_calib_no_high_noise_gate",
            "checkpoint": STRONG_EXPERT_CHECKPOINT,
            "model": strong_expert,
            "report": strong_report,
        },
        {
            "name": "final_frequency_reliability_weak_strong_gate_v2",
            "checkpoint": GATE_CHECKPOINT,
            "model": freeze_eval(final_gate),
            "report": {
                "gate": gate_report,
                "weak_expert": weak_report,
                "strong_expert": strong_report,
            },
        },
    ]


def binary_auc_score(labels, scores):
    """Compute ROC AUC for binary labels without sklearn/scipy."""
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)
    pos = labels == 1
    neg = labels == 0
    n_pos = int(np.sum(pos))
    n_neg = int(np.sum(neg))
    if n_pos == 0 or n_neg == 0:
        return 0.0

    order = np.argsort(scores)
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    i = 0
    while i < len(scores):
        j = i + 1
        while j < len(scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j

    rank_sum_pos = float(np.sum(ranks[pos]))
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def compute_metrics(labels, preds, probs):
    labels = np.asarray(labels).astype(int)
    preds = np.asarray(preds).astype(int)
    probs = np.asarray(probs).astype(float)

    tn = int(np.sum((labels == 0) & (preds == 0)))
    fp = int(np.sum((labels == 0) & (preds == 1)))
    fn = int(np.sum((labels == 1) & (preds == 0)))
    tp = int(np.sum((labels == 1) & (preds == 1)))
    total = max(1, len(labels))
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    auc = binary_auc_score(labels, probs[:, 1]) if probs.ndim == 2 and probs.shape[1] > 1 else 0.0
    return {
        "accuracy": float((tp + tn) / total),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": float(auc),
        "real_acc": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "fake_acc": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


@torch.no_grad()
def evaluate_model(model, loader, device, desc):
    model.eval()
    if hasattr(model, "reset_diagnostics"):
        model.reset_diagnostics()

    labels_all, preds_all, probs_all = [], [], []
    pbar = tqdm(loader, desc=desc, dynamic_ncols=True, leave=True)
    for inputs, labels in pbar:
        inputs = inputs.to(device)
        labels = labels.to(device)
        logits = extract_logits(model(inputs))
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)

        labels_all.extend(labels.detach().cpu().numpy())
        preds_all.extend(preds.detach().cpu().numpy())
        probs_all.extend(probs.detach().cpu().numpy())
        running_labels = np.asarray(labels_all).astype(int)
        running_preds = np.asarray(preds_all).astype(int)
        running_acc = float(np.mean(running_labels == running_preds)) if len(running_labels) else 0.0
        pbar.set_postfix({"acc": "{:.4f}".format(running_acc)})

    metrics = compute_metrics(labels_all, preds_all, probs_all)
    selection_counts = dict(model.last_selection_counts) if hasattr(model, "last_selection_counts") else {}
    p_strong_mean = float(np.mean(model.last_p_strong)) if hasattr(model, "last_p_strong") and model.last_p_strong else ""
    return metrics, selection_counts, p_strong_mean


def row_from_metrics(model_name, checkpoint, degradation, metrics, selection_counts, p_strong_mean):
    row = {
        "model": model_name,
        "checkpoint": checkpoint,
        "degradation": degradation,
        "acc": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "auc": metrics["auc"],
        "real_acc": metrics["real_acc"],
        "fake_acc": metrics["fake_acc"],
        "TN": metrics["TN"],
        "FP": metrics["FP"],
        "FN": metrics["FN"],
        "TP": metrics["TP"],
        "selection_counts": selection_counts,
        "p_strong_mean": p_strong_mean,
    }
    return row


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test final weak/strong gate and individual experts on light degradations"
    )
    parser.add_argument("--real_root", default=REAL_ROOT, type=str)
    parser.add_argument("--fake_root", default=FAKE_ROOT, type=str)
    parser.add_argument("--out_dir", default=OUT_DIR, type=str)
    parser.add_argument("--batch_size", default=CLICK_RUN_BATCH_SIZE, type=int)
    parser.add_argument("--num_workers", default=CLICK_RUN_NUM_WORKERS, type=int)
    parser.add_argument("--gpu", default=CLICK_RUN_GPU, type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 80)
    print("FINAL GATE + WEAK/STRONG EXPERTS LIGHT DEGRADATION TEST")
    print("No training. No checkpoint overwrite.")
    print("Output dir: {}".format(args.out_dir))
    print("Degradations:")
    for mode in DEGRADATION_MODES:
        print("  {}".format(mode))
    print("Gate checkpoint: {}".format(GATE_CHECKPOINT))
    print("Weak expert checkpoint: {}".format(WEAK_EXPERT_CHECKPOINT))
    print("Strong expert checkpoint: {}".format(STRONG_EXPERT_CHECKPOINT))
    print("=" * 80)

    base_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    test_dataset = FFHQGANData(args.real_root, args.fake_root, "test", transform=None)
    print("Test samples: real={}, fake={}, total={}".format(
        test_dataset.real_count,
        test_dataset.fake_count,
        len(test_dataset),
    ))

    model_specs = build_models(device)
    all_rows = []
    for spec in model_specs:
        print("\n" + "=" * 80)
        print("Testing model: {}".format(spec["name"]))
        print("Report: {}".format(spec["report"]))
        print("=" * 80)
        model = spec["model"]
        for degradation, degrade_type, degrade_param in DEGRADATION_MODES:
            eval_dataset = LightDegradationDataset(
                test_dataset,
                base_transform,
                degrade_type,
                degrade_param,
                base_seed=42,
            )
            loader = DataLoader(
                eval_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=False,
            )
            metrics, selection_counts, p_strong_mean = evaluate_model(
                model,
                loader,
                device,
                "{} {}".format(spec["name"], degradation),
            )
            row = row_from_metrics(
                spec["name"],
                spec["checkpoint"],
                degradation,
                metrics,
                selection_counts,
                p_strong_mean,
            )
            all_rows.append(row)
            print(
                "{} / {}: acc={:.4f}, precision={:.4f}, recall={:.4f}, "
                "f1={:.4f}, auc={:.4f}, real_acc={:.4f}, fake_acc={:.4f}, "
                "cm=[[{}, {}], [{}, {}]]".format(
                    spec["name"],
                    degradation,
                    row["acc"],
                    row["precision"],
                    row["recall"],
                    row["f1"],
                    row["auc"],
                    row["real_acc"],
                    row["fake_acc"],
                    row["TN"],
                    row["FP"],
                    row["FN"],
                    row["TP"],
                )
            )
            if selection_counts:
                print("  expert selection counts: {}".format(selection_counts))
                print("  p_strong_mean: {}".format(p_strong_mean))

    csv_path = os.path.join(args.out_dir, "final_gate_and_experts_light_degradations.csv")
    fieldnames = [
        "model",
        "checkpoint",
        "degradation",
        "acc",
        "precision",
        "recall",
        "f1",
        "auc",
        "real_acc",
        "fake_acc",
        "TN",
        "FP",
        "FN",
        "TP",
        "selection_counts",
        "p_strong_mean",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    txt_path = os.path.join(args.out_dir, "final_gate_and_experts_light_degradations.txt")
    sorted_rows = sorted(all_rows, key=lambda r: (r["degradation"], -r["acc"], -r["f1"]))
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Final gate and individual experts light degradation test\n")
        f.write("Degradations: {}\n".format([item[0] for item in DEGRADATION_MODES]))
        f.write("Gate checkpoint: {}\n".format(GATE_CHECKPOINT))
        f.write("Weak expert checkpoint: {}\n".format(WEAK_EXPERT_CHECKPOINT))
        f.write("Strong expert checkpoint: {}\n\n".format(STRONG_EXPERT_CHECKPOINT))
        current_deg = None
        for row in sorted_rows:
            if row["degradation"] != current_deg:
                current_deg = row["degradation"]
                f.write("\n[{}]\n".format(current_deg))
                f.write("{:<52} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8} {}\n".format(
                    "Model", "Acc", "Prec", "Recall", "F1", "AUC", "Real", "Fake", "Selection"))
            f.write("{:<52} {:<8.4f} {:<8.4f} {:<8.4f} {:<8.4f} {:<8.4f} {:<8.4f} {:<8.4f} {}\n".format(
                row["model"],
                row["acc"],
                row["precision"],
                row["recall"],
                row["f1"],
                row["auc"],
                row["real_acc"],
                row["fake_acc"],
                row["selection_counts"],
            ))

    print("\nSaved CSV: {}".format(csv_path))
    print("Saved TXT: {}".format(txt_path))
    print("\nSummary sorted by degradation, then acc/f1:")
    current_deg = None
    for row in sorted_rows:
        if row["degradation"] != current_deg:
            current_deg = row["degradation"]
            print("\n[{}]".format(current_deg))
        print("  {:<52} acc={:.4f} f1={:.4f} auc={:.4f}".format(
            row["model"],
            row["acc"],
            row["f1"],
            row["auc"],
        ))


if __name__ == "__main__":
    main()
