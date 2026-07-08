"""
Per-GAN Gaussian noise robustness evaluation for baselines and the final model.

This script does not train models. It evaluates each selected model on the same
held-out FFHQ/GAN test split under clean/noise10/noise25/noise50, then reports
fake recall separately for ProGAN, StyleGAN2, and StyleGAN3.

Outputs:
  weights/per_gan_all_models_noise/per_gan_summary.csv
  weights/per_gan_all_models_noise/per_gan_summary.txt
"""

import os
import sys
import csv
import json
import argparse
import hashlib
from collections import OrderedDict

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader.ffhq_gan_dataset import FFHQGANData, pil_loader
import test_noise_only as noise_only
from test_frequency_reliability_weak_strong_gate import (
    load_gate,
    load_experts,
    load_strict,
    freeze_eval,
    StabilityRegInferenceWrapper,
    extract_logits as extract_gate_logits,
    EXPERT_NAMES,
    STABILITY_REG_CKPT,
)
from model.frequency_reliability_weak_strong_gate import compute_frequency_reliability_features
from model.inceptionv3_nmfgf_noise_ra_v1_legacy import inceptionv3_nmfgf_noise_ra_v1
from model.inceptionv3_high_noise_bias_calib import inceptionv3_high_noise_bias_calib

try:
    import timm
    _TIMM_AVAILABLE = True
except ImportError:
    timm = None
    _TIMM_AVAILABLE = False


REAL_ROOT = r"E:\FFHQ_off"
FAKE_ROOT = r"E:\archive\GAN_Generated_Fake_Images\fake"
OUT_DIR = r"E:\cbam_ftt_fusion\CBAM.PyTorch-master\weights\per_gan_all_models_noise"

CLICK_RUN_BATCH_SIZE = 1
CLICK_RUN_NUM_WORKERS = 0
CLICK_RUN_GPU = "0"

GAN_SOURCES = ["ProGAN", "StyleGAN2", "StyleGAN3"]

DEGRADATION_MODES = [
    ("clean", None),
    ("noise10", 10),
    ("noise25", 25),
    ("noise50", 50),
]

STANDARD_MODELS = OrderedDict([
    ("resnet50v2", {
        "display": "ResNet50V2",
        "checkpoint": r"E:\cbam_ftt_fusion\CBAM.PyTorch-master\weights\best_model_resnet50v2.pth",
        "model_type": "resnet50v2",
    }),
    ("mobilenetv2", {
        "display": "MobileNetV2",
        "checkpoint": r"E:\cbam_ftt_fusion\CBAM.PyTorch-master\checkpoint\mobilenetv2\best_model.pth",
        "model_type": "mobilenetv2",
    }),
    ("densenet121", {
        "display": "DenseNet121",
        "checkpoint": r"E:\cbam_ftt_fusion\CBAM.PyTorch-master\checkpoint\densenet121\best_model.pth",
        "model_type": "densenet121",
    }),
    ("vgg16", {
        "display": "VGG16",
        "checkpoint": r"E:\cbam_ftt_fusion\CBAM.PyTorch-master\weights\best_model_vgg16.pth",
        "model_type": "vgg16",
    }),
    ("vgg19", {
        "display": "VGG19",
        "checkpoint": r"E:\cbam_ftt_fusion\CBAM.PyTorch-master\weights\best_model_vgg19.pth",
        "model_type": "vgg19",
    }),
    ("inceptionv3", {
        "display": "InceptionV3",
        "checkpoint": r"E:\cbam_ftt_fusion\CBAM.PyTorch-master\weights\best_model_inceptionv3.pth",
        "model_type": "inceptionv3",
    }),
    ("inceptionresnetv2", {
        "display": "InceptionResNetV2",
        "checkpoint": r"E:\cbam_ftt_fusion\CBAM.PyTorch-master\weights\best_model_inceptionresnetv2.pth",
        "model_type": "inceptionresnetv2",
    }),
    ("xception", {
        "display": "Xception",
        "checkpoint": r"E:\cbam_ftt_fusion\CBAM.PyTorch-master\weights\best_model_xception.pth",
        "model_type": "xception",
    }),
])

TIMM_STRONG_MODELS = OrderedDict([
    ("convnext_tiny", {
        "display": "ConvNeXt-Tiny",
        "model_name": "convnext_tiny.fb_in1k",
        "checkpoint": r"E:\cbam_ftt_fusion\CBAM.PyTorch-master\weights\convnext_tiny\best_model.pth",
    }),
    ("swin_tiny", {
        "display": "Swin-Tiny",
        "model_name": "swin_tiny_patch4_window7_224.ms_in1k",
        "checkpoint": r"E:\cbam_ftt_fusion\CBAM.PyTorch-master\weights\swin_tiny\best_model.pth",
    }),
    ("efficientnet_b3", {
        "display": "EfficientNet-B3",
        "model_name": "efficientnet_b3.ra2_in1k",
        "checkpoint": r"E:\cbam_ftt_fusion\CBAM.PyTorch-master\weights\efficientnet_b3\best_model.pth",
    }),
    ("fastvit_sa12", {
        "display": "FastViT-SA12",
        "model_name": "fastvit_sa12.apple_in1k",
        "checkpoint": r"E:\cbam_ftt_fusion\CBAM.PyTorch-master\weights\fastvit_sa12\best_model.pth",
    }),
])

FINAL_MODEL = OrderedDict([
    ("frequency_reliability_weak_strong_gate", {
        "display": "Final Weak-Strong Expert Gate",
        "checkpoint": r"E:\cbam_ftt_fusion\CBAM.PyTorch-master\weights\noise-aware-weak-strong-gate\best_weak_strong_gate.pth",
        "strong_expert": "full_high_noise_gate",
    }),
    ("frequency_reliability_weak_strong_gate_v2_no_high_noise_gate", {
        "display": "Final Weak-Strong Gate v2 (No High-Noise Gate Expert)",
        "checkpoint": r"E:\cbam_ftt_fusion\CBAM.PyTorch-master\weights\noise-aware-weak-strong-gate\best_weak_strong_gate.pth",
        "strong_expert": "no_high_noise_gate",
    })
])

DEFAULT_MODEL_KEYS = list(STANDARD_MODELS.keys()) + list(TIMM_STRONG_MODELS.keys()) + list(FINAL_MODEL.keys())


def noise_degrade(img, sigma, seed):
    img = img.convert("RGB")
    arr = np.asarray(img).astype(np.float32)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, float(sigma), arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


class PerGanNoiseDataset(Dataset):
    def __init__(self, base_dataset, base_transform, degradation_name, noise_sigma=None, base_seed=42):
        self.base_dataset = base_dataset
        self.samples = base_dataset.samples
        self.base_transform = base_transform
        self.degradation_name = degradation_name
        self.noise_sigma = noise_sigma
        self.base_seed = int(base_seed)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label, source = self.samples[index]
        img = pil_loader(path)
        if self.noise_sigma is not None:
            key = "{}|noise|{}|{}".format(path, self.noise_sigma, self.base_seed)
            seed = int(hashlib.md5(key.encode("utf-8")).digest()[:4].hex(), 16)
            img = noise_degrade(img, self.noise_sigma, seed)
        img = self.base_transform(img)
        return img, int(label), source


def make_base_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def checkpoint_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"], "model_state_dict"
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"], "state_dict"
    if isinstance(checkpoint, dict):
        return checkpoint, "raw_state_dict"
    raise RuntimeError("Unsupported checkpoint format: {}".format(type(checkpoint)))


def clean_state_dict(state_dict):
    clean = OrderedDict()
    for key, value in state_dict.items():
        clean[key[7:] if key.startswith("module.") else key] = value
    return clean


def load_timm_model(model_name, checkpoint_path, device):
    if not _TIMM_AVAILABLE:
        raise RuntimeError("timm is not installed in this Python environment.")
    model = timm.create_model(model_name, pretrained=False, num_classes=2)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict, fmt = checkpoint_state_dict(checkpoint)
    state_dict = clean_state_dict(state_dict)
    model.load_state_dict(state_dict, strict=True)
    return model.to(device), fmt, len(state_dict)


def load_standard_model(model_type, checkpoint_path, device):
    model = noise_only.build_model(model_type, num_classes=2, nmfgf_variant="V1")
    loaded = noise_only.load_model_state(model, checkpoint_path, device)
    return model.to(device), "test_noise_only.load_model_state", loaded


def load_no_high_noise_gate_experts(device):
    stability_base = inceptionv3_nmfgf_noise_ra_v1(
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
    stability_model = StabilityRegInferenceWrapper(stability_base).to(device)
    stability_report = load_strict(
        stability_model,
        STABILITY_REG_CKPT,
        device,
        "stability_reg",
        load_target=stability_base,
    )

    bias_ckpt = (
        r"E:\cbam_ftt_fusion\CBAM.PyTorch-master\weights"
        r"\inceptionv3-high-noise-bias-calib-no-high-noise-gate\best_model.pth"
    )
    bias_model = inceptionv3_high_noise_bias_calib(
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
    bias_report = load_strict(
        bias_model,
        bias_ckpt,
        device,
        "high_noise_bias_calib_no_high_noise_gate",
    )
    bias_report["use_high_noise_gate"] = False
    return [freeze_eval(stability_model), freeze_eval(bias_model)], [stability_report, bias_report]


def ensure_checkpoint(path):
    if not os.path.exists(path):
        raise RuntimeError("Checkpoint not found: {}".format(path))


def compute_overall_metrics(labels, preds, probs):
    labels = np.asarray(labels).astype(int)
    preds = np.asarray(preds).astype(int)
    probs = np.asarray(probs).astype(float)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    auc = 0.0
    if len(np.unique(labels)) > 1:
        auc = roc_auc_score(labels, probs[:, 1])
    return {
        "overall_acc": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "auc": float(auc),
        "real_acc": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def add_per_source_metrics(row, labels, preds, sources):
    labels = np.asarray(labels).astype(int)
    preds = np.asarray(preds).astype(int)
    sources = np.asarray(sources, dtype=object)

    real_mask = labels == 0
    row["real_total"] = int(real_mask.sum())
    row["real_correct"] = int(((preds == 0) & real_mask).sum())
    row["real_acc"] = row["real_correct"] / row["real_total"] if row["real_total"] else 0.0

    fake_recalls = []
    for source in GAN_SOURCES:
        mask = (labels == 1) & (sources == source)
        total = int(mask.sum())
        correct = int(((preds == 1) & mask).sum())
        recall_value = correct / total if total else 0.0
        row["{}_total".format(source)] = total
        row["{}_correct".format(source)] = correct
        row["{}_fake_recall".format(source)] = recall_value
        if total:
            fake_recalls.append(recall_value)
    row["mean_gan_fake_recall"] = float(np.mean(fake_recalls)) if fake_recalls else 0.0
    return row


def extract_logits(outputs):
    if isinstance(outputs, tuple):
        return outputs[0]
    if hasattr(outputs, "logits"):
        return outputs.logits
    return outputs


@torch.no_grad()
def evaluate_single_model(model, loader, device, desc):
    model.eval()
    labels_all, preds_all, probs_all, sources_all = [], [], [], []
    pbar = tqdm(loader, desc=desc, dynamic_ncols=True, leave=True)
    for inputs, labels, sources in pbar:
        inputs = inputs.to(device)
        logits = extract_logits(model(inputs))
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        labels_np = labels.detach().cpu().numpy()
        preds_np = preds.detach().cpu().numpy()
        probs_np = probs.detach().cpu().numpy()
        labels_all.extend(labels_np.tolist())
        preds_all.extend(preds_np.tolist())
        probs_all.extend(probs_np.tolist())
        sources_all.extend(list(sources))
        pbar.set_postfix({"acc": "{:.4f}".format(accuracy_score(labels_all, preds_all))})
    return labels_all, preds_all, probs_all, sources_all, {}


@torch.no_grad()
def evaluate_final_gate(gate, experts, loader, device, desc):
    gate.eval()
    for expert in experts:
        expert.eval()
    labels_all, preds_all, probs_all, sources_all = [], [], [], []
    selected_counts = {name: 0 for name in EXPERT_NAMES}
    pbar = tqdm(loader, desc=desc, dynamic_ncols=True, leave=True)
    for inputs, labels, sources in pbar:
        inputs = inputs.to(device)
        gate_features = compute_frequency_reliability_features(inputs)
        gate_logits = gate(gate_features)
        gate_pred = torch.argmax(gate_logits, dim=1)

        expert_logits = []
        for expert in experts:
            expert_logits.append(extract_gate_logits(expert(inputs)))
        expert_logits = torch.stack(expert_logits, dim=1)
        batch_idx = torch.arange(inputs.size(0), device=device)
        final_logits = expert_logits[batch_idx, gate_pred]

        probs = F.softmax(final_logits, dim=1)
        preds = torch.argmax(final_logits, dim=1)

        labels_np = labels.detach().cpu().numpy()
        preds_np = preds.detach().cpu().numpy()
        probs_np = probs.detach().cpu().numpy()
        labels_all.extend(labels_np.tolist())
        preds_all.extend(preds_np.tolist())
        probs_all.extend(probs_np.tolist())
        sources_all.extend(list(sources))
        for expert_id in gate_pred.detach().cpu().tolist():
            selected_counts[EXPERT_NAMES[int(expert_id)]] += 1
        pbar.set_postfix({"acc": "{:.4f}".format(accuracy_score(labels_all, preds_all))})
    return labels_all, preds_all, probs_all, sources_all, selected_counts


def build_loader(base_dataset, base_transform, mode_name, sigma, args):
    dataset = PerGanNoiseDataset(base_dataset, base_transform, mode_name, noise_sigma=sigma, base_seed=42)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Per-GAN noise robustness for all selected models")
    parser.add_argument("--real_root", default=REAL_ROOT, type=str)
    parser.add_argument("--fake_root", default=FAKE_ROOT, type=str)
    parser.add_argument("--out_dir", default=OUT_DIR, type=str)
    parser.add_argument("--batch_size", default=CLICK_RUN_BATCH_SIZE, type=int)
    parser.add_argument("--num_workers", default=CLICK_RUN_NUM_WORKERS, type=int)
    parser.add_argument("--gpu", default=CLICK_RUN_GPU, type=str)
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODEL_KEYS),
        type=str,
        help="Comma-separated model keys. Use default to run all configured models.",
    )
    return parser.parse_args()


def model_specs_from_keys(keys):
    specs = []
    for key in keys:
        key = key.strip()
        if not key:
            continue
        if key in STANDARD_MODELS:
            spec = dict(STANDARD_MODELS[key])
            spec["key"] = key
            spec["kind"] = "standard"
            specs.append(spec)
        elif key in TIMM_STRONG_MODELS:
            spec = dict(TIMM_STRONG_MODELS[key])
            spec["key"] = key
            spec["kind"] = "timm"
            specs.append(spec)
        elif key in FINAL_MODEL:
            spec = dict(FINAL_MODEL[key])
            spec["key"] = key
            spec["kind"] = "final_gate"
            specs.append(spec)
        else:
            raise ValueError("Unknown model key: {}".format(key))
    return specs


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 80)
    print("PER-GAN GAUSSIAN NOISE ROBUSTNESS EVALUATION")
    print("Output dir: {}".format(args.out_dir))
    print("Batch size: {}".format(args.batch_size))
    print("Num workers: {}".format(args.num_workers))
    print("Device: {}".format(device))
    print("All noise is deterministic per image and applied to both real and fake images.")
    print("=" * 80)

    base_transform = make_base_transform()
    base_dataset = FFHQGANData(args.real_root, args.fake_root, "test", transform=None)
    print("Test dataset: real={}, fake={}, total={}".format(
        base_dataset.real_count, base_dataset.fake_count, len(base_dataset)))
    print("Fake source counts: {}".format(base_dataset.get_gan_source_counts()))

    model_keys = [item.strip() for item in args.models.split(",") if item.strip()]
    specs = model_specs_from_keys(model_keys)

    rows = []
    for spec in specs:
        checkpoint = spec["checkpoint"]
        ensure_checkpoint(checkpoint)
        print("\n" + "=" * 80)
        print("Model: {} ({})".format(spec["display"], spec["key"]))
        print("Kind: {}".format(spec["kind"]))
        print("Checkpoint: {}".format(checkpoint))
        print("=" * 80)

        model = None
        gate = None
        experts = None
        load_format = ""
        loaded_keys = ""

        if spec["kind"] == "standard":
            model, load_format, loaded_keys = load_standard_model(spec["model_type"], checkpoint, device)
            model.eval()
        elif spec["kind"] == "timm":
            model, load_format, loaded_keys = load_timm_model(spec["model_name"], checkpoint, device)
            model.eval()
        elif spec["kind"] == "final_gate":
            gate, gate_report = load_gate(device)
            if spec.get("strong_expert") == "no_high_noise_gate":
                experts, expert_reports = load_no_high_noise_gate_experts(device)
                load_format = "weak_strong_gate + stability_reg + no_high_noise_gate_expert"
            else:
                experts, expert_reports = load_experts(device)
                load_format = "weak_strong_gate + stability_reg + full_high_noise_gate_expert"
            loaded_keys = json.dumps({
                "gate": gate_report.get("loaded_state_dict_keys"),
                "experts": [r.get("loaded_state_dict_keys") for r in expert_reports],
            })
        else:
            raise RuntimeError("Unsupported kind: {}".format(spec["kind"]))

        for mode_name, sigma in DEGRADATION_MODES:
            loader = build_loader(base_dataset, base_transform, mode_name, sigma, args)
            desc = "{} {}".format(spec["key"], mode_name)
            if spec["kind"] == "final_gate":
                labels, preds, probs, sources, diag = evaluate_final_gate(gate, experts, loader, device, desc)
            else:
                labels, preds, probs, sources, diag = evaluate_single_model(model, loader, device, desc)

            row = {
                "model": spec["key"],
                "display_name": spec["display"],
                "model_kind": spec["kind"],
                "checkpoint": checkpoint,
                "load_format": load_format,
                "loaded_keys": loaded_keys,
                "degradation": mode_name,
                "noise_sigma": "" if sigma is None else sigma,
            }
            row.update(compute_overall_metrics(labels, preds, probs))
            add_per_source_metrics(row, labels, preds, sources)
            for name in EXPERT_NAMES:
                row["selected_{}".format(name)] = diag.get(name, "")
            rows.append(row)

            print(
                "{} | {} | acc={:.4f} f1={:.4f} real={:.4f} ProGAN={:.4f} StyleGAN2={:.4f} StyleGAN3={:.4f}".format(
                    spec["key"],
                    mode_name,
                    row["overall_acc"],
                    row["f1"],
                    row["real_acc"],
                    row["ProGAN_fake_recall"],
                    row["StyleGAN2_fake_recall"],
                    row["StyleGAN3_fake_recall"],
                )
            )

        del model
        del gate
        del experts
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    fieldnames = [
        "model", "display_name", "model_kind", "checkpoint", "load_format", "loaded_keys",
        "degradation", "noise_sigma",
        "overall_acc", "precision", "recall", "f1", "auc",
        "real_acc", "mean_gan_fake_recall",
        "ProGAN_fake_recall", "StyleGAN2_fake_recall", "StyleGAN3_fake_recall",
        "real_total", "real_correct",
        "ProGAN_total", "ProGAN_correct",
        "StyleGAN2_total", "StyleGAN2_correct",
        "StyleGAN3_total", "StyleGAN3_correct",
        "TN", "FP", "FN", "TP",
        "selected_stability_reg", "selected_high_noise_bias_calib",
    ]

    csv_path = os.path.join(args.out_dir, "per_gan_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    txt_path = os.path.join(args.out_dir, "per_gan_summary.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("PER-GAN GAUSSIAN NOISE ROBUSTNESS SUMMARY\n")
        f.write("Rows: model x degradation\n")
        f.write("Fake recall is reported separately for ProGAN, StyleGAN2, StyleGAN3.\n\n")
        header = "{:<34} {:<8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}\n".format(
            "Model", "Deg", "Acc", "F1", "Real", "ProGAN", "SG2", "SG3")
        f.write(header)
        f.write("-" * 100 + "\n")
        for row in rows:
            f.write("{:<34} {:<8} {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f}\n".format(
                row["model"],
                row["degradation"],
                row["overall_acc"],
                row["f1"],
                row["real_acc"],
                row["ProGAN_fake_recall"],
                row["StyleGAN2_fake_recall"],
                row["StyleGAN3_fake_recall"],
            ))

    print("\nSaved CSV: {}".format(csv_path))
    print("Saved TXT: {}".format(txt_path))


if __name__ == "__main__":
    main()
