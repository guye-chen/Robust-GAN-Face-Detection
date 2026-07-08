"""
Unseen Gaussian noise strength evaluation.

Purpose:
  Verify whether the final weak/strong expert gate generalizes to Gaussian
  noise strengths that were not used as the main clean/noise10/noise25/noise50
  evaluation protocol.

This script does not train models and does not modify checkpoints.

Default comparison models:
  1) InceptionV3 baseline
  2) InceptionV3 + external Gaussian noise augmentation
  3) Weak expert: Stability-Reg
  4) Strong expert: Bias-Calib w/o high-noise gate
  5) Final weak/strong gate v2

Outputs:
  weights/unseen_gaussian_noise_strengths/unseen_noise_results.csv
  weights/unseen_gaussian_noise_strengths/unseen_noise_results.txt
"""

import argparse
import csv
import hashlib
import os
import sys
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from tqdm import tqdm
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader.ffhq_gan_dataset import FFHQGANData, pil_loader
from test_final_gate_and_experts_light_degradations import (
    build_models as build_final_gate_and_experts,
    compute_metrics,
    extract_logits,
    freeze_eval,
)


ROOT = os.path.dirname(os.path.abspath(__file__))
REAL_ROOT = r"E:\FFHQ_off"
FAKE_ROOT = r"E:\archive\GAN_Generated_Fake_Images\fake"

# ==============================
# Click-run default configuration
# ==============================
CLICK_RUN_BATCH_SIZE = 1
CLICK_RUN_NUM_WORKERS = 0
CLICK_RUN_GPU = "0"
CLICK_RUN_DEBUG_SUBSET = None
CLICK_RUN_NOISE_SIGMAS = [0, 5, 10, 15, 20, 25, 30, 40, 50, 60]
CLICK_RUN_MODELS = [
    "inceptionv3_baseline",
    "external_gaussian_aug",
    "weak_expert_stability_reg",
    "strong_expert_bias_calib_no_high_noise_gate",
    "final_gate_v2",
]
OUT_DIR = os.path.join(ROOT, "weights", "unseen_gaussian_noise_strengths")


INCEPTIONV3_BASELINE_CKPT = os.path.join(ROOT, "weights", "best_model_inceptionv3.pth")
EXTERNAL_GAUSSIAN_AUG_CKPT = os.path.join(
    ROOT, "weights", "inceptionv3-gaussian-noise-aug", "best_model.pth"
)

SEEN_PROTOCOL_SIGMAS = {0, 10, 25, 50}


MODEL_DISPLAY = {
    "inceptionv3_baseline": "InceptionV3 baseline",
    "external_gaussian_aug": "InceptionV3 + external Gaussian aug.",
    "weak_expert_stability_reg": "Weak Expert: Stability-Reg",
    "strong_expert_bias_calib_no_high_noise_gate": "Strong Expert: Bias-Calib w/o gate",
    "final_gate_v2": "Final Weak-Strong Gate v2",
}


class GaussianNoiseDataset:
    """Apply Gaussian noise in raw pixel space [0, 255] before normalization."""

    def __init__(self, base_dataset, transform, sigma, base_seed=42, debug_subset=None):
        self.samples = list(base_dataset.samples)
        if debug_subset is not None:
            self.samples = self.samples[: int(debug_subset)]
        self.transform = transform
        self.sigma = float(sigma)
        self.base_seed = int(base_seed)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label, _ = self.samples[index]
        img = pil_loader(path)
        if self.sigma > 0:
            key = "{}|gaussian|{}|{}".format(path, self.sigma, self.base_seed)
            seed = int(hashlib.md5(key.encode("utf-8")).digest()[:4].hex(), 16)
            rng = np.random.default_rng(seed)
            arr = np.asarray(img.convert("RGB")).astype(np.float32)
            noise = rng.normal(0.0, self.sigma, arr.shape).astype(np.float32)
            img = Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))
        return self.transform(img), int(label)


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


def load_strict(model, checkpoint_path, device, name):
    print("Loading {} checkpoint: {}".format(name, checkpoint_path))
    print("  exists: {}".format(os.path.exists(checkpoint_path)))
    if not os.path.exists(checkpoint_path):
        raise RuntimeError("Checkpoint not found: {}".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict, fmt = checkpoint_state_dict(checkpoint)
    state_dict = clean_state_dict(state_dict)
    model.load_state_dict(state_dict, strict=True)
    print("  checkpoint format: {}".format(fmt))
    print("  strict=True: OK")
    print("  loaded state_dict keys: {}".format(len(state_dict)))
    return {
        "checkpoint": checkpoint_path,
        "format": fmt,
        "strict": True,
        "loaded_state_dict_keys": len(state_dict),
    }


def build_inceptionv3_from_checkpoint(checkpoint_path, device, name):
    model = models.inception_v3(pretrained=False, aux_logits=True)
    model.aux_logits = False
    model.AuxLogits = None
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.transform_input = True
    report = load_strict(model, checkpoint_path, device, name)
    return freeze_eval(model.to(device)), report


def build_selected_models(model_keys, device):
    built = {}
    reports = {}

    need_final_family = any(
        key in model_keys
        for key in [
            "weak_expert_stability_reg",
            "strong_expert_bias_calib_no_high_noise_gate",
            "final_gate_v2",
        ]
    )
    if need_final_family:
        specs = build_final_gate_and_experts(device)
        for spec in specs:
            if spec["name"] == "weak_expert_stability_reg":
                built["weak_expert_stability_reg"] = spec["model"]
                reports["weak_expert_stability_reg"] = spec["report"]
            elif spec["name"] == "strong_expert_bias_calib_no_high_noise_gate":
                built["strong_expert_bias_calib_no_high_noise_gate"] = spec["model"]
                reports["strong_expert_bias_calib_no_high_noise_gate"] = spec["report"]
            elif spec["name"] == "final_frequency_reliability_weak_strong_gate_v2":
                built["final_gate_v2"] = spec["model"]
                reports["final_gate_v2"] = spec["report"]

    if "inceptionv3_baseline" in model_keys:
        built["inceptionv3_baseline"], reports["inceptionv3_baseline"] = build_inceptionv3_from_checkpoint(
            INCEPTIONV3_BASELINE_CKPT, device, "inceptionv3_baseline"
        )

    if "external_gaussian_aug" in model_keys:
        built["external_gaussian_aug"], reports["external_gaussian_aug"] = build_inceptionv3_from_checkpoint(
            EXTERNAL_GAUSSIAN_AUG_CKPT, device, "external_gaussian_aug"
        )

    ordered = []
    for key in model_keys:
        if key not in built:
            raise RuntimeError("Model was not built: {}".format(key))
        ordered.append((key, built[key], reports.get(key, {})))
    return ordered


@torch.no_grad()
def evaluate_model(model_key, model, loader, device, desc):
    model.eval()
    if hasattr(model, "reset_diagnostics"):
        model.reset_diagnostics()

    labels_all, preds_all, probs_all = [], [], []
    start = time.time()
    pbar = tqdm(loader, desc=desc, dynamic_ncols=True, leave=True)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = extract_logits(model(images))
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)

        labels_all.extend(labels.detach().cpu().numpy().tolist())
        preds_all.extend(preds.detach().cpu().numpy().tolist())
        probs_all.extend(probs.detach().cpu().numpy().tolist())

        if len(labels_all) > 0:
            running_acc = float(np.mean(np.asarray(labels_all) == np.asarray(preds_all)))
            pbar.set_postfix(acc="{:.4f}".format(running_acc))

    total_time = time.time() - start
    metrics = compute_metrics(labels_all, preds_all, np.asarray(probs_all))
    metrics["total_time"] = float(total_time)
    metrics["time_per_image"] = float(total_time / max(1, len(labels_all)))

    if hasattr(model, "last_selection_counts"):
        counts = dict(model.last_selection_counts)
        total = sum(counts.values()) if counts else 0
        metrics["selected_weak"] = int(counts.get("stability_reg", 0))
        metrics["selected_strong"] = int(counts.get("high_noise_bias_calib_no_high_noise_gate", 0))
        metrics["selected_weak_ratio"] = float(metrics["selected_weak"] / total) if total > 0 else 0.0
        metrics["selected_strong_ratio"] = float(metrics["selected_strong"] / total) if total > 0 else 0.0
    else:
        metrics["selected_weak"] = ""
        metrics["selected_strong"] = ""
        metrics["selected_weak_ratio"] = ""
        metrics["selected_strong_ratio"] = ""

    print(
        "{}: acc={:.4f} f1={:.4f} auc={:.4f} real_acc={:.4f} fake_rec={:.4f} cm=[[{}, {}], [{}, {}]]".format(
            desc,
            metrics["accuracy"],
            metrics["f1"],
            metrics["auc"],
            metrics["real_acc"],
            metrics["fake_acc"],
            metrics["TN"],
            metrics["FP"],
            metrics["FN"],
            metrics["TP"],
        )
    )
    return metrics


def make_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def parse_models(value):
    if value.lower() == "all":
        return list(CLICK_RUN_MODELS)
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_sigmas(value):
    if isinstance(value, list):
        return [float(v) for v in value]
    return [float(item.strip()) for item in value.split(",") if item.strip()]


FIELDNAMES = [
    "model_key",
    "model",
    "degradation",
    "sigma",
    "seen_protocol_sigma",
    "accuracy",
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
    "selected_weak",
    "selected_strong",
    "selected_weak_ratio",
    "selected_strong_ratio",
    "total_time",
    "time_per_image",
]


def save_results(rows, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "unseen_noise_results.csv")
    txt_path = os.path.join(out_dir, "unseen_noise_results.txt")

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("UNSEEN GAUSSIAN NOISE STRENGTH EVALUATION\n")
        f.write("=" * 110 + "\n")
        f.write("Seen protocol sigmas: clean/noise10/noise25/noise50 = 0/10/25/50\n")
        f.write("Unseen sigmas in this test include: 5/15/20/30/40/60\n")
        f.write("Noise is added in raw pixel space [0,255] before ToTensor + Normalize.\n")
        f.write("=" * 110 + "\n\n")
        for row in rows:
            f.write(
                "{:<42s} {:<8s} acc={:.4f} f1={:.4f} auc={:.4f} real={:.4f} fake={:.4f} cm=[[{}, {}], [{}, {}]]".format(
                    row["model"],
                    row["degradation"],
                    row["accuracy"],
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
            if row["model_key"] == "final_gate_v2":
                f.write(
                    " weak={:.4f} strong={:.4f}".format(
                        row["selected_weak_ratio"],
                        row["selected_strong_ratio"],
                    )
                )
            f.write("\n")

    print("\nSaved CSV: {}".format(csv_path))
    print("Saved TXT: {}".format(txt_path))


def parse_args():
    parser = argparse.ArgumentParser(description="Unseen Gaussian noise strength evaluation")
    parser.add_argument("--batch_size", type=int, default=CLICK_RUN_BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=CLICK_RUN_NUM_WORKERS)
    parser.add_argument("--gpu", default=CLICK_RUN_GPU)
    parser.add_argument("--debug_subset", type=int, default=CLICK_RUN_DEBUG_SUBSET)
    parser.add_argument(
        "--sigmas",
        default=",".join(str(v) for v in CLICK_RUN_NOISE_SIGMAS),
        help="Comma-separated Gaussian sigma values in raw pixel scale, e.g. 0,5,10,15,20,25,30,40,50,60",
    )
    parser.add_argument(
        "--models",
        default="all",
        help="all or comma-separated keys: {}".format(",".join(CLICK_RUN_MODELS)),
    )
    parser.add_argument("--out_dir", default=OUT_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sigmas = parse_sigmas(args.sigmas)
    model_keys = parse_models(args.models)

    print("=" * 80)
    print("UNSEEN GAUSSIAN NOISE STRENGTH EVALUATION")
    print("=" * 80)
    print("Device: {}".format(device))
    print("Batch size: {}".format(args.batch_size))
    print("Workers: {}".format(args.num_workers))
    print("Debug subset: {}".format(args.debug_subset))
    print("Sigmas: {}".format(sigmas))
    print("Models: {}".format(model_keys))
    print("Output dir: {}".format(args.out_dir))
    print("=" * 80)

    transform = make_transform()
    base_dataset = FFHQGANData(REAL_ROOT, FAKE_ROOT, "test", transform=None)
    print("Test base samples: real={}, fake={}, total={}".format(
        base_dataset.real_count,
        base_dataset.fake_count,
        len(base_dataset),
    ))

    models_to_eval = build_selected_models(model_keys, device)
    rows = []
    for model_key, model, _report in models_to_eval:
        print("\n" + "=" * 80)
        print("Evaluating model: {}".format(MODEL_DISPLAY[model_key]))
        print("=" * 80)
        for sigma in sigmas:
            degradation = "clean" if sigma == 0 else "noise{}".format(int(sigma) if sigma.is_integer() else sigma)
            dataset = GaussianNoiseDataset(
                base_dataset,
                transform,
                sigma=sigma,
                debug_subset=args.debug_subset,
            )
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=(device.type == "cuda"),
            )
            metrics = evaluate_model(
                model_key,
                model,
                loader,
                device,
                "{} {}".format(MODEL_DISPLAY[model_key], degradation),
            )
            row = {
                "model_key": model_key,
                "model": MODEL_DISPLAY[model_key],
                "degradation": degradation,
                "sigma": float(sigma),
                "seen_protocol_sigma": bool(int(sigma) in SEEN_PROTOCOL_SIGMAS),
            }
            row.update(metrics)
            rows.append(row)

    save_results(rows, args.out_dir)


if __name__ == "__main__":
    main()
