"""
InceptionV3 threshold calibration diagnostic under Gaussian noise.

This script answers a specific reviewer-style question:
Can the noisy-test improvement be reproduced by a simple validation-set
threshold calibration on the original InceptionV3 baseline?

It does not train a model and does not modify checkpoints.

Default protocol:
  1) Load the original InceptionV3 baseline checkpoint with strict=True.
  2) Run val and test under clean/noise10/noise25/noise50.
  3) Find the best fake-probability threshold on val.
  4) Apply that threshold to the corresponding test degradation.
  5) Also report a single global threshold selected on all val modes.

Outputs:
  weights/inceptionv3-threshold-calibration/threshold_calibration_results.csv
  weights/inceptionv3-threshold-calibration/threshold_calibration_results.txt
  weights/inceptionv3-threshold-calibration/fake_probabilities_*.csv
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


ROOT = os.path.dirname(os.path.abspath(__file__))
REAL_ROOT = r"E:\FFHQ_off"
FAKE_ROOT = r"E:\archive\GAN_Generated_Fake_Images\fake"
INCEPTIONV3_BASELINE_CKPT = os.path.join(ROOT, "weights", "best_model_inceptionv3.pth")
OUT_DIR = os.path.join(ROOT, "weights", "inceptionv3-threshold-calibration")

# ==============================
# Click-run default configuration
# ==============================
CLICK_RUN_BATCH_SIZE = 8
CLICK_RUN_NUM_WORKERS = 0
CLICK_RUN_GPU = "0"
CLICK_RUN_DEBUG_SUBSET = None
CLICK_RUN_SIGMAS = [0, 10, 25, 50]
CLICK_RUN_THRESHOLD_GRID_SIZE = 1001


class GaussianNoiseDataset:
    """Apply deterministic Gaussian noise in raw pixel space [0, 255]."""

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
        return self.transform(img), int(label), path


def clean_state_dict(state_dict):
    clean = OrderedDict()
    for key, value in state_dict.items():
        clean[key[7:] if key.startswith("module.") else key] = value
    return clean


def checkpoint_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"], "model_state_dict"
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"], "state_dict"
    if isinstance(checkpoint, dict):
        return checkpoint, "raw_state_dict"
    raise RuntimeError("Unsupported checkpoint format: {}".format(type(checkpoint)))


def build_inceptionv3(checkpoint_path, device):
    print("Loading InceptionV3 baseline checkpoint: {}".format(checkpoint_path))
    if not os.path.exists(checkpoint_path):
        raise RuntimeError("Checkpoint not found: {}".format(checkpoint_path))

    model = models.inception_v3(pretrained=False, aux_logits=True)
    model.aux_logits = False
    model.AuxLogits = None
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.transform_input = True

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict, fmt = checkpoint_state_dict(checkpoint)
    state_dict = clean_state_dict(state_dict)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    print("  checkpoint format: {}".format(fmt))
    print("  strict=True: OK")
    print("  loaded state_dict keys: {}".format(len(state_dict)))
    print("  transform_input: {}".format(model.transform_input))
    return model


def auc_score(labels, scores):
    labels = np.asarray(labels).astype(np.int64)
    scores = np.asarray(scores).astype(np.float64)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    n_pos = len(pos)
    n_neg = len(neg)
    if n_pos == 0 or n_neg == 0:
        return 0.0

    order = np.argsort(scores)
    sorted_scores = scores[order]
    ranks = np.empty_like(sorted_scores, dtype=np.float64)
    i = 0
    while i < len(sorted_scores):
        j = i + 1
        while j < len(sorted_scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[i:j] = avg_rank
        i = j
    original_ranks = np.empty_like(ranks)
    original_ranks[order] = ranks
    rank_sum_pos = original_ranks[labels == 1].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def compute_metrics(labels, fake_probs, threshold):
    labels = np.asarray(labels).astype(np.int64)
    fake_probs = np.asarray(fake_probs).astype(np.float64)
    preds = (fake_probs >= float(threshold)).astype(np.int64)

    tn = int(((labels == 0) & (preds == 0)).sum())
    fp = int(((labels == 0) & (preds == 1)).sum())
    fn = int(((labels == 1) & (preds == 0)).sum())
    tp = int(((labels == 1) & (preds == 1)).sum())

    total = max(1, len(labels))
    acc = float((tn + tp) / total)
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    real_acc = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    fake_rec = recall
    auc = auc_score(labels, fake_probs)

    return {
        "threshold": float(threshold),
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "real_acc": real_acc,
        "fake_rec": fake_rec,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }


def best_threshold(labels, fake_probs, grid_size):
    labels = np.asarray(labels).astype(np.int64)
    fake_probs = np.asarray(fake_probs).astype(np.float64)
    candidates = np.linspace(0.0, 1.0, int(grid_size))

    best = None
    best_key = None
    for threshold in candidates:
        metrics = compute_metrics(labels, fake_probs, threshold)
        key = (metrics["acc"], metrics["f1"], -abs(threshold - 0.5))
        if best is None or key > best_key:
            best = metrics
            best_key = key
    return best


@torch.no_grad()
def collect_predictions(model, loader, device, desc):
    labels_all = []
    probs_all = []
    margins_all = []
    paths_all = []
    start = time.time()
    pbar = tqdm(loader, desc=desc, dynamic_ncols=True)
    for images, labels, paths in pbar:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        if isinstance(logits, tuple):
            logits = logits[0]
        probs = F.softmax(logits, dim=1)[:, 1]
        margins = logits[:, 1] - logits[:, 0]

        labels_np = labels.detach().cpu().numpy().astype(np.int64)
        probs_np = probs.detach().cpu().numpy().astype(np.float64)
        labels_all.extend(labels_np.tolist())
        probs_all.extend(probs_np.tolist())
        margins_all.extend(margins.detach().cpu().numpy().astype(np.float64).tolist())
        paths_all.extend(list(paths))

        if labels_all:
            running = compute_metrics(labels_all, probs_all, 0.5)
            pbar.set_postfix(acc="{:.4f}".format(running["acc"]))

    return {
        "labels": np.asarray(labels_all, dtype=np.int64),
        "fake_probs": np.asarray(probs_all, dtype=np.float64),
        "margins": np.asarray(margins_all, dtype=np.float64),
        "paths": paths_all,
        "time": time.time() - start,
    }


def sigma_name(sigma):
    return "clean" if float(sigma) == 0.0 else "noise{}".format(int(sigma))


def make_loader(raw_dataset, transform, sigma, args, split_name):
    dataset = GaussianNoiseDataset(
        raw_dataset,
        transform,
        sigma=sigma,
        base_seed=42,
        debug_subset=args.debug_subset,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )


def save_probabilities(out_dir, split_name, mode_name, collected):
    path = os.path.join(out_dir, "fake_probabilities_{}_{}.csv".format(split_name, mode_name))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label", "fake_probability", "logit_margin"])
        for img_path, label, prob, margin in zip(
            collected["paths"],
            collected["labels"],
            collected["fake_probs"],
            collected["margins"],
        ):
            writer.writerow([img_path, int(label), "{:.10f}".format(float(prob)), "{:.10f}".format(float(margin))])
    return path


def write_outputs(out_dir, rows, global_threshold, args):
    csv_path = os.path.join(out_dir, "threshold_calibration_results.csv")
    fields = [
        "split",
        "degradation",
        "sigma",
        "method",
        "threshold",
        "acc",
        "precision",
        "recall",
        "f1",
        "auc",
        "real_acc",
        "fake_rec",
        "TN",
        "FP",
        "FN",
        "TP",
        "source",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    txt_path = os.path.join(out_dir, "threshold_calibration_results.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("InceptionV3 threshold calibration diagnostic\n")
        f.write("=" * 80 + "\n")
        f.write("Checkpoint: {}\n".format(args.checkpoint))
        f.write("Real root: {}\nFake root: {}\n".format(args.real_root, args.fake_root))
        f.write("Sigmas: {}\n".format(args.sigmas))
        f.write("Global val threshold: {:.6f}\n".format(global_threshold))
        f.write("Labels: 0=Real, 1=Fake; prediction is fake_probability >= threshold\n\n")
        for row in rows:
            if row["split"] != "test":
                continue
            f.write(
                "{:<8} {:<23} thr={:.4f} acc={:.4f} prec={:.4f} rec={:.4f} "
                "f1={:.4f} auc={:.4f} real={:.4f} fake={:.4f} cm=[[{}, {}], [{}, {}]]\n".format(
                    row["degradation"],
                    row["method"],
                    float(row["threshold"]),
                    float(row["acc"]),
                    float(row["precision"]),
                    float(row["recall"]),
                    float(row["f1"]),
                    float(row["auc"]),
                    float(row["real_acc"]),
                    float(row["fake_rec"]),
                    row["TN"],
                    row["FP"],
                    row["FN"],
                    row["TP"],
                )
            )
        f.write("\nNote:\n")
        f.write("- default_argmax is equivalent to fake_probability >= 0.5.\n")
        f.write("- val_best_threshold is selected independently on the validation split for each degradation.\n")
        f.write("- global_val_threshold is selected once on pooled validation probabilities across all degradations.\n")
        f.write("- Temperature scaling alone does not change argmax decisions for a binary softmax; threshold/logit-bias calibration is the relevant accuracy control.\n")

    return csv_path, txt_path


def parse_sigmas(value):
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    return [float(v.strip()) for v in str(value).split(",") if v.strip()]


def make_row(split, degradation, sigma, method, metrics, source):
    row = {
        "split": split,
        "degradation": degradation,
        "sigma": float(sigma),
        "method": method,
        "source": source,
    }
    for key, value in metrics.items():
        if isinstance(value, float):
            row[key] = "{:.10f}".format(value)
        else:
            row[key] = value
    return row


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_root", default=REAL_ROOT)
    parser.add_argument("--fake_root", default=FAKE_ROOT)
    parser.add_argument("--checkpoint", default=INCEPTIONV3_BASELINE_CKPT)
    parser.add_argument("--out_dir", default=OUT_DIR)
    parser.add_argument("--batch_size", type=int, default=CLICK_RUN_BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=CLICK_RUN_NUM_WORKERS)
    parser.add_argument("--gpu", default=CLICK_RUN_GPU)
    parser.add_argument("--debug_subset", type=int, default=CLICK_RUN_DEBUG_SUBSET)
    parser.add_argument("--sigmas", default=",".join(str(v) for v in CLICK_RUN_SIGMAS))
    parser.add_argument("--threshold_grid_size", type=int, default=CLICK_RUN_THRESHOLD_GRID_SIZE)
    return parser.parse_args()


def main():
    args = parse_args()
    args.sigmas = parse_sigmas(args.sigmas)
    os.makedirs(args.out_dir, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("INCEPTIONV3 THRESHOLD CALIBRATION DIAGNOSTIC")
    print("=" * 80)
    print("Device: {}".format(device))
    print("Checkpoint: {}".format(args.checkpoint))
    print("Sigmas: {}".format(args.sigmas))
    print("Output dir: {}".format(args.out_dir))
    print("Threshold grid size: {}".format(args.threshold_grid_size))
    print("=" * 80)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    raw_val = FFHQGANData(args.real_root, args.fake_root, "val", transform=None)
    raw_test = FFHQGANData(args.real_root, args.fake_root, "test", transform=None)
    print("Val samples: real={} fake={} total={}".format(raw_val.real_count, raw_val.fake_count, len(raw_val)))
    print("Test samples: real={} fake={} total={}".format(raw_test.real_count, raw_test.fake_count, len(raw_test)))

    model = build_inceptionv3(args.checkpoint, device)

    val_cache = {}
    test_cache = {}
    rows = []

    pooled_val_labels = []
    pooled_val_probs = []

    for sigma in args.sigmas:
        mode = sigma_name(sigma)
        val_loader = make_loader(raw_val, transform, sigma, args, "val")
        val_collected = collect_predictions(model, val_loader, device, "Val {}".format(mode))
        val_cache[mode] = val_collected
        save_probabilities(args.out_dir, "val", mode, val_collected)

        pooled_val_labels.extend(val_collected["labels"].tolist())
        pooled_val_probs.extend(val_collected["fake_probs"].tolist())

        default_val = compute_metrics(val_collected["labels"], val_collected["fake_probs"], 0.5)
        best_val = best_threshold(val_collected["labels"], val_collected["fake_probs"], args.threshold_grid_size)
        rows.append(make_row("val", mode, sigma, "default_argmax", default_val, "threshold=0.5"))
        rows.append(make_row("val", mode, sigma, "val_best_threshold", best_val, "selected on same val degradation"))

    global_best = best_threshold(pooled_val_labels, pooled_val_probs, args.threshold_grid_size)
    global_threshold = float(global_best["threshold"])
    print("Global validation threshold: {:.6f}".format(global_threshold))

    for sigma in args.sigmas:
        mode = sigma_name(sigma)
        test_loader = make_loader(raw_test, transform, sigma, args, "test")
        test_collected = collect_predictions(model, test_loader, device, "Test {}".format(mode))
        test_cache[mode] = test_collected
        save_probabilities(args.out_dir, "test", mode, test_collected)

        val_best = best_threshold(
            val_cache[mode]["labels"],
            val_cache[mode]["fake_probs"],
            args.threshold_grid_size,
        )
        test_default = compute_metrics(test_collected["labels"], test_collected["fake_probs"], 0.5)
        test_mode_threshold = compute_metrics(
            test_collected["labels"],
            test_collected["fake_probs"],
            val_best["threshold"],
        )
        test_global_threshold = compute_metrics(
            test_collected["labels"],
            test_collected["fake_probs"],
            global_threshold,
        )

        rows.append(make_row("test", mode, sigma, "default_argmax", test_default, "threshold=0.5"))
        rows.append(make_row("test", mode, sigma, "val_best_threshold", test_mode_threshold, "threshold selected on val {}".format(mode)))
        rows.append(make_row("test", mode, sigma, "global_val_threshold", test_global_threshold, "single threshold selected on pooled val modes"))

        print(
            "{} test: default_acc={:.4f}, val_thr={:.4f}, val_thr_acc={:.4f}, global_thr_acc={:.4f}".format(
                mode,
                test_default["acc"],
                float(val_best["threshold"]),
                test_mode_threshold["acc"],
                test_global_threshold["acc"],
            )
        )

    csv_path, txt_path = write_outputs(args.out_dir, rows, global_threshold, args)
    print("=" * 80)
    print("Saved CSV: {}".format(csv_path))
    print("Saved TXT: {}".format(txt_path))
    print("Saved fake probabilities: {}".format(args.out_dir))
    print("=" * 80)


if __name__ == "__main__":
    main()
