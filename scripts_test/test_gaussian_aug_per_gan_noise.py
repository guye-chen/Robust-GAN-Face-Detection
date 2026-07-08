"""
Per-GAN Gaussian noise evaluation for InceptionV3 + external Gaussian augmentation.

This script is evaluation-only:
  - no training
  - no threshold tuning
  - no checkpoint overwrite

It reports real accuracy and fake recall for ProGAN / StyleGAN2 / StyleGAN3
under clean, noise10, noise25, and noise50.
"""

import argparse
import csv
import hashlib
import os
import sys
from collections import OrderedDict

import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader.ffhq_gan_dataset import FFHQGANData, pil_loader
from test_noise_only import build_model, load_model_state


REAL_ROOT = r"E:\FFHQ_off"
FAKE_ROOT = r"E:\archive\GAN_Generated_Fake_Images\fake"
CHECKPOINT = r"E:\cbam_ftt_fusion\CBAM.PyTorch-master\weights\inceptionv3-gaussian-noise-aug\best_model.pth"
OUT_DIR = r"E:\cbam_ftt_fusion\CBAM.PyTorch-master\weights\inceptionv3-gaussian-noise-aug-per-gan"

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


def make_base_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def noise_degrade(img, sigma, seed):
    img = img.convert("RGB")
    arr = np.asarray(img).astype(np.float32)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, float(sigma), arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


class PerGanNoiseDataset(Dataset):
    def __init__(self, base_dataset, base_transform, noise_sigma=None, base_seed=42):
        self.samples = base_dataset.samples
        self.base_transform = base_transform
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


def extract_logits(outputs):
    if isinstance(outputs, tuple):
        return outputs[0]
    if hasattr(outputs, "logits"):
        return outputs.logits
    return outputs


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


@torch.no_grad()
def evaluate(model, loader, device, desc):
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

    return labels_all, preds_all, probs_all, sources_all


def parse_args():
    parser = argparse.ArgumentParser(description="Per-GAN test for InceptionV3 Gaussian Aug")
    parser.add_argument("--real_root", default=REAL_ROOT, type=str)
    parser.add_argument("--fake_root", default=FAKE_ROOT, type=str)
    parser.add_argument("--checkpoint", default=CHECKPOINT, type=str)
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
    print("INCEPTIONV3 + EXTERNAL GAUSSIAN AUG PER-GAN NOISE TEST")
    print("Checkpoint: {}".format(args.checkpoint))
    print("Output dir: {}".format(args.out_dir))
    print("No training. No checkpoint overwrite.")
    print("Labels: 0=Real, 1=Fake. Prediction: argmax(logits).")
    print("=" * 80)

    if not os.path.exists(args.checkpoint):
        raise RuntimeError("Checkpoint not found: {}".format(args.checkpoint))

    base_transform = make_base_transform()
    base_dataset = FFHQGANData(args.real_root, args.fake_root, "test", transform=None)
    print("Test dataset: real={}, fake={}, total={}".format(
        base_dataset.real_count, base_dataset.fake_count, len(base_dataset)))
    print("Fake source counts: {}".format(base_dataset.get_gan_source_counts()))

    model = build_model("inceptionv3", num_classes=2)
    loaded_keys = load_model_state(model, args.checkpoint, device)
    model = model.to(device)
    model.eval()
    print("Loaded checkpoint keys: {}".format(loaded_keys))

    rows = []
    for mode_name, sigma in DEGRADATION_MODES:
        dataset = PerGanNoiseDataset(base_dataset, base_transform, noise_sigma=sigma, base_seed=42)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        labels, preds, probs, sources = evaluate(model, loader, device, "{} Eval".format(mode_name))
        row = OrderedDict()
        row["model"] = "inceptionv3-gaussian-noise-aug"
        row["checkpoint"] = args.checkpoint
        row["degradation"] = mode_name
        row["noise_sigma"] = "" if sigma is None else sigma
        row.update(compute_overall_metrics(labels, preds, probs))
        add_per_source_metrics(row, labels, preds, sources)
        rows.append(row)

        print(
            "{:<8} acc={:.4f} f1={:.4f} auc={:.4f} real={:.4f} "
            "ProGAN={:.4f} StyleGAN2={:.4f} StyleGAN3={:.4f} cm=[[{}, {}], [{}, {}]]".format(
                mode_name,
                row["overall_acc"],
                row["f1"],
                row["auc"],
                row["real_acc"],
                row["ProGAN_fake_recall"],
                row["StyleGAN2_fake_recall"],
                row["StyleGAN3_fake_recall"],
                row["TN"],
                row["FP"],
                row["FN"],
                row["TP"],
            )
        )

    csv_path = os.path.join(args.out_dir, "per_gan_gaussian_aug_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    txt_path = os.path.join(args.out_dir, "per_gan_gaussian_aug_results.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("PER-GAN RESULTS: InceptionV3 + external Gaussian augmentation\n")
        f.write("Checkpoint: {}\n\n".format(args.checkpoint))
        f.write("{:<12} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8}\n".format(
            "degradation", "acc", "f1", "real", "ProGAN", "SG2", "SG3"))
        for row in rows:
            f.write("{:<12} {:<8.4f} {:<8.4f} {:<8.4f} {:<8.4f} {:<8.4f} {:<8.4f}\n".format(
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
