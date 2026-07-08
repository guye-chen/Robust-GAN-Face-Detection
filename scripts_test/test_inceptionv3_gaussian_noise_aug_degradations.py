"""
Evaluate the InceptionV3 + ordinary Gaussian noise augmentation model
under common external degradations.

This script is evaluation-only:
  - no training
  - no threshold tuning
  - no test-set leakage
"""

import os
import sys
import argparse
import csv
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader.ffhq_gan_dataset import FFHQGANData
from test_noise_only import (
    DeterministicNoiseTestDataset,
    build_model,
    load_model_state,
    compute_metrics,
    _extract_logits,
)


REAL_ROOT = r"E:\FFHQ_off"
FAKE_ROOT = r"E:\archive\GAN_Generated_Fake_Images\fake"
CHECKPOINT = r"E:\cbam_ftt_fusion\CBAM.PyTorch-master\weights\inceptionv3-gaussian-noise-aug\best_model.pth"
OUT_DIR = r"E:\cbam_ftt_fusion\CBAM.PyTorch-master\weights\inceptionv3-gaussian-noise-aug"

# ==============================
# Click-run default configuration
# ==============================
CLICK_RUN_BATCH_SIZE = 1
CLICK_RUN_NUM_WORKERS = 0
CLICK_RUN_GPU = "0"


DEGRADATION_MODES = [
    ("clean", None, None),
    ("noise10", "noise", 10),
    ("noise25", "noise", 25),
    ("noise50", "noise", 50),
    ("jpeg90", "jpeg", 90),
    ("jpeg70", "jpeg", 70),
    ("blur1", "blur", 1.0),
    ("blur2", "blur", 2.0),
    ("downsample0.75", "downsample", 0.75),
    ("downsample0.5", "downsample", 0.5),
]


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, num_classes, desc):
    model.eval()
    total_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []
    pbar = tqdm(loader, desc=desc, dynamic_ncols=True, leave=True)
    for batch_idx, (inputs, labels) in enumerate(pbar):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        logits = _extract_logits(outputs)
        loss = loss_fn(logits, labels)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)

        total_loss += loss.item()
        all_labels.extend(labels.detach().cpu().numpy())
        all_preds.extend(preds.detach().cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy())
        running_acc = (torch.tensor(all_preds) == torch.tensor(all_labels)).float().mean().item()
        pbar.set_postfix({
            "loss": "{:.4f}".format(total_loss / max(1, batch_idx + 1)),
            "acc": "{:.4f}".format(running_acc),
        })
    metrics = compute_metrics(all_labels, all_preds, all_probs, num_classes)
    metrics["loss"] = total_loss / max(1, len(loader))
    return metrics


def metrics_to_row(degradation, checkpoint, metrics, elapsed, total_images):
    cm = metrics["confusion_matrix"]
    tn = int(cm[0, 0]) if cm.shape[0] > 0 and cm.shape[1] > 0 else 0
    fp = int(cm[0, 1]) if cm.shape[0] > 0 and cm.shape[1] > 1 else 0
    fn = int(cm[1, 0]) if cm.shape[0] > 1 and cm.shape[1] > 0 else 0
    tp = int(cm[1, 1]) if cm.shape[0] > 1 and cm.shape[1] > 1 else 0
    return {
        "model": "inceptionv3-gaussian-noise-aug",
        "checkpoint": checkpoint,
        "degradation": degradation,
        "loss": float(metrics["loss"]),
        "acc": float(metrics["accuracy"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1": float(metrics["f1"]),
        "auc": float(metrics["auc"]),
        "real_acc": float(metrics["real_accuracy"]),
        "fake_acc": float(metrics["fake_recall"]),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "total_time": elapsed,
        "time_per_image": elapsed / max(1, total_images),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Test InceptionV3 Gaussian Noise Aug under degradations")
    parser.add_argument("--real_root", default=REAL_ROOT, type=str)
    parser.add_argument("--fake_root", default=FAKE_ROOT, type=str)
    parser.add_argument("--checkpoint", default=CHECKPOINT, type=str)
    parser.add_argument("--out_dir", default=OUT_DIR, type=str)
    parser.add_argument("--batch_size", default=CLICK_RUN_BATCH_SIZE, type=int)
    parser.add_argument("--num_workers", default=CLICK_RUN_NUM_WORKERS, type=int)
    parser.add_argument("--gpu", default=CLICK_RUN_GPU, type=str)
    parser.add_argument("--num_classes", default=2, type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 80)
    print("INCEPTIONV3 GAUSSIAN NOISE AUGMENTATION DEGRADATION TEST")
    print("Checkpoint: {}".format(args.checkpoint))
    print("Batch size: {}".format(args.batch_size))
    print("Workers: {}".format(args.num_workers))
    print("Device: {}".format(device))
    print("Degradations:")
    for name, dtype, dparam in DEGRADATION_MODES:
        print("  {:<16} type={} param={}".format(name, dtype, dparam))
    print("=" * 80)

    base_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    print("\nLoading test dataset...")
    test_dataset = FFHQGANData(args.real_root, args.fake_root, "test", transform=None)
    print("Test samples: real={}, fake={}, total={}".format(
        test_dataset.real_count, test_dataset.fake_count, len(test_dataset)))

    print("\nBuilding model...")
    model = build_model("inceptionv3", args.num_classes)
    model = model.to(device)
    loaded = load_model_state(model, args.checkpoint, device)
    print("Loaded state_dict keys: {}".format(loaded))
    model.eval()

    loss_fn = nn.CrossEntropyLoss()
    rows = []
    for degradation, degrade_type, degrade_param in DEGRADATION_MODES:
        eval_dataset = DeterministicNoiseTestDataset(
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
        start = time.time()
        metrics = evaluate(
            model,
            loader,
            loss_fn,
            device,
            args.num_classes,
            desc="{} Eval".format(degradation),
        )
        elapsed = time.time() - start
        row = metrics_to_row(degradation, args.checkpoint, metrics, elapsed, len(eval_dataset))
        rows.append(row)
        print(
            "{}: acc={:.4f}, precision={:.4f}, recall={:.4f}, f1={:.4f}, "
            "auc={:.4f}, real_acc={:.4f}, fake_acc={:.4f}, cm=[[{}, {}], [{}, {}]]".format(
                degradation,
                row["acc"], row["precision"], row["recall"], row["f1"],
                row["auc"], row["real_acc"], row["fake_acc"],
                row["TN"], row["FP"], row["FN"], row["TP"],
            )
        )

    csv_path = os.path.join(args.out_dir, "all_degradation_results_gaussian_aug.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    txt_path = os.path.join(args.out_dir, "all_degradation_results_gaussian_aug.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Model: inceptionv3-gaussian-noise-aug\n")
        f.write("Checkpoint: {}\n\n".format(args.checkpoint))
        f.write("{:<18} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8} {:<4} {:<4} {:<4} {:<4}\n".format(
            "Degradation", "Acc", "Prec", "Recall", "F1", "AUC", "Real", "Fake", "TN", "FP", "FN", "TP"))
        for row in rows:
            f.write("{:<18} {:<8.4f} {:<8.4f} {:<8.4f} {:<8.4f} {:<8.4f} {:<8.4f} {:<8.4f} {:<4} {:<4} {:<4} {:<4}\n".format(
                row["degradation"], row["acc"], row["precision"], row["recall"],
                row["f1"], row["auc"], row["real_acc"], row["fake_acc"],
                row["TN"], row["FP"], row["FN"], row["TP"]))

    print("\nSaved CSV: {}".format(csv_path))
    print("Saved TXT: {}".format(txt_path))


if __name__ == "__main__":
    main()
