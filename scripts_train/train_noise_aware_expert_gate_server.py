import os
import sys
import argparse
import csv
import hashlib
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader.ffhq_gan_dataset import FFHQGANData, pil_loader
from model.noise_aware_expert_gate import (
    noise_aware_expert_gate,
    compute_noise_gate_features,
)


REAL_ROOT = r"E:\FFHQ_off"
FAKE_ROOT = r"E:\archive\GAN_Generated_Fake_Images\fake"
CHECKPOINT_DIR = r"E:\cbam_ftt_fusion\CBAM.PyTorch-master\weights\noise-aware-expert-gate"

DEFAULT_EPOCHS = 20
DEFAULT_LR = 0.001
DEFAULT_PATIENCE = 5
DEFAULT_GPU = "0"

# ==============================
# Click-run default configuration
# ==============================
CLICK_RUN_DEBUG_SUBSET = None
CLICK_RUN_EPOCHS_OVERRIDE = None
CLICK_RUN_BATCH_SIZE = 64
CLICK_RUN_NUM_WORKERS = 0


DEGRADATION_CLASSES = [
    ("clean", None),
    ("noise10", 10),
    ("noise25", 25),
    ("noise50", 50),
]


def noise_degrade(img, std, seed):
    img = img.convert("RGB")
    arr = np.array(img).astype(np.float32)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, std, arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    from PIL import Image
    return Image.fromarray(arr)


class NoiseGateDataset(Dataset):
    def __init__(self, base_dataset, transform, base_seed=42):
        self.base_dataset = base_dataset
        self.transform = transform
        self.base_seed = base_seed

    def __len__(self):
        return len(self.base_dataset.samples) * len(DEGRADATION_CLASSES)

    def __getitem__(self, index):
        sample_index = index // len(DEGRADATION_CLASSES)
        gate_label = index % len(DEGRADATION_CLASSES)
        path, _, _ = self.base_dataset.samples[sample_index]
        img = pil_loader(path)
        _, noise_std = DEGRADATION_CLASSES[gate_label]
        if noise_std is not None:
            key = "{}|noise|{}|{}".format(path, noise_std, self.base_seed)
            seed = int(hashlib.md5(key.encode()).digest()[:4].hex(), 16)
            img = noise_degrade(img, noise_std, seed)
        return self.transform(img), gate_label


def make_debug_subset(dataset, subset_size):
    if subset_size is None or subset_size <= 0:
        return dataset
    return Subset(dataset, list(range(min(subset_size, len(dataset)))))


def compute_feature_distribution(loader, device, max_batches=20):
    rows = {name: [] for name, _ in DEGRADATION_CLASSES}
    with torch.no_grad():
        for batch_idx, (inputs, gate_labels) in enumerate(loader):
            if batch_idx >= max_batches:
                break
            inputs = inputs.to(device)
            features = compute_noise_gate_features(inputs).cpu().numpy()
            labels = gate_labels.numpy()
            for idx, (name, _) in enumerate(DEGRADATION_CLASSES):
                mask = labels == idx
                if np.any(mask):
                    rows[name].append(features[mask])
    summary = {}
    for name, chunks in rows.items():
        if chunks:
            values = np.concatenate(chunks, axis=0)
            summary[name] = {
                "mean": values.mean(axis=0),
                "std": values.std(axis=0),
            }
    return summary


def run_epoch(model, loader, loss_fn, optimizer, device, train_mode):
    model.train(train_mode)
    total_loss = 0.0
    all_preds, all_labels = [], []
    desc = "Train Gate" if train_mode else "Val Gate"
    pbar = tqdm(loader, desc=desc, dynamic_ncols=True, leave=True)
    for batch_idx, (inputs, gate_labels) in enumerate(pbar):
        inputs = inputs.to(device)
        gate_labels = gate_labels.to(device)
        with torch.set_grad_enabled(train_mode):
            features = compute_noise_gate_features(inputs)
            logits = model(features)
            loss = loss_fn(logits, gate_labels)
            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(gate_labels.detach().cpu().numpy())
        pbar.set_postfix({
            "loss": "{:.4f}".format(total_loss / max(1, batch_idx + 1)),
            "acc": "{:.4f}".format(accuracy_score(all_labels, all_preds)),
        })
    metrics = {
        "loss": total_loss / max(1, len(loader)),
        "accuracy": accuracy_score(all_labels, all_preds),
        "confusion_matrix": confusion_matrix(all_labels, all_preds, labels=[0, 1, 2, 3]),
    }
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train Noise-Aware Expert Gate")
    parser.add_argument("--real_root", default=REAL_ROOT, type=str)
    parser.add_argument("--fake_root", default=FAKE_ROOT, type=str)
    parser.add_argument("--checkpoint_dir", default=CHECKPOINT_DIR, type=str)
    parser.add_argument("--batch_size", default=CLICK_RUN_BATCH_SIZE, type=int)
    parser.add_argument("--num_workers", default=CLICK_RUN_NUM_WORKERS, type=int)
    parser.add_argument("--epochs", default=CLICK_RUN_EPOCHS_OVERRIDE if CLICK_RUN_EPOCHS_OVERRIDE is not None else DEFAULT_EPOCHS, type=int)
    parser.add_argument("--lr", default=DEFAULT_LR, type=float)
    parser.add_argument("--patience", default=DEFAULT_PATIENCE, type=int)
    parser.add_argument("--gpu", default=DEFAULT_GPU, type=str)
    parser.add_argument("--debug_subset", default=CLICK_RUN_DEBUG_SUBSET, type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print("=" * 70)
    print("Noise-Aware Expert Gate Training")
    print("CLICK_RUN_DEBUG_SUBSET = {}".format(CLICK_RUN_DEBUG_SUBSET))
    print("CLICK_RUN_EPOCHS_OVERRIDE = {}".format(CLICK_RUN_EPOCHS_OVERRIDE))
    print("CLICK_RUN_BATCH_SIZE = {}".format(CLICK_RUN_BATCH_SIZE))
    print("CLICK_RUN_NUM_WORKERS = {}".format(CLICK_RUN_NUM_WORKERS))
    print("Checkpoint save dir: {}".format(args.checkpoint_dir))
    print("Gate labels: 0=clean, 1=noise10, 2=noise25, 3=noise50")
    print("Test set is not used for gate training or threshold tuning.")
    print("=" * 70)

    base_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    print("\nLoading train/val datasets...")
    train_base = FFHQGANData(args.real_root, args.fake_root, "train", transform=None)
    val_base = FFHQGANData(args.real_root, args.fake_root, "val", transform=None)
    print("train base samples: {}".format(len(train_base)))
    print("val base samples: {}".format(len(val_base)))

    train_dataset = NoiseGateDataset(train_base, base_transform, base_seed=42)
    val_dataset = NoiseGateDataset(val_base, base_transform, base_seed=43)
    train_dataset = make_debug_subset(train_dataset, args.debug_subset)
    val_dataset = make_debug_subset(val_dataset, args.debug_subset)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = noise_aware_expert_gate(num_features=7, hidden_dim=32, num_classes=4).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    best_epoch = 0
    no_improve = 0
    log_rows = []

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_metrics = run_epoch(model, train_loader, loss_fn, optimizer, device, train_mode=True)
        val_metrics = run_epoch(model, val_loader, loss_fn, optimizer, device, train_mode=False)
        elapsed = time.time() - start
        print(
            "Epoch [{}/{}] | train_gate_loss={:.4f} | train_gate_acc={:.4f} | "
            "val_gate_loss={:.4f} | val_gate_acc={:.4f} | time={:.1f}s".format(
                epoch, args.epochs,
                train_metrics["loss"], train_metrics["accuracy"],
                val_metrics["loss"], val_metrics["accuracy"], elapsed,
            )
        )
        print("Val degradation confusion matrix:")
        print(val_metrics["confusion_matrix"])
        log_rows.append([
            epoch, train_metrics["loss"], train_metrics["accuracy"],
            val_metrics["loss"], val_metrics["accuracy"],
        ])

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = float(val_metrics["accuracy"])
            best_epoch = epoch
            no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_val_gate_acc": best_val_acc,
                "args": vars(args),
                "feature_names": [
                    "E_low_ratio", "E_mid_ratio", "E_high_ratio",
                    "E_high_over_E_mid", "E_mid_over_E_low",
                    "image_std", "high_freq_energy",
                ],
                "degradation_classes": [name for name, _ in DEGRADATION_CLASSES],
            }, os.path.join(args.checkpoint_dir, "best_gate.pth"))
            print("*** Saved best gate (val_gate_acc={:.4f}) ***".format(best_val_acc))
        else:
            no_improve += 1

        if val_metrics["accuracy"] < 0.85:
            print("Gate accuracy is low; feature distribution sample:")
            for name, stats in compute_feature_distribution(val_loader, device).items():
                print("  {} mean={}".format(name, np.round(stats["mean"], 6).tolist()))
                print("  {} std ={}".format(name, np.round(stats["std"], 6).tolist()))

        if no_improve >= args.patience:
            print("Early stopping triggered.")
            break

    with open(os.path.join(args.checkpoint_dir, "gate_training_log.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_gate_loss", "train_gate_accuracy", "val_gate_loss", "val_gate_accuracy"])
        writer.writerows(log_rows)

    print("\nTRAINING FINISHED")
    print("Best epoch: {} | best val_gate_acc: {:.4f}".format(best_epoch, best_val_acc))
    print("Best gate: {}".format(os.path.join(args.checkpoint_dir, "best_gate.pth")))


if __name__ == "__main__":
    main()
