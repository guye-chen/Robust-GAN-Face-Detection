import os
import sys
import csv
import json
import time
import hashlib
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader.ffhq_gan_dataset import FFHQGANData, pil_loader
from model.frequency_reliability_weak_strong_gate import (
    FEATURE_NAMES,
    CLASS_NAMES,
    frequency_reliability_weak_strong_gate,
    compute_frequency_reliability_features,
)


REAL_ROOT = r"E:\FFHQ_off"
FAKE_ROOT = r"E:\archive\GAN_Generated_Fake_Images\fake"
CHECKPOINT_DIR = r"E:\cbam_ftt_fusion\CBAM.PyTorch-master\weights\noise-aware-weak-strong-gate"

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
CLICK_RUN_USE_CLASS_WEIGHT = False


DEGRADATION_CLASSES = [
    ("clean", None, 0),
    ("noise10", 10, 0),
    ("noise25", 25, 1),
    ("noise50", 50, 1),
]


def noise_degrade(img, std, seed):
    img = img.convert("RGB")
    arr = np.array(img).astype(np.float32)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, float(std), arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


class WeakStrongGateDataset(Dataset):
    def __init__(self, base_dataset, transform, base_seed=42):
        self.base_dataset = base_dataset
        self.transform = transform
        self.base_seed = int(base_seed)

    def __len__(self):
        return len(self.base_dataset.samples) * len(DEGRADATION_CLASSES)

    def __getitem__(self, index):
        sample_index = index // len(DEGRADATION_CLASSES)
        degradation_index = index % len(DEGRADATION_CLASSES)
        path, _, _ = self.base_dataset.samples[sample_index]
        img = pil_loader(path)
        _, noise_std, weak_strong_label = DEGRADATION_CLASSES[degradation_index]
        if noise_std is not None:
            key = "{}|noise|{}|{}".format(path, noise_std, self.base_seed)
            seed = int(hashlib.md5(key.encode()).digest()[:4].hex(), 16)
            img = noise_degrade(img, noise_std, seed)
        return self.transform(img), int(weak_strong_label)


def make_debug_subset(dataset, subset_size):
    if subset_size is None or subset_size <= 0:
        return dataset
    return Subset(dataset, list(range(min(int(subset_size), len(dataset)))))


def run_epoch(model, loader, loss_fn, optimizer, device, train_mode):
    model.train(train_mode)
    total_loss = 0.0
    all_preds, all_labels = [], []
    desc = "Train Weak-Strong Gate" if train_mode else "Val Weak-Strong Gate"
    pbar = tqdm(loader, desc=desc, dynamic_ncols=True, leave=True)
    for batch_idx, (inputs, labels) in enumerate(pbar):
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(train_mode):
            features = compute_frequency_reliability_features(inputs)
            logits = model(features)
            loss = loss_fn(logits, labels)
            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())
        pbar.set_postfix({
            "loss": "{:.4f}".format(total_loss / max(1, batch_idx + 1)),
            "acc": "{:.4f}".format(accuracy_score(all_labels, all_preds)),
        })
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    return {
        "loss": total_loss / max(1, len(loader)),
        "accuracy": float(accuracy_score(all_labels, all_preds)),
        "confusion_matrix": cm,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train Frequency-Reliability Weak/Strong Noise Gate")
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
    parser.add_argument("--use_class_weight", action="store_true", default=CLICK_RUN_USE_CLASS_WEIGHT)
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print("=" * 70)
    print("Frequency-Reliability Weak/Strong Noise Gate Training")
    print("CLICK_RUN_DEBUG_SUBSET = {}".format(CLICK_RUN_DEBUG_SUBSET))
    print("CLICK_RUN_EPOCHS_OVERRIDE = {}".format(CLICK_RUN_EPOCHS_OVERRIDE))
    print("CLICK_RUN_BATCH_SIZE = {}".format(CLICK_RUN_BATCH_SIZE))
    print("CLICK_RUN_NUM_WORKERS = {}".format(CLICK_RUN_NUM_WORKERS))
    print("Checkpoint save dir: {}".format(args.checkpoint_dir))
    print("Gate classes: 0=weak_noise(clean/noise10), 1=strong_noise(noise25/noise50)")
    print("Routing after training: weak -> Stability-Reg, strong -> High-Noise Bias-Calib")
    print("Feature names: {}".format(FEATURE_NAMES))
    print("Test set is not used for gate training or threshold tuning.")
    print("=" * 70)

    config = {
        "gate_name": "frequency-reliability-weak-strong-gate",
        "feature_names": FEATURE_NAMES,
        "class_names": CLASS_NAMES,
        "degradation_to_class": {name: CLASS_NAMES[label] for name, _, label in DEGRADATION_CLASSES},
        "routing": {
            "weak_noise": "inceptionv3-nmfgf-noise-ra-stability-reg",
            "strong_noise": "inceptionv3-high-noise-bias-calib",
        },
        "args": vars(args),
    }
    with open(os.path.join(args.checkpoint_dir, "gate_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

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

    train_dataset = make_debug_subset(WeakStrongGateDataset(train_base, base_transform, base_seed=42), args.debug_subset)
    val_dataset = make_debug_subset(WeakStrongGateDataset(val_base, base_transform, base_seed=43), args.debug_subset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = frequency_reliability_weak_strong_gate(num_features=len(FEATURE_NAMES), hidden_dim=32).to(device)
    if args.use_class_weight:
        class_weight = torch.tensor([1.0, 1.0], dtype=torch.float32, device=device)
        print("CrossEntropy class_weight: {}".format(class_weight.detach().cpu().tolist()))
    else:
        class_weight = None
        print("CrossEntropy class_weight: None")
    loss_fn = nn.CrossEntropyLoss(weight=class_weight)
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
        print("Val weak/strong confusion matrix rows=true [weak,strong], cols=pred [weak,strong]:")
        print(val_metrics["confusion_matrix"])
        log_rows.append([
            epoch, train_metrics["loss"], train_metrics["accuracy"],
            val_metrics["loss"], val_metrics["accuracy"],
            int(val_metrics["confusion_matrix"][0, 0]),
            int(val_metrics["confusion_matrix"][0, 1]),
            int(val_metrics["confusion_matrix"][1, 0]),
            int(val_metrics["confusion_matrix"][1, 1]),
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
                "feature_names": FEATURE_NAMES,
                "class_names": CLASS_NAMES,
                "degradation_to_class": config["degradation_to_class"],
                "routing": config["routing"],
            }, os.path.join(args.checkpoint_dir, "best_weak_strong_gate.pth"))
            print("*** Saved best weak/strong gate (val_gate_acc={:.4f}) ***".format(best_val_acc))
        else:
            no_improve += 1
            print("No improvement ({}/{})".format(no_improve, args.patience))

        if no_improve >= args.patience:
            print("Early stopping triggered.")
            break

    with open(os.path.join(args.checkpoint_dir, "weak_strong_gate_training_log.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "train_gate_loss", "train_gate_accuracy", "val_gate_loss", "val_gate_accuracy",
            "val_TW_PW", "val_TW_PS", "val_TS_PW", "val_TS_PS",
        ])
        writer.writerows(log_rows)

    print("\nTRAINING FINISHED")
    print("Best epoch: {} | best val_gate_acc: {:.4f}".format(best_epoch, best_val_acc))
    print("Best weak/strong gate: {}".format(os.path.join(args.checkpoint_dir, "best_weak_strong_gate.pth")))


if __name__ == "__main__":
    main()
