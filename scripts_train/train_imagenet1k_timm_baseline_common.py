import os
import sys
import csv
import time
import json
import argparse
import hashlib
import re
import warnings
import numpy as np
from PIL import Image

warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
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

try:
    import timm
    _TIMM_AVAILABLE = True
except ImportError:
    timm = None
    _TIMM_AVAILABLE = False


REAL_ROOT = r"E:\FFHQ_off"
FAKE_ROOT = r"E:\archive\GAN_Generated_Fake_Images\fake"
NUM_CLASSES = 2
GAN_NAMES = ["ProGAN", "StyleGAN2", "StyleGAN3"]

DEFAULT_BATCH_SIZE = 8
DEFAULT_ACCUMULATION_STEPS = 4
DEFAULT_LR = 0.0001
DEFAULT_EPOCHS = 10
DEFAULT_PATIENCE = 3
DEFAULT_NUM_WORKERS = 4
DEFAULT_GPU = "0"

DISALLOWED_PRETRAIN_TAGS = [
    "in21k", "in22k", "in12k", "jft", "ns", "noisy", "clip",
    "laion", "mae", "fcmae", "ssl", "swag",
]


try:
    from torch.amp import autocast, GradScaler
    _AMP = "new"
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
    _AMP = "legacy"


def amp_context(use_amp):
    if _AMP == "new":
        return autocast("cuda", enabled=use_amp)
    return autocast(enabled=use_amp)


def make_scaler(use_amp):
    if not use_amp:
        return None
    if _AMP == "new":
        return GradScaler("cuda")
    return GradScaler()


def get_gpu_mem():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 ** 3)
    return 0.0


def print_dataset_info(name, dataset, expected_real, expected_fake, expected_sources):
    print("  {}:".format(name))
    print("    real = {}  (expected {})".format(dataset.real_count, expected_real))
    print("    fake = {}  (expected {})".format(dataset.fake_count, expected_fake))
    print("    total = {}  (expected {})".format(len(dataset), expected_real + expected_fake))
    src_counts = dataset.get_gan_source_counts()
    for gan in GAN_NAMES:
        print("    {} = {}  (expected {})".format(
            gan, src_counts.get(gan, 0), expected_sources[gan]))
    assert dataset.real_count == expected_real
    assert dataset.fake_count == expected_fake
    for gan in GAN_NAMES:
        assert src_counts.get(gan, 0) == expected_sources[gan]


def make_debug_subset(dataset, subset_size):
    if subset_size is None or subset_size <= 0:
        return dataset
    return Subset(dataset, list(range(min(int(subset_size), len(dataset)))))


def compute_metrics(labels, preds, probs, num_classes=2):
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    probs = np.asarray(probs)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    auc = 0.0
    if num_classes == 2 and len(np.unique(labels)) > 1:
        auc = roc_auc_score(labels, probs[:, 1])
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "auc": float(auc),
        "real_acc": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "fake_acc": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "confusion_matrix": cm.tolist(),
    }


def noise_degrade(img, std, seed):
    img = img.convert("RGB")
    arr = np.array(img).astype(np.float32)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, float(std), arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


class DeterministicNoiseDataset(Dataset):
    def __init__(self, dataset, base_transform, noise_sigma=None, base_seed=42):
        self.dataset = dataset
        self.base_transform = base_transform
        self.noise_sigma = noise_sigma
        self.base_seed = int(base_seed)
        self.samples = dataset.samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label, _ = self.samples[index]
        img = pil_loader(path)
        if self.noise_sigma is not None:
            key = "{}|noise|{}|{}".format(path, self.noise_sigma, self.base_seed)
            seed = int(hashlib.md5(key.encode()).digest()[:4].hex(), 16)
            img = noise_degrade(img, self.noise_sigma, seed)
        return self.base_transform(img), int(label)


def short_cfg(cfg):
    keys = [
        "architecture", "tag", "url", "hf_hub_id", "input_size",
        "mean", "std", "crop_pct", "interpolation", "num_classes",
        "classifier", "first_conv",
    ]
    return {key: cfg.get(key, None) for key in keys}


def cfg_text(model_name, cfg):
    parts = [model_name]
    for key in ("architecture", "tag", "url", "hf_hub_id"):
        value = cfg.get(key, None)
        if value:
            parts.append(str(value))
    return " ".join(parts).lower()


def cfg_tokens(model_name, cfg):
    values = [model_name]
    for key in ("architecture", "tag", "hf_hub_id"):
        value = cfg.get(key, None)
        if value:
            values.append(str(value))
    text = " ".join(values).lower()
    return [token for token in re.split(r"[^a-z0-9]+", text) if token]


def is_disallowed_pretrain(model_name, cfg):
    tokens = set(cfg_tokens(model_name, cfg))
    return any(tag in tokens for tag in DISALLOWED_PRETRAIN_TAGS)


def has_in1k_signal(model_name, cfg):
    text = cfg_text(model_name, cfg)
    return ("in1k" in text) or ("imagenet-1k" in text) or ("imagenet1k" in text)


def resolve_model_name(requested_name, search_pattern, candidates):
    if not _TIMM_AVAILABLE:
        raise RuntimeError("timm is not installed in this Python environment.")
    available_pretrained = set(timm.list_models(search_pattern, pretrained=True))
    available_all = set(timm.list_models(search_pattern))

    for name in [requested_name] + list(candidates):
        if name in available_pretrained or name in available_all:
            return name

    print("Requested/candidate model names were not found.")
    print("Search pattern: {}".format(search_pattern))
    print("Available pretrained models:")
    for item in sorted(available_pretrained):
        print("  {}".format(item))
    print("Available architectures:")
    for item in sorted(available_all):
        print("  {}".format(item))
    raise RuntimeError("No usable ImageNet-1K candidate found for {}".format(requested_name))


def classifier_out_features(classifier):
    if hasattr(classifier, "out_features"):
        return classifier.out_features
    if isinstance(classifier, nn.Sequential):
        for module in reversed(classifier):
            if hasattr(module, "out_features"):
                return module.out_features
    for module in reversed(list(classifier.modules())):
        if module is not classifier and hasattr(module, "out_features"):
            return module.out_features
    return None


def build_timm_model(requested_name, search_pattern, candidates, display_name, num_classes, pretrained):
    actual_name = resolve_model_name(requested_name, search_pattern, candidates)
    print("\nBuilding {}...".format(display_name))
    print("  requested model name: {}".format(requested_name))
    print("  actual model name: {}".format(actual_name))
    print("  pretrained: {}".format(pretrained))
    print("  fairness requirement: ImageNet-1K supervised pretraining only.")
    print("  disallowed tags: {}".format(DISALLOWED_PRETRAIN_TAGS))

    model = timm.create_model(actual_name, pretrained=pretrained)
    cfg = getattr(model, "pretrained_cfg", None) or getattr(model, "default_cfg", {})
    default_cfg = getattr(model, "default_cfg", {})

    if pretrained:
        if is_disallowed_pretrain(actual_name, cfg):
            raise RuntimeError(
                "Refusing non-fair pretrained weights for {}. cfg={}".format(
                    actual_name, short_cfg(cfg)))
        if not has_in1k_signal(actual_name, cfg):
            print("  WARNING: no explicit in1k tag found in name/cfg.")
            print("  Please verify the printed pretrained_cfg before using this run in the paper.")

    classifier_before = model.get_classifier()
    before_num_classes = cfg.get("num_classes", getattr(model, "num_classes", None))
    print("  pretrained_cfg: {}".format(short_cfg(cfg)))
    print("  default_cfg: {}".format(short_cfg(default_cfg)))
    print("  weights url: {}".format(cfg.get("url", None)))
    print("  hf_hub_id: {}".format(cfg.get("hf_hub_id", None)))
    print("  input_size: {}".format(cfg.get("input_size", None)))
    print("  mean/std: {} / {}".format(cfg.get("mean", None), cfg.get("std", None)))
    print("  crop_pct: {}".format(cfg.get("crop_pct", None)))
    print("  interpolation: {}".format(cfg.get("interpolation", None)))
    print("  classifier before: {}".format(classifier_before))
    print("  num_classes before: {}".format(before_num_classes))

    model.reset_classifier(num_classes=num_classes)
    classifier_after = model.get_classifier()
    after_features = classifier_out_features(classifier_after)
    head_ok = after_features == num_classes
    print("  classifier after: {}".format(classifier_after))
    print("  head out_features after: {}".format(after_features))
    print("  head replaced with 2-class output: {}".format(head_ok))
    if not head_ok:
        raise RuntimeError("{} classifier head was not replaced with {} classes.".format(display_name, num_classes))
    return model, cfg, actual_name


def make_transforms():
    return (
        transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
    )


def train_one_epoch(model, loader, loss_fn, optimizer, scaler, device, args, epoch):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    labels_all, preds_all = [], []
    total_loss = 0.0
    pbar = tqdm(loader, desc="Train Epoch {}/{}".format(epoch, args.epochs),
                dynamic_ncols=True, leave=True)
    for step, (inputs, labels) in enumerate(pbar):
        inputs = inputs.to(device)
        labels = labels.to(device)
        with amp_context(args.use_amp):
            logits = model(inputs)
            loss = loss_fn(logits, labels) / args.accumulation_steps

        if args.use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % args.accumulation_steps == 0 or (step + 1) == len(loader):
            if args.use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * args.accumulation_steps
        preds = torch.argmax(logits, dim=1)
        labels_all.extend(labels.detach().cpu().numpy().tolist())
        preds_all.extend(preds.detach().cpu().numpy().tolist())
        pbar.set_postfix({
            "loss": "{:.4f}".format(total_loss / (step + 1)),
            "acc": "{:.4f}".format(accuracy_score(labels_all, preds_all)),
            "gpu": "{:.1f}G".format(get_gpu_mem()),
        })
    return {
        "loss": total_loss / len(loader),
        "accuracy": float(accuracy_score(labels_all, preds_all)),
    }


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, num_classes, desc):
    model.eval()
    labels_all, preds_all, probs_all = [], [], []
    total_loss = 0.0
    total_images = 0
    start = time.time()
    pbar = tqdm(loader, desc=desc, dynamic_ncols=True, leave=True)
    for inputs, labels in pbar:
        inputs = inputs.to(device)
        labels = labels.to(device)
        logits = model(inputs)
        loss = loss_fn(logits, labels)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        total_loss += loss.item()
        total_images += inputs.size(0)
        labels_all.extend(labels.detach().cpu().numpy().tolist())
        preds_all.extend(preds.detach().cpu().numpy().tolist())
        probs_all.extend(probs.detach().cpu().numpy().tolist())
        pbar.set_postfix({
            "loss": "{:.4f}".format(loss.item()),
            "acc": "{:.4f}".format(accuracy_score(labels_all, preds_all)),
        })
    elapsed = time.time() - start
    metrics = compute_metrics(labels_all, preds_all, probs_all, num_classes)
    metrics["loss"] = total_loss / len(loader)
    metrics["eval_time_sec"] = elapsed
    metrics["images_per_sec"] = total_images / elapsed if elapsed > 0 else 0.0
    return metrics


def run_noise_tests(model, raw_test_dataset, base_transform, loss_fn, device, args):
    rows = []
    modes = [("clean", None), ("noise10", 10), ("noise25", 25), ("noise50", 50)]
    for mode_name, sigma in modes:
        eval_dataset = DeterministicNoiseDataset(raw_test_dataset, base_transform, noise_sigma=sigma)
        loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=False)
        metrics = evaluate(model, loader, loss_fn, device, args.num_classes, "Test {}".format(mode_name))
        row = {
            "degradation": mode_name,
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
            "eval_time_sec": metrics["eval_time_sec"],
            "images_per_sec": metrics["images_per_sec"],
        }
        rows.append(row)
        print("{}: acc={:.4f} precision={:.4f} recall={:.4f} f1={:.4f} auc={:.4f} real_acc={:.4f} fake_acc={:.4f} cm={}".format(
            mode_name, row["acc"], row["precision"], row["recall"], row["f1"],
            row["auc"], row["real_acc"], row["fake_acc"],
            [[row["TN"], row["FP"]], [row["FN"], row["TP"]]],
        ))
    csv_path = os.path.join(args.checkpoint_dir, "results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("Saved test results: {}".format(csv_path))


def parse_common_args(model_spec):
    parser = argparse.ArgumentParser(description="Train {} ImageNet-1K baseline".format(model_spec["display_name"]))
    parser.add_argument("--real_root", default=REAL_ROOT, type=str)
    parser.add_argument("--fake_root", default=FAKE_ROOT, type=str)
    parser.add_argument("--checkpoint_dir", default=model_spec["checkpoint_dir"], type=str)
    parser.add_argument("--model_name", default=model_spec["model_name"], type=str)
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no_pretrained", action="store_false", dest="pretrained")
    parser.add_argument("--batch_size", default=DEFAULT_BATCH_SIZE, type=int)
    parser.add_argument("--accumulation_steps", default=DEFAULT_ACCUMULATION_STEPS, type=int)
    parser.add_argument("--lr", default=DEFAULT_LR, type=float)
    parser.add_argument("--epochs", default=DEFAULT_EPOCHS, type=int)
    parser.add_argument("--patience", default=DEFAULT_PATIENCE, type=int)
    parser.add_argument("--num_workers", default=DEFAULT_NUM_WORKERS, type=int)
    parser.add_argument("--gpu", default=DEFAULT_GPU, type=str)
    parser.add_argument("--num_classes", default=NUM_CLASSES, type=int)
    parser.add_argument("--debug_subset", default=None, type=int)
    return parser.parse_args()


def main_for_model(model_spec):
    args = parse_common_args(model_spec)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.use_amp = torch.cuda.is_available()
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print("=" * 80)
    print("Model: {}".format(model_spec["display_name"]))
    print("Requested timm model: {}".format(args.model_name))
    print("Checkpoint dir: {}".format(args.checkpoint_dir))
    print("Pretrained: {}".format(args.pretrained))
    print("Pretraining requirement: ImageNet-1K supervised only")
    print("optimizer: Adam")
    print("learning_rate: {}".format(args.lr))
    print("epochs: {}".format(args.epochs))
    print("batch_size: {}".format(args.batch_size))
    print("accumulation_steps: {}".format(args.accumulation_steps))
    print("effective batch size: {}".format(args.batch_size * args.accumulation_steps))
    print("AMP: {} ({})".format(args.use_amp, _AMP))
    print("=" * 80)

    model, cfg, actual_model_name = build_timm_model(
        args.model_name,
        model_spec["search_pattern"],
        model_spec["candidates"],
        model_spec["display_name"],
        args.num_classes,
        args.pretrained,
    )
    train_transform, val_transform = make_transforms()

    print("\nLoading datasets...")
    train_dataset = FFHQGANData(args.real_root, args.fake_root, "train", transform=train_transform)
    val_dataset = FFHQGANData(args.real_root, args.fake_root, "val", transform=val_transform)
    raw_test_dataset = FFHQGANData(args.real_root, args.fake_root, "test", transform=None)
    print_dataset_info("train", train_dataset, 15000, 15000,
                       {"ProGAN": 5000, "StyleGAN2": 5000, "StyleGAN3": 5000})
    print_dataset_info("val", val_dataset, 4200, 4200,
                       {"ProGAN": 1400, "StyleGAN2": 1400, "StyleGAN3": 1400})
    print_dataset_info("test", raw_test_dataset, 2100, 2100,
                       {"ProGAN": 700, "StyleGAN2": 700, "StyleGAN3": 700})

    train_dataset = make_debug_subset(train_dataset, args.debug_subset)
    val_dataset = make_debug_subset(val_dataset, args.debug_subset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total params: {:,}".format(total_params))
    print("Trainable params: {:,}".format(trainable_params))

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scaler = make_scaler(args.use_amp)

    best_val_acc = 0.0
    best_val_f1 = 0.0
    best_epoch = 0
    no_improve = 0
    log_rows = []
    start_train = time.time()

    print("\nTRAINING START")
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_metrics = train_one_epoch(model, train_loader, loss_fn, optimizer, scaler, device, args, epoch)
        val_metrics = evaluate(model, val_loader, loss_fn, device, args.num_classes,
                               "Val Epoch {}/{}".format(epoch, args.epochs))
        elapsed = time.time() - start
        print(
            "Epoch [{}/{}] train_loss={:.4f} train_acc={:.4f} "
            "val_loss={:.4f} val_acc={:.4f} val_prec={:.4f} val_rec={:.4f} "
            "val_f1={:.4f} val_auc={:.4f} time={:.1f}s".format(
                epoch, args.epochs,
                train_metrics["loss"], train_metrics["accuracy"],
                val_metrics["loss"], val_metrics["accuracy"], val_metrics["precision"],
                val_metrics["recall"], val_metrics["f1"], val_metrics["auc"], elapsed,
            )
        )
        log_rows.append([
            epoch, train_metrics["loss"], train_metrics["accuracy"],
            val_metrics["loss"], val_metrics["accuracy"], val_metrics["precision"],
            val_metrics["recall"], val_metrics["f1"], val_metrics["auc"], elapsed,
        ])

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_val_f1 = val_metrics["f1"]
            best_epoch = epoch
            no_improve = 0
            torch.save({
                "epoch": int(epoch),
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val_acc": float(best_val_acc),
                "best_val_f1": float(best_val_f1),
                "model_name": actual_model_name,
                "requested_model_name": args.model_name,
                "pretrained": args.pretrained,
                "pretrained_cfg": dict(cfg),
                "total_params": int(total_params),
                "trainable_params": int(trainable_params),
                "args": vars(args),
            }, os.path.join(args.checkpoint_dir, "best_model.pth"))
            print("  *** Saved best model (val_acc={:.4f}, val_f1={:.4f}) ***".format(best_val_acc, best_val_f1))
        else:
            no_improve += 1
            print("  No improvement ({}/{})".format(no_improve, args.patience))
        if no_improve >= args.patience:
            print("Early stopping triggered.")
            break

    torch.save({
        "epoch": int(epoch),
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_val_acc": float(best_val_acc),
        "best_val_f1": float(best_val_f1),
        "best_epoch": int(best_epoch),
        "model_name": actual_model_name,
        "requested_model_name": args.model_name,
        "pretrained": args.pretrained,
        "pretrained_cfg": dict(cfg),
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "args": vars(args),
    }, os.path.join(args.checkpoint_dir, "last_model.pth"))

    with open(os.path.join(args.checkpoint_dir, "training_log.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "train_loss", "train_acc", "val_loss", "val_acc",
            "val_precision", "val_recall", "val_f1", "val_auc", "time_sec",
        ])
        writer.writerows(log_rows)

    summary = {
        "display_name": model_spec["display_name"],
        "actual_model_name": actual_model_name,
        "best_epoch": int(best_epoch),
        "best_val_acc": float(best_val_acc),
        "best_val_f1": float(best_val_f1),
        "total_train_time_sec": time.time() - start_train,
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "pretrained_cfg": short_cfg(cfg),
        "fairness_note": "ImageNet-1K supervised pretrained weights; same FFHQ/GAN split and same training hyperparameters.",
    }
    with open(os.path.join(args.checkpoint_dir, "training_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nTRAINING FINISHED")
    print("Best epoch: {} | best val_acc: {:.4f} | best val_f1: {:.4f}".format(best_epoch, best_val_acc, best_val_f1))
    print("Best model: {}".format(os.path.join(args.checkpoint_dir, "best_model.pth")))
    print("Last model: {}".format(os.path.join(args.checkpoint_dir, "last_model.pth")))
    print("Training total time: {:.1f}s".format(summary["total_train_time_sec"]))

    best_path = os.path.join(args.checkpoint_dir, "best_model.pth")
    if os.path.exists(best_path):
        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        run_noise_tests(model, raw_test_dataset, val_transform, loss_fn, device, args)
