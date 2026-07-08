"""
All Degradation Robustness Evaluation

Tests model robustness under multiple degradation types:
  - Clean
  - JPEG compression  (90 / 80 / 70)
  - Gaussian Blur     (1.0 / 2.0 / 3.0)
  - Gaussian Noise    (10 / 25 / 50)
  - Downsample        (0.75 / 0.5 / 0.25)

Supports all models from test_jpeg_robustness.py and
test_jpeg_robustness_remaining_server.py.

Usage:
    python test_all_degradations.py -m resnet50-cbam-dense-msjf-ra --batch_size 1 --num_workers 0
    python test_all_degradations.py -m resnet50-cbam-dense-msjf-ra -c <path> --batch_size 1 --num_workers 0
"""

import os
import sys
import argparse
import csv
import warnings
import hashlib
import numpy as np
from collections import OrderedDict
from io import BytesIO

warnings.filterwarnings("ignore", category=FutureWarning)

# Global RNG for noise (fallback when no per-image seed is provided)
_rng = np.random.default_rng(42)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)
from tqdm import tqdm
from PIL import Image, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader.ffhq_gan_dataset import FFHQGANData, pil_loader
import model.resnet_cbam as resnet_cbam
import model.resnet_cbam_sfgf as resnet_cbam_sfgf
import model.resnet_cbam_mjfs as resnet_cbam_mjfs
import model.resnet_cbam_msjf_ra as resnet_cbam_msjf_ra
import model.resnet_cbam_dense_msjf_ra as resnet_cbam_dense_msjf_ra
import importlib
resnet50_nmfgf = importlib.import_module('model.resnet50_nmfgf')

try:
    import timm
    _TIMM_AVAILABLE = True
except ImportError:
    _TIMM_AVAILABLE = False


# ============================================================
# Paths & model list
# ============================================================

REAL_ROOT = r"E:\FFHQ_off"
FAKE_ROOT = r"E:\archive\GAN_Generated_Fake_Images\fake"

CHECKPOINT_MAP = {
    "resnet50": r"E:\cbam_ftt_fusion\CBAM.PyTorch-master\checkpoint\resnet50\best_model.pth",
    "mobilenetv2": r"E:\cbam_ftt_fusion\CBAM.PyTorch-master\checkpoint\mobilenetv2\best_model.pth",
    "densenet121": r"E:\cbam_ftt_fusion\CBAM.PyTorch-master\checkpoint\densenet121\best_model.pth",
    "resnet50-cbam": r"E:\cbam_ftt_fusion\CBAM.PyTorch-master\checkpoint\resnet50-cbam\best_model.pth",
    "resnet50-cbam-sfgf": r"E:\cbam_ftt_fusion\CBAM.PyTorch-master\weights\resnet50-cbam-sfgf\best_model.pth",
    "resnet50-cbam-mjfs": r"E:\cbam_ftt_fusion\CBAM.PyTorch-master\weights\resnet50-cbam-mjfs\best_model.pth",
    "resnet50-cbam-msjf-ra": r"E:\cbam_ftt_fusion\CBAM.PyTorch-master\weights\best_model_msjf_ra.pth",
    "resnet50-cbam-dense-msjf-ra": r"E:\cbam_ftt_fusion\CBAM.PyTorch-master\weights\dense_msjf_ra\best_model.pth",
    "resnet50-nmfgf": r"E:\cbam_ftt_fusion\CBAM.PyTorch-master\weights\resnet50-nmfgf\best_model.pth",
    "vgg16": r"E:\cbam_ftt_fusion\CBAM.PyTorch-master\weights\best_model_vgg16.pth",
    "vgg19": r"E:\cbam_ftt_fusion\CBAM.PyTorch-master\weights\vgg19\best_model.pth",
    "inceptionv3": r"E:\cbam_ftt_fusion\CBAM.PyTorch-master\weights\best_model_inceptionv3.pth",
    "inceptionresnetv2": r"E:\cbam_ftt_fusion\CBAM.PyTorch-master\weights\inceptionresnetv2\best_model.pth",
    "resnet50v2": r"E:\cbam_ftt_fusion\CBAM.PyTorch-master\weights\resnet50v2\best_model.pth",
    "xception": r"E:\cbam_ftt_fusion\CBAM.PyTorch-master\weights\xception\best_model.pth",
}

MODEL_CHOICES = sorted(CHECKPOINT_MAP.keys())


# ============================================================
# Degradation transforms (PIL-level)
# ============================================================

def jpeg_degrade(img, quality):
    buffer = BytesIO()
    img = img.convert("RGB")
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def blur_degrade(img, sigma):
    img = img.convert("RGB")
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))


def noise_degrade(img, std, seed=None):
    """Add Gaussian noise in PIL/numpy raw pixel space [0, 255].

    If seed is given, uses a local RNG seeded by it, making noise
    deterministic per image path + degradation.
    """
    img = img.convert("RGB")
    arr = np.array(img).astype(np.float32)
    local_rng = np.random.default_rng(seed) if seed is not None else _rng
    noise = local_rng.normal(0, std, arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def downsample_degrade(img, scale):
    img = img.convert("RGB")
    w, h = img.size
    new_w, new_h = int(w * scale), int(h * scale)
    img_small = img.resize((new_w, new_h), Image.BILINEAR)
    return img_small.resize((w, h), Image.BILINEAR)


class DegradeTransform:
    def __init__(self, base_transform, degrade_type=None, degrade_param=None):
        self.base_transform = base_transform
        self.degrade_type = degrade_type
        self.degrade_param = degrade_param

    def __call__(self, img):
        if self.degrade_type is not None:
            if self.degrade_type == "jpeg":
                img = jpeg_degrade(img, self.degrade_param)
            elif self.degrade_type == "blur":
                img = blur_degrade(img, self.degrade_param)
            elif self.degrade_type == "noise":
                img = noise_degrade(img, self.degrade_param)
            elif self.degrade_type == "downsample":
                img = downsample_degrade(img, self.degrade_param)
        return self.base_transform(img)


class DeterministicNoiseTestDataset:
    """Dataset wrapper that applies per-image deterministic noise.

    Seed = MD5(image_path | degradation | degrad_param | base_seed),
    ensuring the same image always gets identical noise across all models.
    """
    def __init__(self, dataset, base_transform, degrade_type, degrade_param, base_seed=42):
        self.dataset = dataset
        self.base_transform = base_transform
        self.degrade_type = degrade_type
        self.degrade_param = degrade_param
        self.base_seed = base_seed

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        path, label, _ = self.dataset.samples[index]
        img = pil_loader(path)

        if self.degrade_type is not None:
            if self.degrade_type == "noise":
                key = "{}|{}|{}|{}".format(path, self.degrade_type, self.degrade_param, self.base_seed)
                seed = int(hashlib.md5(key.encode()).digest()[:4].hex(), 16)
                img = noise_degrade(img, self.degrade_param, seed=seed)
            elif self.degrade_type == "jpeg":
                img = jpeg_degrade(img, self.degrade_param)
            elif self.degrade_type == "blur":
                img = blur_degrade(img, self.degrade_param)
            elif self.degrade_type == "downsample":
                img = downsample_degrade(img, self.degrade_param)

        if self.base_transform is not None:
            img = self.base_transform(img)
        return img, label


# ============================================================
# Helper
# ============================================================

def _extract_logits(outputs):
    if isinstance(outputs, tuple):
        return outputs[0]
    if hasattr(outputs, "logits"):
        return outputs.logits
    return outputs


# ============================================================
# Model loading
# ============================================================

def load_model_state(model, checkpoint_path, device):
    print("  Loading checkpoint: {}".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        print("  Checkpoint format: model_state_dict")
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        print("  Checkpoint format: state_dict")
    elif isinstance(checkpoint, dict) and any(
        k.startswith(("conv1", "layer", "features", "classifier", "fc"))
        for k in checkpoint.keys()
    ):
        state_dict = checkpoint
        print("  Checkpoint format: raw state_dict")
    else:
        raise RuntimeError(
            'Checkpoint does not contain "model_state_dict", "state_dict", '
            "or raw state_dict keys. Found keys: {}".format(
                list(checkpoint.keys())[:10] if isinstance(checkpoint, dict) else type(checkpoint)
            )
        )

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v

    try:
        model.load_state_dict(new_state_dict, strict=True)
        loaded = len(new_state_dict)
        print("  Loaded {} params (strict=True, all matched)".format(loaded))
    except RuntimeError:
        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
        loaded = len(new_state_dict) - len(unexpected_keys)
        print("  Strict load failed. Loaded {} params (strict=False)".format(loaded))
        if missing_keys:
            print("  Missing keys ({})".format(len(missing_keys)))
            for k in missing_keys[:5]:
                print("    {}".format(k))
            if len(missing_keys) > 5:
                print("    ... and {} more".format(len(missing_keys) - 5))
        if unexpected_keys:
            print("  Unexpected keys ({})".format(len(unexpected_keys)))
            for k in unexpected_keys[:5]:
                print("    {}".format(k))
            if len(unexpected_keys) > 5:
                print("    ... and {} more".format(len(unexpected_keys) - 5))

    if loaded == 0:
        model_keys = list(model.state_dict().keys())
        ckpt_keys = list(new_state_dict.keys())
        print("  ERROR: 0 params matched!")
        print("  Model keys sample (first 10): {}".format(model_keys[:10]))
        print("  Checkpoint keys sample (first 10): {}".format(ckpt_keys[:10]))
        raise RuntimeError(
            "Loaded 0 params from checkpoint! Check model structure vs checkpoint.\n"
            "Checkpoint: {}".format(checkpoint_path)
        )

    return loaded


# ============================================================
# Model building
# ============================================================

def build_model(model_type, num_classes, sfgf_hidden_channels=32):
    if model_type == "resnet50":
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_type == "mobilenetv2":
        model = models.mobilenet_v2(pretrained=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif model_type == "densenet121":
        model = models.densenet121(pretrained=False)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    elif model_type == "resnet50-cbam":
        model = resnet_cbam.resnet50_cbam(pretrained=False, num_classes=num_classes)

    elif model_type == "resnet50-cbam-sfgf":
        model = resnet_cbam_sfgf.resnet50_cbam_sfgf(
            pretrained=False, num_classes=num_classes,
            sfgf_hidden_channels=sfgf_hidden_channels,
        )

    elif model_type == "resnet50-cbam-mjfs":
        model = resnet_cbam_mjfs.resnet50_cbam_mjfs(
            pretrained=False, num_classes=num_classes,
            hidden_dim=128, jpeg_quality=90, use_layer2=True,
        )

    elif model_type == "resnet50-cbam-msjf-ra":
        model = resnet_cbam_msjf_ra.resnet50_cbam_msjf_ra(
            num_classes=num_classes,
            hidden_dim=128,
            jpeg_quality=90,
            use_layer2=True,
            ra_hidden_dim=64,
            alpha=0.5,
        )

    elif model_type == "resnet50-cbam-dense-msjf-ra":
        model = resnet_cbam_dense_msjf_ra.resnet50_cbam_dense_msjf_ra(
            num_classes=num_classes,
            hidden_dim=128,
            jpeg_quality=90,
            use_layer1=True,
            dense_dim=32,
            ra_hidden_dim=64,
            alpha=0.5,
        )

    elif model_type == "resnet50-nmfgf":
        model = resnet50_nmfgf.resnet50_nmfgf(
            num_classes=num_classes,
            pretrained_backbone=False,
        )

    elif model_type == "vgg16":
        from torchvision.models import vgg16
        model = vgg16(pretrained=False)
        model.classifier[6] = nn.Linear(4096, num_classes)

    elif model_type == "vgg19":
        from torchvision.models import vgg19
        model = vgg19(pretrained=False)
        model.classifier[6] = nn.Linear(4096, num_classes)

    elif model_type == "inceptionv3":
        from torchvision.models import inception_v3
        model = inception_v3(pretrained=False, aux_logits=True)
        model.aux_logits = False
        model.AuxLogits = None
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.transform_input = True

    elif model_type == "inceptionresnetv2":
        if not _TIMM_AVAILABLE:
            raise RuntimeError("timm is not installed. Run: pip install timm")
        model = timm.create_model("inception_resnet_v2", pretrained=False, num_classes=num_classes)

    elif model_type == "resnet50v2":
        if not _TIMM_AVAILABLE:
            raise RuntimeError("timm is not installed. Run: pip install timm")
        model = timm.create_model("resnetv2_50", pretrained=False, num_classes=num_classes)

    elif model_type == "xception":
        if not _TIMM_AVAILABLE:
            raise RuntimeError("timm is not installed. Run: pip install timm")
        model = timm.create_model("xception", pretrained=False, num_classes=num_classes)

    else:
        raise ValueError("Unknown model: {}".format(model_type))

    return model


# ============================================================
# Metrics
# ============================================================

def compute_metrics(all_labels, all_preds, all_probs, num_classes=2):
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="binary", zero_division=0)
    rec = recall_score(all_labels, all_preds, average="binary", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="binary", zero_division=0)

    auc = 0.0
    if num_classes == 2 and len(np.unique(all_labels)) > 1:
        try:
            auc = roc_auc_score(all_labels, all_probs[:, 1])
        except ValueError:
            auc = 0.0

    cm = confusion_matrix(all_labels, all_preds)

    tn = cm[0, 0] if cm.shape[0] > 0 and cm.shape[1] > 0 else 0
    fp = cm[0, 1] if cm.shape[0] > 0 and cm.shape[1] > 1 else 0
    fn = cm[1, 0] if cm.shape[0] > 1 and cm.shape[1] > 0 else 0
    tp = cm[1, 1] if cm.shape[0] > 1 and cm.shape[1] > 1 else 0

    real_acc = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fake_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
        "confusion_matrix": cm,
        "real_accuracy": real_acc,
        "fake_recall": fake_recall,
    }


# ============================================================
# Evaluation
# ============================================================

def evaluate(model, dataloader, loss_fn, device, num_classes, desc="Test", is_inception=False):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    total_loss = 0.0

    pbar = tqdm(dataloader, desc=desc, dynamic_ncols=True, leave=True)
    with torch.no_grad():
        for inputs, labels_batch in pbar:
            inputs = inputs.to(device)
            labels_batch = labels_batch.to(device)
            outputs = model(inputs)
            logits = _extract_logits(outputs) if is_inception else outputs
            loss = loss_fn(logits, labels_batch)
            total_loss += loss.item()

            probs = F.softmax(logits, dim=1)
            _, preds = torch.max(logits, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            running_acc = accuracy_score(all_labels, all_preds)
            pbar.set_postfix({
                "loss": "{:.4f}".format(loss.item()),
                "acc": "{:.4f}".format(running_acc),
            })

    avg_loss = total_loss / len(dataloader)
    metrics = compute_metrics(all_labels, all_preds, all_probs, num_classes)
    metrics["loss"] = avg_loss
    return metrics


# ============================================================
# Dataset verification
# ============================================================

def check_dataset_counts(test_dataset):
    expected_real = 2100
    expected_fake = 2100
    expected_sources = {"ProGAN": 700, "StyleGAN2": 700, "StyleGAN3": 700}

    assert test_dataset.real_count == expected_real, \
        "Real count mismatch: {} != {}".format(test_dataset.real_count, expected_real)
    assert test_dataset.fake_count == expected_fake, \
        "Fake count mismatch: {} != {}".format(test_dataset.fake_count, expected_fake)
    assert len(test_dataset) == expected_real + expected_fake, \
        "Total count mismatch: {} != {}".format(len(test_dataset), expected_real + expected_fake)

    src_counts = test_dataset.get_gan_source_counts()
    for src, expected in expected_sources.items():
        actual = src_counts.get(src, 0)
        assert actual == expected, \
            "{} count mismatch: {} != {}".format(src, actual, expected)

    print("Dataset verified: real={}, fake={}, total={}".format(
        test_dataset.real_count, test_dataset.fake_count, len(test_dataset)))
    for src, expected in expected_sources.items():
        print("  {} = {} (ok)".format(src, src_counts.get(src, 0)))


# ============================================================
# Argparse
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="All-Degradation Robustness Evaluation"
    )
    parser.add_argument("--real_root", default=REAL_ROOT, type=str)
    parser.add_argument("--fake_root", default=FAKE_ROOT, type=str)
    parser.add_argument(
        "-m", "--model_type",
        default="mobilenetv2",
        type=str,
        choices=MODEL_CHOICES,
    )
    parser.add_argument(
        "-c", "--checkpoint", default="", type=str,
        help="Path to checkpoint. If empty, auto-detected from --model_type.",
    )
    parser.add_argument("-g", "--gpu", default="0", type=str)
    parser.add_argument("--num_classes", default=2, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--sfgf_hidden_channels", default=32, type=int)
    return parser.parse_args()


# ============================================================
# Degradation specs
# ============================================================

DEGRADATION_MODES = [
    # (display_name, degrade_type, degrade_param)
    ("clean",       None,       None),
    ("jpeg90",      "jpeg",     90),
    ("jpeg80",      "jpeg",     80),
    ("jpeg70",      "jpeg",     70),
    ("blur1.0",     "blur",     1.0),
    ("blur2.0",     "blur",     2.0),
    ("blur3.0",     "blur",     3.0),
    ("noise10",     "noise",    10),
    ("noise25",     "noise",    25),
    ("noise50",     "noise",    50),
    ("downsample0.75", "downsample", 0.75),
    ("downsample0.5",  "downsample", 0.5),
    ("downsample0.25", "downsample", 0.25),
]


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not args.checkpoint:
        args.checkpoint = CHECKPOINT_MAP.get(args.model_type, "")
        if not args.checkpoint:
            raise RuntimeError("No checkpoint found for model: {}. Use -c to specify.".format(args.model_type))
        print("Auto-detected checkpoint: {}".format(args.checkpoint))

    if not os.path.exists(args.checkpoint):
        raise RuntimeError("Checkpoint not found: {}".format(args.checkpoint))

    base_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    print("=" * 70)
    print("ALL-DEGRADATION ROBUSTNESS EVALUATION")
    print("=" * 70)
    print("Model:      {}".format(args.model_type))
    print("Checkpoint: {}".format(args.checkpoint))
    print("GPU:        {}".format(args.gpu))
    print("Device:     {}".format(device))
    print("Batch size: {}".format(args.batch_size))
    print("Workers:    {}".format(args.num_workers))
    print("Degradation modes: {} types".format(len(DEGRADATION_MODES)))
    for name, dtype, dparam in DEGRADATION_MODES:
        print("  {:20s}  type={:10s}  param={}".format(name, str(dtype), dparam))
    print("=" * 70)

    # ---- dataset ----
    print("\nLoading test dataset...")
    test_dataset = FFHQGANData(args.real_root, args.fake_root, "test", transform=None)
    check_dataset_counts(test_dataset)

    # ---- model ----
    print("\nBuilding model...")
    model = build_model(args.model_type, args.num_classes, args.sfgf_hidden_channels)
    model = model.to(device)
    print("  Model built: {} params".format(sum(p.numel() for p in model.parameters())))

    loaded = load_model_state(model, args.checkpoint, device)
    print("  Loaded {} params from checkpoint".format(loaded))

    is_inception = (args.model_type == "inceptionv3")
    loss_fn = nn.CrossEntropyLoss()

    results = []

    # ---- evaluate each degradation ----
    for mode_name, degrade_type, degrade_param in DEGRADATION_MODES:
        print("\n" + "-" * 60)
        print("Testing: {}".format(mode_name.upper()))
        print("-" * 60)

        # Use per-image deterministic noise for fair cross-model comparison
        if degrade_type == "noise":
            eval_dataset = DeterministicNoiseTestDataset(
                test_dataset, base_transform,
                degrade_type=degrade_type,
                degrade_param=degrade_param,
                base_seed=42,
            )
        else:
            test_dataset.transform = DegradeTransform(
                base_transform,
                degrade_type=degrade_type,
                degrade_param=degrade_param,
            )
            eval_dataset = test_dataset

        test_loader = torch.utils.data.DataLoader(
            eval_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=0, pin_memory=False,
        )

        metrics = evaluate(
            model, test_loader, loss_fn, device, args.num_classes,
            desc="{} Eval".format(mode_name),
            is_inception=is_inception,
        )

        results.append({
            "mode": mode_name,
            "degrade_type": str(degrade_type),
            "degrade_param": degrade_param,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "auc": metrics["auc"],
            "real_accuracy": metrics["real_accuracy"],
            "fake_recall": metrics["fake_recall"],
            "confusion_matrix": metrics["confusion_matrix"],
        })

        print("  Accuracy:       {:.4f}".format(metrics["accuracy"]))
        print("  Precision:      {:.4f}".format(metrics["precision"]))
        print("  Recall:         {:.4f}".format(metrics["recall"]))
        print("  F1-score:       {:.4f}".format(metrics["f1"]))
        print("  AUC:            {:.4f}".format(metrics["auc"]))
        print("  Real Accuracy:  {:.4f}".format(metrics["real_accuracy"]))
        print("  Fake Recall:    {:.4f}".format(metrics["fake_recall"]))
        print("  Confusion Matrix: {}".format(metrics["confusion_matrix"].tolist()))

        # ---- clean accuracy check ----
        if mode_name == "clean":
            if metrics["accuracy"] < 0.98:
                print("\n" + "=" * 70)
                print("WARNING: Clean accuracy ({:.4f}) is below 0.98.".format(metrics["accuracy"]))
                print("Model loading or structure may be incorrect. Stopping.")
                print("=" * 70)
                sys.exit(1)

    # ---- compute Drop metrics ----
    clean_acc = results[0]["accuracy"]
    clean_f1 = results[0]["f1"]
    for r in results:
        r["drop_acc"] = clean_acc - r["accuracy"]
        r["drop_f1"] = clean_f1 - r["f1"]

    # ---- print summary table ----
    print("\n" + "=" * 100)
    print("ALL-DEGRADATION ROBUSTNESS RESULTS")
    print("Model: {}".format(args.model_type))
    print("=" * 100)

    header = "{:<18} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<8} {:<10} {:<10}".format(
        "Degradation", "Accuracy", "Drop Acc", "Prec", "Recall",
        "F1", "Drop F1", "AUC", "Real Acc", "Fake Rec",
    )
    print(header)
    print("-" * 100)

    for r in results:
        line = "{:<18} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<8.4f} {:<10.4f} {:<10.4f}".format(
            r["mode"], r["accuracy"], r["drop_acc"], r["precision"], r["recall"],
            r["f1"], r["drop_f1"], r["auc"], r["real_accuracy"], r["fake_recall"],
        )
        print(line)

    print("=" * 100)

    for r in results:
        print("\n{} Confusion Matrix:".format(r["mode"]))
        print(r["confusion_matrix"])

    # ---- save results ----
    out_dir = os.path.dirname(args.checkpoint)
    os.makedirs(out_dir, exist_ok=True)

    txt_path = os.path.join(out_dir, "all_degradation_results.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Model: {}\n".format(args.model_type))
        f.write("Checkpoint: {}\n".format(args.checkpoint))
        f.write("\n")
        f.write(header + "\n")
        f.write("-" * 100 + "\n")
        for r in results:
            line = "{:<18} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<8.4f} {:<10.4f} {:<10.4f}".format(
                r["mode"], r["accuracy"], r["drop_acc"], r["precision"], r["recall"],
                r["f1"], r["drop_f1"], r["auc"], r["real_accuracy"], r["fake_recall"],
            )
            f.write(line + "\n")
        f.write("\n")
        for r in results:
            f.write("\n{} Confusion Matrix:\n".format(r["mode"]))
            f.write(str(r["confusion_matrix"]) + "\n")
    print("\nResults saved to: {}".format(txt_path))

    csv_path = os.path.join(out_dir, "all_degradation_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Degradation", "Accuracy", "Drop Acc", "Precision", "Recall",
            "F1-score", "Drop F1", "AUC", "Real Accuracy", "Fake Recall",
        ])
        for r in results:
            writer.writerow([
                r["mode"],
                "{:.4f}".format(r["accuracy"]),
                "{:.4f}".format(r["drop_acc"]),
                "{:.4f}".format(r["precision"]),
                "{:.4f}".format(r["recall"]),
                "{:.4f}".format(r["f1"]),
                "{:.4f}".format(r["drop_f1"]),
                "{:.4f}".format(r["auc"]),
                "{:.4f}".format(r["real_accuracy"]),
                "{:.4f}".format(r["fake_recall"]),
            ])
    print("Results saved to: {}".format(csv_path))


if __name__ == "__main__":
    main()