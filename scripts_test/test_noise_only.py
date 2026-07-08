"""
Gaussian Noise Robustness Evaluation (Noise-Only)

Tests model robustness under additive Gaussian noise:
  - Clean
  - noise10  (sigma=10)
  - noise25  (sigma=25)
  - noise50  (sigma=50)

Noise is added in PIL/numpy raw pixel space [0,255] before ToTensor + Normalize.
Visualization samples (5 real + 5 fake per level) are saved alongside metrics.

Supports all models from test_jpeg_robustness.py and
test_jpeg_robustness_remaining_server.py.

Usage:
    python test_noise_only.py -m resnet50-cbam-dense-msjf-ra --batch_size 1 --num_workers 0
    python test_noise_only.py -m resnet50-cbam-dense-msjf-ra -c <path> --batch_size 1 --num_workers 0
"""

import os
import sys
import argparse
import csv
import hashlib
import warnings
import numpy as np
from collections import OrderedDict
from io import BytesIO

warnings.filterwarnings("ignore", category=FutureWarning)

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
import model.inceptionv3_nmfgf_noise_ra as inceptionv3_nmfgf_noise_ra
import model.inceptionv3_nmfgf_noise_ra_v1_legacy as inceptionv3_nmfgf_noise_ra_v1_legacy
import model.inceptionv3_frca as inceptionv3_frca
import model.inceptionv3_frca_hns as inceptionv3_frca_hns
import model.inceptionv3_lmfc as inceptionv3_lmfc
import importlib
resnet50_nmfgf = importlib.import_module('model.resnet50_nmfgf')
resnet50_nmfgf_noise_ra = importlib.import_module('model.resnet50_nmfgf_noise_ra')

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

# ==============================
# Click-run default configuration
# ==============================
CLICK_RUN_MODEL = "ALL"
CLICK_RUN_NMFGF_VARIANT = "V1"
CLICK_RUN_BATCH_SIZE = 1
CLICK_RUN_NUM_WORKERS = 0
CLICK_RUN_FORCE_ALPHA = None
CLICK_RUN_DISABLE_INTERNAL_NOISE_COPY = False
CLICK_RUN_FORCE_INTERNAL_SIGMA = None
CLICK_RUN_SAVE_VISUALS = False
ALL_MODELS_OUT_DIR = r"E:\cbam_ftt_fusion\CBAM.PyTorch-master\weights\noise_only_all_models"

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
    "resnet50-nmfgf-noise-ra": r"E:\cbam_ftt_fusion\CBAM.PyTorch-master\weights\resnet50-nmfgf-noise-ra\best_model.pth",
    "inceptionv3-nmfgf-noise-ra": r"E:\cbam_ftt_fusion\CBAM.PyTorch-master\weights\inceptionv3-nmfgf-noise-ra\best_model.pth",
    "inceptionv3-nmfgf-noise-ra-stability-reg": r"E:\cbam_ftt_fusion\CBAM.PyTorch-master\weights\inceptionv3-nmfgf-noise-ra-stability-reg\best_model.pth",
    "inceptionv3-frca": r"E:\cbam_ftt_fusion\CBAM.PyTorch-master\weights\inceptionv3-frca\best_model.pth",
    "inceptionv3-frca-hns": r"E:\cbam_ftt_fusion\CBAM.PyTorch-master\weights\inceptionv3-frca-hns\best_model.pth",
    "inceptionv3-lmfc": r"E:\cbam_ftt_fusion\CBAM.PyTorch-master\weights\inceptionv3-lmfc\best_model.pth",
    "vgg16": r"E:\cbam_ftt_fusion\CBAM.PyTorch-master\weights\best_model_vgg16.pth",
    "vgg19": r"E:\cbam_ftt_fusion\CBAM.PyTorch-master\weights\best_model_vgg19.pth",
    "inceptionv3": r"E:\cbam_ftt_fusion\CBAM.PyTorch-master\weights\best_model_inceptionv3.pth",
    "inceptionresnetv2": r"E:\cbam_ftt_fusion\CBAM.PyTorch-master\weights\best_model_inceptionresnetv2.pth",
    "resnet50v2": r"E:\cbam_ftt_fusion\CBAM.PyTorch-master\weights\best_model_resnet50v2.pth",
    "xception": r"E:\cbam_ftt_fusion\CBAM.PyTorch-master\weights\best_model_xception.pth",
}

INCEPTION_NMFGF_VARIANT_CHECKPOINTS = {
    "V1": r"E:\cbam_ftt_fusion\CBAM.PyTorch-master\weights\inceptionv3-nmfgf-noise-ra\best_model.pth",
    "A": r"E:\cbam_ftt_fusion\CBAM.PyTorch-master\weights\inceptionv3-nmfgf-noise-ra-A\best_model.pth",
    "B": r"E:\cbam_ftt_fusion\CBAM.PyTorch-master\weights\inceptionv3-nmfgf-noise-ra-B\best_model.pth",
    "B_no_ra_loss": r"E:\cbam_ftt_fusion\CBAM.PyTorch-master\weights\inceptionv3-nmfgf-noise-ra-B-no-ra-loss\best_model.pth",
    "B_freeze_cls": r"E:\cbam_ftt_fusion\CBAM.PyTorch-master\weights\inceptionv3-nmfgf-noise-ra-B-freeze-cls\best_model.pth",
}

MODEL_CHOICES = sorted(CHECKPOINT_MAP.keys())
MODEL_TYPE_CHOICES = ["ALL"] + MODEL_CHOICES

# ---- reproducible global RNG for noise (each image gets different noise) ----
_rng = np.random.default_rng(42)


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
    deterministic per image path + degradation — all models see the
    exact same noise for the exact same image.

    If seed is None, falls back to global _rng (used for backward compat).
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

    Seed = MD5(image_path | degradation_name | degrad_param | base_seed),
    ensuring the same image always gets the same noise regardless of
    batch_size, num_workers, shuffle order, or repeat runs.

    This guarantees all models see the identical noise pattern for each image.
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


def _extract_aux(outputs):
    if isinstance(outputs, tuple):
        for item in reversed(outputs):
            if isinstance(item, dict):
                return item
    return {}


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

class StabilityRegInferenceWrapper(nn.Module):
    """Use the stability-reg checkpoint with cls_logits as the test decision."""

    def __init__(self, base_model):
        super(StabilityRegInferenceWrapper, self).__init__()
        self.base_model = base_model

    def forward(self, x, return_aux=False):
        final_logits, cls_logits, stability_logits, aux = self.base_model(x, return_aux=False)
        aux = dict(aux)
        aux["stability_logits_abs_mean"] = stability_logits.detach().abs().mean()
        aux["max_abs_final_minus_cls"] = (final_logits - cls_logits).detach().abs().max()
        if return_aux:
            return cls_logits, aux
        return cls_logits

    def load_state_dict(self, state_dict, strict=True):
        return self.base_model.load_state_dict(state_dict, strict=strict)

    def state_dict(self, *args, **kwargs):
        return self.base_model.state_dict(*args, **kwargs)


def build_model(model_type, num_classes, sfgf_hidden_channels=32, nmfgf_variant="B"):
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

    elif model_type == "resnet50-nmfgf-noise-ra":
        model = resnet50_nmfgf_noise_ra.resnet50_nmfgf_noise_ra(
            num_classes=num_classes,
            dense_dim=16,
            mlp_hidden=64,
            alpha_init=0.05,
        )

    elif model_type == "inceptionv3-nmfgf-noise-ra":
        if nmfgf_variant == "V1":
            fixed_alpha = None
            use_severity_gate = False
            alpha_init = 0.05
        elif nmfgf_variant == "A":
            fixed_alpha = 0.01
            use_severity_gate = False
            alpha_init = 0.01
        elif nmfgf_variant in ("B", "B_no_ra_loss", "B_freeze_cls"):
            fixed_alpha = None
            use_severity_gate = True
            alpha_init = 0.02
        else:
            raise ValueError("Unknown nmfgf_variant: {}".format(nmfgf_variant))
        model = inceptionv3_nmfgf_noise_ra.inceptionv3_nmfgf_noise_ra(
            num_classes=num_classes,
            pretrained_backbone=False,
            alpha_init=alpha_init,
            dense_dim=16,
            mlp_hidden=64,
            alpha_max=0.05,
            fixed_alpha=fixed_alpha,
            use_severity_gate=use_severity_gate,
        )

    elif model_type == "inceptionv3-nmfgf-noise-ra-stability-reg":
        base_model = inceptionv3_nmfgf_noise_ra_v1_legacy.inceptionv3_nmfgf_noise_ra_v1(
            num_classes=num_classes,
            pretrained_backbone=False,
            alpha_init=0.02,
            ra_loss_weight=0.005,
            dense_dim=16,
            mlp_hidden=64,
            sigma_train=[5.0, 10.0, 15.0, 25.0],
            sigma_eval=10.0,
            transform_input=True,
        )
        model = StabilityRegInferenceWrapper(base_model)

    elif model_type == "inceptionv3-frca":
        model = inceptionv3_frca.inceptionv3_frca(
            num_classes=num_classes,
            pretrained_backbone=False,
            adapter_bottleneck=64,
            gate_hidden_dim=16,
            gamma_init=0.01,
            gamma_max=0.05,
        )

    elif model_type == "inceptionv3-frca-hns":
        model = inceptionv3_frca_hns.inceptionv3_frca_hns(
            num_classes=num_classes,
            pretrained_backbone=False,
            adapter_bottleneck=64,
            gate_hidden_dim=16,
            gamma_init=0.01,
            gamma_max=0.05,
            suppress_tau=0.45,
            suppress_k=10.0,
        )

    elif model_type == "inceptionv3-lmfc":
        model = inceptionv3_lmfc.inceptionv3_lmfc(
            num_classes=num_classes,
            pretrained_backbone=False,
            adapter_bottleneck=64,
            gate_hidden_dim=16,
            beta_init=0.01,
            beta_max=0.05,
            low_mid_radius=0.50,
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

def evaluate(model, dataloader, loss_fn, device, num_classes,
             desc="Test", is_inception=False, model_type=None):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    logit_labels = []
    logit_preds = {"final": [], "cls": [], "stability": []}
    logit_probs = {"final": [], "cls": [], "stability": []}
    original_branch_labels, original_branch_preds, original_branch_probs = [], [], []
    total_loss = 0.0

    # ---- diagnostics collectors for noise-ra model ----
    is_noise_ra = model_type in (
        "resnet50-nmfgf-noise-ra",
        "inceptionv3-nmfgf-noise-ra",
    )
    is_frca = model_type in ("inceptionv3-frca", "inceptionv3-frca-hns", "inceptionv3-lmfc")
    diag_collect = OrderedDict()

    def add_diag(name, value):
        diag_collect.setdefault(name, []).append(float(value))

    pbar = tqdm(dataloader, desc=desc, dynamic_ncols=True, leave=True)
    with torch.no_grad():
        for inputs, labels_batch in pbar:
            inputs = inputs.to(device)
            labels_batch = labels_batch.to(device)
            if is_frca:
                outputs = model(inputs, return_aux=True)
            else:
                outputs = model(inputs)
            logits = _extract_logits(outputs)
            loss = loss_fn(logits, labels_batch)
            total_loss += loss.item()

            probs = F.softmax(logits, dim=1)
            _, preds = torch.max(logits, 1)

            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels_batch.detach().cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())
            logit_labels.extend(labels_batch.detach().cpu().numpy())

            # ---- collect NoiseRA diagnostics from tuple ----
            if is_noise_ra:
                aux = _extract_aux(outputs)
                if "alpha" in aux:
                    add_diag("alpha", aux["alpha"].detach().mean().item())
                if "beta" in aux:
                    add_diag("beta", aux["beta"].detach().mean().item())
                if "sigma" in aux:
                    sigma = aux["sigma"]
                    add_diag("sigma", sigma.detach().mean().item() if torch.is_tensor(sigma) else sigma)

                logits_by_name = {"final": logits}
                if isinstance(outputs, tuple) and len(outputs) >= 3:
                    logits_by_name["cls"] = outputs[1]
                    logits_by_name["stability"] = outputs[2]

                for logit_name, logit_value in logits_by_name.items():
                    prob_value = F.softmax(logit_value, dim=1)
                    _, pred_value = torch.max(logit_value, 1)
                    logit_preds[logit_name].extend(pred_value.detach().cpu().numpy())
                    logit_probs[logit_name].extend(prob_value.detach().cpu().numpy())
                    for label_value, label_name in [(0, "real"), (1, "fake")]:
                        mask = labels_batch == label_value
                        if mask.any():
                            add_diag(
                                "{}_fake_prob_mean_for_{}".format(logit_name, label_name),
                                prob_value[mask, 1].mean().item(),
                            )

                for key, value in aux.items():
                    if key in ("alpha", "beta", "sigma", "freq_stats", "dense_diff_score"):
                        continue
                    if key.startswith("cos") or key.startswith("D_") or key.startswith("D"):
                        if torch.is_tensor(value):
                            add_diag(key, value.detach().mean().item())
                        else:
                            add_diag(key, value)

                freq = aux.get('freq_stats')
                if freq is not None and freq.dim() >= 2:
                    add_diag('E_low_ratio', freq[:, 0].mean().item())
                    add_diag('E_mid_ratio', freq[:, 1].mean().item())
                    add_diag('E_high_ratio', freq[:, 2].mean().item())
                    add_diag('E_high_over_E_mid', freq[:, 3].mean().item())
                    add_diag('E_mid_over_E_low', freq[:, 4].mean().item())

                    for label_value, label_name in [(0, "real"), (1, "fake")]:
                        mask = labels_batch == label_value
                        if mask.any():
                            add_diag("{}_E_low_ratio".format(label_name), freq[mask, 0].mean().item())
                            add_diag("{}_E_mid_ratio".format(label_name), freq[mask, 1].mean().item())
                            add_diag("{}_E_high_ratio".format(label_name), freq[mask, 2].mean().item())
                            add_diag("{}_E_high_over_E_mid".format(label_name), freq[mask, 3].mean().item())
                            add_diag("{}_E_mid_over_E_low".format(label_name), freq[mask, 4].mean().item())

                score = aux.get("dense_diff_score")
                if score is not None and torch.is_tensor(score):
                    for label_value, label_name in [(0, "real"), (1, "fake")]:
                        mask = labels_batch == label_value
                        if mask.any():
                            add_diag("{}_cosine_stability".format(label_name),
                                     score[mask].mean().item())

            if is_frca:
                aux = _extract_aux(outputs)
                if "beta" in aux:
                    add_diag("beta", aux["beta"].detach().mean().item())
                if "gamma" in aux:
                    add_diag("gamma", aux["gamma"].detach().mean().item())
                for key in (
                    "adapter_output_abs_mean",
                    "lmfc_feature_abs_mean",
                    "lmfc_adapter_output_abs_mean",
                    "feat_delta_abs_max",
                    "feat_delta_abs_mean",
                    "low_mid_view_mean",
                    "low_mid_view_std",
                ):
                    value = aux.get(key)
                    if value is not None and torch.is_tensor(value):
                        add_diag(key, value.detach().mean().item())
                low_mid_diff = aux.get("low_mid_input_abs_diff")
                if low_mid_diff is not None and torch.is_tensor(low_mid_diff):
                    add_diag("low_mid_input_abs_diff", low_mid_diff.detach().mean().item())
                gate = aux.get("gate")
                if gate is not None and torch.is_tensor(gate):
                    add_diag("gate_mean", gate.mean().item())
                    for label_value, label_name in [(0, "real"), (1, "fake")]:
                        mask = labels_batch == label_value
                        if mask.any():
                            add_diag("gate_{}_mean".format(label_name), gate[mask].mean().item())
                fusion_gate = aux.get("fusion_gate")
                if fusion_gate is not None and torch.is_tensor(fusion_gate):
                    add_diag("fusion_gate_mean", fusion_gate.mean().item())
                original_logits = aux.get("original_logits")
                if original_logits is not None and torch.is_tensor(original_logits):
                    original_probs = F.softmax(original_logits, dim=1)
                    _, original_preds = torch.max(original_logits, 1)
                    original_branch_labels.extend(labels_batch.detach().cpu().numpy())
                    original_branch_preds.extend(original_preds.detach().cpu().numpy())
                    original_branch_probs.extend(original_probs.detach().cpu().numpy())
                suppress = aux.get("suppress")
                if suppress is not None and torch.is_tensor(suppress):
                    add_diag("suppress_mean", suppress.mean().item())
                    for label_value, label_name in [(0, "real"), (1, "fake")]:
                        mask = labels_batch == label_value
                        if mask.any():
                            add_diag("suppress_{}_mean".format(label_name), suppress[mask].mean().item())
                effective_gate = aux.get("effective_gate")
                if effective_gate is not None and torch.is_tensor(effective_gate):
                    add_diag("effective_gate_mean", effective_gate.mean().item())
                freq = aux.get("freq_stats")
                if freq is not None and freq.dim() >= 2:
                    add_diag('E_low_ratio', freq[:, 0].mean().item())
                    add_diag('E_mid_ratio', freq[:, 1].mean().item())
                    add_diag('E_high_ratio', freq[:, 2].mean().item())
                    add_diag('E_high_over_E_mid', freq[:, 3].mean().item())
                    add_diag('E_mid_over_E_low', freq[:, 4].mean().item())

            running_acc = accuracy_score(all_labels, all_preds)
            pbar.set_postfix({
                "loss": "{:.4f}".format(loss.item()),
                "acc": "{:.4f}".format(running_acc),
            })

    metrics = compute_metrics(all_labels, all_preds, all_probs, num_classes)

    # ---- diagnostics summary (only for noise-ra) ----
    diagnostics = {}
    if (is_noise_ra or is_frca) and diag_collect:
        diagnostics = {k: float(np.mean(v)) if v else 0.0
                       for k, v in diag_collect.items()}
        if 'alpha' in diag_collect:
            diagnostics['alpha_final'] = float(diag_collect['alpha'][-1])
        if 'sigma' in diag_collect:
            diagnostics['sigma_eval'] = float(diag_collect['sigma'][0])
        if is_frca:
            diagnostics["final_logits_accuracy"] = float(metrics["accuracy"])
            if original_branch_preds and original_branch_probs:
                original_metrics = compute_metrics(
                    original_branch_labels,
                    original_branch_preds,
                    original_branch_probs,
                    num_classes,
                )
                diagnostics["original_branch_logits_accuracy"] = float(original_metrics["accuracy"])
        if is_noise_ra:
            for logit_name in ["final", "cls", "stability"]:
                if logit_preds[logit_name] and logit_probs[logit_name]:
                    branch_metrics = compute_metrics(
                        logit_labels,
                        logit_preds[logit_name],
                        logit_probs[logit_name],
                        num_classes,
                    )
                    diagnostics["{}_accuracy".format(logit_name)] = float(branch_metrics["accuracy"])
                    diagnostics["{}_real_accuracy".format(logit_name)] = float(branch_metrics["real_accuracy"])
                    diagnostics["{}_fake_recall".format(logit_name)] = float(branch_metrics["fake_recall"])

    return metrics, diagnostics


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
# Sample visualization
# ============================================================

def save_visualization_samples(test_dataset, base_transform, out_dir, mode_name,
                                degrade_type, degrade_param, num_real=5, num_fake=5):
    """
    Save 5 real + 5 fake sample images (degraded version) to disk
    for visual confirmation of noise intensity.
    """
    from torchvision.utils import save_image

    vis_dir = os.path.join(out_dir, "noise_samples", mode_name)
    os.makedirs(vis_dir, exist_ok=True)

    # Collect real and fake image paths
    real_paths = []
    fake_paths = []
    for path, label, _ in test_dataset.samples:
        if label == 0 and len(real_paths) < num_real:
            real_paths.append(path)
        elif label == 1 and len(fake_paths) < num_fake:
            fake_paths.append(path)
        if len(real_paths) >= num_real and len(fake_paths) >= num_fake:
            break

    degrade_tf = DegradeTransform(base_transform, degrade_type=degrade_type,
                                  degrade_param=degrade_param)

    for i, path in enumerate(real_paths):
        img = pil_loader(path)
        img_tensor = degrade_tf(img)
        save_image(img_tensor, os.path.join(vis_dir, "real_{:02d}.png".format(i + 1)))

    for i, path in enumerate(fake_paths):
        img = pil_loader(path)
        img_tensor = degrade_tf(img)
        save_image(img_tensor, os.path.join(vis_dir, "fake_{:02d}.png".format(i + 1)))

    print("  Saved {} real + {} fake samples to: {}".format(
        num_real, num_fake, vis_dir))


# ============================================================
# Argparse
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Gaussian Noise Robustness Evaluation (Noise-Only)"
    )
    parser.add_argument("--real_root", default=REAL_ROOT, type=str)
    parser.add_argument("--fake_root", default=FAKE_ROOT, type=str)
    parser.add_argument(
        "-m", "--model_type",
        default=CLICK_RUN_MODEL,
        type=str,
        choices=MODEL_TYPE_CHOICES,
    )
    parser.add_argument(
        "-c", "--checkpoint", default="", type=str,
        help="Path to checkpoint. If empty, auto-detected from --model_type.",
    )
    parser.add_argument("-g", "--gpu", default="0", type=str)
    parser.add_argument("--num_classes", default=2, type=int)
    parser.add_argument("--batch_size", default=CLICK_RUN_BATCH_SIZE, type=int)
    parser.add_argument("--num_workers", default=CLICK_RUN_NUM_WORKERS, type=int)
    parser.add_argument("--sfgf_hidden_channels", default=32, type=int)
    parser.add_argument("--force_alpha", default=CLICK_RUN_FORCE_ALPHA, type=float,
                        help="Temporarily override Noise-RA alpha during test, e.g. --force_alpha 0.")
    parser.add_argument("--disable_internal_noise_copy", action="store_true",
                        default=CLICK_RUN_DISABLE_INTERNAL_NOISE_COPY,
                        help="Ablation only: make Noise-RA internal noisy copy equal to input during test.")
    parser.add_argument("--force_internal_sigma", default=CLICK_RUN_FORCE_INTERNAL_SIGMA, type=float,
                        help="Ablation only: override Noise-RA sigma_eval during test, e.g. --force_internal_sigma 0.")
    parser.add_argument("--save_visuals", action="store_true", default=CLICK_RUN_SAVE_VISUALS,
                        help="Save 5 real + 5 fake degraded sample images per noise level.")
    parser.add_argument(
        "--nmfgf_variant",
        default=CLICK_RUN_NMFGF_VARIANT,
        choices=["V1", "A", "B", "B_no_ra_loss", "B_freeze_cls"],
        type=str,
                        help="InceptionV3-NMFGF test structure: V1=original checkpoint, A=fixed alpha 0.01, B=bounded alpha + severity gate.")
    return parser.parse_args()


# ============================================================
# Degradation specs (noise-only)
# ============================================================

DEGRADATION_MODES = [
    # (display_name, degrade_type, degrade_param)
    ("clean",    None,     None),
    ("noise10",  "noise",  10),
    ("noise25",  "noise",  25),
    ("noise50",  "noise",  50),
]


# ============================================================
# Shared evaluation runners
# ============================================================

def resolve_checkpoint(model_type, nmfgf_variant):
    if model_type == "inceptionv3-nmfgf-noise-ra":
        return INCEPTION_NMFGF_VARIANT_CHECKPOINTS.get(nmfgf_variant, "")
    return CHECKPOINT_MAP.get(model_type, "")


def make_eval_dataset(test_dataset, base_transform, degrade_type, degrade_param):
    if degrade_type == "noise":
        return DeterministicNoiseTestDataset(
            test_dataset, base_transform,
            degrade_type=degrade_type,
            degrade_param=degrade_param,
            base_seed=42,
        )
    test_dataset.transform = DegradeTransform(
        base_transform,
        degrade_type=degrade_type,
        degrade_param=degrade_param,
    )
    return test_dataset


def evaluate_one_model(args, model_type, checkpoint, device, base_transform,
                       test_dataset, out_dir=None, save_outputs=True,
                       save_visuals=False):
    if not checkpoint:
        raise RuntimeError("No checkpoint configured for model: {}".format(model_type))
    if not os.path.exists(checkpoint):
        raise RuntimeError("Checkpoint not found: {}".format(checkpoint))

    out_dir = out_dir or os.path.dirname(checkpoint)
    os.makedirs(out_dir, exist_ok=True)

    print("\n" + "=" * 100)
    print("Evaluating model: {}".format(model_type))
    print("Checkpoint: {}".format(checkpoint))
    if model_type == "inceptionv3-nmfgf-noise-ra":
        print("NMFGF variant: {}".format(args.nmfgf_variant))
    print("=" * 100)

    model = build_model(
        model_type,
        args.num_classes,
        args.sfgf_hidden_channels,
        nmfgf_variant=args.nmfgf_variant,
    )
    model = model.to(device)
    print("  Model built: {} params".format(sum(p.numel() for p in model.parameters())))
    loaded = load_model_state(model, checkpoint, device)
    print("  Loaded {} params from checkpoint".format(loaded))

    if args.force_alpha is not None:
        if hasattr(model, "set_force_alpha"):
            model.set_force_alpha(args.force_alpha)
            print("  Force alpha enabled for test: {}".format(args.force_alpha))
        else:
            print("  WARNING: --force_alpha ignored; model has no set_force_alpha().")
    if args.force_internal_sigma is not None:
        target_model = getattr(model, "base_model", model)
        if hasattr(target_model, "sigma_eval"):
            target_model.sigma_eval = float(args.force_internal_sigma)
            print("  Force internal sigma_eval enabled for test: {}".format(target_model.sigma_eval))
        else:
            print("  WARNING: --force_internal_sigma ignored; model has no sigma_eval.")
    if args.disable_internal_noise_copy:
        target_model = getattr(model, "base_model", model)
        if hasattr(target_model, "_noise_perturb"):
            def _identity_noise_perturb(x, sigma):
                return x
            target_model._noise_perturb = _identity_noise_perturb
            if hasattr(target_model, "sigma_eval"):
                target_model.sigma_eval = 0.0
            print("  Internal noise copy disabled: _noise_perturb(x, sigma) returns x")
        else:
            print("  WARNING: --disable_internal_noise_copy ignored; model has no _noise_perturb().")

    is_inception = (model_type == "inceptionv3")
    loss_fn = nn.CrossEntropyLoss()
    results = []

    for mode_name, degrade_type, degrade_param in DEGRADATION_MODES:
        print("\n" + "-" * 60)
        print("Testing {} / {}".format(model_type, mode_name.upper()))
        print("-" * 60)

        eval_dataset = make_eval_dataset(test_dataset, base_transform, degrade_type, degrade_param)
        test_loader = torch.utils.data.DataLoader(
            eval_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=False,
        )

        metrics, diagnostics = evaluate(
            model, test_loader, loss_fn, device, args.num_classes,
            desc="{} {}".format(model_type, mode_name),
            is_inception=is_inception,
            model_type=model_type,
        )

        result = {
            "model": model_type,
            "checkpoint": checkpoint,
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
            "diagnostics": diagnostics,
            "status": "ok",
        }
        results.append(result)

        if diagnostics:
            print("  Diagnostics:")
            for key in sorted(diagnostics.keys()):
                print("    {}: {:.6f}".format(key, diagnostics[key]))

        print("  Accuracy:       {:.4f}".format(metrics["accuracy"]))
        print("  Precision:      {:.4f}".format(metrics["precision"]))
        print("  Recall:         {:.4f}".format(metrics["recall"]))
        print("  F1-score:       {:.4f}".format(metrics["f1"]))
        print("  AUC:            {:.4f}".format(metrics["auc"]))
        print("  Real Accuracy:  {:.4f}".format(metrics["real_accuracy"]))
        print("  Fake Recall:    {:.4f}".format(metrics["fake_recall"]))
        print("  Confusion Matrix: {}".format(metrics["confusion_matrix"].tolist()))

        if save_visuals:
            save_visualization_samples(
                test_dataset, base_transform, out_dir, mode_name,
                degrade_type=degrade_type, degrade_param=degrade_param,
                num_real=5, num_fake=5,
            )

    clean_acc = results[0]["accuracy"]
    clean_f1 = results[0]["f1"]
    for r in results:
        r["drop_acc"] = clean_acc - r["accuracy"]
        r["drop_f1"] = clean_f1 - r["f1"]

    if save_outputs:
        save_single_model_results(model_type, checkpoint, results, out_dir, args)

    return results


def save_single_model_results(model_type, checkpoint, results, out_dir, args):
    header = "{:<18} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<8} {:<10} {:<10}".format(
        "Degradation", "Accuracy", "Drop Acc", "Prec", "Recall",
        "F1", "Drop F1", "AUC", "Real Acc", "Fake Rec",
    )
    result_suffix = ""
    if args.disable_internal_noise_copy:
        result_suffix += "_disable_internal_noise_copy"
    if args.force_internal_sigma is not None:
        result_suffix += "_sigma{}".format(str(args.force_internal_sigma).replace(".", "p"))

    txt_path = os.path.join(out_dir, "noise_only_results{}.txt".format(result_suffix))
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Model: {}\n".format(model_type))
        f.write("Checkpoint: {}\n\n".format(checkpoint))
        f.write(header + "\n")
        f.write("-" * 100 + "\n")
        for r in results:
            f.write("{:<18} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<8.4f} {:<10.4f} {:<10.4f}\n".format(
                r["mode"], r["accuracy"], r["drop_acc"], r["precision"], r["recall"],
                r["f1"], r["drop_f1"], r["auc"], r["real_accuracy"], r["fake_recall"],
            ))
        for r in results:
            f.write("\n{} Confusion Matrix:\n".format(r["mode"]))
            f.write(str(r["confusion_matrix"]) + "\n")
            if r.get("diagnostics"):
                f.write("{} Diagnostics:\n".format(r["mode"]))
                for key in sorted(r["diagnostics"].keys()):
                    f.write("  {}: {:.6f}\n".format(key, r["diagnostics"][key]))
    print("\nResults saved to: {}".format(txt_path))

    csv_path = os.path.join(out_dir, "noise_only_results{}.csv".format(result_suffix))
    diagnostic_keys = sorted({key for r in results for key in r.get("diagnostics", {}).keys()})
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Degradation", "Accuracy", "Drop Acc", "Precision", "Recall",
            "F1-score", "Drop F1", "AUC", "Real Accuracy", "Fake Recall",
            "TN", "FP", "FN", "TP",
        ] + diagnostic_keys)
        for r in results:
            cm = r["confusion_matrix"]
            tn, fp = int(cm[0, 0]), int(cm[0, 1])
            fn, tp = int(cm[1, 0]), int(cm[1, 1])
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
                tn, fp, fn, tp,
            ] + [
                "{:.6f}".format(r.get("diagnostics", {}).get(key, 0.0))
                for key in diagnostic_keys
            ])
    print("Results saved to: {}".format(csv_path))


def print_results_table(results, title):
    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)
    header = "{:<40} {:<10} {:<8} {:<8} {:<8} {:<8} {:<8} {:<9} {:<9}".format(
        "Model", "Degrade", "Acc", "Prec", "Recall", "F1", "AUC", "RealAcc", "FakeRec"
    )
    print(header)
    print("-" * 120)
    for r in results:
        print("{:<40} {:<10} {:<8.4f} {:<8.4f} {:<8.4f} {:<8.4f} {:<8.4f} {:<9.4f} {:<9.4f}".format(
            r["model"][:40], r["mode"], r["accuracy"], r["precision"], r["recall"],
            r["f1"], r["auc"], r["real_accuracy"], r["fake_recall"],
        ))
    print("=" * 120)


def save_all_models_summary(results, skipped, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "noise_only_all_models_summary.csv")
    txt_path = os.path.join(out_dir, "noise_only_all_models_summary.txt")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "checkpoint", "degradation", "acc", "drop_acc",
            "precision", "recall", "f1", "drop_f1", "auc",
            "real_acc", "fake_rec", "TN", "FP", "FN", "TP", "status",
        ])
        for r in results:
            cm = r["confusion_matrix"]
            writer.writerow([
                r["model"], r["checkpoint"], r["mode"],
                "{:.6f}".format(r["accuracy"]),
                "{:.6f}".format(r["drop_acc"]),
                "{:.6f}".format(r["precision"]),
                "{:.6f}".format(r["recall"]),
                "{:.6f}".format(r["f1"]),
                "{:.6f}".format(r["drop_f1"]),
                "{:.6f}".format(r["auc"]),
                "{:.6f}".format(r["real_accuracy"]),
                "{:.6f}".format(r["fake_recall"]),
                int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]),
                r.get("status", "ok"),
            ])
        for item in skipped:
            writer.writerow([
                item["model"], item["checkpoint"], "", "", "", "", "", "", "",
                "", "", "", "", "", "", "", item["status"],
            ])

    sorted_results = sorted(results, key=lambda x: (x["mode"], -x["accuracy"], -x["f1"], x["model"]))
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Gaussian noise all-model summary\n")
        f.write("Rows sorted by degradation, then accuracy/f1 descending.\n\n")
        for r in sorted_results:
            f.write("{:<40} {:<10} acc={:.4f} f1={:.4f} auc={:.4f} real_acc={:.4f} fake_rec={:.4f} checkpoint={}\n".format(
                r["model"], r["mode"], r["accuracy"], r["f1"], r["auc"],
                r["real_accuracy"], r["fake_recall"], r["checkpoint"],
            ))
        if skipped:
            f.write("\nSkipped models:\n")
            for item in skipped:
                f.write("{} | {} | {}\n".format(item["model"], item["status"], item["checkpoint"]))

    print("\nAll-model CSV saved to: {}".format(csv_path))
    print("All-model TXT saved to: {}".format(txt_path))


def run_all_models(args, device, base_transform):
    print("\nLoading test dataset once for all-model evaluation...")
    test_dataset = FFHQGANData(args.real_root, args.fake_root, "test", transform=None)
    check_dataset_counts(test_dataset)

    all_results = []
    skipped = []
    for model_type in MODEL_CHOICES:
        checkpoint = resolve_checkpoint(model_type, args.nmfgf_variant)
        if not checkpoint or not os.path.exists(checkpoint):
            print("\nSkipping {}: checkpoint missing ({})".format(model_type, checkpoint))
            skipped.append({
                "model": model_type,
                "checkpoint": checkpoint,
                "status": "missing_checkpoint",
            })
            continue
        try:
            model_out_dir = os.path.join(ALL_MODELS_OUT_DIR, model_type)
            rows = evaluate_one_model(
                args, model_type, checkpoint, device, base_transform,
                test_dataset, out_dir=model_out_dir, save_outputs=True,
                save_visuals=args.save_visuals,
            )
            all_results.extend(rows)
        except Exception as exc:
            print("\nERROR while evaluating {}: {}".format(model_type, exc))
            skipped.append({
                "model": model_type,
                "checkpoint": checkpoint,
                "status": "error: {}".format(exc),
            })
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    sorted_results = sorted(all_results, key=lambda x: (x["mode"], -x["accuracy"], -x["f1"], x["model"]))
    print_results_table(sorted_results, "ALL MODELS GAUSSIAN NOISE SUMMARY")
    if skipped:
        print("\nSkipped / failed models:")
        for item in skipped:
            print("  {} | {} | {}".format(item["model"], item["status"], item["checkpoint"]))
    save_all_models_summary(all_results, skipped, ALL_MODELS_OUT_DIR)


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    if args.model_type == "ALL":
        print("=" * 70)
        print("GAUSSIAN NOISE ROBUSTNESS EVALUATION (ALL MODELS)")
        print("=" * 70)
        print("Click-run default configuration:")
        print("CLICK_RUN_MODEL: {}".format(CLICK_RUN_MODEL))
        print("CLICK_RUN_NMFGF_VARIANT: {}".format(CLICK_RUN_NMFGF_VARIANT))
        print("CLICK_RUN_BATCH_SIZE: {}".format(CLICK_RUN_BATCH_SIZE))
        print("CLICK_RUN_NUM_WORKERS: {}".format(CLICK_RUN_NUM_WORKERS))
        print("CLICK_RUN_SAVE_VISUALS: {}".format(CLICK_RUN_SAVE_VISUALS))
        print("Output dir: {}".format(ALL_MODELS_OUT_DIR))
        print("GPU:        {}".format(args.gpu))
        print("Device:     {}".format(device))
        print("Batch size: {}".format(args.batch_size))
        print("Workers:    {}".format(args.num_workers))
        print("Degradation modes: {} types".format(len(DEGRADATION_MODES)))
        for name, dtype, dparam in DEGRADATION_MODES:
            print("  {:20s}  type={:10s}  param={}".format(name, str(dtype), dparam))
        print("=" * 70)
        run_all_models(args, device, base_transform)
        return

    manual_checkpoint = bool(args.checkpoint)
    if not args.checkpoint:
        if args.model_type == "inceptionv3-nmfgf-noise-ra":
            args.checkpoint = INCEPTION_NMFGF_VARIANT_CHECKPOINTS.get(args.nmfgf_variant, "")
        else:
            args.checkpoint = CHECKPOINT_MAP.get(args.model_type, "")
        if not args.checkpoint:
            raise RuntimeError("No checkpoint found for model: {}. Use -c to specify.".format(args.model_type))
        print("Auto-detected checkpoint: {}".format(args.checkpoint))
    elif args.model_type == "inceptionv3-nmfgf-noise-ra":
        print("Manual checkpoint provided; ignoring variant auto-checkpoint map.")

    if not os.path.exists(args.checkpoint):
        raise RuntimeError("Checkpoint not found: {}".format(args.checkpoint))

    out_dir = os.path.dirname(args.checkpoint)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print("GAUSSIAN NOISE ROBUSTNESS EVALUATION (NOISE-ONLY)")
    print("=" * 70)
    print("Click-run default configuration:")
    print("CLICK_RUN_MODEL: {}".format(CLICK_RUN_MODEL))
    print("CLICK_RUN_NMFGF_VARIANT: {}".format(CLICK_RUN_NMFGF_VARIANT))
    print("CLICK_RUN_BATCH_SIZE: {}".format(CLICK_RUN_BATCH_SIZE))
    print("CLICK_RUN_NUM_WORKERS: {}".format(CLICK_RUN_NUM_WORKERS))
    print("CLICK_RUN_FORCE_ALPHA: {}".format(CLICK_RUN_FORCE_ALPHA))
    print("CLICK_RUN_DISABLE_INTERNAL_NOISE_COPY: {}".format(CLICK_RUN_DISABLE_INTERNAL_NOISE_COPY))
    print("CLICK_RUN_FORCE_INTERNAL_SIGMA: {}".format(CLICK_RUN_FORCE_INTERNAL_SIGMA))
    print("CLICK_RUN_SAVE_VISUALS: {}".format(CLICK_RUN_SAVE_VISUALS))
    print("Model:      {}".format(args.model_type))
    print("Checkpoint: {}".format(args.checkpoint))
    print("Checkpoint source: {}".format("manual" if manual_checkpoint else "auto"))
    print("GPU:        {}".format(args.gpu))
    print("Device:     {}".format(device))
    print("Batch size: {}".format(args.batch_size))
    print("Workers:    {}".format(args.num_workers))
    print("NMFGF variant: {}".format(args.nmfgf_variant))
    print("Force alpha: {}".format(args.force_alpha))
    print("Disable internal noise copy: {}".format(args.disable_internal_noise_copy))
    print("Force internal sigma: {}".format(args.force_internal_sigma))
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
    model = build_model(
        args.model_type,
        args.num_classes,
        args.sfgf_hidden_channels,
        nmfgf_variant=args.nmfgf_variant,
    )
    model = model.to(device)
    print("  Model built: {} params".format(sum(p.numel() for p in model.parameters())))

    loaded = load_model_state(model, args.checkpoint, device)
    print("  Loaded {} params from checkpoint".format(loaded))
    if args.force_alpha is not None:
        if hasattr(model, "set_force_alpha"):
            model.set_force_alpha(args.force_alpha)
            print("  Force alpha enabled for test: {}".format(args.force_alpha))
        else:
            print("  WARNING: --force_alpha ignored; model has no set_force_alpha().")
    if args.force_internal_sigma is not None:
        if hasattr(model, "sigma_eval"):
            model.sigma_eval = float(args.force_internal_sigma)
            print("  Force internal sigma_eval enabled for test: {}".format(model.sigma_eval))
        else:
            print("  WARNING: --force_internal_sigma ignored; model has no sigma_eval.")
    if args.disable_internal_noise_copy:
        if hasattr(model, "_noise_perturb"):
            def _identity_noise_perturb(x, sigma):
                return x
            model._noise_perturb = _identity_noise_perturb
            if hasattr(model, "sigma_eval"):
                model.sigma_eval = 0.0
            print("  Internal noise copy disabled: _noise_perturb(x, sigma) returns x")
        else:
            print("  WARNING: --disable_internal_noise_copy ignored; model has no _noise_perturb().")

    is_inception = (args.model_type == "inceptionv3")
    loss_fn = nn.CrossEntropyLoss()

    results = []

    # ---- evaluate each degradation ----
    for mode_name, degrade_type, degrade_param in DEGRADATION_MODES:
        print("\n" + "-" * 60)
        print("Testing: {}".format(mode_name.upper()))
        print("-" * 60)

        # Use per-image deterministic noise for fair comparison across models
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

        metrics, diagnostics = evaluate(
            model, test_loader, loss_fn, device, args.num_classes,
            desc="{} Eval".format(mode_name),
            is_inception=is_inception,
            model_type=args.model_type,
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
            "diagnostics": diagnostics,
        })

        # ---- Print NoiseRA diagnostics ----
        if diagnostics:
            print("  Diagnostics:")
            for key in sorted(diagnostics.keys()):
                print("    {}: {:.6f}".format(key, diagnostics[key]))

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
                allow_low_clean = (
                    args.model_type == "inceptionv3-nmfgf-noise-ra"
                    and args.nmfgf_variant == "V1"
                )
                print("\n" + "=" * 70)
                print("WARNING: Clean accuracy ({:.4f}) is below 0.98.".format(metrics["accuracy"]))
                if allow_low_clean:
                    print("V1 is allowed to continue so noise10/noise25/noise50 can be evaluated.")
                else:
                    print("Model loading or structure may be incorrect. Stopping.")
                print("=" * 70)
                if not allow_low_clean:
                    sys.exit(1)

        # ---- save visualization samples ----
        if args.save_visuals:
            save_visualization_samples(
                test_dataset, base_transform, out_dir, mode_name,
                degrade_type=degrade_type, degrade_param=degrade_param,
                num_real=5, num_fake=5,
            )

    # ---- compute Drop metrics ----
    clean_acc = results[0]["accuracy"]
    clean_f1 = results[0]["f1"]
    for r in results:
        r["drop_acc"] = clean_acc - r["accuracy"]
        r["drop_f1"] = clean_f1 - r["f1"]

    # ---- print summary table ----
    print("\n" + "=" * 100)
    print("GAUSSIAN NOISE ROBUSTNESS RESULTS")
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
    result_suffix = ""
    if args.disable_internal_noise_copy:
        result_suffix += "_disable_internal_noise_copy"
    if args.force_internal_sigma is not None:
        result_suffix += "_sigma{}".format(str(args.force_internal_sigma).replace(".", "p"))
    txt_path = os.path.join(out_dir, "noise_only_results{}.txt".format(result_suffix))
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
            if r.get("diagnostics"):
                f.write("{} Diagnostics:\n".format(r["mode"]))
                for key in sorted(r["diagnostics"].keys()):
                    f.write("  {}: {:.6f}\n".format(key, r["diagnostics"][key]))
    print("\nResults saved to: {}".format(txt_path))

    csv_path = os.path.join(out_dir, "noise_only_results{}.csv".format(result_suffix))
    diagnostic_keys = sorted({
        key
        for result in results
        for key in result.get("diagnostics", {}).keys()
    })
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Degradation", "Accuracy", "Drop Acc", "Precision", "Recall",
            "F1-score", "Drop F1", "AUC", "Real Accuracy", "Fake Recall",
        ] + diagnostic_keys)
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
            ] + [
                "{:.6f}".format(r.get("diagnostics", {}).get(key, 0.0))
                for key in diagnostic_keys
            ])
    print("Results saved to: {}".format(csv_path))


if __name__ == "__main__":
    main()
