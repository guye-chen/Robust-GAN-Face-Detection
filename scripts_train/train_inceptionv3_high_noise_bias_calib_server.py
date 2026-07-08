import os
import sys
import csv
import json
import time
import shutil
import hashlib
import argparse
import warnings
import numpy as np
from PIL import Image

warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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
from model.inceptionv3_high_noise_bias_calib import (
    inceptionv3_high_noise_bias_calib,
    high_noise_bias_calibration_loss,
)


REAL_ROOT = r"E:\FFHQ_off"
FAKE_ROOT = r"E:\archive\GAN_Generated_Fake_Images\fake"
WEIGHTS_ROOT = r"E:\cbam_ftt_fusion\CBAM.PyTorch-master\weights"
CHECKPOINT_DIR = os.path.join(WEIGHTS_ROOT, "inceptionv3-high-noise-bias-calib")

DEFAULT_BATCH_SIZE = 8
DEFAULT_ACCUMULATION_STEPS = 4
DEFAULT_LR = 0.0001
DEFAULT_EPOCHS = 10
DEFAULT_PATIENCE = 3
DEFAULT_NUM_WORKERS = 4
DEFAULT_GPU = "0"

# ==============================
# Click-run default configuration
# ==============================
CLICK_RUN_DEBUG_SUBSET = None
CLICK_RUN_EPOCHS_OVERRIDE = None
CLICK_RUN_PRETRAINED_BACKBONE = True
CLICK_RUN_TRANSFORM_INPUT = True
CLICK_RUN_STABILITY_LOSS_WEIGHT = 0.005
CLICK_RUN_BIAS_CALIB_LOSS_WEIGHT = 0.05
CLICK_RUN_SIGMA_TRAIN_LIST = [25.0, 35.0, 50.0]
CLICK_RUN_SIGMA_EVAL = 50.0
CLICK_RUN_GAMMA_INIT = 0.02
CLICK_RUN_GAMMA_MAX = 0.20
CLICK_RUN_GATE_TAU = 0.55
CLICK_RUN_GATE_K = 10.0
CLICK_RUN_USE_HIGH_NOISE_GATE = True


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


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def get_gpu_mem():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 ** 3)
    return 0.0


def compute_metrics(labels, preds, probs):
    labels = np.asarray(labels).astype(int)
    preds = np.asarray(preds).astype(int)
    probs = np.asarray(probs).astype(float)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    auc = 0.0
    if len(np.unique(labels)) > 1:
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


def stability_regularization_loss(aux_dict, labels, margin=0.1):
    score = aux_dict["dense_diff_score"]
    mask_real = labels == 0
    mask_fake = labels == 1
    if mask_real.sum() == 0:
        return torch.tensor(0.0, device=score.device)
    real_pull = (1.0 - score[mask_real]).mean()
    if mask_fake.sum() == 0:
        return real_pull
    score_real = score[mask_real].mean()
    score_fake = score[mask_fake].mean()
    margin_term = F.relu(score_fake - score_real + margin)
    return real_pull + margin_term


def print_dataset_info(name, dataset, expected_real, expected_fake):
    print("{}: real={} expected={} | fake={} expected={} | total={}".format(
        name, dataset.real_count, expected_real, dataset.fake_count, expected_fake, len(dataset)))
    assert dataset.real_count == expected_real, "{} real count mismatch".format(name)
    assert dataset.fake_count == expected_fake, "{} fake count mismatch".format(name)


def make_debug_subset(dataset, subset_size):
    if subset_size is None or subset_size <= 0:
        return dataset
    return Subset(dataset, list(range(min(int(subset_size), len(dataset)))))


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


def build_model(args, device):
    model = inceptionv3_high_noise_bias_calib(
        num_classes=2,
        pretrained_backbone=args.pretrained_backbone,
        alpha_init=0.02,
        ra_loss_weight=args.stability_loss_weight,
        dense_dim=16,
        mlp_hidden=64,
        sigma_train=args.sigma_train,
        sigma_eval=args.sigma_eval,
        transform_input=args.transform_input,
        gamma_init=args.gamma_init,
        gamma_max=args.gamma_max,
        gate_tau=args.gate_tau,
        gate_k=args.gate_k,
        use_high_noise_gate=args.use_high_noise_gate,
    )
    return model.to(device)


def save_config_and_snapshot(args):
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    config = vars(args).copy()
    config.update({
        "model_name": "inceptionv3-high-noise-bias-calib",
        "training_objective": (
            "CE(final_logits on clean input) + lambda_stab * feature_stability_loss "
            "+ lambda_bias * high_noise_fake_bias_calibration_loss"
        ),
        "inference_logits": "final_logits",
        "ra_logits_used_for_decision": False,
        "noisy_copy_ce_loss": False,
        "internal_noisy_copy": True,
        "use_high_noise_gate": args.use_high_noise_gate,
        "best_checkpoint_selection": "val_noise50_acc",
    })
    with open(os.path.join(args.checkpoint_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    snapshot_dir = os.path.join(args.checkpoint_dir, "code_snapshot")
    os.makedirs(snapshot_dir, exist_ok=True)
    for path in [
        os.path.abspath(__file__),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "inceptionv3_high_noise_bias_calib.py"),
    ]:
        if os.path.exists(path):
            shutil.copy2(path, os.path.join(snapshot_dir, os.path.basename(path)))


def train_one_epoch(model, loader, loss_fn, optimizer, scaler, device, args, epoch):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    labels_all, preds_all = [], []
    total_loss, total_ce, total_stab, total_bias = 0.0, 0.0, 0.0, 0.0

    pbar = tqdm(loader, desc="Train Epoch {}/{}".format(epoch, args.epochs),
                dynamic_ncols=True, leave=True)
    for step, (inputs, labels) in enumerate(pbar):
        inputs = inputs.to(device)
        labels = labels.to(device)
        with amp_context(args.use_amp):
            final_logits, cls_logits, _, aux = model(inputs, return_aux=False)
            ce_loss = loss_fn(final_logits, labels)
            stab_loss = stability_regularization_loss(aux, labels)
            bias_loss = high_noise_bias_calibration_loss(aux, labels)
            loss = (
                ce_loss
                + args.stability_loss_weight * stab_loss
                + args.bias_calib_loss_weight * bias_loss
            ) / args.accumulation_steps

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
        total_ce += ce_loss.item()
        total_stab += stab_loss.item()
        total_bias += bias_loss.item()
        preds = torch.argmax(final_logits, dim=1)
        labels_all.extend(labels.detach().cpu().numpy().tolist())
        preds_all.extend(preds.detach().cpu().numpy().tolist())
        pbar.set_postfix({
            "loss": "{:.4f}".format(total_loss / (step + 1)),
            "ce": "{:.4f}".format(total_ce / (step + 1)),
            "bias": "{:.4f}".format(total_bias / (step + 1)),
            "acc": "{:.4f}".format(accuracy_score(labels_all, preds_all)),
            "gamma": "{:.4f}".format(float(aux["gamma"].detach().cpu())),
            "gpu": "{:.1f}G".format(get_gpu_mem()),
        })

    return {
        "loss": total_loss / len(loader),
        "ce": total_ce / len(loader),
        "stability": total_stab / len(loader),
        "bias": total_bias / len(loader),
        "accuracy": float(accuracy_score(labels_all, preds_all)),
    }


@torch.no_grad()
def evaluate_final(model, loader, loss_fn, device, desc):
    model.eval()
    labels_all, preds_all, probs_all = [], [], []
    total_loss = 0.0
    diag = {
        "gamma": [],
        "bias_correction_mean": [],
        "high_noise_gate_mean": [],
        "noise_bias_correction_mean": [],
        "noise_high_noise_gate_mean": [],
    }
    pbar = tqdm(loader, desc=desc, dynamic_ncols=True, leave=True)
    for inputs, labels in pbar:
        inputs = inputs.to(device)
        labels = labels.to(device)
        final_logits, cls_logits, _, aux = model(inputs, return_aux=False)
        loss = loss_fn(final_logits, labels)
        probs = F.softmax(final_logits, dim=1)
        preds = torch.argmax(final_logits, dim=1)
        total_loss += loss.item()
        labels_all.extend(labels.detach().cpu().numpy().tolist())
        preds_all.extend(preds.detach().cpu().numpy().tolist())
        probs_all.extend(probs.detach().cpu().numpy().tolist())
        for key in diag.keys():
            value = aux[key]
            if torch.is_tensor(value):
                value = value.detach().float().mean().cpu().item()
            diag[key].append(float(value))
        pbar.set_postfix({
            "loss": "{:.4f}".format(loss.item()),
            "acc": "{:.4f}".format(accuracy_score(labels_all, preds_all)),
        })
    metrics = compute_metrics(labels_all, preds_all, probs_all)
    metrics["loss"] = total_loss / len(loader)
    for key, values in diag.items():
        metrics[key] = float(np.mean(values)) if values else 0.0
    return metrics


def run_noise_tests(model, raw_test_dataset, base_transform, loss_fn, device, args):
    rows = []
    modes = [("clean", None), ("noise10", 10), ("noise25", 25), ("noise50", 50)]
    for mode_name, sigma in modes:
        eval_dataset = DeterministicNoiseDataset(raw_test_dataset, base_transform, noise_sigma=sigma)
        loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=False)
        start = time.time()
        metrics = evaluate_final(model, loader, loss_fn, device, "Test {}".format(mode_name))
        elapsed = time.time() - start
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
            "gamma": metrics["gamma"],
            "bias_correction_mean": metrics["bias_correction_mean"],
            "high_noise_gate_mean": metrics["high_noise_gate_mean"],
            "time_sec": elapsed,
            "time_per_image": elapsed / max(1, len(eval_dataset)),
        }
        rows.append(row)
        print("{}: acc={:.4f} precision={:.4f} recall={:.4f} f1={:.4f} auc={:.4f} real_acc={:.4f} fake_acc={:.4f} gamma={:.4f} gate={:.4f} bias={:.6f} cm={}".format(
            mode_name, row["acc"], row["precision"], row["recall"], row["f1"],
            row["auc"], row["real_acc"], row["fake_acc"], row["gamma"],
            row["high_noise_gate_mean"], row["bias_correction_mean"],
            [[row["TN"], row["FP"]], [row["FN"], row["TP"]]],
        ))

    csv_path = os.path.join(args.checkpoint_dir, "noise_only_results_high_noise_bias_calib.csv")
    txt_path = os.path.join(args.checkpoint_dir, "noise_only_results_high_noise_bias_calib.txt")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Inference uses final_logits with high-noise fake-bias calibration.\n")
        f.write("RA logits are not used for final decision. No noisy-copy CE loss is used.\n")
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("Saved results: {}".format(csv_path))
    print("Saved results: {}".format(txt_path))


def parse_args():
    parser = argparse.ArgumentParser(description="Train InceptionV3 high-noise fake-bias calibration expert")
    parser.add_argument("--real_root", default=REAL_ROOT, type=str)
    parser.add_argument("--fake_root", default=FAKE_ROOT, type=str)
    parser.add_argument("--checkpoint_dir", default=CHECKPOINT_DIR, type=str)
    parser.add_argument("--batch_size", default=DEFAULT_BATCH_SIZE, type=int)
    parser.add_argument("--accumulation_steps", default=DEFAULT_ACCUMULATION_STEPS, type=int)
    parser.add_argument("--lr", default=DEFAULT_LR, type=float)
    parser.add_argument("--epochs", default=CLICK_RUN_EPOCHS_OVERRIDE if CLICK_RUN_EPOCHS_OVERRIDE is not None else DEFAULT_EPOCHS, type=int)
    parser.add_argument("--patience", default=DEFAULT_PATIENCE, type=int)
    parser.add_argument("--num_workers", default=DEFAULT_NUM_WORKERS, type=int)
    parser.add_argument("--gpu", default=DEFAULT_GPU, type=str)
    parser.add_argument("--pretrained_backbone", action="store_true", default=CLICK_RUN_PRETRAINED_BACKBONE)
    parser.add_argument("--no_pretrained_backbone", action="store_false", dest="pretrained_backbone")
    parser.add_argument("--transform_input", action="store_true", default=CLICK_RUN_TRANSFORM_INPUT)
    parser.add_argument("--no_transform_input", action="store_false", dest="transform_input")
    parser.add_argument("--stability_loss_weight", default=CLICK_RUN_STABILITY_LOSS_WEIGHT, type=float)
    parser.add_argument("--bias_calib_loss_weight", default=CLICK_RUN_BIAS_CALIB_LOSS_WEIGHT, type=float)
    parser.add_argument("--sigma_train", default=CLICK_RUN_SIGMA_TRAIN_LIST, nargs="+", type=float)
    parser.add_argument("--sigma_eval", default=CLICK_RUN_SIGMA_EVAL, type=float)
    parser.add_argument("--gamma_init", default=CLICK_RUN_GAMMA_INIT, type=float)
    parser.add_argument("--gamma_max", default=CLICK_RUN_GAMMA_MAX, type=float)
    parser.add_argument("--gate_tau", default=CLICK_RUN_GATE_TAU, type=float)
    parser.add_argument("--gate_k", default=CLICK_RUN_GATE_K, type=float)
    parser.add_argument("--use_high_noise_gate", action="store_true", default=CLICK_RUN_USE_HIGH_NOISE_GATE)
    parser.add_argument("--no_high_noise_gate", action="store_false", dest="use_high_noise_gate")
    parser.add_argument("--debug_subset", default=CLICK_RUN_DEBUG_SUBSET, type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.use_amp = torch.cuda.is_available()
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print("=" * 70)
    print("InceptionV3 High-Noise Frequency Bias Calibration Expert")
    print("Purpose: suppress noise-induced fake-logit bias under severe Gaussian noise.")
    print("No noisy-copy CE loss is used; calibration uses bias regularization.")
    print("Best checkpoint selection: val_noise50_acc")
    print("Checkpoint dir: {}".format(args.checkpoint_dir))
    print("pretrained_backbone: {}".format(args.pretrained_backbone))
    print("transform_input: {}".format(args.transform_input))
    print("stability_loss_weight: {}".format(args.stability_loss_weight))
    print("bias_calib_loss_weight: {}".format(args.bias_calib_loss_weight))
    print("sigma_train: {}".format(args.sigma_train))
    print("sigma_eval: {}".format(args.sigma_eval))
    print("gamma_init: {}".format(args.gamma_init))
    print("gamma_max: {}".format(args.gamma_max))
    print("gate_tau: {}".format(args.gate_tau))
    print("gate_k: {}".format(args.gate_k))
    print("use_high_noise_gate: {}".format(args.use_high_noise_gate))
    print("batch_size: {}".format(args.batch_size))
    print("accumulation_steps: {}".format(args.accumulation_steps))
    print("epochs: {}".format(args.epochs))
    print("lr: {}".format(args.lr))
    print("AMP: {} ({})".format(args.use_amp, _AMP))
    print("=" * 70)

    save_config_and_snapshot(args)
    model = build_model(args, device)
    print("Total params: {:,}".format(sum(p.numel() for p in model.parameters())))
    print("Trainable params: {:,}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    print("\nLoading datasets...")
    train_dataset = FFHQGANData(args.real_root, args.fake_root, "train", transform=train_transform)
    val_dataset = FFHQGANData(args.real_root, args.fake_root, "val", transform=val_transform)
    raw_val_dataset = FFHQGANData(args.real_root, args.fake_root, "val", transform=None)
    raw_test_dataset = FFHQGANData(args.real_root, args.fake_root, "test", transform=None)
    print_dataset_info("train", train_dataset, 15000, 15000)
    print_dataset_info("val", val_dataset, 4200, 4200)
    print_dataset_info("test", raw_test_dataset, 2100, 2100)

    if args.debug_subset is not None and args.debug_subset > 0:
        print("DEBUG subset enabled: {}".format(args.debug_subset))
        train_dataset = make_debug_subset(train_dataset, args.debug_subset)
        val_dataset = make_debug_subset(val_dataset, args.debug_subset)
        raw_val_dataset = make_debug_subset(raw_val_dataset, args.debug_subset)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_clean_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.num_workers, pin_memory=True)
    val_noise50_dataset = DeterministicNoiseDataset(raw_val_dataset, val_transform, noise_sigma=50)
    val_noise50_loader = DataLoader(val_noise50_dataset, batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers, pin_memory=True)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scaler = make_scaler(args.use_amp)

    best_score = -1.0
    best_val_clean_acc = 0.0
    best_val_noise50_acc = 0.0
    best_val_noise50_f1 = 0.0
    best_epoch = 0
    no_improve = 0
    log_lines = []
    start_train = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_metrics = train_one_epoch(model, train_loader, loss_fn, optimizer, scaler, device, args, epoch)
        val_clean = evaluate_final(model, val_clean_loader, loss_fn, device, "Val clean {}/{}".format(epoch, args.epochs))
        val_noise50 = evaluate_final(model, val_noise50_loader, loss_fn, device, "Val noise50 {}/{}".format(epoch, args.epochs))
        elapsed = time.time() - epoch_start
        score = val_noise50["accuracy"]
        line = (
            "Epoch [{}/{}] train_loss={:.4f} train_ce={:.4f} train_stab={:.4f} train_bias={:.4f} "
            "train_acc={:.4f} val_clean_acc={:.4f} val_noise50_acc={:.4f} "
            "val_noise50_real={:.4f} val_noise50_fake={:.4f} val_noise50_f1={:.4f} "
            "gamma={:.4f} gate50={:.4f} bias50={:.6f} score={:.4f} time={:.1f}s"
        ).format(
            epoch, args.epochs,
            train_metrics["loss"], train_metrics["ce"], train_metrics["stability"], train_metrics["bias"],
            train_metrics["accuracy"], val_clean["accuracy"], val_noise50["accuracy"],
            val_noise50["real_acc"], val_noise50["fake_acc"], val_noise50["f1"],
            val_noise50["gamma"], val_noise50["high_noise_gate_mean"],
            val_noise50["bias_correction_mean"], score, elapsed,
        )
        print(line)
        log_lines.append(line)

        if score > best_score:
            best_score = score
            best_val_clean_acc = val_clean["accuracy"]
            best_val_noise50_acc = val_noise50["accuracy"]
            best_val_noise50_f1 = val_noise50["f1"]
            best_epoch = epoch
            no_improve = 0
            torch.save({
                "epoch": int(epoch),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_score": float(best_score),
                "best_val_clean_acc": float(best_val_clean_acc),
                "best_val_noise50_acc": float(best_val_noise50_acc),
                "best_val_noise50_f1": float(best_val_noise50_f1),
                "val_clean_metrics": val_clean,
                "val_noise50_metrics": val_noise50,
                "args": vars(args),
                "inference_logits": "final_logits",
                "training_objective": "CE(final clean) + stability regularization + high-noise fake-bias calibration",
            }, os.path.join(args.checkpoint_dir, "best_model.pth"))
            print("  *** Saved best model (val_noise50_acc={:.4f}, val_clean_acc={:.4f}) ***".format(
                best_val_noise50_acc, best_val_clean_acc))
        else:
            no_improve += 1
            print("  No improvement ({}/{})".format(no_improve, args.patience))
        if no_improve >= args.patience:
            print("Early stopping triggered.")
            break

    torch.save({
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_score": float(best_score),
        "best_val_clean_acc": float(best_val_clean_acc),
        "best_val_noise50_acc": float(best_val_noise50_acc),
        "best_val_noise50_f1": float(best_val_noise50_f1),
        "best_epoch": int(best_epoch),
        "args": vars(args),
    }, os.path.join(args.checkpoint_dir, "last_model.pth"))

    with open(os.path.join(args.checkpoint_dir, "training_log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")

    summary = {
        "best_epoch": int(best_epoch),
        "best_score_val_noise50_acc": float(best_score),
        "best_val_clean_acc": float(best_val_clean_acc),
        "best_val_noise50_acc": float(best_val_noise50_acc),
        "best_val_noise50_f1": float(best_val_noise50_f1),
        "total_train_time_sec": time.time() - start_train,
        "best_model_sha256": sha256_file(os.path.join(args.checkpoint_dir, "best_model.pth")),
        "last_model_sha256": sha256_file(os.path.join(args.checkpoint_dir, "last_model.pth")),
    }
    with open(os.path.join(args.checkpoint_dir, "training_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nTRAINING FINISHED")
    print("Best epoch: {} | best val_noise50_acc: {:.4f} | best val_clean_acc: {:.4f}".format(
        best_epoch, best_val_noise50_acc, best_val_clean_acc))
    print("Best model: {}".format(os.path.join(args.checkpoint_dir, "best_model.pth")))
    print("Last model: {}".format(os.path.join(args.checkpoint_dir, "last_model.pth")))

    best_path = os.path.join(args.checkpoint_dir, "best_model.pth")
    if os.path.exists(best_path):
        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        run_noise_tests(model, raw_test_dataset, val_transform, loss_fn, device, args)


if __name__ == "__main__":
    main()
