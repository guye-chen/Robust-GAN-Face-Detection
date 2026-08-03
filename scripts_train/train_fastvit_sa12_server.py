import os
import sys
import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader.ffhq_gan_dataset import FFHQGANData
from train_convnext_base_server import (
    GAN_NAMES,
    _AMP,
    _make_scaler,
    train_one_epoch,
    evaluate,
    run_noise_tests,
    print_dataset_info,
    make_debug_subset,
)

try:
    import timm
    _TIMM_AVAILABLE = True
except ImportError:
    timm = None
    _TIMM_AVAILABLE = False


REAL_ROOT = r"E:\FFHQ_off"
FAKE_ROOT = r"E:\archive\GAN_Generated_Fake_Images\fake"
CHECKPOINT_DIR = r"E:\cbam_ftt_fusion\CBAM.PyTorch-master\weights\fastvit_sa12"

PRIMARY_MODEL_NAME = "fastvit_sa12.apple_in1k"
FALLBACK_MODEL_NAME = "fastvit_s12.apple_in1k"
NUM_CLASSES = 2

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
CLICK_RUN_MODEL_NAME = PRIMARY_MODEL_NAME
CLICK_RUN_PRETRAINED = True
CLICK_RUN_BATCH_SIZE = DEFAULT_BATCH_SIZE
CLICK_RUN_NUM_WORKERS = DEFAULT_NUM_WORKERS
CLICK_RUN_EPOCHS_OVERRIDE = None
CLICK_RUN_DEBUG_SUBSET = None


def _short_cfg(cfg):
    keys = [
        "architecture", "tag", "url", "hf_hub_id", "input_size",
        "mean", "std", "crop_pct", "interpolation", "num_classes",
        "classifier", "first_conv",
    ]
    return {key: cfg.get(key, None) for key in keys}


def _classifier_out_features(classifier):
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


def resolve_fastvit_model_name(requested_name):
    if not _TIMM_AVAILABLE:
        raise RuntimeError("timm is not installed. FastViT requires timm; install it in new_gpu_env.")

    available_arch = set(timm.list_models("*fastvit*"))
    available_pretrained = set(timm.list_models("*fastvit*", pretrained=True))
    available_all = available_arch | available_pretrained
    if requested_name in available_all:
        return requested_name, available_all

    untagged_name = requested_name.split(".")[0]
    if untagged_name in available_arch:
        print("Requested tagged FastViT model not found in plain list: {}".format(requested_name))
        print("Using available untagged architecture instead: {}".format(untagged_name))
        return untagged_name, available_all

    print("Requested FastViT model not found in this timm version: {}".format(requested_name))
    print("Available timm FastViT pretrained models:")
    for name in sorted(available_pretrained):
        print("  {}".format(name))
    print("Available timm FastViT architectures:")
    for name in sorted(available_arch):
        print("  {}".format(name))
    if FALLBACK_MODEL_NAME in available_all:
        print("Using fallback model: {}".format(FALLBACK_MODEL_NAME))
        return FALLBACK_MODEL_NAME, available_all
    fallback_untagged = FALLBACK_MODEL_NAME.split(".")[0]
    if fallback_untagged in available_arch:
        print("Using fallback untagged architecture: {}".format(fallback_untagged))
        return fallback_untagged, available_all
    raise RuntimeError(
        "Neither {} nor {} exists in this timm version.".format(
            requested_name, FALLBACK_MODEL_NAME)
    )


def build_fastvit(model_name, pretrained, num_classes):
    model_name, _ = resolve_fastvit_model_name(model_name)

    print("\nBuilding FastViT...")
    print("  requested model name: {}".format(CLICK_RUN_MODEL_NAME))
    print("  actual model name: {}".format(model_name))
    print("  pretrained: {}".format(pretrained))
    print("  fairness note: using FastViT ImageNet-1K supervised weights; no YOLO, ConvNeXt, ConvNeXtV2, 22K, or FCMAE weights.")

    try:
        model = timm.create_model(model_name, pretrained=pretrained)
    except Exception as exc:
        print("Available timm FastViT models:")
        for name in timm.list_models("*fastvit*"):
            print("  {}".format(name))
        raise RuntimeError("Failed to create timm model '{}': {}".format(model_name, exc))

    cfg = getattr(model, "pretrained_cfg", None) or getattr(model, "default_cfg", {})
    default_cfg = getattr(model, "default_cfg", {})
    classifier_before = model.get_classifier()
    before_num_classes = cfg.get("num_classes", getattr(model, "num_classes", None))

    print("  model name: {}".format(model_name))
    print("  pretrained_cfg: {}".format(_short_cfg(cfg)))
    print("  default_cfg: {}".format(_short_cfg(default_cfg)))
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
    after_out_features = _classifier_out_features(classifier_after)
    head_ok = after_out_features == num_classes
    print("  classifier after: {}".format(classifier_after))
    print("  num_classes after: {}".format(getattr(model, "num_classes", num_classes)))
    print("  head out_features after: {}".format(after_out_features))
    print("  head replaced with 2-class output: {}".format(head_ok))
    if not head_ok:
        raise RuntimeError("FastViT classifier head was not replaced with {} classes.".format(num_classes))
    return model, cfg, model_name


def make_transforms(cfg):
    mean = cfg.get("mean", (0.485, 0.456, 0.406))
    std = cfg.get("std", (0.229, 0.224, 0.225))
    interpolation = transforms.InterpolationMode.BICUBIC
    if str(cfg.get("interpolation", "")).lower() in ("bilinear", "linear"):
        interpolation = transforms.InterpolationMode.BILINEAR
    train_transform = transforms.Compose([
        transforms.Resize(256, interpolation=interpolation),
        transforms.RandomResizedCrop(224, scale=(0.85, 1.0), interpolation=interpolation),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256, interpolation=interpolation),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_transform, val_transform


def parse_args():
    parser = argparse.ArgumentParser(description="Train FastViT-SA12 on FFHQ/GAN real-fake classification")
    parser.add_argument("--real_root", default=REAL_ROOT, type=str)
    parser.add_argument("--fake_root", default=FAKE_ROOT, type=str)
    parser.add_argument("--checkpoint_dir", default=CHECKPOINT_DIR, type=str)
    parser.add_argument("--model_name", default=CLICK_RUN_MODEL_NAME, type=str)
    parser.add_argument("--pretrained", action="store_true", default=CLICK_RUN_PRETRAINED)
    parser.add_argument("--no_pretrained", action="store_false", dest="pretrained")
    parser.add_argument("--batch_size", default=CLICK_RUN_BATCH_SIZE, type=int)
    parser.add_argument("--accumulation_steps", default=DEFAULT_ACCUMULATION_STEPS, type=int)
    parser.add_argument("--lr", default=DEFAULT_LR, type=float)
    parser.add_argument("--epochs", default=CLICK_RUN_EPOCHS_OVERRIDE if CLICK_RUN_EPOCHS_OVERRIDE is not None else DEFAULT_EPOCHS, type=int)
    parser.add_argument("--patience", default=DEFAULT_PATIENCE, type=int)
    parser.add_argument("--num_workers", default=CLICK_RUN_NUM_WORKERS, type=int)
    parser.add_argument("--gpu", default=DEFAULT_GPU, type=str)
    parser.add_argument("--num_classes", default=NUM_CLASSES, type=int)
    parser.add_argument("--debug_subset", default=CLICK_RUN_DEBUG_SUBSET, type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available()
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print("=" * 70)
    print("Model: FastViT-SA12")
    print("CLICK_RUN_MODEL_NAME = {}".format(CLICK_RUN_MODEL_NAME))
    print("CLICK_RUN_PRETRAINED = {}".format(CLICK_RUN_PRETRAINED))
    print("CHECKPOINT_DIR: {}".format(args.checkpoint_dir))
    print("optimizer: Adam")
    print("learning_rate: {}".format(args.lr))
    print("epochs: {}".format(args.epochs))
    print("batch_size: {}".format(args.batch_size))
    print("accumulation_steps: {}".format(args.accumulation_steps))
    print("effective batch size: {}".format(args.batch_size * args.accumulation_steps))
    print("AMP: {}  (api: {})".format(use_amp, _AMP))
    print("=" * 70)

    model, cfg, actual_model_name = build_fastvit(args.model_name, args.pretrained, args.num_classes)
    train_transform, val_transform = make_transforms(cfg)

    print("\nLoading datasets...")
    train_dataset = FFHQGANData(args.real_root, args.fake_root, "train", transform=train_transform)
    val_dataset = FFHQGANData(args.real_root, args.fake_root, "val", transform=val_transform)
    test_dataset = FFHQGANData(args.real_root, args.fake_root, "test", transform=None)

    print("\nDataset counts:")
    print_dataset_info("train", train_dataset, 15000, 15000,
                       {"ProGAN": 5000, "StyleGAN2": 5000, "StyleGAN3": 5000})
    print_dataset_info("val", val_dataset, 4200, 4200,
                       {"ProGAN": 1400, "StyleGAN2": 1400, "StyleGAN3": 1400})
    print_dataset_info("test", test_dataset, 2100, 2100,
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
    scaler = _make_scaler(use_amp)

    best_val_acc = 0.0
    best_epoch = 0
    no_improve_count = 0
    logs = []
    train_start = time.time()

    print("\n" + "=" * 70)
    print("TRAINING START")
    print("=" * 70)
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, loss_fn, optimizer, scaler, device,
            args.accumulation_steps, use_amp, epoch, args.epochs, args.lr)
        val_metrics = evaluate(model, val_loader, loss_fn, device, args.num_classes,
                               desc="Val Epoch {}/{}".format(epoch, args.epochs))
        elapsed = time.time() - start
        line = (
            "Epoch [{:2d}/{:2d}] | train_loss={:.4f} | train_acc={:.4f} | "
            "val_loss={:.4f} | val_acc={:.4f} | val_prec={:.4f} | "
            "val_rec={:.4f} | val_f1={:.4f} | val_auc={:.4f} | time={:.1f}s"
        ).format(
            epoch, args.epochs, train_loss, train_acc, val_metrics["loss"],
            val_metrics["accuracy"], val_metrics["precision"], val_metrics["recall"],
            val_metrics["f1"], val_metrics["auc"], elapsed)
        print(line)
        logs.append(line)

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = float(val_metrics["accuracy"])
            best_epoch = epoch
            no_improve_count = 0
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val_acc": best_val_acc,
                "model_name": actual_model_name,
                "requested_model_name": args.model_name,
                "pretrained": args.pretrained,
                "pretrained_cfg": dict(cfg),
                "total_params": total_params,
                "trainable_params": trainable_params,
                "args": vars(args),
            }, os.path.join(args.checkpoint_dir, "best_model.pth"))
            print("  *** Saved best model (val_acc={:.4f}) ***".format(best_val_acc))
        else:
            no_improve_count += 1
            print("  No improvement ({}/{})".format(no_improve_count, args.patience))
        if no_improve_count >= args.patience:
            print("Early stopping triggered.")
            break

    total_train_time = time.time() - train_start
    torch.save({
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "model_name": actual_model_name,
        "requested_model_name": args.model_name,
        "pretrained": args.pretrained,
        "pretrained_cfg": dict(cfg),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "args": vars(args),
    }, os.path.join(args.checkpoint_dir, "last_model.pth"))
    with open(os.path.join(args.checkpoint_dir, "training_log.txt"), "w", encoding="utf-8") as f:
        for line in logs:
            f.write(line + "\n")

    print("\n" + "=" * 70)
    print("TRAINING FINISHED")
    print("Best epoch: {} | best val_acc: {:.4f}".format(best_epoch, best_val_acc))
    print("Best model: {}".format(os.path.join(args.checkpoint_dir, "best_model.pth")))
    print("Last model: {}".format(os.path.join(args.checkpoint_dir, "last_model.pth")))
    print("Training total time: {:.1f}s ({:.2f} min)".format(total_train_time, total_train_time / 60.0))
    print("=" * 70)

    best_checkpoint = torch.load(os.path.join(args.checkpoint_dir, "best_model.pth"),
                                 map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["state_dict"], strict=True)
    model = model.to(device)
    run_noise_tests(model, test_dataset, val_transform, args, device, loss_fn)


if __name__ == "__main__":
    main()
