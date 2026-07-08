"""
Light degradation test for modern ImageNet-1K strong baselines and Final Gate v2.

This script does not train anything. It evaluates:
  - ConvNeXt-Tiny
  - Swin-Tiny
  - FastViT-SA12
  - EfficientNet-B3
  - Final Frequency-Reliability Weak-Strong Gate v2

The degradation settings match the existing light-degradation robustness suite:
  clean, jpeg90, jpeg70, blur1, blur2, downsample0.75, downsample0.5

Outputs are written to:
  weights/modern_strong_baselines_light_degradations/
"""

import os
import sys
import csv
import time
import argparse
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

try:
    import timm
except ImportError:
    timm = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader.ffhq_gan_dataset import FFHQGANData
from test_final_gate_and_experts_light_degradations import (
    DEGRADATION_MODES,
    LightDegradationDataset,
    build_models as build_final_gate_and_experts,
    compute_metrics,
    extract_logits,
    freeze_eval,
)


REAL_ROOT = r"E:\FFHQ_off"
FAKE_ROOT = r"E:\archive\GAN_Generated_Fake_Images\fake"
OUT_DIR = (
    r"E:\cbam_ftt_fusion\CBAM.PyTorch-master\weights"
    r"\modern_strong_baselines_light_degradations"
)

# ==============================
# Click-run default configuration
# ==============================
CLICK_RUN_BATCH_SIZE = 1
CLICK_RUN_NUM_WORKERS = 0
CLICK_RUN_GPU = "0"


MODERN_BASELINE_SPECS = [
    {
        "display_name": "ConvNeXt-Tiny",
        "model_name": "convnext_tiny.fb_in1k",
        "checkpoint": (
            r"E:\cbam_ftt_fusion\CBAM.PyTorch-master\weights"
            r"\convnext_tiny\best_model.pth"
        ),
    },
    {
        "display_name": "Swin-Tiny",
        "model_name": "swin_tiny_patch4_window7_224.ms_in1k",
        "checkpoint": (
            r"E:\cbam_ftt_fusion\CBAM.PyTorch-master\weights"
            r"\swin_tiny\best_model.pth"
        ),
    },
    {
        "display_name": "FastViT-SA12",
        "model_name": "fastvit_sa12.apple_in1k",
        "checkpoint": (
            r"E:\cbam_ftt_fusion\CBAM.PyTorch-master\weights"
            r"\fastvit_sa12\best_model.pth"
        ),
    },
    {
        "display_name": "EfficientNet-B3",
        # The trained checkpoint metadata shows this was the actual model name.
        "model_name": "efficientnet_b3.ra2_in1k",
        "checkpoint": (
            r"E:\cbam_ftt_fusion\CBAM.PyTorch-master\weights"
            r"\efficientnet_b3\best_model.pth"
        ),
    },
]


def checkpoint_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"], "state_dict"
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"], "model_state_dict"
    if isinstance(checkpoint, dict):
        return checkpoint, "raw state_dict"
    raise RuntimeError("Unsupported checkpoint format: {}".format(type(checkpoint)))


def clean_state_dict(state_dict):
    clean = OrderedDict()
    for key, value in state_dict.items():
        clean[key[7:] if key.startswith("module.") else key] = value
    return clean


def build_timm_eval_model(spec, device):
    if timm is None:
        raise RuntimeError("timm is not installed in this Python environment.")
    checkpoint_path = spec["checkpoint"]
    if not os.path.exists(checkpoint_path):
        raise RuntimeError("Checkpoint not found: {}".format(checkpoint_path))

    print("\nBuilding {} ({})".format(spec["display_name"], spec["model_name"]))
    model = timm.create_model(spec["model_name"], pretrained=False)
    model.reset_classifier(num_classes=2)
    model = model.to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict, fmt = checkpoint_state_dict(checkpoint)
    state_dict = clean_state_dict(state_dict)
    model.load_state_dict(state_dict, strict=True)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    report = {
        "display_name": spec["display_name"],
        "model_name": spec["model_name"],
        "checkpoint": checkpoint_path,
        "checkpoint_exists": True,
        "checkpoint_format": fmt,
        "strict": True,
        "loaded_state_dict_keys": len(state_dict),
        "total_params": total_params,
        "trainable_params_before_freeze": trainable_params,
    }
    print("  checkpoint: {}".format(checkpoint_path))
    print("  checkpoint format: {}".format(fmt))
    print("  strict=True: OK")
    print("  loaded state_dict keys: {}".format(len(state_dict)))
    print("  total params: {:,}".format(total_params))
    return freeze_eval(model), report


def build_final_gate_v2(device):
    # Reuse the already tested final-gate builder, but keep only the final model.
    model_specs = build_final_gate_and_experts(device)
    for spec in model_specs:
        if spec["name"] == "final_frequency_reliability_weak_strong_gate_v2":
            return spec["model"], {
                "display_name": "Final Gate v2",
                "model_name": "final_frequency_reliability_weak_strong_gate_v2",
                "checkpoint": spec["checkpoint"],
                "checkpoint_exists": os.path.exists(spec["checkpoint"]),
                "strict": True,
                "report": spec["report"],
            }
    raise RuntimeError("Final Gate v2 was not found in build_final_gate_and_experts().")


@torch.no_grad()
def evaluate_model(model, loader, device, desc):
    model.eval()
    if hasattr(model, "reset_diagnostics"):
        model.reset_diagnostics()

    labels_all, preds_all, probs_all = [], [], []
    start_time = time.time()
    pbar = tqdm(loader, desc=desc, dynamic_ncols=True, leave=True)
    for inputs, labels in pbar:
        inputs = inputs.to(device)
        labels = labels.to(device)
        logits = extract_logits(model(inputs))
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)

        labels_all.extend(labels.detach().cpu().numpy())
        preds_all.extend(preds.detach().cpu().numpy())
        probs_all.extend(probs.detach().cpu().numpy())
        running_correct = sum(int(a == b) for a, b in zip(labels_all, preds_all))
        running_acc = running_correct / max(1, len(labels_all))
        pbar.set_postfix({"acc": "{:.4f}".format(running_acc)})

    total_time = time.time() - start_time
    metrics = compute_metrics(labels_all, preds_all, probs_all)
    metrics["total_time"] = total_time
    metrics["time_per_image"] = total_time / max(1, len(labels_all))
    selection_counts = dict(model.last_selection_counts) if hasattr(model, "last_selection_counts") else {}
    p_strong_mean = ""
    if hasattr(model, "last_p_strong") and model.last_p_strong:
        p_strong_mean = sum(model.last_p_strong) / len(model.last_p_strong)
    return metrics, selection_counts, p_strong_mean


def make_row(model_name, checkpoint, degradation, metrics, selection_counts, p_strong_mean):
    return {
        "model": model_name,
        "checkpoint": checkpoint,
        "degradation": degradation,
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
        "total_time": metrics["total_time"],
        "time_per_image": metrics["time_per_image"],
        "selection_counts": selection_counts,
        "p_strong_mean": p_strong_mean,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test modern strong baselines and Final Gate v2 on light degradations"
    )
    parser.add_argument("--real_root", default=REAL_ROOT, type=str)
    parser.add_argument("--fake_root", default=FAKE_ROOT, type=str)
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
    print("MODERN STRONG BASELINES + FINAL GATE V2 LIGHT DEGRADATION TEST")
    print("No training. No checkpoint overwrite.")
    print("Output dir: {}".format(args.out_dir))
    print("Click-run config:")
    print("  CLICK_RUN_BATCH_SIZE = {}".format(CLICK_RUN_BATCH_SIZE))
    print("  CLICK_RUN_NUM_WORKERS = {}".format(CLICK_RUN_NUM_WORKERS))
    print("  CLICK_RUN_GPU = {}".format(CLICK_RUN_GPU))
    print("Degradations:")
    for mode in DEGRADATION_MODES:
        print("  {}".format(mode))
    print("=" * 80)

    base_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    test_dataset = FFHQGANData(args.real_root, args.fake_root, "test", transform=None)
    print("Test samples: real={}, fake={}, total={}".format(
        test_dataset.real_count,
        test_dataset.fake_count,
        len(test_dataset),
    ))

    model_specs = []
    for spec in MODERN_BASELINE_SPECS:
        model, report = build_timm_eval_model(spec, device)
        model_specs.append({
            "name": spec["display_name"],
            "checkpoint": spec["checkpoint"],
            "model": model,
            "report": report,
        })

    final_model, final_report = build_final_gate_v2(device)
    model_specs.append({
        "name": "Final Gate v2",
        "checkpoint": final_report["checkpoint"],
        "model": final_model,
        "report": final_report,
    })

    all_rows = []
    for spec in model_specs:
        print("\n" + "=" * 80)
        print("Testing model: {}".format(spec["name"]))
        print("Report: {}".format(spec["report"]))
        print("=" * 80)
        for degradation, degrade_type, degrade_param in DEGRADATION_MODES:
            eval_dataset = LightDegradationDataset(
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
            metrics, selection_counts, p_strong_mean = evaluate_model(
                spec["model"],
                loader,
                device,
                "{} {}".format(spec["name"], degradation),
            )
            row = make_row(
                spec["name"],
                spec["checkpoint"],
                degradation,
                metrics,
                selection_counts,
                p_strong_mean,
            )
            all_rows.append(row)
            print(
                "{} / {}: acc={:.4f}, precision={:.4f}, recall={:.4f}, "
                "f1={:.4f}, auc={:.4f}, real_acc={:.4f}, fake_acc={:.4f}, "
                "cm=[[{}, {}], [{}, {}]], time_per_image={:.6f}s".format(
                    spec["name"],
                    degradation,
                    row["acc"],
                    row["precision"],
                    row["recall"],
                    row["f1"],
                    row["auc"],
                    row["real_acc"],
                    row["fake_acc"],
                    row["TN"],
                    row["FP"],
                    row["FN"],
                    row["TP"],
                    row["time_per_image"],
                )
            )
            if selection_counts:
                print("  expert selection counts: {}".format(selection_counts))
                print("  p_strong_mean: {}".format(p_strong_mean))

    csv_path = os.path.join(args.out_dir, "modern_strong_baselines_light_degradations.csv")
    fieldnames = [
        "model",
        "checkpoint",
        "degradation",
        "acc",
        "precision",
        "recall",
        "f1",
        "auc",
        "real_acc",
        "fake_acc",
        "TN",
        "FP",
        "FN",
        "TP",
        "total_time",
        "time_per_image",
        "selection_counts",
        "p_strong_mean",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    txt_path = os.path.join(args.out_dir, "modern_strong_baselines_light_degradations.txt")
    sorted_rows = sorted(all_rows, key=lambda r: (r["degradation"], -r["acc"], -r["f1"]))
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Modern strong baselines and Final Gate v2 light degradation test\n")
        f.write("Degradations: {}\n\n".format([item[0] for item in DEGRADATION_MODES]))
        current_deg = None
        for row in sorted_rows:
            if row["degradation"] != current_deg:
                current_deg = row["degradation"]
                f.write("\n[{}]\n".format(current_deg))
                f.write("{:<20} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8} {:<18}\n".format(
                    "Model", "Acc", "Prec", "Recall", "F1", "AUC", "Real", "Fake", "CM(TN/FP/FN/TP)"))
            f.write("{:<20} {:<8.4f} {:<8.4f} {:<8.4f} {:<8.4f} {:<8.4f} {:<8.4f} {:<8.4f} {}/{}/{}/{}\n".format(
                row["model"],
                row["acc"],
                row["precision"],
                row["recall"],
                row["f1"],
                row["auc"],
                row["real_acc"],
                row["fake_acc"],
                row["TN"],
                row["FP"],
                row["FN"],
                row["TP"],
            ))

        f.write("\n[Mean accuracy by model]\n")
        for model_name in [spec["name"] for spec in model_specs]:
            model_rows = [row for row in all_rows if row["model"] == model_name]
            mean_acc = sum(row["acc"] for row in model_rows) / max(1, len(model_rows))
            mean_f1 = sum(row["f1"] for row in model_rows) / max(1, len(model_rows))
            f.write("{}: mean_acc={:.4f}, mean_f1={:.4f}\n".format(model_name, mean_acc, mean_f1))

    print("\nSaved CSV: {}".format(csv_path))
    print("Saved TXT: {}".format(txt_path))
    print("\nSummary sorted by degradation, then acc/f1:")
    current_deg = None
    for row in sorted_rows:
        if row["degradation"] != current_deg:
            current_deg = row["degradation"]
            print("\n[{}]".format(current_deg))
        print("  {:<20} acc={:.4f} f1={:.4f} auc={:.4f}".format(
            row["model"], row["acc"], row["f1"], row["auc"]))

    print("\nMean accuracy by model:")
    for model_name in [spec["name"] for spec in model_specs]:
        model_rows = [row for row in all_rows if row["model"] == model_name]
        mean_acc = sum(row["acc"] for row in model_rows) / max(1, len(model_rows))
        mean_f1 = sum(row["f1"] for row in model_rows) / max(1, len(model_rows))
        print("  {:<20} mean_acc={:.4f}, mean_f1={:.4f}".format(model_name, mean_acc, mean_f1))


if __name__ == "__main__":
    main()
