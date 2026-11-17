"""Optimized-only inference evaluation with the original unseeded FR-WSG weights."""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import numpy as np
import torch
import torch.nn.functional as F

from data_loader.ffhq_gan_dataset import FFHQGANData
import test_generator_heldout_composed_weak_strong_gate_fast as inference


WEIGHTS_ROOT = os.path.join(SCRIPT_DIR, "weights")
WEAK_CHECKPOINT = os.path.join(
    WEIGHTS_ROOT, "inceptionv3-nmfgf-noise-ra-stability-reg", "best_model.pth"
)
STRONG_CHECKPOINT = os.path.join(
    WEIGHTS_ROOT,
    "inceptionv3-high-noise-bias-calib-no-high-noise-gate",
    "best_model.pth",
)
GATE_CHECKPOINT = os.path.join(
    WEIGHTS_ROOT, "noise-aware-weak-strong-gate", "best_weak_strong_gate.pth"
)
REAL_ROOT = r"E:\FFHQ_off"
FAKE_ROOT = r"E:\archive\GAN_Generated_Fake_Images\fake"
OUTPUT_ROOT = os.path.join(WEIGHTS_ROOT, "legacy_unseeded_frwsg_optimized_inference")
STAGES = (
    ("strong_expert", "Strong expert"),
    ("weak_expert", "Weak expert"),
    ("composed_gate", "FR-WSG hard gate"),
)
CONDITIONS = tuple(name for name, _, _ in inference.CONDITIONS)
LIVE_METRICS_EVERY = 25


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run only optimized inference; no original reference forward pass."
    )
    parser.add_argument("--real_root", default=REAL_ROOT)
    parser.add_argument("--fake_root", default=FAKE_ROOT)
    parser.add_argument("--output_root", default=OUTPUT_ROOT)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--noise_seed", type=int, default=42)
    parser.add_argument("--latency_warmup", type=int, default=50,
                        help="Initial images excluded from latency/FPS measurement.")
    return parser.parse_args()


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def build_optimized_weak(checkpoint_path, device):
    base_model = inference.inceptionv3_nmfgf_noise_ra_v1(
        num_classes=2,
        pretrained_backbone=False,
        alpha_init=0.02,
        ra_loss_weight=0.005,
        dense_dim=16,
        mlp_hidden=64,
        sigma_train=[5.0, 10.0, 15.0, 25.0],
        sigma_eval=10.0,
        transform_input=True,
    ).to(device)
    base_model = inference.load_checkpoint_strict(
        base_model, checkpoint_path, device, "legacy weak expert"
    )
    return inference.FastWeakInference(base_model).eval()


def prediction_rows(stage, condition, paths, sources, labels, logits, routes=None):
    probabilities = F.softmax(logits.float(), dim=1)[:, 1].detach().cpu().tolist()
    predictions = logits.argmax(dim=1).detach().cpu().tolist()
    labels = labels.detach().cpu().tolist()
    route_values = [""] * len(labels) if routes is None else routes.detach().cpu().tolist()
    return [{
        "stage": stage,
        "condition": condition,
        "path": path,
        "source": source,
        "true_label": int(label),
        "predicted_label": int(pred),
        "fake_probability": float(prob),
        "gate_route": "" if route == "" else ("weak" if int(route) == 0 else "strong"),
    } for path, source, label, pred, prob, route in zip(
        paths, sources, labels, predictions, probabilities, route_values
    )]


def finalize_metrics(stage, condition, labels_parts, logits_parts,
                     elapsed_seconds, timed_images, route_counts=None):
    labels = torch.cat(labels_parts).numpy().astype(np.int64)
    logits = torch.cat(logits_parts).numpy().astype(np.float64)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    fake_scores = exp_logits[:, 1] / exp_logits.sum(axis=1)
    metrics = inference.classification_metrics(labels, logits, fake_scores)
    latency_ms = 1000.0 * elapsed_seconds / max(1, timed_images)
    fps = timed_images / elapsed_seconds if elapsed_seconds > 0 else 0.0
    row = {
        "stage": stage,
        "condition": condition,
        "n": metrics["n"],
        "accuracy_pct": metrics["accuracy_pct"],
        "f1_fake_pct": metrics["f1_fake_pct"],
        "auc_roc_pct": metrics["auc_roc_pct"],
        "timed_images": timed_images,
        "inference_seconds": elapsed_seconds,
        "latency_ms_per_image": latency_ms,
        "fps": fps,
        "hard_latency_ms_per_image": latency_ms if stage == "composed_gate" else "",
        "hard_fps": fps if stage == "composed_gate" else "",
    }
    if route_counts is not None:
        total_routes = max(1, sum(route_counts.values()))
        row["weak_route_pct"] = 100.0 * route_counts[0] / total_routes
        row["strong_route_pct"] = 100.0 * route_counts[1] / total_routes
    return row


def run_optimized_expert(stage, model, condition, sigma, samples, device, args, label):
    loader = inference.make_loader(
        samples, condition, sigma, args.noise_seed, device,
        args.batch_size, args.num_workers,
    )
    labels_parts, logits_parts, rows = [], [], []
    elapsed_seconds = 0.0
    timed_images = 0
    seen_images = 0
    live_interval = max(1, LIVE_METRICS_EVERY // args.batch_size)
    progress = inference.tqdm(
        loader, desc="{} {} optimized".format(stage, condition), dynamic_ncols=True
    )

    with torch.inference_mode():
        for batch_idx, (images, labels, paths, sources) in enumerate(progress):
            images = images.to(device, non_blocking=(device.type == "cuda"))
            labels_device = labels.to(device, non_blocking=(device.type == "cuda"))
            synchronize(device)
            start = time.perf_counter()
            logits = model(images)
            synchronize(device)
            batch_seconds = time.perf_counter() - start

            if seen_images >= args.latency_warmup:
                elapsed_seconds += batch_seconds
                timed_images += int(images.size(0))
            seen_images += int(images.size(0))
            labels_parts.append(labels_device.cpu())
            logits_parts.append(logits.detach().float().cpu())
            rows.extend(prediction_rows(stage, condition, paths, sources, labels_device, logits))

            if (batch_idx + 1) % live_interval == 0 or batch_idx + 1 == len(loader):
                acc, auc = inference.running_metric_text(labels_parts, logits_parts)
                latency = 1000.0 * elapsed_seconds / max(1, timed_images)
                fps = timed_images / elapsed_seconds if elapsed_seconds > 0 else 0.0
                progress.set_postfix({
                    "ACC": acc,
                    "AUC": auc,
                    "ms/img": "{:.2f}".format(latency) if timed_images else "warmup",
                    "FPS": "{:.2f}".format(fps) if timed_images else "warmup",
                }, refresh=False)

    return finalize_metrics(
        stage, condition, labels_parts, logits_parts,
        elapsed_seconds, timed_images,
    ), rows


def run_optimized_composed(condition, sigma, gate_label, samples, gate,
                           weak_model, strong_model, device, args):
    loader = inference.make_loader(
        samples, condition, sigma, args.noise_seed, device,
        args.batch_size, args.num_workers,
    )
    labels_parts, logits_parts, rows = [], [], []
    route_counts = {0: 0, 1: 0}
    elapsed_seconds = 0.0
    timed_images = 0
    seen_images = 0
    live_interval = max(1, LIVE_METRICS_EVERY // args.batch_size)
    progress = inference.tqdm(
        loader, desc="FR-WSG {} optimized hard route".format(condition),
        dynamic_ncols=True,
    )

    with torch.inference_mode():
        for batch_idx, (images, labels, paths, sources) in enumerate(progress):
            images = images.to(device, non_blocking=(device.type == "cuda"))
            labels_device = labels.to(device, non_blocking=(device.type == "cuda"))
            synchronize(device)
            start = time.perf_counter()

            gate_features = inference.compute_frequency_reliability_features(images)
            routes = gate(gate_features).argmax(dim=1)
            final_logits = torch.empty(
                (images.size(0), 2), device=device, dtype=torch.float32
            )
            for route, expert in ((0, weak_model), (1, strong_model)):
                indices = torch.nonzero(routes == route, as_tuple=False).flatten()
                if indices.numel():
                    selected_images = images.index_select(0, indices)
                    selected_logits = expert(selected_images)
                    final_logits.index_copy_(0, indices, selected_logits.float())

            synchronize(device)
            batch_seconds = time.perf_counter() - start
            if seen_images >= args.latency_warmup:
                elapsed_seconds += batch_seconds
                timed_images += int(images.size(0))
            seen_images += int(images.size(0))
            labels_parts.append(labels_device.cpu())
            logits_parts.append(final_logits.detach().float().cpu())
            route_counts[0] += int((routes == 0).sum().item())
            route_counts[1] += int((routes == 1).sum().item())
            rows.extend(prediction_rows(
                "composed_gate", condition, paths, sources,
                labels_device, final_logits, routes,
            ))

            if (batch_idx + 1) % live_interval == 0 or batch_idx + 1 == len(loader):
                acc, auc = inference.running_metric_text(labels_parts, logits_parts)
                latency = 1000.0 * elapsed_seconds / max(1, timed_images)
                fps = timed_images / elapsed_seconds if elapsed_seconds > 0 else 0.0
                progress.set_postfix({
                    "ACC": acc,
                    "AUC": auc,
                    "ms/img": "{:.2f}".format(latency) if timed_images else "warmup",
                    "HardFPS": "{:.2f}".format(fps) if timed_images else "warmup",
                }, refresh=False)

    return finalize_metrics(
        "composed_gate", condition, labels_parts, logits_parts,
        elapsed_seconds, timed_images, route_counts,
    ), rows


def print_summary(rows):
    print("\nOptimized-only results (percent; latency excludes image loading/preprocessing)")
    print("{:<18} {:<9} {:>9} {:>10} {:>10} {:>13} {:>10}".format(
        "Stage", "Condition", "ACC", "Fake-F1", "ROC-AUC", "Hard ms/img", "Hard FPS"
    ))
    for stage, stage_label in STAGES:
        for condition in CONDITIONS:
            row = next(item for item in rows
                       if item["stage"] == stage and item["condition"] == condition)
            hard_latency = row["hard_latency_ms_per_image"]
            hard_fps = row["hard_fps"]
            print("{:<18} {:<9} {:>8.4f} {:>9.4f} {:>9.4f} {:>13} {:>10}".format(
                stage_label, condition,
                row["accuracy_pct"], row["f1_fake_pct"], row["auc_roc_pct"],
                "{:.3f}".format(hard_latency) if hard_latency != "" else "-",
                "{:.3f}".format(hard_fps) if hard_fps != "" else "-",
            ))


def main():
    args = parse_args()
    if args.batch_size < 1 or args.num_workers < 0 or args.latency_warmup < 0:
        raise ValueError("batch_size must be positive; workers and warmup must be nonnegative.")

    checkpoints = {
        "weak_expert": WEAK_CHECKPOINT,
        "strong_expert": STRONG_CHECKPOINT,
        "gate": GATE_CHECKPOINT,
    }
    for name, path in checkpoints.items():
        if not os.path.isfile(path):
            raise FileNotFoundError("{} checkpoint not found: {}".format(name, path))

    if args.device == "cpu":
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif args.device == "cuda":
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    else:
        device = torch.device("cpu")

    output_dir = os.path.join(args.output_root, datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    os.makedirs(output_dir, exist_ok=False)
    print("Optimized-only inference: original reference forward is disabled.")
    print("Skipped: internal noisy-copy inference and absolute-difference auxiliary branch.")
    print("Retained: clean InceptionV3, clean frequency features, bias calibration, gate, hard routing.")
    print("Selected checkpoints have no fixed training seed. External noise seed={}.".format(
        args.noise_seed
    ))
    print("Device={} | batch_size={} | latency warmup={} images".format(
        device, args.batch_size, args.latency_warmup
    ))
    for name, path in checkpoints.items():
        print("{} checkpoint: {}".format(name, path))

    base_dataset = FFHQGANData(args.real_root, args.fake_root, "test", transform=None)
    if not base_dataset.samples:
        raise RuntimeError("The test split is empty. Check real_root and fake_root.")
    real_samples = [sample for sample in base_dataset.samples if sample[1] == 0]
    fake_samples = [sample for sample in base_dataset.samples if sample[1] == 1]
    samples = []
    for index in range(max(len(real_samples), len(fake_samples))):
        if index < len(real_samples):
            path, label, source = real_samples[index]
            samples.append({"path": path, "label": int(label), "source": source})
        if index < len(fake_samples):
            path, label, source = fake_samples[index]
            samples.append({"path": path, "label": int(label), "source": source})
    missing = [sample["path"] for sample in samples if not os.path.isfile(sample["path"])]
    if missing:
        raise FileNotFoundError("Missing test image(s), first five:\n{}".format(
            "\n".join(missing[:5])
        ))
    print("Test images: real={} fake={} total={}".format(
        base_dataset.real_count, base_dataset.fake_count, len(samples)
    ))
    print("Fake source counts: {}".format(base_dataset.get_gan_source_counts()))

    weak_model = build_optimized_weak(WEAK_CHECKPOINT, device)
    strong_base = inference.build_strong_expert(STRONG_CHECKPOINT, device)
    strong_model = inference.FastStrongInference(strong_base).eval()
    gate = inference.build_gate(GATE_CHECKPOINT, device)

    metrics_rows, prediction_rows_all = [], []
    print("\nStage order: 1) strong expert  2) weak expert  3) FR-WSG hard gate")
    for stage, _ in STAGES:
        for condition, sigma, gate_label in inference.CONDITIONS:
            print("\n=== {} | {} ===".format(stage, condition))
            if stage == "strong_expert":
                row, image_rows = run_optimized_expert(
                    stage, strong_model, condition, sigma, samples,
                    device, args, gate_label,
                )
            elif stage == "weak_expert":
                row, image_rows = run_optimized_expert(
                    stage, weak_model, condition, sigma, samples,
                    device, args, gate_label,
                )
            else:
                row, image_rows = run_optimized_composed(
                    condition, sigma, gate_label, samples, gate,
                    weak_model, strong_model, device, args,
                )
            metrics_rows.append(row)
            prediction_rows_all.extend(image_rows)
            print("{} {}: ACC={:.4f}% Fake-F1={:.4f}% AUC={:.4f}%".format(
                stage, condition, row["accuracy_pct"],
                row["f1_fake_pct"], row["auc_roc_pct"],
            ))
            if stage == "composed_gate":
                print("  Hard latency={:.3f} ms/image | Hard FPS={:.3f}".format(
                    row["hard_latency_ms_per_image"], row["hard_fps"]
                ))

    metrics_path = os.path.join(output_dir, "metrics_optimized_only.csv")
    predictions_path = os.path.join(output_dir, "optimized_per_image_predictions.csv")
    write_csv(metrics_path, metrics_rows)
    write_csv(predictions_path, prediction_rows_all)
    print_summary(metrics_rows)

    summary_path = os.path.join(output_dir, "run_summary.json")
    summary = {
        "inference_mode": "optimized_only",
        "original_reference_forward_ran": False,
        "internal_noisy_copy_inference": False,
        "absolute_difference_auxiliary_inference": False,
        "external_noise_seed": args.noise_seed,
        "device": str(device),
        "batch_size": args.batch_size,
        "latency_warmup_images": args.latency_warmup,
        "test_images": {"real": base_dataset.real_count,
                         "fake": base_dataset.fake_count, "total": len(samples)},
        "checkpoints": checkpoints,
        "hard_latency_definition": "average synchronized forward wall time per image for gate plus selected expert; excludes image loading and preprocessing",
        "hard_fps_definition": "timed images divided by accumulated synchronized forward time",
        "metrics_csv": metrics_path,
        "predictions_csv": predictions_path,
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print("\nMetrics: {}".format(metrics_path))
    print("Predictions: {}".format(predictions_path))
    print("Run summary: {}".format(summary_path))


if __name__ == "__main__":
    main()
