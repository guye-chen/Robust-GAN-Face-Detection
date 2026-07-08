"""
Active-path efficiency test for the final FR-WSG model.

This script does not train anything. It measures the inference cost of:
  gate -> selected expert only

This is different from FinalWeakStrongGateWrapper in
test_final_gate_and_experts_light_degradations.py, whose forward computes both
weak and strong expert logits before selecting one output. That full forward is
useful for evaluation convenience, but it overestimates the intended active-path
inference latency.
"""

import argparse
import csv
import gc
import os
import sys
import time
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.frequency_reliability_weak_strong_gate import (
    FEATURE_NAMES,
    compute_frequency_reliability_features,
    frequency_reliability_weak_strong_gate,
)
from model.inceptionv3_high_noise_bias_calib import inceptionv3_high_noise_bias_calib
from model.inceptionv3_nmfgf_noise_ra_v1_legacy import inceptionv3_nmfgf_noise_ra_v1
from test_final_gate_and_experts_light_degradations import extract_logits


ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT_DIR = os.path.join(ROOT, "weights", "final_gate_active_path_efficiency")

WEIGHTS_DIR = os.path.join(ROOT, "weights")
WEAK_EXPERT_CHECKPOINT = os.path.join(
    WEIGHTS_DIR, "inceptionv3-nmfgf-noise-ra-stability-reg", "best_model.pth"
)
STRONG_EXPERT_CHECKPOINT = os.path.join(
    WEIGHTS_DIR, "inceptionv3-high-noise-bias-calib-no-high-noise-gate", "best_model.pth"
)
GATE_CHECKPOINT = os.path.join(
    WEIGHTS_DIR, "noise-aware-weak-strong-gate", "best_weak_strong_gate.pth"
)

CLICK_RUN_DEVICE = "cuda"
CLICK_RUN_BATCH_SIZE = 1
CLICK_RUN_INPUT_SIZE = 224
CLICK_RUN_WARMUP_ITERS = 5
CLICK_RUN_BENCH_ITERS = 50
CLICK_RUN_ROUTE = "weak_strong"  # weak_strong, all, auto, weak, strong


class StabilityRegInferenceWrapper(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, x):
        _, cls_logits, _, _ = self.base_model(x, return_aux=False)
        return cls_logits


def checkpoint_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise RuntimeError("Unsupported checkpoint format: {}".format(type(checkpoint)))


def clean_state_dict(state_dict):
    return {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state_dict.items()
    }


def load_strict(target, checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        raise RuntimeError("Checkpoint not found: {}".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = clean_state_dict(checkpoint_state_dict(checkpoint))
    target.load_state_dict(state_dict, strict=True)
    return {"loaded_state_dict_keys": len(state_dict), "strict": True}


def freeze_eval(model):
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def count_params(model):
    if model is None:
        return 0
    return int(sum(p.numel() for p in model.parameters()))


def checkpoint_size_mb(path):
    if not path or not os.path.exists(path):
        return 0.0
    return os.path.getsize(path) / (1024.0 ** 2)


def final_gate_checkpoint_size_mb():
    return (
        checkpoint_size_mb(GATE_CHECKPOINT)
        + checkpoint_size_mb(WEAK_EXPERT_CHECKPOINT)
        + checkpoint_size_mb(STRONG_EXPERT_CHECKPOINT)
    )


def build_gate(device):
    gate = frequency_reliability_weak_strong_gate(
        num_features=len(FEATURE_NAMES),
        hidden_dim=32,
    ).to(device)
    report = load_strict(gate, GATE_CHECKPOINT, device)
    return freeze_eval(gate), report


def build_weak_expert(device):
    base_model = inceptionv3_nmfgf_noise_ra_v1(
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
    wrapper = StabilityRegInferenceWrapper(base_model).to(device)
    report = load_strict(base_model, WEAK_EXPERT_CHECKPOINT, device)
    return freeze_eval(wrapper), report


def build_strong_expert(device):
    model = inceptionv3_high_noise_bias_calib(
        num_classes=2,
        pretrained_backbone=False,
        alpha_init=0.02,
        ra_loss_weight=0.005,
        dense_dim=16,
        mlp_hidden=64,
        sigma_train=[25.0, 35.0, 50.0],
        sigma_eval=50.0,
        transform_input=True,
        gamma_init=0.02,
        gamma_max=0.20,
        gate_tau=0.55,
        gate_k=10.0,
        use_high_noise_gate=False,
    ).to(device)
    report = load_strict(model, STRONG_EXPERT_CHECKPOINT, device)
    return freeze_eval(model), report


class ActivePathFinalWeakStrongGate(nn.Module):
    """Run gate first, then execute only the selected expert branch.

    route_mode:
      auto   - use gate prediction per sample
      weak   - force all samples through weak expert
      strong - force all samples through strong expert
    """

    def __init__(self, gate, weak_expert, strong_expert, route_mode="auto"):
        super().__init__()
        if route_mode not in {"auto", "weak", "strong"}:
            raise ValueError("Unsupported route_mode: {}".format(route_mode))
        self.gate = gate
        self.weak_expert = weak_expert
        self.strong_expert = strong_expert
        self.route_mode = route_mode
        self.last_selection_counts = Counter()

    def reset_diagnostics(self):
        self.last_selection_counts = Counter()

    def forward(self, x):
        gate_features = compute_frequency_reliability_features(x)
        gate_logits = self.gate(gate_features)

        if self.route_mode == "weak":
            gate_pred = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        elif self.route_mode == "strong":
            gate_pred = torch.ones(x.size(0), dtype=torch.long, device=x.device)
        else:
            gate_pred = torch.argmax(gate_logits, dim=1)

        final_logits = x.new_empty((x.size(0), 2))
        weak_mask = gate_pred == 0
        strong_mask = gate_pred == 1

        if weak_mask.any():
            final_logits[weak_mask] = extract_logits(self.weak_expert(x[weak_mask]))
        if strong_mask.any():
            final_logits[strong_mask] = extract_logits(self.strong_expert(x[strong_mask]))

        weak_count = int(weak_mask.sum().detach().cpu().item())
        strong_count = int(strong_mask.sum().detach().cpu().item())
        self.last_selection_counts["weak"] += weak_count
        self.last_selection_counts["strong"] += strong_count
        return final_logits


def build_active_model(device, route_mode):
    weak_expert, weak_report = None, {}
    strong_expert, strong_report = None, {}
    gate, gate_report = build_gate(device)

    if route_mode in {"auto", "weak"}:
        weak_expert, weak_report = build_weak_expert(device)
    if route_mode in {"auto", "strong"}:
        strong_expert, strong_report = build_strong_expert(device)

    model = ActivePathFinalWeakStrongGate(
        gate=gate,
        weak_expert=weak_expert,
        strong_expert=strong_expert,
        route_mode=route_mode,
    ).to(device)
    report = {
        "gate_keys": gate_report.get("loaded_state_dict_keys", ""),
        "weak_keys": weak_report.get("loaded_state_dict_keys", ""),
        "strong_keys": strong_report.get("loaded_state_dict_keys", ""),
    }
    return freeze_eval(model), report


def count_checkpoint_params(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        state = checkpoint
    else:
        return 0
    total = 0
    for value in state.values():
        if torch.is_tensor(value):
            total += value.numel()
    del checkpoint
    gc.collect()
    return int(total)


def model_forward(model, x):
    return extract_logits(model(x))


@torch.no_grad()
def benchmark_active_path(model, device, batch_size, input_size, warmup_iters, bench_iters):
    x = torch.randn(batch_size, 3, input_size, input_size, device=device)
    model.reset_diagnostics()

    for _ in range(warmup_iters):
        _ = model_forward(model, x)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    model.reset_diagnostics()
    start = time.perf_counter()
    for _ in range(bench_iters):
        _ = model_forward(model, x)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    n_images = max(1, batch_size * bench_iters)
    ms_per_image = elapsed * 1000.0 / n_images
    fps = 1000.0 / ms_per_image if ms_per_image > 0 else 0.0
    peak_mem = (
        torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)
        if device.type == "cuda"
        else 0.0
    )
    selection_counts = dict(model.last_selection_counts)
    return ms_per_image, fps, peak_mem, selection_counts


def make_row(route_mode, model, report, args, device):
    gate_params = count_params(model.gate)
    weak_params = (
        count_params(model.weak_expert)
        if model.weak_expert is not None
        else count_checkpoint_params(WEAK_EXPERT_CHECKPOINT)
    )
    strong_params = (
        count_params(model.strong_expert)
        if model.strong_expert is not None
        else count_checkpoint_params(STRONG_EXPERT_CHECKPOINT)
    )
    total_params = gate_params + weak_params + strong_params

    ms_per_image, fps, peak_mem, selection_counts = benchmark_active_path(
        model=model,
        device=device,
        batch_size=args.batch_size,
        input_size=args.input_size,
        warmup_iters=args.warmup_iters,
        bench_iters=args.bench_iters,
    )

    if route_mode == "weak":
        active_params = gate_params + weak_params
    elif route_mode == "strong":
        active_params = gate_params + strong_params
    else:
        weak_count = int(selection_counts.get("weak", 0))
        strong_count = int(selection_counts.get("strong", 0))
        total_count = max(1, weak_count + strong_count)
        active_params = gate_params + (
            weak_params * weak_count + strong_params * strong_count
        ) / float(total_count)

    return {
        "model": "Final Gate v2 active-path",
        "route_mode": route_mode,
        "params_total_m": total_params / 1e6,
        "active_params_m": active_params / 1e6,
        "gate_params_m": gate_params / 1e6,
        "weak_expert_params_m": weak_params / 1e6,
        "strong_expert_params_m": strong_params / 1e6,
        "checkpoint_size_mb": final_gate_checkpoint_size_mb(),
        "ms_per_image": ms_per_image,
        "fps": fps,
        "peak_cuda_mem_mb": peak_mem,
        "selected_weak": selection_counts.get("weak", 0),
        "selected_strong": selection_counts.get("strong", 0),
        "batch_size": args.batch_size,
        "warmup_iters": args.warmup_iters,
        "bench_iters": args.bench_iters,
        "input_size": args.input_size,
        "device": str(device),
        "gate_checkpoint": GATE_CHECKPOINT,
        "weak_checkpoint": WEAK_EXPERT_CHECKPOINT,
        "strong_checkpoint": STRONG_EXPERT_CHECKPOINT,
        "gate_keys": report.get("gate_keys", ""),
        "weak_keys": report.get("weak_keys", ""),
        "strong_keys": report.get("strong_keys", ""),
    }


FIELDNAMES = [
    "model",
    "route_mode",
    "params_total_m",
    "active_params_m",
    "gate_params_m",
    "weak_expert_params_m",
    "strong_expert_params_m",
    "checkpoint_size_mb",
    "ms_per_image",
    "fps",
    "peak_cuda_mem_mb",
    "selected_weak",
    "selected_strong",
    "batch_size",
    "warmup_iters",
    "bench_iters",
    "input_size",
    "device",
    "gate_checkpoint",
    "weak_checkpoint",
    "strong_checkpoint",
    "gate_keys",
    "weak_keys",
    "strong_keys",
]


def write_outputs(rows, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "final_gate_active_path_efficiency.csv")
    txt_path = os.path.join(out_dir, "final_gate_active_path_efficiency.txt")

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    def fmt(value, digits=4):
        try:
            return ("{:." + str(digits) + "f}").format(float(value))
        except (TypeError, ValueError):
            return str(value)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("FINAL GATE V2 ACTIVE-PATH EFFICIENCY\n")
        f.write("=" * 100 + "\n")
        f.write("This measures gate -> selected expert only.\n")
        f.write("It does not execute both experts in one forward pass unless route_mode=auto has a mixed batch.\n\n")
        f.write(
            "{:<12s} {:>12s} {:>14s} {:>10s} {:>10s} {:>10s} {:>8s} {:>8s}\n".format(
                "route", "Params(M)", "ActiveParams", "ms/img", "FPS", "Mem(MB)", "weak", "strong"
            )
        )
        f.write("-" * 100 + "\n")
        for row in rows:
            f.write(
                "{:<12s} {:>12s} {:>14s} {:>10s} {:>10s} {:>10s} {:>8s} {:>8s}\n".format(
                    row["route_mode"],
                    fmt(row["params_total_m"]),
                    fmt(row["active_params_m"]),
                    fmt(row["ms_per_image"]),
                    fmt(row["fps"], 2),
                    fmt(row["peak_cuda_mem_mb"], 2),
                    str(row["selected_weak"]),
                    str(row["selected_strong"]),
                )
            )
        f.write("\nCheckpoints:\n")
        f.write("Gate: {}\n".format(GATE_CHECKPOINT))
        f.write("Weak: {}\n".format(WEAK_EXPERT_CHECKPOINT))
        f.write("Strong: {}\n".format(STRONG_EXPERT_CHECKPOINT))

    print("Saved CSV: {}".format(csv_path))
    print("Saved TXT: {}".format(txt_path))


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark FR-WSG active-path inference.")
    parser.add_argument("--device", default=CLICK_RUN_DEVICE, choices=["cuda", "cpu"])
    parser.add_argument("--batch_size", default=CLICK_RUN_BATCH_SIZE, type=int)
    parser.add_argument("--input_size", default=CLICK_RUN_INPUT_SIZE, type=int)
    parser.add_argument("--warmup_iters", default=CLICK_RUN_WARMUP_ITERS, type=int)
    parser.add_argument("--bench_iters", default=CLICK_RUN_BENCH_ITERS, type=int)
    parser.add_argument("--route", default=CLICK_RUN_ROUTE, choices=["weak_strong", "all", "auto", "weak", "strong"])
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR, type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print("=" * 100)
    print("FINAL GATE V2 ACTIVE-PATH EFFICIENCY")
    print("Device: {}".format(device))
    print("Batch size: {}".format(args.batch_size))
    print("Input size: {}x{}".format(args.input_size, args.input_size))
    print("Warmup iters: {}, benchmark iters: {}".format(args.warmup_iters, args.bench_iters))
    print("Route: {}".format(args.route))
    print("Output dir: {}".format(args.out_dir))
    print("=" * 100)

    if args.route == "all":
        route_modes = ["auto", "weak", "strong"]
    elif args.route == "weak_strong":
        route_modes = ["weak", "strong"]
    else:
        route_modes = [args.route]
    rows = []
    for route_mode in route_modes:
        print("\nBenchmarking active-path route_mode={}...".format(route_mode))
        model, report = build_active_model(device, route_mode)
        row = make_row(route_mode, model, report, args, device)
        rows.append(row)
        print(
            "route={} params={:.4f}M active={:.4f}M ms/img={:.4f} fps={:.2f} mem={:.2f}MB weak={} strong={}".format(
                route_mode,
                row["params_total_m"],
                row["active_params_m"],
                row["ms_per_image"],
                row["fps"],
                row["peak_cuda_mem_mb"],
                row["selected_weak"],
                row["selected_strong"],
            )
        )
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    write_outputs(rows, args.out_dir)


if __name__ == "__main__":
    main()
