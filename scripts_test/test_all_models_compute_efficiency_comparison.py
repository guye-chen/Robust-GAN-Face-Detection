"""
Compute / efficiency comparison for all baselines and the final model.

This script does not train any model and does not run the full test set.
It loads checkpoints one by one and reports:
  - parameter count
  - checkpoint size
  - optional FLOPs / MACs when thop is installed
  - dummy-input inference speed
  - peak CUDA memory when CUDA is used

Default device is CUDA because the paper should report GPU inference cost.
Models are loaded and released one by one to reduce peak GPU memory usage.
"""

import argparse
import csv
import gc
import os
import subprocess
import sys
import time
from collections import OrderedDict

import torch
import torch.nn as nn
from torchvision import models

try:
    import timm
except ImportError:
    timm = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.frequency_reliability_weak_strong_gate import (
    FEATURE_NAMES,
    compute_frequency_reliability_features,
    frequency_reliability_weak_strong_gate,
)
from model.inceptionv3_high_noise_bias_calib import inceptionv3_high_noise_bias_calib
from model.inceptionv3_nmfgf_noise_ra_v1_legacy import inceptionv3_nmfgf_noise_ra_v1
from test_final_gate_and_experts_light_degradations import (
    FinalWeakStrongGateWrapper,
    StabilityRegInferenceWrapper,
    extract_logits,
    freeze_eval,
)


ROOT = os.path.dirname(os.path.abspath(__file__))

# ==============================
# Click-run default configuration
# ==============================
CLICK_RUN_DEVICE = "cuda"
CLICK_RUN_BATCH_SIZE = 1
CLICK_RUN_WARMUP_ITERS = 2
CLICK_RUN_BENCH_ITERS = 10
CLICK_RUN_INPUT_SIZE = 224
CLICK_RUN_MODEL = "all"  # "all" or one key in MODEL_SPECS.
CLICK_RUN_COMPUTE_MACS = False  # thop can OOM on 4GB GPUs; enable only if needed.
CLICK_RUN_ISOLATED_SUBPROCESSES = True
CLICK_RUN_OUTPUT_DIR = os.path.join(
    ROOT, "weights", "all_models_compute_efficiency_comparison"
)


WEIGHTS_DIR = os.path.join(ROOT, "weights")
CHECKPOINT_DIR = os.path.join(ROOT, "checkpoint")

WEAK_EXPERT_CHECKPOINT = os.path.join(
    WEIGHTS_DIR, "inceptionv3-nmfgf-noise-ra-stability-reg", "best_model.pth"
)
STRONG_EXPERT_CHECKPOINT = os.path.join(
    WEIGHTS_DIR, "inceptionv3-high-noise-bias-calib-no-high-noise-gate", "best_model.pth"
)
GATE_CHECKPOINT = os.path.join(
    WEIGHTS_DIR, "noise-aware-weak-strong-gate", "best_weak_strong_gate.pth"
)


MODEL_SPECS = OrderedDict([
    # Eight original baselines.
    ("resnet50v2", {
        "category": "Traditional CNN-based Methods",
        "display": "ResNet50V2",
        "kind": "timm",
        "model_name": "resnetv2_50",
        "checkpoint": os.path.join(WEIGHTS_DIR, "best_model_resnet50v2.pth"),
    }),
    ("mobilenetv2", {
        "category": "Traditional CNN-based Methods",
        "display": "MobileNetV2",
        "kind": "torchvision",
        "arch": "mobilenetv2",
        "checkpoint": os.path.join(CHECKPOINT_DIR, "mobilenetv2", "best_model.pth"),
    }),
    ("densenet121", {
        "category": "Traditional CNN-based Methods",
        "display": "DenseNet121",
        "kind": "torchvision",
        "arch": "densenet121",
        "checkpoint": os.path.join(CHECKPOINT_DIR, "densenet121", "best_model.pth"),
    }),
    ("vgg16", {
        "category": "Traditional CNN-based Methods",
        "display": "VGG16",
        "kind": "torchvision",
        "arch": "vgg16",
        "checkpoint": os.path.join(WEIGHTS_DIR, "best_model_vgg16.pth"),
    }),
    ("vgg19", {
        "category": "Traditional CNN-based Methods",
        "display": "VGG19",
        "kind": "torchvision",
        "arch": "vgg19",
        "checkpoint": os.path.join(WEIGHTS_DIR, "best_model_vgg19.pth"),
    }),
    ("inceptionv3", {
        "category": "Traditional CNN-based Methods",
        "display": "InceptionV3",
        "kind": "torchvision",
        "arch": "inceptionv3",
        "checkpoint": os.path.join(WEIGHTS_DIR, "best_model_inceptionv3.pth"),
    }),
    ("inceptionresnetv2", {
        "category": "Traditional CNN-based Methods",
        "display": "InceptionResNetV2",
        "kind": "timm",
        "model_name": "inception_resnet_v2",
        "checkpoint": os.path.join(WEIGHTS_DIR, "best_model_inceptionresnetv2.pth"),
    }),
    ("xception", {
        "category": "Traditional CNN-based Methods",
        "display": "Xception",
        "kind": "timm",
        "model_name": "xception",
        "checkpoint": os.path.join(WEIGHTS_DIR, "best_model_xception.pth"),
    }),

    # Modern strong baselines.
    ("convnext_tiny", {
        "category": "Modern Strong Baselines",
        "display": "ConvNeXt-Tiny",
        "kind": "timm",
        "model_name": "convnext_tiny.fb_in1k",
        "checkpoint": os.path.join(WEIGHTS_DIR, "convnext_tiny", "best_model.pth"),
    }),
    ("swin_tiny", {
        "category": "Modern Strong Baselines",
        "display": "Swin-Tiny",
        "kind": "timm",
        "model_name": "swin_tiny_patch4_window7_224.ms_in1k",
        "checkpoint": os.path.join(WEIGHTS_DIR, "swin_tiny", "best_model.pth"),
    }),
    ("fastvit_sa12", {
        "category": "Modern Strong Baselines",
        "display": "FastViT-SA12",
        "kind": "timm",
        "model_name": "fastvit_sa12.apple_in1k",
        "checkpoint": os.path.join(WEIGHTS_DIR, "fastvit_sa12", "best_model.pth"),
    }),
    ("efficientnet_b3", {
        "category": "Modern Strong Baselines",
        "display": "EfficientNet-B3",
        "kind": "timm",
        "model_name": "efficientnet_b3.ra2_in1k",
        "checkpoint": os.path.join(WEIGHTS_DIR, "efficientnet_b3", "best_model.pth"),
    }),

    # Proposed experts and final gate.
    ("stability_reg", {
        "category": "Ours / Components",
        "display": "Weak Expert: Stability-Reg",
        "kind": "weak_expert",
        "checkpoint": WEAK_EXPERT_CHECKPOINT,
    }),
    ("strong_bias_calib_no_gate", {
        "category": "Ours / Components",
        "display": "Strong Expert: Bias-Calib w/o gate",
        "kind": "strong_expert",
        "checkpoint": STRONG_EXPERT_CHECKPOINT,
    }),
    ("final_gate_v2", {
        "category": "Ours / Components",
        "display": "Final Gate v2",
        "kind": "final_gate",
        "checkpoint": GATE_CHECKPOINT,
    }),
])


def checkpoint_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"], "model_state_dict"
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"], "state_dict"
    if isinstance(checkpoint, dict):
        return checkpoint, "raw_state_dict"
    raise RuntimeError("Unsupported checkpoint format: {}".format(type(checkpoint)))


def clean_state_dict(state_dict):
    clean = OrderedDict()
    for key, value in state_dict.items():
        clean[key[7:] if key.startswith("module.") else key] = value
    return clean


def load_strict(model, checkpoint_path, device, load_target=None):
    target = load_target if load_target is not None else model
    if not os.path.exists(checkpoint_path):
        raise RuntimeError("Checkpoint not found: {}".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict, fmt = checkpoint_state_dict(checkpoint)
    state_dict = clean_state_dict(state_dict)
    target.load_state_dict(state_dict, strict=True)
    return {
        "checkpoint_format": fmt,
        "loaded_state_dict_keys": len(state_dict),
        "strict": True,
    }


def build_torchvision_model(arch):
    if arch == "mobilenetv2":
        model = models.mobilenet_v2(pretrained=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
        return model
    if arch == "densenet121":
        model = models.densenet121(pretrained=False)
        model.classifier = nn.Linear(model.classifier.in_features, 2)
        return model
    if arch == "vgg16":
        model = models.vgg16(pretrained=False)
        model.classifier[6] = nn.Linear(4096, 2)
        return model
    if arch == "vgg19":
        model = models.vgg19(pretrained=False)
        model.classifier[6] = nn.Linear(4096, 2)
        return model
    if arch == "inceptionv3":
        model = models.inception_v3(pretrained=False, aux_logits=True)
        model.aux_logits = False
        model.AuxLogits = None
        model.fc = nn.Linear(model.fc.in_features, 2)
        model.transform_input = True
        return model
    raise ValueError("Unknown torchvision arch: {}".format(arch))


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
    report = load_strict(wrapper, WEAK_EXPERT_CHECKPOINT, device, load_target=base_model)
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


def build_gate(device):
    gate = frequency_reliability_weak_strong_gate(
        num_features=len(FEATURE_NAMES),
        hidden_dim=32,
    ).to(device)
    report = load_strict(gate, GATE_CHECKPOINT, device)
    return freeze_eval(gate), report


def build_model(spec, device):
    kind = spec["kind"]
    report = {}

    if kind == "torchvision":
        model = build_torchvision_model(spec["arch"]).to(device)
        report = load_strict(model, spec["checkpoint"], device)
        return freeze_eval(model), report

    if kind == "timm":
        if timm is None:
            raise RuntimeError("timm is not installed.")
        model = timm.create_model(spec["model_name"], pretrained=False, num_classes=2).to(device)
        report = load_strict(model, spec["checkpoint"], device)
        return freeze_eval(model), report

    if kind == "weak_expert":
        return build_weak_expert(device)

    if kind == "strong_expert":
        return build_strong_expert(device)

    if kind == "final_gate":
        weak_expert, weak_report = build_weak_expert(device)
        strong_expert, strong_report = build_strong_expert(device)
        gate, gate_report = build_gate(device)
        model = FinalWeakStrongGateWrapper(gate, weak_expert, strong_expert).to(device)
        report = {
            "checkpoint_format": "gate + two expert checkpoints",
            "loaded_state_dict_keys": (
                gate_report["loaded_state_dict_keys"]
                + weak_report["loaded_state_dict_keys"]
                + strong_report["loaded_state_dict_keys"]
            ),
            "strict": True,
        }
        return freeze_eval(model), report

    raise ValueError("Unsupported model kind: {}".format(kind))


def extract_single_logits(outputs):
    if isinstance(outputs, tuple):
        return outputs[0]
    if hasattr(outputs, "logits"):
        return outputs.logits
    return outputs


def count_params(model):
    return int(sum(p.numel() for p in model.parameters()))


def count_trainable_params(model):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


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


def model_forward(model, x):
    return extract_single_logits(model(x))


def try_compute_macs(model, device, input_size, enabled):
    if not enabled:
        return "", "MACs disabled by CLICK_RUN_COMPUTE_MACS"
    try:
        from thop import profile
    except Exception as exc:
        return "", "thop unavailable: {}".format(exc)

    dummy = torch.randn(1, 3, input_size, input_size, device=device)
    try:
        with torch.no_grad():
            macs, _ = profile(model, inputs=(dummy,), verbose=False)
        return float(macs) / 1e9, ""
    except Exception as exc:
        return "", "FLOPs failed: {}".format(exc)


@torch.no_grad()
def benchmark_inference(model, device, batch_size, input_size, warmup_iters, bench_iters):
    x = torch.randn(batch_size, 3, input_size, input_size, device=device)
    for _ in range(warmup_iters):
        _ = model_forward(model, x)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

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
    return ms_per_image, fps, peak_mem


def active_params_for_final_gate(model):
    gate_params = count_params(model.gate)
    weak_params = count_params(model.weak_expert)
    strong_params = count_params(model.strong_expert)
    return gate_params, weak_params, strong_params


def benchmark_one(key, spec, device, args):
    print("\nBenchmarking {}...".format(spec["display"]))
    model, report = build_model(spec, device)

    total_params = count_params(model)
    trainable_params = count_trainable_params(model)
    active_weak = ""
    active_strong = ""
    if spec["kind"] == "final_gate":
        gate_params, weak_params, strong_params = active_params_for_final_gate(model)
        active_weak = (gate_params + weak_params) / 1e6
        active_strong = (gate_params + strong_params) / 1e6

    macs_g, flops_note = try_compute_macs(
        model, device, args.input_size, args.compute_macs
    )
    ms_per_image, fps, peak_mem = benchmark_inference(
        model,
        device,
        args.batch_size,
        args.input_size,
        args.warmup_iters,
        args.bench_iters,
    )

    checkpoint_mb = (
        final_gate_checkpoint_size_mb()
        if spec["kind"] == "final_gate"
        else checkpoint_size_mb(spec["checkpoint"])
    )

    row = {
        "key": key,
        "category": spec["category"],
        "model": spec["display"],
        "checkpoint": spec["checkpoint"],
        "checkpoint_exists": os.path.exists(spec["checkpoint"]),
        "checkpoint_size_mb": checkpoint_mb,
        "params_m": total_params / 1e6,
        "trainable_params_m_after_freeze": trainable_params / 1e6,
        "active_params_weak_m": active_weak,
        "active_params_strong_m": active_strong,
        "macs_g": macs_g,
        "ms_per_image": ms_per_image,
        "fps": fps,
        "peak_cuda_mem_mb": peak_mem,
        "loaded_state_dict_keys": report.get("loaded_state_dict_keys", ""),
        "strict_load": report.get("strict", ""),
        "checkpoint_format": report.get("checkpoint_format", ""),
        "flops_note": flops_note,
    }

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    return row


FIELDNAMES = [
    "key",
    "category",
    "model",
    "params_m",
    "active_params_weak_m",
    "active_params_strong_m",
    "checkpoint_size_mb",
    "macs_g",
    "ms_per_image",
    "fps",
    "peak_cuda_mem_mb",
    "checkpoint_exists",
    "strict_load",
    "loaded_state_dict_keys",
    "checkpoint_format",
    "flops_note",
    "checkpoint",
]


def write_outputs(rows, out_dir, args):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "all_models_compute_efficiency_comparison.csv")
    txt_path = os.path.join(out_dir, "all_models_compute_efficiency_comparison.txt")

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("ALL-MODEL COMPUTE / EFFICIENCY COMPARISON\n")
        f.write("=" * 120 + "\n")
        f.write("Device: {}\n".format(args.device))
        f.write("Input: batch_size={}, image={}x{}\n".format(
            args.batch_size, args.input_size, args.input_size
        ))
        f.write("Warmup iters={}, benchmark iters={}\n".format(
            args.warmup_iters, args.bench_iters
        ))
        f.write("FLOPs column reports MACs(G) when thop is available.\n")
        f.write("Final Gate v2 loaded params include gate + weak expert + strong expert.\n")
        f.write("Final Gate v2 active params show gate + selected expert path.\n\n")
        f.write(
            "{:<34s} {:<30s} {:>10s} {:>10s} {:>10s} {:>10s} {:>10s}\n".format(
                "Category", "Model", "Params(M)", "MACs(G)", "ms/img", "FPS", "Mem(MB)"
            )
        )
        f.write("-" * 120 + "\n")
        def fmt_float(value, digits=4):
            if value == "" or value is None:
                return "NA"
            try:
                return ("{:." + str(digits) + "f}").format(float(value))
            except (TypeError, ValueError):
                return str(value)

        for row in rows:
            f.write(
                "{:<34s} {:<30s} {:>10s} {:>10s} {:>10s} {:>10s} {:>10s}\n".format(
                    row["category"][:34],
                    row["model"][:30],
                    fmt_float(row["params_m"]),
                    fmt_float(row["macs_g"]),
                    fmt_float(row["ms_per_image"]),
                    fmt_float(row["fps"], digits=2),
                    fmt_float(row["peak_cuda_mem_mb"], digits=2),
                )
            )

    print("\nSaved CSV: {}".format(csv_path))
    print("Saved TXT: {}".format(txt_path))


def write_single_row(row, out_dir, key):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "single_{}.csv".format(key))
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerow(row)
    print("Saved single-model CSV: {}".format(path))


def read_single_rows(out_dir, keys):
    rows = []
    for key in keys:
        path = os.path.join(out_dir, "single_{}.csv".format(key))
        if not os.path.exists(path):
            spec = MODEL_SPECS[key]
            rows.append({
                "key": key,
                "category": spec["category"],
                "model": spec["display"],
                "checkpoint": spec["checkpoint"],
                "checkpoint_exists": os.path.exists(spec["checkpoint"]),
                "checkpoint_size_mb": checkpoint_size_mb(spec["checkpoint"]),
                "params_m": "",
                "active_params_weak_m": "",
                "active_params_strong_m": "",
                "macs_g": "",
                "ms_per_image": "",
                "fps": "",
                "peak_cuda_mem_mb": "",
                "strict_load": False,
                "loaded_state_dict_keys": "",
                "checkpoint_format": "",
                "flops_note": "FAILED: child process did not produce output",
            })
            continue
        with open(path, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
                break
    return rows


def run_isolated_all(args):
    keys = list(MODEL_SPECS.keys())
    os.makedirs(args.out_dir, exist_ok=True)
    script_path = os.path.abspath(__file__)
    print("Running each model in an isolated Python process to fully release CUDA.")
    for key in keys:
        print("\n" + "=" * 80)
        print("Isolated benchmark: {}".format(key))
        print("=" * 80)
        cmd = [
            sys.executable,
            script_path,
            "--model", key,
            "--device", args.device,
            "--batch_size", str(args.batch_size),
            "--warmup_iters", str(args.warmup_iters),
            "--bench_iters", str(args.bench_iters),
            "--input_size", str(args.input_size),
            "--out_dir", args.out_dir,
            "--child",
        ]
        if args.compute_macs:
            cmd.append("--compute_macs")
        proc = subprocess.run(cmd, cwd=ROOT)
        if proc.returncode != 0:
            print("  WARNING: child process failed for {} with exit code {}".format(
                key, proc.returncode
            ))
    rows = read_single_rows(args.out_dir, keys)
    write_outputs(rows, args.out_dir, args)


def parse_args():
    parser = argparse.ArgumentParser(description="All-model compute / efficiency comparison")
    parser.add_argument("--model", default=CLICK_RUN_MODEL, choices=["all"] + list(MODEL_SPECS.keys()))
    parser.add_argument("--device", default=CLICK_RUN_DEVICE, choices=["cpu", "cuda"])
    parser.add_argument("--batch_size", type=int, default=CLICK_RUN_BATCH_SIZE)
    parser.add_argument("--warmup_iters", type=int, default=CLICK_RUN_WARMUP_ITERS)
    parser.add_argument("--bench_iters", type=int, default=CLICK_RUN_BENCH_ITERS)
    parser.add_argument("--input_size", type=int, default=CLICK_RUN_INPUT_SIZE)
    parser.add_argument(
        "--run_isolated",
        action="store_true",
        default=CLICK_RUN_ISOLATED_SUBPROCESSES,
        help="When model=all, run each model in a separate Python process.",
    )
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--compute_macs",
        action="store_true",
        default=CLICK_RUN_COMPUTE_MACS,
        help="Enable thop MACs profiling. Disabled by default because it can OOM on 4GB GPUs.",
    )
    parser.add_argument("--out_dir", default=CLICK_RUN_OUTPUT_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
    device = torch.device(args.device)

    print("=" * 80)
    print("ALL-MODEL COMPUTE / EFFICIENCY COMPARISON")
    print("=" * 80)
    print("CLICK_RUN_DEVICE = {}".format(CLICK_RUN_DEVICE))
    print("CLICK_RUN_MODEL = {}".format(CLICK_RUN_MODEL))
    print("Device = {}".format(device))
    print("Batch size = {}".format(args.batch_size))
    print("Warmup iters = {}".format(args.warmup_iters))
    print("Benchmark iters = {}".format(args.bench_iters))
    print("Compute MACs = {}".format(args.compute_macs))
    print("Run isolated subprocesses = {}".format(args.run_isolated))
    print("Output dir = {}".format(args.out_dir))
    print("=" * 80)

    if args.model == "all" and args.run_isolated and not args.child:
        run_isolated_all(args)
        return

    keys = list(MODEL_SPECS.keys()) if args.model == "all" else [args.model]
    rows = []
    for key in keys:
        spec = MODEL_SPECS[key]
        try:
            rows.append(benchmark_one(key, spec, device, args))
        except Exception as exc:
            print("  FAILED: {}".format(exc))
            gc.collect()
            if device.type == "cuda":
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
            rows.append({
                "key": key,
                "category": spec["category"],
                "model": spec["display"],
                "checkpoint": spec["checkpoint"],
                "checkpoint_exists": os.path.exists(spec["checkpoint"]),
                "checkpoint_size_mb": checkpoint_size_mb(spec["checkpoint"]),
                "params_m": "",
                "active_params_weak_m": "",
                "active_params_strong_m": "",
                "macs_g": "",
                "ms_per_image": "",
                "fps": "",
                "peak_cuda_mem_mb": "",
                "strict_load": False,
                "loaded_state_dict_keys": "",
                "checkpoint_format": "",
                "flops_note": "FAILED: {}".format(exc),
            })

    if args.child and len(rows) == 1:
        write_single_row(rows[0], args.out_dir, keys[0])
    else:
        write_outputs(rows, args.out_dir, args)


if __name__ == "__main__":
    main()
