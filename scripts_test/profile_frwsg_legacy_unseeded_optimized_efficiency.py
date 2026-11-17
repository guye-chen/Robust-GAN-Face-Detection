"""Kaggle-ready efficiency profile for optimized FR-WSG inference.

Reports the weak expert, strong expert, and final hard-routed model only.
Uses one synthetic 224x224 input, strict checkpoint loading, process-isolated
measurements, resumable per-model JSON records, and the CSV/TXT layout used by
the existing Kaggle 22-detector efficiency scripts.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import subprocess
import sys
import time
import traceback
import shutil
from datetime import datetime
from pathlib import Path


def find_project_root() -> Path:
    explicit = os.environ.get("AIGC_PROJECT_ROOT", "").strip()
    if explicit:
        candidate = Path(explicit)
        if (candidate / "model" / "inceptionv3_nmfgf_noise_ra_v1_legacy.py").is_file():
            return candidate
        raise FileNotFoundError("AIGC_PROJECT_ROOT is not a FR-WSG source root: {}".format(candidate))

    here = Path(__file__).resolve().parent
    for candidate in (here, *here.parents):
        if (candidate / "model" / "inceptionv3_nmfgf_noise_ra_v1_legacy.py").is_file():
            return candidate

    input_root = Path("/kaggle/input")
    if input_root.is_dir():
        matches = list(input_root.rglob("inceptionv3_nmfgf_noise_ra_v1_legacy.py"))
        if len(matches) == 1:
            return matches[0].parents[1]
        if len(matches) > 1:
            raise RuntimeError(
                "Multiple FR-WSG source trees found under /kaggle/input; set AIGC_PROJECT_ROOT."
            )
    raise FileNotFoundError("Could not locate the FR-WSG model source under this script or /kaggle/input.")


ROOT = find_project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
if KAGGLE_WORKING_ROOT.is_dir():
    DEFAULT_OUTPUT_ROOT = KAGGLE_WORKING_ROOT / "frwsg_optimized_efficiency"
else:
    DEFAULT_OUTPUT_ROOT = ROOT / "weights" / "frwsg_optimized_efficiency"

DEVICE = "cuda"
INPUT_SIZE = 224
BATCH_SIZE = 1
WARMUP_ITERS = 5
BENCH_ITERS = 30

WEAK_DIR = "inceptionv3-nmfgf-noise-ra-stability-reg"
STRONG_DIR = "inceptionv3-high-noise-bias-calib-no-high-noise-gate"
GATE_DIR = "noise-aware-weak-strong-gate"

MODEL_SPECS = [
    {"key": "weak_expert", "group": "Ours / Components", "label": "Weak Expert: Stability-Reg"},
    {"key": "strong_expert", "group": "Ours / Components", "label": "Strong Expert: Bias-Calib"},
    {"key": "final_gate", "group": "Ours / Components", "label": "FR-WSG: optimized hard routing"},
]

FIELDS = [
    "key", "group", "model", "status", "device", "precision", "input_shape",
    "parameters_m", "resident_parameters_m", "weak_active_parameters_m",
    "strong_active_parameters_m", "flops_g", "macs_g", "weak_path_flops_g",
    "weak_path_macs_g", "strong_path_flops_g", "strong_path_macs_g",
    "latency_ms_per_image", "fps", "weak_path_latency_ms", "strong_path_latency_ms",
    "weak_path_fps", "strong_path_fps", "peak_cuda_allocated_mb", "checkpoint",
    "notes", "error",
]


def find_checkpoint(env_name: str, directory: str, filename: str) -> Path:
    configured = os.environ.get(env_name, "").strip()
    if configured:
        path = Path(configured)
        if not path.is_file():
            raise FileNotFoundError("{} points to a missing file: {}".format(env_name, path))
        return path

    expected = ROOT / "weights" / directory / filename
    if expected.is_file():
        return expected

    matches = []
    search_roots = [ROOT, ROOT.parent, KAGGLE_INPUT_ROOT, KAGGLE_WORKING_ROOT]
    for root in search_roots:
        if root.is_dir():
            matches.extend(
                p for p in root.rglob(filename)
                if p.is_file() and directory.lower() in [part.lower() for part in p.parts]
            )
    unique = list(dict.fromkeys(matches))
    if len(unique) == 1:
        return unique[0]
    if len(unique) > 1:
        preferred = [p for p in unique if ROOT in p.parents]
        if len(preferred) == 1:
            return preferred[0]
        raise RuntimeError(
            "Multiple {} checkpoints found: {}. Set {} to the intended checkpoint.".format(
                filename, "; ".join(str(p) for p in unique), env_name
            )
        )
    raise FileNotFoundError(
        "Could not find weights/{}/{}. Attach the checkpoint dataset or set {}.".format(
            directory, filename, env_name
        )
    )


def checkpoint_paths():
    return {
        "weak_expert": find_checkpoint(
            "FRWSG_WEAK_CHECKPOINT", WEAK_DIR, "best_model.pth"
        ),
        "strong_expert": find_checkpoint(
            "FRWSG_STRONG_CHECKPOINT", STRONG_DIR, "best_model.pth"
        ),
        "gate": find_checkpoint(
            "FRWSG_GATE_CHECKPOINT", GATE_DIR, "best_weak_strong_gate.pth"
        ),
    }


def checkpoint_state(checkpoint):
    if isinstance(checkpoint, dict):
        for name in ("model_state_dict", "state_dict"):
            if name in checkpoint and isinstance(checkpoint[name], dict):
                return checkpoint[name]
        return checkpoint
    raise RuntimeError("Unsupported checkpoint format: {}".format(type(checkpoint)))


def load_strict(model, checkpoint_path: Path, device, description: str):
    try:
        checkpoint = __import__("torch").load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
    except TypeError:
        checkpoint = __import__("torch").load(checkpoint_path, map_location="cpu")
    state = checkpoint_state(checkpoint)
    clean = {}
    for key, value in state.items():
        clean[key[7:] if key.startswith("module.") else key] = value
    model.load_state_dict(clean, strict=True)
    print("  strict load OK: {} ({} tensors)".format(description, len(clean)), flush=True)
    del checkpoint, state, clean
    return model.to(device).eval()


def build_weak_base(checkpoint: Path, device):
    from model.inceptionv3_nmfgf_noise_ra_v1_legacy import inceptionv3_nmfgf_noise_ra_v1

    model = inceptionv3_nmfgf_noise_ra_v1(
        num_classes=2,
        pretrained_backbone=False,
        alpha_init=0.02,
        ra_loss_weight=0.005,
        dense_dim=16,
        mlp_hidden=64,
        sigma_train=[5.0, 10.0, 15.0, 25.0],
        sigma_eval=10.0,
        transform_input=True,
    )
    return load_strict(model, checkpoint, device, "legacy weak expert")


def build_strong_base(checkpoint: Path, device):
    from model.inceptionv3_high_noise_bias_calib import inceptionv3_high_noise_bias_calib

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
    )
    return load_strict(model, checkpoint, device, "legacy strong expert")


def build_gate(checkpoint: Path, device):
    from model.frequency_reliability_weak_strong_gate import (
        FEATURE_NAMES, frequency_reliability_weak_strong_gate,
    )

    gate = frequency_reliability_weak_strong_gate(
        num_features=len(FEATURE_NAMES), hidden_dim=32
    )
    return load_strict(gate, checkpoint, device, "weak/strong gate")


class FastWeakInference:
    """Clean-input classifier; keeps checkpoint modules but bypasses noisy/RA branches."""

    def __init__(self, base_model):
        self.base_model = base_model

    def __call__(self, image):
        features = self.base_model._forward_features(image)
        return self.base_model._classify_from_feat(features[-1])

    def parameters(self):
        return self.base_model.parameters()

    def eval(self):
        self.base_model.eval()
        return self

    def to(self, device):
        self.base_model.to(device)
        return self


class FastStrongInference:
    """Clean classification plus required frequency Bias-Calib correction only."""

    def __init__(self, base_model):
        self.base_model = base_model

    def __call__(self, image):
        features = self.base_model._forward_features(image)
        cls_logits = self.base_model._classify_from_feat(features[-1])
        freq_stats = self.base_model._compute_freq_stats(image)
        final_logits, _ = self.base_model._apply_bias_calibration(cls_logits, freq_stats)
        return final_logits

    def parameters(self):
        return self.base_model.parameters()

    def eval(self):
        self.base_model.eval()
        return self

    def to(self, device):
        self.base_model.to(device)
        return self


def optimized_parameter_count(model) -> int:
    """Count parameters actually used by its optimized clean-input forward."""
    unused_ids = {id(parameter) for parameter in model.noise_ra.parameters()}
    if hasattr(model, "alpha"):
        unused_ids.add(id(model.alpha))
    return sum(
        parameter.numel() for parameter in model.parameters()
        if id(parameter) not in unused_ids
    )


class ActiveHardRoute:
    """Profile gate plus one selected expert, separately for both possible paths."""

    def __init__(self, gate, weak, strong, route: str):
        self.gate = gate
        self.weak = weak
        self.strong = strong
        self.route = route

    def __call__(self, image):
        from model.frequency_reliability_weak_strong_gate import compute_frequency_reliability_features

        features = compute_frequency_reliability_features(image)
        gate_logits = self.gate(features)
        _ = gate_logits.argmax(dim=1)
        return self.weak(image) if self.route == "weak" else self.strong(image)

    def parameters(self):
        yield from self.gate.parameters()
        yield from self.weak.parameters()
        yield from self.strong.parameters()

    def eval(self):
        self.gate.eval()
        self.weak.eval()
        self.strong.eval()
        return self


def count_parameters(parameters) -> int:
    return int(sum(parameter.numel() for parameter in parameters))


def new_input(device, batch_size: int, input_size: int):
    import torch
    return torch.randn(batch_size, 3, input_size, input_size, device=device)


def call_model(model, image):
    import torch
    with torch.inference_mode():
        return model(image)


def profile_flops(model, image, device):
    import torch
    from torch.profiler import ProfilerActivity, profile

    try:
        with profile(
            activities=[ProfilerActivity.CPU],
            record_shapes=True,
            with_flops=True,
        ) as profiler:
            call_model(model, image)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        total = float(profiler.key_averages().total_average().flops)
        if total <= 0:
            return "", "", "Profiler did not recognize FLOP-countable operators."
        flops_g = total / 1e9
        return flops_g, flops_g / 2.0, (
            "Profiler estimates recognized operators only; FFT/reduction operations may be omitted."
        )
    except Exception as exc:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return "", "", "FLOPs/MACs profiling failed: {}: {}".format(type(exc).__name__, exc)


def benchmark(model, image, device, warmup_iters: int, bench_iters: int):
    import torch

    for _ in range(warmup_iters):
        call_model(model, image)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for _ in range(bench_iters):
        call_model(model, image)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    ms = elapsed * 1000.0 / max(1, bench_iters * image.shape[0])
    fps = 1000.0 / ms if ms > 0 else 0.0
    peak = (
        torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)
        if device.type == "cuda" else ""
    )
    return ms, fps, peak


def checkpoint_label(paths, key: str) -> str:
    if key == "weak_expert":
        return str(paths["weak_expert"])
    if key == "strong_expert":
        return str(paths["strong_expert"])
    return "gate={}; weak={}; strong={}".format(
        paths["gate"], paths["weak_expert"], paths["strong_expert"]
    )


def run_one(key: str, spec: dict, args, paths: dict) -> dict:
    import torch

    device = torch.device(args.device)
    row = {field: "" for field in FIELDS}
    row.update({
        "key": key,
        "group": spec["group"],
        "model": spec["label"],
        "status": "FAILED",
        "device": str(device),
        "precision": "FP32",
        "input_shape": "{}x{}x{}".format(args.batch_size, args.input_size, args.input_size),
        "checkpoint": checkpoint_label(paths, key),
    })
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    image = new_input(device, args.batch_size, args.input_size)
    row["notes"] = (
        "Optimized inference; synthetic input; excludes preprocessing and host-to-device transfer. "
        "Internal noisy-copy and absolute-difference Projection/GAP/auxiliary-MLP paths are skipped."
    )

    if key == "weak_expert":
        base = build_weak_base(paths["weak_expert"], device)
        model = FastWeakInference(base).eval()
        total_params = count_parameters(model.parameters())
        active_params = optimized_parameter_count(base)
        flops, macs, note = profile_flops(model, image, device)
        ms, fps, peak = benchmark(model, image, device, args.warmup_iters, args.bench_iters)
        row.update({
            "parameters_m": total_params / 1e6,
            "resident_parameters_m": total_params / 1e6,
            "weak_active_parameters_m": active_params / 1e6,
            "flops_g": flops,
            "macs_g": macs,
            "latency_ms_per_image": ms,
            "fps": fps,
            "peak_cuda_allocated_mb": peak,
            "notes": row["notes"] + " Active params exclude the bypassed RA branch and alpha. " + note,
        })

    elif key == "strong_expert":
        base = build_strong_base(paths["strong_expert"], device)
        model = FastStrongInference(base).eval()
        total_params = count_parameters(model.parameters())
        active_params = optimized_parameter_count(base)
        flops, macs, note = profile_flops(model, image, device)
        ms, fps, peak = benchmark(model, image, device, args.warmup_iters, args.bench_iters)
        row.update({
            "parameters_m": total_params / 1e6,
            "resident_parameters_m": total_params / 1e6,
            "strong_active_parameters_m": active_params / 1e6,
            "flops_g": flops,
            "macs_g": macs,
            "latency_ms_per_image": ms,
            "fps": fps,
            "peak_cuda_allocated_mb": peak,
            "notes": row["notes"] + " Bias-Calib is retained; active params exclude the bypassed RA branch and alpha. " + note,
        })

    elif key == "final_gate":
        weak_base = build_weak_base(paths["weak_expert"], device)
        strong_base = build_strong_base(paths["strong_expert"], device)
        weak = FastWeakInference(weak_base).eval()
        strong = FastStrongInference(strong_base).eval()
        gate = build_gate(paths["gate"], device)

        model = ActiveHardRoute(gate, weak, strong, "weak").eval()
        total_params = count_parameters(model.parameters())
        gate_params = count_parameters(gate.parameters())
        weak_active = gate_params + optimized_parameter_count(weak_base)
        strong_active = gate_params + optimized_parameter_count(strong_base)
        wf, wm, wn = profile_flops(model, image, device)
        wms, wfps, wpeak = benchmark(
            model, image, device, args.warmup_iters, args.bench_iters
        )

        model.route = "strong"
        sf, sm, sn = profile_flops(model, image, device)
        sms, sfps, speak = benchmark(
            model, image, device, args.warmup_iters, args.bench_iters
        )
        peaks = [value for value in (wpeak, speak) if isinstance(value, (int, float))]
        row.update({
            "parameters_m": total_params / 1e6,
            "resident_parameters_m": total_params / 1e6,
            "weak_active_parameters_m": weak_active / 1e6,
            "strong_active_parameters_m": strong_active / 1e6,
            "weak_path_flops_g": wf,
            "weak_path_macs_g": wm,
            "strong_path_flops_g": sf,
            "strong_path_macs_g": sm,
            "weak_path_latency_ms": wms,
            "strong_path_latency_ms": sms,
            "weak_path_fps": wfps,
            "strong_path_fps": sfps,
            "peak_cuda_allocated_mb": max(peaks) if peaks else "",
            "notes": (
                row["notes"] + " FR-WSG reports gate+weak and gate+strong active paths separately; "
                "all three networks remain resident. Route-specific path is forced for profiling. "
                + wn + " " + sn
            ),
        })
    else:
        raise KeyError("Unknown model key: {}".format(key))

    row["status"] = "OK"
    del image
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return row


def failure_row(key: str, spec: dict, error: str, args, paths: dict) -> dict:
    row = {field: "" for field in FIELDS}
    row.update({
        "key": key,
        "group": spec["group"],
        "model": spec["label"],
        "status": "FAILED",
        "device": args.device,
        "precision": "FP32",
        "input_shape": "{}x{}x{}".format(args.batch_size, args.input_size, args.input_size),
        "checkpoint": checkpoint_label(paths, key),
        "error": error,
    })
    return row


def save_results(rows, out_dir: Path, args):
    csv_path = out_dir / "efficiency_frwsg_optimized.csv"
    txt_path = out_dir / "efficiency_frwsg_optimized.txt"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    with txt_path.open("w", encoding="utf-8") as handle:
        handle.write("OPTIMIZED FR-WSG COMPUTATIONAL EFFICIENCY PROFILE\n")
        handle.write("=" * 100 + "\n")
        handle.write("Device: {} | input: {}x{} | batch: {}\n".format(
            args.device, args.input_size, args.input_size, args.batch_size
        ))
        handle.write("Warmup: {} | timed iterations: {}\n\n".format(
            args.warmup_iters, args.bench_iters
        ))
        handle.write("FLOPs are profiler estimates for recognized operators; MACs = FLOPs / 2.\n")
        handle.write("FFT/reduction operations may be omitted from FLOPs/MACs.\n")
        handle.write("Latency excludes preprocessing and host-to-device transfer.\n")
        handle.write("Optimized inference skips the internal noisy copy and unused RA auxiliary path.\n")
        handle.write("FR-WSG reports gate+weak and gate+strong paths separately.\n\n")
        handle.write("{:<35} {:<8} {:>12} {:>12} {:>12} {:>12}\n".format(
            "Model", "Status", "FLOPs(G)", "MACs(G)", "ms/image", "FPS"
        ))
        handle.write("-" * 100 + "\n")
        for row in rows:
            if row["key"] == "final_gate":
                handle.write(
                    "{:<35} {:<8} weak={:>7}G/{:>7}ms; strong={:>7}G/{:>7}ms\n".format(
                        row["model"][:35], row["status"],
                        row["weak_path_flops_g"] or "NA", row["weak_path_latency_ms"] or "NA",
                        row["strong_path_flops_g"] or "NA", row["strong_path_latency_ms"] or "NA",
                    )
                )
            else:
                handle.write("{:<35} {:<8} {:>12} {:>12} {:>12} {:>12}\n".format(
                    row["model"][:35], row["status"], row["flops_g"] or "NA",
                    row["macs_g"] or "NA", row["latency_ms_per_image"] or "NA",
                    row["fps"] or "NA",
                ))
            if row.get("error"):
                handle.write("  Error: {}\n".format(row["error"]))
    print("Saved CSV: {}".format(csv_path), flush=True)
    print("Saved report: {}".format(txt_path), flush=True)


def run_child(args, spec, out_dir, paths):
    result_path = out_dir / ("child_{}.json".format(spec["key"]))
    print("\nProfiling {} ({})".format(spec["label"], spec["key"]), flush=True)
    try:
        row = run_one(spec["key"], spec, args, paths)
    except Exception as exc:
        error = "{}: {}".format(type(exc).__name__, exc)
        row = failure_row(spec["key"], spec, error, args, paths)
        print("FAILED: {}".format(error), flush=True)
        traceback.print_exc()
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    result_path.write_text(json.dumps(row, ensure_ascii=False, default=str), encoding="utf-8")
    return 0


def config_for(args, paths):
    return {
        "protocol": "optimized FR-WSG inference efficiency; no dataset, no training",
        "device": args.device,
        "input_size": args.input_size,
        "batch_size": args.batch_size,
        "warmup_iters": args.warmup_iters,
        "bench_iters": args.bench_iters,
        "weak_checkpoint": str(paths["weak_expert"]),
        "strong_checkpoint": str(paths["strong_expert"]),
        "gate_checkpoint": str(paths["gate"]),
    }


def parent_main(args, paths):
    if args.device == "cuda" and not __import__("torch").cuda.is_available():
        raise RuntimeError("CUDA requested, but CUDA is unavailable.")

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = DEFAULT_OUTPUT_ROOT / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = out_dir / "run_config.json"
    config = config_for(args, paths)
    if config_path.is_file():
        prior = json.loads(config_path.read_text(encoding="utf-8"))
        if prior != config:
            raise RuntimeError("Existing output directory uses a different benchmark config; choose a new directory.")
    else:
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    print("Optimized FR-WSG efficiency profile; no training or accuracy evaluation.", flush=True)
    print("Device={}, input={}x{}, batch={}, warmup={}, iterations={}".format(
        args.device, args.input_size, args.input_size, args.batch_size,
        args.warmup_iters, args.bench_iters,
    ), flush=True)
    print("Results folder: {}".format(out_dir), flush=True)
    print("Weak checkpoint: {}".format(paths["weak_expert"]), flush=True)
    print("Strong checkpoint: {}".format(paths["strong_expert"]), flush=True)
    print("Gate checkpoint: {}".format(paths["gate"]), flush=True)
    print("Skipped: internal noisy-copy passes and absolute-difference Projection/GAP/RA-MLP.", flush=True)
    print("Retained: clean InceptionV3 path, strong Bias-Calib, frequency gate, hard route.", flush=True)

    rows_by_key = {}
    for spec in MODEL_SPECS:
        key = spec["key"]
        result_path = out_dir / ("child_{}.json".format(key))
        if result_path.is_file():
            try:
                saved = json.loads(result_path.read_text(encoding="utf-8"))
                if saved.get("key") == key:
                    rows_by_key[key] = saved
            except Exception:
                pass

    script_path = Path(__file__).resolve()
    for index, spec in enumerate(MODEL_SPECS, 1):
        key = spec["key"]
        print("\n[{}/{}] {}".format(index, len(MODEL_SPECS), spec["label"]), flush=True)
        if rows_by_key.get(key, {}).get("status") == "OK":
            print("[Resume] Reusing successful saved result; skipping {}.".format(key), flush=True)
        else:
            child_args = [
                sys.executable, str(script_path), "--child", "--model-key", key,
                "--device", args.device,
                "--input-size", str(args.input_size),
                "--batch-size", str(args.batch_size),
                "--warmup-iters", str(args.warmup_iters),
                "--bench-iters", str(args.bench_iters),
                "--output-dir", str(out_dir),
            ]
            env = os.environ.copy()
            env["AIGC_PROJECT_ROOT"] = str(ROOT)
            completed = subprocess.run(child_args, cwd=str(ROOT), env=env)
            result_path = out_dir / ("child_{}.json".format(key))
            if result_path.is_file():
                try:
                    rows_by_key[key] = json.loads(result_path.read_text(encoding="utf-8"))
                except Exception as exc:
                    rows_by_key[key] = failure_row(
                        key, spec, "Could not read child result: {}".format(exc), args, paths
                    )
            else:
                rows_by_key[key] = failure_row(
                    key, spec,
                    "Child exited with code {} without saving a result.".format(completed.returncode),
                    args, paths,
                )

        current = [rows_by_key[s["key"]] for s in MODEL_SPECS if s["key"] in rows_by_key]
        save_results(current, out_dir, args)

    final_rows = [
        rows_by_key.get(spec["key"]) or failure_row(
            spec["key"], spec, "No child result was saved.", args, paths
        )
        for spec in MODEL_SPECS
    ]
    save_results(final_rows, out_dir, args)
    print("\nCompleted: {} / {} stages have usable measurements.".format(
        sum(row["status"] == "OK" for row in final_rows), len(final_rows)
    ), flush=True)
    try:
        archive = shutil.make_archive(str(out_dir), "zip", root_dir=str(out_dir))
        print("Results ZIP: {}".format(archive), flush=True)
    except Exception as exc:
        print("Could not package result ZIP: {}".format(exc), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Profile optimized FR-WSG efficiency on Kaggle")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--model-key", choices=[item["key"] for item in MODEL_SPECS], default="")
    parser.add_argument("--device", choices=("cuda", "cpu"), default=DEVICE)
    parser.add_argument("--input-size", type=int, default=INPUT_SIZE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--warmup-iters", type=int, default=WARMUP_ITERS)
    parser.add_argument("--bench-iters", type=int, default=BENCH_ITERS)
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.input_size <= 0 or args.batch_size <= 0 or args.warmup_iters < 0 or args.bench_iters <= 0:
        raise SystemExit("input/batch/iterations must be positive and warmup nonnegative")
    paths = checkpoint_paths()
    if args.child:
        if not args.model_key:
            raise SystemExit("--model-key is required in child mode")
        spec = next(item for item in MODEL_SPECS if item["key"] == args.model_key)
        child_dir = Path(args.output_dir)
        child_dir.mkdir(parents=True, exist_ok=True)
        raise SystemExit(run_child(args, spec, child_dir, paths))
    parent_main(args, paths)
