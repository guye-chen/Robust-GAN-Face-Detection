"""
Ablation: InceptionV3 High-Noise Bias-Calib without the internal high-noise gate.

This file intentionally keeps the High-Noise Bias Calibration Head, internal
high-noise noisy copy, and stability regularization unchanged. It only disables
the dynamic high-noise gate inside the bias calibration formula:

    Full:
        correction = gamma * high_noise_gate * raw_bias

    This ablation:
        correction = gamma * 1.0 * raw_bias

Purpose:
    Verify whether noise-severity-aware gating is necessary, or whether a
    fixed always-on bias calibration head is enough.

It does NOT overwrite the full model weights.
"""

import train_inceptionv3_high_noise_bias_calib_server as base


base.CHECKPOINT_DIR = (
    r"E:\cbam_ftt_fusion\CBAM.PyTorch-master\weights"
    r"\inceptionv3-high-noise-bias-calib-no-high-noise-gate"
)

# Click-run ablation defaults: match the full high-noise expert except for gate.
base.CLICK_RUN_STABILITY_LOSS_WEIGHT = 0.005
base.CLICK_RUN_BIAS_CALIB_LOSS_WEIGHT = 0.05
base.CLICK_RUN_SIGMA_TRAIN_LIST = [25.0, 35.0, 50.0]
base.CLICK_RUN_SIGMA_EVAL = 50.0
base.CLICK_RUN_GAMMA_INIT = 0.02
base.CLICK_RUN_GAMMA_MAX = 0.20
base.CLICK_RUN_GATE_TAU = 0.55
base.CLICK_RUN_GATE_K = 10.0
base.CLICK_RUN_USE_HIGH_NOISE_GATE = False


if __name__ == "__main__":
    print("=" * 70)
    print("ABLATION: High-Noise Bias-Calib w/o Internal High-Noise Gate")
    print("use_high_noise_gate is forced to False")
    print("Bias calibration remains enabled and always-on: correction = gamma * raw_bias")
    print("Checkpoint dir: {}".format(base.CHECKPOINT_DIR))
    print("=" * 70)
    base.main()
