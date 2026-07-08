"""
Frequency-Reliability Weak/Strong Noise Gate.

The gate routes test images to two frozen experts:
  0 = weak noise  (clean / noise10) -> Stability-Reg expert
  1 = strong noise (noise25 / noise50) -> High-Noise Bias-Calib expert

Unlike a generic image classifier, the gate only consumes compact frequency
reliability statistics. This keeps routing lightweight and interpretable.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    "FrequencyReliabilityWeakStrongGate",
    "compute_frequency_reliability_features",
    "frequency_reliability_weak_strong_gate",
    "FEATURE_NAMES",
    "CLASS_NAMES",
]


FEATURE_NAMES = [
    "E_low_ratio",
    "E_mid_ratio",
    "E_high_ratio",
    "E_high_over_E_mid",
    "E_mid_over_E_low",
    "image_std",
    "log_high_freq_energy",
    "high_frequency_unreliability",
    "mid_frequency_reliability",
    "low_mid_reliability",
]

CLASS_NAMES = ["weak_noise", "strong_noise"]


class FrequencyReliabilityWeakStrongGate(nn.Module):
    def __init__(self, num_features=10, hidden_dim=32, dropout=0.1):
        super(FrequencyReliabilityWeakStrongGate, self).__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(num_features),
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 2),
        )

    def forward(self, features):
        return self.net(features)


def _denormalize_image(x):
    mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (x * std + mean).clamp(0, 1)


def compute_frequency_reliability_features(x, fft_size=64):
    """Return interpretable weak/strong-noise routing features.

    The last three features are explicit reliability cues:
      high_frequency_unreliability = log(1 + E_high / E_mid)
      mid_frequency_reliability    = E_mid / (E_mid + E_high)
      low_mid_reliability          = (E_low + E_mid) / total
    """
    x_01 = _denormalize_image(x)
    image_std = x_01.flatten(1).std(dim=1)

    x_small = F.interpolate(x_01, size=(fft_size, fft_size), mode="bilinear", align_corners=False)
    fft = torch.fft.fft2(x_small, dim=(-2, -1))
    mag = torch.abs(torch.fft.fftshift(fft, dim=(-2, -1)))

    h, w = mag.shape[-2], mag.shape[-1]
    cy, cx = h // 2, w // 2
    yy, xx = torch.meshgrid(
        torch.arange(h, device=x.device, dtype=torch.float32),
        torch.arange(w, device=x.device, dtype=torch.float32),
        indexing="ij",
    )
    dist = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    dist_norm = dist / (max(cy, cx) + 1e-6)

    low = (dist_norm <= (1.0 / 6)).float()
    mid = ((dist_norm > (1.0 / 6)) & (dist_norm <= (3.0 / 6))).float()
    high = (dist_norm > (3.0 / 6)).float()

    e_low = (mag * low[None, None]).sum(dim=(1, 2, 3)) / (low.sum() + 1e-6) / 3.0
    e_mid = (mag * mid[None, None]).sum(dim=(1, 2, 3)) / (mid.sum() + 1e-6) / 3.0
    e_high = (mag * high[None, None]).sum(dim=(1, 2, 3)) / (high.sum() + 1e-6) / 3.0
    e_total = e_low + e_mid + e_high + 1e-6

    e_low_ratio = e_low / e_total
    e_mid_ratio = e_mid / e_total
    e_high_ratio = e_high / e_total
    e_high_over_mid = e_high / (e_mid + 1e-6)
    e_mid_over_low = e_mid / (e_low + 1e-6)
    log_high_freq_energy = torch.log1p(e_high)

    high_frequency_unreliability = torch.log1p(e_high_over_mid)
    mid_frequency_reliability = e_mid / (e_mid + e_high + 1e-6)
    low_mid_reliability = (e_low + e_mid) / e_total

    return torch.stack([
        e_low_ratio,
        e_mid_ratio,
        e_high_ratio,
        e_high_over_mid,
        e_mid_over_low,
        image_std,
        log_high_freq_energy,
        high_frequency_unreliability,
        mid_frequency_reliability,
        low_mid_reliability,
    ], dim=1)


def frequency_reliability_weak_strong_gate(num_features=10, hidden_dim=32, dropout=0.1):
    return FrequencyReliabilityWeakStrongGate(
        num_features=num_features,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )
