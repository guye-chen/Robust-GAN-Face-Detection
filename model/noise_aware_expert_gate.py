import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    "NoiseAwareExpertGate",
    "compute_noise_gate_features",
    "noise_aware_expert_gate",
]


class NoiseAwareExpertGate(nn.Module):
    """Lightweight degradation classifier for expert selection.

    The gate predicts degradation severity from compact frequency/statistical
    features. It does not see real/fake labels and does not update any expert.
    Output classes: 0=clean, 1=noise10, 2=noise25, 3=noise50.
    """

    def __init__(self, num_features=7, hidden_dim=32, num_classes=4):
        super(NoiseAwareExpertGate, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, num_classes),
        )

    def forward(self, features):
        return self.net(features)


def _denormalize_image(x):
    mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (x * std + mean).clamp(0, 1)


def compute_noise_gate_features(x, fft_size=64):
    """Compute frequency features for the noise-aware gate.

    Features:
      E_low_ratio, E_mid_ratio, E_high_ratio, E_high_over_E_mid,
      E_mid_over_E_low, image_std, high_freq_energy.
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
    high_freq_energy = e_high

    return torch.stack([
        e_low_ratio,
        e_mid_ratio,
        e_high_ratio,
        e_high_over_mid,
        e_mid_over_low,
        image_std,
        high_freq_energy,
    ], dim=1)


def noise_aware_expert_gate(num_features=7, hidden_dim=32, num_classes=4):
    return NoiseAwareExpertGate(
        num_features=num_features,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
    )
