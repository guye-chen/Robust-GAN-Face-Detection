"""
InceptionV3 High-Noise Frequency Bias Calibration.

This model is a high-noise expert built on the existing InceptionV3 noisy-copy
feature path. It does not use RA logits for final classification. Instead, a
small calibration head estimates severe-noise frequency bias and subtracts a
bounded correction from the fake logit:

    final_fake_logit = cls_fake_logit - gamma * high_noise_gate * bias

The goal is to reduce the common noise50 failure mode where random
high-frequency Gaussian noise pushes real images toward the fake class.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.inceptionv3_nmfgf_noise_ra_v1_legacy import (
    InceptionV3_NMFGF_NoiseRA_V1,
    safe_load_inceptionv3_baseline_v1,
)


__all__ = [
    "InceptionV3HighNoiseBiasCalib",
    "inceptionv3_high_noise_bias_calib",
    "safe_load_inceptionv3_baseline_v1",
    "high_noise_bias_calibration_loss",
]


class HighNoiseBiasCalibrationHead(nn.Module):
    """Predicts a non-negative fake-logit correction from frequency statistics."""

    def __init__(self, hidden_dim=32):
        super(HighNoiseBiasCalibrationHead, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(7, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)
        # Start as an identity correction: final logits are initially cls logits.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, freq_stats):
        # v1 freq_stats: [E_low, E_mid, E_high, E_high/E_mid, E_high/E_total]
        e_low = freq_stats[:, 0:1]
        e_mid = freq_stats[:, 1:2]
        e_high = freq_stats[:, 2:3]
        high_mid = freq_stats[:, 3:4]
        high_total = freq_stats[:, 4:5]
        mid_low = e_mid / (e_low + 1e-6)
        image_energy = torch.log1p(e_low + e_mid + e_high)

        features = torch.cat([
            torch.log1p(e_low),
            torch.log1p(e_mid),
            torch.log1p(e_high),
            torch.log1p(high_mid),
            high_total,
            torch.log1p(mid_low),
            image_energy,
        ], dim=1)
        return F.softplus(self.mlp(features))


class InceptionV3HighNoiseBiasCalib(InceptionV3_NMFGF_NoiseRA_V1):
    def __init__(self, num_classes=2, pretrained_backbone=False,
                 alpha_init=0.02, ra_loss_weight=0.005,
                 dense_dim=16, mlp_hidden=64,
                 sigma_train=None, sigma_eval=50.0,
                 transform_input=True,
                 gamma_init=0.02, gamma_max=0.20,
                 gate_tau=0.55, gate_k=10.0,
                 calib_hidden_dim=32,
                 use_high_noise_gate=True):
        super(InceptionV3HighNoiseBiasCalib, self).__init__(
            num_classes=num_classes,
            pretrained_backbone=pretrained_backbone,
            alpha_init=alpha_init,
            ra_loss_weight=ra_loss_weight,
            dense_dim=dense_dim,
            mlp_hidden=mlp_hidden,
            sigma_train=sigma_train,
            sigma_eval=sigma_eval,
            transform_input=transform_input,
        )
        self.calib_head = HighNoiseBiasCalibrationHead(hidden_dim=calib_hidden_dim)
        self.gamma_max = float(gamma_max)
        self.raw_gamma = nn.Parameter(self._gamma_to_raw(float(gamma_init)))
        self.gate_tau = float(gate_tau)
        self.gate_k = float(gate_k)
        self.use_high_noise_gate = bool(use_high_noise_gate)

    def _gamma_to_raw(self, gamma):
        gamma = max(1e-6, min(float(gamma), self.gamma_max - 1e-6))
        ratio = torch.tensor(gamma / self.gamma_max)
        return torch.log(ratio / (1.0 - ratio))

    def get_gamma(self):
        return self.gamma_max * torch.sigmoid(self.raw_gamma)

    def _apply_bias_calibration(self, logits, freq_stats):
        high_mid = freq_stats[:, 3:4]
        if self.use_high_noise_gate:
            high_noise_gate = torch.sigmoid(self.gate_k * (high_mid - self.gate_tau))
        else:
            high_noise_gate = torch.ones_like(high_mid)
        raw_bias = self.calib_head(freq_stats)
        gamma = self.get_gamma()
        correction = gamma * high_noise_gate * raw_bias

        final_logits = logits.clone()
        final_logits[:, 1:2] = final_logits[:, 1:2] - correction
        return final_logits, {
            "raw_bias": raw_bias,
            "correction": correction,
            "high_noise_gate": high_noise_gate,
            "gamma": gamma,
        }

    def forward(self, x, return_aux=False):
        sigma = self._select_sigma()

        clean_feats = self._forward_features(x)
        cls_logits = self._classify_from_feat(clean_feats[-1])
        input_freq_stats = self._compute_freq_stats(x)
        final_logits, calib_aux = self._apply_bias_calibration(cls_logits, input_freq_stats)

        x_noise = self._noise_perturb(x, sigma)
        noise_feats = self._forward_features(x_noise)
        noise_freq_stats = self._compute_freq_stats(x_noise)
        noise_cls_logits = self._classify_from_feat(noise_feats[-1])
        noise_final_logits, noise_calib_aux = self._apply_bias_calibration(
            noise_cls_logits, noise_freq_stats
        )

        stability_logits, aux = self.noise_ra(clean_feats, noise_feats, noise_freq_stats)

        aux["alpha"] = self.get_alpha().detach()
        aux["sigma"] = torch.tensor(sigma, device=x.device)
        aux["freq_stats"] = input_freq_stats.detach()
        aux["noise_freq_stats"] = noise_freq_stats.detach()
        aux["cls_logits"] = cls_logits
        aux["final_logits"] = final_logits
        aux["noise_cls_logits"] = noise_cls_logits
        aux["noise_final_logits"] = noise_final_logits
        aux["gamma"] = calib_aux["gamma"]
        aux["bias_raw_mean"] = calib_aux["raw_bias"].mean()
        aux["bias_correction_mean"] = calib_aux["correction"].mean()
        aux["high_noise_gate_mean"] = calib_aux["high_noise_gate"].mean()
        aux["noise_bias_correction_mean"] = noise_calib_aux["correction"].mean()
        aux["noise_high_noise_gate_mean"] = noise_calib_aux["high_noise_gate"].mean()

        if return_aux:
            return final_logits, aux
        return final_logits, cls_logits, stability_logits, aux


def high_noise_bias_calibration_loss(aux_dict, labels, real_margin=0.0, fake_margin=0.25):
    """Calibrate fake-logit bias without noisy-copy CE classification.

    For real images, the internally generated high-noise probe should not raise
    the final fake logit above the clean fake logit. For fake images, correction
    should not suppress fake evidence too aggressively.
    """
    cls_logits = aux_dict["cls_logits"]
    noise_final_logits = aux_dict["noise_final_logits"]
    labels = labels.long()

    mask_real = labels == 0
    mask_fake = labels == 1
    loss = torch.tensor(0.0, device=labels.device)

    if mask_real.sum() > 0:
        real_noise_fake = noise_final_logits[mask_real, 1]
        real_clean_fake = cls_logits[mask_real, 1].detach()
        loss = loss + F.relu(real_noise_fake - real_clean_fake + real_margin).mean()

    if mask_fake.sum() > 0:
        fake_noise_fake = noise_final_logits[mask_fake, 1]
        fake_clean_fake = cls_logits[mask_fake, 1].detach()
        loss = loss + F.relu(fake_clean_fake - fake_noise_fake - fake_margin).mean()

    return loss


def inceptionv3_high_noise_bias_calib(num_classes=2, pretrained_backbone=False,
                                      alpha_init=0.02, ra_loss_weight=0.005,
                                      dense_dim=16, mlp_hidden=64,
                                      sigma_train=None, sigma_eval=50.0,
                                      transform_input=True,
                                      gamma_init=0.02, gamma_max=0.20,
                                      gate_tau=0.55, gate_k=10.0,
                                      calib_hidden_dim=32,
                                      use_high_noise_gate=True):
    return InceptionV3HighNoiseBiasCalib(
        num_classes=num_classes,
        pretrained_backbone=pretrained_backbone,
        alpha_init=alpha_init,
        ra_loss_weight=ra_loss_weight,
        dense_dim=dense_dim,
        mlp_hidden=mlp_hidden,
        sigma_train=sigma_train,
        sigma_eval=sigma_eval,
        transform_input=transform_input,
        gamma_init=gamma_init,
        gamma_max=gamma_max,
        gate_tau=gate_tau,
        gate_k=gate_k,
        calib_hidden_dim=calib_hidden_dim,
        use_high_noise_gate=use_high_noise_gate,
    )
