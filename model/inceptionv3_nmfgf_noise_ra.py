"""
InceptionV3 + Noise-aware Multi-frequency / Stability RA.

The clean InceptionV3 path remains the main classifier. A lightweight branch
compares clean and internally noised features, then adds a small residual:

    final_logits = cls_logits + alpha * noise_stability_logits

Noise is generated inside forward() in tensor space after denormalization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


__all__ = [
    "inceptionv3_nmfgf_noise_ra",
    "safe_load_inceptionv3_baseline",
    "freeze_clean_classifier_path",
    "noise_real_stability_loss",
]


def _build_inception_backbone(pretrained_backbone=False):
    try:
        from torchvision.models import Inception_V3_Weights
        weights = Inception_V3_Weights.IMAGENET1K_V1 if pretrained_backbone else None
        return models.inception_v3(weights=weights, aux_logits=True, transform_input=False)
    except (ImportError, AttributeError):
        return models.inception_v3(
            pretrained=pretrained_backbone,
            aux_logits=True,
            transform_input=False,
        )


class NoiseRobustnessAsymmetry(nn.Module):
    """Small branch using feature differences, cosine stability, and FFT ratios.

    Frequency statistics estimate noise severity and band reliability. They are
    normalized ratios, not raw high-frequency energy as a direct fake cue.
    """

    def __init__(self, dense_dim=16, mlp_hidden=64, num_classes=2):
        super(NoiseRobustnessAsymmetry, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.proj_5d = self._make_proj(288, dense_dim)
        self.proj_6e = self._make_proj(768, dense_dim)
        self.proj_7c = self._make_proj(2048, dense_dim)

        mlp_input_dim = 3 * dense_dim + 3 + 5
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(mlp_hidden, num_classes),
        )

    @staticmethod
    def _make_proj(in_channels, dense_dim):
        return nn.Sequential(
            nn.Conv2d(in_channels, dense_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dense_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, clean_feats, noise_feats, freq_stats):
        projs = [self.proj_5d, self.proj_6e, self.proj_7c]
        names = ["5d", "6e", "7c"]

        diff_features = []
        sim_features = []
        aux = {}

        for name, clean, noisy, proj in zip(names, clean_feats, noise_feats, projs):
            diff = torch.abs(clean - noisy)
            diff_features.append(self.gap(proj(diff)).flatten(1))

            clean_vec = self.gap(clean).flatten(1)
            noisy_vec = self.gap(noisy).flatten(1)
            sim = F.cosine_similarity(clean_vec, noisy_vec, dim=1).unsqueeze(1)
            sim_features.append(sim)

            aux["D_{}_mean".format(name)] = diff.mean()
            aux["cos_{}_mean".format(name)] = sim.mean()

        branch_feat = torch.cat(diff_features + sim_features + [freq_stats], dim=1)
        stability_logits = self.mlp(branch_feat)

        sim_stack = torch.cat(sim_features, dim=1)
        aux["dense_diff_score"] = sim_stack.mean(dim=1)

        return stability_logits, aux


class InceptionV3_NMFGF_NoiseRA(nn.Module):
    def __init__(self, num_classes=2, pretrained_backbone=False,
                 alpha_init=0.02, ra_loss_weight=0.01,
                 dense_dim=16, mlp_hidden=64,
                 sigma_train=None, sigma_eval=10.0,
                 alpha_max=0.05, fixed_alpha=None,
                 use_severity_gate=False, gate_k=5.0, gate_tau=1.0):
        super(InceptionV3_NMFGF_NoiseRA, self).__init__()

        backbone = _build_inception_backbone(pretrained_backbone)
        backbone.aux_logits = False
        backbone.AuxLogits = None
        self.transform_input = True

        self.Conv2d_1a_3x3 = backbone.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = backbone.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = backbone.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = backbone.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = backbone.Conv2d_4a_3x3
        self.Mixed_5b = backbone.Mixed_5b
        self.Mixed_5c = backbone.Mixed_5c
        self.Mixed_5d = backbone.Mixed_5d
        self.Mixed_6a = backbone.Mixed_6a
        self.Mixed_6b = backbone.Mixed_6b
        self.Mixed_6c = backbone.Mixed_6c
        self.Mixed_6d = backbone.Mixed_6d
        self.Mixed_6e = backbone.Mixed_6e
        self.Mixed_7a = backbone.Mixed_7a
        self.Mixed_7b = backbone.Mixed_7b
        self.Mixed_7c = backbone.Mixed_7c
        self.avgpool = backbone.avgpool
        self.dropout = backbone.dropout
        self.fc = nn.Linear(backbone.fc.in_features, num_classes)

        self.noise_ra = NoiseRobustnessAsymmetry(
            dense_dim=dense_dim,
            mlp_hidden=mlp_hidden,
            num_classes=num_classes,
        )
        self.alpha_max = float(alpha_max)
        self.fixed_alpha = fixed_alpha
        self.force_alpha = None
        self.freeze_clean_path = False
        self.use_severity_gate = bool(use_severity_gate)
        self.gate_k = float(gate_k)
        self.gate_tau = float(gate_tau)
        self.raw_alpha = nn.Parameter(self._alpha_to_raw(float(alpha_init)))
        self.ra_loss_weight = ra_loss_weight

        self.sigma_train_list = list(sigma_train) if sigma_train is not None else [5.0, 10.0, 15.0, 25.0]
        self.sigma_eval = float(sigma_eval)

        self.register_buffer("_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self._init_new_modules()

    def _init_new_modules(self):
        nn.init.normal_(self.fc.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.fc.bias)
        for m in self.noise_ra.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        final_linear = self.noise_ra.mlp[-1]
        nn.init.zeros_(final_linear.weight)
        nn.init.zeros_(final_linear.bias)

    def _alpha_to_raw(self, alpha):
        alpha = max(1e-6, min(float(alpha), self.alpha_max - 1e-6))
        ratio = torch.tensor(alpha / self.alpha_max)
        return torch.log(ratio / (1.0 - ratio))

    def set_force_alpha(self, value):
        self.force_alpha = None if value is None else float(value)

    def get_alpha(self):
        if self.force_alpha is not None:
            return torch.tensor(self.force_alpha, device=self.raw_alpha.device)
        if self.fixed_alpha is not None:
            return torch.tensor(float(self.fixed_alpha), device=self.raw_alpha.device)
        return self.alpha_max * torch.sigmoid(self.raw_alpha)

    def _clean_path_modules(self):
        return [
            self.Conv2d_1a_3x3, self.Conv2d_2a_3x3, self.Conv2d_2b_3x3,
            self.Conv2d_3b_1x1, self.Conv2d_4a_3x3,
            self.Mixed_5b, self.Mixed_5c, self.Mixed_5d,
            self.Mixed_6a, self.Mixed_6b, self.Mixed_6c, self.Mixed_6d, self.Mixed_6e,
            self.Mixed_7a, self.Mixed_7b, self.Mixed_7c,
            self.avgpool, self.dropout, self.fc,
        ]

    def train(self, mode=True):
        super(InceptionV3_NMFGF_NoiseRA, self).train(mode)
        if mode and self.freeze_clean_path:
            for module in self._clean_path_modules():
                module.eval()
        return self

    def _transform_input(self, x):
        x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
        x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
        x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        return torch.cat((x_ch0, x_ch1, x_ch2), 1)

    def _forward_features(self, x):
        if self.transform_input:
            x = self._transform_input(x)
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        feat_5d = self.Mixed_5d(x)
        x = self.Mixed_6a(feat_5d)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        feat_6e = self.Mixed_6e(x)
        x = self.Mixed_7a(feat_6e)
        x = self.Mixed_7b(x)
        feat_7c = self.Mixed_7c(x)
        return feat_5d, feat_6e, feat_7c

    def _classify_from_feat(self, feat_7c):
        x = self.avgpool(feat_7c)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)

    def _select_sigma(self):
        if self.training:
            idx = torch.randint(0, len(self.sigma_train_list), (1,)).item()
            return float(self.sigma_train_list[idx])
        return self.sigma_eval

    def _noise_perturb(self, x, sigma):
        with torch.no_grad():
            x_denorm = (x * self._std + self._mean).clamp(0, 1)
            noise = torch.randn_like(x_denorm) * (sigma / 255.0)
            x_noise = (x_denorm + noise).clamp(0, 1)
            x_noise = (x_noise - self._mean) / self._std
        return x_noise

    def _compute_freq_stats(self, x):
        with torch.no_grad():
            x_01 = (x * self._std + self._mean).clamp(0, 1)
            x_01 = F.interpolate(x_01, size=(64, 64), mode="bilinear", align_corners=False)
            fft = torch.fft.fft2(x_01, dim=(-2, -1))
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

            e_low = (mag * low[None, None]).sum(dim=(1, 2, 3)) / (low.sum() + 1e-6) / 3
            e_mid = (mag * mid[None, None]).sum(dim=(1, 2, 3)) / (mid.sum() + 1e-6) / 3
            e_high = (mag * high[None, None]).sum(dim=(1, 2, 3)) / (high.sum() + 1e-6) / 3
            e_total = e_low + e_mid + e_high + 1e-6
            low_ratio = e_low / e_total
            mid_ratio = e_mid / e_total
            high_ratio = e_high / e_total
            high_mid_ratio = e_high / (e_mid + 1e-6)
            mid_low_ratio = e_mid / (e_low + 1e-6)
        return torch.stack(
            [low_ratio, mid_ratio, high_ratio, high_mid_ratio, mid_low_ratio],
            dim=1,
        )

    def forward(self, x, return_aux=False):
        sigma = self._select_sigma()

        clean_feats = self._forward_features(x)
        cls_logits = self._classify_from_feat(clean_feats[-1])

        x_noise = self._noise_perturb(x, sigma)
        noise_feats = self._forward_features(x_noise)
        freq_stats = self._compute_freq_stats(x_noise)

        stability_logits, aux = self.noise_ra(clean_feats, noise_feats, freq_stats)
        alpha = self.get_alpha()
        if self.use_severity_gate:
            noise_score = freq_stats[:, 3]
            beta = torch.sigmoid(self.gate_k * (self.gate_tau - noise_score)).unsqueeze(1)
        else:
            beta = torch.ones((x.shape[0], 1), device=x.device, dtype=x.dtype)
        final_logits = cls_logits + alpha * beta * stability_logits

        aux["alpha"] = alpha.detach()
        aux["beta"] = beta.detach()
        aux["sigma"] = torch.tensor(sigma, device=x.device)
        aux["freq_stats"] = freq_stats.detach()

        if return_aux:
            return final_logits, aux
        return final_logits, cls_logits, stability_logits, aux


def noise_real_stability_loss(aux_dict, labels, margin=0.1):
    score = aux_dict["dense_diff_score"]
    mask_real = labels == 0
    mask_fake = labels == 1
    if mask_real.sum() == 0 or mask_fake.sum() == 0:
        return torch.tensor(0.0, device=score.device)
    score_real = score[mask_real].mean()
    score_fake = score[mask_fake].mean()
    hinge = F.relu(score_fake - score_real + margin)
    real_pull = (1.0 - score[mask_real]).mean()
    return hinge + real_pull


def inceptionv3_nmfgf_noise_ra(num_classes=2, pretrained_backbone=False,
                               alpha_init=0.02, ra_loss_weight=0.01,
                               dense_dim=16, mlp_hidden=64,
                               sigma_train=None, sigma_eval=10.0,
                               alpha_max=0.05, fixed_alpha=None,
                               use_severity_gate=False, gate_k=5.0,
                               gate_tau=1.0):
    return InceptionV3_NMFGF_NoiseRA(
        num_classes=num_classes,
        pretrained_backbone=pretrained_backbone,
        alpha_init=alpha_init,
        ra_loss_weight=ra_loss_weight,
        dense_dim=dense_dim,
        mlp_hidden=mlp_hidden,
        sigma_train=sigma_train,
        sigma_eval=sigma_eval,
        alpha_max=alpha_max,
        fixed_alpha=fixed_alpha,
        use_severity_gate=use_severity_gate,
        gate_k=gate_k,
        gate_tau=gate_tau,
    )


def safe_load_inceptionv3_baseline(model, checkpoint_path, map_location="cpu"):
    """Load baseline InceptionV3 backbone/fc weights into the clean path.

    Noise-RA, raw_alpha, gate, and other new module parameters are intentionally
    left randomly initialized. Missing new-branch keys are expected.
    """
    if not checkpoint_path:
        return 0, list(model.state_dict().keys()), []

    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise RuntimeError("Unsupported checkpoint format: {}".format(type(checkpoint)))

    model_dict = model.state_dict()
    filtered = {}
    unexpected = []

    def normalize_key(key):
        name = key[7:] if key.startswith("module.") else key
        for prefix in ("model.", "backbone.", "inception."):
            if name.startswith(prefix):
                name = name[len(prefix):]
        return name

    for key, value in state_dict.items():
        name = normalize_key(key)
        if name in model_dict and model_dict[name].shape == value.shape:
            filtered[name] = value
        else:
            unexpected.append(name)

    model.load_state_dict(filtered, strict=False)
    missing = [key for key in model_dict.keys() if key not in filtered]
    return len(filtered), missing, unexpected


def freeze_clean_classifier_path(model):
    """Freeze the InceptionV3 clean backbone and fc classifier.

    Leaves Noise-RA modules and raw_alpha trainable for B_freeze_cls ablation.
    """
    trainable_prefixes = ("noise_ra", "raw_alpha")
    model.freeze_clean_path = True
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith(trainable_prefixes)
    for module in model._clean_path_modules():
        module.eval()
