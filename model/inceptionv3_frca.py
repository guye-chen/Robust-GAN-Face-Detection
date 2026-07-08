"""
InceptionV3-FRCA: Frequency Reliability Calibration Adapter.

FRCA calibrates intermediate/high-level InceptionV3 features. It does not
produce real/fake logits directly and does not fuse classification logits.
When gamma=0, the output path is identical to the original InceptionV3 clean
baseline after loading the same checkpoint.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


__all__ = [
    "inceptionv3_frca",
    "safe_load_inceptionv3_baseline",
    "freeze_inception_backbone_and_fc",
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


class FRCAdapter(nn.Module):
    def __init__(self, channels, bottleneck=64):
        super(FRCAdapter, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, bottleneck, kernel_size=1, bias=False),
            nn.BatchNorm2d(bottleneck),
            nn.GELU(),
            nn.Conv2d(bottleneck, channels, kernel_size=1, bias=True),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


class FrequencyReliabilityGate(nn.Module):
    def __init__(self, hidden_dim=16):
        super(FrequencyReliabilityGate, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, freq_stats):
        return torch.sigmoid(self.mlp(freq_stats))


class InceptionV3_FRCA(nn.Module):
    def __init__(self, num_classes=2, pretrained_backbone=False,
                 adapter_channels=2048, adapter_bottleneck=64,
                 gate_hidden_dim=16, gamma_init=0.01,
                 gamma_max=0.05):
        super(InceptionV3_FRCA, self).__init__()

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

        self.adapter = FRCAdapter(adapter_channels, adapter_bottleneck)
        self.freq_gate = FrequencyReliabilityGate(gate_hidden_dim)
        self.gamma_max = float(gamma_max)
        self.raw_gamma = nn.Parameter(self._gamma_to_raw(float(gamma_init)))
        self.force_gamma = None
        self.freeze_clean_path = False

        self.register_buffer("_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def set_force_gamma(self, value):
        self.force_gamma = None if value is None else float(value)

    def get_gamma(self):
        if self.force_gamma is not None:
            return torch.tensor(self.force_gamma, device=self.raw_gamma.device)
        return self.gamma_max * torch.sigmoid(self.raw_gamma)

    def _gamma_to_raw(self, gamma):
        gamma = max(1e-6, min(float(gamma), self.gamma_max - 1e-6))
        ratio = torch.tensor(gamma / self.gamma_max)
        return torch.log(ratio / (1.0 - ratio))

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
        super(InceptionV3_FRCA, self).train(mode)
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
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)
        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)
        return x

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
        feat = self._forward_features(x)
        freq_stats = self._compute_freq_stats(x)
        gate = self.freq_gate(freq_stats).view(-1, 1, 1, 1)
        gamma = self.get_gamma()
        adapted = self.adapter(feat)
        feat_new = feat + gamma * gate * adapted
        feat_delta = feat_new - feat
        pooled = self.avgpool(feat_new)
        pooled = torch.flatten(pooled, 1)
        pooled = self.dropout(pooled)
        logits = self.fc(pooled)

        if return_aux:
            aux = {
                "gamma": gamma.detach(),
                "gate": gate.detach().view(-1),
                "freq_stats": freq_stats.detach(),
                "adapter_output_abs_mean": adapted.detach().abs().mean(),
                "feat_delta_abs_max": feat_delta.detach().abs().max(),
                "feat_delta_abs_mean": feat_delta.detach().abs().mean(),
            }
            return logits, aux
        return logits


def _normalize_key(key):
    name = key[7:] if key.startswith("module.") else key
    for prefix in ("model.", "backbone.", "inception."):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name


def safe_load_inceptionv3_baseline(model, checkpoint_path, map_location="cpu"):
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
    for key, value in state_dict.items():
        name = _normalize_key(key)
        if name in model_dict and model_dict[name].shape == value.shape:
            filtered[name] = value
        else:
            unexpected.append(name)
    model.load_state_dict(filtered, strict=False)
    missing = [key for key in model_dict if key not in filtered]
    return len(filtered), missing, unexpected


def freeze_inception_backbone_and_fc(model):
    model.freeze_clean_path = True
    for name, param in model.named_parameters():
        param.requires_grad = (
            name.startswith("adapter")
            or name.startswith("freq_gate")
            or name == "raw_gamma"
        )
    for module in model._clean_path_modules():
        module.eval()


def inceptionv3_frca(num_classes=2, pretrained_backbone=False,
                     adapter_bottleneck=64, gate_hidden_dim=16,
                     gamma_init=0.01, gamma_max=0.05):
    return InceptionV3_FRCA(
        num_classes=num_classes,
        pretrained_backbone=pretrained_backbone,
        adapter_bottleneck=adapter_bottleneck,
        gate_hidden_dim=gate_hidden_dim,
        gamma_init=gamma_init,
        gamma_max=gamma_max,
    )
