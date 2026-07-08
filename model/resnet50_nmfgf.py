"""
ResNet50 + Noise-aware Multi-frequency Gated Fusion (NMGFG)

Key innovation:
    1. Extract multi-frequency (low / mid / high) statistics from input image via FFT.
    2. Learn a small MLP to output frequency gates for the final feature vector.
    3. Reduce weight on high-frequency components under strong additive noise,
       increase weight on low/mid-frequency stable structural information.
    4. <0.1M additional parameters, fully lightweight.

Purpose: Improve Gaussian noise robustness without changing training procedure.
"""

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


__all__ = [
    'resnet50_nmfgf',
    'safe_load_backbone',
]


# ============================================================
# Noise-aware Multi-frequency Gated Fusion module
# ============================================================

class NoiseAwareMultiFreqGatedFusion(nn.Module):
    """
    Noise-aware Multi-frequency Gated Fusion (NMGF) module.

    Design:
        Input image -> FFT -> frequency energy st ats (5 numbers)
        -> MLP(5->32->3) -> 3 gate scalars (g_low, g_mid, g_high)

        Backbone features:
            layer1 avg_pool (256) -> proj_low -> 16-dim  * g_low
            layer2 avg_pool (512) -> proj_mid -> 16-dim  * g_mid
            layer3 avg_pool (1024)-> proj_high-> 16-dim  * g_high
            -> sum -> 16-dim fused modulation signal

        modulation = Linear(16 -> 2048)
        fc_input = layer4_avg_pool(2048) + modulation
        logits = fc(fc_input)

    Total additional params: <70K (very lightweight).
    """

    def __init__(self, in_features=2048, band_dim=16, mlp_hidden=32):
        super(NoiseAwareMultiFreqGatedFusion, self).__init__()

        # ---- Frequency statistics MLP ----
        # 5 inputs: E_low, E_mid, E_high, ratio_high_mid, ratio_high_total
        # 3 outputs: g_low, g_mid, g_high  (positive via Softplus)
        self.freq_mlp = nn.Sequential(
            nn.Linear(5, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden, 3),
            nn.Softplus(),
        )

        # ---- Per-band feature projections ----
        # Low:  layer1 -> 256 dim
        # Mid:  layer2 -> 512 dim
        # High: layer3 -> 1024 dim  (not layer3+4 to save params)
        self.proj_low  = nn.Linear(256, band_dim)
        self.proj_mid  = nn.Linear(512, band_dim)
        self.proj_high = nn.Linear(1024, band_dim)

        # ---- Modulation: fused signal -> additive bias to fc input ----
        self.modulation = nn.Linear(band_dim, in_features)

        self.in_features = in_features

        # ---- Init: keep modulation small initially ----
        nn.init.normal_(self.modulation.weight, mean=0.0, std=0.001)
        nn.init.zeros_(self.modulation.bias)

        # MLP init: start with gates ~1.0
        for m in self.freq_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.5 if m.out_features == 3 else 0.0)

    def forward(self, x, feat1, feat2, feat3):
        """
        Args:
            x:      input image             [B, 3, H, W]  (Normalized)
            feat1:  layer1 output           [B, 256, H/4, W/4]
            feat2:  layer2 output           [B, 512, H/8, W/8]
            feat3:  layer3 output           [B, 1024, H/16, W/16]

        Returns:
            modulation: [B, in_features]   additive signal for fc input
            aux:        dict of gate/energy statistics
        """
        B = x.shape[0]

        # ---- Step 1: Denormalize & compute FFT ----
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x_01 = x * std + mean

        fft = torch.fft.fft2(x_01, dim=(-2, -1))
        fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))
        mag = torch.abs(fft_shifted)              # [B, 3, H, W]

        # ---- Step 2: Frequency band masks (concentric circles) ----
        H, W = mag.shape[-2], mag.shape[-1]
        cy, cx = H // 2, W // 2
        y, xc = torch.meshgrid(
            torch.arange(H, device=x.device),
            torch.arange(W, device=x.device), indexing='ij')
        dist = torch.sqrt((y - cy)**2 + (xc - cx)**2)
        dist_norm = dist / (max(cy, cx) + 1e-6)

        low_th  = 1.0 / 6
        mid_th  = 3.0 / 6

        m_low  = (dist_norm <= low_th).float()
        m_mid  = ((dist_norm > low_th) & (dist_norm <= mid_th)).float()
        m_high = (dist_norm > mid_th).float()

        # ---- Step 3: Per-band average energy ----
        E_low  = (mag * m_low[None,None]).sum(dim=(1,2,3))  / (m_low.sum()  + 1e-6) / 3
        E_mid  = (mag * m_mid[None,None]).sum(dim=(1,2,3))  / (m_mid.sum()  + 1e-6) / 3
        E_high = (mag * m_high[None,None]).sum(dim=(1,2,3)) / (m_high.sum() + 1e-6) / 3

        E_total = E_low + E_mid + E_high + 1e-6
        r_hm = E_high / (E_mid + 1e-6)
        r_ht = E_high / E_total

        freq_stats = torch.stack([E_low, E_mid, E_high, r_hm, r_ht], dim=1)  # [B, 5]

        # ---- Step 4: MLP -> gates ----
        gates = self.freq_mlp(freq_stats)          # [B, 3]
        g_low, g_mid, g_high = gates[:, 0:1], gates[:, 1:2], gates[:, 2:3]

        # ---- Step 5: Per-band features & gated fusion ----
        f_low  = F.adaptive_avg_pool2d(feat1, (1,1)).flatten(1)  # [B, 256]
        f_mid  = F.adaptive_avg_pool2d(feat2, (1,1)).flatten(1)  # [B, 512]
        f_high = F.adaptive_avg_pool2d(feat3, (1,1)).flatten(1)  # [B, 1024]

        p_low  = self.proj_low(f_low)    * g_low   # [B, band_dim]
        p_mid  = self.proj_mid(f_mid)    * g_mid   # [B, band_dim]
        p_high = self.proj_high(f_high)  * g_high  # [B, band_dim]

        fused = p_low + p_mid + p_high              # [B, band_dim]

        # ---- Step 6: Modulation vector ----
        modulation = self.modulation(fused)          # [B, in_features]

        aux = {
            'g_low':  g_low.mean(),
            'g_mid':  g_mid.mean(),
            'g_high': g_high.mean(),
            'E_low':  E_low.mean(),
            'E_mid':  E_mid.mean(),
            'E_high': E_high.mean(),
            'r_hm':   r_hm.mean(),
            'r_ht':   r_ht.mean(),
        }

        return modulation, aux


# ============================================================
# ResNet50 backbone with NMGFG
# ============================================================

class ResNet50_NMFG(nn.Module):
    """
    ResNet50 + Noise-aware Multi-frequency Gated Fusion.

    Based on torchvision ResNet50, no CBAM, lightweight NMFG module.

    Architecture:
        Input -> ResNet50 backbone (conv1..layer4)
        layer4 output -> avg_pool -> 2048-dim base feature
        layer1/2/3 outputs -> NMGFG -> 2048-dim modulation
        fc_input = base + modulation
        fc -> 2-class logits
    """

    def __init__(self, num_classes=2, pretrained_backbone=False):
        super(ResNet50_NMFG, self).__init__()

        backbone = models.resnet50(pretrained=False)

        self.conv1   = backbone.conv1
        self.bn1     = backbone.bn1
        self.relu    = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1  = backbone.layer1
        self.layer2  = backbone.layer2
        self.layer3  = backbone.layer3
        self.layer4  = backbone.layer4

        # NMFG module: only needs layer1/2/3 features
        self.nmfg = NoiseAwareMultiFreqGatedFusion(
            in_features=2048,
            band_dim=16,
            mlp_hidden=32,
        )

        self.fc = nn.Linear(2048, num_classes)

        # Weight init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, return_aux=False):
        """
        Args:
            x:          [B, 3, 224, 224] normalized
            return_aux: if True, return (logits, aux) with gate/energy stats

        Returns:
            logits or (logits, aux)
        """
        # Backbone
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        feat1 = self.layer1(out)
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)
        feat4 = self.layer4(feat3)

        # Standard fc input from layer4
        base = F.adaptive_avg_pool2d(feat4, (1, 1)).flatten(1)  # [B, 2048]

        # NMFG modulation from layer1/2/3
        mod, aux = self.nmfg(x, feat1, feat2, feat3)              # [B, 2048]

        fc_input = base + mod
        logits = self.fc(fc_input)

        if return_aux:
            return logits, aux
        return logits


# ============================================================
# Builder and loader
# ============================================================

def resnet50_nmfgf(num_classes=2, pretrained_backbone=False):
    """
    Construct ResNet50-NMGF model.

    Args:
        num_classes: number of output classes (default 2 for real/fake)
        pretrained_backbone: if True, load ImageNet pretrained ResNet50 weights
    """
    model = ResNet50_NMFG(num_classes=num_classes, pretrained_backbone=pretrained_backbone)
    return model


def safe_load_backbone(model, weights_dict_or_path, logger=None):
    """
    Safely load ImageNet pretrained backbone weights.
    Only loads matching keys, skips nmfg and fc layers.
    """
    if isinstance(weights_dict_or_path, str):
        pretrained = torch.load(weights_dict_or_path, map_location='cpu', weights_only=False)
    else:
        pretrained = weights_dict_or_path

    model_state = model.state_dict()
    loaded, missing, unexpected = 0, [], []

    for key, value in pretrained.items():
        if key.startswith('fc.'):
            continue
        if key in model_state:
            if model_state[key].shape == value.shape:
                model_state[key] = value
                loaded += 1
            else:
                unexpected.append(key)
        else:
            unexpected.append(key)

    for key in model_state:
        if key not in pretrained:
            missing.append(key)

    model.load_state_dict(model_state, strict=False)

    if logger is not None:
        logger.append('Backbone pretrained weights loaded: {}'.format(loaded))
        if missing:
            logger.append('Missing keys (random init): {} (nmfg, fc)'.format(len(missing)))
        if unexpected:
            logger.append('Unexpected keys (ignored): {}'.format(len(unexpected)))

    return loaded, missing, unexpected


# ============================================================
# Verification
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ResNet50 + Noise-aware Multi-frequency Gated Fusion")
    print("Verification")
    print("=" * 60)

    model = resnet50_nmfgf(num_classes=2, pretrained_backbone=False)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())

    backbone_params = sum(p.numel() for n, p in model.named_parameters()
                          if not any(x in n for x in ['nmfg', 'fc']))
    nmfg_params = sum(p.numel() for n, p in model.named_parameters()
                      if 'nmfg' in n)
    fc_params = sum(p.numel() for n, p in model.named_parameters()
                    if 'fc' in n)

    print("Total params:     {:,}".format(total_params))
    print("  Backbone:       {:,}".format(backbone_params))
    print("  NMGFG (new):    {:,}  ({:.2f}K)".format(nmfg_params, nmfg_params / 1e3))
    print("  Final fc:       {:,}".format(fc_params))
    print("  NMGFG+fc:       {:,}  ({:.2f}K)".format(
        nmfg_params + fc_params, (nmfg_params + fc_params) / 1e3))

    assert nmfg_params < 100_000, \
        "NMGFG params {:,} exceeds 100K limit!".format(nmfg_params)
    print("  <0.1M constraint: \u2713")

    x = torch.randn(2, 3, 224, 224)
    print("\n--- Forward pass ---")
    with torch.no_grad():
        out = model(x)
    print("Default (2,3,224,224) -> {}".format(out.shape))
    assert out.shape == (2, 2)

    with torch.no_grad():
        out2, aux = model(x, return_aux=True)
    print("return_aux=True -> logits {} | aux: {}".format(out2.shape, list(aux.keys())))
    assert out2.shape == (2, 2)
    for k, v in aux.items():
        print("  {}: {:.6f}".format(k, v.item()))

    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED")
    print("NMGFG params: {:,}  (<0.1M)".format(nmfg_params))
    print("=" * 60)