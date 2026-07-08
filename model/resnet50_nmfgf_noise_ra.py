"""
ResNet50 + Noise-aware MFGF + Real-Stability RA (NMGFG-Noise-RA)

Key innovations over the original NMFG model:
  1. Internal noise perturbation within forward() — NOT dataset-side transform.
  2. Clean and noisy features extracted from the same forward pass.
  3. Lightweight Noise Robustness Asymmetry branch (layer1-4 diff + cosine sim).
  4. Multi-frequency FFT stats feed into the RA branch as auxiliary signal.
  5. Residual fusion: final_logits = cls_logits + alpha * stability_logits.
  6. Optional noise_real_stability_loss to keep real images stable under noise.

Architecture:
    Input x (Normalized tensor)
      |
    ├─→ ResNet50 backbone ──→ feat1-4 & cls_logits (original path)
    │
    ├─→ Internal noise perturb ──→ x_noise (sigma ∈ {5,10,15,25})
    │       └─→ ResNet50 backbone ──→ feat1-4_noise
    │
    ├─→ NoiseRA branch:
    │       per-layer abs(feat_x - feat_noise) → 1x1 proj → dense_dim
    │       per-layer cosine_sim(GAP(feat_x), GAP(feat_noise))
    │       + FFT frequency stats (E_low, E_mid, E_high, r_hm, r_ht)
    │       → MLP → noise_stability_logits
    │
    └─→ final_logits = cls_logits + alpha * noise_stability_logits

Lightweight: ~130K-180K additional parameters over vanilla ResNet50.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


__all__ = [
    'resnet50_nmfgf_noise_ra',
    'safe_load_backbone',
    'noise_real_stability_loss',
]


# ============================================================
# Utility: safe backbone weight loading
# ============================================================

def safe_load_backbone(model, pretrained_path=None):
    """Load ImageNet-pretrained backbone weights for the model."""
    if pretrained_path is None or pretrained_path == '':
        loaded = 0
        missing_keys = list(model.state_dict().keys())
        unexpected_keys = []
    else:
        state_dict = torch.load(pretrained_path, map_location='cpu',
                                weights_only=True)
        model_dict = model.state_dict()
        filtered_dict = {}
        unexpected_keys = []
        for k, v in state_dict.items():
            if k in model_dict:
                if v.shape == model_dict[k].shape:
                    filtered_dict[k] = v
                else:
                    unexpected_keys.append(k)
            else:
                unexpected_keys.append(k)
        missing_keys = [k for k in model_dict if k not in filtered_dict]
        model.load_state_dict(filtered_dict, strict=False)
        loaded = len(filtered_dict)
    return loaded, missing_keys, unexpected_keys


# ============================================================
# Noise Robustness Asymmetry branch
# ============================================================

class NoiseRobustnessAsymmetry(nn.Module):
    """
    Noise RA branch: learns feature stability under internal noise perturbation.

    For each layer (1-4):
      - abs(feat_clean - feat_noisy) → 1x1 Conv proj → dense_dim
      - cosine_sim(GAP(clean), GAP(noisy))

    All compressed diffs + cosine sims + FFT frequency stats → MLP → 2-class logits.

    Total parameters: ~120K.
    """

    def __init__(self,
                 layer1_channels=256, layer2_channels=512,
                 layer3_channels=1024, layer4_channels=2048,
                 dense_dim=16, mlp_hidden=64):
        super(NoiseRobustnessAsymmetry, self).__init__()

        self.dense_dim = dense_dim
        self.gap = nn.AdaptiveAvgPool2d(1)

        # ---- Per-layer 1x1 projection for absolute difference ----
        self.layer1_proj = nn.Sequential(
            nn.Conv2d(layer1_channels, dense_dim, 1, bias=False),
            nn.BatchNorm2d(dense_dim),
            nn.ReLU(inplace=True),
        )
        self.layer2_proj = nn.Sequential(
            nn.Conv2d(layer2_channels, dense_dim, 1, bias=False),
            nn.BatchNorm2d(dense_dim),
            nn.ReLU(inplace=True),
        )
        self.layer3_proj = nn.Sequential(
            nn.Conv2d(layer3_channels, dense_dim, 1, bias=False),
            nn.BatchNorm2d(dense_dim),
            nn.ReLU(inplace=True),
        )
        self.layer4_proj = nn.Sequential(
            nn.Conv2d(layer4_channels, dense_dim, 1, bias=False),
            nn.BatchNorm2d(dense_dim),
            nn.ReLU(inplace=True),
        )

        # ---- MLP input: 4 * dense_dim (diffs) + 4 (cosine sims) + 5 (freq stats) ----
        mlp_input_dim = 4 * dense_dim + 4 + 5
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(mlp_hidden, 2),
        )

    def forward(self, feat1_x, feat2_x, feat3_x, feat4_x,
                feat1_noise, feat2_noise, feat3_noise, feat4_noise,
                freq_stats):
        """
        Args:
            feat*_x:     clean features from layer1-4  [B, C, H, W]
            feat*_noise: noisy features from layer1-4  [B, C, H, W]
            freq_stats:  [B, 5]  (E_low, E_mid, E_high, r_hm, r_ht)

        Returns:
            stability_logits: [B, 2]
            aux: dict with per-layer stats
        """
        layers_x     = [feat1_x, feat2_x, feat3_x, feat4_x]
        layers_noise = [feat1_noise, feat2_noise, feat3_noise, feat4_noise]
        projs = [self.layer1_proj, self.layer2_proj,
                 self.layer3_proj, self.layer4_proj]

        diff_features = []
        sim_features = []
        D_means = {}
        cos_means = {}

        for idx, (f_x, f_noise, proj) in enumerate(zip(
                layers_x, layers_noise, projs)):
            # Absolute difference → project
            D = torch.abs(f_x - f_noise)
            d_proj = self.gap(proj(D)).flatten(1)
            diff_features.append(d_proj)

            # Cosine similarity
            g_x = self.gap(f_x).flatten(1)
            g_noise = self.gap(f_noise).flatten(1)
            s = F.cosine_similarity(g_x, g_noise, dim=1).unsqueeze(1)
            sim_features.append(s)

            # Track for diagnostics
            D_means['D{}_mean'.format(idx + 1)] = D.mean()
            cos_means['cos{}_mean'.format(idx + 1)] = s.mean()

        # ---- Concatenate all features ----
        feat = torch.cat(diff_features + sim_features, dim=1)      # [B, 4*d+4]
        feat = torch.cat([feat, freq_stats], dim=1)                 # [B, 4*d+4+5]
        stability_logits = self.mlp(feat)

        # ---- dense_diff_score for real-stability loss ----
        sim_stack = torch.cat(sim_features, dim=1)   # [B, 4]
        dense_diff_score = sim_stack.mean(dim=1)      # [B]

        aux = {
            'dense_diff_score': dense_diff_score,
        }
        aux.update(D_means)
        aux.update(cos_means)

        return stability_logits, aux


# ============================================================
# Main model: ResNet50 + internal noise RA
# ============================================================

class ResNet50_NMFG_NoiseRA(nn.Module):
    """
    ResNet50 + internal noise perturbation + Noise RA branch.

    Training:
      - sigma randomly chosen from {5, 10, 15, 25} per batch.
      - cls_logits preserved from clean path.
      - noise_stability_logits from NoiseRA branch as residual.

    Eval:
      - fixed sigma (default 10) for stable evaluation.

    Output:
      - Always returns (final_logits, cls_logits, stability_logits, aux_dict).
      - The caller should apply CE(cls_logits) optionally plus
        noise_real_stability_loss.
    """

    def __init__(self, num_classes=2, dense_dim=16, mlp_hidden=64,
                 alpha_init=0.05, sigma_train=None, sigma_eval=10.0):
        super(ResNet50_NMFG_NoiseRA, self).__init__()

        # ---- Backbone: torchvision ResNet50 ----
        backbone = models.resnet50(pretrained=False)

        self.conv1   = backbone.conv1
        self.bn1     = backbone.bn1
        self.relu    = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1  = backbone.layer1   # 256 channels
        self.layer2  = backbone.layer2   # 512
        self.layer3  = backbone.layer3   # 1024
        self.layer4  = backbone.layer4   # 2048
        self.avgpool = backbone.avgpool  # AdaptiveAvgPool2d(1)

        # ---- Original classification head (clean path) ----
        backbone_fc_in = backbone.fc.in_features  # 2048
        self.fc = nn.Linear(backbone_fc_in, num_classes)

        # ---- Noise Robustness Asymmetry branch ----
        self.noise_ra = NoiseRobustnessAsymmetry(
            layer1_channels=256,
            layer2_channels=512,
            layer3_channels=1024,
            layer4_channels=2048,
            dense_dim=dense_dim,
            mlp_hidden=mlp_hidden,
        )

        # ---- Learnable alpha for residual fusion ----
        self.alpha = nn.Parameter(torch.tensor(alpha_init))

        # ---- Training sigma set ----
        if sigma_train is None:
            self.sigma_train_list = [5.0, 10.0, 15.0, 25.0]
        else:
            self.sigma_train_list = list(sigma_train)
        self.sigma_eval = sigma_eval

        # ---- Normalization buffers ----
        self.register_buffer('_mean', torch.tensor(
            [0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('_std', torch.tensor(
            [0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # ---- Weight init for new modules ----
        for m in self.noise_ra.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for m in self.noise_ra.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _forward_backbone(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        feat1 = self.layer1(x)
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)
        feat4 = self.layer4(feat3)

        return feat1, feat2, feat3, feat4

    def _noise_perturb(self, x, sigma):
        """Add Gaussian noise to denormalized image in tensor space (no PIL)."""
        with torch.no_grad():
            x_denorm = x * self._std + self._mean
            x_denorm = x_denorm.clamp(0, 1)
            noise = torch.randn_like(x_denorm) * (sigma / 255.0)
            x_noise = (x_denorm + noise).clamp(0, 1)
            x_noise = (x_noise - self._mean) / self._std
        return x_noise

    def _compute_freq_stats(self, x):
        """Compute FFT frequency band statistics from input tensor.

        Returns [B, 5]: E_low, E_mid, E_high, r_hm, r_ht
        """
        with torch.no_grad():
            # x is already normalized; denormalize for FFT
            x_01 = x * self._std + self._mean
            x_01 = x_01.clamp(0, 1)

            fft = torch.fft.fft2(x_01, dim=(-2, -1))
            fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))
            mag = torch.abs(fft_shifted)  # [B, 3, H, W]

            H, W = mag.shape[-2], mag.shape[-1]
            cy, cx = H // 2, W // 2
            yc, xc = torch.meshgrid(
                torch.arange(H, device=x.device, dtype=torch.float32),
                torch.arange(W, device=x.device, dtype=torch.float32),
                indexing='ij')
            dist = torch.sqrt((yc - cy)**2 + (xc - cx)**2)
            dist_norm = dist / (max(cy, cx) + 1e-6)

            low_th = 1.0 / 6
            mid_th = 3.0 / 6

            m_low  = (dist_norm <= low_th).float()
            m_mid  = ((dist_norm > low_th) & (dist_norm <= mid_th)).float()
            m_high = (dist_norm > mid_th).float()

            E_low  = (mag * m_low[None, None]).sum(dim=(1, 2, 3)) / (m_low.sum() + 1e-6) / 3
            E_mid  = (mag * m_mid[None, None]).sum(dim=(1, 2, 3)) / (m_mid.sum() + 1e-6) / 3
            E_high = (mag * m_high[None, None]).sum(dim=(1, 2, 3)) / (m_high.sum() + 1e-6) / 3

            E_total = E_low + E_mid + E_high + 1e-6
            r_hm = E_high / (E_mid + 1e-6)
            r_ht = E_high / E_total

        return torch.stack([E_low, E_mid, E_high, r_hm, r_ht], dim=1)  # [B, 5]

    def forward(self, x, return_aux=False):
        """
        Args:
            x:          [B, 3, 224, 224]  Normalized
            return_aux: if True, return aux dict with diagnostic info

        Returns:
            If return_aux=False:
                (final_logits, cls_logits, stability_logits, aux_dict)
            If return_aux=True:
                final_logits, aux_dict
              (for compatibility with training loop expecting (logits, aux))
        """
        B = x.shape[0]
        is_training = self.training

        # ---- Select sigma ----
        if is_training:
            idx = torch.randint(0, len(self.sigma_train_list), (1,)).item()
            sigma = self.sigma_train_list[idx]
        else:
            sigma = self.sigma_eval

        # ---- Clean backbone features ----
        feat1, feat2, feat3, feat4 = self._forward_backbone(x)

        # ---- Clean classification logits ----
        base_feat = self.avgpool(feat4).flatten(1)  # [B, 2048]
        cls_logits = self.fc(base_feat)               # [B, num_classes]

        # ---- Internal noise perturbation ----
        x_noise = self._noise_perturb(x, sigma)

        # ---- Noisy backbone features ----
        feat1_noise, feat2_noise, feat3_noise, feat4_noise = \
            self._forward_backbone(x_noise)

        # ---- Frequency statistics from the noisy input ----
        freq_stats = self._compute_freq_stats(x_noise)  # [B, 5]

        # ---- Noise RA branch ----
        stability_logits, aux = self.noise_ra(
            feat1, feat2, feat3, feat4,
            feat1_noise, feat2_noise, feat3_noise, feat4_noise,
            freq_stats,
        )

        # ---- Residual fusion ----
        final_logits = cls_logits + self.alpha * stability_logits

        # ---- Diagnostic info ----
        aux['alpha'] = self.alpha.detach()
        aux['sigma'] = torch.tensor(sigma, device=x.device)
        aux['freq_stats'] = freq_stats.detach()

        if return_aux:
            return final_logits, aux
        else:
            return final_logits, cls_logits, stability_logits, aux


# ============================================================
# Loss: noise_real_stability_loss
# ============================================================

def noise_real_stability_loss(aux_dict, labels, margin=0.1):
    """
    Encourage real images to maintain high feature similarity under noise.

    L = ReLU(s_fake - s_real + margin) + (1 - s_real).mean()

    where s = dense_diff_score (mean cosine similarity across layer1-4).

    Args:
        aux_dict: from model forward (must contain 'dense_diff_score')
        labels:   [B]  (0=real, 1=fake)
        margin:   hinge margin

    Returns:
        scalar loss
    """
    s = aux_dict['dense_diff_score']  # [B]

    mask_real = (labels == 0)
    mask_fake = (labels == 1)

    if mask_real.sum() == 0 or mask_fake.sum() == 0:
        return torch.tensor(0.0, device=s.device)

    s_real = s[mask_real].mean()
    s_fake = s[mask_fake].mean()

    hinge = F.relu(s_fake - s_real + margin)
    real_pull = (1.0 - s[mask_real]).mean()

    return hinge + real_pull


# ============================================================
# Factory function
# ============================================================

def resnet50_nmfgf_noise_ra(num_classes=2, pretrained_backbone=False,
                             dense_dim=16, mlp_hidden=64,
                             alpha_init=0.05, sigma_train=None,
                             sigma_eval=10.0):
    """Build ResNet50 + Noise-aware MFGF + Noise RA model."""
    model = ResNet50_NMFG_NoiseRA(
        num_classes=num_classes,
        dense_dim=dense_dim,
        mlp_hidden=mlp_hidden,
        alpha_init=alpha_init,
        sigma_train=sigma_train,
        sigma_eval=sigma_eval,
    )
    return model