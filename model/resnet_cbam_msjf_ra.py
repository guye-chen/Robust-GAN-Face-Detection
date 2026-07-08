"""
ResNet50 + CBAM + MSJF-RA
Multi-Scale JPEG Feature Stability with Robustness Asymmetry

Extends resnet_cbam_mjfs.py with a lightweight RA (Robustness Asymmetry) branch
that models feature stability under random degradation (JPEG / Gaussian Blur / Downsample).

The RA branch computes:
  - Absolute feature differences between original and degraded inputs
  - Cosine similarity of GAP-pooled features
  - A lightweight MLP that produces stability_logits

Final output: final_logits = cls_logits + alpha * stability_logits

Author: MSJF-RA Extension
"""

import io
import math
import random
import numpy as np
from PIL import Image, ImageFilter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo


__all__ = [
    'resnet50_cbam_msjf_ra',
    'safe_load_backbone',
    'robustness_asymmetry_loss',
]


model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}


# ============================================================
# CBAM building blocks (reused from resnet_cbam.py)
# ============================================================

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // 16, in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

        self.ca = ChannelAttention(planes * 4)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# ============================================================
# MJFS module (reused from resnet_cbam_mjfs.py)
# ============================================================

class MJFS(nn.Module):
    """Multi-Scale JPEG Feature Stability module."""

    def __init__(self, layer2_channels=512, layer3_channels=1024, layer4_channels=2048,
                 hidden_dim=128, use_layer2=True):
        super(MJFS, self).__init__()

        self.use_layer2 = use_layer2

        if self.use_layer2:
            self.layer2_proj = nn.Sequential(
                nn.Conv2d(layer2_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
            )

        self.layer3_proj = nn.Sequential(
            nn.Conv2d(layer3_channels, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.layer4_proj = nn.Sequential(
            nn.Conv2d(layer4_channels, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, feat2_x, feat3_x, feat4_x, feat2_jpeg, feat3_jpeg, feat4_jpeg):
        d3 = torch.abs(feat3_x - feat3_jpeg)
        d4 = torch.abs(feat4_x - feat4_jpeg)

        d3 = self.layer3_proj(d3)
        d4 = self.layer4_proj(d4)

        d_l3 = self.gap(d3).flatten(1)
        d_l4 = self.gap(d4).flatten(1)

        if self.use_layer2:
            d2 = torch.abs(feat2_x - feat2_jpeg)
            d2 = self.layer2_proj(d2)
            d_l2 = self.gap(d2).flatten(1)
            return d_l2, d_l3, d_l4
        else:
            return d_l3, d_l4


# ============================================================
# RA (Robustness Asymmetry) branch
# ============================================================

class RobustnessAsymmetry(nn.Module):
    """
    Lightweight degradation stability branch.

    Extracts layer3 and layer4 features from both the original and a randomly
    degraded version of the input, then computes:
      - Absolute difference (L1) after projection
      - Cosine similarity of GAP-pooled features
      - MLP → stability_logits

    Total parameters: ~200K (well under 1M budget).
    """

    def __init__(self, layer3_channels=1024, layer4_channels=2048, hidden_dim=64):
        super(RobustnessAsymmetry, self).__init__()

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.layer3_proj = nn.Sequential(
            nn.Conv2d(layer3_channels, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.layer4_proj = nn.Sequential(
            nn.Conv2d(layer4_channels, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Input: d3(hidden_dim) + d4(hidden_dim) + s3(1) + s4(1)
        mlp_input_dim = hidden_dim * 2 + 2
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, f3_x, f4_x, f3_deg, f4_deg):
        # ---- absolute differences ----
        D3 = torch.abs(f3_x - f3_deg)
        D4 = torch.abs(f4_x - f4_deg)

        d3 = self.gap(self.layer3_proj(D3)).flatten(1)
        d4 = self.gap(self.layer4_proj(D4)).flatten(1)

        # ---- cosine similarity ----
        g3_x = self.gap(f3_x).flatten(1)
        g3_deg = self.gap(f3_deg).flatten(1)
        g4_x = self.gap(f4_x).flatten(1)
        g4_deg = self.gap(f4_deg).flatten(1)

        s3 = F.cosine_similarity(g3_x, g3_deg, dim=1).unsqueeze(1)
        s4 = F.cosine_similarity(g4_x, g4_deg, dim=1).unsqueeze(1)

        # ---- MLP prediction ----
        feat = torch.cat([d3, d4, s3, s4], dim=1)
        stability_logits = self.mlp(feat)

        # ---- auxiliary outputs for interpretability ----
        aux = {
            's3': s3.squeeze(1),
            's4': s4.squeeze(1),
            'D3_mean': D3.mean(),
            'D4_mean': D4.mean(),
        }

        return stability_logits, aux


# ============================================================
# Main model: ResNet50 + CBAM + MSJF + RA
# ============================================================

class ResNet_CBAM_MSJF_RA(nn.Module):
    """
    ResNet50 + CBAM + MSJF-RA

    Architecture:
        Input x
          |
        ResNet50 backbone + CBAM
          ├── feat2, feat3, feat4 (from x)
          ├── feat2_jpeg, feat3_jpeg, feat4_jpeg (from JPEG-perturbed x)
          └── feat3_deg, feat4_deg (from randomly degraded x)
          |
        MSJF branch: multi-scale JPEG feature differences → cls_logits
        RA branch:   degradation stability analysis → stability_logits
          |
        final_logits = cls_logits + alpha * stability_logits
    """

    def __init__(self, block, layers, num_classes=2,
                 hidden_dim=128, jpeg_quality=90, use_layer2=True,
                 ra_hidden_dim=64, alpha=0.5):
        self.inplanes = 64
        super(ResNet_CBAM_MSJF_RA, self).__init__()

        # ---- backbone ----
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # ---- MSJF branch ----
        self.use_layer2 = use_layer2
        backbone_dim = 512 * block.expansion  # 2048
        stability_dim = hidden_dim * 3 if use_layer2 else hidden_dim * 2

        self.mjfs = MJFS(
            layer2_channels=128 * block.expansion,
            layer3_channels=256 * block.expansion,
            layer4_channels=512 * block.expansion,
            hidden_dim=hidden_dim,
            use_layer2=use_layer2,
        )

        self.classifier = nn.Sequential(
            nn.Linear(backbone_dim + stability_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes),
        )

        self.jpeg_quality = jpeg_quality

        # ---- RA branch ----
        self.ra = RobustnessAsymmetry(
            layer3_channels=256 * block.expansion,
            layer4_channels=512 * block.expansion,
            hidden_dim=ra_hidden_dim,
        )

        # learnable alpha for combining cls_logits and stability_logits
        self.alpha = nn.Parameter(torch.tensor(alpha))

        # ---- eval degradation mode ----
        # When set to a string (e.g. "jpeg80"), _generate_degradation
        # uses a fixed degradation instead of random, for reproducible eval.
        self._eval_degradation = None

        # ---- normalization buffers ----
        self.register_buffer('_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # ---- weight init ----
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _forward_backbone(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        feat2 = self.layer2(x)
        feat3 = self.layer3(feat2)
        feat4 = self.layer4(feat3)

        return feat2, feat3, feat4

    # ---- MSJF JPEG perturbation (fixed quality) ----

    def _jpeg_perturb(self, x):
        """JPEG perturbation with fixed quality for the MSJF branch."""
        with torch.no_grad():
            x_denorm = x * self._std + self._mean
            x_denorm = x_denorm.clamp(0, 1)

            x_jpeg_list = []
            for i in range(x_denorm.size(0)):
                img = (x_denorm[i] * 255).byte().permute(1, 2, 0).cpu().numpy()
                pil_img = Image.fromarray(img, mode='RGB')

                buffer = io.BytesIO()
                pil_img.save(buffer, format='JPEG', quality=self.jpeg_quality)
                buffer.seek(0)
                pil_jpeg = Image.open(buffer).convert('RGB')

                img_jpeg = np.array(pil_jpeg).astype(np.float32) / 255.0
                img_jpeg = torch.from_numpy(img_jpeg).permute(2, 0, 1).to(x.device)
                x_jpeg_list.append(img_jpeg)

            x_jpeg = torch.stack(x_jpeg_list)
            x_jpeg = (x_jpeg - self._mean) / self._std

        return x_jpeg

    # ---- RA random degradation ----

    def _generate_degradation(self, x):
        """
        Generate a degraded version of x in memory.

        In training mode (_eval_degradation=None): randomly selects one of
          JPEG(70/80/90), Gaussian Blur(0.8/1.0/1.5), Downsample(0.5/0.75).

        In eval mode (_eval_degradation set): uses the fixed degradation
          specified by set_eval_degradation(), e.g. "jpeg80".
        """
        with torch.no_grad():
            if self._eval_degradation is not None:
                deg_type = self._eval_degradation[0]
            else:
                deg_type = random.choice(['jpeg', 'blur', 'downsample'])

            # Denormalize
            x_denorm = x * self._std + self._mean
            x_denorm = x_denorm.clamp(0, 1)

            if deg_type == 'jpeg':
                if self._eval_degradation is not None:
                    quality = self._eval_degradation[1]
                else:
                    quality = random.choice([70, 80, 90])
                x_deg_list = []
                for i in range(x_denorm.size(0)):
                    img = (x_denorm[i] * 255).byte().permute(1, 2, 0).cpu().numpy()
                    pil_img = Image.fromarray(img, mode='RGB')
                    buffer = io.BytesIO()
                    pil_img.save(buffer, format='JPEG', quality=quality)
                    buffer.seek(0)
                    pil_jpeg = Image.open(buffer).convert('RGB')
                    img_jpeg = np.array(pil_jpeg).astype(np.float32) / 255.0
                    img_jpeg = torch.from_numpy(img_jpeg).permute(2, 0, 1).to(x.device)
                    x_deg_list.append(img_jpeg)
                x_deg = torch.stack(x_deg_list)

            elif deg_type == 'blur':
                if self._eval_degradation is not None:
                    sigma = self._eval_degradation[1]
                else:
                    sigma = random.choice([0.8, 1.0, 1.5])
                # Use PIL GaussianBlur for simplicity
                x_deg_list = []
                for i in range(x_denorm.size(0)):
                    img = (x_denorm[i] * 255).byte().permute(1, 2, 0).cpu().numpy()
                    pil_img = Image.fromarray(img, mode='RGB').filter(
                        ImageFilter.GaussianBlur(radius=sigma)
                    )
                    img_blur = np.array(pil_img).astype(np.float32) / 255.0
                    img_blur = torch.from_numpy(img_blur).permute(2, 0, 1).to(x.device)
                    x_deg_list.append(img_blur)
                x_deg = torch.stack(x_deg_list)

            elif deg_type == 'downsample':
                if self._eval_degradation is not None:
                    scale = self._eval_degradation[1]
                else:
                    scale = random.choice([0.5, 0.75])
                H, W = x_denorm.shape[2], x_denorm.shape[3]
                new_h, new_w = int(H * scale), int(W * scale)
                x_small = F.interpolate(x_denorm, size=(new_h, new_w),
                                        mode='bilinear', align_corners=False)
                x_deg = F.interpolate(x_small, size=(H, W),
                                      mode='bilinear', align_corners=False)

            # Renormalize
            x_deg = (x_deg - self._mean) / self._std

        return x_deg

    def set_eval_degradation(self, deg_spec):
        """
        Set a fixed degradation for reproducible evaluation.

        Args:
            deg_spec: string like "jpeg80", "blur1.0", "downsample0.75",
                      or None to restore random degradation.
        """
        if deg_spec is None:
            self._eval_degradation = None
        elif deg_spec.startswith('jpeg'):
            quality = int(deg_spec[4:])
            self._eval_degradation = ('jpeg', quality)
        elif deg_spec.startswith('blur'):
            sigma = float(deg_spec[4:])
            self._eval_degradation = ('blur', sigma)
        elif deg_spec.startswith('downsample'):
            scale = float(deg_spec[10:])
            self._eval_degradation = ('downsample', scale)
        else:
            raise ValueError(
                "Unknown degradation spec: {}. "
                "Use 'jpeg80', 'blur1.0', 'downsample0.75', or None.".format(deg_spec)
            )

    # ---- forward ----

    def forward(self, x, return_aux=False):
        # 1. MSJF perturbation (fixed JPEG quality)
        x_jpeg = self._jpeg_perturb(x)

        # 2. RA perturbation (random degradation)
        x_deg = self._generate_degradation(x)

        # 3. Backbone passes
        feat2_x, feat3_x, feat4_x = self._forward_backbone(x)
        feat2_jpeg, feat3_jpeg, feat4_jpeg = self._forward_backbone(x_jpeg)
        _, feat3_deg, feat4_deg = self._forward_backbone(x_deg)

        # 4. MSJF branch → cls_logits
        backbone_feat = self.avgpool(feat4_x).flatten(1)

        stability = self.mjfs(
            feat2_x, feat3_x, feat4_x,
            feat2_jpeg, feat3_jpeg, feat4_jpeg,
        )

        F_final = torch.cat([backbone_feat] + list(stability), dim=1)
        cls_logits = self.classifier(F_final)

        # 5. RA branch → stability_logits
        stability_logits, ra_aux = self.ra(feat3_x, feat4_x, feat3_deg, feat4_deg)

        # 6. Final output
        final_logits = cls_logits + self.alpha * stability_logits

        if return_aux:
            aux_dict = {
                'cls_logits': cls_logits,
                'stability_logits': stability_logits,
                's3': ra_aux['s3'],
                's4': ra_aux['s4'],
                'D3_mean': ra_aux['D3_mean'],
                'D4_mean': ra_aux['D4_mean'],
            }
            return final_logits, aux_dict

        return final_logits


# ============================================================
# Build function
# ============================================================

def resnet50_cbam_msjf_ra(num_classes=2, pretrained_backbone=False,
                          hidden_dim=128, jpeg_quality=90, use_layer2=True,
                          ra_hidden_dim=64, alpha=0.5, logger=None, **kwargs):
    """
    Construct ResNet50-CBAM-MSJF-RA model.

    Args:
        num_classes:         number of output classes (default 2 for real/fake)
        pretrained_backbone: if True, load ImageNet pretrained ResNet50 weights
        hidden_dim:          hidden dimension for MSJF branch
        jpeg_quality:        JPEG quality for MSJF perturbation
        use_layer2:          whether to use layer2 in MSJF
        ra_hidden_dim:       hidden dimension for RA branch
        alpha:               initial alpha weight for stability_logits
        logger:              optional list for logging
    """
    model = ResNet_CBAM_MSJF_RA(
        Bottleneck, [3, 4, 6, 3],
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        jpeg_quality=jpeg_quality,
        use_layer2=use_layer2,
        ra_hidden_dim=ra_hidden_dim,
        alpha=alpha,
        **kwargs,
    )

    if pretrained_backbone:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet50'])
        loaded_count, _, _ = safe_load_backbone(model, pretrained_state_dict, logger)

        if logger is not None:
            mjfs_params = sum(p.numel() for n, p in model.named_parameters()
                              if 'mjfs' in n)
            ra_params = sum(p.numel() for n, p in model.named_parameters()
                            if 'ra' in n)
            cls_params = sum(p.numel() for n, p in model.named_parameters()
                             if 'classifier' in n)
            total_new = mjfs_params + ra_params + cls_params
            logger.append('Model: ResNet50-CBAM-MSJF-RA')
            logger.append('MSJF hidden_dim: {}'.format(hidden_dim))
            logger.append('JPEG quality: {}'.format(jpeg_quality))
            logger.append('RA hidden_dim: {}'.format(ra_hidden_dim))
            logger.append('RA alpha: {}'.format(alpha))
            logger.append('MSJF + RA + classifier params: {:,}'.format(total_new))
            logger.append('Backbone pretrained weights loaded: {}'.format(loaded_count))

    return model


# ============================================================
# Weight loading helper
# ============================================================

def safe_load_backbone(model, weights_dict_or_path, logger=None):
    """
    Safely load backbone weights from a dict or file path.
    Only loads keys that exist in the model and have matching shapes.
    """
    if isinstance(weights_dict_or_path, str):
        pretrained = torch.load(weights_dict_or_path, map_location='cpu', weights_only=False)
    else:
        pretrained = weights_dict_or_path

    model_state = model.state_dict()

    loaded = 0
    for key, value in pretrained.items():
        if key in model_state:
            if model_state[key].shape == value.shape:
                model_state[key] = value
                loaded += 1

    missing_keys = [k for k in model_state if k not in pretrained]
    unexpected_keys = [k for k in pretrained if k not in model_state]

    model.load_state_dict(model_state, strict=False)

    if logger is not None:
        logger.append('Backbone pretrained weights loaded: {}'.format(loaded))
        if len(missing_keys) > 0:
            logger.append('Missing keys (random init): {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            logger.append('Unexpected keys (ignored): {}'.format(unexpected_keys))

    return loaded, missing_keys, unexpected_keys


# ============================================================
# Loss function
# ============================================================

def robustness_asymmetry_loss(aux_dict, labels, margin=0.1):
    """
    Robustness Asymmetry Loss.

    Encourages:
      - real images (label=0): high similarity before/after degradation
      - fake images (label=1): lower similarity before/after degradation

    L_ra = ReLU((s_fake - s_real) + margin)

    Args:
        aux_dict: auxiliary dict from model.forward(..., return_aux=True)
        labels:   ground-truth labels (0=real, 1=fake)
        margin:   margin for the hinge loss

    Returns:
        scalar loss tensor
    """
    s3 = aux_dict['s3']
    s4 = aux_dict['s4']
    s = (s3 + s4) / 2.0

    mask_real = (labels == 0)
    mask_fake = (labels == 1)

    if mask_real.sum() == 0 or mask_fake.sum() == 0:
        return torch.tensor(0.0, device=s.device)

    s_real = s[mask_real].mean()
    s_fake = s[mask_fake].mean()

    L_ra = F.relu((s_fake - s_real) + margin)

    return L_ra


# ============================================================
# Verification
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ResNet50-CBAM-MSJF-RA Verification")
    print("=" * 60)

    # Build model
    model = resnet50_cbam_msjf_ra(num_classes=2, pretrained_backbone=False)
    model.eval()

    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ra_params = sum(p.numel() for n, p in model.named_parameters() if 'ra' in n)
    print("Total params:    {:,}".format(total))
    print("Trainable params:{:,}".format(trainable))
    print("RA branch params:{:,}".format(ra_params))

    # Test forward (default)
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        y = model(x)
    print("\nDefault forward:")
    print("  Input:  {}".format(x.shape))
    print("  Output: {}".format(y.shape))
    assert y.shape == (2, 2), "Expected (2, 2), got {}".format(y.shape)

    # Test forward with return_aux
    with torch.no_grad():
        y, aux = model(x, return_aux=True)
    print("\nForward with return_aux=True:")
    print("  Output: {}".format(y.shape))
    print("  Aux keys: {}".format(list(aux.keys())))
    for k, v in aux.items():
        if isinstance(v, torch.Tensor):
            print("    {}: shape={}".format(k, v.shape))
        else:
            print("    {}: {}".format(k, v))
    assert y.shape == (2, 2), "Expected (2, 2), got {}".format(y.shape)
    assert 'cls_logits' in aux
    assert 'stability_logits' in aux
    assert 's3' in aux and 's4' in aux
    assert 'D3_mean' in aux and 'D4_mean' in aux

    # Test loss function
    labels = torch.tensor([0, 1])
    L_ra = robustness_asymmetry_loss(aux, labels, margin=0.1)
    print("\nRobustness Asymmetry Loss:")
    print("  L_ra = {:.6f}".format(L_ra.item()))

    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED")
    print("=" * 60)