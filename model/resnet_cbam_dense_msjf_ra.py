"""
ResNet50 + CBAM + Dense-MSJF + Real-Stability RA

Dense Multi-Scale JPEG Feature Stability with Real-Stability Robustness Asymmetry.

Key innovations over resnet_cbam_msjf_ra.py:

  1. Dense Feature Reuse: MSJF/RA branches now use layer1-layer4 features
     (not just layer3/4), enabling finer-grained degradation detection.

  2. Real-Stability RA Loss: Focuses on keeping real images stable under
     strong JPEG compression, rather than pushing fake images further apart.
     This directly addresses the real-accuracy drop at JPEG70.

Architecture:
    Input x
      |
    ResNet50 backbone + CBAM
      +-- feat1, feat2, feat3, feat4 (from x)
      +-- feat1_jpeg, feat2_jpeg, feat3_jpeg, feat4_jpeg (from JPEG-perturbed x)
      +-- feat1_deg, feat2_deg, feat3_deg, feat4_deg (from randomly degraded x)
      |
    Dense-MSJF branch: layer1-4 feature differences -> cls_logits
    Dense-RA branch:   layer1-4 degradation stability -> stability_logits
      |
    final_logits = cls_logits + alpha * stability_logits

Lightweight: ~0.15M-0.25M additional parameters over the base MSJF-RA.

Author: Dense-MSJF-RA Extension
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
    'resnet50_cbam_dense_msjf_ra',
    'safe_load_backbone',
    'real_stability_ra_loss',
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
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv1(x))


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
# Dense-MSJF module (layer1-4 feature reuse)
# ============================================================

class DenseMSJF(nn.Module):
    """
    Dense Multi-Scale JPEG Feature Stability module.

    Unlike the original MSJF which only uses layer2-4, this version uses
    layer1-4 for richer multi-scale degradation analysis.
    """

    def __init__(self, layer1_channels=256, layer2_channels=512,
                 layer3_channels=1024, layer4_channels=2048,
                 hidden_dim=128, dense_dim=32, use_layer1=True):
        super(DenseMSJF, self).__init__()

        self.use_layer1 = use_layer1
        self.dense_dim = dense_dim

        # Project each layer's difference to dense_dim
        if self.use_layer1:
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

        self.gap = nn.AdaptiveAvgPool2d(1)

        # Number of features fed into classifier
        if self.use_layer1:
            self.stability_dim = dense_dim * 4
        else:
            self.stability_dim = dense_dim * 3

    def forward(self, feat1_x, feat2_x, feat3_x, feat4_x,
                feat1_jpeg, feat2_jpeg, feat3_jpeg, feat4_jpeg):
        results = []

        if self.use_layer1:
            d1 = torch.abs(feat1_x - feat1_jpeg)
            d1 = self.layer1_proj(d1)
            results.append(self.gap(d1).flatten(1))

        d2 = torch.abs(feat2_x - feat2_jpeg)
        d2 = self.layer2_proj(d2)
        results.append(self.gap(d2).flatten(1))

        d3 = torch.abs(feat3_x - feat3_jpeg)
        d3 = self.layer3_proj(d3)
        results.append(self.gap(d3).flatten(1))

        d4 = torch.abs(feat4_x - feat4_jpeg)
        d4 = self.layer4_proj(d4)
        results.append(self.gap(d4).flatten(1))

        return tuple(results)


# ============================================================
# Dense-RA (Robustness Asymmetry) branch
# ============================================================

class DenseRobustnessAsymmetry(nn.Module):
    """
    Dense degradation stability branch using layer1-4 features.

    Extracts layer1-4 features from both the original and degraded inputs,
    computes absolute differences and cosine similarities, then feeds
    into a lightweight MLP.

    Total parameters: ~150K-250K.
    """

    def __init__(self, layer1_channels=256, layer2_channels=512,
                 layer3_channels=1024, layer4_channels=2048,
                 hidden_dim=64, dense_dim=32, use_layer1=True):
        super(DenseRobustnessAsymmetry, self).__init__()

        self.use_layer1 = use_layer1
        self.dense_dim = dense_dim
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Project each layer's difference to dense_dim
        if self.use_layer1:
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

        # MLP input: dense_dim * num_layers (differences) + num_layers (similarities)
        if self.use_layer1:
            num_diff_features = dense_dim * 4
            num_sim_features = 4
        else:
            num_diff_features = dense_dim * 3
            num_sim_features = 3

        mlp_input_dim = num_diff_features + num_sim_features
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, feat1_x, feat2_x, feat3_x, feat4_x,
                feat1_deg, feat2_deg, feat3_deg, feat4_deg):
        diff_features = []
        sim_features = []

        layers_x = []
        layers_deg = []
        if self.use_layer1:
            layers_x.append(feat1_x)
            layers_deg.append(feat1_deg)
        layers_x.extend([feat2_x, feat3_x, feat4_x])
        layers_deg.extend([feat2_deg, feat3_deg, feat4_deg])

        projs = []
        if self.use_layer1:
            projs.append(self.layer1_proj)
        projs.extend([self.layer2_proj, self.layer3_proj, self.layer4_proj])

        # ---- per-layer differences and similarities ----
        D_means = {}
        for idx, (f_x, f_deg, proj) in enumerate(zip(layers_x, layers_deg, projs)):
            # Absolute difference
            D = torch.abs(f_x - f_deg)
            d_proj = self.gap(proj(D)).flatten(1)
            diff_features.append(d_proj)

            # Cosine similarity
            g_x = self.gap(f_x).flatten(1)
            g_deg = self.gap(f_deg).flatten(1)
            s = F.cosine_similarity(g_x, g_deg, dim=1).unsqueeze(1)
            sim_features.append(s)

            # Track mean difference for logging
            D_means['D{}_mean'.format(idx + 1)] = D.mean()

        # ---- MLP ----
        feat = torch.cat(diff_features + sim_features, dim=1)
        stability_logits = self.mlp(feat)

        # ---- dense_diff_score for real-stability loss ----
        # Average of all cosine similarities as a stability score
        sim_stack = torch.cat(sim_features, dim=1)  # [B, num_layers]
        dense_diff_score = sim_stack.mean(dim=1)     # [B]

        aux = {
            'dense_diff_score': dense_diff_score,
            'D1_mean': D_means.get('D1_mean', torch.tensor(0.0)),
            'D2_mean': D_means.get('D2_mean', torch.tensor(0.0)),
            'D3_mean': D_means.get('D3_mean', torch.tensor(0.0)),
            'D4_mean': D_means.get('D4_mean', torch.tensor(0.0)),
        }

        return stability_logits, aux


# ============================================================
# Main model: ResNet50 + CBAM + Dense-MSJF + Real-Stability RA
# ============================================================

class ResNet_CBAM_Dense_MSJF_RA(nn.Module):
    """
    ResNet50 + CBAM + Dense-MSJF + Real-Stability RA

    Architecture:
        Input x
          |
        ResNet50 backbone + CBAM
          +-- feat1, feat2, feat3, feat4 (from x)
          +-- feat1_jpeg, feat2_jpeg, feat3_jpeg, feat4_jpeg (from JPEG-perturbed x)
          +-- feat1_deg, feat2_deg, feat3_deg, feat4_deg (from randomly degraded x)
          |
        Dense-MSJF branch: layer1-4 JPEG feature differences -> cls_logits
        Dense-RA branch:   layer1-4 degradation stability -> stability_logits
          |
        final_logits = cls_logits + alpha * stability_logits
    """

    def __init__(self, block, layers, num_classes=2,
                 hidden_dim=128, jpeg_quality=90, use_layer1=True,
                 dense_dim=32, ra_hidden_dim=64, alpha=0.5):
        self.inplanes = 64
        super(ResNet_CBAM_Dense_MSJF_RA, self).__init__()

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

        # ---- Dense-MSJF branch ----
        self.use_layer1 = use_layer1
        backbone_dim = 512 * block.expansion  # 2048

        self.dense_msjf = DenseMSJF(
            layer1_channels=64 * block.expansion,
            layer2_channels=128 * block.expansion,
            layer3_channels=256 * block.expansion,
            layer4_channels=512 * block.expansion,
            hidden_dim=hidden_dim,
            dense_dim=dense_dim,
            use_layer1=use_layer1,
        )

        stability_dim = self.dense_msjf.stability_dim

        self.classifier = nn.Sequential(
            nn.Linear(backbone_dim + stability_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes),
        )

        self.jpeg_quality = jpeg_quality

        # ---- Dense-RA branch ----
        self.dense_ra = DenseRobustnessAsymmetry(
            layer1_channels=64 * block.expansion,
            layer2_channels=128 * block.expansion,
            layer3_channels=256 * block.expansion,
            layer4_channels=512 * block.expansion,
            hidden_dim=ra_hidden_dim,
            dense_dim=dense_dim,
            use_layer1=use_layer1,
        )

        # learnable alpha for combining cls_logits and stability_logits
        self.alpha = nn.Parameter(torch.tensor(alpha))

        # ---- eval degradation mode ----
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

        feat1 = self.layer1(x)
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)
        feat4 = self.layer4(feat3)

        return feat1, feat2, feat3, feat4

    # ---- MSJF JPEG perturbation (fixed quality) ----

    def _jpeg_perturb(self, x):
        """JPEG perturbation with fixed quality for the Dense-MSJF branch."""
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
        feat1_x, feat2_x, feat3_x, feat4_x = self._forward_backbone(x)
        feat1_jpeg, feat2_jpeg, feat3_jpeg, feat4_jpeg = self._forward_backbone(x_jpeg)
        feat1_deg, feat2_deg, feat3_deg, feat4_deg = self._forward_backbone(x_deg)

        # 4. Dense-MSJF branch -> cls_logits
        backbone_feat = self.avgpool(feat4_x).flatten(1)

        stability = self.dense_msjf(
            feat1_x, feat2_x, feat3_x, feat4_x,
            feat1_jpeg, feat2_jpeg, feat3_jpeg, feat4_jpeg,
        )

        F_final = torch.cat([backbone_feat] + list(stability), dim=1)
        cls_logits = self.classifier(F_final)

        # 5. Dense-RA branch -> stability_logits
        stability_logits, ra_aux = self.dense_ra(
            feat1_x, feat2_x, feat3_x, feat4_x,
            feat1_deg, feat2_deg, feat3_deg, feat4_deg,
        )

        # 6. Final output
        final_logits = cls_logits + self.alpha * stability_logits

        if return_aux:
            aux_dict = {
                'cls_logits': cls_logits,
                'stability_logits': stability_logits,
                'dense_diff_score': ra_aux['dense_diff_score'],
                'D1_mean': ra_aux['D1_mean'],
                'D2_mean': ra_aux['D2_mean'],
                'D3_mean': ra_aux['D3_mean'],
                'D4_mean': ra_aux['D4_mean'],
            }
            return final_logits, aux_dict

        return final_logits


# ============================================================
# Build function
# ============================================================

def resnet50_cbam_dense_msjf_ra(num_classes=2, pretrained_backbone=False,
                                hidden_dim=128, jpeg_quality=90, use_layer1=True,
                                dense_dim=32, ra_hidden_dim=64, alpha=0.5,
                                logger=None, **kwargs):
    """
    Construct ResNet50-CBAM-Dense-MSJF-RA model.

    Args:
        num_classes:         number of output classes (default 2 for real/fake)
        pretrained_backbone: if True, load ImageNet pretrained ResNet50 weights
        hidden_dim:          hidden dimension for classifier MLP
        jpeg_quality:        JPEG quality for MSJF perturbation
        use_layer1:          whether to use layer1 in Dense-MSJF and Dense-RA
        dense_dim:           channel dimension for dense feature projections
        ra_hidden_dim:       hidden dimension for RA branch MLP
        alpha:               initial alpha weight for stability_logits
        logger:              optional list for logging
    """
    model = ResNet_CBAM_Dense_MSJF_RA(
        Bottleneck, [3, 4, 6, 3],
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        jpeg_quality=jpeg_quality,
        use_layer1=use_layer1,
        dense_dim=dense_dim,
        ra_hidden_dim=ra_hidden_dim,
        alpha=alpha,
        **kwargs,
    )

    if pretrained_backbone:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet50'])
        loaded_count, _, _ = safe_load_backbone(model, pretrained_state_dict, logger)

        if logger is not None:
            dense_msjf_params = sum(p.numel() for n, p in model.named_parameters()
                                    if 'dense_msjf' in n)
            dense_ra_params = sum(p.numel() for n, p in model.named_parameters()
                                  if 'dense_ra' in n)
            cls_params = sum(p.numel() for n, p in model.named_parameters()
                             if 'classifier' in n)
            total_new = dense_msjf_params + dense_ra_params + cls_params
            logger.append('Model: ResNet50-CBAM-Dense-MSJF-RA')
            logger.append('Dense-MSJF dense_dim: {}'.format(dense_dim))
            logger.append('JPEG quality: {}'.format(jpeg_quality))
            logger.append('RA hidden_dim: {}'.format(ra_hidden_dim))
            logger.append('RA alpha: {}'.format(alpha))
            logger.append('Dense-MSJF + RA + classifier params: {:,}'.format(total_new))
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
# Real-Stability RA Loss
# ============================================================

def real_stability_ra_loss(aux_dict, labels, margin=0.1):
    """
    Real-Stability Robustness Asymmetry Loss.

    Focuses on keeping real images stable under degradation, rather than
    pushing fake images to be unstable. This directly addresses the problem
    of real images being misclassified as fake under strong JPEG compression.

    L_ra = ReLU(s_fake - s_real + margin) + lambda_real * (1 - s_real)

    where:
      - s = dense_diff_score (average cosine similarity across all layers)
      - s_real = mean similarity for real images
      - s_fake = mean similarity for fake images
      - The first term is the original RA loss (margin between real and fake)
      - The second term explicitly encourages real images to have high similarity

    Args:
        aux_dict: auxiliary dict from model.forward(..., return_aux=True)
        labels:   ground-truth labels (0=real, 1=fake)
        margin:   margin for the hinge loss

    Returns:
        scalar loss tensor
    """
    s = aux_dict['dense_diff_score']  # [B]

    mask_real = (labels == 0)
    mask_fake = (labels == 1)

    if mask_real.sum() == 0 or mask_fake.sum() == 0:
        return torch.tensor(0.0, device=s.device)

    # ---- core RA term: fake should be less stable than real ----
    s_real = s[mask_real].mean()
    s_fake = s[mask_fake].mean()
    L_ra_margin = F.relu((s_fake - s_real) + margin)

    # ---- real-stability term: real images should be highly stable ----
    # Encourage real cosine similarity to be close to 1.0
    L_real_stability = (1.0 - s_real)

    # Combined loss: primary RA margin + real-stability regularization
    # Weight the real-stability term lower to avoid dominating the margin loss
    L_ra = L_ra_margin + 0.5 * L_real_stability

    return L_ra


# ============================================================
# Verification
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ResNet50-CBAM-Dense-MSJF-RA Verification")
    print("=" * 60)

    # Build model
    model = resnet50_cbam_dense_msjf_ra(num_classes=2, pretrained_backbone=False)
    model.eval()

    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    backbone_params = sum(p.numel() for n, p in model.named_parameters()
                          if not any(x in n for x in ['dense_msjf', 'dense_ra', 'classifier', 'alpha']))
    dense_msjf_params = sum(p.numel() for n, p in model.named_parameters()
                            if 'dense_msjf' in n)
    dense_ra_params = sum(p.numel() for n, p in model.named_parameters()
                          if 'dense_ra' in n)
    classifier_params = sum(p.numel() for n, p in model.named_parameters()
                            if 'classifier' in n)
    alpha_params = sum(p.numel() for n, p in model.named_parameters()
                       if 'alpha' in n)

    new_params = dense_msjf_params + dense_ra_params + classifier_params + alpha_params

    print("Total params:              {:,}".format(total))
    print("Trainable params:          {:,}".format(trainable))
    print("  Backbone (ResNet50+CBAM): {:,}".format(backbone_params))
    print("  Dense-MSJF branch:        {:,}".format(dense_msjf_params))
    print("  Dense-RA branch:          {:,}".format(dense_ra_params))
    print("  Classifier MLP:           {:,}".format(classifier_params))
    print("  Alpha (learnable):        {:,}".format(alpha_params))
    print("  New params (non-backbone): {:,}".format(new_params))

    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    print("\n--- Forward pass test ---")

    # Default forward (no aux)
    with torch.no_grad():
        out = model(x)
    print("Default forward  (2,3,224,224) -> {}".format(out.shape))
    assert out.shape == (2, 2), "Expected (2,2), got {}".format(out.shape)

    # Forward with aux
    model.set_eval_degradation("jpeg80")
    with torch.no_grad():
        out_aux = model(x, return_aux=True)
    assert isinstance(out_aux, tuple) and len(out_aux) == 2, "Expected tuple of 2"
    final_logits, aux = out_aux
    print("return_aux=True  (2,3,224,224) -> logits: {}, aux keys: {}".format(
        final_logits.shape, list(aux.keys())))
    assert final_logits.shape == (2, 2), "Expected logits (2,2)"
    assert 'dense_diff_score' in aux, "Missing dense_diff_score in aux"
    assert 'D1_mean' in aux, "Missing D1_mean in aux"
    assert 'D4_mean' in aux, "Missing D4_mean in aux"

    # Test loss function
    labels = torch.tensor([0, 1])
    loss = real_stability_ra_loss(aux, labels, margin=0.1)
    print("\nreal_stability_ra_loss: {:.6f}".format(loss.item()))
    print("  dense_diff_score: {}".format(aux['dense_diff_score']))

    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED")
    print("=" * 60)