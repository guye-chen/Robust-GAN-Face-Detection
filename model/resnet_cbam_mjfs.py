import io
import math
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


__all__ = ['resnet50_cbam_mjfs', 'safe_load_backbone']


model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
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


class MJFS(nn.Module):

    def __init__(self, layer2_channels=512, layer3_channels=1024, layer4_channels=2048,
                 hidden_dim=128, use_layer2=True):
        super(MJFS, self).__init__()

        self.use_layer2 = use_layer2

        if self.use_layer2:
            self.layer2_proj = nn.Sequential(
                nn.Conv2d(layer2_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True)
            )

        self.layer3_proj = nn.Sequential(
            nn.Conv2d(layer3_channels, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )

        self.layer4_proj = nn.Sequential(
            nn.Conv2d(layer4_channels, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
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


class ResNet_CBAM_MJFS(nn.Module):

    def __init__(self, block, layers, num_classes=2, hidden_dim=128, jpeg_quality=90,
                 use_layer2=True):
        self.inplanes = 64
        super(ResNet_CBAM_MJFS, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.use_layer2 = use_layer2
        backbone_dim = 512 * block.expansion
        stability_dim = hidden_dim * 3 if use_layer2 else hidden_dim * 2

        self.mjfs = MJFS(
            layer2_channels=128 * block.expansion,
            layer3_channels=256 * block.expansion,
            layer4_channels=512 * block.expansion,
            hidden_dim=hidden_dim,
            use_layer2=use_layer2
        )

        self.classifier = nn.Sequential(
            nn.Linear(backbone_dim + stability_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes)
        )

        self.jpeg_quality = jpeg_quality

        self.register_buffer('_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

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

    def _jpeg_perturb(self, x):
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

    def forward(self, x):
        x_jpeg = self._jpeg_perturb(x)

        feat2_x, feat3_x, feat4_x = self._forward_backbone(x)
        feat2_jpeg, feat3_jpeg, feat4_jpeg = self._forward_backbone(x_jpeg)

        backbone_feat = self.avgpool(feat4_x).flatten(1)

        stability = self.mjfs(
            feat2_x, feat3_x, feat4_x,
            feat2_jpeg, feat3_jpeg, feat4_jpeg
        )

        F_final = torch.cat([backbone_feat] + list(stability), dim=1)
        logits = self.classifier(F_final)

        return logits


def safe_load_backbone(model, weights_path, logger=None):
    pretrained = torch.load(weights_path, map_location='cpu', weights_only=False)
    model_state = model.state_dict()

    loaded = 0
    skipped = []
    for key, value in pretrained.items():
        if key in model_state:
            if model_state[key].shape == value.shape:
                model_state[key] = value
                loaded += 1
            else:
                skipped.append(key)
        else:
            skipped.append(key)

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


def resnet50_cbam_mjfs(pretrained=False, num_classes=2, hidden_dim=128,
                       jpeg_quality=90, use_layer2=True, logger=None, **kwargs):
    model = ResNet_CBAM_MJFS(Bottleneck, [3, 4, 6, 3], num_classes=num_classes,
                             hidden_dim=hidden_dim, jpeg_quality=jpeg_quality,
                             use_layer2=use_layer2, **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet50'])
        loaded_count, skipped_keys = safe_load_backbone(
            model, pretrained_state_dict, logger)

        if logger is not None:
            mjfs_params = sum(p.numel() for n, p in model.named_parameters()
                              if 'mjfs' in n) + sum(
                                  p.numel() for n, p in model.named_parameters()
                                  if 'classifier' in n)
            logger.append('Model: ResNet50-CBAM-MJFS')
            logger.append('MJFS hidden_dim: {}'.format(hidden_dim))
            logger.append('JPEG quality: {}'.format(jpeg_quality))
            logger.append('MJFS + classifier params: {:,}'.format(mjfs_params))
            logger.append(
                'F_final = concat(backbone_feat, d_l2, d_l3, d_l4) -> MLP classifier')

    return model