import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo


__all__ = ['resnet50_cbam_sfgf']


model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
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


class LightweightSFGF(nn.Module):

    def __init__(self, in_channels=1024, hidden_channels=32):
        super(LightweightSFGF, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels

        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )

        self.freq_fusion = nn.Sequential(
            nn.Conv2d(hidden_channels * 3, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )

        self.restore = nn.Sequential(
            nn.Conv2d(hidden_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels)
        )

        self.gate_pool = nn.AdaptiveAvgPool2d(1)
        self.gate_conv1 = nn.Conv2d(in_channels, 64, kernel_size=1, bias=False)
        self.gate_relu = nn.ReLU(inplace=True)
        self.gate_conv2 = nn.Conv2d(64, in_channels, kernel_size=1, bias=False)
        self.gate_sigmoid = nn.Sigmoid()

        self.beta = nn.Parameter(torch.tensor(0.1))

    def _build_freq_masks(self, H, W, device):
        y = torch.arange(H, device=device).float()
        x = torch.arange(W, device=device).float()
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        center_y = H / 2.0
        center_x = W / 2.0
        dist = torch.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2)
        max_dist = dist.max()

        low_mask = (dist <= 0.25 * max_dist).float()
        mid_mask = ((dist > 0.25 * max_dist) & (dist <= 0.55 * max_dist)).float()
        high_mask = (dist > 0.55 * max_dist).float()

        low_mask = low_mask.unsqueeze(0).unsqueeze(0)
        mid_mask = mid_mask.unsqueeze(0).unsqueeze(0)
        high_mask = high_mask.unsqueeze(0).unsqueeze(0)

        return low_mask, mid_mask, high_mask

    def forward(self, x):
        B, C, H, W = x.shape
        input_dtype = x.dtype

        x_reduced = self.reduce(x)

        fft = torch.fft.fft2(x_reduced.float(), norm='ortho')
        fft_shift = torch.fft.fftshift(fft)

        low_mask, mid_mask, high_mask = self._build_freq_masks(H, W, x.device)

        fft_low = fft_shift * low_mask
        fft_mid = fft_shift * mid_mask
        fft_high = fft_shift * high_mask

        fft_low_shift = torch.fft.ifftshift(fft_low)
        fft_mid_shift = torch.fft.ifftshift(fft_mid)
        fft_high_shift = torch.fft.ifftshift(fft_high)

        low_spatial = torch.fft.ifft2(fft_low_shift, norm='ortho').real
        mid_spatial = torch.fft.ifft2(fft_mid_shift, norm='ortho').real
        high_spatial = torch.fft.ifft2(fft_high_shift, norm='ortho').real

        if input_dtype == torch.half:
            low_spatial = low_spatial.half()
            mid_spatial = mid_spatial.half()
            high_spatial = high_spatial.half()

        freq_cat = torch.cat([low_spatial, mid_spatial, high_spatial], dim=1)

        freq_fused = self.freq_fusion(freq_cat)
        freq_feature = self.restore(freq_fused)

        gate = self.gate_pool(x)
        gate = self.gate_conv1(gate)
        gate = self.gate_relu(gate)
        gate = self.gate_conv2(gate)
        gate = self.gate_sigmoid(gate)

        out = x + self.beta * gate * freq_feature

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, sfgf_hidden_channels=32):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)

        self.sfgf = LightweightSFGF(256 * block.expansion, hidden_channels=sfgf_hidden_channels)

        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

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

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.sfgf(x)

        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def safe_load_pretrained(model, pretrained_state_dict, logger=None):
    model_state_dict = model.state_dict()
    loaded_keys = []
    skipped_keys = []

    for key, value in pretrained_state_dict.items():
        if key in model_state_dict:
            if model_state_dict[key].shape == value.shape:
                model_state_dict[key] = value
                loaded_keys.append(key)
            else:
                skipped_keys.append(key)
        else:
            skipped_keys.append(key)

    model.load_state_dict(model_state_dict)

    if logger is not None:
        logger.append('Pretrained weights loaded: {}/{}'.format(len(loaded_keys), len(pretrained_state_dict)))
        if len(skipped_keys) > 0:
            logger.append('Skipped keys (shape mismatch or new params):')
            for sk in skipped_keys:
                logger.append('  ' + sk)

    return len(loaded_keys), skipped_keys


def count_sfgf_params(model):
    sfgf_param_count = 0
    for name, param in model.named_parameters():
        if 'sfgf' in name:
            sfgf_param_count += param.numel()
    return sfgf_param_count


def resnet50_cbam_sfgf(pretrained=False, num_classes=1000, sfgf_hidden_channels=32, logger=None, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes,
                   sfgf_hidden_channels=sfgf_hidden_channels, **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet50'])
        loaded_count, skipped_keys = safe_load_pretrained(model, pretrained_state_dict, logger)

        if logger is not None:
            sfgf_params = count_sfgf_params(model)
            logger.append('Model: ResNet50-CBAM-SFGF')
            logger.append('SFGF: inserted after layer3')
            logger.append('hidden_channels: {}'.format(sfgf_hidden_channels))
            logger.append('SFGF module params: {:,}'.format(sfgf_params))
            logger.append('beta initial value: {:.4f}'.format(model.sfgf.beta.item()))
            logger.append('Fusion: out = x + beta * gate * freq_feature')
            logger.append('Pretrained ImageNet ResNet50 params loaded: {}'.format(loaded_count))

    return model