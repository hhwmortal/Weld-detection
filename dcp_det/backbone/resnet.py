"""ResNet-50 backbone and final DCP-EVC pyramid assembly."""

from collections import OrderedDict
from typing import Dict

import torch.nn as nn
from torchvision.models import ResNet50_Weights
from torchvision.models._utils import IntermediateLayerGetter

from ..modules.dcp_evc_fpn import DCPEVCFPN
from .fpn import LastLevelMaxPool


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, channels, stride=1, downsample=None, norm_layer=None):
        super().__init__()
        norm_layer = nn.BatchNorm2d if norm_layer is None else norm_layer
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=1, bias=False)
        self.bn1 = norm_layer(channels)
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = norm_layer(channels)
        self.conv3 = nn.Conv2d(channels, channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm_layer(channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, inputs):
        identity = inputs if self.downsample is None else self.downsample(inputs)
        output = self.relu(self.bn1(self.conv1(inputs)))
        output = self.relu(self.bn2(self.conv2(output)))
        output = self.bn3(self.conv3(output))
        return self.relu(output + identity)


class ResNet(nn.Module):
    def __init__(self, norm_layer=None):
        super().__init__()
        norm_layer = nn.BatchNorm2d if norm_layer is None else norm_layer
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 3, norm_layer=norm_layer)
        self.layer2 = self._make_layer(128, 4, stride=2, norm_layer=norm_layer)
        self.layer3 = self._make_layer(256, 6, stride=2, norm_layer=norm_layer)
        self.layer4 = self._make_layer(512, 3, stride=2, norm_layer=norm_layer)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")

    def _make_layer(self, channels, count, stride=1, norm_layer=None):
        downsample = None
        if stride != 1 or self.in_channels != channels * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    channels * Bottleneck.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                norm_layer(channels * Bottleneck.expansion),
            )
        blocks = [
            Bottleneck(
                self.in_channels,
                channels,
                stride=stride,
                downsample=downsample,
                norm_layer=norm_layer,
            )
        ]
        self.in_channels = channels * Bottleneck.expansion
        blocks.extend(
            Bottleneck(self.in_channels, channels, norm_layer=norm_layer)
            for _ in range(1, count)
        )
        return nn.Sequential(*blocks)

    def forward(self, inputs):
        output = self.maxpool(self.relu(self.bn1(self.conv1(inputs))))
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        return self.layer4(output)


class DCPDetBackbone(nn.Module):
    def __init__(self, resnet: ResNet, kernel_size: int, phys_gamma_init: float):
        super().__init__()
        return_layers: Dict[str, str] = {
            "layer1": "0",
            "layer2": "1",
            "layer3": "2",
            "layer4": "3",
        }
        self.body = IntermediateLayerGetter(resnet, return_layers=return_layers)
        self.custom_fpn = DCPEVCFPN(kernel_size=kernel_size, phys_gamma_init=phys_gamma_init)
        self.extra_blocks = LastLevelMaxPool()
        self.out_channels = 256

    def forward(self, images):
        backbone_features = self.body(images)
        pyramid = self.custom_fpn(backbone_features, images)
        names = list(pyramid.keys())
        outputs, names = self.extra_blocks(
            list(pyramid.values()), list(backbone_features.values()), names
        )
        return OrderedDict((str(index), value) for index, value in enumerate(outputs))


def build_resnet50_dcp_evc_backbone(
    pretrained: bool = True,
    trainable_layers: int = 3,
    kernel_size: int = 31,
    phys_gamma_init: float = 0.0,
) -> DCPDetBackbone:
    """Build the final backbone using public torchvision ImageNet weights."""

    if not 0 <= trainable_layers <= 5:
        raise ValueError("trainable_layers must be in [0, 5]")
    backbone = ResNet(norm_layer=nn.BatchNorm2d)
    if pretrained:
        state_dict = ResNet50_Weights.IMAGENET1K_V1.get_state_dict(
            progress=True, check_hash=True
        )
        incompatible = backbone.load_state_dict(state_dict, strict=False)
        if set(incompatible.unexpected_keys) != {"fc.weight", "fc.bias"}:
            raise RuntimeError("Unexpected ImageNet initialization mismatch: {}".format(incompatible))
        if incompatible.missing_keys:
            raise RuntimeError(
                "Missing ImageNet backbone parameters: {}".format(incompatible.missing_keys)
            )

    layers_to_train = ["layer4", "layer3", "layer2", "layer1", "conv1"][:trainable_layers]
    if trainable_layers == 5:
        layers_to_train.append("bn1")
    for name, parameter in backbone.named_parameters():
        if all(not name.startswith(layer) for layer in layers_to_train):
            parameter.requires_grad_(False)
    return DCPDetBackbone(backbone, kernel_size=kernel_size, phys_gamma_init=phys_gamma_init)

