"""Defect Contrast Prior extraction and the final DCP-EVC feature pyramid."""

from collections import OrderedDict
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .evc import EVCBlock


class LocalContrastPriorExtractor(nn.Module):
    """Extract the parameter-free local grayscale-deviation prior."""

    def __init__(
        self,
        kernel_size: int = 31,
        prior_input: str = "normalized_padded",
        image_mean=None,
        image_std=None,
    ) -> None:
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("DCP kernel_size must be a positive odd integer")
        if prior_input != "normalized_padded":
            raise ValueError("Final DCP-Det requires prior_input=normalized_padded")
        self.kernel_size = int(kernel_size)
        self.padding = self.kernel_size // 2
        self.prior_input = prior_input
        image_mean = [0.485, 0.456, 0.406] if image_mean is None else image_mean
        image_std = [0.229, 0.224, 0.225] if image_std is None else image_std
        self.register_buffer("image_mean", torch.tensor(image_mean).reshape(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor(image_std).reshape(1, 3, 1, 1))

    def forward(self, normalized_padded_images: Tensor) -> Tensor:
        valid_mask = (
            normalized_padded_images.abs().sum(dim=1, keepdim=True) > 1e-8
        ).to(normalized_padded_images.dtype)
        gray = normalized_padded_images.mean(dim=1, keepdim=True)
        background_sum = F.avg_pool2d(
            gray * valid_mask,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.padding,
        )
        background_count = F.avg_pool2d(
            valid_mask,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.padding,
        )
        background = background_sum / (background_count + 1e-6)
        contrast = torch.abs(gray - background) * valid_mask

        batch, channels, height, width = contrast.shape
        flat = contrast.reshape(batch, -1)
        flat_mask = valid_mask.reshape(batch, -1)
        minimum_pool = flat.clone()
        minimum_pool[flat_mask == 0] = float("inf")
        maximum_pool = flat.clone()
        maximum_pool[flat_mask == 0] = float("-inf")
        minimum = minimum_pool.min(dim=1, keepdim=True)[0]
        maximum = maximum_pool.max(dim=1, keepdim=True)[0]
        normalized = (flat - minimum) / (maximum - minimum + 1e-6)
        return normalized.reshape(batch, channels, height, width) * valid_mask


DefectContrastPriorExtractor = LocalContrastPriorExtractor


class DCPEVCFPN(nn.Module):
    """Final DCP-EVC pyramid: EVC and DCP modulation are applied to P4/P5."""

    def __init__(
        self,
        in_channels_list=(256, 512, 1024, 2048),
        out_channels: int = 256,
        kernel_size: int = 31,
        phys_gamma_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.contrast_prior_extractor = LocalContrastPriorExtractor(kernel_size=kernel_size)
        self.reduce_c2 = nn.Conv2d(in_channels_list[0], out_channels, 1)
        self.reduce_c3 = nn.Conv2d(in_channels_list[1], out_channels, 1)
        self.reduce_c4 = nn.Conv2d(in_channels_list[2], out_channels, 1)
        self.reduce_c5 = nn.Conv2d(in_channels_list[3], out_channels, 1)
        self.evc_c4 = EVCBlock(in_channels=out_channels, out_channels=out_channels)
        self.evc_c5 = EVCBlock(in_channels=out_channels, out_channels=out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.conv3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.conv4 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.conv5 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.phys_gamma = nn.Parameter(torch.tensor([float(phys_gamma_init)]))

    def _modulate(self, feature: Tensor, contrast: Tensor) -> Tensor:
        mask = F.adaptive_avg_pool2d(contrast, feature.shape[-2:])
        return feature * (1.0 + self.phys_gamma * mask)

    def forward(self, features: Dict[str, Tensor], transformed_images: Tensor):
        c2, c3, c4, c5 = (features[str(index)] for index in range(4))
        contrast = self.contrast_prior_extractor(transformed_images)
        p2_lateral = self.reduce_c2(c2)
        p3_lateral = self.reduce_c3(c3)
        p4 = self._modulate(self.evc_c4(self.reduce_c4(c4)), contrast)
        p5 = self._modulate(self.evc_c5(self.reduce_c5(c5)), contrast)
        p4_topdown = p4 + F.interpolate(p5, size=p4.shape[-2:], mode="nearest")
        p3 = p3_lateral + F.interpolate(p4_topdown, size=p3_lateral.shape[-2:], mode="nearest")
        p2 = p2_lateral + F.interpolate(p3, size=p2_lateral.shape[-2:], mode="nearest")
        return OrderedDict(
            (
                ("0", self.conv2(p2)),
                ("1", self.conv3(p3)),
                ("2", self.conv4(p4_topdown)),
                ("3", self.conv5(p5)),
            )
        )
