"""Final pyramid support blocks used by DCP-Det."""

from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


class LastLevelMaxPool(torch.nn.Module):
    """Create the P6 feature by stride-2 max pooling P5."""

    def forward(
        self, features: List[Tensor], backbone_features: List[Tensor], names: List[str]
    ) -> Tuple[List[Tensor], List[str]]:
        del backbone_features
        names.append("pool")
        features.append(F.max_pool2d(features[-1], kernel_size=1, stride=2, padding=0))
        return features, names

