"""MiniResNet demo model with skip connections and non-Sequential Conv+BN attributes."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ptq_sim.models.registry import register_model


class ResidualBlock(nn.Module):
    """Basic residual block with optional projection shortcut."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        identity = self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + identity)
        return out


@register_model("mini_resnet")
class MiniResNet(nn.Module):
    """Compact ResNet architecture ideal for verifying torch.fx BatchNorm folding and PTQ."""

    def __init__(self, in_channels: int = 3, num_classes: int = 10, base_channels: int = 16) -> None:
        super().__init__()
        self.stem_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(base_channels)
        self.stage1 = ResidualBlock(base_channels, base_channels, stride=1)
        self.stage2 = ResidualBlock(base_channels, base_channels * 2, stride=2)
        self.stage3 = ResidualBlock(base_channels * 2, base_channels * 4, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_channels * 4, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        out = F.relu(self.stem_bn(self.stem_conv(x)))
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.pool(out)
        out = torch.flatten(out, 1)
        return self.fc(out)
