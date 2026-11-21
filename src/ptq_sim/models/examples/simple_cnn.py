"""SimpleCNN demo model with Conv2d + BatchNorm2d + ReLU blocks and Linear classifier."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ptq_sim.models.registry import register_model


@register_model("simple_cnn")
class SimpleCNN(nn.Module):
    """Lightweight CNN with Sequential Conv+BN+ReLU blocks for PTQ verification."""

    def __init__(self, in_channels: int = 3, num_classes: int = 10, base_channels: int = 16) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_channels * 4, base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Linear(base_channels * 4, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        return self.classifier(x)
