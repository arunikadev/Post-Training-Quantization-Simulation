"""SimpleMLP demo model with optional Linear + BatchNorm1d folding support."""

from __future__ import annotations

from typing import Sequence

from torch import Tensor, nn

from ptq_sim.models.registry import register_model


@register_model("simple_mlp")
class SimpleMLP(nn.Module):
    """Multi-layer perceptron for tabular/vector inputs with optional BatchNorm1d."""

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dims: Sequence[int] = (128, 64),
        num_classes: int = 4,
        use_bn: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, int(h_dim), bias=not use_bn))
            if use_bn:
                layers.append(nn.BatchNorm1d(int(h_dim)))
            layers.append(nn.ReLU(inplace=True))
            prev_dim = int(h_dim)
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev_dim, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = x.flatten(1)
        return self.head(self.backbone(x))
