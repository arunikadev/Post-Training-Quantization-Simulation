"""Template demonstrating how to plug in your own PyTorch model for PTQ simulation.

Option A (No edits to framework code needed - Dynamic Import):
1. Put your model class in any Python file, e.g. `my_models/custom_net.py`:
       class MyCustomNet(nn.Module):
           def __init__(self, num_classes=10): ...
2. In your YAML config or CLI:
       model:
         target: "my_models.custom_net:MyCustomNet"   # or "my_models/custom_net.py:MyCustomNet"
         kwargs:
           num_classes: 10
         checkpoint: "checkpoints/my_weights.pth"

Option B (Register in MODEL_REGISTRY):
Decorate your nn.Module with `@register_model("my_custom_net")` and import it before running.
"""

from __future__ import annotations

from torch import Tensor, nn

# Uncomment the line below when registering directly into MODEL_REGISTRY:
# from ptq_sim.models.registry import register_model
# @register_model("my_custom_net")
class CustomModelTemplate(nn.Module):
    """Minimal boilerplate for a custom user PyTorch architecture."""

    def __init__(self, in_features: int = 32, num_classes: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
