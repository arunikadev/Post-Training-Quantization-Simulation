"""Template showing how to plug in custom calibration and evaluation DataLoaders.

Option A (Dynamic Import without modifying framework):
1. Create a function in e.g. `my_data/loaders.py`:
       def build_my_loaders(calib_batch_size=32, eval_batch_size=64, num_workers=0, seed=42, **kwargs):
           ...
           return calib_loader, eval_loader
2. Set in YAML config:
       data:
         target: "my_data.loaders:build_my_loaders"
         kwargs:
           data_dir: "/path/to/dataset"

Option B (Python API):
Pass your PyTorch DataLoaders directly to `PTQPipeline(cfg).run(model=my_model, calib_loader=..., eval_loader=...)`.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset


def build_custom_loaders_template(
    calib_batch_size: int = 32,
    eval_batch_size: int = 64,
    num_workers: int = 0,
    seed: int = 42,
    input_dim: int = 32,
    num_classes: int = 2,
) -> tuple[DataLoader, DataLoader]:
    """Minimal boilerplate returning (calib_loader, eval_loader)."""
    gen = torch.Generator().manual_seed(seed)
    calib_x = torch.randn(128, input_dim, generator=gen)
    calib_y = torch.randint(0, num_classes, (128,), generator=gen)
    eval_x = torch.randn(256, input_dim, generator=gen)
    eval_y = torch.randint(0, num_classes, (256,), generator=gen)

    calib_loader = DataLoader(TensorDataset(calib_x, calib_y), batch_size=calib_batch_size, shuffle=False)
    eval_loader = DataLoader(TensorDataset(eval_x, eval_y), batch_size=eval_batch_size, shuffle=False)
    return calib_loader, eval_loader
