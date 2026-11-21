"""Dataset registry and built-in DataLoader adapters."""

# Import modules so @register_dataset decorators run automatically.
from ptq_sim.data import synthetic, vision  # noqa: F401
from ptq_sim.data.registry import (
    DATASET_REGISTRY,
    build_dataloaders,
    list_registered_datasets,
    register_dataset,
)
from ptq_sim.data.synthetic import build_synthetic_train_loader

__all__ = [
    "DATASET_REGISTRY",
    "build_dataloaders",
    "build_synthetic_train_loader",
    "list_registered_datasets",
    "register_dataset",
]
