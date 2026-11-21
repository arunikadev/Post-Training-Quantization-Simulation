"""Dataset registry and DataLoader factory for calibration and evaluation."""

from __future__ import annotations

from typing import Callable

from torch.utils.data import DataLoader

from ptq_sim.config.schema import DataConfig
from ptq_sim.utils.io import import_from_target

DatasetBuilder = Callable[..., tuple[DataLoader, DataLoader]]
DATASET_REGISTRY: dict[str, DatasetBuilder] = {}


def register_dataset(name: str) -> Callable[[DatasetBuilder], DatasetBuilder]:
    """Decorator to register a DataLoader factory returning (calib_loader, eval_loader)."""

    def _decorator(fn: DatasetBuilder) -> DatasetBuilder:
        DATASET_REGISTRY[name.lower()] = fn
        return fn

    return _decorator


def list_registered_datasets() -> list[str]:
    """Return sorted list of registered dataset builders."""
    return sorted(DATASET_REGISTRY.keys())


def build_dataloaders(data_cfg: DataConfig, seed: int = 42) -> tuple[DataLoader, DataLoader]:
    """Build (calib_loader, eval_loader) from DataConfig."""
    kwargs = dict(data_cfg.kwargs or {})
    common_args = {
        "calib_batch_size": data_cfg.calib_batch_size,
        "eval_batch_size": data_cfg.eval_batch_size,
        "num_workers": data_cfg.num_workers,
        "seed": seed,
        **kwargs,
    }

    if data_cfg.target:
        builder = import_from_target(data_cfg.target)
        return builder(**common_args)

    key = data_cfg.name.lower()
    if key not in DATASET_REGISTRY:
        raise ValueError(
            f"Dataset '{data_cfg.name}' is not registered. Available: {list_registered_datasets()}. "
            "You can also specify data.target='my_module:build_loaders'."
        )
    return DATASET_REGISTRY[key](**common_args)
