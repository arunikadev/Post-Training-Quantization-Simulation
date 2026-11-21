"""Forward-pass calibration engine for static activation PTQ."""

from __future__ import annotations

import logging
from typing import Any, Iterable

import torch
from torch import nn
from tqdm import tqdm

from ptq_sim.quantization.preparer import ModelQuantizer

logger = logging.getLogger("ptq_sim")


def extract_model_inputs(batch: Any, device: torch.device) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Normalize common DataLoader batch structures into (*args, **kwargs) on target device."""
    if isinstance(batch, torch.Tensor):
        return (batch.to(device),), {}
    if isinstance(batch, (list, tuple)):
        if len(batch) == 0:
            raise ValueError("Received an empty batch tuple/list.")
        first = batch[0]
        if isinstance(first, torch.Tensor):
            return (first.to(device),), {}
        if isinstance(first, dict):
            return (), {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in first.items()}
    if isinstance(batch, dict):
        # Drop common label keys if a dictionary batch contains targets alongside inputs.
        ignored = {"label", "labels", "target", "targets", "y"}
        if "x" in batch and len(batch) <= 2:
            val = batch["x"]
            return (val.to(device) if isinstance(val, torch.Tensor) else val,), {}
        if "input" in batch and len(batch) <= 2:
            val = batch["input"]
            return (val.to(device) if isinstance(val, torch.Tensor) else val,), {}
        kwargs = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
            if k not in ignored
        }
        return (), kwargs
    raise TypeError(f"Unsupported batch type for model forward pass: {type(batch).__name__}.")


class PTQCalibrator:
    """Runs activation range calibration and freezes quantization parameters."""

    @staticmethod
    def needs_activation_calibration(model: nn.Module) -> bool:
        """Return True if at least one layer uses a static enabled activation quantizer."""
        for layer in ModelQuantizer.get_quantized_layers(model).values():
            if layer.has_static_activation_quantizer():
                return True
        return False

    @staticmethod
    def calibrate(
        model: nn.Module,
        calib_loader: Iterable[Any] | None,
        device: torch.device,
        num_batches: int = 16,
        show_progress: bool = False,
    ) -> dict[str, Any]:
        """Calibrate static activation quantizers (skipped automatically for weight-only/dynamic)."""
        model.eval()
        model.to(device)

        if not PTQCalibrator.needs_activation_calibration(model) or num_batches == 0:
            ModelQuantizer.freeze_quantization(model)
            return {
                "skipped_activation_calib": True,
                "batches_run": 0,
            }

        if calib_loader is None:
            raise ValueError(
                "Static activation quantization requires a calibration DataLoader, "
                "or enable weight-only / dynamic activation mode."
            )

        ModelQuantizer.enable_calibration(model)

        batches_run = 0
        iterator: Iterable[Any] = calib_loader
        if show_progress:
            iterator = tqdm(calib_loader, total=num_batches, desc="Calibrating", leave=False)

        with torch.inference_mode():
            for batch in iterator:
                if batches_run >= num_batches:
                    break
                args, kwargs = extract_model_inputs(batch, device)
                model(*args, **kwargs)
                batches_run += 1

        ModelQuantizer.freeze_quantization(model)
        logger.info("Calibration completed across %d batch(es).", batches_run)
        return {
            "skipped_activation_calib": False,
            "batches_run": batches_run,
        }
