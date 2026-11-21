"""Device resolution and hardware inspection helpers."""

from __future__ import annotations

from typing import Any

import torch


def resolve_device(device: str = "auto") -> torch.device:
    """Resolve a device specification ('auto', 'cuda', 'cuda:0', 'cpu')."""
    normalized = str(device).strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def get_device_info(device: torch.device) -> dict[str, Any]:
    """Return metadata describing the active compute device."""
    info: dict[str, Any] = {
        "device": str(device),
        "type": device.type,
        "cuda_available": torch.cuda.is_available(),
    }
    if device.type == "cuda" and torch.cuda.is_available():
        idx = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        info["gpu_name"] = props.name
        info["total_vram_mb"] = round(props.total_memory / (1024.0 * 1024.0), 2)
        info["cuda_version"] = torch.version.cuda
    return info
