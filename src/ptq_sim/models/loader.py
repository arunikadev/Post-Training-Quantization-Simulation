"""Smart PyTorch checkpoint and state_dict loader & inspector."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from torch import nn

logger = logging.getLogger("ptq_sim")

COMMON_STATE_DICT_KEYS = (
    "model_state_dict",
    "state_dict",
    "model",
    "net",
    "weights",
    "ema_state_dict",
)
COMMON_PREFIXES = ("module.", "_orig_mod.")


def _safe_torch_load(path: Path, map_location: str | torch.device = "cpu") -> Any:
    """Load a PyTorch checkpoint safely, falling back when full objects are pickled."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception:
        logger.warning(
            "Falling back to torch.load(..., weights_only=False) for '%s'. "
            "Only load checkpoints from trusted sources.",
            path,
        )
        return torch.load(path, map_location=map_location, weights_only=False)


def _extract_state_dict(
    raw_obj: Any,
    state_dict_key: str | None = None,
) -> tuple[dict[str, Any], str]:
    """Unwrap raw state_dict, wrapped dicts, or pickled nn.Module checkpoints."""
    if isinstance(raw_obj, nn.Module):
        return raw_obj.state_dict(), "nn.Module"

    if not isinstance(raw_obj, dict):
        raise TypeError(
            f"Unsupported checkpoint object type: {type(raw_obj).__name__}. "
            "Expected state_dict dict, wrapped dict, or nn.Module."
        )

    if state_dict_key is not None:
        if state_dict_key not in raw_obj:
            raise KeyError(
                f"Specified state_dict_key '{state_dict_key}' not found. "
                f"Available keys: {list(raw_obj.keys())}."
            )
        candidate = raw_obj[state_dict_key]
        if isinstance(candidate, nn.Module):
            return candidate.state_dict(), f"wrapped[{state_dict_key}:nn.Module]"
        if isinstance(candidate, dict):
            return candidate, f"wrapped[{state_dict_key}]"
        raise TypeError(f"Value under '{state_dict_key}' is {type(candidate).__name__}, not dict/nn.Module.")

    for key in COMMON_STATE_DICT_KEYS:
        if key in raw_obj:
            candidate = raw_obj[key]
            if isinstance(candidate, nn.Module):
                return candidate.state_dict(), f"wrapped[{key}:nn.Module]"
            if isinstance(candidate, dict) and any(isinstance(v, torch.Tensor) for v in candidate.values()):
                return candidate, f"wrapped[{key}]"

    # Otherwise assume raw_obj itself is a flat state_dict.
    return raw_obj, "raw_state_dict"


def _strip_known_prefixes(
    state_dict: dict[str, Any],
    model_keys: set[str],
) -> tuple[dict[str, Any], str | None]:
    """Strip common wrapper prefixes ('module.', '_orig_mod.') if the target model lacks them."""
    sd_keys = list(state_dict.keys())
    if not sd_keys:
        return state_dict, None

    for prefix in COMMON_PREFIXES:
        prefixed_count = sum(1 for k in sd_keys if k.startswith(prefix))
        model_has_prefix = any(k.startswith(prefix) for k in model_keys)
        if prefixed_count == len(sd_keys) and not model_has_prefix:
            cleaned = {k[len(prefix) :]: v for k, v in state_dict.items()}
            return cleaned, prefix
    return state_dict, None


def load_checkpoint_into_model(
    model: nn.Module,
    checkpoint_path: str | Path,
    state_dict_key: str | None = None,
    strict: bool = True,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load weights from a .pt/.pth checkpoint or state_dict into an existing nn.Module."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: '{path}'.")

    raw_obj = _safe_torch_load(path, map_location=map_location)
    state_dict, detected_format = _extract_state_dict(raw_obj, state_dict_key=state_dict_key)

    model_keys = set(model.state_dict().keys())
    state_dict, stripped_prefix = _strip_known_prefixes(state_dict, model_keys)

    incompatible = model.load_state_dict(state_dict, strict=strict)
    report = {
        "checkpoint_path": str(path),
        "detected_format": detected_format,
        "stripped_prefix": stripped_prefix,
        "strict": strict,
        "num_tensors_in_checkpoint": len(state_dict),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }
    logger.info(
        "Loaded checkpoint '%s' (format=%s, stripped_prefix=%s, missing=%d, unexpected=%d).",
        path.name,
        detected_format,
        stripped_prefix,
        len(incompatible.missing_keys),
        len(incompatible.unexpected_keys),
    )
    return report


def inspect_checkpoint(
    checkpoint_path: str | Path,
    state_dict_key: str | None = None,
    max_preview: int = 25,
) -> dict[str, Any]:
    """Inspect a .pt/.pth file and summarize its keys, shapes, dtypes, and FP32 footprint."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: '{path}'.")

    raw_obj = _safe_torch_load(path, map_location="cpu")
    top_keys = list(raw_obj.keys()) if isinstance(raw_obj, dict) else []
    state_dict, detected_format = _extract_state_dict(raw_obj, state_dict_key=state_dict_key)

    tensor_entries: list[dict[str, Any]] = []
    total_params = 0
    total_bytes = 0
    dtype_counts: dict[str, int] = {}

    for key, val in state_dict.items():
        if isinstance(val, torch.Tensor):
            numel = int(val.numel())
            nbytes = int(numel * val.element_size())
            total_params += numel
            total_bytes += nbytes
            dt = str(val.dtype)
            dtype_counts[dt] = dtype_counts.get(dt, 0) + 1
            if len(tensor_entries) < max_preview:
                tensor_entries.append(
                    {
                        "key": key,
                        "shape": list(val.shape),
                        "dtype": dt,
                        "numel": numel,
                    }
                )

    return {
        "checkpoint_path": str(path),
        "file_size_mb": round(path.stat().st_size / (1024.0 * 1024.0), 4),
        "detected_format": detected_format,
        "top_level_keys": top_keys[:30],
        "num_state_dict_entries": len(state_dict),
        "total_tensor_elements": total_params,
        "tensor_memory_mb": round(total_bytes / (1024.0 * 1024.0), 4),
        "fp32_equivalent_mb": round((total_params * 4) / (1024.0 * 1024.0), 4),
        "dtype_counts": dtype_counts,
        "preview_tensors": tensor_entries,
    }
