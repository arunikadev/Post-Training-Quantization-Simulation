"""Filesystem, YAML/JSON serialization, and run directory helpers."""

from __future__ import annotations

import importlib
import importlib.util
import json
import re
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def sanitize_name(name: str) -> str:
    """Sanitize an experiment or scenario name for safe directory creation."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    return cleaned.strip("_") or "experiment"


def create_run_dir(base_dir: str | Path, experiment_name: str) -> Path:
    """Create a unique timestamped run directory under base_dir."""
    root = Path(base_dir)
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = sanitize_name(experiment_name)
    candidate = root / f"{safe_name}_{stamp}"
    suffix = 1
    while candidate.exists():
        candidate = root / f"{safe_name}_{stamp}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=True)
    return candidate


def to_serializable(obj: Any) -> Any:
    """Recursively convert tensors, numpy scalars/arrays, Paths, and dataclasses to JSON/YAML types."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return to_serializable(asdict(obj))
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return obj.detach().cpu().item()
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    return obj


def save_json(data: Any, path: str | Path, indent: int = 2) -> Path:
    """Write data safely as UTF-8 JSON."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = to_serializable(data)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=indent, ensure_ascii=True)
    return out_path


def load_json(path: str | Path) -> Any:
    """Read UTF-8 JSON file."""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_yaml(data: Any, path: str | Path) -> Path:
    """Write data safely as UTF-8 YAML."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = to_serializable(data)
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=False)
    return out_path


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Read UTF-8 YAML file and return a dictionary."""
    with Path(path).open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected YAML mapping at {path}, got {type(loaded).__name__}.")
    return loaded


def import_from_target(target: str) -> Any:
    """Dynamically import a class or function from 'pkg.mod:Attr' or 'path/to/file.py:Attr'."""
    if ":" not in target:
        raise ValueError(
            f"Dynamic target '{target}' must use 'module.path:AttrName' or 'path/to/file.py:AttrName' format."
        )
    module_ref, attr_name = target.rsplit(":", 1)
    module_path = Path(module_ref)
    if module_ref.endswith(".py") or module_path.is_file():
        if not module_path.is_file():
            raise FileNotFoundError(f"Python file not found for target: '{module_path}'.")
        mod_name = f"ptq_dynamic_{module_path.stem}"
        spec = importlib.util.spec_from_file_location(mod_name, str(module_path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load module spec from '{module_path}'.")
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)
    else:
        cwd = str(Path.cwd())
        if cwd not in sys.path:
            sys.path.insert(0, cwd)
        module = importlib.import_module(module_ref)

    if not hasattr(module, attr_name):
        raise AttributeError(f"Attribute '{attr_name}' not found in '{module_ref}'.")
    return getattr(module, attr_name)
