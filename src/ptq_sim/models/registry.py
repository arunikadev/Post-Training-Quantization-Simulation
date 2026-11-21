"""Model registry, torchvision adapter, and dynamic Python target loader."""

from __future__ import annotations

from typing import Any, Callable

from torch import nn

from ptq_sim.config.schema import ModelConfig
from ptq_sim.models.loader import load_checkpoint_into_model
from ptq_sim.utils.io import import_from_target

MODEL_REGISTRY: dict[str, Callable[..., nn.Module]] = {}


def register_model(name: str) -> Callable[[Callable[..., nn.Module]], Callable[..., nn.Module]]:
    """Decorator to register a model factory or nn.Module class under a name."""

    def _decorator(obj: Callable[..., nn.Module]) -> Callable[..., nn.Module]:
        MODEL_REGISTRY[name.lower()] = obj
        return obj

    return _decorator


def list_registered_models() -> list[str]:
    """Return sorted list of built-in/registered model names."""
    return sorted(MODEL_REGISTRY.keys())


def _build_torchvision_model(arch_name: str, kwargs: dict[str, Any]) -> nn.Module:
    try:
        import torchvision.models as tv_models
    except ImportError as exc:
        raise ImportError("torchvision is required to load 'torchvision:<arch>' models.") from exc

    if not hasattr(tv_models, arch_name):
        raise ValueError(f"torchvision.models has no architecture named '{arch_name}'.")
    factory = getattr(tv_models, arch_name)
    return factory(**kwargs)


def build_model(model_cfg: ModelConfig) -> tuple[nn.Module, dict[str, Any] | None]:
    """Instantiate a model from target, torchvision, or registry and optionally load checkpoint."""
    kwargs = dict(model_cfg.kwargs or {})

    if model_cfg.target:
        factory = import_from_target(model_cfg.target)
        model = factory(**kwargs)
    elif model_cfg.name.lower().startswith("torchvision:"):
        arch = model_cfg.name.split(":", 1)[1]
        model = _build_torchvision_model(arch, kwargs)
    else:
        key = model_cfg.name.lower()
        if key not in MODEL_REGISTRY:
            raise ValueError(
                f"Model '{model_cfg.name}' is not registered. Available: {list_registered_models()}. "
                "You can also pass target='my_module:MyModel' or 'torchvision:resnet18'."
            )
        model = MODEL_REGISTRY[key](**kwargs)

    if not isinstance(model, nn.Module):
        raise TypeError(f"Model factory did not return an nn.Module (got {type(model).__name__}).")

    load_report: dict[str, Any] | None = None
    if model_cfg.checkpoint:
        load_report = load_checkpoint_into_model(
            model=model,
            checkpoint_path=model_cfg.checkpoint,
            state_dict_key=model_cfg.state_dict_key,
            strict=model_cfg.strict_load,
            map_location="cpu",
        )

    model.eval()
    return model, load_report
