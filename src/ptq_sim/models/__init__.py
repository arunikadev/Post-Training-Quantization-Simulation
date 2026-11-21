"""Model registry, smart checkpoint loader, and built-in example architectures."""

# Import examples so their @register_model decorators execute automatically.
from ptq_sim.models import examples  # noqa: F401
from ptq_sim.models.loader import inspect_checkpoint, load_checkpoint_into_model
from ptq_sim.models.registry import (
    MODEL_REGISTRY,
    build_model,
    list_registered_models,
    register_model,
)

__all__ = [
    "MODEL_REGISTRY",
    "build_model",
    "inspect_checkpoint",
    "list_registered_models",
    "load_checkpoint_into_model",
    "register_model",
]
