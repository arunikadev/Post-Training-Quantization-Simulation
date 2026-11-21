"""Typed experiment configuration schema with YAML loading and dot-notation overrides."""

from __future__ import annotations

import copy
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ptq_sim.utils.io import load_yaml, save_yaml

VALID_GRANULARITIES = {"per_tensor", "per_channel", "per_group"}
VALID_OBSERVERS = {"minmax", "ema", "percentile", "mse", "kl"}
VALID_TASKS = {"classification", "regression", "language_modeling"}


@dataclass
class QuantizerConfig:
    """Configuration for a single weight or activation FakeQuantizer."""

    enabled: bool = True
    bit_width: int = 8
    symmetric: bool = True
    signed: bool = True
    narrow_range: bool = True
    granularity: str = "per_tensor"
    group_size: int = 64
    ch_axis: int = 0
    observer: str = "minmax"
    observer_kwargs: dict[str, Any] = field(default_factory=dict)
    dynamic: bool = False
    include_zero: bool = True

    def __post_init__(self) -> None:
        self.bit_width = int(self.bit_width)
        self.group_size = int(self.group_size)
        self.ch_axis = int(self.ch_axis)
        self.granularity = str(self.granularity).lower()
        self.observer = str(self.observer).lower()

        if not (2 <= self.bit_width <= 16):
            raise ValueError(f"bit_width must be in [2, 16], got {self.bit_width}.")
        if self.granularity not in VALID_GRANULARITIES:
            raise ValueError(
                f"granularity must be one of {sorted(VALID_GRANULARITIES)}, got '{self.granularity}'."
            )
        if self.group_size <= 0:
            raise ValueError(f"group_size must be > 0, got {self.group_size}.")
        if self.observer not in VALID_OBSERVERS:
            raise ValueError(
                f"observer must be one of {sorted(VALID_OBSERVERS)}, got '{self.observer}'."
            )
        if self.observer_kwargs is None:
            self.observer_kwargs = {}

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> QuantizerConfig:
        if not data:
            return cls()
        return cls(**data)

    def with_updates(self, updates: dict[str, Any] | None) -> QuantizerConfig:
        """Return a validated copy with partial dictionary updates applied."""
        if not updates:
            return copy.deepcopy(self)
        merged = asdict(self)
        for key, val in updates.items():
            if key == "observer_kwargs" and isinstance(val, dict):
                merged["observer_kwargs"] = {**merged.get("observer_kwargs", {}), **val}
            else:
                merged[key] = copy.deepcopy(val)
        return QuantizerConfig(**merged)


@dataclass
class LayerOverrideConfig:
    """Per-layer quantization override for mixed-precision or selective disabling."""

    enabled: bool | None = None
    weight: dict[str, Any] | None = None
    activation: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> LayerOverrideConfig:
        if not data:
            return cls()
        return cls(
            enabled=data.get("enabled"),
            weight=copy.deepcopy(data.get("weight")),
            activation=copy.deepcopy(data.get("activation")),
        )


def _default_weight_config() -> QuantizerConfig:
    return QuantizerConfig(
        enabled=True,
        bit_width=8,
        symmetric=True,
        signed=True,
        narrow_range=True,
        granularity="per_channel",
        group_size=64,
        ch_axis=0,
        observer="minmax",
        dynamic=False,
        include_zero=True,
    )


def _default_activation_config() -> QuantizerConfig:
    return QuantizerConfig(
        enabled=True,
        bit_width=8,
        symmetric=False,
        signed=False,
        narrow_range=False,
        granularity="per_tensor",
        group_size=64,
        ch_axis=0,
        observer="minmax",
        dynamic=False,
        include_zero=True,
    )


@dataclass
class QuantizationConfig:
    """Global PTQ configuration controlling folding, target modules, schemes, and overrides."""

    fold_bn: bool = True
    quantize_types: list[str] = field(default_factory=lambda: ["Conv2d", "Linear"])
    skip_layers: list[str] = field(default_factory=list)
    quantize_input: bool = True
    quantize_output: bool = False
    weight: QuantizerConfig = field(default_factory=_default_weight_config)
    activation: QuantizerConfig = field(default_factory=_default_activation_config)
    layer_overrides: dict[str, LayerOverrideConfig] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.weight, dict):
            self.weight = _default_weight_config().with_updates(self.weight)
        if isinstance(self.activation, dict):
            self.activation = _default_activation_config().with_updates(self.activation)
        parsed_overrides: dict[str, LayerOverrideConfig] = {}
        for name, val in (self.layer_overrides or {}).items():
            if isinstance(val, LayerOverrideConfig):
                parsed_overrides[name] = val
            elif isinstance(val, dict):
                parsed_overrides[name] = LayerOverrideConfig.from_dict(val)
            else:
                raise TypeError(f"Invalid layer_override value for '{name}': {type(val)}")
        self.layer_overrides = parsed_overrides
        self.quantize_types = list(self.quantize_types or ["Conv2d", "Linear"])
        self.skip_layers = list(self.skip_layers or [])

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> QuantizationConfig:
        if not data:
            return cls()
        raw = copy.deepcopy(data)
        weight_cfg = _default_weight_config().with_updates(raw.pop("weight", None))
        act_cfg = _default_activation_config().with_updates(raw.pop("activation", None))
        overrides = raw.pop("layer_overrides", {}) or {}
        return cls(weight=weight_cfg, activation=act_cfg, layer_overrides=overrides, **raw)

    def resolve_layer_quant_configs(
        self, layer_name: str
    ) -> tuple[bool, QuantizerConfig, QuantizerConfig]:
        """Resolve (enabled, weight_cfg, activation_cfg) for a specific layer name."""
        matched_override: LayerOverrideConfig | None = None
        for pattern, override in self.layer_overrides.items():
            if layer_name == pattern or re.fullmatch(pattern, layer_name) or re.search(pattern, layer_name):
                matched_override = override
                break

        if matched_override is None:
            return (
                True,
                copy.deepcopy(self.weight),
                copy.deepcopy(self.activation),
            )

        enabled = True if matched_override.enabled is None else bool(matched_override.enabled)
        w_cfg = self.weight.with_updates(matched_override.weight)
        a_cfg = self.activation.with_updates(matched_override.activation)
        return enabled, w_cfg, a_cfg


@dataclass
class ModelConfig:
    """Configuration for instantiating a model and loading a checkpoint/state_dict."""

    name: str = "simple_cnn"
    target: str | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)
    checkpoint: str | None = None
    state_dict_key: str | None = None
    strict_load: bool = True

    def __post_init__(self) -> None:
        if self.kwargs is None:
            self.kwargs = {}

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ModelConfig:
        if not data:
            return cls()
        return cls(**data)


@dataclass
class DataConfig:
    """Configuration for calibration and evaluation DataLoaders."""

    name: str = "synthetic_image"
    target: str | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)
    calib_batch_size: int = 32
    eval_batch_size: int = 64
    num_calib_batches: int = 16
    num_workers: int = 0

    def __post_init__(self) -> None:
        if self.kwargs is None:
            self.kwargs = {}
        self.calib_batch_size = int(self.calib_batch_size)
        self.eval_batch_size = int(self.eval_batch_size)
        self.num_calib_batches = int(self.num_calib_batches)
        self.num_workers = int(self.num_workers)
        if self.calib_batch_size <= 0 or self.eval_batch_size <= 0:
            raise ValueError("Batch sizes must be positive integers.")
        if self.num_calib_batches < 0:
            raise ValueError("num_calib_batches must be >= 0.")

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> DataConfig:
        if not data:
            return cls()
        return cls(**data)


@dataclass
class EvaluationConfig:
    """Configuration for evaluation, diagnostics, plots, and artifact export."""

    task: str = "classification"
    max_eval_batches: int | None = None
    run_layer_analysis: bool = True
    layer_analysis_batches: int = 4
    run_sensitivity: bool = False
    sensitivity_max_batches: int | None = 8
    generate_plots: bool = True
    export_artifacts: bool = True

    def __post_init__(self) -> None:
        self.task = str(self.task).lower()
        if self.task not in VALID_TASKS:
            raise ValueError(f"task must be one of {sorted(VALID_TASKS)}, got '{self.task}'.")
        if self.layer_analysis_batches <= 0:
            raise ValueError("layer_analysis_batches must be > 0.")

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> EvaluationConfig:
        if not data:
            return cls()
        return cls(**data)


def parse_override_value(raw_value: Any) -> Any:
    """Parse CLI override strings cleanly using YAML literal rules."""
    if not isinstance(raw_value, str):
        return raw_value
    stripped = raw_value.strip()
    if stripped == "":
        return ""
    try:
        return yaml.safe_load(stripped)
    except yaml.YAMLError:
        return raw_value


def _set_nested(mapping: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = [p.strip() for p in dotted_key.split(".") if p.strip()]
    if not parts:
        raise ValueError(f"Invalid override key: '{dotted_key}'")
    cursor = mapping
    for part in parts[:-1]:
        if part not in cursor or not isinstance(cursor[part], dict):
            cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value


@dataclass
class PTQExperimentConfig:
    """Top-level configuration object for a full PTQ simulation run."""

    experiment_name: str = "ptq_experiment"
    seed: int = 42
    device: str = "auto"
    output_dir: str = "outputs"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def __post_init__(self) -> None:
        if isinstance(self.model, dict):
            self.model = ModelConfig.from_dict(self.model)
        if isinstance(self.data, dict):
            self.data = DataConfig.from_dict(self.data)
        if isinstance(self.quantization, dict):
            self.quantization = QuantizationConfig.from_dict(self.quantization)
        if isinstance(self.evaluation, dict):
            self.evaluation = EvaluationConfig.from_dict(self.evaluation)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> PTQExperimentConfig:
        if not data:
            return cls()
        raw = copy.deepcopy(data)
        return cls(
            experiment_name=raw.get("experiment_name", "ptq_experiment"),
            seed=int(raw.get("seed", 42)),
            device=raw.get("device", "auto"),
            output_dir=raw.get("output_dir", "outputs"),
            model=ModelConfig.from_dict(raw.get("model")),
            data=DataConfig.from_dict(raw.get("data")),
            quantization=QuantizationConfig.from_dict(raw.get("quantization")),
            evaluation=EvaluationConfig.from_dict(raw.get("evaluation")),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> PTQExperimentConfig:
        return cls.from_dict(load_yaml(path))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_yaml(self, path: str | Path) -> Path:
        return save_yaml(self.to_dict(), path)

    def copy(self) -> PTQExperimentConfig:
        return PTQExperimentConfig.from_dict(self.to_dict())

    def apply_overrides(self, overrides: dict[str, Any]) -> PTQExperimentConfig:
        """Return a new validated config with dot-notation overrides applied."""
        raw = self.to_dict()
        for key, value in overrides.items():
            _set_nested(raw, key, parse_override_value(value))
        return PTQExperimentConfig.from_dict(raw)

    def resolve_layer_quant_configs(
        self, layer_name: str
    ) -> tuple[bool, QuantizerConfig, QuantizerConfig]:
        """Delegate layer-level quantization config resolution to QuantizationConfig."""
        return self.quantization.resolve_layer_quant_configs(layer_name)
