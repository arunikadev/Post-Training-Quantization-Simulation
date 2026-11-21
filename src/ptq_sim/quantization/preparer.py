"""Model preparation for PTQ: BN folding, layer swapping, selective overrides, and state toggles."""

from __future__ import annotations

import copy
import re
from typing import Any

from torch import nn

from ptq_sim.config.schema import QuantizationConfig
from ptq_sim.quantization.bn_folding import fold_batch_norms
from ptq_sim.quantization.modules import QuantBaseMixin, QuantConv2d, QuantLinear

SUPPORTED_QUANT_MODULES: dict[str, tuple[type[nn.Module], type[QuantBaseMixin]]] = {
    "Conv2d": (nn.Conv2d, QuantConv2d),
    "Linear": (nn.Linear, QuantLinear),
}


def _matches_any_pattern(name: str, patterns: list[str]) -> bool:
    for pat in patterns:
        if name == pat or re.fullmatch(pat, name) or re.search(pat, name):
            return True
    return False


class ModelQuantizer:
    """Prepares floating-point PyTorch models for PTQ simulation and manages quantizer states."""

    @staticmethod
    def prepare(
        model: nn.Module,
        quant_cfg: QuantizationConfig,
    ) -> tuple[nn.Module, dict[str, Any]]:
        """Deep-copy model, optionally fold BatchNorms, and replace target layers with Quant* wrappers."""
        if quant_cfg.fold_bn:
            prepared_model, folded_pairs = fold_batch_norms(model)
        else:
            prepared_model = copy.deepcopy(model)
            prepared_model.eval()
            folded_pairs = []

        target_map: dict[type[nn.Module], type[QuantBaseMixin]] = {}
        for type_name in quant_cfg.quantize_types:
            if type_name not in SUPPORTED_QUANT_MODULES:
                raise ValueError(
                    f"Unsupported quantize_type '{type_name}'. "
                    f"Supported: {sorted(SUPPORTED_QUANT_MODULES.keys())}."
                )
            float_cls, quant_cls = SUPPORTED_QUANT_MODULES[type_name]
            target_map[float_cls] = quant_cls

        quantized_layers: list[str] = []
        skipped_layers: list[str] = []
        override_applied: list[str] = []

        def _replace_recursive(parent: nn.Module, prefix: str = "") -> None:
            for child_name, child in list(parent.named_children()):
                full_name = f"{prefix}.{child_name}" if prefix else child_name
                if isinstance(child, (QuantConv2d, QuantLinear)):
                    continue

                matched_quant_cls = None
                for float_cls, quant_cls in target_map.items():
                    if isinstance(child, float_cls):
                        matched_quant_cls = quant_cls
                        break

                if matched_quant_cls is not None:
                    if _matches_any_pattern(full_name, quant_cfg.skip_layers):
                        skipped_layers.append(full_name)
                        continue

                    enabled, w_cfg, a_cfg = quant_cfg.resolve_layer_quant_configs(full_name)
                    if not enabled:
                        skipped_layers.append(full_name)
                        continue

                    if (
                        w_cfg != quant_cfg.weight
                        or a_cfg != quant_cfg.activation
                    ):
                        override_applied.append(full_name)

                    qmod = matched_quant_cls.from_float(
                        child,
                        quant_cfg=quant_cfg,
                        weight_cfg=w_cfg,
                        act_cfg=a_cfg,
                    )
                    if isinstance(parent, nn.Sequential) and child_name.isdigit():
                        parent[int(child_name)] = qmod
                    else:
                        setattr(parent, child_name, qmod)
                    quantized_layers.append(full_name)
                else:
                    _replace_recursive(child, full_name)

        _replace_recursive(prepared_model)

        prep_info = {
            "folded_bn_pairs": folded_pairs,
            "quantized_layers": quantized_layers,
            "skipped_layers": skipped_layers,
            "override_applied": override_applied,
        }
        return prepared_model, prep_info

    @staticmethod
    def get_quantized_layers(model: nn.Module) -> dict[str, QuantBaseMixin]:
        """Return an ordered dict of {layer_name: quantized_module} in the model."""
        layers: dict[str, QuantBaseMixin] = {}
        for name, mod in model.named_modules():
            if isinstance(mod, (QuantConv2d, QuantLinear)):
                layers[name] = mod
        return layers

    @staticmethod
    def enable_calibration(model: nn.Module) -> None:
        """Put all quantized layers into activation-observation mode."""
        for layer in ModelQuantizer.get_quantized_layers(model).values():
            layer.enable_calibration()

    @staticmethod
    def freeze_quantization(model: nn.Module) -> None:
        """Compute static activation qparams, disable observers, and enable fake-quantization."""
        for layer in ModelQuantizer.get_quantized_layers(model).values():
            layer.freeze_quantization()

    @staticmethod
    def enable_fake_quant(model: nn.Module) -> None:
        """Enable fake quantization on all wrapped layers."""
        for layer in ModelQuantizer.get_quantized_layers(model).values():
            layer.set_fake_quant_enabled(True)

    @staticmethod
    def disable_fake_quant(model: nn.Module) -> None:
        """Bypass fake quantization on all wrapped layers (emulates unquantized float pass)."""
        for layer in ModelQuantizer.get_quantized_layers(model).values():
            layer.set_fake_quant_enabled(False)

    @staticmethod
    def set_only_layer_quant_enabled(model: nn.Module, target_layer_name: str) -> None:
        """Enable fake quantization exclusively on target_layer_name for sensitivity analysis."""
        layers = ModelQuantizer.get_quantized_layers(model)
        if target_layer_name not in layers:
            raise KeyError(f"Layer '{target_layer_name}' not found among quantized layers.")
        for name, layer in layers.items():
            layer.set_fake_quant_enabled(name == target_layer_name)

    @staticmethod
    def capture_fake_quant_states(model: nn.Module) -> dict[str, tuple[bool, bool | None, bool | None]]:
        """Snapshot current fake_quant_enabled flags per layer so states can be restored."""
        states: dict[str, tuple[bool, bool | None, bool | None]] = {}
        for name, layer in ModelQuantizer.get_quantized_layers(model).items():
            w_state = layer.weight_quantizer.fake_quant_enabled
            in_state = layer.input_quantizer.fake_quant_enabled if layer.input_quantizer is not None else None
            out_state = layer.output_quantizer.fake_quant_enabled if layer.output_quantizer is not None else None
            states[name] = (w_state, in_state, out_state)
        return states

    @staticmethod
    def restore_fake_quant_states(
        model: nn.Module,
        states: dict[str, tuple[bool, bool | None, bool | None]],
    ) -> None:
        """Restore previously captured fake_quant_enabled flags."""
        for name, layer in ModelQuantizer.get_quantized_layers(model).items():
            if name not in states:
                continue
            w_state, in_state, out_state = states[name]
            layer.weight_quantizer.fake_quant_enabled = w_state
            if layer.input_quantizer is not None and in_state is not None:
                layer.input_quantizer.fake_quant_enabled = in_state
            if layer.output_quantizer is not None and out_state is not None:
                layer.output_quantizer.fake_quant_enabled = out_state
