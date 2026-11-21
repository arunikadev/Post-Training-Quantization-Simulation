"""Model footprint, compression ratio, qparam overhead, and BitOps profiler."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from ptq_sim.quantization.calibrator import extract_model_inputs
from ptq_sim.quantization.modules import QuantConv2d, QuantLinear
from ptq_sim.quantization.preparer import ModelQuantizer


def _estimate_conv2d_macs(mod: QuantConv2d, out_tensor: Tensor) -> int:
    # out_tensor shape: (B, C_out, H_out, W_out); compute per-sample MACs.
    if out_tensor.ndim != 4:
        return 0
    _, c_out, h_out, w_out = out_tensor.shape
    k_h, k_w = mod.kernel_size
    c_in_per_group = mod.in_channels // max(1, mod.groups)
    return int(c_out * h_out * w_out * c_in_per_group * k_h * k_w)


def _estimate_linear_macs(mod: QuantLinear, in_tensor: Tensor) -> int:
    # Compute per-sample MACs across sequence/spatial dims excluding batch dim 0.
    if in_tensor.ndim < 2:
        return int(mod.in_features * mod.out_features)
    tokens_per_sample = 1
    for dim in in_tensor.shape[1:-1]:
        tokens_per_sample *= int(dim)
    return int(tokens_per_sample * mod.in_features * mod.out_features)


class ModelSizeProfiler:
    """Profiles theoretical FP32 vs quantized memory footprint and BitOps."""

    @staticmethod
    def profile(
        quant_model: nn.Module,
        sample_batch: Any = None,
        device: torch.device | None = None,
    ) -> dict[str, Any]:
        """Compute size (MB), compression ratio, effective bits/param, and optional BitOps."""
        quant_layers = ModelQuantizer.get_quantized_layers(quant_model)

        total_params = 0
        quantized_weight_params = 0
        non_quantized_params = 0

        fp32_bits = 0
        quant_payload_bits = 0
        qparam_overhead_bits = 0
        non_quant_bits = 0

        # Track weight parameter ids belonging to quantized layers.
        quant_weight_param_ids: set[int] = set()
        for layer in quant_layers.values():
            w_param = layer.weight
            quant_weight_param_ids.add(id(w_param))
            numel = int(w_param.numel())
            total_params += numel
            fp32_bits += numel * 32

            if layer.weight_quantizer.cfg.enabled:
                quantized_weight_params += numel
                w_bits = int(layer.weight_quantizer.cfg.bit_width)
                quant_payload_bits += numel * w_bits

                num_scales = int(layer.weight_quantizer.scale.numel())
                # Scale stored in FP32 (32b); zero_point stored as int8/int16 if asymmetric else 0.
                zp_bits_each = 0 if layer.weight_quantizer.cfg.symmetric else (8 if w_bits <= 8 else 16)
                qparam_overhead_bits += num_scales * (32 + zp_bits_each)
            else:
                non_quantized_params += numel
                non_quant_bits += numel * 32

        for param in quant_model.parameters():
            if id(param) in quant_weight_param_ids:
                continue
            numel = int(param.numel())
            total_params += numel
            non_quantized_params += numel
            fp32_bits += numel * 32
            non_quant_bits += numel * 32

        total_quant_bits = quant_payload_bits + qparam_overhead_bits + non_quant_bits
        bits_to_mb = 1.0 / (8.0 * 1024.0 * 1024.0)

        fp32_mb = fp32_bits * bits_to_mb
        quant_mb = total_quant_bits * bits_to_mb
        compression_ratio = (fp32_bits / total_quant_bits) if total_quant_bits > 0 else 1.0
        avg_bits = (total_quant_bits / total_params) if total_params > 0 else 32.0

        summary: dict[str, Any] = {
            "total_params": total_params,
            "quantized_params": quantized_weight_params,
            "non_quantized_params": non_quantized_params,
            "quantized_param_ratio": round(quantized_weight_params / max(1, total_params), 4),
            "fp32_size_mb": round(fp32_mb, 6),
            "quantized_size_mb": round(quant_mb, 6),
            "weight_payload_mb": round(quant_payload_bits * bits_to_mb, 6),
            "qparam_overhead_mb": round(qparam_overhead_bits * bits_to_mb, 6),
            "non_quantized_size_mb": round(non_quant_bits * bits_to_mb, 6),
            "compression_ratio": round(compression_ratio, 4),
            "size_reduction_pct": round((1.0 - (quant_mb / max(fp32_mb, 1e-12))) * 100.0, 2),
            "avg_effective_bits_per_param": round(avg_bits, 3),
        }

        if sample_batch is not None and quant_layers:
            bitops_info = ModelSizeProfiler._profile_macs_and_bitops(
                quant_model, quant_layers, sample_batch, device or torch.device("cpu")
            )
            summary.update(bitops_info)

        return summary

    @staticmethod
    def _profile_macs_and_bitops(
        quant_model: nn.Module,
        quant_layers: dict[str, Any],
        sample_batch: Any,
        device: torch.device,
    ) -> dict[str, Any]:
        total_macs = 0
        fp32_bitops = 0
        quant_bitops = 0
        handles: list[Any] = []

        for layer in quant_layers.values():
            def _make_hook(mod: Any):
                def _hook(_m: nn.Module, inputs: tuple[Tensor, ...], output: Tensor) -> None:
                    nonlocal total_macs, fp32_bitops, quant_bitops
                    if isinstance(mod, QuantConv2d):
                        macs = _estimate_conv2d_macs(mod, output)
                    elif isinstance(mod, QuantLinear) and inputs:
                        macs = _estimate_linear_macs(mod, inputs[0])
                    else:
                        macs = 0
                    w_bits = mod.weight_quantizer.cfg.bit_width if mod.weight_quantizer.cfg.enabled else 32
                    in_q = mod.input_quantizer
                    a_bits = in_q.cfg.bit_width if (in_q is not None and in_q.cfg.enabled) else 32
                    total_macs += macs
                    fp32_bitops += macs * 32 * 32
                    quant_bitops += macs * w_bits * a_bits
                return _hook
            handles.append(layer.register_forward_hook(_make_hook(layer)))

        try:
            quant_model.eval()
            quant_model.to(device)
            args, kwargs = extract_model_inputs(sample_batch, device)
            # Run on a single sample slice if tensor batch to keep MAC count strictly per-sample.
            if args and isinstance(args[0], torch.Tensor) and args[0].shape[0] > 1:
                args = (args[0][:1],) + args[1:]
            elif kwargs:
                kwargs = {
                    k: (v[:1] if isinstance(v, torch.Tensor) and v.ndim >= 1 and v.shape[0] > 1 else v)
                    for k, v in kwargs.items()
                }
            with torch.inference_mode():
                quant_model(*args, **kwargs)
        finally:
            for h in handles:
                h.remove()

        bitops_reduction = (fp32_bitops / quant_bitops) if quant_bitops > 0 else 1.0
        return {
            "macs_per_sample": total_macs,
            "fp32_gbitops": round(fp32_bitops / 1e9, 6),
            "quant_gbitops": round(quant_bitops / 1e9, 6),
            "bitops_reduction_ratio": round(bitops_reduction, 4),
        }
