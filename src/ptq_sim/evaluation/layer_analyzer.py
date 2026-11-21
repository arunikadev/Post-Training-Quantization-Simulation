"""Layer-wise weight and activation quantization error analyzer."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn

from ptq_sim.evaluation.metrics import compute_tensor_error_metrics
from ptq_sim.quantization.calibrator import extract_model_inputs
from ptq_sim.quantization.preparer import ModelQuantizer


@dataclass
class LayerAnalysisResult:
    """Container for per-layer error metrics and subsampled tensor distributions."""

    records: list[dict[str, Any]]
    dataframe: pd.DataFrame
    weight_samples: dict[str, dict[str, Any]]
    activation_samples: dict[str, dict[str, Any]]
    summary: dict[str, Any]


def _subsample_np(tensor: Tensor, max_elements: int) -> np.ndarray:
    flat = tensor.detach().cpu().to(torch.float32).reshape(-1)
    n = flat.numel()
    if n > max_elements:
        stride = max(1, n // max_elements)
        flat = flat[::stride][:max_elements]
    return flat.numpy()


class _StreamingTensorStats:
    """Accumulates exact tensor error metrics across multiple batches without storing full activations."""

    def __init__(self) -> None:
        self.count = 0
        self.signal_sq = 0.0
        self.quant_sq = 0.0
        self.noise_sq = 0.0
        self.abs_err_sum = 0.0
        self.dot_sum = 0.0
        self.max_abs_err = 0.0

    def update(self, ref: Tensor, quant: Tensor) -> None:
        r = ref.detach().cpu().to(torch.float64).reshape(-1)
        q = quant.detach().cpu().to(torch.float64).reshape(-1)
        diff = r - q
        self.count += int(r.numel())
        self.signal_sq += float(torch.sum(r * r).item())
        self.quant_sq += float(torch.sum(q * q).item())
        self.noise_sq += float(torch.sum(diff * diff).item())
        self.abs_err_sum += float(torch.sum(diff.abs()).item())
        self.dot_sum += float(torch.dot(r, q).item())
        if diff.numel() > 0:
            self.max_abs_err = max(self.max_abs_err, float(diff.abs().max().item()))

    def finalize(self) -> dict[str, float]:
        if self.count == 0:
            return {
                "mse": 0.0,
                "rmse": 0.0,
                "mae": 0.0,
                "max_abs_err": 0.0,
                "sqnr_db": 120.0,
                "cosine_sim": 1.0,
            }
        mse = self.noise_sq / self.count
        rmse = math.sqrt(mse)
        mae = self.abs_err_sum / self.count
        if self.noise_sq <= 1e-12:
            sqnr_db = 120.0
        else:
            sqnr_db = 10.0 * math.log10(max(self.signal_sq, 1e-12) / self.noise_sq)
        denom = math.sqrt(self.signal_sq) * math.sqrt(self.quant_sq)
        if self.signal_sq <= 1e-12 and self.quant_sq <= 1e-12:
            cos_sim = 1.0
        elif denom <= 1e-12:
            cos_sim = 0.0
        else:
            cos_sim = max(-1.0, min(1.0, self.dot_sum / denom))
        return {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "max_abs_err": self.max_abs_err,
            "sqnr_db": sqnr_db,
            "cosine_sim": cos_sim,
        }


class LayerWiseAnalyzer:
    """Computes weight and activation quantization errors for every wrapped layer."""

    @staticmethod
    def analyze(
        quant_model: nn.Module,
        eval_loader: Iterable[Any],
        device: torch.device,
        max_batches: int = 4,
        max_sample_elements: int = 20000,
    ) -> LayerAnalysisResult:
        """Compare unquantized vs quantized layer weights and forward activations."""
        quant_model.eval()
        quant_model.to(device)

        layers = ModelQuantizer.get_quantized_layers(quant_model)
        if not layers:
            empty_df = pd.DataFrame()
            return LayerAnalysisResult(
                records=[],
                dataframe=empty_df,
                weight_samples={},
                activation_samples={},
                summary={"num_analyzed_layers": 0},
            )

        # 1. Weight metrics & subsampled distributions.
        weight_metrics_map: dict[str, dict[str, float]] = {}
        weight_samples: dict[str, dict[str, Any]] = {}
        for name, mod in layers.items():
            fp32_w = mod.weight.detach()
            q_w = mod.get_dequantized_weight()
            w_err = compute_tensor_error_metrics(fp32_w, q_w)
            weight_metrics_map[name] = w_err
            weight_samples[name] = {
                "fp32": _subsample_np(fp32_w, max_sample_elements),
                "quant": _subsample_np(q_w, max_sample_elements),
                "sqnr_db": w_err["sqnr_db"],
                "bit_width": mod.weight_quantizer.cfg.bit_width,
                "granularity": mod.weight_quantizer.cfg.granularity,
            }

        # Cache up to max_batches so float and quant passes observe identical inputs.
        cached_batches: list[Any] = []
        for idx, batch in enumerate(eval_loader):
            if idx >= max_batches:
                break
            cached_batches.append(batch)

        ref_outputs: dict[str, list[Tensor]] = {name: [] for name in layers}
        act_stats: dict[str, _StreamingTensorStats] = {name: _StreamingTensorStats() for name in layers}
        act_samples: dict[str, dict[str, Any]] = {}
        act_float_buf: dict[str, list[np.ndarray]] = {name: [] for name in layers}
        act_quant_buf: dict[str, list[np.ndarray]] = {name: [] for name in layers}

        saved_states = ModelQuantizer.capture_fake_quant_states(quant_model)
        handles: list[Any] = []

        try:
            # Pass 1: Float outputs (fake_quant disabled).
            ModelQuantizer.disable_fake_quant(quant_model)
            for name, mod in layers.items():
                def _make_ref_hook(layer_name: str):
                    def _hook(_module: nn.Module, _inputs: Any, output: Tensor) -> None:
                        ref_outputs[layer_name].append(output.detach().cpu())
                    return _hook
                handles.append(mod.register_forward_hook(_make_ref_hook(name)))

            with torch.inference_mode():
                for batch in cached_batches:
                    args, kwargs = extract_model_inputs(batch, device)
                    quant_model(*args, **kwargs)

            for h in handles:
                h.remove()
            handles.clear()

            # Pass 2: Quantized outputs (restored enabled states).
            ModelQuantizer.restore_fake_quant_states(quant_model, saved_states)
            batch_counter = {"idx": 0}

            for name, mod in layers.items():
                def _make_quant_hook(layer_name: str):
                    def _hook(_module: nn.Module, _inputs: Any, output: Tensor) -> None:
                        b_idx = batch_counter["idx"]
                        ref_list = ref_outputs[layer_name]
                        if b_idx < len(ref_list):
                            ref_t = ref_list[b_idx]
                            q_t = output.detach().cpu()
                            act_stats[layer_name].update(ref_t, q_t)
                            # Keep bounded distribution samples from first batch(es).
                            if sum(arr.size for arr in act_float_buf[layer_name]) < max_sample_elements:
                                per_batch_cap = max(512, max_sample_elements // max(1, len(cached_batches)))
                                act_float_buf[layer_name].append(_subsample_np(ref_t, per_batch_cap))
                                act_quant_buf[layer_name].append(_subsample_np(q_t, per_batch_cap))
                    return _hook
                handles.append(mod.register_forward_hook(_make_quant_hook(name)))

            with torch.inference_mode():
                for b_idx, batch in enumerate(cached_batches):
                    batch_counter["idx"] = b_idx
                    args, kwargs = extract_model_inputs(batch, device)
                    quant_model(*args, **kwargs)

        finally:
            for h in handles:
                h.remove()
            ModelQuantizer.restore_fake_quant_states(quant_model, saved_states)

        records: list[dict[str, Any]] = []
        for name, mod in layers.items():
            w_err = weight_metrics_map[name]
            a_err = act_stats[name].finalize()
            in_q = mod.input_quantizer
            act_bits = in_q.cfg.bit_width if (in_q is not None and in_q.cfg.enabled) else 32

            if act_float_buf[name]:
                act_samples[name] = {
                    "fp32": np.concatenate(act_float_buf[name])[:max_sample_elements],
                    "quant": np.concatenate(act_quant_buf[name])[:max_sample_elements],
                    "sqnr_db": a_err["sqnr_db"],
                    "bit_width": act_bits,
                }

            records.append(
                {
                    "layer_name": name,
                    "layer_type": mod.__class__.__name__,
                    "weight_bits": mod.weight_quantizer.cfg.bit_width if mod.weight_quantizer.cfg.enabled else 32,
                    "act_bits": act_bits,
                    "weight_granularity": mod.weight_quantizer.cfg.granularity,
                    "num_weights": int(mod.weight.numel()),
                    "weight_mse": w_err["mse"],
                    "weight_rmse": w_err["rmse"],
                    "weight_mae": w_err["mae"],
                    "weight_max_abs_err": w_err["max_abs_err"],
                    "weight_sqnr_db": w_err["sqnr_db"],
                    "weight_cosine_sim": w_err["cosine_sim"],
                    "act_mse": a_err["mse"],
                    "act_rmse": a_err["rmse"],
                    "act_mae": a_err["mae"],
                    "act_max_abs_err": a_err["max_abs_err"],
                    "act_sqnr_db": a_err["sqnr_db"],
                    "act_cosine_sim": a_err["cosine_sim"],
                }
            )

        df = pd.DataFrame(records)
        summary = {
            "num_analyzed_layers": len(records),
            "mean_weight_sqnr_db": float(df["weight_sqnr_db"].mean()) if not df.empty else None,
            "min_weight_sqnr_db": float(df["weight_sqnr_db"].min()) if not df.empty else None,
            "mean_act_sqnr_db": float(df["act_sqnr_db"].mean()) if not df.empty else None,
            "min_act_sqnr_db": float(df["act_sqnr_db"].min()) if not df.empty else None,
            "worst_act_sqnr_layer": (
                str(df.loc[df["act_sqnr_db"].idxmin(), "layer_name"]) if not df.empty else None
            ),
            "worst_weight_sqnr_layer": (
                str(df.loc[df["weight_sqnr_db"].idxmin(), "layer_name"]) if not df.empty else None
            ),
        }
        return LayerAnalysisResult(
            records=records,
            dataframe=df,
            weight_samples=weight_samples,
            activation_samples=act_samples,
            summary=summary,
        )
