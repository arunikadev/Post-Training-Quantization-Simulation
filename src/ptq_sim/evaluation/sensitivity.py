"""Single-layer quantization sensitivity analysis for mixed-precision guidance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import pandas as pd
import torch
from torch import nn

from ptq_sim.evaluation.evaluator import CustomEvalFn, ModelEvaluator
from ptq_sim.quantization.preparer import ModelQuantizer


@dataclass
class SensitivityResult:
    """Ranked single-layer sensitivity measurements."""

    records: list[dict[str, Any]]
    dataframe: pd.DataFrame
    most_sensitive_layers: list[str]
    primary_metric_name: str
    baseline_value: float


class SensitivityAnalyzer:
    """Evaluates metric degradation when quantizing one layer at a time."""

    @staticmethod
    def analyze(
        quant_model: nn.Module,
        fp32_metrics: dict[str, Any],
        eval_loader: Iterable[Any],
        evaluator: ModelEvaluator,
        device: torch.device,
        max_batches: int | None = 8,
        eval_fn: CustomEvalFn | None = None,
    ) -> SensitivityResult:
        """Measure single-layer sensitivity against the FP32 baseline."""
        layers = ModelQuantizer.get_quantized_layers(quant_model)
        primary_name = str(fp32_metrics["primary_metric_name"])
        higher_is_better = bool(fp32_metrics.get("higher_is_better", True))

        saved_states = ModelQuantizer.capture_fake_quant_states(quant_model)

        try:
            # Evaluate float reference using the same max_batches slice for apples-to-apples delta.
            ModelQuantizer.disable_fake_quant(quant_model)
            ref_metrics = evaluator.evaluate(
                quant_model,
                eval_loader,
                device=device,
                max_batches=max_batches,
                eval_fn=eval_fn,
            )
            baseline_val = float(ref_metrics["primary_metric"])
            baseline_loss = ref_metrics.get("loss")

            records: list[dict[str, Any]] = []
            for layer_name, layer_mod in layers.items():
                ModelQuantizer.set_only_layer_quant_enabled(quant_model, layer_name)
                layer_metrics = evaluator.evaluate(
                    quant_model,
                    eval_loader,
                    device=device,
                    max_batches=max_batches,
                    eval_fn=eval_fn,
                )
                layer_val = float(layer_metrics["primary_metric"])
                degradation = (baseline_val - layer_val) if higher_is_better else (layer_val - baseline_val)
                loss_val = layer_metrics.get("loss")
                loss_delta = (
                    (float(loss_val) - float(baseline_loss))
                    if (loss_val is not None and baseline_loss is not None)
                    else None
                )
                in_q = layer_mod.input_quantizer
                records.append(
                    {
                        "layer_name": layer_name,
                        "layer_type": layer_mod.__class__.__name__,
                        "weight_bits": layer_mod.weight_quantizer.cfg.bit_width,
                        "act_bits": in_q.cfg.bit_width if (in_q is not None and in_q.cfg.enabled) else 32,
                        "baseline_metric": baseline_val,
                        "layer_quant_metric": layer_val,
                        "degradation": degradation,
                        "loss_delta": loss_delta,
                    }
                )
        finally:
            ModelQuantizer.restore_fake_quant_states(quant_model, saved_states)

        records.sort(key=lambda r: r["degradation"], reverse=True)
        for rank, rec in enumerate(records, start=1):
            rec["sensitivity_rank"] = rank

        df = pd.DataFrame(records)
        top_layers = [r["layer_name"] for r in records[:5]]
        return SensitivityResult(
            records=records,
            dataframe=df,
            most_sensitive_layers=top_layers,
            primary_metric_name=primary_name,
            baseline_value=baseline_val,
        )
