"""Unified evaluation engine for FP32 and quantized simulation models."""

from __future__ import annotations

import math
import time
from typing import Any, Callable, Iterable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ptq_sim.evaluation.metrics import compute_topk_correct
from ptq_sim.quantization.calibrator import extract_model_inputs

CustomEvalFn = Callable[[nn.Module, Iterable[Any], torch.device], dict[str, float]]


def _extract_inputs_and_targets(
    batch: Any,
    device: torch.device,
) -> tuple[tuple[Any, ...], dict[str, Any], Tensor | None]:
    """Extract model forward inputs and ground-truth target tensor from common batch formats."""
    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
        args, kwargs = extract_model_inputs(batch[0], device)
        target = batch[1].to(device) if isinstance(batch[1], torch.Tensor) else None
        return args, kwargs, target

    if isinstance(batch, dict):
        target = None
        for key in ("labels", "label", "targets", "target", "y"):
            if key in batch and isinstance(batch[key], torch.Tensor):
                target = batch[key].to(device)
                break
        args, kwargs = extract_model_inputs(batch, device)
        return args, kwargs, target

    args, kwargs = extract_model_inputs(batch, device)
    return args, kwargs, None


class ModelEvaluator:
    """Evaluates classification, regression, language modeling, or custom tasks."""

    def __init__(self, task: str = "classification") -> None:
        self.task = task.lower()

    def evaluate(
        self,
        model: nn.Module,
        eval_loader: Iterable[Any],
        device: torch.device,
        max_batches: int | None = None,
        eval_fn: CustomEvalFn | None = None,
    ) -> dict[str, Any]:
        """Run evaluation and return a standardized metric dictionary."""
        model.eval()
        model.to(device)

        if eval_fn is not None:
            t0 = time.perf_counter()
            custom_metrics = dict(eval_fn(model, eval_loader, device))
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            first_key = next(iter(custom_metrics.keys()))
            custom_metrics.setdefault("primary_metric_name", first_key)
            custom_metrics.setdefault("primary_metric", float(custom_metrics[first_key]))
            custom_metrics.setdefault("higher_is_better", True)
            custom_metrics.setdefault("total_eval_time_ms", elapsed_ms)
            return custom_metrics

        total_samples = 0
        total_batches = 0
        total_loss = 0.0
        total_mae = 0.0
        top1_correct = 0
        top5_correct = 0
        has_top5 = False
        total_tokens = 0
        total_time_s = 0.0

        with torch.inference_mode():
            for batch in eval_loader:
                if max_batches is not None and total_batches >= max_batches:
                    break
                args, kwargs, targets = _extract_inputs_and_targets(batch, device)
                if targets is None:
                    raise ValueError(
                        "Evaluation batch does not contain targets/labels. "
                        "Provide (x, y) tuples or a custom eval_fn."
                    )

                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                t0 = time.perf_counter()
                outputs = model(*args, **kwargs)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                total_time_s += time.perf_counter() - t0

                bsz = int(targets.shape[0])
                total_batches += 1

                if self.task == "classification":
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs
                    loss = F.cross_entropy(logits, targets, reduction="sum")
                    total_loss += float(loss.item())
                    total_samples += bsz
                    topk_hits = compute_topk_correct(logits, targets, topk=(1, 5))
                    top1_correct += topk_hits.get(1, 0)
                    if 5 in topk_hits:
                        has_top5 = True
                        top5_correct += topk_hits[5]

                elif self.task == "regression":
                    preds = outputs.logits if hasattr(outputs, "logits") else outputs
                    preds_f = preds.reshape_as(targets).to(torch.float32)
                    targets_f = targets.to(torch.float32)
                    total_loss += float(F.mse_loss(preds_f, targets_f, reduction="sum").item())
                    total_mae += float(F.l1_loss(preds_f, targets_f, reduction="sum").item())
                    total_samples += int(targets_f.numel())

                elif self.task == "language_modeling":
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs
                    vocab_size = logits.shape[-1]
                    loss = F.cross_entropy(
                        logits.reshape(-1, vocab_size),
                        targets.reshape(-1),
                        reduction="sum",
                    )
                    total_loss += float(loss.item())
                    total_tokens += int(targets.numel())
                    total_samples += bsz

                else:
                    raise ValueError(f"Unsupported task '{self.task}'.")

        if total_batches == 0:
            raise ValueError("Evaluation DataLoader yielded 0 batches.")

        avg_latency_ms = (total_time_s / total_batches) * 1000.0
        result: dict[str, Any] = {
            "task": self.task,
            "num_batches": total_batches,
            "num_samples": total_samples,
            "avg_latency_ms": round(avg_latency_ms, 4),
        }

        if self.task == "classification":
            top1 = top1_correct / max(1, total_samples)
            avg_loss = total_loss / max(1, total_samples)
            result["top1_acc"] = top1
            if has_top5:
                result["top5_acc"] = top5_correct / max(1, total_samples)
            result["loss"] = avg_loss
            result["primary_metric_name"] = "top1_acc"
            result["primary_metric"] = top1
            result["higher_is_better"] = True

        elif self.task == "regression":
            mse = total_loss / max(1, total_samples)
            rmse = math.sqrt(mse)
            mae = total_mae / max(1, total_samples)
            result["mse"] = mse
            result["rmse"] = rmse
            result["mae"] = mae
            result["primary_metric_name"] = "rmse"
            result["primary_metric"] = rmse
            result["higher_is_better"] = False

        elif self.task == "language_modeling":
            avg_nll = total_loss / max(1, total_tokens)
            ppl = math.exp(min(avg_nll, 50.0))
            result["loss"] = avg_nll
            result["perplexity"] = ppl
            result["primary_metric_name"] = "perplexity"
            result["primary_metric"] = ppl
            result["higher_is_better"] = False

        return result
