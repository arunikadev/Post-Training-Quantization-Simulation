"""Task performance and tensor quantization fidelity metrics."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def compute_mse(ref: Tensor, quant: Tensor) -> float:
    """Compute Mean Squared Error between reference and quantized tensors."""
    diff = ref.detach().to(torch.float64) - quant.detach().to(torch.float64)
    return float(torch.mean(diff * diff).item())


def compute_rmse(ref: Tensor, quant: Tensor) -> float:
    """Compute Root Mean Squared Error."""
    return math.sqrt(compute_mse(ref, quant))


def compute_mae(ref: Tensor, quant: Tensor) -> float:
    """Compute Mean Absolute Error."""
    diff = (ref.detach().to(torch.float64) - quant.detach().to(torch.float64)).abs()
    return float(torch.mean(diff).item())


def compute_max_abs_error(ref: Tensor, quant: Tensor) -> float:
    """Compute maximum absolute deviation between reference and quantized tensors."""
    if ref.numel() == 0:
        return 0.0
    diff = (ref.detach().to(torch.float64) - quant.detach().to(torch.float64)).abs()
    return float(diff.max().item())


def compute_sqnr_db(ref: Tensor, quant: Tensor, eps: float = 1e-12) -> float:
    """Compute Signal-to-Quantization-Noise Ratio in decibels (dB)."""
    ref_d = ref.detach().to(torch.float64)
    diff_d = ref_d - quant.detach().to(torch.float64)
    signal_power = float(torch.sum(ref_d * ref_d).item())
    noise_power = float(torch.sum(diff_d * diff_d).item())
    if signal_power <= eps and noise_power <= eps:
        return 120.0
    if noise_power <= eps:
        return 120.0
    return 10.0 * math.log10(max(signal_power, eps) / max(noise_power, eps))


def compute_cosine_similarity(ref: Tensor, quant: Tensor, eps: float = 1e-12) -> float:
    """Compute flattened cosine similarity between reference and quantized tensors."""
    ref_flat = ref.detach().to(torch.float64).reshape(-1)
    q_flat = quant.detach().to(torch.float64).reshape(-1)
    norm_r = float(torch.linalg.vector_norm(ref_flat).item())
    norm_q = float(torch.linalg.vector_norm(q_flat).item())
    if norm_r <= eps and norm_q <= eps:
        return 1.0
    if norm_r <= eps or norm_q <= eps:
        return 0.0
    dot = float(torch.dot(ref_flat, q_flat).item())
    return max(-1.0, min(1.0, dot / (norm_r * norm_q)))


def compute_tensor_error_metrics(ref: Tensor, quant: Tensor) -> dict[str, float]:
    """Compute all tensor fidelity metrics in one pass."""
    ref_d = ref.detach().to(torch.float64).reshape(-1)
    q_d = quant.detach().to(torch.float64).reshape(-1)
    diff = ref_d - q_d
    n = max(1, ref_d.numel())

    mse = float(torch.sum(diff * diff).item() / n)
    rmse = math.sqrt(mse)
    mae = float(torch.sum(diff.abs()).item() / n)
    max_abs = float(diff.abs().max().item()) if ref_d.numel() > 0 else 0.0

    sig_pow = float(torch.sum(ref_d * ref_d).item())
    noise_pow = float(torch.sum(diff * diff).item())
    if noise_pow <= 1e-12:
        sqnr_db = 120.0
    else:
        sqnr_db = 10.0 * math.log10(max(sig_pow, 1e-12) / noise_pow)

    norm_r = math.sqrt(sig_pow)
    norm_q = float(torch.linalg.vector_norm(q_d).item())
    if norm_r <= 1e-12 and norm_q <= 1e-12:
        cos_sim = 1.0
    elif norm_r <= 1e-12 or norm_q <= 1e-12:
        cos_sim = 0.0
    else:
        cos_sim = max(-1.0, min(1.0, float(torch.dot(ref_d, q_d).item()) / (norm_r * norm_q)))

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "max_abs_err": max_abs,
        "sqnr_db": sqnr_db,
        "cosine_sim": cos_sim,
    }


def compute_topk_correct(logits: Tensor, targets: Tensor, topk: tuple[int, ...] = (1, 5)) -> dict[int, int]:
    """Count top-k correct predictions for classification logits."""
    num_classes = logits.shape[-1]
    valid_ks = [k for k in topk if k <= num_classes]
    if not valid_ks:
        return {}
    max_k = max(valid_ks)
    _, pred = logits.topk(max_k, dim=-1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(targets.reshape(1, -1).expand_as(pred))
    return {k: int(correct[:k].reshape(-1).float().sum().item()) for k in valid_ks}
