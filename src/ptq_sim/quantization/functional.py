"""Pure functional uniform quantization, dequantization, and group-reshaping primitives."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def get_qmin_qmax(
    bit_width: int,
    signed: bool = True,
    narrow_range: bool = True,
) -> tuple[int, int]:
    """Compute integer quantization bounds (qmin, qmax) for a given bit-width and signedness."""
    if not (2 <= bit_width <= 16):
        raise ValueError(f"bit_width must be in [2, 16], got {bit_width}.")
    if signed:
        qmax = (1 << (bit_width - 1)) - 1
        qmin = -qmax if narrow_range else -(1 << (bit_width - 1))
    else:
        qmin = 0
        qmax = (1 << bit_width) - 1
    return qmin, qmax


def compute_qparams(
    min_val: Tensor,
    max_val: Tensor,
    bit_width: int,
    symmetric: bool = True,
    signed: bool = True,
    narrow_range: bool = True,
    include_zero: bool = True,
    eps: float = 1e-8,
) -> tuple[Tensor, Tensor]:
    """Compute uniform quantization (scale, zero_point) in float32 with zero-range guards."""
    min_f = min_val.detach().to(torch.float32)
    max_f = max_val.detach().to(torch.float32)
    qmin, qmax = get_qmin_qmax(bit_width=bit_width, signed=signed, narrow_range=narrow_range)

    if symmetric:
        if signed:
            max_abs = torch.maximum(min_f.abs(), max_f.abs())
            scale = max_abs / float(qmax)
        else:
            scale = torch.clamp(max_f, min=0.0) / float(qmax)
        zero_point = torch.zeros_like(scale)
    else:
        if include_zero:
            min_f = torch.minimum(min_f, torch.zeros_like(min_f))
            max_f = torch.maximum(max_f, torch.zeros_like(max_f))

        raw_range = torch.clamp(max_f - min_f, min=0.0)
        scale = raw_range / float(qmax - qmin)
        zero_range = scale < eps
        safe_scale = torch.where(zero_range, torch.ones_like(scale), scale)
        zero_point = torch.round(float(qmin) - min_f / safe_scale)
        if include_zero:
            zero_point = torch.clamp(zero_point, float(qmin), float(qmax))

        # Preserve a valid zero_point when tensor range is degenerate/constant.
        default_zp = 0.0 if signed else float(qmin)
        zero_point = torch.where(zero_range, torch.full_like(zero_point, default_zp), zero_point)

    scale = torch.where(scale < eps, torch.ones_like(scale), scale)
    return scale, zero_point


def uniform_quantize(
    x: Tensor,
    scale: Tensor,
    zero_point: Tensor,
    qmin: int,
    qmax: int,
) -> Tensor:
    """Quantize a floating-point tensor to rounded clamped integer values (still float dtype)."""
    q = torch.round(x / scale) + zero_point
    return torch.clamp(q, float(qmin), float(qmax))


def uniform_dequantize(
    q: Tensor,
    scale: Tensor,
    zero_point: Tensor,
) -> Tensor:
    """Dequantize an integer-valued tensor back to floating-point representation."""
    return (q - zero_point) * scale


def fake_quantize_tensor(
    x: Tensor,
    scale: Tensor,
    zero_point: Tensor,
    qmin: int,
    qmax: int,
) -> Tensor:
    """Simulate quantization error via quantize -> clamp -> dequantize in float space."""
    orig_dtype = x.dtype
    x_f = x.to(torch.float32) if orig_dtype != torch.float32 else x
    scale_f = scale.to(x_f.device, dtype=torch.float32)
    zp_f = zero_point.to(x_f.device, dtype=torch.float32)
    q = uniform_quantize(x_f, scale_f, zp_f, qmin, qmax)
    dq = uniform_dequantize(q, scale_f, zp_f)
    return dq.to(orig_dtype) if orig_dtype != torch.float32 else dq


def reshape_for_group_quant(
    x: Tensor,
    group_size: int,
) -> tuple[Tensor, tuple[int, ...], int]:
    """Reshape tensor into (rows, num_groups, group_size) using edge-replication padding.

    Using edge replication rather than zero padding ensures non-divisible trailing
    groups do not have their min/max range distorted by artificial zeros.
    """
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}.")
    orig_shape = tuple(x.shape)
    if x.ndim == 0:
        x_2d = x.reshape(1, 1)
    elif x.ndim == 1:
        x_2d = x.reshape(1, -1)
    elif x.ndim == 2:
        x_2d = x
    else:
        x_2d = x.reshape(orig_shape[0], -1)

    rows, cols = x_2d.shape
    rem = cols % group_size
    pad_len = (group_size - rem) if rem != 0 else 0
    if pad_len > 0:
        # Replicate the last element so padded entries stay within existing group bounds.
        pad_values = x_2d[:, -1:].expand(rows, pad_len)
        x_2d = torch.cat([x_2d, pad_values], dim=1)

    num_groups = x_2d.shape[1] // group_size
    x_grouped = x_2d.reshape(rows, num_groups, group_size)
    return x_grouped, orig_shape, pad_len


def restore_from_group_quant(
    x_grouped: Tensor,
    orig_shape: tuple[int, ...],
    pad_len: int,
) -> Tensor:
    """Restore a group-quantized (rows, num_groups, group_size) tensor to its original shape."""
    rows = x_grouped.shape[0]
    x_2d = x_grouped.reshape(rows, -1)
    if pad_len > 0:
        x_2d = x_2d[:, :-pad_len]
    if len(orig_shape) == 0:
        return x_2d.reshape(())
    return x_2d.reshape(orig_shape)


def broadcast_qparams_for_tensor(
    scale: Tensor,
    zero_point: Tensor,
    x: Tensor,
    granularity: str,
    ch_axis: int = 0,
) -> tuple[Tensor, Tensor]:
    """Reshape scale and zero_point tensors to broadcast cleanly against x."""
    if granularity == "per_tensor" or scale.numel() == 1:
        return scale.reshape(()), zero_point.reshape(())

    if granularity == "per_channel":
        axis = ch_axis if ch_axis >= 0 else x.ndim + ch_axis
        if not (0 <= axis < x.ndim):
            raise ValueError(f"ch_axis {ch_axis} out of range for tensor with shape {tuple(x.shape)}.")
        shape = [1] * x.ndim
        shape[axis] = -1
        return scale.reshape(shape), zero_point.reshape(shape)

    if granularity == "per_group":
        return scale.unsqueeze(-1), zero_point.unsqueeze(-1)

    raise ValueError(f"Unsupported granularity '{granularity}'.")


def normalize_tensor_for_granularity(
    x: Tensor,
    granularity: str,
    ch_axis: int = 0,
    group_size: int = 64,
) -> tuple[Tensor, tuple[int, ...], int]:
    """Return a reduced view of x suitable for statistics collection per granularity."""
    if granularity == "per_tensor":
        return x.reshape(1, -1), tuple(x.shape), 0
    if granularity == "per_channel":
        axis = ch_axis if ch_axis >= 0 else x.ndim + ch_axis
        if not (0 <= axis < x.ndim):
            raise ValueError(f"ch_axis {ch_axis} out of range for tensor with shape {tuple(x.shape)}.")
        x_perm = x.movedim(axis, 0).reshape(x.shape[axis], -1)
        return x_perm, tuple(x.shape), 0
    if granularity == "per_group":
        x_grouped, orig_shape, pad_len = reshape_for_group_quant(x, group_size=group_size)
        rows, num_groups, g_size = x_grouped.shape
        return x_grouped.reshape(rows * num_groups, g_size), (rows, num_groups), pad_len
    raise ValueError(f"Unsupported granularity '{granularity}'.")


def optimal_mse_ratio_candidates(max_val_ratio_floor: float = 0.01, num_steps: int = 100) -> list[float]:
    """Generate a hybrid log+linear ratio grid to handle both mild clipping and extreme outliers."""
    n_log = max(10, num_steps // 3)
    n_lin = max(10, num_steps - n_log)
    log_start = math.log10(max(max_val_ratio_floor, 1e-4))
    log_part = [10.0 ** (log_start + i * (math.log10(0.25) - log_start) / max(1, n_log - 1)) for i in range(n_log)]
    lin_part = [0.25 + i * (1.0 - 0.25) / max(1, n_lin - 1) for i in range(n_lin)]
    merged = sorted({round(v, 6) for v in (log_part + lin_part) if 0.0 < v <= 1.0})
    if merged[-1] != 1.0:
        merged.append(1.0)
    return merged
