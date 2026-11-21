"""Calibration observers for PTQ: MinMax, EMA, Percentile, MSE, and KL Divergence."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

import numpy as np
import torch
from torch import Tensor, nn

from ptq_sim.config.schema import QuantizerConfig
from ptq_sim.quantization.functional import (
    compute_qparams,
    fake_quantize_tensor,
    get_qmin_qmax,
    normalize_tensor_for_granularity,
    optimal_mse_ratio_candidates,
    reshape_for_group_quant,
)

OBSERVER_REGISTRY: dict[str, type[BaseObserver]] = {}


def register_observer(name: str) -> Callable[[type[BaseObserver]], type[BaseObserver]]:
    """Decorator registering an observer class under a canonical lowercase name."""

    def _decorator(cls: type[BaseObserver]) -> type[BaseObserver]:
        OBSERVER_REGISTRY[name.lower()] = cls
        return cls

    return _decorator


def build_observer(cfg: QuantizerConfig) -> BaseObserver:
    """Instantiate an observer matching QuantizerConfig."""
    key = cfg.observer.lower()
    if key not in OBSERVER_REGISTRY:
        raise ValueError(
            f"Unknown observer '{cfg.observer}'. Registered: {sorted(OBSERVER_REGISTRY.keys())}."
        )
    obs_cls = OBSERVER_REGISTRY[key]
    return obs_cls(cfg=cfg, **(cfg.observer_kwargs or {}))


class StreamingHistogram:
    """Constant-memory streaming histogram for per-tensor activation calibration."""

    def __init__(self, num_bins: int = 2048) -> None:
        if num_bins < 16:
            raise ValueError(f"num_bins must be >= 16, got {num_bins}.")
        self.num_bins = int(num_bins)
        self.hist: torch.Tensor | None = None
        self.hist_min: float | None = None
        self.hist_max: float | None = None
        self.total_count: float = 0.0

    def reset(self) -> None:
        self.hist = None
        self.hist_min = None
        self.hist_max = None
        self.total_count = 0.0

    def update(self, x: Tensor) -> None:
        if x.numel() == 0:
            return
        flat = x.detach().to(torch.float32).reshape(-1)
        # Filter out non-finite values if any to keep histogram numerically safe.
        finite_mask = torch.isfinite(flat)
        if not torch.all(finite_mask):
            flat = flat[finite_mask]
            if flat.numel() == 0:
                return

        b_min = float(flat.min().item())
        b_max = float(flat.max().item())
        b_count = float(flat.numel())

        if self.hist is None:
            self.hist_min = b_min
            self.hist_max = b_max
            self.total_count = b_count
            if b_max > b_min:
                counts = torch.histc(flat, bins=self.num_bins, min=b_min, max=b_max)
                self.hist = counts.detach().cpu().to(torch.float64)
            else:
                self.hist = torch.zeros(self.num_bins, dtype=torch.float64)
                self.hist[0] = b_count
            return

        assert self.hist_min is not None and self.hist_max is not None
        old_min, old_max = self.hist_min, self.hist_max
        new_min = min(old_min, b_min)
        new_max = max(old_max, b_max)

        if new_max == new_min:
            self.total_count += b_count
            self.hist[0] = self.total_count
            return

        if new_min < old_min or new_max > old_max:
            self._rescale_histogram(old_min, old_max, new_min, new_max)
            self.hist_min = new_min
            self.hist_max = new_max

        if b_max == b_min:
            new_w = (self.hist_max - self.hist_min) / float(self.num_bins)
            idx = int(min(max((b_min - self.hist_min) / new_w, 0.0), float(self.num_bins - 1)))
            self.hist[idx] += b_count
        else:
            # Compute bin histogram over the expanded global range so bins align exactly.
            counts = torch.histc(flat, bins=self.num_bins, min=self.hist_min, max=self.hist_max)
            self.hist += counts.detach().cpu().to(torch.float64)

        self.total_count += b_count

    def _rescale_histogram(
        self,
        old_min: float,
        old_max: float,
        new_min: float,
        new_max: float,
    ) -> None:
        """Redistribute existing histogram counts proportionally into the expanded range."""
        assert self.hist is not None
        n = self.num_bins
        new_hist = torch.zeros(n, dtype=torch.float64)
        new_w = (new_max - new_min) / float(n)

        if old_max == old_min:
            idx = int(min(max((old_min - new_min) / new_w, 0.0), float(n - 1)))
            new_hist[idx] = self.total_count
            self.hist = new_hist
            return

        old_w = (old_max - old_min) / float(n)
        k = torch.arange(n, dtype=torch.float64)
        old_left = old_min + k * old_w
        old_right = old_left + old_w

        f0 = (old_left - new_min) / new_w
        f1 = (old_right - new_min) / new_w

        j0 = torch.clamp(torch.floor(f0).to(torch.long), 0, n - 1)
        j1 = torch.clamp(torch.floor(f1 - 1e-12).to(torch.long), 0, n - 1)

        same_bin = j0 == j1
        width = torch.clamp(f1 - f0, min=1e-12)
        boundary = (j0 + 1).to(torch.float64)
        frac0 = torch.where(same_bin, torch.ones_like(width), torch.clamp((boundary - f0) / width, 0.0, 1.0))
        frac1 = 1.0 - frac0

        new_hist.scatter_add_(0, j0, self.hist * frac0)
        split_mask = ~same_bin
        if torch.any(split_mask):
            new_hist.scatter_add_(0, j1[split_mask], self.hist[split_mask] * frac1[split_mask])
        self.hist = new_hist

    def bin_centers_and_counts(self) -> tuple[Tensor, Tensor]:
        if self.hist is None or self.hist_min is None or self.hist_max is None:
            raise RuntimeError("Histogram has not observed any data.")
        if self.hist_max == self.hist_min:
            return torch.tensor([self.hist_min], dtype=torch.float32), torch.tensor(
                [self.total_count], dtype=torch.float64
            )
        step = (self.hist_max - self.hist_min) / float(self.num_bins)
        centers = torch.linspace(
            self.hist_min + 0.5 * step,
            self.hist_max - 0.5 * step,
            steps=self.num_bins,
            dtype=torch.float32,
        )
        return centers, self.hist


class BaseObserver(nn.Module, ABC):
    """Abstract base class for PTQ range/threshold observers."""

    def __init__(self, cfg: QuantizerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.register_buffer("min_val", torch.tensor(float("inf"), dtype=torch.float32))
        self.register_buffer("max_val", torch.tensor(float("-inf"), dtype=torch.float32))
        self.has_observed = False

    def _compute_batch_minmax(self, x: Tensor) -> tuple[Tensor, Tensor, tuple[int, ...]]:
        """Reduce x to (b_min, b_max, target_qparam_shape) based on configured granularity."""
        x_f = x.detach().to(torch.float32)
        gran = self.cfg.granularity
        if gran == "per_tensor":
            return x_f.min(), x_f.max(), ()
        if gran == "per_channel":
            x_2d, _, _ = normalize_tensor_for_granularity(
                x_f, granularity="per_channel", ch_axis=self.cfg.ch_axis
            )
            return x_2d.min(dim=1).values, x_2d.max(dim=1).values, (x_2d.shape[0],)
        if gran == "per_group":
            x_grouped, _, _ = reshape_for_group_quant(x_f, group_size=self.cfg.group_size)
            b_min = x_grouped.min(dim=-1).values
            b_max = x_grouped.max(dim=-1).values
            return b_min, b_max, tuple(b_min.shape)
        raise ValueError(f"Unsupported granularity '{gran}'.")

    def _ensure_min_max_shape(self, target_shape: tuple[int, ...], device: torch.device) -> None:
        if tuple(self.min_val.shape) != target_shape or not self.has_observed:
            self.min_val = torch.full(target_shape, float("inf"), dtype=torch.float32, device=device)
            self.max_val = torch.full(target_shape, float("-inf"), dtype=torch.float32, device=device)

    @abstractmethod
    def observe(self, x: Tensor) -> None:
        """Observe a tensor batch and update internal statistics."""

    def calculate_qparams(self) -> tuple[Tensor, Tensor]:
        """Compute (scale, zero_point) from observed statistics."""
        if not self.has_observed:
            raise RuntimeError("Observer has not observed any tensor before calculate_qparams().")
        return compute_qparams(
            min_val=self.min_val,
            max_val=self.max_val,
            bit_width=self.cfg.bit_width,
            symmetric=self.cfg.symmetric,
            signed=self.cfg.signed,
            narrow_range=self.cfg.narrow_range,
            include_zero=self.cfg.include_zero,
        )

    def reset(self) -> None:
        """Reset observer state for a fresh calibration pass."""
        device = self.min_val.device
        self.min_val = torch.tensor(float("inf"), dtype=torch.float32, device=device)
        self.max_val = torch.tensor(float("-inf"), dtype=torch.float32, device=device)
        self.has_observed = False


@register_observer("minmax")
class MinMaxObserver(BaseObserver):
    """Tracks running global min and max across observed batches."""

    def observe(self, x: Tensor) -> None:
        if x.numel() == 0:
            return
        b_min, b_max, shape = self._compute_batch_minmax(x)
        self._ensure_min_max_shape(shape, x.device)
        if not self.has_observed:
            self.min_val.copy_(b_min)
            self.max_val.copy_(b_max)
            self.has_observed = True
        else:
            self.min_val.copy_(torch.minimum(self.min_val, b_min))
            self.max_val.copy_(torch.maximum(self.max_val, b_max))


@register_observer("ema")
class MovingAverageMinMaxObserver(BaseObserver):
    """Tracks exponential moving average of min and max across observed batches."""

    def __init__(self, cfg: QuantizerConfig, averaging_constant: float = 0.01) -> None:
        super().__init__(cfg)
        if not (0.0 < averaging_constant <= 1.0):
            raise ValueError(f"averaging_constant must be in (0, 1], got {averaging_constant}.")
        self.averaging_constant = float(averaging_constant)

    def observe(self, x: Tensor) -> None:
        if x.numel() == 0:
            return
        b_min, b_max, shape = self._compute_batch_minmax(x)
        self._ensure_min_max_shape(shape, x.device)
        if not self.has_observed:
            self.min_val.copy_(b_min)
            self.max_val.copy_(b_max)
            self.has_observed = True
        else:
            c = self.averaging_constant
            self.min_val.mul_(1.0 - c).add_(b_min * c)
            self.max_val.mul_(1.0 - c).add_(b_max * c)


@register_observer("percentile")
class PercentileObserver(BaseObserver):
    """Clips extreme outliers using a streaming histogram (per-tensor) or exact quantiles."""

    def __init__(
        self,
        cfg: QuantizerConfig,
        percentile: float = 99.99,
        num_bins: int = 2048,
    ) -> None:
        super().__init__(cfg)
        if not (50.0 < percentile <= 100.0):
            raise ValueError(f"percentile must be in (50, 100], got {percentile}.")
        self.percentile = float(percentile)
        self.num_bins = int(num_bins)
        self._histogram = StreamingHistogram(num_bins=self.num_bins)

    def reset(self) -> None:
        super().reset()
        self._histogram.reset()

    def observe(self, x: Tensor) -> None:
        if x.numel() == 0:
            return
        x_f = x.detach().to(torch.float32)

        if self.cfg.granularity == "per_tensor":
            self._histogram.update(x_f)
            assert self._histogram.hist_min is not None and self._histogram.hist_max is not None
            self.min_val = torch.tensor(self._histogram.hist_min, dtype=torch.float32, device=x.device)
            self.max_val = torch.tensor(self._histogram.hist_max, dtype=torch.float32, device=x.device)
            self.has_observed = True
            return

        # For per_channel / per_group (typically weights), compute exact quantiles directly.
        if self.cfg.granularity == "per_channel":
            x_2d, _, _ = normalize_tensor_for_granularity(x_f, "per_channel", ch_axis=self.cfg.ch_axis)
            target_shape = (x_2d.shape[0],)
        else:
            x_grouped, _, _ = reshape_for_group_quant(x_f, group_size=self.cfg.group_size)
            target_shape = tuple(x_grouped.shape[:2])
            x_2d = x_grouped.reshape(-1, x_grouped.shape[-1])

        p_high = self.percentile / 100.0
        p_low = 1.0 - p_high
        if self.cfg.symmetric and self.cfg.signed:
            thr = torch.quantile(x_2d.abs(), q=p_high, dim=1)
            q_min, q_max = -thr, thr
        else:
            q_low = torch.quantile(x_2d, q=p_low, dim=1)
            q_high = torch.quantile(x_2d, q=p_high, dim=1)
            q_min = q_low.reshape(target_shape)
            q_max = q_high.reshape(target_shape)

        q_min = q_min.reshape(target_shape)
        q_max = q_max.reshape(target_shape)
        self._ensure_min_max_shape(target_shape, x.device)
        if not self.has_observed:
            self.min_val.copy_(q_min)
            self.max_val.copy_(q_max)
            self.has_observed = True
        else:
            self.min_val.copy_(torch.minimum(self.min_val, q_min))
            self.max_val.copy_(torch.maximum(self.max_val, q_max))

    def calculate_qparams(self) -> tuple[Tensor, Tensor]:
        if not self.has_observed:
            raise RuntimeError("PercentileObserver has not observed any tensor.")
        if self.cfg.granularity != "per_tensor":
            return super().calculate_qparams()

        centers, counts = self._histogram.bin_centers_and_counts()
        total = float(counts.sum().item())
        if total <= 0.0 or centers.numel() == 1:
            return super().calculate_qparams()

        p_high = self.percentile / 100.0
        p_low = 1.0 - p_high

        if self.cfg.symmetric and self.cfg.signed:
            abs_centers = centers.abs()
            sorted_abs, order = torch.sort(abs_centers)
            sorted_counts = counts[order]
            cdf = torch.cumsum(sorted_counts, dim=0) / total
            idx = int(torch.searchsorted(cdf, torch.tensor(p_high, dtype=cdf.dtype)).clamp(0, len(sorted_abs) - 1))
            thr = float(sorted_abs[idx].item())
            min_t = torch.tensor(-thr, dtype=torch.float32, device=self.min_val.device)
            max_t = torch.tensor(thr, dtype=torch.float32, device=self.max_val.device)
        else:
            cdf = torch.cumsum(counts, dim=0) / total
            idx_low = int(torch.searchsorted(cdf, torch.tensor(p_low, dtype=cdf.dtype)).clamp(0, len(centers) - 1))
            idx_high = int(torch.searchsorted(cdf, torch.tensor(p_high, dtype=cdf.dtype)).clamp(0, len(centers) - 1))
            min_t = torch.tensor(float(centers[idx_low].item()), dtype=torch.float32, device=self.min_val.device)
            max_t = torch.tensor(float(centers[idx_high].item()), dtype=torch.float32, device=self.max_val.device)

        return compute_qparams(
            min_val=min_t,
            max_val=max_t,
            bit_width=self.cfg.bit_width,
            symmetric=self.cfg.symmetric,
            signed=self.cfg.signed,
            narrow_range=self.cfg.narrow_range,
            include_zero=self.cfg.include_zero,
        )


@register_observer("mse")
class MSEObserver(BaseObserver):
    """Searches clipping scale ratios to minimize L2 fake-quantization error."""

    def __init__(
        self,
        cfg: QuantizerConfig,
        num_steps: int = 100,
        min_ratio: float = 0.01,
        num_bins: int = 2048,
    ) -> None:
        super().__init__(cfg)
        self.num_steps = int(num_steps)
        self.min_ratio = float(min_ratio)
        self.num_bins = int(num_bins)
        self._histogram = StreamingHistogram(num_bins=self.num_bins)
        self._ratios = optimal_mse_ratio_candidates(self.min_ratio, self.num_steps)

    def reset(self) -> None:
        super().reset()
        self._histogram.reset()

    def _optimize_2d_mse(self, rows_2d: Tensor) -> tuple[Tensor, Tensor]:
        """Find per-row (min, max) minimizing row-wise quantization MSE via vectorized search."""
        row_min = rows_2d.min(dim=1).values
        row_max = rows_2d.max(dim=1).values
        qmin, qmax = get_qmin_qmax(self.cfg.bit_width, self.cfg.signed, self.cfg.narrow_range)

        best_err = torch.full((rows_2d.shape[0],), float("inf"), dtype=torch.float32, device=rows_2d.device)
        best_min = row_min.clone()
        best_max = row_max.clone()

        for r in self._ratios:
            if self.cfg.symmetric and self.cfg.signed:
                max_abs = torch.maximum(row_min.abs(), row_max.abs()) * r
                c_min, c_max = -max_abs, max_abs
            else:
                center = 0.5 * (row_min + row_max) if not self.cfg.include_zero else torch.zeros_like(row_min)
                c_min = center + (row_min - center) * r
                c_max = center + (row_max - center) * r

            scale, zp = compute_qparams(
                c_min,
                c_max,
                bit_width=self.cfg.bit_width,
                symmetric=self.cfg.symmetric,
                signed=self.cfg.signed,
                narrow_range=self.cfg.narrow_range,
                include_zero=self.cfg.include_zero,
            )
            fq = fake_quantize_tensor(rows_2d, scale.unsqueeze(1), zp.unsqueeze(1), qmin, qmax)
            err = torch.mean((rows_2d - fq) ** 2, dim=1)
            improved = err < best_err
            best_err = torch.where(improved, err, best_err)
            best_min = torch.where(improved, c_min, best_min)
            best_max = torch.where(improved, c_max, best_max)

        return best_min, best_max

    def observe(self, x: Tensor) -> None:
        if x.numel() == 0:
            return
        x_f = x.detach().to(torch.float32)

        if self.cfg.granularity == "per_tensor":
            self._histogram.update(x_f)
            assert self._histogram.hist_min is not None and self._histogram.hist_max is not None
            self.min_val = torch.tensor(self._histogram.hist_min, dtype=torch.float32, device=x.device)
            self.max_val = torch.tensor(self._histogram.hist_max, dtype=torch.float32, device=x.device)
            self.has_observed = True
            return

        if self.cfg.granularity == "per_channel":
            rows_2d, _, _ = normalize_tensor_for_granularity(x_f, "per_channel", ch_axis=self.cfg.ch_axis)
            target_shape = (rows_2d.shape[0],)
        else:
            x_grouped, _, _ = reshape_for_group_quant(x_f, group_size=self.cfg.group_size)
            target_shape = tuple(x_grouped.shape[:2])
            rows_2d = x_grouped.reshape(-1, x_grouped.shape[-1])

        opt_min, opt_max = self._optimize_2d_mse(rows_2d)
        opt_min = opt_min.reshape(target_shape)
        opt_max = opt_max.reshape(target_shape)
        self._ensure_min_max_shape(target_shape, x.device)
        if not self.has_observed:
            self.min_val.copy_(opt_min)
            self.max_val.copy_(opt_max)
            self.has_observed = True
        else:
            self.min_val.copy_(torch.minimum(self.min_val, opt_min))
            self.max_val.copy_(torch.maximum(self.max_val, opt_max))

    def calculate_qparams(self) -> tuple[Tensor, Tensor]:
        if not self.has_observed:
            raise RuntimeError("MSEObserver has not observed any tensor.")
        if self.cfg.granularity != "per_tensor":
            return super().calculate_qparams()

        centers, counts = self._histogram.bin_centers_and_counts()
        if centers.numel() == 1 or float(counts.sum().item()) <= 0.0:
            return super().calculate_qparams()

        weights = counts.to(torch.float32)
        weights = weights / weights.sum()
        g_min = self.min_val.detach().cpu()
        g_max = self.max_val.detach().cpu()
        qmin, qmax = get_qmin_qmax(self.cfg.bit_width, self.cfg.signed, self.cfg.narrow_range)

        best_err = float("inf")
        best_min = g_min
        best_max = g_max

        for r in self._ratios:
            if self.cfg.symmetric and self.cfg.signed:
                max_abs = torch.maximum(g_min.abs(), g_max.abs()) * r
                c_min, c_max = -max_abs, max_abs
            else:
                center = 0.5 * (g_min + g_max) if not self.cfg.include_zero else torch.tensor(0.0)
                c_min = center + (g_min - center) * r
                c_max = center + (g_max - center) * r

            scale, zp = compute_qparams(
                c_min,
                c_max,
                bit_width=self.cfg.bit_width,
                symmetric=self.cfg.symmetric,
                signed=self.cfg.signed,
                narrow_range=self.cfg.narrow_range,
                include_zero=self.cfg.include_zero,
            )
            fq = fake_quantize_tensor(centers, scale, zp, qmin, qmax)
            err = float(torch.sum(weights * (centers - fq) ** 2).item())
            if err < best_err:
                best_err = err
                best_min = c_min
                best_max = c_max

        return compute_qparams(
            min_val=best_min.to(self.min_val.device),
            max_val=best_max.to(self.max_val.device),
            bit_width=self.cfg.bit_width,
            symmetric=self.cfg.symmetric,
            signed=self.cfg.signed,
            narrow_range=self.cfg.narrow_range,
            include_zero=self.cfg.include_zero,
        )


@register_observer("kl")
class KLObserver(BaseObserver):
    """TensorRT-style KL-divergence (entropy) calibration observer."""

    def __init__(
        self,
        cfg: QuantizerConfig,
        num_bins: int = 2048,
        num_candidates: int = 256,
    ) -> None:
        super().__init__(cfg)
        self.num_bins = int(num_bins)
        self.num_candidates = int(num_candidates)
        self._abs_histogram = StreamingHistogram(num_bins=self.num_bins)
        self._signed_min: float | None = None
        self._signed_max: float | None = None

    def reset(self) -> None:
        super().reset()
        self._abs_histogram.reset()
        self._signed_min = None
        self._signed_max = None

    def _kl_threshold_from_1d_hist(self, hist_np: np.ndarray, max_range: float, target_bins: int) -> float:
        """Select clipping threshold minimizing KL(P || Q) over a positive histogram [0, max_range]."""
        n = len(hist_np)
        if n <= target_bins or float(hist_np.sum()) <= 0.0 or max_range <= 0.0:
            return max_range

        start_bin = max(target_bins, 2)
        if n - start_bin <= self.num_candidates:
            candidates = np.arange(start_bin, n + 1)
        else:
            candidates = np.unique(np.linspace(start_bin, n, num=self.num_candidates, dtype=np.int64))

        bin_width = max_range / float(n)
        best_kl = float("inf")
        best_threshold = max_range

        for i in candidates:
            ref = hist_np[:i].copy()
            if i < n:
                ref[-1] += float(hist_np[i:].sum())
            q_in = hist_np[:i].copy()
            p_sum = float(ref.sum())
            q_in_sum = float(q_in.sum())
            if p_sum <= 0.0 or q_in_sum <= 0.0:
                continue

            # Map i reference bins onto target_bins quantized buckets (without pre-folding tail).
            stride = float(i) / float(target_bins)
            q_idx = np.minimum((np.arange(i, dtype=np.float64) / stride).astype(np.int64), target_bins - 1)
            quantized_bins = np.bincount(q_idx, weights=q_in, minlength=target_bins)
            nonzero = (q_in > 0).astype(np.float64)
            nonzero_per_bucket = np.bincount(q_idx, weights=nonzero, minlength=target_bins)

            q = np.zeros_like(q_in)
            valid_mask = (q_in > 0) & (nonzero_per_bucket[q_idx] > 0)
            q[valid_mask] = quantized_bins[q_idx[valid_mask]] / nonzero_per_bucket[q_idx[valid_mask]]
            q_sum = float(q.sum())
            if q_sum <= 0.0:
                continue

            p_norm = ref / p_sum
            q_norm = q / q_sum
            nz = p_norm > 0
            kl = float(np.sum(p_norm[nz] * np.log(p_norm[nz] / np.maximum(q_norm[nz], 1e-12))))
            if kl < best_kl:
                best_kl = kl
                best_threshold = (float(i) - 0.5) * bin_width

        return max(best_threshold, 1e-8)

    def _target_bins_count(self) -> int:
        if self.cfg.symmetric and self.cfg.signed:
            return max(2, (1 << (self.cfg.bit_width - 1)) - 1)
        if self._signed_min is not None and self._signed_min >= 0.0 and not self.cfg.signed:
            return max(2, (1 << self.cfg.bit_width) - 1)
        return max(2, (1 << (self.cfg.bit_width - 1)) - 1)

    def observe(self, x: Tensor) -> None:
        if x.numel() == 0:
            return
        x_f = x.detach().to(torch.float32)

        if self.cfg.granularity == "per_tensor":
            b_min = float(x_f.min().item())
            b_max = float(x_f.max().item())
            self._signed_min = b_min if self._signed_min is None else min(self._signed_min, b_min)
            self._signed_max = b_max if self._signed_max is None else max(self._signed_max, b_max)
            self._abs_histogram.update(x_f.abs())
            self.min_val = torch.tensor(self._signed_min, dtype=torch.float32, device=x.device)
            self.max_val = torch.tensor(self._signed_max, dtype=torch.float32, device=x.device)
            self.has_observed = True
            return

        # Support per_channel / per_group cleanly (e.g., if user sets KL for weights).
        if self.cfg.granularity == "per_channel":
            rows_2d, _, _ = normalize_tensor_for_granularity(x_f, "per_channel", ch_axis=self.cfg.ch_axis)
            target_shape = (rows_2d.shape[0],)
        else:
            x_grouped, _, _ = reshape_for_group_quant(x_f, group_size=self.cfg.group_size)
            target_shape = tuple(x_grouped.shape[:2])
            rows_2d = x_grouped.reshape(-1, x_grouped.shape[-1])

        target_bins = max(2, (1 << (self.cfg.bit_width - 1)) - 1)
        n_rows = rows_2d.shape[0]
        row_mins = torch.empty(n_rows, dtype=torch.float32, device=x.device)
        row_maxs = torch.empty(n_rows, dtype=torch.float32, device=x.device)
        row_bins = min(self.num_bins, max(128, rows_2d.shape[1] * 2))

        rows_cpu = rows_2d.cpu().numpy()
        for idx in range(n_rows):
            row = rows_cpu[idx]
            r_min, r_max = float(row.min()), float(row.max())
            max_abs = max(abs(r_min), abs(r_max))
            if max_abs <= 0.0:
                row_mins[idx] = r_min
                row_maxs[idx] = r_max
                continue
            hist_np, _ = np.histogram(np.abs(row), bins=row_bins, range=(0.0, max_abs))
            thr = self._kl_threshold_from_1d_hist(hist_np.astype(np.float64), max_abs, target_bins)
            if self.cfg.symmetric and self.cfg.signed:
                row_mins[idx] = -thr
                row_maxs[idx] = thr
            else:
                row_mins[idx] = max(r_min, -thr)
                row_maxs[idx] = min(r_max, thr)

        row_mins = row_mins.reshape(target_shape)
        row_maxs = row_maxs.reshape(target_shape)
        self._ensure_min_max_shape(target_shape, x.device)
        if not self.has_observed:
            self.min_val.copy_(row_mins)
            self.max_val.copy_(row_maxs)
            self.has_observed = True
        else:
            self.min_val.copy_(torch.minimum(self.min_val, row_mins))
            self.max_val.copy_(torch.maximum(self.max_val, row_maxs))

    def calculate_qparams(self) -> tuple[Tensor, Tensor]:
        if not self.has_observed:
            raise RuntimeError("KLObserver has not observed any tensor.")
        if self.cfg.granularity != "per_tensor":
            return super().calculate_qparams()

        assert self._signed_min is not None and self._signed_max is not None
        assert self._abs_histogram.hist is not None and self._abs_histogram.hist_max is not None
        max_abs = float(self._abs_histogram.hist_max)
        hist_np = self._abs_histogram.hist.numpy()
        thr = self._kl_threshold_from_1d_hist(hist_np, max_abs, self._target_bins_count())

        if self.cfg.symmetric and self.cfg.signed:
            min_t = torch.tensor(-thr, dtype=torch.float32, device=self.min_val.device)
            max_t = torch.tensor(thr, dtype=torch.float32, device=self.max_val.device)
        else:
            min_t = torch.tensor(max(self._signed_min, -thr), dtype=torch.float32, device=self.min_val.device)
            max_t = torch.tensor(min(self._signed_max, thr), dtype=torch.float32, device=self.max_val.device)

        return compute_qparams(
            min_val=min_t,
            max_val=max_t,
            bit_width=self.cfg.bit_width,
            symmetric=self.cfg.symmetric,
            signed=self.cfg.signed,
            narrow_range=self.cfg.narrow_range,
            include_zero=self.cfg.include_zero,
        )
