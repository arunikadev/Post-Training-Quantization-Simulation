"""FakeQuantizer nn.Module supporting static and dynamic PTQ across granularities."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from ptq_sim.config.schema import QuantizerConfig
from ptq_sim.quantization.functional import (
    broadcast_qparams_for_tensor,
    compute_qparams,
    fake_quantize_tensor,
    get_qmin_qmax,
    normalize_tensor_for_granularity,
    reshape_for_group_quant,
    restore_from_group_quant,
    uniform_quantize,
)
from ptq_sim.quantization.observers import build_observer


class FakeQuantizer(nn.Module):
    """Stateful fake-quantization operator backed by an observer and qparam buffers."""

    def __init__(self, cfg: QuantizerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.observer = build_observer(cfg)
        self.observer_enabled = False
        self.fake_quant_enabled = bool(cfg.enabled)

        self.register_buffer("scale", torch.ones(1, dtype=torch.float32))
        self.register_buffer("zero_point", torch.zeros(1, dtype=torch.float32))
        self.register_buffer("is_calibrated", torch.tensor(False, dtype=torch.bool))

    @property
    def qmin_qmax(self) -> tuple[int, int]:
        return get_qmin_qmax(
            bit_width=self.cfg.bit_width,
            signed=self.cfg.signed,
            narrow_range=self.cfg.narrow_range,
        )

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Any],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        # Resize scale/zero_point buffers dynamically to match loaded per-channel/group shapes.
        for buf_name in ("scale", "zero_point"):
            key = prefix + buf_name
            if key in state_dict:
                loaded_t = state_dict[key]
                current_t = getattr(self, buf_name)
                if current_t.shape != loaded_t.shape:
                    setattr(self, buf_name, torch.empty_like(loaded_t, device=current_t.device))
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def enable_observer(self) -> None:
        self.observer_enabled = True

    def disable_observer(self) -> None:
        self.observer_enabled = False

    def enable_fake_quant(self) -> None:
        if self.cfg.enabled:
            self.fake_quant_enabled = True

    def disable_fake_quant(self) -> None:
        self.fake_quant_enabled = False

    def reset_observer(self) -> None:
        self.observer.reset()
        self.is_calibrated.fill_(False)

    def calibrate_from_observer(self) -> None:
        """Compute and store scale & zero_point from the internal observer."""
        if not self.cfg.enabled or self.cfg.dynamic:
            return
        if not self.observer.has_observed:
            return
        scale, zp = self.observer.calculate_qparams()
        device = self.scale.device
        self.scale = scale.to(device=device, dtype=torch.float32).clone()
        self.zero_point = zp.to(device=device, dtype=torch.float32).clone()
        self.is_calibrated = torch.tensor(True, dtype=torch.bool, device=device)

    def _compute_dynamic_qparams(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Compute per-tensor, per-channel/token, or per-group qparams directly from x."""
        x_f = x.detach().to(torch.float32)
        gran = self.cfg.granularity
        if gran == "per_tensor":
            min_v, max_v = x_f.min(), x_f.max()
        elif gran == "per_channel":
            # For 2D/3D activations (e.g. Transformer per-token), reduce over the last dimension
            # when ch_axis == -1 or when ch_axis == 0 on 2D batches.
            if self.cfg.ch_axis == -1 or x_f.ndim >= 3:
                flat = x_f.reshape(-1, x_f.shape[-1])
                min_v = flat.min(dim=1).values
                max_v = flat.max(dim=1).values
            else:
                x_2d, _, _ = normalize_tensor_for_granularity(x_f, "per_channel", ch_axis=self.cfg.ch_axis)
                min_v = x_2d.min(dim=1).values
                max_v = x_2d.max(dim=1).values
        elif gran == "per_group":
            x_grouped, _, _ = reshape_for_group_quant(x_f, group_size=self.cfg.group_size)
            min_v = x_grouped.min(dim=-1).values
            max_v = x_grouped.max(dim=-1).values
        else:
            raise ValueError(f"Unsupported granularity '{gran}'.")

        return compute_qparams(
            min_val=min_v,
            max_val=max_v,
            bit_width=self.cfg.bit_width,
            symmetric=self.cfg.symmetric,
            signed=self.cfg.signed,
            narrow_range=self.cfg.narrow_range,
            include_zero=self.cfg.include_zero,
        )

    def _apply_fake_quant_with_qparams(self, x: Tensor, scale: Tensor, zero_point: Tensor) -> Tensor:
        qmin, qmax = self.qmin_qmax
        gran = self.cfg.granularity

        if gran == "per_group":
            x_grouped, orig_shape, pad_len = reshape_for_group_quant(x, group_size=self.cfg.group_size)
            s_b, z_b = broadcast_qparams_for_tensor(scale, zero_point, x_grouped, granularity="per_group")
            fq_grouped = fake_quantize_tensor(x_grouped, s_b, z_b, qmin, qmax)
            return restore_from_group_quant(fq_grouped, orig_shape, pad_len)

        if self.cfg.dynamic and gran == "per_channel" and (self.cfg.ch_axis == -1 or x.ndim >= 3):
            flat = x.reshape(-1, x.shape[-1])
            s_b = scale.reshape(-1, 1)
            z_b = zero_point.reshape(-1, 1)
            return fake_quantize_tensor(flat, s_b, z_b, qmin, qmax).reshape(x.shape)

        s_b, z_b = broadcast_qparams_for_tensor(
            scale, zero_point, x, granularity=gran, ch_axis=self.cfg.ch_axis
        )
        return fake_quantize_tensor(x, s_b, z_b, qmin, qmax)

    def forward(self, x: Tensor) -> Tensor:
        if not self.cfg.enabled:
            return x

        if self.cfg.dynamic:
            if not self.fake_quant_enabled:
                return x
            scale, zp = self._compute_dynamic_qparams(x)
            return self._apply_fake_quant_with_qparams(x, scale, zp)

        if self.observer_enabled:
            self.observer.observe(x.detach())

        if not self.fake_quant_enabled:
            return x

        if not bool(self.is_calibrated.item()):
            if self.observer.has_observed:
                self.calibrate_from_observer()
            else:
                # If fake_quant is invoked before calibration, observe on the fly cleanly.
                self.observer.observe(x.detach())
                self.calibrate_from_observer()

        return self._apply_fake_quant_with_qparams(x, self.scale, self.zero_point)

    def quantize_to_int(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Quantize a tensor into clamped integer dtype plus (scale, zero_point)."""
        if not bool(self.is_calibrated.item()):
            self.observer.observe(x.detach())
            self.calibrate_from_observer()

        qmin, qmax = self.qmin_qmax
        gran = self.cfg.granularity
        x_f = x.detach().to(torch.float32)
        s = self.scale.to(x_f.device, dtype=torch.float32)
        z = self.zero_point.to(x_f.device, dtype=torch.float32)

        if gran == "per_group":
            x_grouped, orig_shape, pad_len = reshape_for_group_quant(x_f, group_size=self.cfg.group_size)
            s_b, z_b = broadcast_qparams_for_tensor(s, z, x_grouped, granularity="per_group")
            q_grouped = uniform_quantize(x_grouped, s_b, z_b, qmin, qmax)
            q_tensor = restore_from_group_quant(q_grouped, orig_shape, pad_len)
        else:
            s_b, z_b = broadcast_qparams_for_tensor(s, z, x_f, granularity=gran, ch_axis=self.cfg.ch_axis)
            q_tensor = uniform_quantize(x_f, s_b, z_b, qmin, qmax)

        if self.cfg.bit_width <= 8:
            target_dtype = torch.int8 if self.cfg.signed else torch.uint8
        else:
            target_dtype = torch.int16 if self.cfg.signed else torch.int32
        return q_tensor.to(target_dtype), s.detach().cpu(), z.detach().cpu()

    def export_metadata(self) -> dict[str, Any]:
        """Return JSON-serializable quantization parameter summary."""
        s_cpu = self.scale.detach().cpu()
        z_cpu = self.zero_point.detach().cpu()
        return {
            "enabled": bool(self.cfg.enabled),
            "bit_width": int(self.cfg.bit_width),
            "symmetric": bool(self.cfg.symmetric),
            "signed": bool(self.cfg.signed),
            "narrow_range": bool(self.cfg.narrow_range),
            "granularity": self.cfg.granularity,
            "group_size": int(self.cfg.group_size),
            "observer": self.cfg.observer,
            "dynamic": bool(self.cfg.dynamic),
            "is_calibrated": bool(self.is_calibrated.item()),
            "scale_shape": list(s_cpu.shape),
            "scale_min": float(s_cpu.min().item()),
            "scale_max": float(s_cpu.max().item()),
            "scale_mean": float(s_cpu.mean().item()),
            "zero_point_min": float(z_cpu.min().item()),
            "zero_point_max": float(z_cpu.max().item()),
            "scale": s_cpu.tolist() if s_cpu.numel() <= 128 else None,
            "zero_point": z_cpu.tolist() if z_cpu.numel() <= 128 else None,
        }
