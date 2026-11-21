"""Quantized drop-in wrapper modules for Conv2d and Linear layers."""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ptq_sim.config.schema import QuantizationConfig, QuantizerConfig
from ptq_sim.quantization.fake_quant import FakeQuantizer


class QuantBaseMixin:
    """Shared lifecycle, weight calibration, and export methods for quantized layers."""

    weight: nn.Parameter
    bias: nn.Parameter | None
    weight_quantizer: FakeQuantizer
    input_quantizer: FakeQuantizer | None
    output_quantizer: FakeQuantizer | None

    def _init_quantizers(
        self,
        weight_cfg: QuantizerConfig,
        act_cfg: QuantizerConfig,
        quantize_input: bool,
        quantize_output: bool,
    ) -> None:
        self.weight_quantizer = FakeQuantizer(copy.deepcopy(weight_cfg))
        if quantize_input and act_cfg.enabled:
            self.input_quantizer = FakeQuantizer(copy.deepcopy(act_cfg))
        else:
            self.input_quantizer = None
        if quantize_output and act_cfg.enabled:
            self.output_quantizer = FakeQuantizer(copy.deepcopy(act_cfg))
        else:
            self.output_quantizer = None

    def _quantizers(self) -> list[FakeQuantizer]:
        quants: list[FakeQuantizer] = [self.weight_quantizer]
        if self.input_quantizer is not None:
            quants.append(self.input_quantizer)
        if self.output_quantizer is not None:
            quants.append(self.output_quantizer)
        return quants

    def calibrate_weights(self) -> None:
        """Run one-shot observation and calibration on the layer weight tensor."""
        if not self.weight_quantizer.cfg.enabled:
            return
        self.weight_quantizer.observer.reset()
        self.weight_quantizer.observer.observe(self.weight.detach())
        self.weight_quantizer.calibrate_from_observer()
        self.weight_quantizer.disable_observer()

    def enable_calibration(self) -> None:
        """Observe activations while keeping calibrated weights fake-quantized."""
        if not bool(self.weight_quantizer.is_calibrated.item()) and self.weight_quantizer.cfg.enabled:
            self.calibrate_weights()
        for q in (self.input_quantizer, self.output_quantizer):
            if q is not None and q.cfg.enabled and not q.cfg.dynamic:
                q.reset_observer()
                q.enable_observer()
                q.disable_fake_quant()

    def freeze_quantization(self) -> None:
        """Compute activation qparams from observers, freeze observers, and enable fake-quant."""
        if self.weight_quantizer.cfg.enabled and not bool(self.weight_quantizer.is_calibrated.item()):
            self.calibrate_weights()
        self.weight_quantizer.disable_observer()
        self.weight_quantizer.enable_fake_quant()

        for q in (self.input_quantizer, self.output_quantizer):
            if q is not None and q.cfg.enabled:
                if not q.cfg.dynamic:
                    q.calibrate_from_observer()
                    q.disable_observer()
                q.enable_fake_quant()

    def set_fake_quant_enabled(self, enabled: bool) -> None:
        """Toggle all fake quantizers inside this layer without resetting qparams."""
        for q in self._quantizers():
            if enabled:
                q.enable_fake_quant()
            else:
                q.disable_fake_quant()

    def has_static_activation_quantizer(self) -> bool:
        for q in (self.input_quantizer, self.output_quantizer):
            if q is not None and q.cfg.enabled and not q.cfg.dynamic:
                return True
        return False

    def get_dequantized_weight(self) -> Tensor:
        """Return simulated quantized weight tensor without modifying stored FP32 weights."""
        if not self.weight_quantizer.cfg.enabled:
            return self.weight.detach()
        was_obs = self.weight_quantizer.observer_enabled
        was_fq = self.weight_quantizer.fake_quant_enabled
        try:
            self.weight_quantizer.disable_observer()
            self.weight_quantizer.enable_fake_quant()
            return self.weight_quantizer(self.weight.detach())
        finally:
            self.weight_quantizer.observer_enabled = was_obs
            self.weight_quantizer.fake_quant_enabled = was_fq

    def get_quantized_int_weight(self) -> tuple[Tensor, Tensor, Tensor]:
        """Return integer-clamped weight tensor plus scale and zero_point."""
        return self.weight_quantizer.quantize_to_int(self.weight.detach())

    def export_qparams(self) -> dict[str, Any]:
        """Return layer quantization metadata dictionary."""
        info: dict[str, Any] = {
            "layer_type": self.__class__.__name__,
            "weight_shape": list(self.weight.shape),
            "weight": self.weight_quantizer.export_metadata(),
        }
        if self.input_quantizer is not None:
            info["input_activation"] = self.input_quantizer.export_metadata()
        if self.output_quantizer is not None:
            info["output_activation"] = self.output_quantizer.export_metadata()
        return info


class QuantConv2d(nn.Module, QuantBaseMixin):
    """Quantized simulation replacement for torch.nn.Conv2d."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        stride: tuple[int, int] = (1, 1),
        padding: tuple[int, int] | str = (0, 0),
        dilation: tuple[int, int] = (1, 1),
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        weight_cfg: QuantizerConfig | None = None,
        act_cfg: QuantizerConfig | None = None,
        quantize_input: bool = True,
        quantize_output: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, *kernel_size, dtype=torch.float32)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels, dtype=torch.float32))
        else:
            self.register_parameter("bias", None)

        w_cfg = weight_cfg or QuantizerConfig()
        a_cfg = act_cfg or QuantizerConfig()
        self._init_quantizers(w_cfg, a_cfg, quantize_input, quantize_output)

    @classmethod
    def from_float(
        cls,
        float_mod: nn.Conv2d,
        quant_cfg: QuantizationConfig,
        weight_cfg: QuantizerConfig | None = None,
        act_cfg: QuantizerConfig | None = None,
    ) -> QuantConv2d:
        """Create a QuantConv2d copying weights and hyperparameters from a float nn.Conv2d."""
        qmod = cls(
            in_channels=float_mod.in_channels,
            out_channels=float_mod.out_channels,
            kernel_size=float_mod.kernel_size,
            stride=float_mod.stride,
            padding=float_mod.padding,
            dilation=float_mod.dilation,
            groups=float_mod.groups,
            bias=float_mod.bias is not None,
            padding_mode=float_mod.padding_mode,
            weight_cfg=weight_cfg or quant_cfg.weight,
            act_cfg=act_cfg or quant_cfg.activation,
            quantize_input=quant_cfg.quantize_input,
            quantize_output=quant_cfg.quantize_output,
        )
        qmod.to(device=float_mod.weight.device, dtype=float_mod.weight.dtype)
        with torch.no_grad():
            qmod.weight.copy_(float_mod.weight)
            if float_mod.bias is not None and qmod.bias is not None:
                qmod.bias.copy_(float_mod.bias)
        qmod.calibrate_weights()
        return qmod

    def forward(self, x: Tensor) -> Tensor:
        if self.input_quantizer is not None:
            x = self.input_quantizer(x)
        w = self.weight_quantizer(self.weight)
        out = F.conv2d(
            x,
            w,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        if self.output_quantizer is not None:
            out = self.output_quantizer(out)
        return out


class QuantLinear(nn.Module, QuantBaseMixin):
    """Quantized simulation replacement for torch.nn.Linear (supports 2D and ND inputs)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        weight_cfg: QuantizerConfig | None = None,
        act_cfg: QuantizerConfig | None = None,
        quantize_input: bool = True,
        quantize_output: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.float32))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.float32))
        else:
            self.register_parameter("bias", None)

        w_cfg = weight_cfg or QuantizerConfig()
        a_cfg = act_cfg or QuantizerConfig()
        self._init_quantizers(w_cfg, a_cfg, quantize_input, quantize_output)

    @classmethod
    def from_float(
        cls,
        float_mod: nn.Linear,
        quant_cfg: QuantizationConfig,
        weight_cfg: QuantizerConfig | None = None,
        act_cfg: QuantizerConfig | None = None,
    ) -> QuantLinear:
        """Create a QuantLinear copying weights and bias from a float nn.Linear."""
        qmod = cls(
            in_features=float_mod.in_features,
            out_features=float_mod.out_features,
            bias=float_mod.bias is not None,
            weight_cfg=weight_cfg or quant_cfg.weight,
            act_cfg=act_cfg or quant_cfg.activation,
            quantize_input=quant_cfg.quantize_input,
            quantize_output=quant_cfg.quantize_output,
        )
        qmod.to(device=float_mod.weight.device, dtype=float_mod.weight.dtype)
        with torch.no_grad():
            qmod.weight.copy_(float_mod.weight)
            if float_mod.bias is not None and qmod.bias is not None:
                qmod.bias.copy_(float_mod.bias)
        qmod.calibrate_weights()
        return qmod

    def forward(self, x: Tensor) -> Tensor:
        if self.input_quantizer is not None:
            x = self.input_quantizer(x)
        w = self.weight_quantizer(self.weight)
        out = F.linear(x, w, self.bias)
        if self.output_quantizer is not None:
            out = self.output_quantizer(out)
        return out
