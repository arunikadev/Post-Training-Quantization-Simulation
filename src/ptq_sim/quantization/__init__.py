"""Core Post-Training Quantization simulation engine."""

from ptq_sim.quantization.bn_folding import (
    fold_batch_norms,
    fold_conv_bn_weights,
    fold_linear_bn_weights,
)
from ptq_sim.quantization.calibrator import PTQCalibrator, extract_model_inputs
from ptq_sim.quantization.fake_quant import FakeQuantizer
from ptq_sim.quantization.functional import (
    compute_qparams,
    fake_quantize_tensor,
    get_qmin_qmax,
    reshape_for_group_quant,
    restore_from_group_quant,
    uniform_dequantize,
    uniform_quantize,
)
from ptq_sim.quantization.modules import QuantBaseMixin, QuantConv2d, QuantLinear
from ptq_sim.quantization.observers import (
    OBSERVER_REGISTRY,
    BaseObserver,
    KLObserver,
    MinMaxObserver,
    MovingAverageMinMaxObserver,
    MSEObserver,
    PercentileObserver,
    build_observer,
    register_observer,
)
from ptq_sim.quantization.preparer import ModelQuantizer

__all__ = [
    "OBSERVER_REGISTRY",
    "BaseObserver",
    "FakeQuantizer",
    "KLObserver",
    "MinMaxObserver",
    "ModelQuantizer",
    "MovingAverageMinMaxObserver",
    "MSEObserver",
    "PTQCalibrator",
    "PercentileObserver",
    "QuantBaseMixin",
    "QuantConv2d",
    "QuantLinear",
    "build_observer",
    "compute_qparams",
    "extract_model_inputs",
    "fake_quantize_tensor",
    "fold_batch_norms",
    "fold_conv_bn_weights",
    "fold_linear_bn_weights",
    "get_qmin_qmax",
    "register_observer",
    "reshape_for_group_quant",
    "restore_from_group_quant",
    "uniform_dequantize",
    "uniform_quantize",
]
