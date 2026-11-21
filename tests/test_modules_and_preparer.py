"""Tests for QuantConv2d, QuantLinear, ModelQuantizer, overrides, weight-only, and dynamic modes."""

from __future__ import annotations

import unittest

import torch
from torch import nn

from ptq_sim.config.schema import LayerOverrideConfig, QuantizationConfig, QuantizerConfig
from ptq_sim.models.examples.mini_transformer import MiniTransformerClassifier
from ptq_sim.models.examples.simple_cnn import SimpleCNN
from ptq_sim.quantization.calibrator import PTQCalibrator
from ptq_sim.quantization.modules import QuantConv2d, QuantLinear
from ptq_sim.quantization.preparer import ModelQuantizer


class TestModulesAndPreparer(unittest.TestCase):
    def test_skip_layers_and_mixed_precision_overrides(self) -> None:
        model = SimpleCNN(in_channels=3, num_classes=10)
        cfg = QuantizationConfig(
            fold_bn=True,
            skip_layers=["classifier\\.3"],  # skip final linear layer via regex
            weight=QuantizerConfig(bit_width=4, granularity="per_channel"),
            activation=QuantizerConfig(bit_width=8, symmetric=False, signed=False),
            layer_overrides={
                "features.0": LayerOverrideConfig(weight={"bit_width": 8}),
            },
        )
        q_model, info = ModelQuantizer.prepare(model, cfg)
        layers = ModelQuantizer.get_quantized_layers(q_model)

        self.assertIn("classifier.3", info["skipped_layers"])
        self.assertNotIn("classifier.3", layers)
        self.assertIsInstance(q_model.classifier[3], nn.Linear)
        self.assertIsInstance(layers["features.0"], QuantConv2d)
        self.assertEqual(layers["features.0"].weight_quantizer.cfg.bit_width, 8)
        self.assertEqual(layers["features.4"].weight_quantizer.cfg.bit_width, 4)

    def test_weight_only_mode_requires_no_calib_loader(self) -> None:
        model = SimpleCNN(in_channels=3, num_classes=10)
        cfg = QuantizationConfig(
            fold_bn=True,
            quantize_input=False,
            weight=QuantizerConfig(bit_width=4, granularity="per_group", group_size=16),
            activation=QuantizerConfig(enabled=False),
        )
        q_model, _ = ModelQuantizer.prepare(model, cfg)
        calib_report = PTQCalibrator.calibrate(q_model, calib_loader=None, device=torch.device("cpu"))
        self.assertTrue(calib_report["skipped_activation_calib"])

        x = torch.randn(2, 3, 32, 32)
        with torch.inference_mode():
            out = q_model(x)
        self.assertEqual(tuple(out.shape), (2, 10))
        self.assertTrue(torch.isfinite(out).all())

    def test_dynamic_activation_on_mini_transformer_per_group(self) -> None:
        model = MiniTransformerClassifier(
            vocab_size=128,
            d_model=64,
            n_heads=4,
            num_layers=2,
            d_ff=128,
            num_classes=4,
            max_seq_len=32,
        )
        cfg = QuantizationConfig(
            fold_bn=False,
            weight=QuantizerConfig(bit_width=4, granularity="per_group", group_size=32, symmetric=True, signed=True),
            activation=QuantizerConfig(
                enabled=True,
                bit_width=8,
                dynamic=True,
                granularity="per_channel",
                ch_axis=-1,
                symmetric=False,
                signed=False,
            ),
        )
        q_model, info = ModelQuantizer.prepare(model, cfg)
        # 2 blocks * (q, k, v, out, fc1, fc2 = 6) + head (1) = 13 Linear layers
        self.assertEqual(len(info["quantized_layers"]), 13)
        for mod in ModelQuantizer.get_quantized_layers(q_model).values():
            self.assertIsInstance(mod, QuantLinear)

        calib_report = PTQCalibrator.calibrate(q_model, calib_loader=None, device=torch.device("cpu"))
        self.assertTrue(calib_report["skipped_activation_calib"])

        tokens = torch.randint(0, 128, (3, 16))
        with torch.inference_mode():
            logits = q_model(tokens)
        self.assertEqual(tuple(logits.shape), (3, 4))
        self.assertTrue(torch.isfinite(logits).all())


if __name__ == "__main__":
    unittest.main()
