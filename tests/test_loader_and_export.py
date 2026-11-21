"""Tests for SmartCheckpointLoader, inspect_checkpoint, and ArtifactExporter reloadability."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from ptq_sim.config.schema import PTQExperimentConfig
from ptq_sim.export.exporter import ArtifactExporter
from ptq_sim.models.examples.simple_cnn import SimpleCNN
from ptq_sim.models.loader import inspect_checkpoint, load_checkpoint_into_model
from ptq_sim.quantization.calibrator import PTQCalibrator
from ptq_sim.quantization.preparer import ModelQuantizer


class TestLoaderAndExport(unittest.TestCase):
    def test_smart_loader_variants(self) -> None:
        torch.manual_seed(3)
        ref_model = SimpleCNN(in_channels=3, num_classes=10)
        ref_sd = ref_model.state_dict()
        x = torch.randn(2, 3, 32, 32)
        ref_model.eval()
        with torch.inference_mode():
            expected_out = ref_model(x)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            raw_path = tmp_dir / "raw.pth"
            wrapped_path = tmp_dir / "wrapped.pth"
            ddp_path = tmp_dir / "ddp.pth"

            torch.save(ref_sd, raw_path)
            torch.save({"epoch": 5, "model_state_dict": ref_sd}, wrapped_path)
            torch.save({"state_dict": {f"module.{k}": v for k, v in ref_sd.items()}}, ddp_path)

            info = inspect_checkpoint(wrapped_path)
            self.assertEqual(info["detected_format"], "wrapped[model_state_dict]")
            self.assertGreater(info["total_tensor_elements"], 0)

            for path in (raw_path, wrapped_path, ddp_path):
                target = SimpleCNN(in_channels=3, num_classes=10)
                report = load_checkpoint_into_model(target, path, strict=True)
                self.assertEqual(report["missing_keys"], [])
                self.assertEqual(report["unexpected_keys"], [])
                target.eval()
                with torch.inference_mode():
                    out = target(x)
                self.assertTrue(torch.allclose(expected_out, out, atol=1e-6))

    def test_export_simulated_reload_and_int_package_bounds(self) -> None:
        torch.manual_seed(5)
        cfg = PTQExperimentConfig()
        cfg.quantization.weight.bit_width = 4
        cfg.quantization.weight.granularity = "per_channel"
        cfg.quantization.activation.bit_width = 8

        float_model = SimpleCNN(in_channels=3, num_classes=10)
        q_model, prep_info = ModelQuantizer.prepare(float_model, cfg.quantization)
        calib_batches = [torch.randn(8, 3, 32, 32) for _ in range(2)]
        PTQCalibrator.calibrate(q_model, calib_batches, device=torch.device("cpu"), num_batches=2)

        x = torch.randn(3, 3, 32, 32)
        with torch.inference_mode():
            expected = q_model(x)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            summary = {
                "fp32_metrics": {"primary_metric_name": "top1_acc", "primary_metric": 0.9},
                "quantized_metrics": {"primary_metric_name": "top1_acc", "primary_metric": 0.89},
                "metric_delta": {"primary_metric_delta": -0.01},
                "profile": {"fp32_size_mb": 0.2, "quantized_size_mb": 0.05, "compression_ratio": 4.0},
                "preparation": prep_info,
            }
            artifacts = ArtifactExporter.export_all(
                run_dir=tmp_dir,
                cfg=cfg,
                quant_model=q_model,
                summary_payload=summary,
            )

            # 1. Reload simulated checkpoint into a fresh prepared model
            reloaded_q, _ = ModelQuantizer.prepare(SimpleCNN(in_channels=3, num_classes=10), cfg.quantization)
            load_checkpoint_into_model(reloaded_q, artifacts["quantized_simulated_pt"], strict=True)
            reloaded_q.eval()
            with torch.inference_mode():
                actual = reloaded_q(x)
            self.assertTrue(torch.allclose(expected, actual, atol=1e-6))

            # 2. Verify integer package bounds for 4-bit narrow signed weights [-7, 7]
            int_pkg = torch.load(artifacts["quantized_int_package_pt"], weights_only=True)
            self.assertEqual(int_pkg["format"], "ptq_sim_integer_package_v1")
            for layer_entry in int_pkg["quantized_layers"].values():
                int_w = layer_entry["int_weight"]
                self.assertEqual(int_w.dtype, torch.int8)
                self.assertGreaterEqual(int(int_w.min().item()), -7)
                self.assertLessEqual(int(int_w.max().item()), 7)


if __name__ == "__main__":
    unittest.main()
