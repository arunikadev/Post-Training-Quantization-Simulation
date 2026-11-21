"""End-to-end integration tests for PTQPipeline and SweepRunner."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from ptq_sim.config.schema import PTQExperimentConfig
from ptq_sim.models.examples.resnet_mini import MiniResNet
from ptq_sim.pipeline.runner import PTQPipeline, SweepRunner


class TestPipelineEndToEnd(unittest.TestCase):
    def test_full_pipeline_with_checkpoint_analysis_sensitivity_plots_export(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            ckpt_path = tmp_dir / "mini_resnet.pth"
            model = MiniResNet(in_channels=3, num_classes=10)
            torch.save(model.state_dict(), ckpt_path)

            cfg = PTQExperimentConfig.from_yaml("configs/base.yaml").apply_overrides(
                {
                    "experiment_name": "e2e_test_run",
                    "device": "cpu",
                    "output_dir": str(tmp_dir / "outputs"),
                    "model.name": "mini_resnet",
                    "model.checkpoint": str(ckpt_path),
                    "data.name": "synthetic_image",
                    "data.kwargs.num_calib": 64,
                    "data.kwargs.num_eval": 64,
                    "data.num_calib_batches": 2,
                    "evaluation.run_layer_analysis": True,
                    "evaluation.layer_analysis_batches": 2,
                    "evaluation.run_sensitivity": True,
                    "evaluation.sensitivity_max_batches": 1,
                    "evaluation.generate_plots": True,
                    "evaluation.export_artifacts": True,
                }
            )

            result = PTQPipeline(cfg).run()
            self.assertIsNotNone(result.run_dir)
            run_dir = result.run_dir
            assert run_dir is not None

            expected_files = [
                run_dir / "config_resolved.yaml",
                run_dir / "metrics_summary.json",
                run_dir / "summary_report.md",
                run_dir / "layer_analysis.csv",
                run_dir / "sensitivity_analysis.csv",
                run_dir / "qparams.json",
                run_dir / "quantized_simulated.pt",
                run_dir / "quantized_int_package.pt",
                run_dir / "plots" / "weight_distributions.png",
                run_dir / "plots" / "activation_distributions.png",
                run_dir / "plots" / "layer_error_metrics.png",
                run_dir / "plots" / "sensitivity_ranking.png",
            ]
            for fpath in expected_files:
                self.assertTrue(fpath.is_file(), msg=f"Missing expected artifact: {fpath}")
                self.assertGreater(fpath.stat().st_size, 0)

            self.assertGreater(result.profile["compression_ratio"], 2.0)

    def test_sweep_runner_generates_summary_and_pareto(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            base_cfg = PTQExperimentConfig.from_yaml("configs/base.yaml").apply_overrides(
                {
                    "device": "cpu",
                    "model.name": "simple_cnn",
                    "data.name": "synthetic_image",
                    "data.kwargs.num_calib": 64,
                    "data.kwargs.num_eval": 64,
                    "data.num_calib_batches": 2,
                    "evaluation.layer_analysis_batches": 1,
                    "evaluation.generate_plots": False,
                }
            )
            scenarios = [
                {"name": "W8A8", "overrides": {"quantization.weight.bit_width": 8, "quantization.activation.bit_width": 8}},
                {"name": "W4A8", "overrides": {"quantization.weight.bit_width": 4, "quantization.activation.bit_width": 8}},
                {
                    "name": "W4_Only",
                    "overrides": {
                        "quantization.weight.bit_width": 4,
                        "quantization.activation.enabled": False,
                        "quantization.quantize_input": False,
                    },
                },
            ]
            sweep = SweepRunner(
                base_config=base_cfg,
                scenarios=scenarios,
                sweep_name="e2e_sweep",
                output_dir=tmp_dir / "outputs",
            )
            res = sweep.run()
            self.assertEqual(len(res.dataframe), 3)
            self.assertTrue(res.summary_csv_path.is_file())
            self.assertTrue(res.summary_json_path.is_file())
            self.assertIsNotNone(res.pareto_plot_path)
            assert res.pareto_plot_path is not None
            self.assertTrue(res.pareto_plot_path.is_file())


if __name__ == "__main__":
    unittest.main()
