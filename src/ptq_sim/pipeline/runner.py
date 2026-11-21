"""End-to-end PTQ simulation pipeline and multi-scenario sweep orchestrator."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import torch
from torch import nn

from ptq_sim.config.schema import PTQExperimentConfig
from ptq_sim.data.registry import build_dataloaders
from ptq_sim.evaluation.evaluator import CustomEvalFn, ModelEvaluator
from ptq_sim.evaluation.layer_analyzer import LayerAnalysisResult, LayerWiseAnalyzer
from ptq_sim.evaluation.profiler import ModelSizeProfiler
from ptq_sim.evaluation.sensitivity import SensitivityAnalyzer, SensitivityResult
from ptq_sim.export.exporter import ArtifactExporter
from ptq_sim.models.loader import load_checkpoint_into_model
from ptq_sim.models.registry import build_model
from ptq_sim.quantization.calibrator import PTQCalibrator
from ptq_sim.quantization.preparer import ModelQuantizer
from ptq_sim.utils.device import get_device_info, resolve_device
from ptq_sim.utils.io import create_run_dir, load_yaml, save_json
from ptq_sim.utils.logger import close_file_handlers, setup_logger
from ptq_sim.utils.seed import set_seed
from ptq_sim.visualization.plots import (
    plot_activation_distributions,
    plot_bitwidth_pareto,
    plot_layer_error_metrics,
    plot_sensitivity_ranking,
    plot_weight_distributions,
)


@dataclass
class PTQRunResult:
    """Complete result object returned by PTQPipeline.run()."""

    config: PTQExperimentConfig
    run_dir: Path | None
    quant_model: nn.Module
    fp32_metrics: dict[str, Any]
    quant_metrics: dict[str, Any]
    metric_delta: dict[str, Any]
    profile: dict[str, Any]
    preparation: dict[str, Any]
    calibration: dict[str, Any]
    layer_analysis: LayerAnalysisResult | None
    sensitivity: SensitivityResult | None
    artifacts: dict[str, str]


class PTQPipeline:
    """Orchestrates baseline eval, BN folding, quantization prep, calibration, analysis, and export."""

    def __init__(self, config: PTQExperimentConfig) -> None:
        self.config = config

    def run(
        self,
        model: nn.Module | None = None,
        calib_loader: Iterable[Any] | None = None,
        eval_loader: Iterable[Any] | None = None,
        eval_fn: CustomEvalFn | None = None,
        run_dir_override: str | Path | None = None,
        _skip_checkpoint_load: bool = False,
    ) -> PTQRunResult:
        """Execute full PTQ simulation workflow and return structured diagnostics."""
        cfg = self.config
        set_seed(cfg.seed)
        device = resolve_device(cfg.device)

        need_dir = cfg.evaluation.export_artifacts or cfg.evaluation.generate_plots or run_dir_override is not None
        if run_dir_override is not None:
            run_dir: Path | None = Path(run_dir_override)
            run_dir.mkdir(parents=True, exist_ok=True)
        elif need_dir:
            run_dir = create_run_dir(cfg.output_dir, cfg.experiment_name)
        else:
            run_dir = None

        log_file = (run_dir / "run.log") if run_dir is not None else None
        logger = setup_logger("ptq_sim", log_file=log_file)
        try:
            logger.info("Starting PTQ simulation '%s' on device=%s", cfg.experiment_name, device)
            logger.info("Device info: %s", get_device_info(device))

            # 1. Load or prepare model
            checkpoint_report: dict[str, Any] | None = None
            if model is None:
                float_model, checkpoint_report = build_model(cfg.model)
            else:
                float_model = model
                if cfg.model.checkpoint and not _skip_checkpoint_load:
                    checkpoint_report = load_checkpoint_into_model(
                        float_model,
                        checkpoint_path=cfg.model.checkpoint,
                        state_dict_key=cfg.model.state_dict_key,
                        strict=cfg.model.strict_load,
                        map_location="cpu",
                    )
            float_model.eval()
            float_model.to(device)

            # 2. Load DataLoaders if not directly provided
            needs_calib_data = bool(cfg.quantization.activation.enabled and not cfg.quantization.activation.dynamic)
            if eval_loader is None or (calib_loader is None and needs_calib_data):
                built_calib, built_eval = build_dataloaders(cfg.data, seed=cfg.seed)
                if calib_loader is None:
                    calib_loader = built_calib
                if eval_loader is None:
                    eval_loader = built_eval

            # 3. Baseline FP32 evaluation
            evaluator = ModelEvaluator(task=cfg.evaluation.task)
            fp32_metrics = evaluator.evaluate(
                float_model,
                eval_loader,
                device=device,
                max_batches=cfg.evaluation.max_eval_batches,
                eval_fn=eval_fn,
            )
            logger.info(
                "FP32 Baseline | %s=%.4f | latency=%.3f ms/batch",
                fp32_metrics["primary_metric_name"],
                fp32_metrics["primary_metric"],
                fp32_metrics.get("avg_latency_ms", 0.0),
            )

            # 4. Prepare quantized simulation model (BN folding + layer swap + weight calibration)
            quant_model, prep_info = ModelQuantizer.prepare(float_model, cfg.quantization)
            logger.info(
                "Prepared model: folded_bn=%d, quantized_layers=%d, skipped_layers=%d, overrides=%d",
                len(prep_info["folded_bn_pairs"]),
                len(prep_info["quantized_layers"]),
                len(prep_info["skipped_layers"]),
                len(prep_info["override_applied"]),
            )

            # 5. Calibrate static activation quantizers (skipped automatically for weight-only / dynamic)
            calib_info = PTQCalibrator.calibrate(
                quant_model,
                calib_loader=calib_loader,
                device=device,
                num_batches=cfg.data.num_calib_batches,
            )
            if calib_info["skipped_activation_calib"]:
                logger.info("Activation calibration skipped (weight-only or dynamic activation mode).")

            # 6. Evaluate Quantized model
            quant_metrics = evaluator.evaluate(
                quant_model,
                eval_loader,
                device=device,
                max_batches=cfg.evaluation.max_eval_batches,
                eval_fn=eval_fn,
            )
            p_delta = float(quant_metrics["primary_metric"]) - float(fp32_metrics["primary_metric"])
            higher_better = bool(fp32_metrics.get("higher_is_better", True))
            degradation = (-p_delta) if higher_better else p_delta
            metric_delta = {
                "primary_metric_name": fp32_metrics["primary_metric_name"],
                "primary_metric_delta": p_delta,
                "degradation": degradation,
            }
            logger.info(
                "Quantized Sim | %s=%.4f | delta=%+.4f | latency=%.3f ms/batch",
                quant_metrics["primary_metric_name"],
                quant_metrics["primary_metric"],
                p_delta,
                quant_metrics.get("avg_latency_ms", 0.0),
            )

            # 7. Profile footprint & BitOps
            sample_batch = next(iter(eval_loader), None)
            profile = ModelSizeProfiler.profile(quant_model, sample_batch=sample_batch, device=device)
            logger.info(
                "Footprint | FP32=%.4f MB -> Quant=%.4f MB (%.2fx compression, %.2f bits/param)",
                profile["fp32_size_mb"],
                profile["quantized_size_mb"],
                profile["compression_ratio"],
                profile["avg_effective_bits_per_param"],
            )

            # 8. Layer-wise error analysis
            layer_result: LayerAnalysisResult | None = None
            if cfg.evaluation.run_layer_analysis:
                layer_result = LayerWiseAnalyzer.analyze(
                    quant_model=quant_model,
                    eval_loader=eval_loader,
                    device=device,
                    max_batches=cfg.evaluation.layer_analysis_batches,
                )
                logger.info(
                    "Layer Analysis | mean_w_sqnr=%.2f dB, mean_act_sqnr=%.2f dB, worst_act_layer=%s",
                    layer_result.summary.get("mean_weight_sqnr_db") or 0.0,
                    layer_result.summary.get("mean_act_sqnr_db") or 0.0,
                    layer_result.summary.get("worst_act_sqnr_layer"),
                )

            # 9. Single-layer sensitivity analysis (optional)
            sens_result: SensitivityResult | None = None
            if cfg.evaluation.run_sensitivity:
                sens_result = SensitivityAnalyzer.analyze(
                    quant_model=quant_model,
                    fp32_metrics=fp32_metrics,
                    eval_loader=eval_loader,
                    evaluator=evaluator,
                    device=device,
                    max_batches=cfg.evaluation.sensitivity_max_batches,
                    eval_fn=eval_fn,
                )
                logger.info(
                    "Sensitivity Top-3: %s",
                    ", ".join(sens_result.most_sensitive_layers[:3]),
                )

            # 10. Plots
            plot_paths: dict[str, str] = {}
            if cfg.evaluation.generate_plots and run_dir is not None:
                plots_dir = run_dir / "plots"
                if layer_result is not None:
                    p1 = plot_weight_distributions(layer_result.weight_samples, plots_dir / "weight_distributions.png")
                    if p1 is not None:
                        plot_paths["weight_distributions"] = str(p1)
                    p2 = plot_activation_distributions(
                        layer_result.activation_samples, plots_dir / "activation_distributions.png"
                    )
                    if p2 is not None:
                        plot_paths["activation_distributions"] = str(p2)
                    p3 = plot_layer_error_metrics(layer_result.dataframe, plots_dir / "layer_error_metrics.png")
                    if p3 is not None:
                        plot_paths["layer_error_metrics"] = str(p3)
                if sens_result is not None:
                    p4 = plot_sensitivity_ranking(
                        sens_result.dataframe,
                        plots_dir / "sensitivity_ranking.png",
                        metric_name=sens_result.primary_metric_name,
                    )
                    if p4 is not None:
                        plot_paths["sensitivity_ranking"] = str(p4)

            # 11. Export artifacts
            artifacts: dict[str, str] = {}
            summary_payload = {
                "experiment_name": cfg.experiment_name,
                "device_info": get_device_info(device),
                "checkpoint_load": checkpoint_report,
                "preparation": prep_info,
                "calibration": calib_info,
                "fp32_metrics": fp32_metrics,
                "quantized_metrics": quant_metrics,
                "metric_delta": metric_delta,
                "profile": profile,
                "layer_analysis_summary": layer_result.summary if layer_result is not None else None,
                "most_sensitive_layers": sens_result.most_sensitive_layers if sens_result is not None else None,
            }

            if cfg.evaluation.export_artifacts and run_dir is not None:
                artifacts = ArtifactExporter.export_all(
                    run_dir=run_dir,
                    cfg=cfg,
                    quant_model=quant_model,
                    summary_payload=summary_payload,
                    layer_result=layer_result,
                    sens_result=sens_result,
                    plot_paths=plot_paths,
                )
                logger.info("Artifacts exported to '%s'.", run_dir)
            elif plot_paths:
                artifacts.update({f"plot_{k}": v for k, v in plot_paths.items()})

            return PTQRunResult(
                config=cfg,
                run_dir=run_dir,
                quant_model=quant_model,
                fp32_metrics=fp32_metrics,
                quant_metrics=quant_metrics,
                metric_delta=metric_delta,
                profile=profile,
                preparation=prep_info,
                calibration=calib_info,
                layer_analysis=layer_result,
                sensitivity=sens_result,
                artifacts=artifacts,
            )
        finally:
            close_file_handlers(logger)


@dataclass
class SweepRunResult:
    """Aggregated results from a multi-scenario PTQ sweep."""

    sweep_dir: Path
    dataframe: pd.DataFrame
    scenario_results: dict[str, PTQRunResult]
    pareto_plot_path: Path | None
    summary_csv_path: Path
    summary_json_path: Path


class SweepRunner:
    """Runs multiple quantization scenarios against a shared model & dataset."""

    def __init__(
        self,
        base_config: PTQExperimentConfig,
        scenarios: list[dict[str, Any]],
        sweep_name: str = "ptq_sweep",
        output_dir: str | Path = "outputs",
    ) -> None:
        self.base_config = base_config
        self.scenarios = scenarios
        self.sweep_name = sweep_name
        self.output_dir = Path(output_dir)

    @classmethod
    def from_yaml(
        cls,
        sweep_yaml_path: str | Path,
        global_overrides: dict[str, Any] | None = None,
    ) -> SweepRunner:
        raw = load_yaml(sweep_yaml_path)
        base_cfg_path = raw.get("base_config", "configs/base.yaml")
        base_cfg = PTQExperimentConfig.from_yaml(base_cfg_path)
        if global_overrides:
            base_cfg = base_cfg.apply_overrides(global_overrides)

        scenarios = raw.get("scenarios", [])
        if not scenarios:
            raise ValueError(f"Sweep config '{sweep_yaml_path}' contains no scenarios.")

        return cls(
            base_config=base_cfg,
            scenarios=scenarios,
            sweep_name=raw.get("sweep_name", "ptq_sweep"),
            output_dir=global_overrides.get("output_dir", raw.get("output_dir", base_cfg.output_dir))
            if global_overrides
            else raw.get("output_dir", base_cfg.output_dir),
        )

    def run(
        self,
        model: nn.Module | None = None,
        calib_loader: Iterable[Any] | None = None,
        eval_loader: Iterable[Any] | None = None,
        eval_fn: CustomEvalFn | None = None,
    ) -> SweepRunResult:
        """Execute all sweep scenarios and save aggregated tables and Pareto plot."""
        sweep_dir = create_run_dir(self.output_dir, self.sweep_name)
        logger = setup_logger("ptq_sim.sweep", log_file=sweep_dir / "sweep.log")
        try:
            logger.info("Starting PTQ sweep '%s' with %d scenario(s).", self.sweep_name, len(self.scenarios))

            set_seed(self.base_config.seed)
            if model is None:
                shared_model, _ = build_model(self.base_config.model)
            else:
                shared_model = model
                if self.base_config.model.checkpoint:
                    load_checkpoint_into_model(
                        shared_model,
                        self.base_config.model.checkpoint,
                        state_dict_key=self.base_config.model.state_dict_key,
                        strict=self.base_config.model.strict_load,
                        map_location="cpu",
                    )

            if calib_loader is None or eval_loader is None:
                built_calib, built_eval = build_dataloaders(self.base_config.data, seed=self.base_config.seed)
                calib_loader = calib_loader or built_calib
                eval_loader = eval_loader or built_eval

            rows: list[dict[str, Any]] = []
            scenario_results: dict[str, PTQRunResult] = {}

            for sc in self.scenarios:
                sc_name = str(sc["name"])
                sc_overrides = dict(sc.get("overrides", {}) or {})
                sc_overrides.setdefault("experiment_name", sc_name)
                sc_cfg = self.base_config.apply_overrides(sc_overrides)

                sc_dir = sweep_dir / sc_name
                res = PTQPipeline(sc_cfg).run(
                    model=shared_model,
                    calib_loader=calib_loader,
                    eval_loader=eval_loader,
                    eval_fn=eval_fn,
                    run_dir_override=sc_dir,
                    _skip_checkpoint_load=True,
                )
                scenario_results[sc_name] = res

                rows.append(
                    {
                        "scenario": sc_name,
                        "w_bits": sc_cfg.quantization.weight.bit_width if sc_cfg.quantization.weight.enabled else 32,
                        "a_bits": sc_cfg.quantization.activation.bit_width
                        if sc_cfg.quantization.activation.enabled
                        else 32,
                        "w_granularity": sc_cfg.quantization.weight.granularity,
                        "a_observer": sc_cfg.quantization.activation.observer,
                        "primary_metric_name": res.fp32_metrics["primary_metric_name"],
                        "fp32_primary_metric": res.fp32_metrics["primary_metric"],
                        "quant_primary_metric": res.quant_metrics["primary_metric"],
                        "metric_delta": res.metric_delta["primary_metric_delta"],
                        "fp32_size_mb": res.profile["fp32_size_mb"],
                        "quantized_size_mb": res.profile["quantized_size_mb"],
                        "compression_ratio": res.profile["compression_ratio"],
                        "avg_bits_per_param": res.profile["avg_effective_bits_per_param"],
                        "mean_act_sqnr_db": (
                            res.layer_analysis.summary.get("mean_act_sqnr_db")
                            if res.layer_analysis is not None
                            else None
                        ),
                    }
                )

            df = pd.DataFrame(rows)
            csv_path = sweep_dir / "sweep_summary.csv"
            df.to_csv(csv_path, index=False, encoding="utf-8")
            json_path = save_json(rows, sweep_dir / "sweep_summary.json")
            pareto_path = plot_bitwidth_pareto(df, sweep_dir / "bitwidth_pareto.png")

            logger.info("Sweep finished. Summary saved to '%s'.", sweep_dir)
            return SweepRunResult(
                sweep_dir=sweep_dir,
                dataframe=df,
                scenario_results=scenario_results,
                pareto_plot_path=pareto_path,
                summary_csv_path=csv_path,
                summary_json_path=json_path,
            )
        finally:
            close_file_handlers(logger)
