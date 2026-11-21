"""Artifact export for PTQ metrics, reports, qparams, simulated & integer checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from ptq_sim.config.schema import PTQExperimentConfig
from ptq_sim.evaluation.layer_analyzer import LayerAnalysisResult
from ptq_sim.evaluation.sensitivity import SensitivityResult
from ptq_sim.quantization.preparer import ModelQuantizer
from ptq_sim.utils.io import save_json, to_serializable


def _format_num(val: Any, digits: int = 4) -> str:
    if val is None:
        return "N/A"
    if isinstance(val, (int, float)):
        return f"{val:.{digits}f}" if isinstance(val, float) else str(val)
    return str(val)


class ArtifactExporter:
    """Exports structured reports, CSV diagnostics, qparams, and PyTorch model packages."""

    @staticmethod
    def build_qparams_dict(quant_model: nn.Module) -> dict[str, Any]:
        """Collect per-layer quantization parameters and schemes."""
        layers = ModelQuantizer.get_quantized_layers(quant_model)
        return {name: mod.export_qparams() for name, mod in layers.items()}

    @staticmethod
    def build_integer_package(quant_model: nn.Module, cfg: PTQExperimentConfig) -> dict[str, Any]:
        """Pack quantized integer weights, scales, zero_points, biases, and float residuals."""
        layers = ModelQuantizer.get_quantized_layers(quant_model)
        quant_weight_param_ids: set[int] = set()
        quant_layers_payload: dict[str, Any] = {}

        for name, mod in layers.items():
            quant_weight_param_ids.add(id(mod.weight))
            int_w, scale, zp = mod.get_quantized_int_weight()
            layer_entry: dict[str, Any] = {
                "layer_type": mod.__class__.__name__,
                "int_weight": int_w,
                "scale": scale,
                "zero_point": zp,
                "bit_width": int(mod.weight_quantizer.cfg.bit_width),
                "signed": bool(mod.weight_quantizer.cfg.signed),
                "symmetric": bool(mod.weight_quantizer.cfg.symmetric),
                "narrow_range": bool(mod.weight_quantizer.cfg.narrow_range),
                "granularity": mod.weight_quantizer.cfg.granularity,
                "group_size": int(mod.weight_quantizer.cfg.group_size),
                "bias": mod.bias.detach().cpu().clone() if mod.bias is not None else None,
            }
            if mod.input_quantizer is not None and mod.input_quantizer.cfg.enabled:
                layer_entry["input_activation"] = {
                    "scale": mod.input_quantizer.scale.detach().cpu().clone(),
                    "zero_point": mod.input_quantizer.zero_point.detach().cpu().clone(),
                    "bit_width": int(mod.input_quantizer.cfg.bit_width),
                    "dynamic": bool(mod.input_quantizer.cfg.dynamic),
                    "granularity": mod.input_quantizer.cfg.granularity,
                }
            quant_layers_payload[name] = layer_entry

        # Preserve remaining non-quantized parameters (biases of skipped layers, norms, embeddings).
        non_quant_state: dict[str, torch.Tensor] = {}
        quant_layer_prefixes = tuple(f"{k}." for k in layers)
        for full_key, tensor in quant_model.state_dict().items():
            if full_key.startswith(quant_layer_prefixes):
                continue
            non_quant_state[full_key] = tensor.detach().cpu().clone()

        return {
            "format": "ptq_sim_integer_package_v1",
            "config": cfg.to_dict(),
            "quantized_layers": quant_layers_payload,
            "non_quantized_state_dict": non_quant_state,
        }

    @staticmethod
    def _build_markdown_report(
        cfg: PTQExperimentConfig,
        summary_payload: dict[str, Any],
        layer_result: LayerAnalysisResult | None,
        sens_result: SensitivityResult | None,
        artifact_paths: dict[str, str],
    ) -> str:
        fp32 = summary_payload["fp32_metrics"]
        q_m = summary_payload["quantized_metrics"]
        delta = summary_payload["metric_delta"]
        prof = summary_payload["profile"]
        prep = summary_payload["preparation"]

        lines: list[str] = [
            f"# PTQ Simulation Report: {cfg.experiment_name}",
            "",
            "## Configuration Summary",
            f"- **Model**: `{cfg.model.target or cfg.model.name}` (checkpoint: `{cfg.model.checkpoint or 'none/init'}`)",
            f"- **Dataset**: `{cfg.data.target or cfg.data.name}` (task: `{cfg.evaluation.task}`)",
            f"- **Weight Quantization**: enabled={cfg.quantization.weight.enabled}, "
            f"{cfg.quantization.weight.bit_width}-bit, "
            f"sym={cfg.quantization.weight.symmetric}, "
            f"granularity={cfg.quantization.weight.granularity} "
            f"(group_size={cfg.quantization.weight.group_size}), "
            f"observer={cfg.quantization.weight.observer}",
            f"- **Activation Quantization**: enabled={cfg.quantization.activation.enabled}, "
            f"{cfg.quantization.activation.bit_width}-bit, "
            f"sym={cfg.quantization.activation.symmetric}, "
            f"dynamic={cfg.quantization.activation.dynamic}, "
            f"granularity={cfg.quantization.activation.granularity}, "
            f"observer={cfg.quantization.activation.observer}",
            f"- **BatchNorm Folding**: enabled={cfg.quantization.fold_bn} "
            f"(folded {len(prep.get('folded_bn_pairs', []))} pairs)",
            f"- **Quantized Layers**: {len(prep.get('quantized_layers', []))} "
            f"(skipped: {len(prep.get('skipped_layers', []))})",
            "",
            "## Performance Comparison (FP32 vs Quantized)",
            "",
            "| Metric | FP32 | Quantized | Delta (Quant - FP32) |",
            "|---|---:|---:|---:|",
        ]

        primary_name = fp32.get("primary_metric_name", "primary_metric")
        lines.append(
            f"| **{primary_name}** | {_format_num(fp32.get('primary_metric'))} | "
            f"{_format_num(q_m.get('primary_metric'))} | {_format_num(delta.get('primary_metric_delta'))} |"
        )
        for key in ("top1_acc", "top5_acc", "loss", "mse", "rmse", "mae", "perplexity", "avg_latency_ms"):
            if key == primary_name:
                continue
            if key in fp32 or key in q_m:
                v_fp = fp32.get(key)
                v_q = q_m.get(key)
                v_d = (v_q - v_fp) if (isinstance(v_fp, (int, float)) and isinstance(v_q, (int, float))) else None
                lines.append(f"| {key} | {_format_num(v_fp)} | {_format_num(v_q)} | {_format_num(v_d)} |")

        lines.extend(
            [
                "",
                "## Footprint & Compression Profile",
                f"- **Total Parameters**: {prof.get('total_params', 0):,} "
                f"(quantized weights: {prof.get('quantized_params', 0):,})",
                f"- **FP32 Model Size**: {_format_num(prof.get('fp32_size_mb'), 4)} MB",
                f"- **Quantized Model Size**: {_format_num(prof.get('quantized_size_mb'), 4)} MB "
                f"(weights: {_format_num(prof.get('weight_payload_mb'), 4)} MB, "
                f"qparams overhead: {_format_num(prof.get('qparam_overhead_mb'), 4)} MB)",
                f"- **Compression Ratio**: {_format_num(prof.get('compression_ratio'), 2)}x "
                f"({prof.get('size_reduction_pct', 0)}% reduction)",
                f"- **Avg Effective Bits / Param**: {_format_num(prof.get('avg_effective_bits_per_param'), 2)} bits",
            ]
        )
        if "bitops_reduction_ratio" in prof:
            lines.append(
                f"- **BitOps Reduction**: {_format_num(prof.get('bitops_reduction_ratio'), 2)}x "
                f"(FP32 {prof.get('fp32_gbitops')} GBitOps -> Quant {prof.get('quant_gbitops')} GBitOps)"
            )

        if layer_result is not None and not layer_result.dataframe.empty:
            df = layer_result.dataframe.sort_values("act_sqnr_db", ascending=True).head(5)
            lines.extend(
                [
                    "",
                    "## Lowest Activation SQNR Layers (Top 5)",
                    "",
                    "| Layer | Type | W/A Bits | Weight SQNR (dB) | Act SQNR (dB) | Act Cosine Sim |",
                    "|---|---|---:|---:|---:|---:|",
                ]
            )
            for _, row in df.iterrows():
                lines.append(
                    f"| `{row['layer_name']}` | {row['layer_type']} | "
                    f"W{int(row['weight_bits'])}A{int(row['act_bits'])} | "
                    f"{_format_num(row['weight_sqnr_db'], 2)} | "
                    f"{_format_num(row['act_sqnr_db'], 2)} | "
                    f"{_format_num(row['act_cosine_sim'], 5)} |"
                )

        if sens_result is not None and not sens_result.dataframe.empty:
            df_s = sens_result.dataframe.head(5)
            lines.extend(
                [
                    "",
                    "## Most Sensitive Layers (Single-Layer Sensitivity Top 5)",
                    "",
                    "| Rank | Layer | W/A Bits | Degradation | Layer Quant Metric |",
                    "|---:|---|---:|---:|---:|",
                ]
            )
            for _, row in df_s.iterrows():
                lines.append(
                    f"| {int(row['sensitivity_rank'])} | `{row['layer_name']}` | "
                    f"W{int(row['weight_bits'])}A{int(row['act_bits'])} | "
                    f"{_format_num(row['degradation'], 5)} | "
                    f"{_format_num(row['layer_quant_metric'], 5)} |"
                )

        lines.extend(["", "## Generated Artifacts"])
        for k, v in artifact_paths.items():
            lines.append(f"- `{k}`: `{v}`")
        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def export_all(
        run_dir: str | Path,
        cfg: PTQExperimentConfig,
        quant_model: nn.Module,
        summary_payload: dict[str, Any],
        layer_result: LayerAnalysisResult | None = None,
        sens_result: SensitivityResult | None = None,
        plot_paths: dict[str, str] | None = None,
    ) -> dict[str, str]:
        """Persist resolved config, metrics, CSVs, qparams, checkpoints, and Markdown report."""
        out_dir = Path(run_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        artifacts: dict[str, str] = {}

        # 1. Config
        cfg_path = cfg.to_yaml(out_dir / "config_resolved.yaml")
        artifacts["config_resolved"] = str(cfg_path)

        # 2. Metrics JSON
        metrics_path = save_json(summary_payload, out_dir / "metrics_summary.json")
        artifacts["metrics_summary"] = str(metrics_path)

        # 3. Quantization params JSON
        qparams_dict = ArtifactExporter.build_qparams_dict(quant_model)
        qparams_path = save_json(qparams_dict, out_dir / "qparams.json")
        artifacts["qparams"] = str(qparams_path)

        # 4. Layer & sensitivity CSVs
        if layer_result is not None and not layer_result.dataframe.empty:
            layer_csv = out_dir / "layer_analysis.csv"
            layer_result.dataframe.to_csv(layer_csv, index=False, encoding="utf-8")
            artifacts["layer_analysis_csv"] = str(layer_csv)

        if sens_result is not None and not sens_result.dataframe.empty:
            sens_csv = out_dir / "sensitivity_analysis.csv"
            sens_result.dataframe.to_csv(sens_csv, index=False, encoding="utf-8")
            artifacts["sensitivity_analysis_csv"] = str(sens_csv)

        # 5. Simulated quantized checkpoint (reloadable into prepared quant_model).
        sim_ckpt_path = out_dir / "quantized_simulated.pt"
        torch.save(
            {
                "format": "ptq_sim_simulated_v1",
                "model_state_dict": {k: v.detach().cpu() for k, v in quant_model.state_dict().items()},
                "config": cfg.to_dict(),
                "summary": to_serializable(summary_payload),
            },
            sim_ckpt_path,
        )
        artifacts["quantized_simulated_pt"] = str(sim_ckpt_path)

        # 6. Integer weights + scale/zp package.
        int_pkg_path = out_dir / "quantized_int_package.pt"
        int_pkg = ArtifactExporter.build_integer_package(quant_model, cfg)
        torch.save(int_pkg, int_pkg_path)
        artifacts["quantized_int_package_pt"] = str(int_pkg_path)

        if plot_paths:
            for k, v in plot_paths.items():
                artifacts[f"plot_{k}"] = v

        # 7. Markdown summary report.
        report_md = ArtifactExporter._build_markdown_report(
            cfg=cfg,
            summary_payload=summary_payload,
            layer_result=layer_result,
            sens_result=sens_result,
            artifact_paths=artifacts,
        )
        report_path = out_dir / "summary_report.md"
        report_path.write_text(report_md, encoding="utf-8")
        artifacts["summary_report_md"] = str(report_path)

        return artifacts
