"""Command-line entrypoints for running PTQ simulations and bit-width sweeps."""

from __future__ import annotations

import argparse
from typing import Sequence

from ptq_sim.config.schema import PTQExperimentConfig
from ptq_sim.models.registry import list_registered_models
from ptq_sim.data.registry import list_registered_datasets
from ptq_sim.pipeline.runner import PTQPipeline, SweepRunner


def _parse_kv_overrides(pairs: list[str] | None) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in pairs or []:
        if "=" not in item:
            raise ValueError(f"Override '{item}' must be in 'key.path=value' format.")
        key, val = item.split("=", 1)
        parsed[key.strip()] = val.strip()
    return parsed


def _collect_common_overrides(args: argparse.Namespace) -> dict[str, object]:
    overrides: dict[str, object] = {}
    if getattr(args, "model", None):
        overrides["model.name"] = args.model
        overrides["model.target"] = None
    if getattr(args, "target", None):
        overrides["model.target"] = args.target
    if getattr(args, "checkpoint", None):
        overrides["model.checkpoint"] = args.checkpoint
    if getattr(args, "state_dict_key", None):
        overrides["model.state_dict_key"] = args.state_dict_key
    if getattr(args, "dataset", None):
        overrides["data.name"] = args.dataset
        overrides["data.target"] = None
    if getattr(args, "device", None):
        overrides["device"] = args.device
    if getattr(args, "output_dir", None):
        overrides["output_dir"] = args.output_dir
    if getattr(args, "seed", None) is not None:
        overrides["seed"] = args.seed

    if getattr(args, "override", None):
        overrides.update(_parse_kv_overrides(args.override))
    return overrides


def build_ptq_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run modular Post-Training Quantization (PTQ) simulation."
    )
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to YAML config.")
    parser.add_argument("--experiment-name", type=str, default=None, help="Override experiment name.")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"Model name or torchvision:<arch>. Built-in: {list_registered_models()}",
    )
    parser.add_argument("--target", type=str, default=None, help="Dynamic import 'module:ClassName' or 'file.py:ClassName'.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to .pt/.pth state_dict checkpoint.")
    parser.add_argument("--state-dict-key", type=str, default=None, help="Nested state_dict key inside checkpoint.")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help=f"Dataset name. Built-in: {list_registered_datasets()}",
    )
    parser.add_argument("--w-bits", type=int, default=None, help="Weight bit-width (2..16).")
    parser.add_argument("--a-bits", type=int, default=None, help="Activation bit-width (2..16).")
    parser.add_argument(
        "--w-granularity",
        type=str,
        choices=["per_tensor", "per_channel", "per_group"],
        default=None,
        help="Weight quantization granularity.",
    )
    parser.add_argument(
        "--a-granularity",
        type=str,
        choices=["per_tensor", "per_channel", "per_group"],
        default=None,
        help="Activation quantization granularity.",
    )
    parser.add_argument("--group-size", type=int, default=None, help="Group size for per_group quantization.")
    parser.add_argument(
        "--w-observer",
        type=str,
        choices=["minmax", "ema", "percentile", "mse", "kl"],
        default=None,
        help="Weight calibration observer.",
    )
    parser.add_argument(
        "--calibrator",
        type=str,
        choices=["minmax", "ema", "percentile", "mse", "kl"],
        default=None,
        help="Activation calibration observer.",
    )
    parser.add_argument("--weight-only", action="store_true", help="Disable activation quantization (weight-only PTQ).")
    parser.add_argument("--dynamic-act", action="store_true", help="Use dynamic activation quantization.")
    parser.add_argument("--no-fold-bn", action="store_true", help="Disable BatchNorm folding.")
    parser.add_argument("--skip-layers", nargs="*", default=None, help="Layer names or regex patterns to skip.")
    parser.add_argument("--sensitivity", action="store_true", help="Run single-layer sensitivity analysis.")
    parser.add_argument("--no-layer-analysis", action="store_true", help="Skip layer-wise error analysis.")
    parser.add_argument("--no-plots", action="store_true", help="Disable plot generation.")
    parser.add_argument("--no-export", action="store_true", help="Disable disk artifact export.")
    parser.add_argument("--device", type=str, default=None, help="Compute device ('auto', 'cuda', 'cpu').")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for output artifacts.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument(
        "--override",
        action="append",
        default=None,
        help="Generic dotted override e.g. --override quantization.weight.symmetric=false",
    )
    return parser


def run_ptq_main(argv: Sequence[str] | None = None) -> None:
    parser = build_ptq_parser()
    args = parser.parse_args(argv)

    cfg = PTQExperimentConfig.from_yaml(args.config)
    overrides = _collect_common_overrides(args)

    if args.experiment_name:
        overrides["experiment_name"] = args.experiment_name
    if args.w_bits is not None:
        overrides["quantization.weight.bit_width"] = args.w_bits
    if args.a_bits is not None:
        overrides["quantization.activation.bit_width"] = args.a_bits
    if args.w_granularity is not None:
        overrides["quantization.weight.granularity"] = args.w_granularity
    if args.a_granularity is not None:
        overrides["quantization.activation.granularity"] = args.a_granularity
    if args.group_size is not None:
        overrides["quantization.weight.group_size"] = args.group_size
        overrides["quantization.activation.group_size"] = args.group_size
    if args.w_observer is not None:
        overrides["quantization.weight.observer"] = args.w_observer
    if args.calibrator is not None:
        overrides["quantization.activation.observer"] = args.calibrator
    if args.weight_only:
        overrides["quantization.activation.enabled"] = False
        overrides["quantization.quantize_input"] = False
        overrides["quantization.quantize_output"] = False
    if args.dynamic_act:
        overrides["quantization.activation.enabled"] = True
        overrides["quantization.activation.dynamic"] = True
    if args.no_fold_bn:
        overrides["quantization.fold_bn"] = False
    if args.skip_layers is not None:
        overrides["quantization.skip_layers"] = args.skip_layers
    if args.sensitivity:
        overrides["evaluation.run_sensitivity"] = True
    if args.no_layer_analysis:
        overrides["evaluation.run_layer_analysis"] = False
    if args.no_plots:
        overrides["evaluation.generate_plots"] = False
    if args.no_export:
        overrides["evaluation.export_artifacts"] = False

    if overrides:
        cfg = cfg.apply_overrides(overrides)

    res = PTQPipeline(cfg).run()
    print("\n=== PTQ Simulation Summary ===")
    print(f"Experiment   : {cfg.experiment_name}")
    print(f"Primary      : {res.fp32_metrics['primary_metric_name']}")
    print(f"FP32         : {res.fp32_metrics['primary_metric']:.4f}")
    print(f"Quantized    : {res.quant_metrics['primary_metric']:.4f}")
    print(f"Delta        : {res.metric_delta['primary_metric_delta']:+.4f}")
    print(
        f"Footprint    : {res.profile['fp32_size_mb']:.4f} MB -> {res.profile['quantized_size_mb']:.4f} MB "
        f"({res.profile['compression_ratio']:.2f}x)"
    )
    if res.run_dir is not None:
        print(f"Run Directory: {res.run_dir}")


def build_sweep_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run multi-configuration PTQ sweep.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/sweep_bitwidths.yaml",
        help="Path to sweep YAML config.",
    )
    parser.add_argument("--model", type=str, default=None, help="Override model name.")
    parser.add_argument("--target", type=str, default=None, help="Override dynamic model target.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Override checkpoint path.")
    parser.add_argument("--state-dict-key", type=str, default=None, help="Override state_dict key.")
    parser.add_argument("--dataset", type=str, default=None, help="Override dataset name.")
    parser.add_argument("--device", type=str, default=None, help="Compute device.")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--override", action="append", default=None, help="Generic dotted override key=value.")
    return parser


def run_sweep_main(argv: Sequence[str] | None = None) -> None:
    parser = build_sweep_parser()
    args = parser.parse_args(argv)
    overrides = _collect_common_overrides(args)
    runner = SweepRunner.from_yaml(args.config, global_overrides=overrides or None)
    res = runner.run()
    print("\n=== PTQ Sweep Results ===")
    print(res.dataframe.to_string(index=False))
    print(f"\nSweep Directory: {res.sweep_dir}")
