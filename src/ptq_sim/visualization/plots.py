"""Headless Matplotlib visualization utilities for PTQ distributions, errors, sensitivity, and sweeps."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _select_representative_layers(keys: list[str], max_layers: int) -> list[str]:
    if len(keys) <= max_layers:
        return keys
    indices = np.linspace(0, len(keys) - 1, num=max_layers, dtype=int)
    return [keys[i] for i in indices]


def _plot_distribution_grid(
    samples_map: dict[str, dict[str, Any]],
    save_path: str | Path,
    title: str,
    xlabel: str,
    max_layers: int = 6,
) -> Path | None:
    """Shared helper to plot FP32 vs Fake-Quantized distribution histograms across layers."""
    if not samples_map:
        return None

    selected = _select_representative_layers(list(samples_map.keys()), max_layers=max_layers)
    n_plots = len(selected)
    ncols = min(3, n_plots)
    nrows = math.ceil(n_plots / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5.2 * ncols, 3.8 * nrows))
    axes_arr = np.atleast_1d(axes).reshape(-1)

    for idx, layer_name in enumerate(selected):
        ax = axes_arr[idx]
        entry = samples_map[layer_name]
        fp32_vals = np.asarray(entry["fp32"], dtype=np.float32)
        quant_vals = np.asarray(entry["quant"], dtype=np.float32)
        sqnr = entry.get("sqnr_db", float("nan"))
        bits = entry.get("bit_width", "?")

        vmin = float(min(fp32_vals.min(), quant_vals.min()))
        vmax = float(max(fp32_vals.max(), quant_vals.max()))
        if vmax - vmin < 1e-8:
            vmin -= 0.5
            vmax += 0.5

        bins = np.linspace(vmin, vmax, 60)
        ax.hist(fp32_vals, bins=bins, density=True, alpha=0.55, color="#1f77b4", label="FP32")
        ax.hist(quant_vals, bins=bins, density=True, alpha=0.55, color="#ff7f0e", label=f"Quant ({bits}b)")

        # Draw clipping bounds from observed quantized range if tighter than FP32 extrema.
        q_min_v, q_max_v = float(quant_vals.min()), float(quant_vals.max())
        if q_min_v > float(fp32_vals.min()) + 1e-6:
            ax.axvline(q_min_v, color="#d62728", linestyle="--", linewidth=1.0, alpha=0.8)
        if q_max_v < float(fp32_vals.max()) - 1e-6:
            ax.axvline(q_max_v, color="#d62728", linestyle="--", linewidth=1.0, alpha=0.8)

        short_name = layer_name if len(layer_name) <= 28 else f"...{layer_name[-25:]}"
        ax.set_title(f"{short_name}\nSQNR: {sqnr:.2f} dB", fontsize=10)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.grid(True, alpha=0.25, linestyle=":")
        ax.legend(fontsize=8, loc="upper right")

    for j in range(n_plots, len(axes_arr)):
        axes_arr[j].axis("off")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_weight_distributions(
    weight_samples: dict[str, dict[str, Any]],
    save_path: str | Path,
    max_layers: int = 6,
) -> Path | None:
    """Plot FP32 vs fake-quantized weight histograms for representative layers."""
    return _plot_distribution_grid(
        samples_map=weight_samples,
        save_path=save_path,
        title="Weight Distributions: FP32 vs Quantized",
        xlabel="Weight Value",
        max_layers=max_layers,
    )


def plot_activation_distributions(
    act_samples: dict[str, dict[str, Any]],
    save_path: str | Path,
    max_layers: int = 6,
) -> Path | None:
    """Plot FP32 vs quantized output activation histograms for representative layers."""
    return _plot_distribution_grid(
        samples_map=act_samples,
        save_path=save_path,
        title="Layer Output Activations: FP32 vs Quantized",
        xlabel="Activation Value",
        max_layers=max_layers,
    )


def plot_layer_error_metrics(layer_df: pd.DataFrame, save_path: str | Path) -> Path | None:
    """Plot per-layer Weight/Activation SQNR (dB) and Cosine Similarity horizontal bar charts."""
    if layer_df is None or layer_df.empty:
        return None

    df = layer_df.copy()
    names = [n if len(n) <= 30 else f"...{n[-27:]}" for n in df["layer_name"].tolist()]
    y_pos = np.arange(len(names))
    fig_height = max(4.2, 0.42 * len(names) + 1.8)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, fig_height), sharey=True)
    bar_h = 0.36

    ax1.barh(y_pos - bar_h / 2, df["weight_sqnr_db"], height=bar_h, color="#2b7bba", label="Weight SQNR")
    ax1.barh(y_pos + bar_h / 2, df["act_sqnr_db"], height=bar_h, color="#e66101", label="Activation SQNR")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(names, fontsize=9)
    ax1.invert_yaxis()
    ax1.set_xlabel("SQNR (dB) [Higher is Better]")
    ax1.set_title("Per-Layer SQNR (dB)")
    ax1.grid(True, axis="x", alpha=0.3, linestyle=":")
    ax1.legend(fontsize=8)

    ax2.barh(y_pos - bar_h / 2, df["weight_cosine_sim"], height=bar_h, color="#1b9e77", label="Weight Cosine Sim")
    ax2.barh(y_pos + bar_h / 2, df["act_cosine_sim"], height=bar_h, color="#7570b3", label="Activation Cosine Sim")
    min_cos = float(min(df["weight_cosine_sim"].min(), df["act_cosine_sim"].min()))
    ax2.set_xlim(max(-1.0, min(0.95, min_cos - 0.05)), 1.005)
    ax2.set_xlabel("Cosine Similarity [Higher is Better]")
    ax2.set_title("Per-Layer Cosine Similarity")
    ax2.grid(True, axis="x", alpha=0.3, linestyle=":")
    ax2.legend(fontsize=8)

    fig.suptitle("Layer-Wise Quantization Fidelity", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_sensitivity_ranking(
    sens_df: pd.DataFrame,
    save_path: str | Path,
    metric_name: str = "primary_metric",
) -> Path | None:
    """Plot horizontal bar chart of single-layer quantization degradation ranking."""
    if sens_df is None or sens_df.empty:
        return None

    df = sens_df.sort_values("degradation", ascending=False).reset_index(drop=True)
    names = [n if len(n) <= 32 else f"...{n[-29:]}" for n in df["layer_name"].tolist()]
    y_pos = np.arange(len(names))
    fig_height = max(4.0, 0.4 * len(names) + 1.5)

    fig, ax = plt.subplots(figsize=(9.5, fig_height))
    colors = ["#c0392b" if v > 0 else "#27ae60" for v in df["degradation"]]
    ax.barh(y_pos, df["degradation"], color=colors, alpha=0.88)
    ax.axvline(0.0, color="#333333", linewidth=0.9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(f"Degradation in '{metric_name}' when ONLY this layer is quantized")
    ax.set_title("Single-Layer Quantization Sensitivity Ranking")
    ax.grid(True, axis="x", alpha=0.3, linestyle=":")

    fig.tight_layout()
    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_bitwidth_pareto(sweep_df: pd.DataFrame, save_path: str | Path) -> Path | None:
    """Plot trade-off curves of primary metric vs quantized size (MB) and scenario name."""
    if sweep_df is None or sweep_df.empty:
        return None

    df = sweep_df.copy()
    metric_col = "quant_primary_metric"
    baseline_col = "fp32_primary_metric"
    metric_label = str(df["primary_metric_name"].iloc[0]) if "primary_metric_name" in df.columns else "Metric"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2))

    # Panel 1: Metric vs Quantized Size (MB) with grouped & staggered annotations.
    ax1.scatter(df["quantized_size_mb"], df[metric_col], s=85, color="#1f77b4", zorder=3)
    grouped_pts: dict[tuple[float, float], tuple[float, float, list[str]]] = {}
    for _, row in df.iterrows():
        px = float(row["quantized_size_mb"])
        py = float(row[metric_col])
        key = (round(px, 4), round(py, 4))
        if key not in grouped_pts:
            grouped_pts[key] = (px, py, [])
        grouped_pts[key][2].append(str(row["scenario"]))

    sorted_pts = sorted(grouped_pts.values(), key=lambda item: (item[0], item[1]))
    for idx, (px, py, names) in enumerate(sorted_pts):
        label_text = "\n".join(names)
        place_above = (idx % 2 == 0)
        y_off = 22 if place_above else -24
        va = "bottom" if place_above else "top"
        ax1.annotate(
            label_text,
            (px, py),
            textcoords="offset points",
            xytext=(0, y_off),
            ha="center",
            va=va,
            fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#cccccc", alpha=0.88),
            arrowprops=dict(arrowstyle="-", color="#777777", lw=0.8, shrinkA=0, shrinkB=4),
        )

    x_vals = df["quantized_size_mb"].astype(float)
    y_vals = df[metric_col].astype(float)
    x_span = max(float(x_vals.max() - x_vals.min()), 1e-4)
    y_span = max(float(y_vals.max() - y_vals.min()), 0.05)
    ax1.set_xlim(float(x_vals.min()) - 0.18 * x_span, float(x_vals.max()) + 0.18 * x_span)
    ax1.set_ylim(float(y_vals.min()) - 0.55 * y_span, float(y_vals.max()) + 0.55 * y_span)

    if baseline_col in df.columns and df[baseline_col].notna().any():
        fp32_val = float(df[baseline_col].iloc[0])
        ax1.axhline(fp32_val, color="#2ca02c", linestyle="--", linewidth=1.3, label=f"FP32 Baseline ({fp32_val:.4f})")
        ax1.legend(fontsize=8, loc="upper right")
    ax1.set_xlabel("Quantized Footprint (MB)")
    ax1.set_ylabel(metric_label)
    ax1.set_title("Performance vs Footprint Trade-off")
    ax1.grid(True, alpha=0.3, linestyle=":")

    # Panel 2: Bar comparison across scenarios.
    x_pos = np.arange(len(df))
    ax2.bar(x_pos, df[metric_col], color="#ff7f0e", alpha=0.88)
    if baseline_col in df.columns and df[baseline_col].notna().any():
        fp32_val = float(df[baseline_col].iloc[0])
        ax2.axhline(fp32_val, color="#2ca02c", linestyle="--", linewidth=1.3, label=f"FP32 ({fp32_val:.4f})")
        ax2.legend(fontsize=8)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(df["scenario"], rotation=30, ha="right", fontsize=8)
    ax2.set_ylabel(metric_label)
    ax2.set_title("Scenario Comparison")
    ax2.grid(True, axis="y", alpha=0.3, linestyle=":")

    fig.suptitle("PTQ Bit-Width & Calibrator Sweep", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
