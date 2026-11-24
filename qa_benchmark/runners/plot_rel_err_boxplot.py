# -*- coding: utf-8 -*-
"""Plot rel_err raincloud plots for the four-model QA benchmark.

The input scored CSV may contain repeated DeepSeek records from earlier
stability runs, so this script explicitly keeps rep == 1 for a fair
four-model comparison.

Each group is drawn as a compact raincloud:
left half-violin (density), centered boxplot, and right-side jittered points.
The y position is log10(rel_err + EPS), while tick labels show original rel_err.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
from scipy.stats import gaussian_kde


HERE = Path(__file__).resolve().parent
QA_ROOT = HERE.parent
DATA_PATH = QA_ROOT / "results" / "model_sweep" / "scored" / "all_models_rep1_scored.csv"
OUT_DIR = QA_ROOT / "results" / "model_sweep" / "figures"
OUT_STEM = OUT_DIR / "rel_err_raincloud_by_framework_model"
SUMMARY_CSV = OUT_DIR / "rel_err_raincloud_summary.csv"

MM = 1 / 25.4
FIG_WIDTH = 140 * MM
FIG_HEIGHT = 70 * MM
LOG_EPS = 1e-4

MODEL_ORDER = ["deepseek_chat", "gemini_3_5_flash", "gpt_5_5", "qwen3_7_plus"]
MODEL_LABELS = {
    "deepseek_chat": "DeepSeekv4",
    "gemini_3_5_flash": "Genimi3.5",
    "gpt_5_5": "GPT5.5",
    "qwen3_7_plus": "Qwen3.7",
}

FRAMEWORK_ORDER = ["llm_only", "llm_rag", "pag"]
FRAMEWORK_LABELS = {
    "llm_only": "LLM-only",
    "llm_rag": "LLM-RAG",
    "pag": "LLM-PAG",
}

MODEL_COLORS = {
    "deepseek_chat": "#0072B2",
    "gemini_3_5_flash": "#D55E00",
    "gpt_5_5": "#009E73",
    "qwen3_7_plus": "#7C3AED",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": ["Times New Roman", "SimSun"],
            "font.sans-serif": ["SimSun", "Microsoft YaHei"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "figure.dpi": 300,
            "savefig.dpi": 600,
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 8,
            "axes.linewidth": 0.6,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def to_plot_y(values: np.ndarray) -> np.ndarray:
    return np.log10(np.asarray(values, dtype=float) + LOG_EPS)


def major_tick_formatter(value: float, _pos: int) -> str:
    rel_err = (10 ** value) - LOG_EPS
    exponent = int(np.round(np.log10(rel_err)))
    return rf"$10^{{{exponent}}}$"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df[df["rep"].astype(int) == 1].copy()
    df = df[df["rel_err"].notna()].copy()
    df = df[df["model_alias"].isin(MODEL_ORDER) & df["framework"].isin(FRAMEWORK_ORDER)].copy()
    df["model_alias"] = pd.Categorical(df["model_alias"], MODEL_ORDER, ordered=True)
    df["framework"] = pd.Categorical(df["framework"], FRAMEWORK_ORDER, ordered=True)
    return df.sort_values(["framework", "model_alias", "tier", "id"])


def write_summary(df: pd.DataFrame) -> None:
    summary = (
        df.groupby(["framework", "model_alias"], observed=True)["rel_err"]
        .agg(
            n="count",
            median="median",
            q1=lambda s: s.quantile(0.25),
            q3=lambda s: s.quantile(0.75),
            p90=lambda s: s.quantile(0.90),
            max="max",
        )
        .reset_index()
    )
    summary["framework"] = summary["framework"].map(FRAMEWORK_LABELS)
    summary["model_alias"] = summary["model_alias"].map(MODEL_LABELS)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")


def draw_half_violin(ax: plt.Axes, values_y: np.ndarray, x_center: float, color: str) -> None:
    if len(values_y) < 3 or np.isclose(values_y.min(), values_y.max()):
        return
    y_grid = np.linspace(values_y.min() - 0.15, values_y.max() + 0.15, 180)
    kde = gaussian_kde(values_y)
    density = kde(y_grid)
    density = density / density.max() * 0.36
    ax.fill_betweenx(
        y_grid,
        x_center - density,
        x_center,
        facecolor=color,
        edgecolor="none",
        alpha=0.24,
        zorder=1,
    )


def draw_group_box(ax: plt.Axes, values_y: np.ndarray, x_center: float, color: str) -> None:
    q1, median, q3 = np.quantile(values_y, [0.25, 0.5, 0.75])
    iqr = q3 - q1
    lower = values_y[values_y >= q1 - 1.5 * iqr].min()
    upper = values_y[values_y <= q3 + 1.5 * iqr].max()
    width = 0.34

    rect = plt.Rectangle(
        (x_center - width / 2, q1),
        width,
        q3 - q1,
        facecolor=color,
        edgecolor="#111827",
        linewidth=0.8,
        alpha=0.68,
        zorder=4,
    )
    ax.add_patch(rect)
    ax.plot([x_center - width / 2, x_center + width / 2], [median, median],
            color="#111827", linewidth=0.95, zorder=5)
    ax.plot([x_center, x_center], [lower, q1], color="#111827", linewidth=0.75, zorder=4)
    ax.plot([x_center, x_center], [q3, upper], color="#111827", linewidth=0.75, zorder=4)
    cap = width * 0.45
    ax.plot([x_center - cap / 2, x_center + cap / 2], [lower, lower],
            color="#111827", linewidth=0.75, zorder=4)
    ax.plot([x_center - cap / 2, x_center + cap / 2], [upper, upper],
            color="#111827", linewidth=0.75, zorder=4)

    mean_y = float(np.mean(values_y))
    ax.scatter(
        [x_center],
        [mean_y],
        marker="*",
        s=62,
        facecolor="white",
        edgecolor="#111827",
        linewidth=0.75,
        zorder=6,
    )


def draw_raincloud(df: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(
        1,
        len(FRAMEWORK_ORDER),
        figsize=(FIG_WIDTH, FIG_HEIGHT),
        sharey=True,
        constrained_layout=False,
    )
    rng = np.random.default_rng(20260705)
    positions = np.arange(1, len(MODEL_ORDER) + 1)

    for panel_idx, (ax, framework) in enumerate(zip(axes, FRAMEWORK_ORDER)):
        sub = df[df["framework"] == framework]
        arrays_raw = [
            sub.loc[sub["model_alias"] == model, "rel_err"].to_numpy(dtype=float)
            for model in MODEL_ORDER
        ]
        arrays = [to_plot_y(values) for values in arrays_raw]

        for x_pos, model, values in zip(positions, MODEL_ORDER, arrays):
            if len(values) == 0:
                continue
            color = MODEL_COLORS[model]
            draw_half_violin(ax, values, x_pos, color)
            draw_group_box(ax, values, x_pos, color)
            jitter = rng.normal(loc=0.0, scale=0.055, size=len(values))
            jitter = np.clip(jitter, -0.13, 0.13)
            ax.scatter(
                np.full(len(values), x_pos + 0.36) + jitter,
                values,
                s=15,
                facecolor=color,
                edgecolor=color,
                alpha=0.42,
                linewidths=0.55,
                zorder=3,
            )

        labels = [
            f"{MODEL_LABELS[model]}\n$(n={len(values)})$"
            for model, values in zip(MODEL_ORDER, arrays_raw)
        ]
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor")
        ax.set_title(f"({chr(ord('a') + panel_idx)}) {FRAMEWORK_LABELS[framework]}", loc="left", pad=3)
        ax.set_xlim(0.48, len(MODEL_ORDER) + 0.82)

        major_rel_ticks = np.array([1e-3, 1e-2, 1e-1, 1, 10, 100], dtype=float)
        minor_rel_ticks = np.concatenate([np.arange(2, 10) * 10.0**exp for exp in range(-4, 2)])
        ax.set_ylim(np.log10(LOG_EPS), np.log10(120 + LOG_EPS))
        ax.yaxis.set_major_locator(FixedLocator(np.log10(major_rel_ticks + LOG_EPS)))
        ax.yaxis.set_major_formatter(FuncFormatter(major_tick_formatter))
        ax.yaxis.set_minor_locator(FixedLocator(np.log10(minor_rel_ticks + LOG_EPS)))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(True, axis="y", which="major", color="#CBD5E1", linewidth=0.45, alpha=0.65)
        ax.grid(True, axis="y", which="minor", color="#E2E8F0", linewidth=0.28, alpha=0.35)
        ax.tick_params(axis="both", which="major", length=3.5, width=0.6)
        ax.tick_params(axis="y", which="minor", length=2.2, width=0.5)
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)

    axes[0].set_ylabel("Rel. err.")
    fig.tight_layout(pad=0.35, w_pad=0.75)
    return fig


def main() -> None:
    setup_style()
    df = load_data()
    write_summary(df)
    fig = draw_raincloud(df)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".svg"):
        out = OUT_STEM.with_suffix(suffix)
        fig.savefig(out, bbox_inches="tight", pad_inches=0.02, facecolor="white")
        print(out)
    print(SUMMARY_CSV)
    plt.close(fig)


if __name__ == "__main__":
    main()
