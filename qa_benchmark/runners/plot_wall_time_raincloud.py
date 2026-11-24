# -*- coding: utf-8 -*-
"""Plot wall-clock time raincloud plots for the four-model QA benchmark."""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator
from scipy.stats import gaussian_kde


HERE = Path(__file__).resolve().parent
QA_ROOT = HERE.parent
DATA_PATH = QA_ROOT / "results" / "model_sweep" / "scored" / "all_models_rep1_scored.csv"
OUT_DIR = QA_ROOT / "results" / "model_sweep" / "figures"
OUT_STEM = OUT_DIR / "wall_time_raincloud_by_framework_model"
SUMMARY_CSV = OUT_DIR / "wall_time_raincloud_summary.csv"

MM = 1 / 25.4
FIG_WIDTH = 140 * MM
FIG_HEIGHT = 70 * MM

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


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df[df["rep"].astype(int) == 1].copy()
    df = df[df["wall_s"].notna()].copy()
    df = df[df["model_alias"].isin(MODEL_ORDER) & df["framework"].isin(FRAMEWORK_ORDER)].copy()
    df["model_alias"] = pd.Categorical(df["model_alias"], MODEL_ORDER, ordered=True)
    df["framework"] = pd.Categorical(df["framework"], FRAMEWORK_ORDER, ordered=True)
    return df.sort_values(["framework", "model_alias", "tier", "id"])


def write_summary(df: pd.DataFrame) -> None:
    summary = (
        df.groupby(["framework", "model_alias"], observed=True)["wall_s"]
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


def draw_half_violin(ax: plt.Axes, values: np.ndarray, x_center: float, color: str) -> None:
    if len(values) < 3 or np.isclose(values.min(), values.max()):
        return
    y_grid = np.linspace(max(0.0, values.min() - 4.0), values.max() + 4.0, 180)
    kde = gaussian_kde(values)
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


def draw_group_box(ax: plt.Axes, values: np.ndarray, x_center: float, color: str) -> None:
    q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
    iqr = q3 - q1
    lower = values[values >= q1 - 1.5 * iqr].min()
    upper = values[values <= q3 + 1.5 * iqr].max()
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

    mean_y = float(np.mean(values))
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
    rng = np.random.default_rng(20260706)
    positions = np.arange(1, len(MODEL_ORDER) + 1)

    for panel_idx, (ax, framework) in enumerate(zip(axes, FRAMEWORK_ORDER)):
        sub = df[df["framework"] == framework]
        arrays = [
            sub.loc[sub["model_alias"] == model, "wall_s"].to_numpy(dtype=float)
            for model in MODEL_ORDER
        ]

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
            for model, values in zip(MODEL_ORDER, arrays)
        ]
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor")
        ax.set_title(f"({chr(ord('a') + panel_idx)}) {FRAMEWORK_LABELS[framework]}", loc="left", pad=3)
        ax.set_xlim(0.48, len(MODEL_ORDER) + 0.82)

        ax.set_ylim(0, 180)
        ax.yaxis.set_major_locator(MultipleLocator(30))
        ax.yaxis.set_minor_locator(MultipleLocator(10))
        ax.grid(True, axis="y", which="major", color="#CBD5E1", linewidth=0.45, alpha=0.65)
        ax.grid(True, axis="y", which="minor", color="#E2E8F0", linewidth=0.28, alpha=0.35)
        ax.tick_params(axis="both", which="major", length=3.5, width=0.6)
        ax.tick_params(axis="y", which="minor", length=2.2, width=0.5)
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)

    axes[0].set_ylabel("Wall time / s")
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
