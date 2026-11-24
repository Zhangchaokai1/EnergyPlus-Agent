# -*- coding: utf-8 -*-
"""框架 B 的检索语料：代理模型训练分割（716 条）的参数矩阵 + 年度结果 + 全局统计。

信息对等设计（规划 §1.2-4）：B 的语料与 C 的代理模型出自同一份仿真数据的同一训练分割，
差异只在"读表检索"vs"学到的物理代理+门控"。检索 = 10 维参数归一化欧氏距离 k 近邻。

直接运行本文件可（重新）构建 corpus/train_corpus.csv 与 corpus/global_stats.json。
"""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import pandas as pd

from common import CORPUS_DIR, PARAM_COLS, PARAM_LABELS, REPO_ROOT

CORPUS_CSV = CORPUS_DIR / "train_corpus.csv"
STATS_JSON = CORPUS_DIR / "global_stats.json"

SUMMARY_CSV = REPO_ROOT / "datasets" / "office_sample_summary.csv"
SPLIT_CSV = REPO_ROOT / "models" / "office_hc_deeponet_ensemble" / "split_sample_ids.csv"

OUT_COLS = ["annual_heating_input_kwh", "annual_cooling_input_kwh", "annual_hvac_input_kwh"]
OUT_LABELS = {
    "annual_heating_input_kwh": "年供暖输入 kWh",
    "annual_cooling_input_kwh": "年供冷输入 kWh",
    "annual_hvac_input_kwh": "年空调总电耗 kWh",
}

# 归一化距离用的训练范围（office_sampling.VARIABLE_BOUNDS；键 window_u 对应列 window_u_w_m2k）
def _bounds() -> dict[str, tuple[float, float]]:
    import office_sampling as s
    b = dict(s.VARIABLE_BOUNDS)
    b["window_u_w_m2k"] = b.pop("window_u")
    return b


def build_corpus() -> pd.DataFrame:
    df = pd.read_csv(SUMMARY_CSV, encoding="utf-8-sig")
    split = pd.read_csv(SPLIT_CSV, encoding="utf-8-sig")
    train_ids = set(split.loc[split["split"] == "train", "sample_id"].astype(int))
    df = df[(df["sample_id"].astype(int).isin(train_ids)) & (df["simulation_status"] == "ok")].copy()
    # 口径：空调总电耗 = 供暖输入 + 供冷输入（annual_total_input_kwh 是含照明/设备的全楼电耗，不用）
    df["annual_hvac_input_kwh"] = df["annual_heating_input_kwh"] + df["annual_cooling_input_kwh"]
    corpus = df[["sample_id", *PARAM_COLS, *OUT_COLS]].reset_index(drop=True)

    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    corpus.to_csv(CORPUS_CSV, index=False, encoding="utf-8-sig")

    bounds = _bounds()
    stats = {
        "n_samples": int(len(corpus)),
        "weather": "南京典型气象年 TMY（库内无异常气候年数据）",
        "param_bounds": {k: list(bounds[k]) for k in PARAM_COLS},
        "outputs": {
            c: {"min": round(float(corpus[c].min()), 1),
                "median": round(float(corpus[c].median()), 1),
                "mean": round(float(corpus[c].mean()), 1),
                "max": round(float(corpus[c].max()), 1)}
            for c in OUT_COLS
        },
    }
    STATS_JSON.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return corpus


@lru_cache(maxsize=1)
def _load():
    if not (CORPUS_CSV.exists() and STATS_JSON.exists()):
        build_corpus()
    corpus = pd.read_csv(CORPUS_CSV, encoding="utf-8-sig")
    stats = json.loads(STATS_JSON.read_text(encoding="utf-8"))
    bounds = _bounds()
    lo = np.array([bounds[k][0] for k in PARAM_COLS])
    hi = np.array([bounds[k][1] for k in PARAM_COLS])
    mat = (corpus[PARAM_COLS].to_numpy(dtype=float) - lo) / (hi - lo)
    return corpus, stats, mat, lo, hi


def retrieve(params: dict, k: int = 5) -> pd.DataFrame:
    """按归一化欧氏距离取 k 近邻，附 distance 列。"""
    corpus, _stats, mat, lo, hi = _load()
    q = (np.array([float(params[c]) for c in PARAM_COLS]) - lo) / (hi - lo)
    d = np.sqrt(((mat - q) ** 2).sum(axis=1))
    idx = np.argsort(d)[:k]
    rows = corpus.iloc[idx].copy()
    rows.insert(0, "distance", np.round(d[idx], 3))
    return rows


def _table(rows: pd.DataFrame) -> str:
    heads = ["归一化距离", *[PARAM_LABELS[c] for c in PARAM_COLS], *[OUT_LABELS[c] for c in OUT_COLS]]
    lines = ["| " + " | ".join(heads) + " |",
             "| " + " | ".join(["---"] * len(heads)) + " |"]
    for _, r in rows.iterrows():
        cells = [f"{r['distance']:.3f}"]
        cells += [f"{r[c]:g}" for c in PARAM_COLS]
        cells += [f"{r[c]:.1f}" for c in OUT_COLS]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _queries_for_question(q: dict) -> list[tuple[str, dict]]:
    """每题的检索查询：(标签, 参数字典) 列表。查询参数取题面结构化参数（B 免语言解析，视为其优势）。"""
    qtype = q["qtype"]
    if qtype == "compare":
        return [("方案 A", q["params_a"]), ("方案 B", q["params_b"])]
    if qtype == "saving":
        base = dict(q["params"])
        after = dict(base)
        after[q["change"]["param"]] = q["change"]["to"]
        return [("当前方案（改动前）", base), ("改动后方案", after)]
    return [("题目方案", q["params"])]


def format_retrieval_block(q: dict, k: int = 5) -> str:
    """组装某题的完整检索资料块（若干近邻表 + 全局统计）。"""
    _corpus, stats, _mat, _lo, _hi = _load()
    parts = []
    for label, params in _queries_for_question(q):
        rows = retrieve(params, k=k)
        parts.append(f"#### 检索：与「{label}」最接近的 {k} 条样本\n{_table(rows)}")
    b = stats["param_bounds"]
    rng = "；".join(f"{PARAM_LABELS[c]} {b[c][0]:g}–{b[c][1]:g}" for c in PARAM_COLS)
    o = stats["outputs"]
    parts.append(
        f"#### 样本库全局统计（{stats['n_samples']} 条，全部为{stats['weather']}）\n"
        f"- 参数覆盖范围：{rng}\n"
        + "\n".join(
            f"- {OUT_LABELS[c]}：min {o[c]['min']:.0f} / 中位 {o[c]['median']:.0f} / "
            f"均值 {o[c]['mean']:.0f} / max {o[c]['max']:.0f}"
            for c in OUT_COLS
        )
    )
    return "\n\n".join(parts)


if __name__ == "__main__":
    c = build_corpus()
    print(f"corpus built: {len(c)} rows -> {CORPUS_CSV}")
    print(STATS_JSON.read_text(encoding="utf-8"))
