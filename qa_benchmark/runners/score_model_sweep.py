# -*- coding: utf-8 -*-
"""Score multi-model QA benchmark outputs.

This keeps the original judge() rules from score.py and adds model_alias as a
first-class grouping dimension. It can also include the completed DeepSeek
baseline by reading results/raw/{llm_only,llm_rag,pag}.jsonl.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from common import QA_ROOT, RESULTS_RAW, load_questions
from score import VERDICTS, flip_rate, judge

MODEL_SWEEP_ROOT = QA_ROOT / "results" / "model_sweep"
RAW_ROOT = MODEL_SWEEP_ROOT / "raw"
SCORED_ROOT = MODEL_SWEEP_ROOT / "scored"


def discover_model_sweep_raw() -> list[Path]:
    if not RAW_ROOT.exists():
        return []
    return sorted(RAW_ROOT.glob("*/*.jsonl"))


def deepseek_raw() -> list[Path]:
    return [p for p in (RESULTS_RAW / n for n in ("llm_only.jsonl", "llm_rag.jsonl", "pag.jsonl")) if p.exists()]


def score_records(raw_paths: list[Path]) -> pd.DataFrame:
    qindex = {q["id"]: q for q in load_questions()}
    rows = []
    for path in raw_paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            q = qindex[r["id"]]
            parsed = r.get("parsed") or {}
            j = judge(q, parsed)
            gates = r.get("gate_verdicts") or []
            observed_gate = ("fallback" if any(g.get("verdict") == "fallback" for g in gates)
                             else ("trust" if gates else None))
            model_alias = r.get("model_alias") or "deepseek_chat"
            rows.append({
                "model_alias": model_alias,
                "provider_label": r.get("provider_label") or ("deepseek" if model_alias == "deepseek_chat" else None),
                "model": r.get("model") or ("deepseek-chat" if model_alias == "deepseek_chat" else None),
                "base_host": r.get("base_host"),
                "run_group": r.get("run_group"),
                "framework": r["framework"],
                "id": r["id"],
                "tier": r["tier"],
                "qtype": r["qtype"],
                "rep": int(r["rep"]),
                "confidence": parsed.get("confidence"),
                "answer_value": parsed.get("answer_value"),
                "truth": q["truth"]["value"],
                "choice": parsed.get("choice"),
                "choice_truth": q["truth"].get("choice"),
                "abs_err": j["abs_err"],
                "rel_err": j["rel_err"],
                "direction_ok": j["direction_ok"],
                "verdict": j["verdict"],
                "review_reason": j["review_reason"],
                "wall_s": r.get("wall_s"),
                "fallback_used": r.get("fallback_used"),
                "observed_gate": observed_gate,
                "expected_gate": r.get("expected_gate") or q.get("expected_gate"),
                "parse_ok": parsed.get("parse_ok"),
            })
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    main = df[df["rep"] == 1]
    out = []
    for (model_alias, framework, tier), g in main.groupby(["model_alias", "framework", "tier"], sort=False):
        n = len(g)
        counts = {v: int((g["verdict"] == v).sum()) for v in VERDICTS}
        errs = g["rel_err"].dropna()
        out.append({
            "model_alias": model_alias,
            "model": g["model"].dropna().iloc[0] if g["model"].notna().any() else None,
            "framework": framework,
            "tier": tier,
            "n": n,
            **counts,
            "hallucination_rate": round(counts["hallucination"] / n, 3) if n else None,
            "honest_rate": round(counts["honest_uncertain"] / n, 3) if n else None,
            "correct_or_acceptable_rate": round((counts["correct"] + counts["acceptable"]) / n, 3) if n else None,
            "median_rel_err": round(float(errs.median()), 4) if len(errs) else None,
            "median_wall_s": round(float(g["wall_s"].median()), 2) if g["wall_s"].notna().any() else None,
        })
    return pd.DataFrame(out)


def model_flip_rate(df: pd.DataFrame) -> pd.DataFrame | None:
    if df.empty or df["rep"].max() <= 1:
        return None
    rows = []
    for (model_alias, framework), g in df.groupby(["model_alias", "framework"], sort=False):
        per_q = g.groupby("id")["verdict"].nunique()
        rows.append({
            "model_alias": model_alias,
            "framework": framework,
            "n_questions": len(per_q),
            "flipped": int((per_q > 1).sum()),
            "flip_rate": round(float((per_q > 1).mean()), 3),
        })
    return pd.DataFrame(rows)


def delta_vs_deepseek(summary: pd.DataFrame) -> pd.DataFrame:
    metric = "correct_or_acceptable_rate"
    base = summary[summary["model_alias"] == "deepseek_chat"][
        ["framework", "tier", metric, "median_rel_err", "median_wall_s"]
    ].rename(columns={
        metric: "deepseek_usable_rate",
        "median_rel_err": "deepseek_median_rel_err",
        "median_wall_s": "deepseek_median_wall_s",
    })
    comp = summary[summary["model_alias"] != "deepseek_chat"].merge(base, on=["framework", "tier"], how="left")
    if comp.empty:
        return comp
    comp["delta_usable_rate"] = (comp[metric] - comp["deepseek_usable_rate"]).round(3)
    comp["delta_median_rel_err"] = (comp["median_rel_err"] - comp["deepseek_median_rel_err"]).round(4)
    comp["delta_median_wall_s"] = (comp["median_wall_s"] - comp["deepseek_median_wall_s"]).round(2)
    return comp


def main() -> None:
    ap = argparse.ArgumentParser(description="Score multi-model QA benchmark outputs")
    ap.add_argument("--raw", nargs="*", default=None)
    ap.add_argument("--include-deepseek", action="store_true")
    ap.add_argument("--out-dir", default=str(SCORED_ROOT))
    ap.add_argument("--prefix", default="model_sweep")
    args = ap.parse_args()

    raw_paths = [Path(p) for p in args.raw] if args.raw else discover_model_sweep_raw()
    if args.include_deepseek:
        raw_paths = deepseek_raw() + raw_paths
    raw_paths = [p for p in raw_paths if p.exists()]
    if not raw_paths:
        raise SystemExit("没有可判分的原始记录文件")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = score_records(raw_paths)
    df.to_csv(out_dir / f"{args.prefix}_scored.csv", index=False, encoding="utf-8-sig")

    summary = summarize(df)
    tiers = ["in_dist", "edge", "ood"]
    summary["tier"] = pd.Categorical(summary["tier"], tiers, ordered=True)
    summary = summary.sort_values(["model_alias", "framework", "tier"])
    summary.to_csv(out_dir / f"{args.prefix}_summary.csv", index=False, encoding="utf-8-sig")

    review = df[df["verdict"] == "needs_review"]
    if len(review):
        review.to_csv(out_dir / f"{args.prefix}_review_queue.csv", index=False, encoding="utf-8-sig")

    delta = delta_vs_deepseek(summary)
    if len(delta):
        delta.to_csv(out_dir / f"{args.prefix}_delta_vs_deepseek.csv", index=False, encoding="utf-8-sig")

    fr = model_flip_rate(df)
    if fr is not None:
        fr.to_csv(out_dir / f"{args.prefix}_flip_rate.csv", index=False, encoding="utf-8-sig")

    print("\n=== 多模型汇总（主表口径 rep=1）===")
    print(summary.to_string(index=False))
    print(f"\n复核队列：{len(review)} 条" + (f" -> {args.prefix}_review_queue.csv" if len(review) else ""))
    if len(delta):
        print(f"DeepSeek 对照差值：{args.prefix}_delta_vs_deepseek.csv")
    print(f"输出目录：{out_dir}")


if __name__ == "__main__":
    main()
