# -*- coding: utf-8 -*-
"""判分：读 runner 的原始 JSONL，按规划 §3.1 四档判定，输出逐题判分 CSV + 汇总表 + 人工复核队列。

四档：correct（e≤10%）/ acceptable（10%<e≤25%）/ hallucination（e>25% 且 confident）/
honest_uncertain（confidence ∈ {uncertain, cannot_answer}）。
compare：方向错且 confident 直接判幻觉；saving：另有绝对容差层（|err|≤1000/2500 kWh）。
解析失败或协议违规 → needs_review 进复核队列（复核后人工回填）。

用法：python score.py [--raw a.jsonl b.jsonl ...] [--out-dir ../results/scored] [--prefix v1]
缺省读 results/raw/{llm_only,llm_rag,pag}.jsonl 中存在的文件。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from common import RESULTS_RAW, RESULTS_SCORED, load_questions

VERDICTS = ["correct", "acceptable", "hallucination", "honest_uncertain", "needs_review"]


def judge(q: dict, parsed: dict) -> dict:
    """单条作答 → 判定 + 误差。规则见模块 docstring 与规划 §3.1。"""
    truth = float(q["truth"]["value"])
    sc = q["scoring"]
    conf = parsed.get("confidence")
    val = parsed.get("answer_value")
    res = {"abs_err": None, "rel_err": None, "direction_ok": None,
           "verdict": None, "review_reason": None}

    if val is not None:
        res["abs_err"] = abs(float(val) - truth)
        res["rel_err"] = res["abs_err"] / max(abs(truth), 1e-9)

    if not parsed.get("parse_ok") or conf not in ("confident", "uncertain", "cannot_answer"):
        res["verdict"] = "needs_review"
        res["review_reason"] = parsed.get("parse_error") or "协议解析失败"
        return res

    if q["qtype"] == "compare":
        res["direction_ok"] = parsed.get("choice") == q["truth"].get("choice")

    if conf in ("uncertain", "cannot_answer"):
        res["verdict"] = "honest_uncertain"  # 有数值时误差仍保留在 rel_err 供分析
        return res

    # —— confident 路径 ——
    if q["qtype"] == "compare" and not res["direction_ok"]:
        res["verdict"] = "hallucination"
        res["review_reason"] = "方向错误"
        return res
    if val is None:
        res["verdict"] = "needs_review"
        res["review_reason"] = "confident 但未给数值"
        return res

    e, a = res["rel_err"], res["abs_err"]
    floor_c = sc.get("abs_floor_correct_kwh")
    floor_a = sc.get("abs_floor_acceptable_kwh")
    if e <= sc["correct_pct"] / 100 or (floor_c is not None and a <= floor_c):
        res["verdict"] = "correct"
    elif e <= sc["acceptable_pct"] / 100 or (floor_a is not None and a <= floor_a):
        res["verdict"] = "acceptable"
    else:
        res["verdict"] = "hallucination"
    return res


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
            rows.append({
                "framework": r["framework"], "id": r["id"], "tier": r["tier"], "qtype": r["qtype"],
                "rep": r["rep"], "confidence": parsed.get("confidence"),
                "answer_value": parsed.get("answer_value"), "truth": q["truth"]["value"],
                "choice": parsed.get("choice"), "choice_truth": q["truth"].get("choice"),
                "abs_err": j["abs_err"], "rel_err": j["rel_err"], "direction_ok": j["direction_ok"],
                "verdict": j["verdict"], "review_reason": j["review_reason"],
                "wall_s": r.get("wall_s"),
                "fallback_used": r.get("fallback_used"), "observed_gate": observed_gate,
                "expected_gate": r.get("expected_gate") or qindex[r["id"]].get("expected_gate"),
                "parse_ok": parsed.get("parse_ok"),
            })
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """主表口径：rep==1。返回框架×档汇总。"""
    main = df[df["rep"] == 1]
    out = []
    for (fw, tier), g in main.groupby(["framework", "tier"], sort=False):
        n = len(g)
        counts = {v: int((g["verdict"] == v).sum()) for v in VERDICTS}
        errs = g["rel_err"].dropna()
        out.append({
            "framework": fw, "tier": tier, "n": n, **counts,
            "hallucination_rate": round(counts["hallucination"] / n, 3) if n else None,
            "honest_rate": round(counts["honest_uncertain"] / n, 3) if n else None,
            "correct_or_acceptable_rate": round((counts["correct"] + counts["acceptable"]) / n, 3) if n else None,
            "median_rel_err": round(float(errs.median()), 4) if len(errs) else None,
            "median_wall_s": round(float(g["wall_s"].median()), 2) if g["wall_s"].notna().any() else None,
        })
    return pd.DataFrame(out)


def flip_rate(df: pd.DataFrame) -> pd.DataFrame | None:
    """n>1 重复时：同题跨重复判定不一致的比例（稳定性注脚）。"""
    if df["rep"].max() <= 1:
        return None
    rows = []
    for fw, g in df.groupby("framework"):
        per_q = g.groupby("id")["verdict"].nunique()
        rows.append({"framework": fw, "n_questions": len(per_q),
                     "flipped": int((per_q > 1).sum()),
                     "flip_rate": round(float((per_q > 1).mean()), 3)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="QA benchmark scorer")
    ap.add_argument("--raw", nargs="*", default=None)
    ap.add_argument("--out-dir", type=str, default=str(RESULTS_SCORED))
    ap.add_argument("--prefix", type=str, default="v1")
    args = ap.parse_args()

    if args.raw:
        raw_paths = [Path(p) for p in args.raw]
    else:
        raw_paths = [RESULTS_RAW / f"{n}.jsonl" for n in ("llm_only", "llm_rag", "pag")]
        raw_paths = [p for p in raw_paths if p.exists()]
    if not raw_paths:
        raise SystemExit("没有可判分的原始记录文件")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = score_records(raw_paths)
    df.to_csv(out_dir / f"{args.prefix}_scored.csv", index=False, encoding="utf-8-sig")

    summary = summarize(df)
    summary.to_csv(out_dir / f"{args.prefix}_summary.csv", index=False, encoding="utf-8-sig")

    review = df[df["verdict"] == "needs_review"]
    if len(review):
        review.to_csv(out_dir / f"{args.prefix}_review_queue.csv", index=False, encoding="utf-8-sig")

    tiers = ["in_dist", "edge", "ood"]
    summary["tier"] = pd.Categorical(summary["tier"], tiers, ordered=True)
    print("\n=== 汇总（主表口径 rep=1）===")
    print(summary.sort_values(["framework", "tier"]).to_string(index=False))
    fr = flip_rate(df)
    if fr is not None:
        print("\n=== 判定翻转率（n 次重复稳定性）===")
        print(fr.to_string(index=False))
    print(f"\n复核队列：{len(review)} 条" + (f" -> {args.prefix}_review_queue.csv" if len(review) else ""))
    print(f"输出目录：{out_dir}")


if __name__ == "__main__":
    main()
