"""Render factual Pareto comparisons from tools, with priorities selected by the LLM.

This is an explicit tool-backed presentation, not a claim that free-form LLM
prose has been fact-checked. No candidate IDs or building results are hardcoded.
"""
from __future__ import annotations

from .comparison_summary import build_comparison_summary


PRESENTATION_SPEC = {
    "type": "function",
    "function": {
        "name": "present_pareto_comparison",
        "description": "Finish the current Pareto answer with computed tables, rankings and conditional trade-offs. If the user requested feature drivers, call explain_features first. Select discussion priority order from the user's intent; this does not change optimization weights. The returned presentation is shown directly, without another prose rewrite.",
        "parameters": {
            "type": "object",
            "properties": {
                "priority_order": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["cost", "electricity", "comfort"]},
                    "description": "Priorities to discuss in order; use all three when the user requests a balance.",
                },
            },
            "required": ["priority_order"],
            "additionalProperties": False,
        },
    },
}


def normalize_priorities(priorities) -> list[str]:
    valid = ("cost", "electricity", "comfort")
    if not isinstance(priorities, list):
        return list(valid)
    return list(dict.fromkeys(p for p in priorities if isinstance(p, str) and p in valid)) or list(valid)


def render_pareto_comparison(result: dict, shap: dict | None = None, *, priority_order=None, language="en") -> str:
    zh = language == "zh"
    rows = result["topSolutions"]
    summary = build_comparison_summary(rows, language=language)
    by_id = {r["id"]: r for r in rows}
    context = result.get("modelContext") or {}
    basis = " · ".join(str(context[k]) for k in ("name", "model_id", "weather_file") if context.get(k))
    fmt = lambda value: f"{value:,}"
    lines = [f"**{'计算依据' if zh else 'Basis'}:** {basis}", ""]
    if zh:
        lines += [f"比较 {result['candidateCount']} 个候选方案，其中 {result['paretoCount']} 个非支配；下表展示按成本升序选出的 {len(rows)} 个方案。以下排序仅针对展示方案，并非综合评分或全空间最优。", "",
                  "| 方案 | 成本（元） | 平均用电（kWh/年） | 稳健目标（kWh/年） | 节能率 | 不舒适（h/年） |"]
    else:
        lines += [f"Screened {result['candidateCount']} candidates; {result['paretoCount']} are non-dominated. The table shows the {len(rows)} lowest-cost options from that candidate Pareto front. All comparisons below concern these displayed options, not an overall score or a global optimum.", "",
                  "| ID | Cost (CNY) | Mean electricity (kWh/year) | Robust objective (kWh/year) | Savings | Discomfort (h/year) |"]
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(f"| {row['id']} | {fmt(row['retrofitCostCny'])} | {fmt(row['annualTotalInputKwh'])} | {fmt(row['annualRobustTotalInputKwh'])} | {row['savingsVsBaselinePct']:.2f}% | {row['annualDiscomfortHours']:.2f} |")
    baseline = fmt(result["baselineHvacInputKwh"])
    lines += ["", (f"节能率按含不确定性惩罚的稳健用电目标相对基线 {baseline} kWh/年计算，不等于平均预测用电的节能率。" if zh else f"Savings use the uncertainty-penalized robust electricity objective relative to the {baseline} kWh/year baseline, rather than the mean prediction."), "",
              "**展示方案的比较摘要**" if zh else "**Computed comparisons within the displayed options**", ""]
    lines.extend("- " + item for item in summary["highlights"])
    lines += ["", "**按偏好选择与取舍**" if zh else "**Conditional choices and trade-offs**", ""]
    axes = {
        "cost": ("retrofitCostCny", "优先降低投资" if zh else "If upfront cost is the priority"),
        "electricity": ("annualRobustTotalInputKwh", "优先降低稳健用电目标" if zh else "If robust electricity is the priority"),
        "comfort": ("annualDiscomfortHours", "优先降低不舒适时数" if zh else "If comfort is the priority"),
    }
    # Each preference gets the actual extremum and all three objectives, avoiding
    # invented 'balanced', 'best value', causal, or Pareto-dominance judgments.
    for priority in normalize_priorities(priority_order):
        field, label = axes[priority]
        minimum = summary["metrics"][field]["minimum"]
        if minimum is None:
            continue
        for sid in minimum["ids"]:
            row = by_id[sid]
            details = (f"成本 {fmt(row['retrofitCostCny'])} 元，稳健用电 {fmt(row['annualRobustTotalInputKwh'])} kWh/年，节能率 {row['savingsVsBaselinePct']:.2f}%，不舒适 {row['annualDiscomfortHours']:.2f} h/年。" if zh else f"cost CNY {fmt(row['retrofitCostCny'])}, robust electricity {fmt(row['annualRobustTotalInputKwh'])} kWh/year, savings {row['savingsVsBaselinePct']:.2f}%, discomfort {row['annualDiscomfortHours']:.2f} h/year.")
            lines.append(f"- {label}: **#{sid}** — {details}")
    lines += ["", ("所有展示方案均属于本次候选集的非支配前沿。未计算综合权重或单位投资收益评分，兼顾多个目标时应按可接受的预算和不舒适时数进一步筛选。" if zh else "All displayed options belong to the non-dominated front of the candidate set. No overall weights or value-per-cost score were calculated. For a compromise, narrow the table using an acceptable budget and discomfort level." )]
    if shap and shap.get("target") == "energy_total" and shap.get("topFeatures"):
        lines += ["", "**用电影响因素（全局 SHAP）**" if zh else "**Drivers of electricity use (global SHAP)**", "",
                  "| 变量 | 重要度 | 高取值对应的贡献方向 |" if zh else "| Variable | Importance | Contribution pattern at high values |",
                  "|---|---:|---|"]
        directions = ({"negative": "通常为负，相对参考点降低指标", "positive": "通常为正，相对参考点提高指标", "mixed": "方向混合或关联较弱"} if zh else {"negative": "Generally negative; lower target relative to reference", "positive": "Generally positive; higher target relative to reference", "mixed": "Mixed or weak directional pattern"})
        for feat in shap["topFeatures"]:
            label = str(feat.get("label") or feat["feature"]).replace("|", "\\|")
            lines.append(f"| {label} | {feat['importancePct']:.2f}% | {directions.get(feat.get('directionCode'), directions['mixed'])} |")
        lines += ["", ("重要度为平均绝对 SHAP 的归一化占比，不是能耗占比或节能率。高取值的正负趋势相对于归因参考点；全局关联不证明因果或全范围单调性，也不能确定某个具体方案的性能成因。" if zh else "Importance is the normalized share of mean absolute SHAP, not an energy share or savings rate. Signs are relative to the attribution reference. These global associations do not establish causality, monotonicity across the full range, or the cause of an individual option's performance." )]
    return "\n".join(lines)
