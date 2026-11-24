"""Deterministic comparisons of the displayed Pareto options (no LLM ranking)."""
from __future__ import annotations

from itertools import groupby
from math import isfinite


METRICS = {
    "retrofitCostCny": ("CNY", "lower"),
    "annualTotalInputKwh": ("kWh/year", "lower"),
    "annualRobustTotalInputKwh": ("kWh/year", "lower"),
    "annualDiscomfortHours": ("h/year", "lower"),
    "savingsVsBaselinePct": ("%", "higher"),
}


def build_comparison_summary(solutions: list[dict], *, language: str = "en") -> dict:
    """Use the same rounded values the user sees; preserve ties, including at rank 2."""
    ids = [row["id"] for row in solutions]
    if len(set(ids)) != len(ids):
        raise ValueError("Comparison requires unique solution IDs")
    metrics = {}
    for field, (unit, preferred) in METRICS.items():
        if any(not isfinite(float(row[field])) for row in solutions):
            raise ValueError(f"Non-finite comparison metric: {field}")
        ordered = sorted(solutions, key=lambda row: row[field])
        groups = [
            {"value": value, "ids": [row["id"] for row in rows]}
            for value, rows in groupby(ordered, key=lambda row: row[field])
        ]
        metrics[field] = {
            "unit": unit,
            "preferredDirection": preferred,
            "ascendingGroups": groups,
            "minimum": groups[0] if groups else None,
            "maximum": groups[-1] if groups else None,
        }

    # Include everyone tied at the second position; never claim an arbitrary winner.
    lowest_two = []
    for group in metrics["annualRobustTotalInputKwh"]["ascendingGroups"]:
        lowest_two.append(group)
        if sum(len(item["ids"]) for item in lowest_two) >= 2:
            break

    def describe(group: dict) -> str:
        names = ", ".join(f"#{sid}" for sid in group["ids"])
        return f"{names} ({group['value']:,})"

    highlights = []
    if solutions:
        cost = describe(metrics["retrofitCostCny"]["minimum"])
        energy = "; ".join(describe(group) for group in lowest_two)
        discomfort = describe(metrics["annualDiscomfortHours"]["minimum"])
        savings = describe(metrics["savingsVsBaselinePct"]["minimum"])
        if language == "zh":
            highlights = [
                f"成本最低（元）：{cost}",
                f"稳健用电目标从低到高前两位（含并列；kWh/年）：{energy}",
                f"不舒适时数最低（h/年）：{discomfort}",
                f"节能率最低（%）：{savings}",
            ]
        else:
            highlights = [
                f"Lowest cost (CNY): {cost}",
                f"Lowest robust electricity, up to two positions including ties (kWh/year): {energy}",
                f"Lowest discomfort (h/year): {discomfort}",
                f"Lowest savings (%): {savings}",
            ]

    return {
        "scope": "displayed topSolutions only, not all candidates or the full Pareto front",
        "displayedCount": len(solutions),
        "displayedIds": ids,
        "selectionBasis": "lowest-cost subset of the candidate Pareto front, ordered by cost; not a weighted overall ranking",
        "rankingBasis": "displayed rounded values; equal displayed values share a rank group",
        "metrics": metrics,
        "lowestTwoRobustElectricity": lowest_two,
        "highlights": highlights,
        "energyBasis": {
            "meanPrediction": "annualTotalInputKwh",
            "uncertaintyPenalizedObjective": "annualRobustTotalInputKwh",
            "savingsNumeratorBasis": "annualRobustTotalInputKwh",
            "savingsFormula": "100 * (baselineHvacInputKwh - annualRobustTotalInputKwh) / baselineHvacInputKwh; calculated before display rounding",
        },
        "recommendationBasis": "State the user's priority (cost, robust electricity, or discomfort). No overall score, comfort-per-yuan metric, or unique best compromise has been calculated. Preserve candidate-set non-dominance.",
    }
