"""Energy and comfort optimization report generation routes."""
from __future__ import annotations

import json
import math
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict

from backend.paths import ROOT  # noqa: E402
REPORT_DIR = Path(__file__).resolve().parent / "generated_reports"
CHART_DIR = REPORT_DIR / "charts"
FONT_CANDIDATES = [
    Path(r"C:\Windows\Fonts\simhei.ttf"),
    Path(r"C:\Windows\Fonts\msyh.ttc"),
    Path(r"C:\Windows\Fonts\simsun.ttc"),
]

router = APIRouter(tags=["reports"])


class EnergyComfortReportRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    building_id: str | None = None
    model_id: str | None = None


def _app():
    from backend import app

    return app


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff_-]+", "_", value.strip())
    return slug.strip("_") or "report"


def _num(value: Any, digits: int = 1) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    if not math.isfinite(number):
        return "-"
    return f"{number:,.{digits}f}"


def _pdf_text(value: Any) -> str:
    text = str(value)
    replacements = {
        "㎡": "m2",
        "m²": "m2",
        "°C": "degC",
        "℃": "degC",
        "·": ".",
        "→": "->",
        "μ": "mu",
        "σ": "sigma",
        "–": "-",
        "—": "-",
        "−": "-",
        "≤": "<=",
        "≥": ">=",
        "\u202f": " ",
        "\u00a0": " ",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def _pct(value: Any, digits: int = 1) -> str:
    return f"{_num(value, digits)}%"


def _kwh(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    if abs(number) >= 10000:
        return f"{number / 10000:,.1f} 万 kWh"
    return f"{number:,.0f} kWh"


def _cny(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    if abs(number) >= 10000:
        return f"{number / 10000:,.1f} 万元"
    return f"{number:,.0f} 元"


def _load_spec(manifest: dict[str, Any]) -> dict[str, Any]:
    from building_project import default_spec_dict, project_dir

    spec = manifest.get("spec")
    if isinstance(spec, dict):
        return spec
    building_id = str(manifest.get("building_id") or "")
    spec_path = project_dir(building_id) / "building_spec.json"
    if spec_path.exists():
        try:
            return json.loads(spec_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass

    fallback = default_spec_dict()
    identity = fallback.setdefault("building_identity", {})
    identity["building_name"] = manifest.get("building_name") or manifest.get("name") or "南京高层办公默认模型"
    geometry = fallback.setdefault("geometry", {})
    for key, value in (manifest.get("geometry") or {}).items():
        if key in geometry and value is not None:
            geometry[key] = value
    return fallback


def _design_inputs_from_spec(spec: dict[str, Any]) -> dict[str, float]:
    app = _app()
    envelope = spec.get("envelope") or {}
    operation = spec.get("operation") or {}
    values = dict(app.BASE_DESIGN)
    values.update({
        "wall_insulation_m": envelope.get("wall_insulation_m", values["wall_insulation_m"]),
        "roof_insulation_m": envelope.get("roof_insulation_m", values["roof_insulation_m"]),
        "window_u_w_m2k": envelope.get("window_u_w_m2k", values["window_u_w_m2k"]),
        "window_shgc": envelope.get("window_shgc", values["window_shgc"]),
        "window_wall_ratio": envelope.get("window_wall_ratio", values["window_wall_ratio"]),
        "infiltration_ach": envelope.get("infiltration_ach", values["infiltration_ach"]),
        "cooling_setpoint_c": operation.get("cooling_setpoint_c", values["cooling_setpoint_c"]),
        "heating_setpoint_c": operation.get("heating_setpoint_c", values["heating_setpoint_c"]),
        "cooling_cop": operation.get("cooling_cop", values["cooling_cop"]),
        "heating_cop": operation.get("heating_cop", values["heating_cop"]),
    })
    return {key: float(values[key]) for key in app.STATIC_COLUMNS}


def _floor_area(spec: dict[str, Any], manifest: dict[str, Any]) -> float | None:
    manifest_area = (manifest.get("geometry") or {}).get("floor_area_m2")
    if manifest_area is not None:
        try:
            return float(manifest_area)
        except (TypeError, ValueError):
            pass
    geometry = spec.get("geometry") or {}
    try:
        area = float(geometry.get("length_m", 0)) * float(geometry.get("width_m", 0))
        if geometry.get("plan_shape") == "l_shape":
            area -= float(geometry.get("notch_length_m", 0)) * float(geometry.get("notch_width_m", 0))
        return area * int(geometry.get("floor_count", 1))
    except (TypeError, ValueError):
        return None


def _building_info(spec: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    identity = spec.get("building_identity") or {}
    geometry = spec.get("geometry") or {}
    envelope = spec.get("envelope") or {}
    operation = spec.get("operation") or {}
    dataset = manifest.get("dataset") or {}
    model = manifest.get("model") or {}
    height = (manifest.get("geometry") or {}).get("height_m")
    if height is None:
        try:
            height = float(geometry.get("ground_floor_height_m", 0)) + max(0, int(geometry.get("floor_count", 1)) - 1) * float(
                geometry.get("typical_floor_height_m", 0)
            )
        except (TypeError, ValueError):
            height = None
    return {
        "name": identity.get("building_name") or manifest.get("building_name") or "当前建筑",
        "type": identity.get("building_type") or "office",
        "city": identity.get("city") or "Nanjing",
        "weather": dataset.get("weather_file") or identity.get("weather_file") or "Nanjing TMY EPW",
        "plan_shape": geometry.get("plan_shape", "rectangle"),
        "length_m": geometry.get("length_m"),
        "width_m": geometry.get("width_m"),
        "floor_count": geometry.get("floor_count"),
        "height_m": height,
        "floor_area_m2": _floor_area(spec, manifest),
        "orientation_deg": geometry.get("orientation_deg", 0),
        "perimeter_depth_m": geometry.get("perimeter_depth_m"),
        "wall_insulation_m": envelope.get("wall_insulation_m"),
        "roof_insulation_m": envelope.get("roof_insulation_m"),
        "window_u_w_m2k": envelope.get("window_u_w_m2k"),
        "window_shgc": envelope.get("window_shgc"),
        "window_wall_ratio": envelope.get("window_wall_ratio"),
        "infiltration_ach": envelope.get("infiltration_ach"),
        "cooling_setpoint_c": operation.get("cooling_setpoint_c"),
        "heating_setpoint_c": operation.get("heating_setpoint_c"),
        "cooling_cop": operation.get("cooling_cop"),
        "heating_cop": operation.get("heating_cop"),
        "sample_count": dataset.get("sample_count"),
        "model_type": model.get("type") or "CHU-DeepONet",
        "ensemble_members": model.get("ensemble_members"),
    }


def _ensure_plot_font() -> str | None:
    from matplotlib import font_manager

    for path in FONT_CANDIDATES:
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            return font_manager.FontProperties(fname=str(path)).get_name()
    return None


def _plot_charts(report_id: str, building: dict[str, Any], prediction: dict[str, Any],
                 pareto: dict[str, Any], shap: dict[str, Any]) -> dict[str, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    CHART_DIR.mkdir(parents=True, exist_ok=True)
    font_name = _ensure_plot_font()
    if font_name:
        plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 140
    paths: dict[str, Path] = {}

    def save(name: str) -> Path:
        path = CHART_DIR / f"{report_id}_{name}.png"
        plt.tight_layout()
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        paths[name] = path
        return path

    daily = prediction.get("daily") or []
    x = list(range(1, len(daily) + 1))
    heating = [d.get("heatingInputKwh", 0) for d in daily]
    cooling = [d.get("coolingInputKwh", 0) for d in daily]
    plt.figure(figsize=(8.2, 3.6))
    plt.plot(x, heating, label="供暖输入", color="#d97706", linewidth=1.4)
    plt.plot(x, cooling, label="供冷输入", color="#0284c7", linewidth=1.4)
    plt.title("全年冷热输入曲线")
    plt.xlabel("日序")
    plt.ylabel("kWh/day")
    plt.legend()
    plt.grid(alpha=0.22)
    save("daily_curve")

    points = pareto.get("points") or []
    px = [p.get("retrofitCostCny", 0) / 10000 for p in points]
    py = [p.get("annualRobustTotalInputKwh", 0) / 10000 for p in points]
    colors = ["#0f6b8f" if p.get("pareto") else "#cbd5e1" for p in points]
    sizes = [28 if p.get("pareto") else 14 for p in points]
    plt.figure(figsize=(8.2, 3.8))
    plt.scatter(px, py, c=colors, s=sizes, alpha=0.82, edgecolors="none")
    plt.title("成本-能耗-舒适 Pareto 优化结果")
    plt.xlabel("改造成本 (万元)")
    plt.ylabel("稳健冷热输入 (万 kWh)")
    plt.grid(alpha=0.2)
    save("pareto")

    baseline = pareto.get("baseline") or {}
    chosen = _representative_solutions(pareto)
    names = ["基线"] + [label for label, _ in chosen]
    energy = [baseline.get("annual_hvac_input_kwh", 0) / 10000] + [p.get("annualRobustTotalInputKwh", 0) / 10000 for _, p in chosen]
    discomfort = [baseline.get("annual_discomfort_hours", 0)] + [p.get("annualDiscomfortHours", 0) for _, p in chosen]
    plt.figure(figsize=(8.2, 3.8))
    axis = plt.gca()
    bars = axis.bar(names, energy, color=["#64748b", "#0f6b8f", "#2f855a", "#d97706"][: len(names)], label="冷热输入")
    axis.set_ylabel("稳健冷热输入 (万 kWh)")
    axis.tick_params(axis="x", labelrotation=8)
    axis2 = axis.twinx()
    axis2.plot(names, discomfort, color="#7c3aed", marker="o", linewidth=2, label="不舒适小时")
    axis2.set_ylabel("等效不舒适小时")
    axis.set_title("典型方案量化对比")
    for bar, value in zip(bars, energy):
        axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.1f}", ha="center", va="bottom", fontsize=9)
    save("comparison")

    feats = shap.get("topFeatures") or []
    labels = [f.get("label") or f.get("feature") for f in feats][::-1]
    values = [f.get("importancePct", 0) for f in feats][::-1]
    plt.figure(figsize=(8.2, 3.8))
    plt.barh(labels, values, color="#4f46e5")
    plt.title("特征重要性分析")
    plt.xlabel("相对重要性 (%)")
    plt.grid(axis="x", alpha=0.2)
    save("shap")
    return paths


def _representative_solutions(pareto: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    front = pareto.get("front") or []
    if not front:
        return []
    best_cost = min(front, key=lambda p: p.get("retrofitCostCny", float("inf")))
    best_energy = min(front, key=lambda p: p.get("annualRobustTotalInputKwh", float("inf")))
    cost_values = [p.get("retrofitCostCny", 0) for p in front]
    energy_values = [p.get("annualRobustTotalInputKwh", 0) for p in front]
    discomfort_values = [p.get("annualDiscomfortHours", 0) for p in front]

    def norm(value: float, values: list[float]) -> float:
        lo = min(values)
        hi = max(values)
        return 0.0 if hi <= lo else (value - lo) / (hi - lo)

    balanced = min(
        front,
        key=lambda p: (
            norm(p.get("retrofitCostCny", 0), cost_values)
            + norm(p.get("annualRobustTotalInputKwh", 0), energy_values)
            + norm(p.get("annualDiscomfortHours", 0), discomfort_values)
        ),
    )
    chosen: list[tuple[str, dict[str, Any]]] = []
    for label, solution in [("低成本方案", best_cost), ("均衡方案", balanced), ("最低能耗方案", best_energy)]:
        if not any(item.get("id") == solution.get("id") for _, item in chosen):
            chosen.append((label, solution))
    return chosen


def _energy_curve_summary(prediction: dict[str, Any]) -> dict[str, Any]:
    daily = prediction.get("daily") or []
    if not daily:
        return {}
    heating_peak = max(daily, key=lambda d: d.get("heatingInputKwh", 0))
    cooling_peak = max(daily, key=lambda d: d.get("coolingInputKwh", 0))
    total_peak = max(daily, key=lambda d: d.get("totalInputKwh", 0))
    season_months = {
        "winter_heating": {12, 1, 2},
        "transition": {3, 4, 5, 10, 11},
        "summer_cooling": {6, 7, 8, 9},
    }
    season_totals = {}
    for key, months in season_months.items():
        rows = [d for d in daily if d.get("month") in months]
        season_totals[key] = {
            "heating_input_kwh": sum(float(d.get("heatingInputKwh", 0)) for d in rows),
            "cooling_input_kwh": sum(float(d.get("coolingInputKwh", 0)) for d in rows),
            "total_input_kwh": sum(float(d.get("totalInputKwh", 0)) for d in rows),
            "discomfort_hours": sum(float(d.get("discomfortHours", 0)) for d in rows),
        }
    return {
        "heating_peak_day": {
            "label": heating_peak.get("label"),
            "heating_input_kwh": heating_peak.get("heatingInputKwh"),
            "drybulb_mean_c": heating_peak.get("drybulbMeanC"),
        },
        "cooling_peak_day": {
            "label": cooling_peak.get("label"),
            "cooling_input_kwh": cooling_peak.get("coolingInputKwh"),
            "drybulb_mean_c": cooling_peak.get("drybulbMeanC"),
        },
        "total_peak_day": {
            "label": total_peak.get("label"),
            "total_input_kwh": total_peak.get("totalInputKwh"),
            "drybulb_mean_c": total_peak.get("drybulbMeanC"),
        },
        "season_totals": season_totals,
    }


def _analysis_payload(building: dict[str, Any], prediction: dict[str, Any],
                      pareto: dict[str, Any], shap: dict[str, Any]) -> dict[str, Any]:
    annual = prediction.get("annual") or {}
    representative = _representative_solutions(pareto)
    return {
        "report_instruction": "这是自动报告，不是自由问答；所有数值都来自物理代理模型、Pareto 工具和 SHAP 工具，不允许补造数值。",
        "building": building,
        "model_context": {
            "model_type": building.get("model_type"),
            "ensemble_members": building.get("ensemble_members"),
            "sample_count": building.get("sample_count"),
            "weather": building.get("weather"),
            "note": "HC-DeepONet 代理模型来自 EnergyPlus 仿真数据；报告采用 PAG 物理增强生成口径。",
        },
        "annual_prediction": {
            "heating_input_kwh": annual.get("heatingInputKwh"),
            "cooling_input_kwh": annual.get("coolingInputKwh"),
            "total_input_kwh": annual.get("totalInputKwh"),
            "discomfort_hours": annual.get("discomfortHours"),
            "heating_share_pct": annual.get("heatingInputKwh", 0) / max(annual.get("totalInputKwh", 0), 1e-9) * 100.0,
            "cooling_share_pct": annual.get("coolingInputKwh", 0) / max(annual.get("totalInputKwh", 0), 1e-9) * 100.0,
            "total_input_intensity_kwh_m2": annual.get("totalInputKwh", 0) / max(building.get("floor_area_m2") or 0, 1e-9),
            "sigma_heating_load_kwh": annual.get("sigmaHeatingLoadKwh"),
            "sigma_cooling_load_kwh": annual.get("sigmaCoolingLoadKwh"),
            "sigma_discomfort_hours": annual.get("sigmaDiscomfortHours"),
        },
        "energy_curve": _energy_curve_summary(prediction),
        "pareto": {
            "candidate_count": pareto.get("count"),
            "pareto_count": pareto.get("paretoCount"),
            "baseline": pareto.get("baseline"),
            "representative_solutions": [
                {"label": label, **(_slim_solution(solution) or {})}
                for label, solution in representative
            ],
        },
        "shap": shap.get("topFeatures", []),
    }


def _slim_solution(solution: dict[str, Any] | None) -> dict[str, Any] | None:
    if not solution:
        return None
    keys = [
        "id",
        "retrofitCostCny",
        "annualRobustTotalInputKwh",
        "annualDiscomfortHours",
        "savingsVsBaselinePct",
        "discomfortChangeVsBaselineH",
        "sigmaTotalInputKwh",
    ]
    slim = {key: solution.get(key) for key in keys}
    slim["inputs"] = solution.get("inputs")
    return slim


def _llm_sections(payload: dict[str, Any]) -> tuple[str, dict[str, list[str]], str | None]:
    try:
        from openai import OpenAI
        from backend.agent.config import get_llm_config

        cfg = get_llm_config()
        if not cfg["api_key"]:
            raise RuntimeError("LLM API Key not configured")
        client = OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])
        prompt = """你是 EnergyPlus-Agent 的自动报告撰写器，面向业主、物业、社区更新和设计院人员。

你的技术身份与对话端一致：
- 这是物理增强生成(PAG)报告，数值来自 HC-DeepONet 5 成员深度集成代理模型、Pareto 优化工具和 SHAP 归因工具。
- 代理模型来自 EnergyPlus 仿真数据，输出年供暖/供冷输入电耗、等效不舒适小时，以及集成不确定性相关指标。
- 你负责解释、归纳、权衡和给建议；不要自己心算建筑热工，不要编造任何输入 JSON 里没有的精确数值。
- 若看到 sigma 或不确定性信息，只能以“代理不确定性提示/建议必要时用 EnergyPlus 复核”的方式表述，不要声称已完成高保真仿真。
- 工具结果含 model_context/building 信息时，必须说明结果基于哪个建筑/模型/天气文件；不要把默认南京模型的几何尺寸套用到用户专属模型之外。

报告写作要求：
- 用简体中文，专业但不要像论文摘要，要像智能体对话端那样解释“为什么这么判断、该怎么取舍、下一步怎么做”。
- 每个章节写成多条完整句子，句子要有分析含义，不要只复述表格。
- 必须覆盖：建筑信息、当前方案能耗曲线解读、Pareto 优化与典型方案权衡、SHAP 特征重要性、总结与建议。
- 对 Pareto 方案要明确低成本、均衡、最低能耗三类代表方案的适用场景，说明节能与舒适代价。
- 对建筑画像只做文字描述，不要求绘图。

输出严格 JSON，不要 Markdown，不要额外解释。字段固定如下，每个字段是 5-8 条中文短句数组：
{
  "building_overview": [],
  "energy_curve_analysis": [],
  "pareto_analysis": [],
  "feature_importance": [],
  "recommendations": []
}"""
        response = client.chat.completions.create(
            model=cfg["model"],
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            temperature=0.25,
            max_tokens=3200,
        )
        content = response.choices[0].message.content or ""
        parsed = _extract_json(content)
        if not parsed:
            raise RuntimeError("LLM response is not valid JSON")
        sections = {
            "building_overview": [_pdf_text(item) for item in _string_list(parsed.get("building_overview"))],
            "energy_curve_analysis": [_pdf_text(item) for item in _string_list(parsed.get("energy_curve_analysis"))],
            "pareto_analysis": [_pdf_text(item) for item in _string_list(parsed.get("pareto_analysis"))],
            "feature_importance": [_pdf_text(item) for item in _string_list(parsed.get("feature_importance"))],
            "recommendations": [_pdf_text(item) for item in _string_list(parsed.get("recommendations"))],
        }
        if not any(sections.values()):
            raise RuntimeError("LLM response has no report sections")
        return "llm", sections, None
    except Exception as exc:  # noqa: BLE001 - report generation must fall back.
        return "template_fallback", _template_sections(payload), str(exc)


def _extract_json(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _template_sections(payload: dict[str, Any]) -> dict[str, list[str]]:
    building = payload["building"]
    annual = payload["annual_prediction"]
    pareto = payload["pareto"]
    curve = payload.get("energy_curve") or {}
    seasons = curve.get("season_totals") or {}
    shap = payload["shap"]
    representatives = pareto.get("representative_solutions") or []
    low_cost = next((item for item in representatives if item.get("label") == "低成本方案"), representatives[0] if representatives else {})
    balanced = next((item for item in representatives if item.get("label") == "均衡方案"), representatives[0] if representatives else {})
    best_energy = next((item for item in representatives if item.get("label") == "最低能耗方案"), representatives[-1] if representatives else {})
    top_feature = shap[0] if shap else {}
    sections = {
        "building_overview": [
            f"本报告对象为 {building.get('city', 'Nanjing')} 的 {building.get('name', '当前建筑')}，类型为 {building.get('type', 'office')}，气象口径为 {building.get('weather', '南京典型气象年')}。",
            f"建筑平面为 {building.get('plan_shape')}，约 {building.get('floor_count', '-')} 层，总建筑面积约 {_num(building.get('floor_area_m2'), 0)} m2，高度约 {_num(building.get('height_m'), 1)} m。",
            f"围护结构基准参数包括窗墙比 {_pct((building.get('window_wall_ratio') or 0) * 100, 1)}、窗 U 值 {_num(building.get('window_u_w_m2k'), 2)} W/m2K、SHGC {_num(building.get('window_shgc'), 2)}。",
            f"外墙保温为 {_num((building.get('wall_insulation_m') or 0) * 100, 1)} cm，屋面保温为 {_num((building.get('roof_insulation_m') or 0) * 100, 1)} cm，渗透换气为 {_num(building.get('infiltration_ach'), 2)} ACH。",
            f"当前运行设定为供冷 {building.get('cooling_setpoint_c', '-')} degC、供暖 {building.get('heating_setpoint_c', '-')} degC，制冷 COP {building.get('cooling_cop', '-')}、制热 COP {building.get('heating_cop', '-')}。",
            f"该报告基于 {building.get('model_type', 'CHU-DeepONet')} 代理模型，集成成员数为 {building.get('ensemble_members', '-')}，训练样本数为 {building.get('sample_count', '默认模型样本库')}。",
        ],
        "energy_curve_analysis": [
            f"当前方案预测年冷热输入为 {_kwh(annual.get('total_input_kwh'))}，其中供暖 {_kwh(annual.get('heating_input_kwh'))}、供冷 {_kwh(annual.get('cooling_input_kwh'))}。",
            f"按建筑面积折算，冷热输入强度约 {_num(annual.get('total_input_intensity_kwh_m2'), 2)} kWh/m2，供冷占比约 {_pct(annual.get('cooling_share_pct'), 1)}，供暖占比约 {_pct(annual.get('heating_share_pct'), 1)}。",
            f"全年等效不舒适为 {_num(annual.get('discomfort_hours'), 2)} h，说明当前基准运行设定下舒适风险总体较低，但优化方案仍需要同时检查不舒适小时变化。",
            f"供暖峰值出现在 {curve.get('heating_peak_day', {}).get('label', '-')}，日供暖输入约 {_kwh(curve.get('heating_peak_day', {}).get('heating_input_kwh'))}；供冷峰值出现在 {curve.get('cooling_peak_day', {}).get('label', '-')}，日供冷输入约 {_kwh(curve.get('cooling_peak_day', {}).get('cooling_input_kwh'))}。",
            f"夏季供冷季冷热输入约 {_kwh((seasons.get('summer_cooling') or {}).get('total_input_kwh'))}，冬季供暖季冷热输入约 {_kwh((seasons.get('winter_heating') or {}).get('total_input_kwh'))}，曲线解读应优先关注夏季峰值和过渡季零散负荷。",
            "这些曲线来自代理模型批量预测，适合快速筛选方案；若后续选择明显超出训练范围的参数，应追加 EnergyPlus 高保真仿真复核。",
        ],
        "pareto_analysis": [
            f"Pareto 优化共评估 {pareto.get('candidate_count', '-')} 个候选方案，识别 {pareto.get('pareto_count', '-')} 个非支配方案，用于权衡成本、稳健能耗与舒适度。",
            f"低成本方案 #{low_cost.get('id', '-')} 的改造成本约 {_cny(low_cost.get('retrofitCostCny'))}，稳健冷热输入为 {_kwh(low_cost.get('annualRobustTotalInputKwh'))}，节能率约 {_pct(low_cost.get('savingsVsBaselinePct'), 2)}。",
            f"均衡方案 #{balanced.get('id', '-')} 的稳健冷热输入为 {_kwh(balanced.get('annualRobustTotalInputKwh'))}，节能率约 {_pct(balanced.get('savingsVsBaselinePct'), 2)}，成本约 {_cny(balanced.get('retrofitCostCny'))}。",
            f"最低能耗方案 #{best_energy.get('id', '-')} 的稳健冷热输入为 {_kwh(best_energy.get('annualRobustTotalInputKwh'))}，节能率约 {_pct(best_energy.get('savingsVsBaselinePct'), 2)}，但需要特别关注不舒适小时和投资强度。",
            "低成本方案适合快速改造或预算约束场景；均衡方案适合作为优先推荐；最低能耗方案更适合深度节能示范或长期收益导向项目。",
            "若某个方案的节能率很高但不舒适小时显著上升，应把它作为技术上限参考，而不是直接作为施工推荐。",
        ],
        "feature_importance": [
            f"SHAP 归因显示，最重要特征为 {top_feature.get('label') or top_feature.get('feature') or '-'}，相对重要性约 {_pct(top_feature.get('importancePct'), 2)}。",
            *[
                f"{item.get('label') or item.get('feature')} 的影响方向为：{item.get('direction', '未标注')}。"
                for item in shap[:4]
            ],
            "特征重要性用于解释模型在训练样本空间内的平均敏感性，不等同于单一工程方案的因果证明。",
            "当高重要性参数同时具备可施工性和可运维性时，应优先纳入改造清单；若参数难以改造，则用于制定监测和控制策略。",
        ],
        "recommendations": [
            "优先选择 Pareto 前沿中的均衡方案，不建议只按单一最低能耗目标决策。",
            "若预算有限，先优化窗 U 值、窗墙比、渗透换气与系统 COP 等高敏感参数。",
            "对门控不确定性偏高或超出训练范围的方案，应追加 EnergyPlus 高保真仿真复核。",
            "运维阶段可在舒适约束允许时微调供冷设定点，并持续监测峰值日能耗变化。",
            "落地时建议把方案分成低成本快改、围护结构改造、系统效率提升和运维策略优化四类，分别评估投资、施工扰动和舒适风险。",
            "最终决策应把能耗、舒适、预算、施工窗口和使用者接受度一起纳入，而不是只选择图上最低能耗点。",
        ],
    }
    return {key: [_pdf_text(item) for item in values] for key, values in sections.items()}


def _pdf_font_name() -> str:
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    for path in FONT_CANDIDATES:
        if path.exists():
            pdfmetrics.registerFont(TTFont("ReportCJK", str(path)))
            return "ReportCJK"
    return "Helvetica"


def _build_pdf(report_id: str, filename: str, building: dict[str, Any], prediction: dict[str, Any],
               pareto: dict[str, Any], shap: dict[str, Any], sections: dict[str, list[str]],
               charts: dict[str, Path], ai_mode: str, ai_error: str | None) -> Path:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / filename
    font_name = _pdf_font_name()
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("CJKTitle", parent=styles["Title"], fontName=font_name, fontSize=22, leading=30, alignment=TA_CENTER))
    styles.add(ParagraphStyle("CJKHeading", parent=styles["Heading2"], fontName=font_name, fontSize=14, leading=20, spaceBefore=8, spaceAfter=8))
    styles.add(ParagraphStyle("CJKBody", parent=styles["BodyText"], fontName=font_name, fontSize=9.8, leading=15, spaceAfter=5))
    styles.add(ParagraphStyle("CJKSmall", parent=styles["BodyText"], fontName=font_name, fontSize=8.4, leading=12, textColor=colors.HexColor("#64748b")))
    styles.add(ParagraphStyle("CJKCell", parent=styles["BodyText"], fontName=font_name, fontSize=8.8, leading=12))

    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        leftMargin=1.45 * cm,
        rightMargin=1.45 * cm,
        topMargin=1.35 * cm,
        bottomMargin=1.35 * cm,
        title="建筑能耗与舒适度优化报告",
    )
    story: list[Any] = []
    story.append(Paragraph("建筑能耗与舒适度优化报告", styles["CJKTitle"]))
    mode_label = "AI 分析" if ai_mode == "llm" else "模板兜底"
    story.append(Paragraph(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  分析模式：{mode_label}", styles["CJKSmall"]))
    story.append(Spacer(1, 8))

    annual = prediction.get("annual") or {}
    metrics = [
        ["指标", "数值", "说明"],
        ["年冷热总输入", _kwh(annual.get("totalInputKwh")), "供暖输入 + 供冷输入"],
        ["年供暖输入", _kwh(annual.get("heatingInputKwh")), "按制热 COP 折算"],
        ["年供冷输入", _kwh(annual.get("coolingInputKwh")), "按制冷 COP 折算"],
        ["年等效不舒适", f"{_num(annual.get('discomfortHours'), 2)} h", "|PMV| 超限等效小时"],
    ]
    story.append(_table(metrics, styles))

    story.extend(_section("一、建筑信息", sections.get("building_overview", []), styles))
    story.append(_table(_building_table(building), styles))
    story.append(PageBreak())

    story.extend(_section("二、能耗曲线分析", sections.get("energy_curve_analysis", []), styles))
    story.append(_image(charts["daily_curve"], 16.2 * cm, 7.0 * cm))
    story.append(_table(_season_table(prediction), styles))
    story.append(PageBreak())

    story.extend(_section("三、Pareto 优化结果", sections.get("pareto_analysis", []), styles))
    story.append(_table(_solution_table(pareto), styles))
    story.append(Spacer(1, 8))
    story.append(_image(charts["pareto"], 16.2 * cm, 7.1 * cm))
    story.append(PageBreak())

    story.append(Paragraph("四、典型方案量化对比", styles["CJKHeading"]))
    story.append(Paragraph("下图把基线、低成本、均衡和最低能耗三类方案放在同一坐标下，便于同时观察能耗收益和舒适代价。", styles["CJKBody"]))
    story.append(_image(charts["comparison"], 16.2 * cm, 7.0 * cm))
    story.append(PageBreak())

    story.extend(_section("五、特征重要性分析", sections.get("feature_importance", []), styles))
    story.append(_table(_shap_table(shap), styles))
    story.append(_image(charts["shap"], 16.2 * cm, 7.0 * cm))
    story.extend(_section("六、总结与建议", sections.get("recommendations", []), styles))
    if ai_mode == "template_fallback":
        note = "AI 文本生成失败，已使用规则模板生成。"
        if ai_error:
            note += f" 失败原因：{ai_error}"
        story.append(Paragraph(note, styles["CJKSmall"]))

    def footer(canvas, document):
        canvas.saveState()
        canvas.setFont(font_name, 8)
        canvas.setFillColor(colors.HexColor("#64748b"))
        canvas.drawString(1.45 * cm, 0.72 * cm, "EnergyPlus-Agent")
        canvas.drawRightString(A4[0] - 1.45 * cm, 0.72 * cm, f"第 {document.page} 页")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return path


def _section(title: str, items: list[str], styles) -> list[Any]:
    from reportlab.lib import colors
    from reportlab.platypus import ListFlowable, ListItem, Paragraph, Spacer

    flows: list[Any] = [Paragraph(title, styles["CJKHeading"])]
    if items:
        flows.append(ListFlowable(
            [ListItem(Paragraph(item, styles["CJKBody"]), bulletColor=colors.HexColor("#0f6b8f")) for item in items],
            bulletType="bullet",
            leftIndent=14,
        ))
    flows.append(Spacer(1, 6))
    return flows


def _table(rows: list[list[Any]], styles):
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.platypus import Paragraph, Table, TableStyle

    data = [[Paragraph(_pdf_text(cell), styles["CJKCell"]) for cell in row] for row in rows]
    column_count = max((len(row) for row in rows), default=1)
    if column_count == 3:
        col_widths = [2.8 * cm, 4.0 * cm, 9.2 * cm]
    elif column_count == 5:
        col_widths = [2.5 * cm, 3.0 * cm, 4.0 * cm, 3.0 * cm, 3.5 * cm]
    else:
        col_widths = None
    table = Table(data, repeatRows=1, hAlign="LEFT", colWidths=col_widths)
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), styles["CJKCell"].fontName),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2ff")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#312e81")),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cbd5e1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return table


def _image(path: Path, width: float, height: float):
    from reportlab.platypus import Image

    image = Image(str(path))
    image._restrictSize(width, height)  # noqa: SLF001 - ReportLab public examples use this helper.
    return image


def _building_table(building: dict[str, Any]) -> list[list[Any]]:
    return [
        ["类别", "参数", "取值"],
        ["基本信息", "建筑名称", building.get("name", "-")],
        ["基本信息", "城市/气象", f"{building.get('city', '-')} / {building.get('weather', '-')}"],
        ["几何", "平面/层数/面积", f"{building.get('plan_shape', '-')} / {building.get('floor_count', '-')} 层 / {_num(building.get('floor_area_m2'), 0)} m2"],
        ["几何", "长宽/高度", f"{_num(building.get('length_m'), 1)} m x {_num(building.get('width_m'), 1)} m / {_num(building.get('height_m'), 1)} m"],
        ["围护", "窗墙比 / 窗 U / SHGC", f"{_num(building.get('window_wall_ratio'), 2)} / {_num(building.get('window_u_w_m2k'), 2)} W/m2K / {_num(building.get('window_shgc'), 2)}"],
        ["围护", "外墙/屋面保温/渗透", f"{_num(building.get('wall_insulation_m'), 3)} m / {_num(building.get('roof_insulation_m'), 3)} m / {_num(building.get('infiltration_ach'), 2)} 1/h"],
        ["运行", "供冷/供暖设定", f"{_num(building.get('cooling_setpoint_c'), 1)} degC / {_num(building.get('heating_setpoint_c'), 1)} degC"],
        ["模型", "模型类型/样本数", f"{building.get('model_type', '-')} / {building.get('sample_count', '-')}"],
    ]


def _season_table(prediction: dict[str, Any]) -> list[list[Any]]:
    curve = _energy_curve_summary(prediction)
    seasons = curve.get("season_totals") or {}
    labels = {
        "winter_heating": "冬季供暖季",
        "transition": "过渡季",
        "summer_cooling": "夏季供冷季",
    }
    rows = [["时段", "供暖输入", "供冷输入", "冷热总输入", "不舒适小时"]]
    for key in ["winter_heating", "transition", "summer_cooling"]:
        item = seasons.get(key) or {}
        rows.append([
            labels[key],
            _kwh(item.get("heating_input_kwh")),
            _kwh(item.get("cooling_input_kwh")),
            _kwh(item.get("total_input_kwh")),
            f"{_num(item.get('discomfort_hours'), 2)} h",
        ])
    return rows


def _solution_table(pareto: dict[str, Any]) -> list[list[Any]]:
    baseline = pareto.get("baseline") or {}
    chosen = _representative_solutions(pareto)
    rows = [["方案", "成本", "稳健冷热输入", "节能率", "等效不舒适"]]
    rows.append(["基线", "-", _kwh(baseline.get("annual_hvac_input_kwh")), "0.00%", f"{_num(baseline.get('annual_discomfort_hours'), 2)} h"])
    for label, item in chosen:
        rows.append([
            f"{label} #{item.get('id')}",
            _cny(item.get("retrofitCostCny")),
            _kwh(item.get("annualRobustTotalInputKwh")),
            _pct(item.get("savingsVsBaselinePct"), 2),
            f"{_num(item.get('annualDiscomfortHours'), 2)} h",
        ])
    return rows


def _shap_table(shap: dict[str, Any]) -> list[list[Any]]:
    rows = [["特征", "重要性", "影响方向"]]
    for item in shap.get("topFeatures", []):
        rows.append([
            item.get("label") or item.get("feature"),
            _pct(item.get("importancePct"), 2),
            _pdf_text(item.get("direction", "-")),
        ])
    return rows


def generate_report(request: EnergyComfortReportRequest) -> dict[str, Any]:
    app = _app()
    if bool(request.building_id) != bool(request.model_id):
        raise HTTPException(status_code=422, detail="building_id 和 model_id 需要同时提供。")
    try:
        manifest = app._model_manifest_for_ref(request.building_id, request.model_id, strict=bool(request.building_id))
        model_context = app._active_model_context(manifest)
        spec = _load_spec(manifest)
        design_inputs = _design_inputs_from_spec(spec)
        prediction = app._predict_design(app.DesignInputs(**design_inputs), manifest=manifest)
        pareto = app._design_pareto_result(app.ParetoRequest(count=160, seed=2026, lock_setpoints=False), manifest)
        from backend.agent.tools import explain_features

        shap = explain_features(target="energy_total", top_k=8)
        building = _building_info(spec, manifest)
        report_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        charts = _plot_charts(report_id, building, prediction, pareto, shap)
        payload = _analysis_payload(building, prediction, pareto, shap)
        ai_mode, sections, ai_error = _llm_sections(payload)
        filename = f"{_safe_slug(building.get('name') or '建筑')}_energy_comfort_{report_id}.pdf"
        pdf_path = _build_pdf(report_id, filename, building, prediction, pareto, shap, sections, charts, ai_mode, ai_error)
    except HTTPException:
        raise
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=f"报告生成依赖缺失：{exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"报告生成失败：{type(exc).__name__}: {exc}") from exc
    return {
        "report_id": report_id,
        "filename": pdf_path.name,
        "download_url": f"/api/reports/{report_id}/download",
        "model_context": model_context,
        "ai_mode": ai_mode,
    }


@router.post("/api/reports/energy-comfort")
def create_energy_comfort_report(request: EnergyComfortReportRequest | None = None):
    return generate_report(request or EnergyComfortReportRequest())


@router.get("/api/reports/{report_id}/download")
def download_report(report_id: str):
    return _download_response(report_id)


@router.head("/api/reports/{report_id}/download")
def head_report(report_id: str):
    return _download_response(report_id)


def _download_response(report_id: str):
    if not re.fullmatch(r"\d{8}_\d{6}_[0-9a-f]{8}", report_id):
        raise HTTPException(status_code=404, detail="报告不存在")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    matches = sorted(REPORT_DIR.glob(f"*_{report_id}.pdf"))
    if not matches:
        raise HTTPException(status_code=404, detail="报告不存在")
    path = matches[-1].resolve()
    try:
        path.relative_to(REPORT_DIR.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="报告不存在") from exc
    return FileResponse(path, media_type="application/pdf", filename=path.name)
