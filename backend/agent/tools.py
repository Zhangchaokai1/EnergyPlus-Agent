"""Function Calling 工具层。

把已训练的 HC-DeepONet 代理模型与既有计算函数包装成大模型可调用的工具：
  - predict_energy   单个设计的年能耗 μ_E + σ_E + 门控判定
  - optimize_pareto  成本-能耗-舒适三目标 Pareto（场景 5.1 设计/改造期）
  - operation_savings 设定点与"升温一度"节能（场景 5.2 运维期）
  - estimate_extreme 异常年 OOD 能耗峰值（场景 5.3 极端气候）
  - explain_features SHAP 特征归因

工具在请求时惰性导入 backend.app（彼时已完整加载），复用其缓存的模型 (_load_bundle)
与既有函数 (_predict / design_pareto / operation_saving / extreme_estimate)，不重复造轮子。
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from . import gating
from .attribution_direction import direction_details
from .comparison_summary import build_comparison_summary
from backend.training_routes import list_models

from backend.paths import ROOT  # noqa: E402
SHAP_CSV = ROOT / "models" / "office_shap" / "shap_importance.csv"


def _app():
    """惰性导入主应用模块（避免与 routes 的循环依赖；请求时 app 已加载完毕）。"""
    from backend import app
    return app


def _model_manifest_from_kwargs(app, kwargs: dict) -> dict:
    building_id = kwargs.get("building_id")
    model_id = kwargs.get("model_id")
    if building_id or model_id:
        if not building_id or not model_id:
            raise ValueError("building_id 和 model_id 需要同时提供。")
        return app._model_manifest_for_ref(str(building_id), str(model_id), strict=True)
    return app._active_model_manifest()


def _daily_curve(daily_mean, weather, heating_cop, cooling_cop) -> list:
    """把逐日负荷均值(365,3)折算为逐日供暖/供冷输入电耗曲线（供前端绘图）。"""
    labels = weather["day_label"].tolist()
    months = weather["month"].tolist()
    hc = max(float(heating_cop), 1e-9)
    cc = max(float(cooling_cop), 1e-9)
    return [
        {"label": str(labels[i]), "month": int(months[i]),
         "heating": round(float(daily_mean[i, 0]) / hc, 1),
         "cooling": round(float(daily_mean[i, 1]) / cc, 1)}
        for i in range(len(labels))
    ]


# ---------------- 工具实现 ----------------

def predict_energy(climate_scenario: str = "base", **kwargs) -> dict:
    """给定（部分）建筑/系统参数，返回年能耗、等效不舒适与不确定性门控判定。

    未提供的参数回退到基准办公楼；允许超出训练范围（由 σ_E 门控体现可靠性）。
    climate_scenario 可选异常年情景（base/anom_warm/anom_cold/anom_balanced）：
    代理在对应年份的逐日气候特征上外推，σ_E 门控自然反映其可信度。
    """
    app = _app()
    manifest = _model_manifest_from_kwargs(app, kwargs)
    selected_model = app._active_model_context(manifest)
    design = dict(app.BASE_DESIGN)
    for key in app.STATIC_COLUMNS:
        val = kwargs.get(key)
        if val is not None:
            design[key] = float(val)

    out_of_range = {}
    for key in app.STATIC_COLUMNS:
        lo, hi = app.DESIGN_BOUNDS[key]
        if design[key] < lo or design[key] > hi:
            out_of_range[key] = {"value": design[key], "min": lo, "max": hi}

    scenario = str(climate_scenario or "base")
    try:
        weather = app._weather_features_for_scenario(scenario)
    except Exception:
        return {"error": f"未知气候情景：{scenario}", "validScenarios": app._list_weather_scenarios()}

    static = np.array([[design[k] for k in app.STATIC_COLUMNS]], dtype=float)
    dynamic = np.repeat(weather[app.DYNAMIC_COLUMNS].to_numpy(dtype=float)[None, :, :], 1, axis=0)
    raw = app._predict(static, dynamic, app._active_model_dir(manifest))

    annual = raw["annual_mean"][0]
    heating_load, cooling_load, discomfort = float(annual[0]), float(annual[1]), float(annual[2])
    heating_input = heating_load / max(design["heating_cop"], 1e-9)
    cooling_input = cooling_load / max(design["cooling_cop"], 1e-9)
    total_input = heating_input + cooling_input

    sigma_e = gating.energy_total_sigma(raw["member_annual"])
    verdict = gating.gate_verdict(sigma_e)

    base = app._baseline_summary_for_manifest(manifest)
    savings_pct = (base["annual_hvac_input_kwh"] - total_input) / max(base["annual_hvac_input_kwh"], 1e-9) * 100.0

    return {
        "design": {k: round(design[k], 4) for k in app.STATIC_COLUMNS},
        "modelContext": selected_model,
        "climateScenario": scenario,
        "outOfTrainingRange": out_of_range or None,
        "annual": {
            "heatingLoadKwh": round(heating_load, 1),
            "coolingLoadKwh": round(cooling_load, 1),
            "heatingInputKwh": round(heating_input, 1),
            "coolingInputKwh": round(cooling_input, 1),
            "totalInputKwh": round(total_input, 1),
            "discomfortHours": round(discomfort, 2),
        },
        "vsBaseline": {
            "baselineHvacInputKwh": round(base["annual_hvac_input_kwh"], 1),
            "savingsPct": round(savings_pct, 2),
        },
        "gating": verdict,
        "chart": {"kind": "dailyCurve",
                  "daily": _daily_curve(raw["daily_mean"][0], weather, design["heating_cop"], design["cooling_cop"])},
    }


def optimize_pareto(count: int = 100, lock_setpoints: bool = False, top_k: int = 8, **kwargs) -> dict:
    """成本-能耗-舒适三目标 Pareto 优化，返回若干非支配高性价比方案。"""
    app = _app()
    manifest = _model_manifest_from_kwargs(app, kwargs)
    selected_model = app._active_model_context(manifest)
    count = int(max(20, min(int(count or 100), 400)))
    req = app.ParetoRequest(count=count, seed=2026, lock_setpoints=bool(lock_setpoints))
    result = app._design_pareto_result(req, manifest)

    def slim(p: dict) -> dict:
        return {
            "id": p["id"],
            "retrofitCostCny": round(p["retrofitCostCny"]),
            "annualTotalInputKwh": round(p["annualTotalInputKwh"]),
            "annualRobustTotalInputKwh": round(p["annualRobustTotalInputKwh"]),
            "savingsVsBaselinePct": round(p["savingsVsBaselinePct"], 2),
            "annualDiscomfortHours": round(p["annualDiscomfortHours"], 2),
            "design": {k: round(v, 4) for k, v in p["inputs"].items()},
        }

    front = [slim(p) for p in result["front"][: max(1, int(top_k or 8))]]
    return {
        "candidateCount": result["count"],
        "modelContext": selected_model,
        "paretoCount": result["paretoCount"],
        "objectives": result["objectives"],
        "baselineHvacInputKwh": round(result["baseline"]["annual_hvac_input_kwh"]),
        "topSolutions": front,
        "comparisonSummary": build_comparison_summary(front, language="zh"),
        "chart": {
            "kind": "pareto",
            "points": [
                {
                    "x": round(p["retrofitCostCny"] / 10000.0, 2),
                    "y": round(p["annualRobustTotalInputKwh"] / 10000.0, 2),
                    "z": round(p["annualDiscomfortHours"], 1),
                    "pareto": bool(p["pareto"]),
                }
                for p in result["points"]
            ],
        },
    }


def operation_savings(cooling_setpoint_c: float = 26.0, heating_setpoint_c: float = 20.0, **kwargs) -> dict:
    """固定基准建筑，给定空调设定点，返回年冷热输入与"夏季升温 1°C"的节能量。"""
    app = _app()
    manifest = _model_manifest_from_kwargs(app, kwargs)
    base_inputs = app._fixed_building_inputs(float(cooling_setpoint_c), float(heating_setpoint_c))
    raised_inputs = app._fixed_building_inputs(
        min(float(cooling_setpoint_c) + 1.0, app.COOLING_SETPOINT_MAX),
        float(heating_setpoint_c),
    )
    base = app._predict_design(base_inputs, manifest=manifest)
    raised = app._predict_design(raised_inputs, manifest=manifest)
    base_cooling = base["annual"]["coolingInputKwh"]
    raised_cooling = raised["annual"]["coolingInputKwh"]
    saved = base_cooling - raised_cooling
    summer_saved = 0.0
    for b, r in zip(base["daily"], raised["daily"]):
        if 6 <= b["month"] <= 9:
            summer_saved += b["coolingInputKwh"] - r["coolingInputKwh"]
    ann = base["annual"]
    return {
        "modelContext": app._active_model_context(manifest),
        "setpoints": {
            "coolingSetpointC": float(cooling_setpoint_c),
            "heatingSetpointC": float(heating_setpoint_c),
        },
        "annual": {
            "coolingInputKwh": round(ann["coolingInputKwh"]),
            "heatingInputKwh": round(ann["heatingInputKwh"]),
            "totalInputKwh": round(ann["totalInputKwh"]),
            "discomfortHours": round(ann["discomfortHours"], 2),
        },
        "raiseCoolingBy1C": {
            "summerCoolingSavedKwh": round(summer_saved),
            "annualCoolingSavedKwh": round(saved),
            "summerCoolingSavingPct": round(summer_saved / max(
                sum(d["coolingInputKwh"] for d in base["daily"] if 6 <= d["month"] <= 9), 1e-9
            ) * 100.0, 2),
        },
        "chart": {
            "kind": "dailyCurve",
            "daily": [
                {"label": d["label"], "month": d["month"],
                 "heating": round(d["heatingInputKwh"], 1), "cooling": round(d["coolingInputKwh"], 1)}
                for d in base["daily"]
            ],
        },
    }


def estimate_extreme(scenario: str = "anom_warm", cooling_setpoint_c: float = 26.0,
                     heating_setpoint_c: float = 20.0, **kwargs) -> dict:
    """异常年 (OOD) 天气下的能耗峰值与增幅估计。scenario: anom_warm/anom_cold/anom_balanced。"""
    app = _app()
    from fastapi import HTTPException
    manifest = _model_manifest_from_kwargs(app, kwargs)
    normal_inputs = app._fixed_building_inputs(float(cooling_setpoint_c), float(heating_setpoint_c))
    try:
        scenario_weather = app._weather_features_for_scenario(str(scenario))
    except HTTPException as exc:
        return {"error": str(exc.detail), "validScenarios": app._list_weather_scenarios()}

    normal = app._predict_design(normal_inputs, app._base_weather_features(), manifest=manifest)
    extreme = app._predict_design(normal_inputs, scenario_weather, manifest=manifest)
    normal_avg = normal["annual"]["totalInputKwh"] / max(len(normal["daily"]), 1)
    extreme_peak = max(extreme["daily"], key=lambda d: d["totalInputKwh"])
    extreme_avg = extreme["annual"]["totalInputKwh"] / max(len(extreme["daily"]), 1)
    increase = (extreme_peak["totalInputKwh"] - normal_avg) / max(normal_avg, 1e-9) * 100.0
    avg_increase = (extreme_avg - normal_avg) / max(normal_avg, 1e-9) * 100.0
    return {
        "scenario": str(scenario),
        "modelContext": app._active_model_context(manifest),
        "normalAvgDailyTotalInputKwh": round(normal_avg),
        "extremeAvgDailyTotalInputKwh": round(extreme_avg),
        "extremePeakDayKwh": round(extreme_peak["totalInputKwh"]),
        "peakDayLabel": extreme_peak["label"],
        "peakIncreasePctVsNormalAvg": round(increase, 1),
        "avgIncreasePct": round(avg_increase, 1),
        "note": "异常年属于训练分布边缘(OOD)，σ_E 通常升高；可调用 run_energyplus_simulation 仿真复核。",
        "chart": {
            "kind": "bars",
            "unit": "kWh",
            "bars": [
                {"name": "正常日均", "value": round(normal_avg)},
                {"name": "极端峰值日", "value": round(extreme_peak["totalInputKwh"])},
            ],
        },
    }


@lru_cache(maxsize=1)
def _read_shap_frame(path: str, mtime_ns: int, size: int) -> pd.DataFrame:
    return pd.read_csv(path)


def _shap_frame() -> pd.DataFrame:
    # Data refreshes must invalidate long-lived server caches.
    stat = SHAP_CSV.stat()
    return _read_shap_frame(str(SHAP_CSV), stat.st_mtime_ns, stat.st_size)


def explain_features(target: str = "energy_total", top_k: int = 6, **kwargs) -> dict:
    """用 SHAP 解释某指标的主要驱动因素。target: energy_total/heating/cooling/discomfort。"""
    alias = {
        "energy": "energy_total", "total": "energy_total", "能耗": "energy_total",
        "heating": "heating", "供暖": "heating",
        "cooling": "cooling", "供冷": "cooling",
        "comfort": "discomfort", "discomfort": "discomfort", "舒适": "discomfort",
    }
    key = alias.get(str(target).lower(), str(target).lower())
    df = _shap_frame()
    sub = df[df["target"] == key]
    if sub.empty:
        return {"error": f"未知 target：{target}", "validTargets": sorted(df["target"].unique().tolist())}
    sub = sub.sort_values("relative_importance_pct", ascending=False).head(max(1, min(10, int(top_k or 6))))
    feats = [
        {
            "feature": r["feature"],
            "label": str(r["feature_label"]).strip("$"),
            "importancePct": round(float(r["relative_importance_pct"]), 2),
            **direction_details(r, language="zh"),
        }
        for _, r in sub.iterrows()
    ]
    return {
        "target": key,
        "attributionScope": "300 个验证设计的全局归因，仅适用于默认南京办公楼及固定气象",
        "method": "固定参考点的精确 Shapley 值；参考点为 100 个固定训练设计的均值",
        "signConvention": "正 SHAP 表示相对参考点增加指标，负 SHAP 表示降低指标；重要度为平均绝对 SHAP，不含方向。",
        "directionRule": "描述性显示规则：特征与 SHAP 的 Pearson |r| >= 0.5，且低/高四分位组平均 SHAP 符号相反、高取值组至少 75% 同号；否则显示混合。不是因果或全局单调结论。",
        "targetLabel": str(sub.iloc[0]["target_label"]),
        "unit": str(sub.iloc[0]["target_unit"]),
        "topFeatures": feats,
        "chart": {
            "kind": "featureBars",
            "features": [{"label": f["label"], "importancePct": f["importancePct"]} for f in feats],
        },
    }


# ---------------- 工具调度 + OpenAI 函数 schema ----------------

def run_energyplus_simulation(**kwargs) -> dict:
    """真实 EnergyPlus 高保真仿真（约 1 分钟）。门控判红或用户要求权威核验时使用。"""
    from . import eplus_fallback
    return eplus_fallback.run_fallback(kwargs)


def list_available_models(**kwargs) -> dict:
    """List registered surrogate models and the current model selection."""
    return list_models()


TOOL_IMPLS = {
    "predict_energy": predict_energy,
    "optimize_pareto": optimize_pareto,
    "operation_savings": operation_savings,
    "estimate_extreme": estimate_extreme,
    "explain_features": explain_features,
    "run_energyplus_simulation": run_energyplus_simulation,
    "list_available_models": list_available_models,
}


def dispatch_tool(name: str, arguments: dict | None) -> dict:
    fn = TOOL_IMPLS.get(name)
    if fn is None:
        return {"error": f"未知工具：{name}"}
    try:
        return fn(**(arguments or {}))
    except Exception as exc:  # noqa: BLE001 - 工具错误回传给模型而非中断
        return {"error": f"{type(exc).__name__}: {exc}"}


_DESIGN_PROPS = {
    "wall_insulation_m": {"type": "number", "description": "外墙保温厚度 m，训练范围 0.02–0.12"},
    "roof_insulation_m": {"type": "number", "description": "屋面保温厚度 m，训练范围 0.02–0.15"},
    "window_u_w_m2k": {"type": "number", "description": "窗传热系数 U (W/m²K)，训练范围 1.4–5.8，越低越保温"},
    "window_shgc": {"type": "number", "description": "太阳得热系数 SHGC，训练范围 0.30–0.55"},
    "infiltration_ach": {"type": "number", "description": "渗透换气次数 1/h，训练范围 0.30–0.80"},
    "window_wall_ratio": {"type": "number", "description": "窗墙比，训练范围 0.30–0.60"},
    "cooling_setpoint_c": {"type": "number", "description": "供冷设定温度 °C，训练范围 24–28"},
    "heating_setpoint_c": {"type": "number", "description": "供暖设定温度 °C，训练范围 16–20"},
    "cooling_cop": {"type": "number", "description": "制冷 COP，训练范围 3.0–6.0"},
    "heating_cop": {"type": "number", "description": "制热 COP，训练范围 1.8–4.0"},
}

_CLIMATE_PROP = {
    "climate_scenario": {
        "type": "string",
        "description": (
            "气候情景，默认 base=南京典型气象年(TMY)。情景库另含三个异常年："
            "anom_warm（偏暖年：年均约 +3.3°C，夏季峰值热浪约 +11°C，全年约 66 天日均温超出典型年最高）；"
            "anom_balanced（均衡异常年：年均基本持平约 -0.1°C，但夏季约 +9.4°C 强热浪、冬季约 -10.7°C 强冷潮）；"
            "anom_cold（偏冷年：年均约 -3.4°C，冬季峰值冷潮约 -11°C）。"
            "用户描述异常/极端气候年份时，选统计特征最接近的情景。"
        ),
    },
}

_MODEL_PROPS = {
    "building_id": {"type": "string", "description": "可选：模型库中的建筑 ID；未提供时使用当前选中模型"},
    "model_id": {"type": "string", "description": "可选：模型库中的代理模型 ID；未提供时使用当前选中模型"},
}

TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "predict_energy",
            "description": (
                "用 HC-DeepONet 代理模型预测某建筑方案在南京气候下的年供暖/供冷输入电耗、"
                "年等效不舒适小时、相对基准办公楼的节能率，并给出不确定性门控判定"
                "（σ_E 与验证集 p95 阈值比较：低=信任代理，高=建议回退 EnergyPlus 仿真）。"
                "只需传用户改动的参数，其余按基准办公楼取值。可选 climate_scenario 切换异常年气候。"
            ),
            "parameters": {"type": "object", "properties": {**_DESIGN_PROPS, **_CLIMATE_PROP, **_MODEL_PROPS}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "optimize_pareto",
            "description": (
                "成本-能耗-舒适三目标 Pareto 优化（设计/改造期）。在大量候选围护/系统方案上跑代理"
                "集成，返回非支配的高性价比方案（成本、稳健冷热输入、节能率、等效不舒适、对应设计参数）。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "count": {"type": "integer", "description": "候选方案数量，20–400，默认 100"},
                    "lock_setpoints": {"type": "boolean", "description": "是否锁定空调设定点为基准值（只优化围护与系统），默认 false"},
                    "top_k": {"type": "integer", "description": "返回前 K 个 Pareto 方案，默认 8"},
                    **_MODEL_PROPS,
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "operation_savings",
            "description": (
                "运维期：固定基准建筑，给定空调供冷/供暖设定温度，返回年供冷/供暖输入电耗、"
                "等效不舒适，以及夏季供冷设定值提高 1°C 可节省的电量与百分比。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "cooling_setpoint_c": {"type": "number", "description": "供冷设定温度 °C，24–28，默认 26"},
                    "heating_setpoint_c": {"type": "number", "description": "供暖设定温度 °C，16–20，默认 20"},
                    **_MODEL_PROPS,
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "estimate_extreme",
            "description": (
                "极端气候情景：固定建筑与设定点，估计异常年 (OOD) 天气下的日均能耗、峰值日能耗与增幅。"
                "scenario 取 anom_warm(异常暖年)/anom_cold(异常冷年)/anom_balanced(均衡异常年)。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "scenario": {"type": "string", "description": "异常年情景：anom_warm / anom_cold / anom_balanced"},
                    "cooling_setpoint_c": {"type": "number", "description": "供冷设定温度 °C，默认 26"},
                    "heating_setpoint_c": {"type": "number", "description": "供暖设定温度 °C，默认 20"},
                    **_MODEL_PROPS,
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "explain_features",
            "description": "用 SHAP 解释某能耗/舒适指标的主要驱动因素（特征重要性与影响方向）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {"type": "string", "description": "指标：energy_total / heating / cooling / discomfort"},
                    "top_k": {"type": "integer", "description": "返回前 K 个特征，默认 6"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_energyplus_simulation",
            "description": (
                "用高保真 EnergyPlus 对某建筑方案做真实参数化仿真（南京气候，75 区全年模拟，约 1 分钟）。"
                "当 predict_energy 的门控判定为 fallback（σ_E 高 / 疑似 OOD），或用户要求权威核验时调用；"
                "返回与代理同口径的年供暖/供冷输入电耗与等效不舒适，作为可信地面真值。只需传改动的参数。"
                "可选 climate_scenario 在异常年 EPW 气象文件下仿真。"
            ),
            "parameters": {"type": "object", "properties": {**_DESIGN_PROPS, **_CLIMATE_PROP, **_MODEL_PROPS}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_available_models",
            "description": "列出 EnergyPlus-Agent 模型库中的默认模型和用户训练出的建筑专属代理模型，以及当前选中的模型。",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]
