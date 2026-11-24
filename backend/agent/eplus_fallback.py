"""真实 EnergyPlus 回退仿真（门控判红时的高保真复核）。

把一个设计字典组装成 simulate_office_sample 的 payload，跑一次参数化 EnergyPlus
（南京 EPW，75 区全年模拟，约 70–80 秒），解析出年供暖/供冷负荷、HVAC 输入电耗与
等效不舒适小时（与代理模型同口径），作为可信的"地面真值"返回。

run_with_progress() 在后台线程跑仿真、每隔几秒产出一次进度事件，供编排器边等边汇报，
避免长达一分多钟的 SSE 静默。
"""
from __future__ import annotations

import itertools
import sys
import threading
from functools import lru_cache
from pathlib import Path

from backend.paths import ROOT  # noqa: E402
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_id_counter = itertools.count(900001)
_id_lock = threading.Lock()


def _next_sample_id() -> int:
    with _id_lock:
        return next(_id_counter)


@lru_cache(maxsize=1)
def _weather():
    """选定天气文件并解析表头（缓存；与数据集构建一致：南京 TMYx）。"""
    from office_geometry import choose_weather_file, parse_epw_header
    wf, mode = choose_weather_file()
    return str(wf), mode, parse_epw_header(wf)


@lru_cache(maxsize=8)
def _weather_for_scenario(scenario: str):
    """按气候情景选 EPW：base=数据集同款 TMYx；异常年取 datasets/weather_ood 对应文件。"""
    if scenario in ("", "base"):
        return _weather()
    from office_geometry import parse_epw_header
    matches = sorted((ROOT / "datasets" / "weather_ood").glob(f"*__{scenario}.epw"))
    if not matches:
        raise ValueError(f"未知气候情景：{scenario}（datasets/weather_ood 下无对应 EPW）")
    wf = matches[0]
    return str(wf), f"anom:{scenario}", parse_epw_header(wf)


def _app():
    from backend import app
    return app


def _eplus_daily_curve(app, rows) -> list:
    """从 E+ 逐日输出构造逐日供暖/供冷输入电耗曲线（day_index 映射回基准天气的标签）。"""
    wf = app._base_weather_features()
    idx2label = dict(zip(wf["day_index"].tolist(), wf["day_label"].tolist()))
    idx2month = dict(zip(wf["day_index"].tolist(), wf["month"].tolist()))
    out = []
    for r in sorted(rows, key=lambda x: int(x.get("day_index", 0))):
        di = int(r.get("day_index", 0))
        out.append({
            "label": str(idx2label.get(di, di)),
            "month": int(idx2month.get(di, 0)),
            "heating": round(float(r.get("daily_heating_input_kwh", 0) or 0), 1),
            "cooling": round(float(r.get("daily_cooling_input_kwh", 0) or 0), 1),
        })
    return out


def run_fallback(design: dict | None) -> dict:
    """跑一次真实 EnergyPlus，返回与代理同口径的年度结果（HVAC 输入电耗，不含基础电耗）。

    design 中可带 climate_scenario（base/anom_warm/anom_cold/anom_balanced）切换气象年。
    """
    import generate_office_dataset as gen

    app = _app()
    design = dict(design or {})
    scenario = str(design.pop("climate_scenario", None) or "base")
    building_id = design.pop("building_id", None)
    model_id = design.pop("model_id", None)
    if building_id or model_id:
        if not building_id or not model_id:
            return {"error": "building_id 和 model_id 需要同时提供。"}
        try:
            manifest = app._model_manifest_for_ref(str(building_id), str(model_id), strict=True)
        except Exception as exc:
            return {"error": str(exc)}
    else:
        manifest = app._active_model_manifest()
    full = dict(app.BASE_DESIGN)
    for key in app.STATIC_COLUMNS:
        val = design.get(key)
        if val is not None:
            full[key] = float(val)

    try:
        weather_path, _mode, weather_meta = _weather_for_scenario(scenario)
    except ValueError as exc:
        return {"error": str(exc)}
    payload = {
        "sample_id": _next_sample_id(),
        "sample_type": "agent_fallback" if scenario == "base" else f"agent_fallback_{scenario}",
        **{k: full[k] for k in app.STATIC_COLUMNS},
        "keep_artifacts": False,
        "weather_path": weather_path,
        "weather_meta": weather_meta,
        "work_dir": str(ROOT / "_tmp_agent_fallback_runs"),
        "building_id": manifest.get("building_id") if not manifest.get("is_default") else None,
        "building_spec_path": "",
    }
    if not manifest.get("is_default"):
        spec_path = ROOT / "building_projects" / str(manifest.get("building_id")) / "building_spec.json"
        if spec_path.exists():
            payload["building_spec_path"] = str(spec_path)
    res = gen.simulate_office_sample(payload)
    s = res["summary"]
    if s.get("simulation_status") != "ok":
        return {"error": f"EnergyPlus 仿真失败：{s.get('error_message') or '未知错误'}"}

    heating_input = float(s["annual_heating_input_kwh"])
    cooling_input = float(s["annual_cooling_input_kwh"])
    total_input = heating_input + cooling_input
    base = app._baseline_summary_for_manifest(manifest)
    savings = (base["annual_hvac_input_kwh"] - total_input) / max(base["annual_hvac_input_kwh"], 1e-9) * 100.0

    return {
        "source": "EnergyPlus 高保真仿真",
        "runtimeSeconds": s.get("runtime_s"),
        "modelContext": app._active_model_context(manifest),
        "climateScenario": scenario,
        "design": {k: round(full[k], 4) for k in app.STATIC_COLUMNS},
        "annual": {
            "heatingLoadKwh": round(float(s["annual_heating_load_kwh"]), 1),
            "coolingLoadKwh": round(float(s["annual_cooling_load_kwh"]), 1),
            "heatingInputKwh": round(heating_input, 1),
            "coolingInputKwh": round(cooling_input, 1),
            "totalInputKwh": round(total_input, 1),
            "discomfortHours": round(float(s["annual_discomfort_hours"]), 2),
        },
        "vsBaseline": {
            "baselineHvacInputKwh": round(base["annual_hvac_input_kwh"], 1),
            "savingsPct": round(savings, 2),
        },
        "chart": {"kind": "dailyCurve", "daily": _eplus_daily_curve(app, res.get("daily_rows") or [])},
    }


def run_with_progress(design: dict | None, heartbeat: float = 3.0):
    """生成器：后台线程跑仿真，每 heartbeat 秒产出 {'elapsedS': n}；最后产出 {'result': ...}。"""
    holder: dict = {}

    def worker():
        try:
            holder["result"] = run_fallback(design)
        except Exception as exc:  # noqa: BLE001
            holder["result"] = {"error": f"{type(exc).__name__}: {exc}"}

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    elapsed = 0.0
    while True:
        th.join(timeout=heartbeat)
        if not th.is_alive():
            break
        elapsed += heartbeat
        yield {"elapsedS": round(elapsed)}
    yield {"result": holder.get("result", {"error": "仿真线程未返回结果"})}
