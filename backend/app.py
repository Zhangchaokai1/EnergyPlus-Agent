from __future__ import annotations

import math
import os
import sys
import json
from functools import lru_cache
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from scipy.stats import qmc

from backend.paths import ROOT  # noqa: E402
MODEL_DIR = ROOT / "models" / "office_hc_deeponet_ensemble"
DATASET_DIR = ROOT / "datasets"
WEATHER_OOD_DIR = DATASET_DIR / "weather_ood"
PARETO_DIR = ROOT / "models" / "office_pareto_optimization"
PARETO_RUNTIME_SUMMARY = PARETO_DIR / "runtime_summary.json"
PAPER_DIR = ROOT / "paper"
SELECTED_MODEL_PATH = ROOT / "models" / "building_surrogates" / "selected_model.json"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import office_sampling as sampling  # noqa: E402
import train_office_surrogate as train  # noqa: E402
from building_project import DEFAULT_MODEL_ID, default_model_manifest, project_dir  # noqa: E402
from office_dataset_utils import build_daily_weather_features  # noqa: E402

app = FastAPI(
    title="Nanjing Office Surrogate Showcase API",
    description="Local API for the trained HC-DeepONet office energy surrogate.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5183",
        "http://127.0.0.1:5183",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],
    # 允许任意本地端口（Vite 端口被占用时会自动切到 5174/5175…）
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_COLUMNS = train.STATIC_COLUMNS
DYNAMIC_COLUMNS = train.DYNAMIC_COLUMNS
TARGET_COLUMNS = train.TARGET_COLUMNS
ENERGY_IDX = train.ENERGY_IDX
COOLING_SETPOINT_MIN = 24.0
COOLING_SETPOINT_MAX = 28.0
HEATING_SETPOINT_MIN = 16.0
HEATING_SETPOINT_MAX = 20.0

BASE_DESIGN = {
    "wall_insulation_m": 0.020,
    "roof_insulation_m": 0.020,
    "window_u_w_m2k": 3.50,
    "window_shgc": 0.50,
    "infiltration_ach": 0.80,
    "window_wall_ratio": 0.40,
    "cooling_setpoint_c": 26.0,
    "heating_setpoint_c": 20.0,
    "cooling_cop": 3.00,
    "heating_cop": 1.80,
}

DESIGN_BOUNDS = {
    "wall_insulation_m": (0.020, 0.120),
    "roof_insulation_m": (0.020, 0.150),
    "window_u_w_m2k": (1.40, 5.80),
    "window_shgc": (0.30, 0.55),
    "infiltration_ach": (0.30, 0.80),
    "window_wall_ratio": (0.30, 0.60),
    "cooling_setpoint_c": (COOLING_SETPOINT_MIN, COOLING_SETPOINT_MAX),
    "heating_setpoint_c": (HEATING_SETPOINT_MIN, HEATING_SETPOINT_MAX),
    "cooling_cop": (3.00, 6.00),
    "heating_cop": (1.80, 4.00),
}

DISPLAY_LABELS = {
    "wall_insulation_m": "外墙保温厚度",
    "roof_insulation_m": "屋面保温厚度",
    "window_u_w_m2k": "窗传热系数 U",
    "window_shgc": "太阳得热系数 SHGC",
    "infiltration_ach": "渗透换气次数",
    "window_wall_ratio": "窗墙比",
    "cooling_setpoint_c": "供冷设定温度",
    "heating_setpoint_c": "供暖设定温度",
    "cooling_cop": "制冷 COP",
    "heating_cop": "制热 COP",
}

BUILDING_FEATURE_DESCRIPTIONS = {
    "wall_insulation_m": {
        "unit": "m",
        "description": "外墙保温层厚度，影响冬季传热损失和夏季围护结构得热。",
    },
    "roof_insulation_m": {
        "unit": "m",
        "description": "屋面保温层厚度，主要影响顶层热损失和太阳辐射引起的屋面传热。",
    },
    "window_u_w_m2k": {
        "unit": "W/m2K",
        "description": "窗户传热系数，数值越低表示窗体隔热性能越好。",
    },
    "window_shgc": {
        "unit": "-",
        "description": "太阳得热系数，描述透过窗户进入室内的太阳辐射比例。",
    },
    "infiltration_ach": {
        "unit": "1/h",
        "description": "渗透换气次数，表示室外空气通过缝隙进入建筑的强度。",
    },
    "window_wall_ratio": {
        "unit": "-",
        "description": "窗墙比，表示外立面窗面积占围护立面面积的比例。",
    },
    "cooling_setpoint_c": {
        "unit": "degC",
        "description": "供冷设定温度，影响夏季空调供冷需求和舒适水平。",
    },
    "heating_setpoint_c": {
        "unit": "degC",
        "description": "供暖设定温度，影响冬季供暖需求和舒适水平。",
    },
    "cooling_cop": {
        "unit": "-",
        "description": "制冷系统性能系数，用于把供冷负荷换算为输入电耗。",
    },
    "heating_cop": {
        "unit": "-",
        "description": "制热系统性能系数，用于把供暖负荷换算为输入电耗。",
    },
}

CLIMATE_FEATURE_DESCRIPTIONS = [
    {
        "name": "干球温度",
        "fields": ["drybulbMeanC", "drybulbMinC", "drybulbMaxC"],
        "unit": "degC",
        "description": "刻画南京全年冷热变化，是供暖与供冷负荷的核心驱动因素。",
    },
    {
        "name": "相对湿度",
        "fields": ["rhMeanPct"],
        "unit": "%",
        "description": "反映空气湿度条件，会影响潜热负荷和人体热舒适。",
    },
    {
        "name": "风速",
        "fields": ["windMeanMS"],
        "unit": "m/s",
        "description": "影响围护结构换热和渗透换气带来的负荷变化。",
    },
    {
        "name": "太阳辐射",
        "fields": ["ghiKwhM2"],
        "unit": "kWh/m2",
        "description": "全球水平辐射表示太阳能输入强度，是窗户得热和供冷负荷的重要来源。",
    },
    {
        "name": "冷热度时",
        "fields": ["hdh16CH", "cdh26CH"],
        "unit": "degC·h",
        "description": "供暖度时 HDH16 与供冷度时 CDH26 概括一天内偏冷或偏热的累计强度。",
    },
]

PRINCIPLE_DOCS = [
    {
        "id": "heat-balance",
        "source": "EnergyPlus的底层原理：热平衡法.md",
        "summary": "解释 EnergyPlus 如何通过热平衡法计算室内温度、表面温度和冷热负荷。",
    },
    {
        "id": "pmv",
        "source": "Fanger PMV模型.md",
        "summary": "说明 Fanger PMV 模型如何把热环境转换为人体热舒适评价。",
    },
    {
        "id": "cop",
        "source": "COP.md",
        "summary": "说明 COP 如何把冷热负荷换算为设备输入电耗。",
    },
]


class DesignInputs(BaseModel):
    wall_insulation_m: float = Field(BASE_DESIGN["wall_insulation_m"], ge=0.02, le=0.12)
    roof_insulation_m: float = Field(BASE_DESIGN["roof_insulation_m"], ge=0.02, le=0.15)
    window_u_w_m2k: float = Field(BASE_DESIGN["window_u_w_m2k"], ge=1.4, le=5.8)
    window_shgc: float = Field(BASE_DESIGN["window_shgc"], ge=0.3, le=0.55)
    infiltration_ach: float = Field(BASE_DESIGN["infiltration_ach"], ge=0.3, le=0.8)
    window_wall_ratio: float = Field(BASE_DESIGN["window_wall_ratio"], ge=0.3, le=0.6)
    cooling_setpoint_c: float = Field(BASE_DESIGN["cooling_setpoint_c"], ge=COOLING_SETPOINT_MIN, le=COOLING_SETPOINT_MAX)
    heating_setpoint_c: float = Field(BASE_DESIGN["heating_setpoint_c"], ge=HEATING_SETPOINT_MIN, le=HEATING_SETPOINT_MAX)
    cooling_cop: float = Field(BASE_DESIGN["cooling_cop"], ge=3.0, le=6.0)
    heating_cop: float = Field(BASE_DESIGN["heating_cop"], ge=1.8, le=4.0)

    def as_vector(self) -> np.ndarray:
        data = self.model_dump()
        return np.asarray([[float(data[c]) for c in STATIC_COLUMNS]], dtype=float)

    def as_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in self.model_dump().items()}


class ScenarioRequest(BaseModel):
    inputs: DesignInputs = Field(default_factory=DesignInputs)


class ParetoRequest(BaseModel):
    count: int = Field(120, ge=20, le=800)
    seed: int = Field(2026, ge=1, le=999999)
    lock_setpoints: bool = False


class OperationInputs(BaseModel):
    cooling_setpoint_c: float = Field(26.0, ge=COOLING_SETPOINT_MIN, le=COOLING_SETPOINT_MAX)
    heating_setpoint_c: float = Field(20.0, ge=HEATING_SETPOINT_MIN, le=HEATING_SETPOINT_MAX)


class OperationRequest(BaseModel):
    inputs: OperationInputs = Field(default_factory=OperationInputs)


class ExtremeRequest(BaseModel):
    scenario: str = Field("anom_warm")
    cooling_setpoint_c: float = Field(26.0, ge=COOLING_SETPOINT_MIN, le=COOLING_SETPOINT_MAX)
    heating_setpoint_c: float = Field(20.0, ge=HEATING_SETPOINT_MIN, le=HEATING_SETPOINT_MAX)


def _select_device() -> torch.device:
    prefer_cpu = os.environ.get("SURROGATE_SHOWCASE_CPU", "").lower() in {"1", "true", "yes"}
    if prefer_cpu:
        return torch.device("cpu")
    return train.select_device()


def _default_model_ref() -> dict[str, str]:
    return {"building_id": "default_office", "model_id": DEFAULT_MODEL_ID}


def _load_selected_model_ref() -> dict[str, str]:
    if not SELECTED_MODEL_PATH.exists():
        return _default_model_ref()
    try:
        selected = json.loads(SELECTED_MODEL_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return _default_model_ref()
    if not selected.get("building_id") or not selected.get("model_id"):
        return _default_model_ref()
    return {"building_id": str(selected["building_id"]), "model_id": str(selected["model_id"])}


def _model_manifest_for_ref(building_id: str | None = None, model_id: str | None = None, *, strict: bool = False) -> dict:
    selected = (
        {"building_id": str(building_id), "model_id": str(model_id)}
        if building_id and model_id
        else _load_selected_model_ref()
    )
    if selected == _default_model_ref():
        return default_model_manifest()
    manifest_path = ROOT / "models" / "building_surrogates" / selected["building_id"] / selected["model_id"] / "model_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        if strict:
            raise RuntimeError(f"model manifest not found or invalid: {manifest_path}")
        return default_model_manifest()
    model_dir = _resolve_model_dir(manifest)
    if not model_dir.exists():
        if strict:
            raise RuntimeError(f"model directory not found: {model_dir}")
        return default_model_manifest()
    manifest["model_dir"] = str(model_dir)
    return manifest


def _active_model_manifest() -> dict:
    return _model_manifest_for_ref()


def _resolve_model_dir(manifest: dict) -> Path:
    raw_dir = str(manifest.get("model_dir") or "").strip()
    if raw_dir:
        path = Path(raw_dir)
        if path.exists():
            return path
        if not path.is_absolute():
            root_relative = ROOT / path
            if root_relative.exists():
                return root_relative

    building_id = manifest.get("building_id")
    model_id = manifest.get("model_id")
    if manifest.get("is_default") or model_id == DEFAULT_MODEL_ID:
        return ROOT / "models" / DEFAULT_MODEL_ID
    if building_id and model_id:
        return ROOT / "models" / "building_surrogates" / str(building_id) / str(model_id)
    return MODEL_DIR


def _active_model_dir(manifest: dict | None = None) -> Path:
    manifest = manifest or _active_model_manifest()
    return _resolve_model_dir(manifest)


def _active_model_context(manifest: dict | None = None) -> dict:
    manifest = manifest or _active_model_manifest()
    dataset = manifest.get("dataset") or {}
    return {
        "building_id": manifest.get("building_id"),
        "model_id": manifest.get("model_id"),
        "name": manifest.get("building_name") or manifest.get("name"),
        "is_default": bool(manifest.get("is_default")),
        "status": manifest.get("status", "ready"),
        "sample_count": dataset.get("sample_count"),
        "weather_file": dataset.get("weather_file"),
    }


def clear_model_cache() -> None:
    _load_bundle.cache_clear()


@lru_cache(maxsize=4)
def _load_bundle(model_dir: str | None = None):
    model_path = Path(model_dir) if model_dir else MODEL_DIR
    if not model_path.exists():
        raise RuntimeError(f"Model directory not found: {model_path}")
    device = _select_device()
    static_scaler = joblib.load(model_path / "static_scaler.joblib")
    dynamic_scaler = joblib.load(model_path / "dynamic_scaler.joblib")
    target_scaler = joblib.load(model_path / "target_scaler.joblib")
    members = []
    for member_dir in sorted(d for d in model_path.glob("member_*") if d.is_dir()):
        checkpoint = torch.load(member_dir / "model_state.pt", map_location=device)
        cfg = train.TrainingConfig(**checkpoint["config"])
        model = train.HCDeepOnet(len(STATIC_COLUMNS), len(DYNAMIC_COLUMNS), len(TARGET_COLUMNS), cfg).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        members.append(model)
    if not members:
        raise RuntimeError(f"No ensemble members found under {model_path}")
    mean, scale = train.target_scaler_tensors(target_scaler, device)
    return {
        "device": device,
        "members": members,
        "static_scaler": static_scaler,
        "dynamic_scaler": dynamic_scaler,
        "target_mean": mean,
        "target_scale": scale,
        "model_dir": str(model_path),
    }


@lru_cache(maxsize=1)
def _base_weather_features() -> pd.DataFrame:
    if not Path(train.DATASET_PATH).exists():
        # Public release ships no training dataset: derive the same daily
        # features directly from the base Nanjing EPW used to build it.
        from office_geometry import choose_weather_file

        epw, _ = choose_weather_file()
        features = build_daily_weather_features(epw)
        return features[["day_index", "month", "day", "day_label", *DYNAMIC_COLUMNS]].reset_index(drop=True)
    frame = train.load_sequence_frame()
    base = frame[frame["sample_id"] == 2].sort_values("day_index")
    if base.empty:
        base = frame.groupby("day_index", sort=True).first().reset_index()
    return base[["day_index", "month", "day", "day_label", *DYNAMIC_COLUMNS]].reset_index(drop=True)


@lru_cache(maxsize=32)
def _weather_features_for_scenario(scenario: str) -> pd.DataFrame:
    if scenario == "base":
        return _base_weather_features()
    matches = sorted(WEATHER_OOD_DIR.glob(f"*__{scenario}.epw"))
    if not matches:
        valid = ", ".join(_list_weather_scenarios())
        raise HTTPException(status_code=404, detail=f"Unknown scenario '{scenario}'. Valid: {valid}")
    features = build_daily_weather_features(matches[0])
    return features[["day_index", "month", "day", "day_label", *DYNAMIC_COLUMNS]].reset_index(drop=True)


def _list_weather_scenarios() -> list[str]:
    names = ["base"]
    for path in sorted(WEATHER_OOD_DIR.glob("*.epw")):
        if "__" in path.stem:
            names.append(path.stem.split("__", 1)[1])
    return names


def _baseline_summary_for_manifest(manifest: dict) -> dict[str, float]:
    defaults = {
        "annual_total_input_kwh": 2267242.85,
        "annual_discomfort_hours": 4.140020576131692,
        "annual_electricity_kwh": 1720906.56,
        "annual_hvac_input_kwh": 546336.29,
    }
    if not manifest.get("is_default"):
        summary_path = project_dir(str(manifest.get("building_id"))) / "datasets" / "office_sample_summary.csv"
        if summary_path.exists():
            try:
                frame = pd.read_csv(summary_path)
                row = frame.loc[frame["sample_id"].astype(int) == 1].iloc[0]
                heating_input = float(row.get("annual_heating_input_kwh", 0.0) or 0.0)
                cooling_input = float(row.get("annual_cooling_input_kwh", 0.0) or 0.0)
                hvac_input = heating_input + cooling_input
                if hvac_input <= 0:
                    hvac_input = float(row["annual_total_input_kwh"]) - float(row.get("annual_electricity_kwh", 0.0) or 0.0)
                return {
                    "annual_total_input_kwh": float(row["annual_total_input_kwh"]),
                    "annual_discomfort_hours": float(row["annual_discomfort_hours"]),
                    "annual_electricity_kwh": float(row.get("annual_electricity_kwh", 0.0) or 0.0),
                    "annual_hvac_input_kwh": hvac_input,
                }
            except Exception:
                pass
    if not PARETO_RUNTIME_SUMMARY.exists():
        return defaults
    try:
        with PARETO_RUNTIME_SUMMARY.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError):
        return defaults
    total_input = float(data.get("annual_total_input_kwh", defaults["annual_total_input_kwh"]))
    electricity = float(data.get("annual_electricity_kwh", defaults["annual_electricity_kwh"]))
    return {
        "annual_total_input_kwh": total_input,
        "annual_discomfort_hours": float(data.get("annual_discomfort_hours", defaults["annual_discomfort_hours"])),
        "annual_electricity_kwh": electricity,
        "annual_hvac_input_kwh": float(data.get("annual_hvac_input_kwh", total_input - electricity)),
    }


def _baseline_summary() -> dict[str, float]:
    return _baseline_summary_for_manifest(_active_model_manifest())


def _parse_markdown_table(lines: list[str]) -> dict:
    rows = []
    for line in lines:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if cells and all(set(cell.replace(" ", "")) <= {"-", ":"} for cell in cells):
            continue
        rows.append(cells)
    header = rows[0] if rows else []
    body = rows[1:] if len(rows) > 1 else []
    return {"type": "table", "header": header, "rows": body}


def _parse_markdown_blocks(text: str) -> list[dict]:
    blocks = []
    paragraph: list[str] = []
    lines = text.splitlines()
    i = 0

    def flush_paragraph():
        nonlocal paragraph
        if paragraph:
            blocks.append({"type": "paragraph", "text": " ".join(part.strip() for part in paragraph).strip()})
            paragraph = []

    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()
        if not stripped:
            flush_paragraph()
            i += 1
            continue
        if stripped.startswith("```"):
            flush_paragraph()
            code = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code.append(lines[i].rstrip())
                i += 1
            blocks.append({"type": "code", "text": "\n".join(code)})
            i += 1
            continue
        if stripped == "$$":
            flush_paragraph()
            formula = []
            i += 1
            while i < len(lines) and lines[i].strip() != "$$":
                formula.append(lines[i].strip())
                i += 1
            blocks.append({"type": "formula", "text": "\n".join(formula)})
            i += 1
            continue
        if stripped.startswith("#"):
            flush_paragraph()
            level = len(stripped) - len(stripped.lstrip("#"))
            blocks.append({"type": "heading", "level": min(level, 3), "text": stripped[level:].strip()})
            i += 1
            continue
        if stripped.startswith("|") and "|" in stripped[1:]:
            flush_paragraph()
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i].strip())
                i += 1
            blocks.append(_parse_markdown_table(table_lines))
            continue
        if stripped.startswith("- ") or stripped.startswith("* "):
            flush_paragraph()
            items = []
            while i < len(lines):
                item = lines[i].strip()
                if not (item.startswith("- ") or item.startswith("* ")):
                    break
                items.append(item[2:].strip())
                i += 1
            blocks.append({"type": "list", "ordered": False, "items": items})
            continue
        if len(stripped) > 3 and stripped[0].isdigit() and ". " in stripped[:5]:
            flush_paragraph()
            items = []
            while i < len(lines):
                item = lines[i].strip()
                if not (len(item) > 3 and item[0].isdigit() and ". " in item[:5]):
                    break
                items.append(item.split(". ", 1)[1].strip())
                i += 1
            blocks.append({"type": "list", "ordered": True, "items": items})
            continue
        paragraph.append(stripped)
        i += 1
    flush_paragraph()
    return blocks


def _load_principle_doc(spec: dict[str, str]) -> dict:
    path = PAPER_DIR / spec["source"]
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Principle document not found: {spec['source']}")
    text = path.read_text(encoding="utf-8")
    blocks = _parse_markdown_blocks(text)
    title = next((b["text"] for b in blocks if b["type"] == "heading" and b["level"] == 1), path.stem)
    return {
        "id": spec["id"],
        "title": title,
        "sourceName": spec["source"],
        "summary": spec["summary"],
        "blocks": blocks,
        "markdown": text,
    }


def _predict(static_raw: np.ndarray, dynamic_raw: np.ndarray, model_dir: str | Path | None = None) -> dict[str, np.ndarray]:
    bundle = _load_bundle(str(model_dir or _active_model_dir()))
    device = bundle["device"]
    sx = torch.tensor(bundle["static_scaler"].transform(static_raw).astype(np.float32), device=device)
    dyn_scaled = bundle["dynamic_scaler"].transform(dynamic_raw.reshape(-1, dynamic_raw.shape[-1])).reshape(dynamic_raw.shape)
    dx = torch.tensor(dyn_scaled.astype(np.float32), device=device)
    member_preds = []
    with torch.no_grad():
        for model in bundle["members"]:
            pred = train.to_original(model(sx, dx), bundle["target_mean"], bundle["target_scale"])
            member_preds.append(pred.cpu().numpy())
    stack = np.stack(member_preds, axis=0)
    member_annual = stack.sum(axis=2)
    return {
        "daily_mean": stack.mean(axis=0),
        "daily_sigma": stack.std(axis=0),
        "annual_mean": member_annual.mean(axis=0),
        "annual_sigma": member_annual.std(axis=0),
        "member_annual": member_annual,
    }


def _predict_design(inputs: DesignInputs, weather: pd.DataFrame | None = None, manifest: dict | None = None) -> dict:
    manifest = manifest or _active_model_manifest()
    weather = weather if weather is not None else _base_weather_features()
    static = inputs.as_vector()
    dynamic = np.repeat(weather[DYNAMIC_COLUMNS].to_numpy(dtype=float)[None, :, :], len(static), axis=0)
    raw = _predict(static, dynamic, _active_model_dir(manifest))
    result = _format_prediction(raw, inputs.as_dict(), weather)
    result["modelContext"] = _active_model_context(manifest)
    return result


def _format_prediction(raw: dict[str, np.ndarray], inputs: dict[str, float], weather: pd.DataFrame) -> dict:
    daily = raw["daily_mean"][0]
    annual = raw["annual_mean"][0]
    sigma = raw["annual_sigma"][0]
    heating_load = daily[:, 0]
    cooling_load = daily[:, 1]
    discomfort = daily[:, 2]
    heating_input = heating_load / max(inputs["heating_cop"], 1e-9)
    cooling_input = cooling_load / max(inputs["cooling_cop"], 1e-9)
    total_input = heating_input + cooling_input
    days = []
    for idx, row in weather.iterrows():
        days.append({
            "dayIndex": int(row["day_index"]),
            "label": str(row["day_label"]),
            "month": int(row["month"]),
            "day": int(row["day"]),
            "heatingLoadKwh": float(heating_load[idx]),
            "coolingLoadKwh": float(cooling_load[idx]),
            "heatingInputKwh": float(heating_input[idx]),
            "coolingInputKwh": float(cooling_input[idx]),
            "totalInputKwh": float(total_input[idx]),
            "discomfortHours": float(discomfort[idx]),
            "drybulbMeanC": float(row["drybulb_mean_c"]),
        })
    annual_heating_input = float(heating_input.sum())
    annual_cooling_input = float(cooling_input.sum())
    return {
        "inputs": inputs,
        "annual": {
            "heatingLoadKwh": float(annual[0]),
            "coolingLoadKwh": float(annual[1]),
            "discomfortHours": float(annual[2]),
            "heatingInputKwh": annual_heating_input,
            "coolingInputKwh": annual_cooling_input,
            "totalInputKwh": annual_heating_input + annual_cooling_input,
            "sigmaHeatingLoadKwh": float(sigma[0]),
            "sigmaCoolingLoadKwh": float(sigma[1]),
            "sigmaDiscomfortHours": float(sigma[2]),
        },
        "daily": days,
    }


def _candidate_to_inputs(row: np.ndarray, lock_setpoints: bool) -> DesignInputs:
    keys = [
        "wall_insulation_m",
        "roof_insulation_m",
        "window_u_w_m2k",
        "window_shgc",
        "infiltration_ach",
        "window_wall_ratio",
        "cooling_setpoint_c",
        "heating_setpoint_c",
        "cooling_cop",
        "heating_cop",
    ]
    data = {key: float(row[i]) for i, key in enumerate(keys)}
    if lock_setpoints:
        data["cooling_setpoint_c"] = BASE_DESIGN["cooling_setpoint_c"]
        data["heating_setpoint_c"] = BASE_DESIGN["heating_setpoint_c"]
    return DesignInputs(**data)


def _generate_candidates(count: int, seed: int, lock_setpoints: bool) -> list[DesignInputs]:
    keys = [
        "wall_insulation_m",
        "roof_insulation_m",
        "window_u_w_m2k",
        "window_shgc",
        "infiltration_ach",
        "window_wall_ratio",
        "cooling_setpoint_c",
        "heating_setpoint_c",
        "cooling_cop",
        "heating_cop",
    ]
    lower = [DESIGN_BOUNDS[k][0] for k in keys]
    upper = [DESIGN_BOUNDS[k][1] for k in keys]
    power = int(math.ceil(math.log2(count)))
    sampler = qmc.Sobol(d=len(keys), scramble=True, seed=seed)
    raw = qmc.scale(sampler.random_base2(power), lower, upper)[:count]
    return [_candidate_to_inputs(row, lock_setpoints) for row in raw]


def _dominates(a: dict, b: dict) -> bool:
    return (
        a["annualRobustTotalInputKwh"] <= b["annualRobustTotalInputKwh"]
        and a["retrofitCostCny"] <= b["retrofitCostCny"]
        and a["annualDiscomfortHours"] <= b["annualDiscomfortHours"]
        and (
            a["annualRobustTotalInputKwh"] < b["annualRobustTotalInputKwh"]
            or a["retrofitCostCny"] < b["retrofitCostCny"]
            or a["annualDiscomfortHours"] < b["annualDiscomfortHours"]
        )
    )


def _pareto_flags(points: list[dict]) -> list[bool]:
    flags = []
    for i, point in enumerate(points):
        dominated = any(i != j and _dominates(other, point) for j, other in enumerate(points))
        flags.append(not dominated)
    return flags


def _estimate_cost(inputs: DesignInputs) -> float:
    data = inputs.as_dict()
    sample = sampling.SampleDefinition(
        sample_id=999999,
        sample_type="candidate",
        wall_insulation_m=data["wall_insulation_m"],
        roof_insulation_m=data["roof_insulation_m"],
        window_u=data["window_u_w_m2k"],
        window_shgc=data["window_shgc"],
        infiltration_ach=data["infiltration_ach"],
        window_wall_ratio=data["window_wall_ratio"],
        cooling_setpoint_c=data["cooling_setpoint_c"],
        heating_setpoint_c=data["heating_setpoint_c"],
        cooling_cop=data["cooling_cop"],
        heating_cop=data["heating_cop"],
    )
    return float(sampling.estimate_retrofit_cost_cny(sample))


def _fixed_building_inputs(cooling_setpoint: float, heating_setpoint: float) -> DesignInputs:
    data = dict(BASE_DESIGN)
    data["cooling_setpoint_c"] = float(cooling_setpoint)
    data["heating_setpoint_c"] = float(heating_setpoint)
    return DesignInputs(**data)


@app.get("/api/health")
def health():
    bundle = _load_bundle(str(_active_model_dir()))
    return {
        "status": "ok",
        "device": str(bundle["device"]),
        "members": len(bundle["members"]),
        "modelDir": str(bundle["model_dir"]),
        "selectedModel": _active_model_context(),
    }


@app.get("/api/metadata")
def metadata():
    baseline = _baseline_summary()
    manifest = _active_model_manifest()
    geometry = manifest.get("geometry") or {}
    dataset = manifest.get("dataset") or {}
    return {
        "building": {
            "name": manifest.get("building_name") or manifest.get("name") or "南京高层办公原型",
            "floorAreaM2": geometry.get("floor_area_m2", 19440),
            "floors": geometry.get("floor_count", 15),
            "heightM": geometry.get("height_m", 58.5),
            "weather": dataset.get("weather_file", "Nanjing TMYx 2011-2025"),
        },
        "model": {
            "selected": _active_model_context(),
            "staticFeatures": len(STATIC_COLUMNS),
            "dailyWeatherFeatures": len(DYNAMIC_COLUMNS),
            "weatherDegreeHour": "HDH16",
            "comfortMetric": "人员加权等效不舒适小时（|PMV|>1.5）",
            "baselineAnnualTotalInputKwh": baseline["annual_total_input_kwh"],
            "baselineAnnualHvacInputKwh": baseline["annual_hvac_input_kwh"],
            "baselineAnnualDiscomfortHours": baseline["annual_discomfort_hours"],
            "baselineAnnualElectricityKwh": baseline["annual_electricity_kwh"],
        },
        "inputs": [
            {
                "key": key,
                "label": DISPLAY_LABELS[key],
                "min": DESIGN_BOUNDS[key][0],
                "max": DESIGN_BOUNDS[key][1],
                "default": BASE_DESIGN[key],
            }
            for key in STATIC_COLUMNS
        ],
        "weatherScenarios": _list_weather_scenarios(),
    }


@app.get("/api/input-features")
def input_features():
    weather = _base_weather_features()
    climate_days = []
    for row in weather.itertuples(index=False):
        climate_days.append({
            "dayIndex": int(row.day_index),
            "label": str(row.day_label),
            "month": int(row.month),
            "day": int(row.day),
            "drybulbMeanC": float(row.drybulb_mean_c),
            "drybulbMinC": float(row.drybulb_min_c),
            "drybulbMaxC": float(row.drybulb_max_c),
            "rhMeanPct": float(row.rh_mean_pct),
            "windMeanMS": float(row.wind_mean_m_s),
            "ghiKwhM2": float(row.ghi_kwh_m2),
            "hdh16CH": float(row.hdh16_c_h),
            "cdh26CH": float(row.cdh26_c_h),
        })
    building_features = []
    for key in STATIC_COLUMNS:
        info = BUILDING_FEATURE_DESCRIPTIONS[key]
        lo, hi = DESIGN_BOUNDS[key]
        building_features.append({
            "key": key,
            "label": DISPLAY_LABELS[key],
            "unit": info["unit"],
            "min": lo,
            "max": hi,
            "default": BASE_DESIGN[key],
            "description": info["description"],
        })
    return {
        "summary": {
            "climateFeatureCount": len(CLIMATE_FEATURE_DESCRIPTIONS),
            "buildingFeatureCount": len(building_features),
            "dailyWeatherFeatureCount": len(DYNAMIC_COLUMNS),
            "climateDays": len(climate_days),
            "weather": "Nanjing TMYx 2011-2025",
            "engineeringNote": "模型内部还使用 day_of_year_sin/cos、3 日滚动温度/辐射/风速等工程特征。",
        },
        "climateFeatures": CLIMATE_FEATURE_DESCRIPTIONS,
        "climateDaily": climate_days,
        "buildingFeatures": building_features,
    }


@app.get("/api/principle-docs")
def principle_docs():
    docs = [_load_principle_doc(spec) for spec in PRINCIPLE_DOCS]
    return {"docs": docs}


@app.post("/api/design/predict")
def design_predict(request: ScenarioRequest):
    return _predict_design(request.inputs)


def _design_pareto_result(request: ParetoRequest, manifest: dict | None = None) -> dict:
    manifest = manifest or _active_model_manifest()
    baseline = _baseline_summary_for_manifest(manifest)
    candidates = _generate_candidates(request.count, request.seed, request.lock_setpoints)
    weather = _base_weather_features()
    static = np.asarray([candidate.as_vector()[0] for candidate in candidates], dtype=float)
    dynamic = np.repeat(weather[DYNAMIC_COLUMNS].to_numpy(dtype=float)[None, :, :], len(candidates), axis=0)
    raw = _predict(static, dynamic, _active_model_dir(manifest))
    points = []
    for idx, candidate in enumerate(candidates):
        data = candidate.as_dict()
        annual_load = raw["annual_mean"][idx]
        sigma_load = raw["annual_sigma"][idx]
        total_input = annual_load[0] / data["heating_cop"] + annual_load[1] / data["cooling_cop"]
        sigma_total = math.sqrt(
            (sigma_load[0] / max(data["heating_cop"], 1e-9)) ** 2
            + (sigma_load[1] / max(data["cooling_cop"], 1e-9)) ** 2
        )
        robust_total = total_input + sigma_total
        points.append({
            "id": idx + 1,
            "annualTotalInputKwh": float(total_input),
            "annualRobustTotalInputKwh": float(robust_total),
            "sigmaTotalInputKwh": float(sigma_total),
            "annualHeatingLoadKwh": float(annual_load[0]),
            "annualCoolingLoadKwh": float(annual_load[1]),
            "annualDiscomfortHours": float(annual_load[2]),
            "sigmaDiscomfortHours": float(sigma_load[2]),
            "savingsVsBaselinePct": float(
                (baseline["annual_hvac_input_kwh"] - robust_total)
                / max(baseline["annual_hvac_input_kwh"], 1e-9)
                * 100.0
            ),
            "discomfortChangeVsBaselineH": float(annual_load[2] - baseline["annual_discomfort_hours"]),
            "retrofitCostCny": _estimate_cost(candidate),
            "inputs": data,
        })
    flags = _pareto_flags(points)
    for point, is_front in zip(points, flags):
        point["pareto"] = bool(is_front)
    front = sorted((p for p in points if p["pareto"]), key=lambda p: p["retrofitCostCny"])
    return {
        "count": len(points),
        "modelContext": _active_model_context(manifest),
        "paretoCount": len(front),
        "objectives": ["retrofitCostCny", "annualRobustTotalInputKwh", "annualDiscomfortHours"],
        "baseline": baseline,
        "points": points,
        "front": front,
    }


@app.post("/api/design/pareto")
def design_pareto(request: ParetoRequest):
    return _design_pareto_result(request)


@app.post("/api/operation/predict")
def operation_predict(request: OperationRequest):
    inputs = _fixed_building_inputs(request.inputs.cooling_setpoint_c, request.inputs.heating_setpoint_c)
    return _predict_design(inputs)


@app.post("/api/operation/saving")
def operation_saving(request: OperationRequest):
    base_inputs = _fixed_building_inputs(request.inputs.cooling_setpoint_c, request.inputs.heating_setpoint_c)
    raised_inputs = _fixed_building_inputs(min(request.inputs.cooling_setpoint_c + 1.0, COOLING_SETPOINT_MAX), request.inputs.heating_setpoint_c)
    base = _predict_design(base_inputs)
    raised = _predict_design(raised_inputs)
    base_cooling = base["annual"]["coolingInputKwh"]
    raised_cooling = raised["annual"]["coolingInputKwh"]
    saved = base_cooling - raised_cooling
    summer_saved = 0.0
    for b, r in zip(base["daily"], raised["daily"]):
        if 6 <= b["month"] <= 9:
            summer_saved += b["coolingInputKwh"] - r["coolingInputKwh"]
    return {
        "base": base,
        "raised": raised,
        "saving": {
            "annualCoolingInputSavedKwh": float(saved),
            "summerCoolingInputSavedKwh": float(summer_saved),
            "annualCoolingSavingPct": float(saved / max(base_cooling, 1e-9) * 100.0),
            "summerCoolingSavingPct": float(summer_saved / max(
                sum(d["coolingInputKwh"] for d in base["daily"] if 6 <= d["month"] <= 9), 1e-9
            ) * 100.0),
        },
    }


@app.post("/api/extreme/estimate")
def extreme_estimate(request: ExtremeRequest):
    normal_inputs = _fixed_building_inputs(request.cooling_setpoint_c, request.heating_setpoint_c)
    normal = _predict_design(normal_inputs, _base_weather_features())
    scenario_weather = _weather_features_for_scenario(request.scenario)
    extreme = _predict_design(normal_inputs, scenario_weather)
    normal_avg = normal["annual"]["totalInputKwh"] / max(len(normal["daily"]), 1)
    extreme_peak = max(extreme["daily"], key=lambda d: d["totalInputKwh"])
    extreme_avg = extreme["annual"]["totalInputKwh"] / max(len(extreme["daily"]), 1)
    increase = (extreme_peak["totalInputKwh"] - normal_avg) / max(normal_avg, 1e-9) * 100.0
    avg_increase = (extreme_avg - normal_avg) / max(normal_avg, 1e-9) * 100.0
    return {
        "scenario": request.scenario,
        "normal": {
            "averageDailyTotalInputKwh": float(normal_avg),
            "averageDailyHeatingInputKwh": float(normal["annual"]["heatingInputKwh"] / len(normal["daily"])),
            "averageDailyCoolingInputKwh": float(normal["annual"]["coolingInputKwh"] / len(normal["daily"])),
        },
        "extreme": {
            "averageDailyTotalInputKwh": float(extreme_avg),
            "averageIncreasePct": float(avg_increase),
            "peakDay": extreme_peak,
            "increasePctVsNormalAverage": float(increase),
            "daily": extreme["daily"],
        },
    }


# ---- 智能体路由（DeepSeek 物理增强对话；工具调用/门控将在后续阶段接入）----
from .agent.routes import router as agent_router  # noqa: E402
from .report_routes import router as report_router  # noqa: E402
from .training_routes import router as training_router  # noqa: E402

app.include_router(agent_router)
app.include_router(report_router)
app.include_router(training_router)
