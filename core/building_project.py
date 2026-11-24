"""Project schema for natural-language driven surrogate training.

Supports rectangular and L-shaped, multi-floor office buildings with
perimeter/core zoning and IdealLoads HVAC. Floor height is split into a
ground-floor height and a typical-floor height. It is shared by the
EnergyPlus generator and the Agent training API.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
PROJECTS_DIR = ROOT / "building_projects"
SURROGATE_LIBRARY_DIR = ROOT / "models" / "building_surrogates"
DEFAULT_MODEL_ID = "office_hc_deeponet_ensemble"


@dataclass(frozen=True)
class BuildingIdentity:
    building_name: str = "用户确认建筑"
    building_type: str = "office"
    city: str = "Nanjing"
    weather_file: str = "Nanjing TMY EPW"


@dataclass(frozen=True)
class GeometrySpec:
    plan_shape: str = "rectangle"          # "rectangle" | "l_shape"
    length_m: float = 36.0                 # full bounding-box extent (x)
    width_m: float = 36.0                  # full bounding-box extent (y)
    floor_count: int = 15
    ground_floor_height_m: float = 4.5     # 首层层高（大堂/裙房常见更高）
    typical_floor_height_m: float = 3.9    # 标准层层高（第2层及以上）
    orientation_deg: float = 0.0
    perimeter_depth_m: float = 5.0
    zoning_mode: str = "perimeter_core"
    # L 形专用：从矩形包络的一角切掉一个矩形缺口，形成 L 形。
    # notch_corner 指定缺口所在角（该角连同其周边区域被移除）。
    notch_corner: str = "ne"               # "ne" | "nw" | "se" | "sw"
    notch_length_m: float = 14.0           # 缺口沿 x 方向的尺寸
    notch_width_m: float = 14.0            # 缺口沿 y 方向的尺寸


@dataclass(frozen=True)
class EnvelopeSpec:
    wall_insulation_m: float = 0.02
    roof_insulation_m: float = 0.02
    window_u_w_m2k: float = 3.5
    window_shgc: float = 0.5
    window_wall_ratio: float = 0.4
    infiltration_ach: float = 0.8


@dataclass(frozen=True)
class OperationSpec:
    cooling_setpoint_c: float = 26.0
    heating_setpoint_c: float = 20.0
    cooling_cop: float = 3.0
    heating_cop: float = 1.8
    schedule_template: str = "office_weekday"


@dataclass(frozen=True)
class DatasetTrainingSpec:
    sample_count: int = 128
    sampling_method: str = "sobol"
    seed: int = 2026
    train_ensemble_members: int = 5


@dataclass(frozen=True)
class BuildingProjectSpec:
    building_identity: BuildingIdentity = field(default_factory=BuildingIdentity)
    geometry: GeometrySpec = field(default_factory=GeometrySpec)
    envelope: EnvelopeSpec = field(default_factory=EnvelopeSpec)
    operation: OperationSpec = field(default_factory=OperationSpec)
    dataset_training: DatasetTrainingSpec = field(default_factory=DatasetTrainingSpec)
    parameter_sources: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_scenario(self, label: str | None = None) -> dict[str, object]:
        return {
            "label": label or self.building_identity.building_name,
            "wall_insulation_m": self.envelope.wall_insulation_m,
            "roof_insulation_m": self.envelope.roof_insulation_m,
            "window_u": self.envelope.window_u_w_m2k,
            "window_shgc": self.envelope.window_shgc,
            "infiltration_ach": self.envelope.infiltration_ach,
            "window_wall_ratio": self.envelope.window_wall_ratio,
            "heating_setpoint_c": self.operation.heating_setpoint_c,
            "cooling_setpoint_c": self.operation.cooling_setpoint_c,
            "heating_cop": self.operation.heating_cop,
            "cooling_cop": self.operation.cooling_cop,
        }


def _section(cls, data: dict[str, Any] | None):
    values = data or {}
    allowed = cls.__dataclass_fields__.keys()
    return cls(**{key: values[key] for key in allowed if key in values})


def spec_from_dict(data: dict[str, Any] | None) -> BuildingProjectSpec:
    data = data or {}
    return BuildingProjectSpec(
        building_identity=_section(BuildingIdentity, data.get("building_identity")),
        geometry=_section(GeometrySpec, data.get("geometry")),
        envelope=_section(EnvelopeSpec, data.get("envelope")),
        operation=_section(OperationSpec, data.get("operation")),
        dataset_training=_section(DatasetTrainingSpec, data.get("dataset_training")),
        parameter_sources=dict(data.get("parameter_sources") or {}),
    )


VALID_PLAN_SHAPES = ("rectangle", "l_shape")
VALID_NOTCH_CORNERS = ("ne", "nw", "se", "sw")


def validate_spec(spec: BuildingProjectSpec) -> list[str]:
    errors: list[str] = []
    g, e, o, d = spec.geometry, spec.envelope, spec.operation, spec.dataset_training
    if g.plan_shape not in VALID_PLAN_SHAPES:
        errors.append(f"平面形状 plan_shape 目前仅支持 {VALID_PLAN_SHAPES}。")
    if g.length_m <= 0:
        errors.append("建筑长度 length_m 必须大于 0。")
    if g.width_m <= 0:
        errors.append("建筑宽度 width_m 必须大于 0。")
    if g.floor_count < 1:
        errors.append("楼层数 floor_count 必须至少为 1。")
    if not 2.8 <= g.ground_floor_height_m <= 6.0:
        errors.append("首层层高 ground_floor_height_m 建议在 2.8-6.0 m。")
    if not 2.6 <= g.typical_floor_height_m <= 5.0:
        errors.append("标准层层高 typical_floor_height_m 建议在 2.6-5.0 m。")
    if g.perimeter_depth_m <= 0:
        errors.append("周边区深度 perimeter_depth_m 必须大于 0。")
    if g.perimeter_depth_m * 2 >= min(g.length_m, g.width_m):
        errors.append("周边区深度过大，核心区宽度会小于等于 0。")
    if g.plan_shape == "l_shape":
        if g.notch_corner not in VALID_NOTCH_CORNERS:
            errors.append(f"缺角位置 notch_corner 目前仅支持 {VALID_NOTCH_CORNERS}。")
        d_ = g.perimeter_depth_m
        # Both the L 形主翼(去掉缺角后剩余部分)和副翼(缺角同层的窄边)都要能各自
        # 独立容纳一圈"周边区+核心区"，否则程序化分区无法生成闭合、无重叠的几何。
        if not (2 * d_ < g.notch_length_m < g.length_m - 2 * d_):
            errors.append(
                f"缺角长度 notch_length_m 需满足 2×perimeter_depth_m({2 * d_:.2f}) < notch_length_m < "
                f"length_m - 2×perimeter_depth_m({g.length_m - 2 * d_:.2f})，"
                f"否则主翼或副翼进深不足以容纳周边区+核心区。"
            )
        if not (2 * d_ < g.notch_width_m < g.width_m - 2 * d_):
            errors.append(
                f"缺角宽度 notch_width_m 需满足 2×perimeter_depth_m({2 * d_:.2f}) < notch_width_m < "
                f"width_m - 2×perimeter_depth_m({g.width_m - 2 * d_:.2f})，"
                f"否则主翼或副翼进深不足以容纳周边区+核心区。"
            )
    if not 0.1 <= e.window_wall_ratio <= 0.8:
        errors.append("窗墙比 window_wall_ratio 建议在 0.1-0.8。")
    if o.cooling_setpoint_c <= o.heating_setpoint_c:
        errors.append("供冷设定温度必须高于供暖设定温度。")
    if d.sample_count < 1:
        errors.append("训练样本数 sample_count 必须大于 0。")
    if d.train_ensemble_members < 1:
        errors.append("集成成员数 train_ensemble_members 必须大于 0。")
    return errors


def slugify_name(value: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff_-]+", "_", value.strip())
    slug = slug.strip("_")
    return slug or "building"


def new_building_id(name: str, now: datetime | None = None) -> str:
    stamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return f"{slugify_name(name)}_{stamp}"


def project_dir(building_id: str) -> Path:
    return PROJECTS_DIR / building_id


def save_spec(building_id: str, spec: BuildingProjectSpec) -> Path:
    path = project_dir(building_id)
    path.mkdir(parents=True, exist_ok=True)
    spec_path = path / "building_spec.json"
    spec_path.write_text(json.dumps(spec.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return spec_path


def load_spec(building_id: str) -> BuildingProjectSpec:
    path = project_dir(building_id) / "building_spec.json"
    return spec_from_dict(json.loads(path.read_text(encoding="utf-8")))


def load_spec_path(path: str | Path) -> BuildingProjectSpec:
    return spec_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def notch_area_m2(spec: BuildingProjectSpec) -> float:
    g = spec.geometry
    if g.plan_shape != "l_shape":
        return 0.0
    return float(g.notch_length_m) * float(g.notch_width_m)


def floor_plate_area_m2(spec: BuildingProjectSpec) -> float:
    g = spec.geometry
    return float(g.length_m) * float(g.width_m) - notch_area_m2(spec)


def total_floor_area_m2(spec: BuildingProjectSpec) -> float:
    return floor_plate_area_m2(spec) * spec.geometry.floor_count


def total_height_m(spec: BuildingProjectSpec) -> float:
    g = spec.geometry
    return float(g.ground_floor_height_m) + float(g.typical_floor_height_m) * max(0, g.floor_count - 1)


def default_spec_dict() -> dict[str, Any]:
    spec = BuildingProjectSpec()
    data = spec.to_dict()
    data["parameter_sources"] = {}
    for section, values in data.items():
        if section == "parameter_sources" or not isinstance(values, dict):
            continue
        for key in values:
            data["parameter_sources"][f"{section}.{key}"] = "default_assumption"
    return data


def default_model_manifest() -> dict[str, Any]:
    return {
        "building_id": "default_office",
        "model_id": DEFAULT_MODEL_ID,
        "building_name": "南京高层办公默认模型",
        "status": "ready",
        "is_default": True,
        "model_dir": str(ROOT / "models" / DEFAULT_MODEL_ID),
        "geometry": {
            "plan_shape": "rectangle",
            "length_m": 36.0,
            "width_m": 36.0,
            "floor_count": 15,
            "ground_floor_height_m": 3.9,
            "typical_floor_height_m": 3.9,
            "floor_area_m2": 19440.0,
        },
        "dataset": {
            "sample_count": None,
            "weather_file": "Nanjing TMYx 2011-2025",
        },
        "model": {
            "type": "CHU-DeepONet",
            "ensemble_members": 5,
        },
    }
