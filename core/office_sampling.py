"""Office design-variable sampling + research-grade retrofit cost (forked from demo2/generate_retrofit_dataset.py).

10 sampled dimensions: 6 envelope/design + WWR + 2 occupied setpoints + (COP x2 fold downstream).
Window area is WWR-dynamic in the cost model. Setpoints carry no capital cost.
"""
from __future__ import annotations

from dataclasses import dataclass

from scipy.stats import qmc

from office_geometry import BUILDING_X_M, BUILDING_Y_M, FLOOR_AREA_M2, TOTAL_HEIGHT_M

GROSS_FACADE_M2 = 2.0 * (BUILDING_X_M + BUILDING_Y_M) * TOTAL_HEIGHT_M   # 8424
ROOF_AREA_M2 = BUILDING_X_M * BUILDING_Y_M                              # 1296

# sample_id=1 baseline (reference "before-retrofit" office)
BASELINE = {
    "wall_insulation_m": 0.020,
    "roof_insulation_m": 0.020,
    "window_u": 3.50,
    "window_shgc": 0.50,
    "infiltration_ach": 0.80,
    "window_wall_ratio": 0.40,
    "cooling_setpoint_c": 26.0,
    "heating_setpoint_c": 20.0,
    "cooling_cop": 3.00,
    "heating_cop": 1.80,
}

# ordered for the 10-D Sobol cube
VAR_ORDER = [
    "wall_insulation_m", "roof_insulation_m", "window_u", "window_shgc",
    "infiltration_ach", "window_wall_ratio", "cooling_setpoint_c",
    "heating_setpoint_c", "cooling_cop", "heating_cop",
]
VARIABLE_BOUNDS = {
    "wall_insulation_m": (0.020, 0.120),
    "roof_insulation_m": (0.020, 0.150),
    "window_u": (1.40, 5.80),
    "window_shgc": (0.30, 0.55),
    "infiltration_ach": (0.30, 0.80),
    "window_wall_ratio": (0.30, 0.60),
    "cooling_setpoint_c": (24.0, 28.0),
    "heating_setpoint_c": (16.0, 20.0),
    "cooling_cop": (3.00, 6.00),
    "heating_cop": (1.80, 4.00),
}
VARIABLE_STEPS = {
    "wall_insulation_m": 0.005,
    "roof_insulation_m": 0.005,
    "window_u": 0.05,
    "window_shgc": 0.02,
    "infiltration_ach": 0.02,
    "window_wall_ratio": 0.02,
    "cooling_setpoint_c": 0.5,
    "heating_setpoint_c": 0.5,
    "cooling_cop": 0.10,
    "heating_cop": 0.10,
}

# research-grade unit costs (CNY)
WALL_FIXED = 120.0
WALL_THICK = 5500.0
ROOF_FIXED = 60.0
ROOF_THICK = 650.0
WINDOW_U_PREMIUM = 620.0
WINDOW_SHGC_PREMIUM = 130.0
WINDOW_BASE_GLAZING = 90.0          # baseline glazing provision per m2 (so higher WWR costs more)
AIRTIGHT_PREMIUM_PER_FLOOR = 30.0
COOLING_COP_PREMIUM_PER_FLOOR = 48.0
HEATING_COP_PREMIUM_PER_FLOOR = 42.0


@dataclass(frozen=True)
class SampleDefinition:
    sample_id: int
    sample_type: str
    wall_insulation_m: float
    roof_insulation_m: float
    window_u: float
    window_shgc: float
    infiltration_ach: float
    window_wall_ratio: float
    cooling_setpoint_c: float
    heating_setpoint_c: float
    cooling_cop: float
    heating_cop: float

    def to_scenario(self) -> dict[str, object]:
        return {
            "id": f"sample_{self.sample_id:05d}",
            "label": f"office_sample_{self.sample_id:05d}",
            "wall_insulation_m": self.wall_insulation_m,
            "roof_insulation_m": self.roof_insulation_m,
            "window_u": self.window_u,
            "window_shgc": self.window_shgc,
            "infiltration_ach": self.infiltration_ach,
            "window_wall_ratio": self.window_wall_ratio,
            "cooling_setpoint_c": self.cooling_setpoint_c,
            "heating_setpoint_c": self.heating_setpoint_c,
            "cooling_cop": self.cooling_cop,
            "heating_cop": self.heating_cop,
        }

    def to_record(self) -> dict[str, object]:
        r = self.to_scenario()
        r.pop("id"); r.pop("label")
        r["window_u_w_m2k"] = r.pop("window_u")
        return {"sample_id": self.sample_id, "sample_type": self.sample_type, **r}


def quantize(value: float, lower: float, upper: float, step: float) -> float:
    snapped = round((value - lower) / step) * step + lower
    snapped = max(lower, min(upper, snapped))
    decimals = max(0, len(str(step).split(".")[-1].rstrip("0")))
    return round(snapped, decimals)


def estimate_retrofit_cost_cny(s: SampleDefinition) -> float:
    window_area = s.window_wall_ratio * GROSS_FACADE_M2
    opaque_wall_area = GROSS_FACADE_M2 - window_area

    wall_cost = 0.0
    if s.wall_insulation_m > BASELINE["wall_insulation_m"]:
        wall_cost = opaque_wall_area * (WALL_FIXED + WALL_THICK * s.wall_insulation_m)
    roof_cost = 0.0
    if s.roof_insulation_m > BASELINE["roof_insulation_m"]:
        roof_cost = ROOF_AREA_M2 * (ROOF_FIXED + ROOF_THICK * s.roof_insulation_m)

    glazing_base = window_area * WINDOW_BASE_GLAZING
    window_u_cost = window_area * WINDOW_U_PREMIUM * max(0.0, BASELINE["window_u"] - s.window_u) / (
        BASELINE["window_u"] - VARIABLE_BOUNDS["window_u"][0])
    window_shgc_cost = window_area * WINDOW_SHGC_PREMIUM * max(0.0, BASELINE["window_shgc"] - s.window_shgc) / (
        BASELINE["window_shgc"] - VARIABLE_BOUNDS["window_shgc"][0])
    airtight_cost = FLOOR_AREA_M2 * AIRTIGHT_PREMIUM_PER_FLOOR * max(0.0, BASELINE["infiltration_ach"] - s.infiltration_ach) / (
        BASELINE["infiltration_ach"] - VARIABLE_BOUNDS["infiltration_ach"][0])
    cooling_cost = FLOOR_AREA_M2 * COOLING_COP_PREMIUM_PER_FLOOR * max(0.0, s.cooling_cop - BASELINE["cooling_cop"]) / (
        VARIABLE_BOUNDS["cooling_cop"][1] - BASELINE["cooling_cop"])
    heating_cost = FLOOR_AREA_M2 * HEATING_COP_PREMIUM_PER_FLOOR * max(0.0, s.heating_cop - BASELINE["heating_cop"]) / (
        VARIABLE_BOUNDS["heating_cop"][1] - BASELINE["heating_cop"])

    return round(glazing_base + wall_cost + roof_cost + window_u_cost + window_shgc_cost
                 + airtight_cost + cooling_cost + heating_cost, 2)


def _baseline_sample() -> SampleDefinition:
    return SampleDefinition(sample_id=1, sample_type="baseline", **BASELINE)


def generate_samples(total_samples: int, seed: int) -> list[SampleDefinition]:
    if total_samples < 1:
        raise ValueError("total_samples must be positive")
    samples = [_baseline_sample()]
    if total_samples == 1:
        return samples

    d = len(VAR_ORDER)
    lower = [VARIABLE_BOUNDS[k][0] for k in VAR_ORDER]
    upper = [VARIABLE_BOUNDS[k][1] for k in VAR_ORDER]
    sampler = qmc.Sobol(d=d, scramble=True, seed=seed)
    used = {tuple(getattr(samples[0], k) for k in VAR_ORDER)}
    next_id = 2
    batch, max_batches, count = 128, 128, 0
    while len(samples) < total_samples and count < max_batches:
        scaled = qmc.scale(sampler.random(batch), lower, upper)
        for row in scaled:
            vals = {k: quantize(float(row[i]), *VARIABLE_BOUNDS[k], VARIABLE_STEPS[k]) for i, k in enumerate(VAR_ORDER)}
            key = tuple(vals[k] for k in VAR_ORDER)
            if key in used:
                continue
            used.add(key)
            samples.append(SampleDefinition(sample_id=next_id, sample_type="sobol", **vals))
            next_id += 1
            if len(samples) == total_samples:
                break
        count += 1
    if len(samples) != total_samples:
        raise RuntimeError(f"Generated only {len(samples)}/{total_samples} unique samples")
    return samples
