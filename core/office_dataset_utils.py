"""Weather features + daily load/comfort summarisation (forked from demo3/daily_dataset_utils.py).

- build_daily_weather_features: daily weather vector with HDH16/CDH26 descriptors.
- summarize_daily_simulation: daily electricity/lights/equipment/heating/cooling from the
  DAILY meter file (eplusmtr.csv). Heating/cooling come from Heating/Cooling:EnergyTransfer.
- summarize_daily_comfort: from the HOURLY variable file (eplusout.csv), computes daily
  occupant-weighted equivalent discomfort hours from zone PMV + occupant count.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

EPW_COLUMNS = [
    "year", "month", "day", "hour", "minute", "data_source",
    "dry_bulb_c", "dew_point_c", "relative_humidity_pct", "station_pressure_pa",
    "etr_wh_m2", "etrn_wh_m2", "horizontal_ir_wh_m2",
    "global_horizontal_radiation_wh_m2", "direct_normal_radiation_wh_m2",
    "diffuse_horizontal_radiation_wh_m2", "global_horizontal_illuminance_lux",
    "direct_normal_illuminance_lux", "diffuse_horizontal_illuminance_lux",
    "zenith_luminance_cd_m2", "wind_direction_deg", "wind_speed_m_s",
    "total_sky_cover_tenths", "opaque_sky_cover_tenths", "visibility_km",
    "ceiling_height_m", "present_weather_observation", "present_weather_codes",
    "precipitable_water_mm", "aerosol_optical_depth_thousandths", "snow_depth_cm",
    "days_since_last_snowfall", "albedo", "liquid_precipitation_depth_mm",
    "liquid_precipitation_quantity_hr",
]

METER_ELECTRICITY = ["electricity:facility"]
METER_LIGHTS = ["interiorlights:electricity"]
METER_EQUIPMENT = ["interiorequipment:electricity"]
# total ideal-loads (sensible+latent) custom meters first; sensible EnergyTransfer as fallback
METER_HEATING = ["idealtotalheating", "heating:energytransfer", "districtheatingwater:facility", "districtheating:facility"]
METER_COOLING = ["idealtotalcooling", "cooling:energytransfer", "districtcooling:facility"]

PMV_DISCOMFORT_THRESHOLD = 1.5


def parse_epw_location(epw_path: Path) -> dict[str, object]:
    first_line = epw_path.read_text(encoding="utf-8", errors="ignore").splitlines()[0]
    parts = [p.strip() for p in first_line.split(",")]
    if len(parts) < 10 or parts[0].upper() != "LOCATION":
        raise RuntimeError(f"Unexpected EPW header in {epw_path}")
    return {
        "latitude": float(parts[6]), "longitude": float(parts[7]),
        "timezone": float(parts[8]), "elevation_m": float(parts[9]),
        "location_name": f"{parts[1]}, {parts[2]}, {parts[3]}",
    }


def load_epw_dataframe(epw_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(epw_path, skiprows=8, header=None, names=EPW_COLUMNS, encoding="utf-8")
    numeric = ["month", "day", "hour", "dry_bulb_c", "dew_point_c", "relative_humidity_pct",
               "global_horizontal_radiation_wh_m2", "direct_normal_radiation_wh_m2",
               "diffuse_horizontal_radiation_wh_m2", "wind_speed_m_s"]
    for c in numeric:
        frame[c] = pd.to_numeric(frame[c], errors="coerce")
    frame.loc[frame["dry_bulb_c"] >= 99.0, "dry_bulb_c"] = np.nan
    frame.loc[frame["dew_point_c"] >= 99.0, "dew_point_c"] = np.nan
    frame.loc[frame["relative_humidity_pct"] >= 999.0, "relative_humidity_pct"] = np.nan
    frame.loc[frame["wind_speed_m_s"] >= 999.0, "wind_speed_m_s"] = np.nan
    for c in ["global_horizontal_radiation_wh_m2", "direct_normal_radiation_wh_m2", "diffuse_horizontal_radiation_wh_m2"]:
        frame.loc[frame[c] >= 999999.0, c] = 0.0
        frame[c] = frame[c].fillna(0.0)
    return frame


def build_daily_weather_features(epw_path: Path) -> pd.DataFrame:
    location = parse_epw_location(epw_path)
    frame = load_epw_dataframe(epw_path)
    day_lookup = frame[["month", "day"]].drop_duplicates().reset_index(drop=True)
    day_lookup["day_index"] = np.arange(1, len(day_lookup) + 1)
    day_lookup["day_label"] = day_lookup.apply(lambda r: f"{int(r['month']):02d}-{int(r['day']):02d}", axis=1)
    angle = 2.0 * np.pi * (day_lookup["day_index"] - 1) / len(day_lookup)
    day_lookup["day_of_year_sin"] = np.sin(angle)
    day_lookup["day_of_year_cos"] = np.cos(angle)
    frame = frame.merge(day_lookup[["month", "day", "day_index"]], on=["month", "day"], how="left")
    # weather-only degree-hours (fixed base, kept as descriptors; setpoint enters model as input)
    frame["hdh16_c_h"] = np.clip(16.0 - frame["dry_bulb_c"].fillna(16.0), 0.0, None)
    frame["cdh26_c_h"] = np.clip(frame["dry_bulb_c"].fillna(26.0) - 26.0, 0.0, None)
    g = frame.groupby("day_index", sort=True)
    daily = g.agg(
        month=("month", "first"), day=("day", "first"),
        drybulb_mean_c=("dry_bulb_c", "mean"), drybulb_min_c=("dry_bulb_c", "min"), drybulb_max_c=("dry_bulb_c", "max"),
        dewpoint_mean_c=("dew_point_c", "mean"), rh_mean_pct=("relative_humidity_pct", "mean"),
        rh_min_pct=("relative_humidity_pct", "min"), rh_max_pct=("relative_humidity_pct", "max"),
        wind_mean_m_s=("wind_speed_m_s", "mean"), wind_max_m_s=("wind_speed_m_s", "max"),
        ghi_wh_m2=("global_horizontal_radiation_wh_m2", "sum"), dni_wh_m2=("direct_normal_radiation_wh_m2", "sum"),
        dhi_wh_m2=("diffuse_horizontal_radiation_wh_m2", "sum"),
        hdh16_c_h=("hdh16_c_h", "sum"), cdh26_c_h=("cdh26_c_h", "sum"),
    ).reset_index()
    daily["temp_range_c"] = daily["drybulb_max_c"] - daily["drybulb_min_c"]
    daily["ghi_kwh_m2"] = daily["ghi_wh_m2"] / 1000.0
    daily["dni_kwh_m2"] = daily["dni_wh_m2"] / 1000.0
    daily["dhi_kwh_m2"] = daily["dhi_wh_m2"] / 1000.0
    daily = daily.merge(day_lookup, on=["day_index", "month", "day"], how="left")
    daily["drybulb_mean_3d_c"] = daily["drybulb_mean_c"].rolling(3, min_periods=1).mean()
    daily["ghi_3d_kwh_m2"] = daily["ghi_kwh_m2"].rolling(3, min_periods=1).mean()
    daily["wind_mean_3d_m_s"] = daily["wind_mean_m_s"].rolling(3, min_periods=1).mean()
    daily["latitude"] = location["latitude"]
    daily["longitude"] = location["longitude"]
    daily["elevation_m"] = location["elevation_m"]
    daily["location_name"] = location["location_name"]
    daily["weather_file"] = str(epw_path)
    ordered = [
        "day_index", "month", "day", "day_label", "day_of_year_sin", "day_of_year_cos",
        "latitude", "longitude", "elevation_m", "location_name", "weather_file",
        "drybulb_mean_c", "drybulb_min_c", "drybulb_max_c", "temp_range_c", "dewpoint_mean_c",
        "rh_mean_pct", "rh_min_pct", "rh_max_pct", "wind_mean_m_s", "wind_max_m_s",
        "ghi_kwh_m2", "dni_kwh_m2", "dhi_kwh_m2", "hdh16_c_h", "cdh26_c_h",
        "drybulb_mean_3d_c", "ghi_3d_kwh_m2", "wind_mean_3d_m_s",
    ]
    return daily[ordered]


def save_daily_weather_features(epw_path: Path, output_csv_path: Path) -> pd.DataFrame:
    daily = build_daily_weather_features(epw_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
    return daily


def _first_meter_col(df: pd.DataFrame, needles: list[str]) -> pd.Series | None:
    for needle in needles:
        for c in df.columns:
            if needle.lower() in c.lower():
                return pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return None


def summarize_daily_simulation(meter_csv_path: Path) -> pd.DataFrame:
    """Daily energy from the DAILY meter file. Returns one row per day with loads in kWh."""
    if not meter_csv_path.exists():
        raise RuntimeError(f"meter csv not found: {meter_csv_path}")
    df = pd.read_csv(meter_csv_path, encoding="utf-8-sig")
    n = len(df)
    to_kwh = 1.0 / 3_600_000.0

    def series(needles):
        s = _first_meter_col(df, needles)
        return (s.reset_index(drop=True).astype(float) if s is not None
                else pd.Series(np.zeros(n), dtype=float))

    out = pd.DataFrame({"day_index": np.arange(1, n + 1)})
    out["daily_electricity_kwh"] = series(METER_ELECTRICITY) * to_kwh
    out["daily_lights_kwh"] = series(METER_LIGHTS) * to_kwh
    out["daily_equipment_kwh"] = series(METER_EQUIPMENT) * to_kwh
    out["daily_heating_load_kwh"] = series(METER_HEATING) * to_kwh
    out["daily_cooling_load_kwh"] = series(METER_COOLING) * to_kwh
    return out


def _zone_key_from_output_column(column: str) -> str:
    """Extract the EnergyPlus zone name from an output CSV column."""
    key = column.split(":", 1)[0].strip().upper()
    suffix = " OFFICE PEOPLE"
    if key.endswith(suffix):
        key = key[: -len(suffix)].strip()
    return key


def summarize_daily_comfort(output_csv_path: Path, hours_per_day: int = 24) -> pd.DataFrame:
    """Daily occupant-weighted equivalent discomfort hours from hourly variables.

    Per hour: sum occupied people in zones where |PMV| exceeds PMV_DISCOMFORT_THRESHOLD,
    divide by total occupied people.
    Daily discomfort hours are the sum of hourly discomfort fractions.
    Returns columns: day_index, daily_discomfort_hours, daily_occupied_hours, daily_mean_pmv_occ.
    """
    if not output_csv_path.exists():
        raise RuntimeError(f"hourly output csv not found: {output_csv_path}")
    df = pd.read_csv(output_csv_path, encoding="utf-8-sig")
    pmv_cols = [c for c in df.columns if "fanger model pmv" in c.lower()]
    occ_cols = [c for c in df.columns if "occupant count" in c.lower()]
    if not pmv_cols or not occ_cols:
        raise RuntimeError(f"PMV/occupant columns missing in {output_csv_path.name} "
                           f"(pmv={len(pmv_cols)}, occ={len(occ_cols)})")
    pmv_by_zone = {_zone_key_from_output_column(c): c for c in pmv_cols}
    occ_by_zone = {_zone_key_from_output_column(c): c for c in occ_cols}
    common_zones = sorted(set(pmv_by_zone).intersection(occ_by_zone))
    if not common_zones:
        raise RuntimeError(f"No matched PMV/occupant zones in {output_csv_path.name} "
                           f"(pmv={len(pmv_cols)}, occ={len(occ_cols)})")

    pmv = df[[pmv_by_zone[z] for z in common_zones]].apply(pd.to_numeric, errors="coerce")
    occ = df[[occ_by_zone[z] for z in common_zones]].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    pmv_arr = pmv.to_numpy(dtype=float)
    occ_arr = occ.to_numpy(dtype=float)
    occupied_people = occ_arr.sum(axis=1)
    occupied = occupied_people > 0.0
    uncomfortable_people = (occ_arr * (np.abs(pmv_arr) > PMV_DISCOMFORT_THRESHOLD)).sum(axis=1)
    discomfort_fraction = np.divide(
        uncomfortable_people,
        occupied_people,
        out=np.zeros_like(occupied_people, dtype=float),
        where=occupied,
    )
    weighted_pmv = np.divide(
        (occ_arr * pmv_arr).sum(axis=1),
        occupied_people,
        out=np.full_like(occupied_people, np.nan, dtype=float),
        where=occupied,
    )
    n = len(df)
    day_index = np.arange(n) // hours_per_day + 1
    frame = pd.DataFrame({
        "day_index": day_index,
        "occupied": occupied.astype(int),
        "discomfort": discomfort_fraction,
        "pmv_occ": weighted_pmv,
    })
    daily = frame.groupby("day_index").agg(
        daily_occupied_hours=("occupied", "sum"),
        daily_discomfort_hours=("discomfort", "sum"),
        daily_mean_pmv_occ=("pmv_occ", "mean"),
    ).reset_index()
    return daily


def augment_daily_with_system_inputs(daily: pd.DataFrame, heating_cop: float, cooling_cop: float) -> pd.DataFrame:
    a = daily.copy()
    a["daily_heating_input_kwh"] = a["daily_heating_load_kwh"] / heating_cop
    a["daily_cooling_input_kwh"] = a["daily_cooling_load_kwh"] / cooling_cop
    a["daily_total_input_kwh"] = a["daily_electricity_kwh"] + a["daily_heating_input_kwh"] + a["daily_cooling_input_kwh"]
    return a


def summarize_annual_from_daily(daily: pd.DataFrame) -> dict[str, float]:
    out = {
        "annual_electricity_kwh": round(float(daily["daily_electricity_kwh"].sum()), 2),
        "annual_lights_kwh": round(float(daily["daily_lights_kwh"].sum()), 2),
        "annual_equipment_kwh": round(float(daily["daily_equipment_kwh"].sum()), 2),
        "annual_heating_load_kwh": round(float(daily["daily_heating_load_kwh"].sum()), 2),
        "annual_cooling_load_kwh": round(float(daily["daily_cooling_load_kwh"].sum()), 2),
        "annual_heating_input_kwh": round(float(daily["daily_heating_input_kwh"].sum()), 2),
        "annual_cooling_input_kwh": round(float(daily["daily_cooling_input_kwh"].sum()), 2),
        "annual_total_input_kwh": round(float(daily["daily_total_input_kwh"].sum()), 2),
    }
    if "daily_discomfort_hours" in daily.columns:
        out["annual_discomfort_hours"] = float(daily["daily_discomfort_hours"].sum())
    return out
