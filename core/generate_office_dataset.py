"""Generate the high-rise office daily heating/cooling/comfort dataset with EnergyPlus.

Forked from demo3/generate_daily_curve_dataset.py. Differences:
  - office geometry/template (office_geometry, office_template.idf.in),
  - 10-D Sobol design space incl. WWR + occupied setpoints (office_sampling),
  - energy AND comfort summarised per day: daily_heating_load / daily_cooling_load
    (from eplusmtr.csv) merged with daily_discomfort_hours / occupied_hours / mean_pmv
    (from eplusout.csv hourly Fanger PMV + occupant count),
  - three surrogate targets carried through: heating load, cooling load, discomfort hours.

Resumable (skip sample_ids already 'ok' in the summary) and process-parallel.
Run, e.g.:  python generate_office_dataset.py --samples 32 --workers 6
"""
from __future__ import annotations

import argparse
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from office_dataset_utils import (
    PMV_DISCOMFORT_THRESHOLD,
    augment_daily_with_system_inputs,
    build_daily_weather_features,
    summarize_annual_from_daily,
    summarize_daily_comfort,
    summarize_daily_simulation,
)
from office_geometry import (
    FLOOR_AREA_M2,
    TEMPLATE_PATH,
    build_zone_specs,
    choose_weather_file,
    geometry_from_project_spec,
    parse_epw_header,
    render_idf,
    run_energyplus,
)
from building_project import load_spec_path, total_floor_area_m2
from office_sampling import (
    SampleDefinition,
    estimate_retrofit_cost_cny,
    generate_samples,
)

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets"
WORK_DIR = ROOT / "_tmp_office_runs"
MASTER_SAMPLE_PLAN_PATH = DATASET_DIR / "office_sample_plan.csv"
MASTER_DAILY_DATASET_PATH = DATASET_DIR / "office_daily_dataset.csv"
MASTER_SAMPLE_SUMMARY_PATH = DATASET_DIR / "office_sample_summary.csv"
MASTER_SUMMARY_MD_PATH = DATASET_DIR / "office_dataset_summary.md"

# 10 design variables (model inputs), in the order to_record() emits them
DESIGN_VARS = [
    "wall_insulation_m", "roof_insulation_m", "window_u_w_m2k", "window_shgc",
    "infiltration_ach", "window_wall_ratio", "cooling_setpoint_c",
    "heating_setpoint_c", "cooling_cop", "heating_cop",
]
WEATHER_FEATURES = [
    "day_of_year_sin", "day_of_year_cos", "latitude", "longitude", "elevation_m",
    "drybulb_mean_c", "drybulb_min_c", "drybulb_max_c", "temp_range_c", "dewpoint_mean_c",
    "rh_mean_pct", "rh_min_pct", "rh_max_pct", "wind_mean_m_s", "wind_max_m_s",
    "ghi_kwh_m2", "dni_kwh_m2", "dhi_kwh_m2", "hdh16_c_h", "cdh26_c_h",
    "drybulb_mean_3d_c", "ghi_3d_kwh_m2", "wind_mean_3d_m_s",
]
DAILY_TARGETS = [
    "daily_electricity_kwh", "daily_lights_kwh", "daily_equipment_kwh",
    "daily_heating_load_kwh", "daily_cooling_load_kwh",
    "daily_heating_input_kwh", "daily_cooling_input_kwh", "daily_total_input_kwh",
    "daily_occupied_hours", "daily_discomfort_hours", "daily_mean_pmv_occ",
]
ANNUAL_COLS = [
    "annual_electricity_kwh", "annual_heating_load_kwh", "annual_cooling_load_kwh",
    "annual_heating_input_kwh", "annual_cooling_input_kwh", "annual_total_input_kwh",
    "annual_discomfort_hours", "annual_savings_vs_baseline_pct",
]
DAILY_COLUMNS = (
    ["sample_id", "sample_type", "simulation_status", "weather_file", "weather_mode",
     "day_index", "month", "day", "day_label", "location_name"]
    + DESIGN_VARS + WEATHER_FEATURES + DAILY_TARGETS + ANNUAL_COLS
    + ["estimated_retrofit_cost_cny"]
)
SUMMARY_COLUMNS = (
    ["sample_id", "sample_type", "simulation_status", "runtime_s"]
    + DESIGN_VARS
    + ["annual_electricity_kwh", "annual_lights_kwh", "annual_equipment_kwh",
       "annual_heating_load_kwh", "annual_cooling_load_kwh",
       "annual_heating_input_kwh", "annual_cooling_input_kwh", "annual_total_input_kwh",
       "annual_discomfort_hours", "annual_savings_vs_baseline_pct",
       "eui_equiv_kwh_m2", "estimated_retrofit_cost_cny",
       "weather_file", "weather_mode", "error_message"]
)


def reconstruct_sample(payload: dict[str, object]) -> SampleDefinition:
    return SampleDefinition(
        sample_id=int(payload["sample_id"]),
        sample_type=str(payload["sample_type"]),
        wall_insulation_m=float(payload["wall_insulation_m"]),
        roof_insulation_m=float(payload["roof_insulation_m"]),
        window_u=float(payload["window_u_w_m2k"]),
        window_shgc=float(payload["window_shgc"]),
        infiltration_ach=float(payload["infiltration_ach"]),
        window_wall_ratio=float(payload["window_wall_ratio"]),
        cooling_setpoint_c=float(payload["cooling_setpoint_c"]),
        heating_setpoint_c=float(payload["heating_setpoint_c"]),
        cooling_cop=float(payload["cooling_cop"]),
        heating_cop=float(payload["heating_cop"]),
    )


def simulate_office_sample(payload: dict[str, object]) -> dict[str, object]:
    sample = reconstruct_sample(payload)
    keep_artifacts = bool(payload["keep_artifacts"])
    weather_path = Path(str(payload["weather_path"]))
    weather_meta = dict(payload["weather_meta"])
    work_dir = Path(str(payload.get("work_dir") or WORK_DIR))

    idf_dir = work_dir / "idf"
    run_dir = work_dir / "runs" / f"sample_{sample.sample_id:05d}"
    idf_dir.mkdir(parents=True, exist_ok=True)
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    idf_path = idf_dir / f"sample_{sample.sample_id:05d}.idf"

    building_spec_path = payload.get("building_spec_path")
    if building_spec_path:
        spec = load_spec_path(str(building_spec_path))
        zones = build_zone_specs(geometry_from_project_spec(spec))
    else:
        zones = build_zone_specs()
    template_text = TEMPLATE_PATH.read_text(encoding="utf-8")
    idf_text = render_idf(template_text, weather_meta, sample.to_scenario(), zones)
    idf_path.write_text(idf_text, encoding="utf-8")

    started = time.perf_counter()
    try:
        run_energyplus(weather_path, idf_path, run_dir)
        daily = summarize_daily_simulation(run_dir / "eplusmtr.csv")
        comfort = summarize_daily_comfort(run_dir / "eplusout.csv")
        daily = daily.merge(comfort, on="day_index", how="left")
        daily = augment_daily_with_system_inputs(daily, sample.heating_cop, sample.cooling_cop)
        annual = summarize_annual_from_daily(daily)
        runtime_s = round(time.perf_counter() - started, 2)
        summary = {
            **sample.to_record(),
            "simulation_status": "ok",
            "runtime_s": runtime_s,
            **annual,
            "estimated_retrofit_cost_cny": estimate_retrofit_cost_cny(sample),
            "error_message": "",
        }
        daily_rows = daily.to_dict(orient="records")
        if not keep_artifacts:
            shutil.rmtree(run_dir, ignore_errors=True)
            idf_path.unlink(missing_ok=True)
        return {"summary": summary, "daily_rows": daily_rows}
    except Exception as exc:  # runtime safeguard: record failure, keep going
        runtime_s = round(time.perf_counter() - started, 2)
        if not keep_artifacts:  # clean up so failures don't accumulate on disk
            shutil.rmtree(run_dir, ignore_errors=True)
            idf_path.unlink(missing_ok=True)
        summary = {
            **sample.to_record(),
            "simulation_status": "failed",
            "runtime_s": runtime_s,
            "estimated_retrofit_cost_cny": estimate_retrofit_cost_cny(sample),
            "error_message": str(exc),
        }
        return {"summary": summary, "daily_rows": []}


def build_daily_frame(
    summary: dict[str, object],
    daily_rows: list[dict[str, object]],
    weather_features: pd.DataFrame,
    weather_file: Path,
    weather_mode: str,
    baseline_total_input_kwh: float,
) -> pd.DataFrame:
    daily = pd.DataFrame(daily_rows)
    wf = weather_features[["day_index", "month", "day", "day_label", "location_name", *WEATHER_FEATURES]]
    df = daily.merge(wf, on="day_index", how="left")

    annual_total = float(summary["annual_total_input_kwh"])
    savings_pct = round((baseline_total_input_kwh - annual_total) / baseline_total_input_kwh * 100.0, 2)

    df["sample_id"] = summary["sample_id"]
    df["sample_type"] = summary["sample_type"]
    df["simulation_status"] = summary["simulation_status"]
    df["weather_file"] = str(weather_file)
    df["weather_mode"] = weather_mode
    for key in DESIGN_VARS:
        df[key] = summary[key]
    for key in ["annual_electricity_kwh", "annual_heating_load_kwh", "annual_cooling_load_kwh",
                "annual_heating_input_kwh", "annual_cooling_input_kwh", "annual_total_input_kwh",
                "annual_discomfort_hours"]:
        df[key] = summary[key]
    df["annual_savings_vs_baseline_pct"] = savings_pct
    df["estimated_retrofit_cost_cny"] = summary["estimated_retrofit_cost_cny"]

    for col in DAILY_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df[DAILY_COLUMNS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the office daily heating/cooling/comfort dataset.")
    parser.add_argument("--samples", type=int, default=32, help="Target total design samples (incl. baseline). Default 32")
    parser.add_argument("--workers", type=int, default=min(6, max(2, (os.cpu_count() or 8) // 3)),
                        help="Parallel EnergyPlus workers")
    parser.add_argument("--seed", type=int, default=20260621, help="Sobol sampling seed")
    parser.add_argument("--keep-artifacts", action="store_true", help="Keep per-sample IDF/E+ outputs")
    parser.add_argument("--output-dir", type=Path, default=DATASET_DIR, help="Dataset output directory")
    parser.add_argument("--work-dir", type=Path, default=WORK_DIR, help="Temporary EnergyPlus work directory")
    parser.add_argument("--building-spec", type=Path, default=None, help="Confirmed building_spec.json for project-specific geometry")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global DATASET_DIR, WORK_DIR, MASTER_SAMPLE_PLAN_PATH, MASTER_DAILY_DATASET_PATH
    global MASTER_SAMPLE_SUMMARY_PATH, MASTER_SUMMARY_MD_PATH
    DATASET_DIR = args.output_dir
    WORK_DIR = args.work_dir
    MASTER_SAMPLE_PLAN_PATH = DATASET_DIR / "office_sample_plan.csv"
    MASTER_DAILY_DATASET_PATH = DATASET_DIR / "office_daily_dataset.csv"
    MASTER_SAMPLE_SUMMARY_PATH = DATASET_DIR / "office_sample_summary.csv"
    MASTER_SUMMARY_MD_PATH = DATASET_DIR / "office_dataset_summary.md"
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    weather_file, weather_mode = choose_weather_file()
    weather_meta = parse_epw_header(weather_file)
    weather_features = build_daily_weather_features(weather_file)

    project_spec = None
    building_id = None
    floor_area_m2 = FLOOR_AREA_M2
    if args.building_spec is not None:
        building_id = args.building_spec.parent.name
        project_spec = load_spec_path(args.building_spec)
        floor_area_m2 = total_floor_area_m2(project_spec)

    samples = generate_samples(total_samples=args.samples, seed=args.seed)
    pd.DataFrame([s.to_record() for s in samples]).to_csv(
        MASTER_SAMPLE_PLAN_PATH, index=False, encoding="utf-8-sig")

    # resume: keep ids already 'ok'
    if MASTER_SAMPLE_SUMMARY_PATH.exists():
        existing = pd.read_csv(MASTER_SAMPLE_SUMMARY_PATH, encoding="utf-8-sig")
        existing["sample_id"] = existing["sample_id"].astype(int)
        ok_ids = set(existing.loc[existing["simulation_status"] == "ok", "sample_id"].astype(int))
        max_id = int(existing["sample_id"].max()) if not existing.empty else 0
        if args.samples < max_id:
            raise RuntimeError(f"--samples={args.samples} < existing max sample_id {max_id}; raise --samples to append.")
    else:
        existing = pd.DataFrame(columns=SUMMARY_COLUMNS)
        ok_ids = set()

    pending = [s for s in samples if s.sample_id not in ok_ids]
    print(f"Weather: {weather_file.name} ({weather_mode}) | floor area {floor_area_m2:.0f} m2")
    print(f"Target samples: {args.samples} | already ok: {len(ok_ids)} | to run: {len(pending)} | workers: {args.workers}")

    payloads = [{**s.to_record(), "weather_path": str(weather_file),
                 "weather_meta": weather_meta, "keep_artifacts": args.keep_artifacts,
                 "work_dir": str(WORK_DIR),
                 "building_id": building_id, "building_spec_path": str(args.building_spec) if args.building_spec else ""}
                for s in pending]

    # baseline (sample_id=1) first, so we have the savings denominator
    if 1 in ok_ids:
        baseline_total_input = float(existing.loc[existing["sample_id"] == 1, "annual_total_input_kwh"].iloc[0])
        baseline_result = None
    else:
        baseline_payload = next(p for p in payloads if int(p["sample_id"]) == 1)
        baseline_result = simulate_office_sample(baseline_payload)
        if baseline_result["summary"]["simulation_status"] != "ok":
            raise RuntimeError(f"Baseline failed: {baseline_result['summary']['error_message']}")
        baseline_total_input = float(baseline_result["summary"]["annual_total_input_kwh"])

    header_needed = not MASTER_DAILY_DATASET_PATH.exists()

    def write_daily(summary, daily_rows):
        nonlocal header_needed
        if summary["simulation_status"] != "ok":
            return
        frame = build_daily_frame(summary, daily_rows, weather_features,
                                  weather_file, weather_mode, baseline_total_input)
        frame.to_csv(MASTER_DAILY_DATASET_PATH, mode="a", header=header_needed,
                     index=False, encoding="utf-8-sig")
        header_needed = False

    new_summaries: list[dict[str, object]] = []
    if baseline_result is not None:
        baseline_result["summary"]["weather_file"] = str(weather_file)
        baseline_result["summary"]["weather_mode"] = weather_mode
        write_daily(baseline_result["summary"], baseline_result["daily_rows"])
        new_summaries.append(baseline_result["summary"])

    rest = [p for p in payloads if int(p["sample_id"]) != 1]
    results: list[dict[str, object]] = []
    if rest:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(simulate_office_sample, p): int(p["sample_id"]) for p in rest}
            for i, fut in enumerate(as_completed(futures), start=1):
                results.append(fut.result())
                if i % 5 == 0 or i == len(futures):
                    print(f"  completed {i}/{len(futures)} samples")

        # retry failures once, serially (transient E+/IO issues)
        failed = [r for r in results if r["summary"]["simulation_status"] == "failed"]
        if failed:
            print(f"Retrying {len(failed)} failed samples serially ...")
            by_id = {int(p["sample_id"]): p for p in rest}
            rerun = {int(r["summary"]["sample_id"]): simulate_office_sample(by_id[int(r["summary"]["sample_id"])])
                     for r in failed}
            results = [rerun.get(int(r["summary"]["sample_id"]), r) for r in results]

        results.sort(key=lambda r: int(r["summary"]["sample_id"]))
        for r in results:
            r["summary"]["weather_file"] = str(weather_file)
            r["summary"]["weather_mode"] = weather_mode
            write_daily(r["summary"], r["daily_rows"])
            new_summaries.append(r["summary"])

    # merge summaries (existing + new), add EUI, persist
    summary_by_id = {int(row["sample_id"]): row for row in existing.to_dict(orient="records")}
    for s in new_summaries:
        s["eui_equiv_kwh_m2"] = (round(float(s["annual_total_input_kwh"]) / floor_area_m2, 2)
                                 if s.get("annual_total_input_kwh") is not None else None)
        s.setdefault("annual_savings_vs_baseline_pct",
                     round((baseline_total_input - float(s["annual_total_input_kwh"])) / baseline_total_input * 100.0, 2)
                     if s["simulation_status"] == "ok" else None)
        summary_by_id[int(s["sample_id"])] = s

    target_ids = {s.sample_id for s in samples}
    rows = [summary_by_id[i] for i in sorted(summary_by_id) if i in target_ids]
    summary_frame = pd.DataFrame(rows)
    for col in SUMMARY_COLUMNS:
        if col not in summary_frame.columns:
            summary_frame[col] = None
    summary_frame = summary_frame[SUMMARY_COLUMNS]
    summary_frame.to_csv(MASTER_SAMPLE_SUMMARY_PATH, index=False, encoding="utf-8-sig")

    ok = summary_frame[summary_frame["simulation_status"] == "ok"]
    n_ok = len(ok)
    n_failed = int((summary_frame["simulation_status"] == "failed").sum())
    if not args.keep_artifacts and n_failed == 0:
        shutil.rmtree(WORK_DIR, ignore_errors=True)

    # ---- quality-gate / reasonableness report ----
    print("\n=== DATASET QUALITY GATE ===")
    print(f"  samples ok: {n_ok} | failed: {n_failed}")
    if n_ok:
        eui = pd.to_numeric(ok["eui_equiv_kwh_m2"], errors="coerce")
        heat = pd.to_numeric(ok["annual_heating_load_kwh"], errors="coerce") / floor_area_m2
        cool = pd.to_numeric(ok["annual_cooling_load_kwh"], errors="coerce") / floor_area_m2
        disc = pd.to_numeric(ok["annual_discomfort_hours"], errors="coerce")
        print(f"  equiv EUI kWh/m2 : min {eui.min():.1f}  med {eui.median():.1f}  max {eui.max():.1f}")
        print(f"  heating load     : min {heat.min():.1f}  med {heat.median():.1f}  max {heat.max():.1f} kWh/m2")
        print(f"  cooling load     : min {cool.min():.1f}  med {cool.median():.1f}  max {cool.max():.1f} kWh/m2")
        print(f"  discomfort h/yr  : min {disc.min():.2f}  med {disc.median():.2f}  max {disc.max():.2f}")
        gate = []
        if (eui <= 0).any() or (eui > 400).any(): gate.append("EUI out of (0,400] for some sample")
        if (heat < 0).any() or (cool < 0).any(): gate.append("negative heating/cooling load")
        if eui.isna().any(): gate.append("NaN EUI present")
        print("  GATE: " + ("PASS" if not gate else "REVIEW -> " + "; ".join(gate)))

    write_summary_md(args.samples, weather_file, n_ok, n_failed, baseline_total_input, floor_area_m2)
    print(f"\nPlan:    {MASTER_SAMPLE_PLAN_PATH}")
    print(f"Daily:   {MASTER_DAILY_DATASET_PATH}")
    print(f"Summary: {MASTER_SAMPLE_SUMMARY_PATH}")
    print(f"Baseline equiv total: {baseline_total_input:.0f} kWh ({baseline_total_input/floor_area_m2:.1f} kWh/m2)")
    print(f"Newly simulated this run: {len(pending)} | elapsed {time.perf_counter()-started:.1f}s")


def write_summary_md(total_samples, weather_file, n_ok, n_failed, baseline_total, floor_area_m2=FLOOR_AREA_M2) -> None:
    lines = [
        "# 高层办公逐日数据集（供暖/供冷/舒适度）",
        "",
        f"- 目标样本数：`{total_samples}`（含基线 sample_id=1）",
        f"- 天气文件：`{weather_file.name}`",
        f"- 楼面积：`{floor_area_m2:.0f} m2`",
        f"- 成功 / 失败：`{n_ok}` / `{n_failed}`",
        f"- 基线年等效总输入：`{baseline_total:.0f} kWh`（`{baseline_total/floor_area_m2:.1f} kWh/m2`）",
        "",
        "## 数据组织",
        "- 长表 `office_daily_dataset.csv`：1 行 = 1 个设计样本在 1 天上的气象+设计变量+能耗+舒适度。",
        "- 代理模型输入：10 维设计变量 + 20 维逐日气象；目标：`daily_heating_load_kwh`、",
        f"  `daily_cooling_load_kwh`、`daily_discomfort_hours`（人员加权等效不舒适小时，基于占用热区 |PMV|>{PMV_DISCOMFORT_THRESHOLD:g}）。",
        "- 续跑：提高 `--samples` 重新运行本脚本，仅补新的 Sobol 样本。",
        "",
    ]
    MASTER_SUMMARY_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
