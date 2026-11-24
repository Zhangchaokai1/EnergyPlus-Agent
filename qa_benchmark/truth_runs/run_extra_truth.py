# -*- coding: utf-8 -*-
"""补跑基准题集所需的 EnergyPlus 真值（任务清单 extra_jobs.csv，由 make_questions.py 生成）。

- 走与数据集完全相同的仿真管线（generate_office_dataset.simulate_office_sample，南京 TMY）。
- 增量：extra_truth.csv 中已 ok 的 job 自动跳过，可断点续跑。
- 跑完后重跑 ../questions/make_questions.py 即可把真值绑定进 questions_v1.yaml。

用法：python run_extra_truth.py [--workers 5]
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

JOBS_CSV = HERE / "extra_jobs.csv"
TRUTH_CSV = HERE / "extra_truth.csv"

PARAM_COLS = [
    "wall_insulation_m", "roof_insulation_m", "window_u_w_m2k", "window_shgc",
    "infiltration_ach", "window_wall_ratio", "cooling_setpoint_c", "heating_setpoint_c",
    "cooling_cop", "heating_cop",
]

SUMMARY_FIELDS = [
    "runtime_s", "annual_heating_input_kwh", "annual_cooling_input_kwh",
    "annual_heating_load_kwh", "annual_cooling_load_kwh", "annual_total_input_kwh",
    "annual_discomfort_hours",
]


def main() -> None:
    ap = argparse.ArgumentParser(description="Run extra EnergyPlus truths for the QA benchmark.")
    ap.add_argument("--workers", type=int, default=5)
    args = ap.parse_args()

    import generate_office_dataset as gen  # noqa: PLC0415 惰性导入（含 E+ 路径检查）
    from office_geometry import choose_weather_file, parse_epw_header  # noqa: PLC0415

    wf, _mode = choose_weather_file()
    meta = parse_epw_header(wf)

    jobs = pd.read_csv(JOBS_CSV, encoding="utf-8-sig")
    old, done = None, set()
    if TRUTH_CSV.exists():
        old = pd.read_csv(TRUTH_CSV, encoding="utf-8-sig")
        old = old[old["simulation_status"] == "ok"]
        done = set(old["job_id"])
    todo = jobs[~jobs["job_id"].isin(done)]
    print(f"jobs: {len(jobs)} total | cached ok: {len(done)} | to run: {len(todo)}", flush=True)
    if todo.empty:
        return

    payloads = []
    for pos, (_, r) in enumerate(todo.iterrows()):
        payloads.append((str(r["job_id"]), {
            **{c: float(r[c]) for c in PARAM_COLS},
            "sample_id": 400001 + pos,
            "sample_type": "qa_extra",
            "weather_path": str(wf),
            "weather_meta": meta,
            "keep_artifacts": False,
        }))

    rows, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(gen.simulate_office_sample, p): jid for jid, p in payloads}
        n = 0
        for fut in as_completed(futs):
            jid = futs[fut]
            try:
                s = fut.result()["summary"]
                row = {"job_id": jid, "simulation_status": s.get("simulation_status")}
                row.update({k: s.get(k) for k in SUMMARY_FIELDS})
            except Exception as exc:  # noqa: BLE001 记录失败原因，主流程不中断
                row = {"job_id": jid, "simulation_status": f"error: {exc}"}
            rows.append(row)
            n += 1
            print(f"[{n}/{len(payloads)}] {jid} -> {row['simulation_status']} "
                  f"({row.get('runtime_s', '?')} s)", flush=True)

    frame = pd.DataFrame(rows)
    if old is not None and not old.empty:
        frame = pd.concat([old, frame], ignore_index=True)
    frame = frame.sort_values("job_id").reset_index(drop=True)
    frame.to_csv(TRUTH_CSV, index=False, encoding="utf-8-sig")
    ok = int((frame["simulation_status"] == "ok").sum())
    print(f"done in {time.time() - t0:.0f} s | ok {ok}/{len(jobs)} -> {TRUTH_CSV}", flush=True)


if __name__ == "__main__":
    main()
