"""Workflow helpers for project-specific IDF, simulation and surrogate training.

The long-running steps are exposed as regular functions so the FastAPI backend can
run them in a background thread, while scripts can reuse them later.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Callable

from building_project import (
    BuildingProjectSpec,
    SURROGATE_LIBRARY_DIR,
    load_spec,
    project_dir,
    total_floor_area_m2,
)
from office_geometry import (
    choose_weather_file,
    geometry_from_project_spec,
    geometry_summary,
    parse_epw_header,
    render_idf_for_spec,
)


Progress = Callable[[str, int, str], None]


def _simulation_pct(completed: int, total: int) -> int:
    if total <= 0:
        return 15
    return 15 + round(50 * max(0, min(completed, total)) / total)


def _noop_progress(status: str, progress_pct: int, message: str) -> None:
    return None


def write_baseline_idf(building_id: str, spec: BuildingProjectSpec | None = None) -> Path:
    spec = spec or load_spec(building_id)
    weather, _ = choose_weather_file()
    meta = parse_epw_header(weather)
    out_dir = project_dir(building_id) / "idf"
    out_dir.mkdir(parents=True, exist_ok=True)
    idf_path = out_dir / "baseline.idf"
    idf_path.write_text(
        render_idf_for_spec(spec, meta, label=f"{building_id}_baseline"),
        encoding="utf-8",
    )
    return idf_path


def run_dataset_generation(building_id: str, progress: Progress = _noop_progress) -> Path:
    """Generate a project dataset by reusing the current office generator.

    V1 keeps the existing 10-D sampling logic and patches the confirmed building
    geometry into the IDF rendering path. The underlying script still runs
    EnergyPlus, so this can take minutes to hours depending on sample_count.
    """
    spec = load_spec(building_id)
    sample_count = int(spec.dataset_training.sample_count)
    seed = int(spec.dataset_training.seed)
    proj = project_dir(building_id)
    dataset_dir = proj / "datasets"
    work_dir = proj / "_tmp_runs"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    progress("simulating", 10, f"准备生成 {sample_count} 个 EnergyPlus 样本")

    cmd = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve().parent / "generate_office_dataset.py"),
        "--samples",
        str(sample_count),
        "--seed",
        str(seed),
        "--output-dir",
        str(dataset_dir),
        "--work-dir",
        str(work_dir),
        "--building-spec",
        str(proj / "building_spec.json"),
    ]
    progress("simulating", 15, f"启动 EnergyPlus 数据集生成脚本，样本 0/{sample_count}")
    log_lines: list[str] = []
    completed_seen = 0
    process = subprocess.Popen(
        cmd,
        cwd=str(Path(__file__).resolve().parent),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    for raw_line in process.stdout:
        log_lines.append(raw_line)
        line = raw_line.strip()
        if not line:
            continue

        target_match = re.search(
            r"Target samples:\s*(\d+)\s*\|\s*already ok:\s*(\d+)\s*\|\s*to run:\s*(\d+)",
            line,
            re.IGNORECASE,
        )
        if target_match:
            completed_seen = min(sample_count, int(target_match.group(2)))
            pending = int(target_match.group(3))
            progress(
                "simulating",
                _simulation_pct(completed_seen, sample_count),
                f"EnergyPlus 样本 {completed_seen}/{sample_count} 已完成，待运行 {pending}",
            )
            continue

        complete_match = re.search(r"completed\s+(\d+)\s*/\s*(\d+)\s+samples", line, re.IGNORECASE)
        if complete_match:
            rest_done = int(complete_match.group(1))
            rest_total = max(1, int(complete_match.group(2)))
            completed = min(sample_count, sample_count - rest_total + rest_done)
            completed_seen = max(completed_seen, completed)
            progress(
                "simulating",
                _simulation_pct(completed_seen, sample_count),
                f"EnergyPlus 样本 {completed_seen}/{sample_count} 已完成",
            )
            continue

        quality_match = re.search(r"samples ok:\s*(\d+)\s*\|\s*failed:\s*(\d+)", line, re.IGNORECASE)
        if quality_match:
            completed_seen = min(sample_count, int(quality_match.group(1)) + int(quality_match.group(2)))
            progress(
                "simulating",
                _simulation_pct(completed_seen, sample_count),
                f"EnergyPlus 样本 {completed_seen}/{sample_count} 已完成",
            )
            continue

        if line.startswith("Retrying"):
            progress("simulating", _simulation_pct(completed_seen, sample_count), line)

    returncode = process.wait()
    (proj / "dataset_generation.log").write_text(
        f"COMMAND: {' '.join(cmd)}\n\nOUTPUT:\n{''.join(log_lines)}",
        encoding="utf-8",
    )
    if returncode != 0:
        raise RuntimeError(f"dataset generation failed; see {proj / 'dataset_generation.log'}")
    progress("simulating", 65, f"EnergyPlus 样本 {sample_count}/{sample_count} 已完成，数据集生成完成")
    return dataset_dir


def run_surrogate_training(building_id: str, progress: Progress = _noop_progress) -> Path:
    spec = load_spec(building_id)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_id = f"surrogate_{timestamp}"
    model_dir = SURROGATE_LIBRARY_DIR / building_id / model_id
    dataset_dir = project_dir(building_id) / "datasets"
    model_dir.mkdir(parents=True, exist_ok=True)
    progress("training", 70, f"开始训练代理模型 {model_id}")

    cmd = [
        "python",
        str(Path(__file__).resolve().parent / "train_office_surrogate.py"),
        "--dataset-path",
        str(dataset_dir / "office_daily_dataset.csv"),
        "--summary-path",
        str(dataset_dir / "office_sample_summary.csv"),
        "--output-dir",
        str(model_dir),
        "--members",
        str(spec.dataset_training.train_ensemble_members),
    ]
    result = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parent), capture_output=True, text=True)
    (project_dir(building_id) / "surrogate_training.log").write_text(
        f"COMMAND: {' '.join(cmd)}\n\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}",
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeError(f"surrogate training failed; see {project_dir(building_id) / 'surrogate_training.log'}")

    manifest = build_model_manifest(building_id, model_id, model_dir, spec)
    (model_dir / "model_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    progress("registering_model", 95, "模型训练完成，正在注册模型")
    return model_dir


def build_model_manifest(
    building_id: str,
    model_id: str,
    model_dir: Path,
    spec: BuildingProjectSpec,
) -> dict:
    weather_file, _ = choose_weather_file()
    sample_summary = project_dir(building_id) / "datasets" / "office_sample_summary.csv"
    success_count = None
    if sample_summary.exists():
        try:
            import pandas as pd

            frame = pd.read_csv(sample_summary)
            success_count = int((frame["simulation_status"] == "ok").sum())
        except Exception:
            success_count = None
    return {
        "building_id": building_id,
        "model_id": model_id,
        "building_name": spec.building_identity.building_name,
        "status": "ready",
        "is_default": False,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_dir": str(model_dir),
        "geometry": {
            **geometry_summary(geometry_from_project_spec(spec)),
            "floor_area_m2": total_floor_area_m2(spec),
        },
        "dataset": {
            "sample_count": spec.dataset_training.sample_count,
            "weather_file": str(weather_file),
            "simulation_success_count": success_count,
        },
        "model": {
            "type": "CHU-DeepONet",
            "ensemble_members": spec.dataset_training.train_ensemble_members,
            "targets": [
                "daily_heating_load_kwh",
                "daily_cooling_load_kwh",
                "daily_discomfort_hours",
            ],
        },
        "spec": spec.to_dict(),
    }


def run_full_training(building_id: str, progress: Progress = _noop_progress) -> Path:
    spec = load_spec(building_id)
    progress("generating_idf", 5, "生成基准 IDF")
    write_baseline_idf(building_id, spec)
    run_dataset_generation(building_id, progress)
    model_dir = run_surrogate_training(building_id, progress)
    progress("completed", 100, "新代理模型已进入模型库")
    return model_dir


def copy_existing_dataset_for_smoke(building_id: str) -> None:
    """Developer helper for very fast local smoke tests.

    Not used by production endpoints. Kept as an explicit helper instead of
    silently short-circuiting the scientific workflow.
    """
    root = Path(__file__).resolve().parent
    dst = project_dir(building_id) / "datasets"
    dst.mkdir(parents=True, exist_ok=True)
    for name in ["office_daily_dataset.csv", "office_sample_summary.csv", "office_sample_plan.csv"]:
        src = root / "datasets" / name
        if src.exists():
            shutil.copy2(src, dst / name)
