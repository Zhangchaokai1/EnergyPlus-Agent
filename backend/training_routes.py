"""Training workbench routes for project-specific surrogate models."""
from __future__ import annotations

import base64
import csv
import json
import mimetypes
import os
import re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, File, HTTPException, UploadFile
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

from backend.paths import ROOT  # noqa: E402
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from building_project import (  # noqa: E402
    DEFAULT_MODEL_ID,
    SURROGATE_LIBRARY_DIR,
    default_model_manifest,
    default_spec_dict,
    new_building_id,
    project_dir,
    save_spec,
    spec_from_dict,
    validate_spec,
)
from building_training_workflow import run_full_training, write_baseline_idf  # noqa: E402

router = APIRouter(tags=["training"])

JOBS: dict[str, dict[str, Any]] = {}
SELECTED_MODEL_PATH = ROOT / "models" / "building_surrogates" / "selected_model.json"


class ConfirmSpecRequest(BaseModel):
    spec: dict[str, Any] = Field(default_factory=dict)


class TrainingChatMessage(BaseModel):
    role: Literal["user", "assistant"] = "user"
    content: str


class TrainingChatRequest(BaseModel):
    messages: list[TrainingChatMessage] = Field(default_factory=list)
    spec: dict[str, Any] = Field(default_factory=dict)
    vision_analysis: dict[str, Any] | None = None


class StartTrainingRequest(BaseModel):
    building_id: str


class SelectModelRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    building_id: str
    model_id: str


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_env_file(Path(__file__).resolve().parent / ".env")


def _qwen_config() -> dict[str, str]:
    csv_path = ROOT / "千问API.csv"
    csv_conf: dict[str, str] = {}
    if csv_path.exists():
        rows = list(csv.DictReader(csv_path.open(encoding="utf-8-sig")))
        if rows:
            value_col = next((c for c in rows[0].keys() if c != "id"), None)
            if value_col:
                csv_conf = {row["id"]: row[value_col] for row in rows if row.get("id")}
    return {
        "base_url": os.environ.get("QWEN_VL_BASE_URL") or csv_conf.get("openAiCompatible") or "",
        "api_key": os.environ.get("QWEN_VL_API_KEY") or csv_conf.get("apiKey") or "",
        "model": os.environ.get("QWEN_VL_MODEL") or "qwen3-vl-235b-a22b-thinking",
    }


def _vision_prompt(user_note: str = "") -> str:
    return (
        "你正在为 EnergyPlus 参数化建模流程分析一张建筑照片/渲染图/BIM 截图。"
        "目标不是精确测绘，而是给后续对话提供一个合理的初始草案，所有关键尺寸最终都会由用户确认。"
        "\n\n请重点观察并在 JSON 中给出判断：\n"
        "1. 平面形状 shape：矩形(rectangle)还是 L 形(l_shape，两段体量拼接、有明显凹角或退台)？"
        "   若图片是沿街或鸟瞰视角能看到转角，优先判断是否为 L 形；无法判断时默认 rectangle。\n"
        "2. 若判断为 L 形，尝试估计缺角方位 notch_corner（ne/nw/se/sw，以主入口朝南、图中拍摄视角为参照的粗略猜测即可，"
        "不确定就填 null 并在 uncertainties 中说明）和缺角相对主体的长宽比例。\n"
        "3. 楼层数 floor_count_estimate：数窗排或数层线估计。\n"
        "4. 首层/裙房是否比标准层更高（常见于有大堂、商业裙房、架空层的办公楼，层高可能是标准层的 1.2~1.8 倍）；"
        "在 draft_spec.geometry 中分别给出 ground_floor_height_m（首层层高）与 typical_floor_height_m（标准层层高）的估计，"
        "而不是一个统一的层高。\n"
        "5. 屋顶类型 roof_type、开窗规律 window_pattern、窗墙比区间 wwr_range、立面材质等可见特征。\n\n"
        "不要编造精确尺寸或臆测你看不到的信息；无法判断的字段一律写 null，并在 uncertainties 列出需要用户明确确认的项。"
        "长度、宽度、朝向这类无法从单张图片可靠获得的尺寸，也应给出保守的默认量级并在 uncertainties 中说明这是粗估。"
        "\n\n输出 JSON，顶层字段包括：visible_description, building_type, floor_count_estimate, shape, "
        "notch_corner_guess, roof_type, window_pattern, wwr_range, uncertainties（字符串数组）, draft_spec。"
        "draft_spec 必须使用 building_identity / geometry / envelope / operation / dataset_training 五段结构，"
        "其中 geometry 段的字段名固定为：plan_shape, length_m, width_m, floor_count, ground_floor_height_m, "
        "typical_floor_height_m, orientation_deg, perimeter_depth_m, notch_corner, notch_length_m, notch_width_m"
        "（plan_shape=rectangle 时 notch_* 字段可省略或填 null）。"
        + (f"\n用户补充说明：{user_note}" if user_note else "")
    )


def _extract_json_object(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def _deepseek_training_config() -> dict[str, str]:
    from backend.agent.config import get_llm_config

    cfg = get_llm_config()
    if not cfg["api_key"]:
        raise HTTPException(status_code=400, detail="未配置 DeepSeek API Key：请在 backend/.env 设置 DEEPSEEK_API_KEY 或 LLM_API_KEY")
    return cfg


def _training_chat_prompt(spec: dict[str, Any], vision_analysis: dict[str, Any] | None) -> str:
    geometry = (spec or {}).get("geometry") or {}
    plan_shape = geometry.get("plan_shape", "rectangle")
    return (
        "# 角色\n"
        "你是「EnergyPlus-Agent 自然语言训练向导」，通过对话帮用户把一栋建筑变成结构化参数草案，"
        "供程序化 IDF 生成器、EnergyPlus 批量仿真和代理模型训练使用。\n\n"
        "# 硬性边界（不可违反）\n"
        "- 你只维护结构化参数草案（spec_patch），绝不自己生成、拼接或描述完整 IDF 文本——"
        "IDF 生成、仿真调度、模型训练全部由确定性程序完成，你不参与，也不要假装自己算出了能耗数字。\n"
        "- 支持的建筑范围：矩形(rectangle)或 L 形(l_shape)平屋顶多层办公建筑，每层按"
        "「东/南/西/北周边区 + 核心区」（L 形则是两段体量各自东南西北周边区+核心区）划分，"
        "HVAC 统一用 ZoneHVAC:IdealLoadsAirSystem。超出这个范围的建筑（坡屋顶、中庭、幕墙细部、复杂平面、"
        "真实空调系统等）要明确告知用户目前不支持，可作为近似简化处理。\n"
        "- 单张图片或用户口头描述都不能作为精确几何来源；长度、宽度、层数、首层层高、标准层层高、朝向、"
        "L 形的缺角位置与尺寸等关键参数，必须让用户明确确认或明确接受默认值，才能写入 parameter_sources 的 "
        "user_confirmed。\n\n"
        "# 关键字段说明（回答用户问题、生成 next_questions 时依据）\n"
        "- geometry.plan_shape: \"rectangle\" 或 \"l_shape\"。\n"
        "- geometry.length_m / width_m: 矩形时是建筑实际长宽；L 形时是「补齐缺角后的完整包络矩形」尺寸"
        "（不是两段体量分别的尺寸）。\n"
        "- geometry.ground_floor_height_m: 首层层高（大堂、门厅、裙房常见 4.2-6.0 m，比标准层高）。\n"
        "- geometry.typical_floor_height_m: 第 2 层及以上的标准层层高（办公楼常见 3.3-4.2 m）。"
        "这两个层高字段是分开的，不要把它们合并成一个统一层高去回填。\n"
        "- geometry.perimeter_depth_m: 周边区进深，不能超过建筑短边的一半，否则核心区宽度会小于等于 0。\n"
        "- geometry.notch_corner: 仅 l_shape 时有意义，取值 ne/nw/se/sw，表示从包络矩形的哪个角切掉一块"
        "矩形缺口来形成 L 形（如 ne=从东北角切掉）。\n"
        "- geometry.notch_length_m / notch_width_m: 仅 l_shape 时有意义，缺口沿 x/y 方向的尺寸。"
        "两者都必须大于 2×perimeter_depth_m，且分别小于 length_m-2×perimeter_depth_m 与 "
        "width_m-2×perimeter_depth_m，否则某一翼的进深会不够放下「周边区+核心区」——"
        "如果用户给的缺角尺寸不满足这个关系，明确指出冲突并给出一个满足条件的合理建议值，不要直接硬塞进 spec_patch。\n"
        "- 若用户此前一直在聊矩形，现在描述转角、退台、两段体量、L 形平面等特征，应主动确认是否要把 "
        "plan_shape 切换为 l_shape，并在切换时一并询问或给出缺角位置与尺寸的草案。\n\n"
        "# 对话风格\n"
        "- 回答自然、简洁、像真人对话，一次只聚焦 1-2 个最需要确认的问题，不要一次性列出十几个参数轰炸用户。\n"
        "- 不要把完整 JSON 或字段名堆给用户看；用中文人话解释你的假设和理由（例如\"裙房大堂层高一般比标准层高，"
        "我先假设首层 4.5 米、标准层 3.9 米，你可以直接告诉我实际值\"）。\n"
        "- 用户表达了修改或确认意图时，把改动放进 spec_patch；只是提问或闲聊时，spec_patch 返回空对象 {}。\n\n"
        "# 输出格式\n"
        "必须只输出一个 JSON 对象，不要有多余文字：\n"
        "{\"reply\":\"给用户看的中文回复\",\"spec_patch\":{},\"next_questions\":[],\"ready_to_confirm\":false}\n"
        "spec_patch 可包含 building_identity/geometry/envelope/operation/dataset_training/parameter_sources 六段，"
        "字段名必须与当前参数草案中的字段名完全一致（尤其是 ground_floor_height_m / typical_floor_height_m / "
        "notch_corner / notch_length_m / notch_width_m，不要发明新的字段名或沿用旧的 floor_height_m）。"
        "更新某个字段时，把对应 parameter_sources 里的 \"section.key\" 写成 \"user_confirmed\"。"
        "next_questions 是你认为还需要用户确认的关键问题列表（简短中文短语）。"
        "ready_to_confirm 只有在用户明确表示参数无误、可以确认生成 IDF 时才为 true，其余情况一律 false。\n\n"
        f"# 当前参数草案（plan_shape={plan_shape}）\n{json.dumps(spec, ensure_ascii=False)}\n"
        f"# 视觉识别结果（仅供参考，不可越过用户确认直接采用）\n{json.dumps(vision_analysis or {}, ensure_ascii=False)}"
    )


def _apply_spec_patch(spec: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    merged = json.loads(json.dumps(spec or default_spec_dict(), ensure_ascii=False))
    aliases = {
        "geometry": {
            "length": "length_m",
            "building_length": "length_m",
            "长度": "length_m",
            "width": "width_m",
            "building_width": "width_m",
            "宽度": "width_m",
            "floors": "floor_count",
            "floor_number": "floor_count",
            "stories": "floor_count",
            "层数": "floor_count",
            # 无歧义："首层"/"底层"/"ground floor" 一律指首层层高。
            "ground_floor_height": "ground_floor_height_m",
            "first_floor_height": "ground_floor_height_m",
            "首层层高": "ground_floor_height_m",
            "首层高度": "ground_floor_height_m",
            "底层层高": "ground_floor_height_m",
            "一层层高": "ground_floor_height_m",
            "大堂层高": "ground_floor_height_m",
            # 无歧义："标准层"/"typical floor" 一律指标准层层高。
            "typical_floor_height": "typical_floor_height_m",
            "standard_floor_height": "typical_floor_height_m",
            "标准层层高": "typical_floor_height_m",
            "标准层高度": "typical_floor_height_m",
            # 裸的"层高"/"floor_height"没有指明首层还是标准层：多数场景下用户是在说
            # 全楼统一层高或标准层层高，默认落到 typical_floor_height_m；LLM 侧的系统
            # 提示已要求它优先分别给出两个层高，这里只是兜底正则/前端快速指令的解释。
            "floor_height": "typical_floor_height_m",
            "story_height": "typical_floor_height_m",
            "storey_height": "typical_floor_height_m",
            "层高": "typical_floor_height_m",
            "楼层高度": "typical_floor_height_m",
            "每层高度": "typical_floor_height_m",
            "perimeter_depth": "perimeter_depth_m",
            "周边区深度": "perimeter_depth_m",
            "外围区深度": "perimeter_depth_m",
            "orientation": "orientation_deg",
            "朝向": "orientation_deg",
            "shape": "plan_shape",
            "平面形状": "plan_shape",
            "建筑形状": "plan_shape",
            "notch": "notch_corner",
            "notch_position": "notch_corner",
            "缺角位置": "notch_corner",
            "缺角方向": "notch_corner",
            "缺角长度": "notch_length_m",
            "缺角宽度": "notch_width_m",
        },
        "envelope": {
            "wwr": "window_wall_ratio",
            "window_to_wall_ratio": "window_wall_ratio",
            "窗墙比": "window_wall_ratio",
            "wall_insulation": "wall_insulation_m",
            "外墙保温": "wall_insulation_m",
            "roof_insulation": "roof_insulation_m",
            "屋面保温": "roof_insulation_m",
            "window_u": "window_u_w_m2k",
            "窗U值": "window_u_w_m2k",
            "infiltration": "infiltration_ach",
            "渗透率": "infiltration_ach",
        },
        "operation": {
            "cooling_setpoint": "cooling_setpoint_c",
            "供冷设定温度": "cooling_setpoint_c",
            "制冷设定温度": "cooling_setpoint_c",
            "heating_setpoint": "heating_setpoint_c",
            "供暖设定温度": "heating_setpoint_c",
            "采暖设定温度": "heating_setpoint_c",
        },
        "dataset_training": {
            "samples": "sample_count",
            "样本数": "sample_count",
            "ensemble_members": "train_ensemble_members",
            "集成成员数": "train_ensemble_members",
        },
    }
    for section, values in (patch or {}).items():
        if section == "parameter_sources" and isinstance(values, dict):
            merged.setdefault("parameter_sources", {}).update({str(k): str(v) for k, v in values.items()})
            continue
        if not isinstance(values, dict) or not isinstance(merged.get(section), dict):
            continue
        for key, value in values.items():
            key = aliases.get(section, {}).get(str(key), key)
            if key in merged[section] and value is not None:
                merged[section][key] = value
                merged.setdefault("parameter_sources", {})[f"{section}.{key}"] = "user_confirmed"
    return merged


@router.get("/api/training/template")
def training_template():
    return {"spec": default_spec_dict()}


@router.post("/api/training/chat")
def training_chat(request: TrainingChatRequest):
    cfg = _deepseek_training_config()
    client = OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])
    messages = [
        {"role": "system", "content": _training_chat_prompt(request.spec or default_spec_dict(), request.vision_analysis)},
        *[m.model_dump() for m in request.messages[-12:]],
    ]
    try:
        response = client.chat.completions.create(
            model=cfg["model"],
            messages=messages,
            temperature=0.25,
            max_tokens=1200,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"DeepSeek 训练向导调用失败：{exc}") from exc
    content = response.choices[0].message.content or ""
    parsed = _extract_json_object(content) or {}
    reply = str(parsed.get("reply") or content or "我需要再确认一些参数。")
    patch = parsed.get("spec_patch") if isinstance(parsed.get("spec_patch"), dict) else {}
    next_spec = _apply_spec_patch(request.spec or default_spec_dict(), patch)
    return {
        "reply": reply,
        "spec": next_spec,
        "spec_patch": patch,
        "next_questions": parsed.get("next_questions") if isinstance(parsed.get("next_questions"), list) else [],
        "ready_to_confirm": bool(parsed.get("ready_to_confirm")),
        "model": cfg["model"],
    }


@router.post("/api/training/analyze-image")
async def analyze_image(file: UploadFile = File(...), note: str = ""):
    cfg = _qwen_config()
    if not cfg["base_url"] or not cfg["api_key"]:
        raise HTTPException(status_code=400, detail="QWEN_VL_BASE_URL/QWEN_VL_API_KEY 未配置")
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="上传图片为空")
    mime = file.content_type or mimetypes.guess_type(file.filename or "")[0] or "image/png"
    data_url = f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"
    client = OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])
    try:
        response = client.chat.completions.create(
            model=cfg["model"],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _vision_prompt(note)},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            max_tokens=1400,
            temperature=0.1,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"视觉模型调用失败：{exc}") from exc
    content = response.choices[0].message.content or ""
    parsed = _extract_json_object(content)
    template = default_spec_dict()
    if parsed and isinstance(parsed.get("draft_spec"), dict):
        template = _merge_spec_defaults(template, parsed["draft_spec"])
    return {
        "model": cfg["model"],
        "raw_text": content,
        "analysis": parsed or {"visible_description": content},
        "draft_spec": template,
        "parameter_sources": {
            "vision": "vision_estimated",
            "defaults": "default_assumption",
        },
    }


def _merge_spec_defaults(base: dict[str, Any], draft: dict[str, Any]) -> dict[str, Any]:
    merged = json.loads(json.dumps(base, ensure_ascii=False))
    for section, values in draft.items():
        if isinstance(values, dict) and isinstance(merged.get(section), dict):
            for key, value in values.items():
                if value is not None and key in merged[section]:
                    merged[section][key] = value
                    merged.setdefault("parameter_sources", {})[f"{section}.{key}"] = "vision_estimated"
    return merged


def _mark_confirmed_sources(spec_dict: dict[str, Any]) -> dict[str, Any]:
    confirmed = json.loads(json.dumps(spec_dict, ensure_ascii=False))
    sources = dict(confirmed.get("parameter_sources") or {})
    for section, values in confirmed.items():
        if section == "parameter_sources" or not isinstance(values, dict):
            continue
        for key in values:
            sources[f"{section}.{key}"] = "user_confirmed"
    confirmed["parameter_sources"] = sources
    return confirmed


@router.post("/api/training/confirm-spec")
def confirm_spec(request: ConfirmSpecRequest):
    confirmed_dict = _mark_confirmed_sources(request.spec)
    spec = spec_from_dict(confirmed_dict)
    errors = validate_spec(spec)
    if errors:
        raise HTTPException(status_code=422, detail={"errors": errors})
    building_id = new_building_id(spec.building_identity.building_name)
    spec_path = save_spec(building_id, spec)
    idf_path = write_baseline_idf(building_id, spec)
    return {
        "building_id": building_id,
        "spec_path": str(spec_path),
        "baseline_idf": str(idf_path),
        "spec": spec.to_dict(),
    }


@router.post("/api/training/start")
def start_training(request: StartTrainingRequest):
    spec_path = project_dir(request.building_id) / "building_spec.json"
    if not spec_path.exists():
        raise HTTPException(status_code=404, detail=f"building_id 不存在：{request.building_id}")
    spec = spec_from_dict(json.loads(spec_path.read_text(encoding="utf-8")))
    sample_total = int(spec.dataset_training.sample_count)
    job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{request.building_id}"
    JOBS[job_id] = {
        "job_id": job_id,
        "building_id": request.building_id,
        "model_id": None,
        "status": "queued",
        "progress_pct": 0,
        "current_step": "等待后台任务启动",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "started_at": None,
        "finished_at": None,
        "elapsed_s": 0,
        "sample_total": sample_total,
        "sample_completed": 0,
        "error": None,
    }
    thread = threading.Thread(target=_run_job, args=(job_id,), daemon=True)
    thread.start()
    return JOBS[job_id]


def _run_job(job_id: str) -> None:
    job = JOBS[job_id]
    started = time.perf_counter()
    job["started_at"] = datetime.now().isoformat(timespec="seconds")

    def progress(status: str, pct: int, message: str) -> None:
        sample_match = re.search(r"样本\s*(\d+)\s*/\s*(\d+)", message)
        sample_update: dict[str, Any] = {}
        if sample_match:
            sample_update = {
                "sample_completed": int(sample_match.group(1)),
                "sample_total": int(sample_match.group(2)),
            }
        elif status in {"training", "registering_model", "completed"}:
            total = int(job.get("sample_total") or 0)
            if total:
                sample_update = {"sample_completed": total, "sample_total": total}
        job.update({
            "status": status,
            "progress_pct": max(0, min(100, int(pct))),
            "current_step": message,
            "elapsed_s": round(time.perf_counter() - started, 1),
            **sample_update,
        })

    try:
        model_dir = run_full_training(job["building_id"], progress)
        manifest_path = model_dir / "model_manifest.json"
        model_id = model_dir.name
        if manifest_path.exists():
            try:
                model_id = json.loads(manifest_path.read_text(encoding="utf-8")).get("model_id", model_id)
            except json.JSONDecodeError:
                pass
        job.update({
            "model_id": model_id,
            "status": "completed",
            "progress_pct": 100,
            "current_step": "训练完成",
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "elapsed_s": round(time.perf_counter() - started, 1),
        })
    except Exception as exc:
        job.update({
            "status": "failed",
            "error": str(exc),
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "elapsed_s": round(time.perf_counter() - started, 1),
        })


@router.get("/api/training/jobs/{job_id}")
def get_job(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="job_id 不存在")
    return JOBS[job_id]


@router.get("/api/models")
def list_models():
    models = [default_model_manifest()]
    for manifest_path in SURROGATE_LIBRARY_DIR.glob("*/*/model_manifest.json"):
        try:
            models.append(json.loads(manifest_path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    selected = _load_selected_model()
    return {"selected": selected, "models": models}


@router.post("/api/models/select")
def select_model(request: SelectModelRequest):
    available = list_models()["models"]
    match = next((m for m in available if m["building_id"] == request.building_id and m["model_id"] == request.model_id), None)
    if not match:
        raise HTTPException(status_code=404, detail="指定模型不存在")
    SELECTED_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    selected = {"building_id": request.building_id, "model_id": request.model_id}
    SELECTED_MODEL_PATH.write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        from backend import app as backend_app

        backend_app.clear_model_cache()
    except Exception:
        pass
    return {"selected": selected, "model": match}


def _load_selected_model() -> dict[str, str]:
    if not SELECTED_MODEL_PATH.exists():
        return {"building_id": "default_office", "model_id": DEFAULT_MODEL_ID}
    try:
        return json.loads(SELECTED_MODEL_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"building_id": "default_office", "model_id": DEFAULT_MODEL_ID}
