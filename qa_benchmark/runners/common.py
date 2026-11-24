# -*- coding: utf-8 -*-
"""问答基准公共设施：路径引导、LLM 客户端、共享 prompt 骨架、作答协议解析、JSONL 落盘。

公平性协议（../基准实验规划.md §1.2）：三框架同一 LLM(deepseek-chat)、temperature=0、
逐字相同的题面、相同的 JSON 作答协议；仅 system prompt 的"增强源"段不同。
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent            # qa_benchmark/runners
QA_ROOT = HERE.parent                              # qa_benchmark
REPO_ROOT = QA_ROOT.parent                         # 仓库根
AGENT_ROOT = REPO_ROOT / "EnergyPlus-Agent"        # backend 包所在

for _p in (str(REPO_ROOT), str(AGENT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

QUESTIONS_YAML = QA_ROOT / "questions" / "questions_v1.yaml"
CORPUS_DIR = QA_ROOT / "corpus"
RESULTS_RAW = QA_ROOT / "results" / "raw"
RESULTS_SCORED = QA_ROOT / "results" / "scored"
CACHE_DIR = QA_ROOT / "results" / "cache"
PILOT_DIR = QA_ROOT / "pilot"

PARAM_COLS = [
    "wall_insulation_m", "roof_insulation_m", "window_u_w_m2k", "window_shgc",
    "infiltration_ach", "window_wall_ratio", "cooling_setpoint_c", "heating_setpoint_c",
    "cooling_cop", "heating_cop",
]

# 展示用中文短标签（B 检索表表头等）
PARAM_LABELS = {
    "wall_insulation_m": "外墙保温 m",
    "roof_insulation_m": "屋面保温 m",
    "window_u_w_m2k": "窗U W/(m²K)",
    "window_shgc": "SHGC",
    "infiltration_ach": "渗透 ACH",
    "window_wall_ratio": "窗墙比",
    "cooling_setpoint_c": "冷设定 °C",
    "heating_setpoint_c": "热设定 °C",
    "cooling_cop": "冷COP",
    "heating_cop": "热COP",
}


def load_questions(ids: list[str] | None = None) -> list[dict]:
    import yaml
    qs = yaml.safe_load(QUESTIONS_YAML.read_text(encoding="utf-8"))
    if ids:
        idset = {i.strip() for i in ids if i.strip()}
        qs = [q for q in qs if q["id"] in idset]
        missing = idset - {q["id"] for q in qs}
        if missing:
            raise SystemExit(f"题号不存在：{sorted(missing)}")
    return qs


# ---------------- LLM 客户端（复用 agent 的 provider 配置） ----------------

@lru_cache(maxsize=1)
def llm_client():
    from backend.agent.config import get_llm_config, get_llm_request_options  # 读 EnergyPlus-Agent/backend/.env
    cfg = get_llm_config()
    if not cfg["api_key"]:
        raise RuntimeError("未配置 API Key：请检查 EnergyPlus-Agent/backend/.env")
    from openai import OpenAI
    return OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"]), cfg["model"], get_llm_request_options()


def chat_once(system: str, user: str, temperature: float = 0.0, max_retries: int = 5) -> dict:
    """一次非流式对话（无工具，框架 A/B 用）。指数退避重试，返回 {text, wall_s, usage, model}。"""
    client, model, request_options = llm_client()
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            t0 = time.time()
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=temperature,
                stream=False,
                **request_options,
            )
            wall = time.time() - t0
            usage = getattr(resp, "usage", None)
            return {
                "text": resp.choices[0].message.content or "",
                "wall_s": round(wall, 2),
                "model": model,
                "usage": ({"prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens}
                          if usage else None),
            }
        except Exception as exc:  # noqa: BLE001 限速/网络抖动重试
            last_exc = exc
            wait = min(2 ** attempt * 2, 60)
            print(f"    LLM 调用失败（{type(exc).__name__}），{wait}s 后重试 {attempt + 1}/{max_retries}", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"LLM 调用连续失败：{last_exc}")


# ---------------- 共享 prompt 骨架与作答协议 ----------------

ANSWER_PROTOCOL = """【作答要求】
1. 可以先简要展示你的分析，但回答的最后必须单独一行输出一个 JSON 对象（不要放进代码块），格式：
   {"answer_value": <数值或 null>, "unit": "kWh", "confidence": "confident|uncertain|cannot_answer", "basis": "一句话依据"}
2. answer_value 用阿拉伯数字，单位一律 kWh（不要千分位逗号、不要"万"等中文数量词）；无法给出数值时填 null。
3. confidence 三选一：confident=有可靠依据；uncertain=给出估计但把握不足；cannot_answer=无法可靠回答。
   若没有可靠依据，请诚实选择 uncertain 或 cannot_answer——诚实优于编造。
4. 方案比较题（问"哪个方案更低/更省电"）：JSON 中额外加 "choice": "A" 或 "B"（全年空调总电耗更低的一方），
   answer_value 填两方案全年空调总电耗差值的绝对值（kWh）。"""

SKELETON = """你是一名建筑能耗领域的专业助手，回答用户关于某栋办公建筑全年空调能耗的问题。
口径说明：题目所问"空调总电耗"指全年供冷输入电耗与供暖输入电耗之和（HVAC 用电，不含照明与插座设备用电）。

{augmentation}

{protocol}"""


def build_system_prompt(augmentation: str) -> str:
    prompt = SKELETON.format(augmentation=augmentation.strip(), protocol=ANSWER_PROTOCOL)
    suffix = os.environ.get("LLM_SYSTEM_SUFFIX", "").strip()
    return f"{prompt}\n\n{suffix}" if suffix else prompt


# ---------------- 作答协议解析 ----------------

_JSON_RE = re.compile(r"\{[^{}]*\}", re.S)
_NUM_RE = re.compile(r"-?[\d,]+(?:\.\d+)?")

_UNIT_FACTOR = [
    ("万kwh", 1e4), ("万千瓦时", 1e4), ("mwh", 1e3), ("gwh", 1e6),
    ("kwh", 1.0), ("千瓦时", 1.0), ("度", 1.0),
]


def _coerce_value(val, unit) -> tuple[float | None, str | None]:
    """把 answer_value/unit 归一到 kWh 浮点数。返回 (数值, 解析警告)。"""
    if val is None:
        return None, None
    factor = 1.0
    unit_s = str(unit or "").strip().lower().replace(" ", "")
    for token, f in _UNIT_FACTOR:
        if token in unit_s:
            factor = f
            break
    if isinstance(val, (int, float)):
        return float(val) * factor, None
    s = str(val).strip().replace(",", "").replace("，", "")
    mul = 1.0
    if "亿" in s:
        mul, s = 1e8, s.replace("亿", "")
    elif "万" in s:
        mul, s = 1e4, s.replace("万", "")
    m = _NUM_RE.search(s)
    if not m:
        return None, f"无法解析数值：{val!r}"
    return float(m.group()) * mul * factor, None


def parse_answer(text: str) -> dict:
    """从回答文本提取协议 JSON（从后往前找第一个含 answer_value/confidence 键的对象）。"""
    out = {"answer_value": None, "unit": None, "confidence": None, "basis": None,
           "choice": None, "parse_ok": False, "parse_error": None}
    if not text or not text.strip():
        out["parse_error"] = "空回答"
        return out
    obj = None
    for cand in reversed(_JSON_RE.findall(text)):
        for attempt in (cand, cand.replace("“", '"').replace("”", '"').replace("'", '"')):
            try:
                o = json.loads(attempt)
            except json.JSONDecodeError:
                continue
            if isinstance(o, dict) and ("answer_value" in o or "confidence" in o):
                obj = o
                break
        if obj is not None:
            break
    if obj is None:
        out["parse_error"] = "未找到协议 JSON"
        return out

    value, warn = _coerce_value(obj.get("answer_value"), obj.get("unit"))
    conf = str(obj.get("confidence") or "").strip().lower()
    if conf not in {"confident", "uncertain", "cannot_answer"}:
        out["parse_error"] = f"confidence 非法：{obj.get('confidence')!r}"
    choice = str(obj.get("choice") or "").strip().upper()
    choice = choice.replace("方案", "").strip() or None
    if choice not in {"A", "B", None}:
        choice = None

    out.update({
        "answer_value": value,
        "unit": str(obj.get("unit") or ""),
        "confidence": conf or None,
        "basis": str(obj.get("basis") or "")[:300],
        "choice": choice,
        "parse_ok": out["parse_error"] is None and (warn is None),
        "parse_error": out["parse_error"] or warn,
    })
    return out


# ---------------- JSONL 落盘与断点续跑 ----------------

def append_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_done_keys(path: Path) -> set[tuple[str, int]]:
    """已完成 (题号, 重复序号)，供断点续跑跳过。"""
    done = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                done.add((r["id"], int(r["rep"])))
            except Exception:  # noqa: BLE001 半行损坏时忽略
                continue
    return done


def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")
