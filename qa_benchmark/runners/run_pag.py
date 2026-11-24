# -*- coding: utf-8 -*-
"""框架 C：PAG。headless 驱动 EnergyPlus-Agent 的部署编排器（同一套工具/门控/回退代码），
仅按公平性协议把 temperature 固定为 0、system prompt 换成共享骨架 + PAG 增强源段。

E+ 回退缓存：按 (完整设计向量, 气候情景) 哈希缓存仿真结果，跨重复复用；
首跑耗时记录在 first_runtime_s（Table 7 延迟列采用首跑实测值）。

用法：python run_pag.py [--reps 1] [--ids ...] [--out ../results/raw/pag.jsonl]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from common import (CACHE_DIR, RESULTS_RAW, append_jsonl, build_system_prompt,
                    load_done_keys, load_questions, now_ts, parse_answer)

# 触发 backend 包导入（common 已把 EnergyPlus-Agent 加进 sys.path；模型在首次工具调用时惰性加载）
from backend.agent import eplus_fallback, orchestrator  # noqa: E402

C_AUGMENT = """你是"物理增强生成(PAG)"智能体：数值不靠心算，而是调用背后的物理引擎获得。
物理引擎与对象：用 EnergyPlus 仿真数据训练的 HC-DeepONet 5 成员深度集成代理模型（输出预测均值与
集成分歧 σ_E——模型对自己"知不知道"的自认证信号），以及高保真 EnergyPlus 参数化仿真器；
对象为南京一栋 19440 m² 高层办公原型（与题目建筑同一原型）。

工具使用规则：
- 一切数值必须来自工具调用，禁止心算或编造。
- predict_energy：传入题面给出的全部 10 维参数（毫秒级），返回年供暖/供冷输入电耗与【不确定性门控】判定；
  题目描述异常气候年份时，用 climate_scenario 选统计特征最接近的情景（anom_warm/anom_cold/anom_balanced）。
- 门控判定为 "fallback"（σ_E 超阈值，疑似外推/OOD）时：必须调用 run_energyplus_simulation
  （参数与 climate_scenario 与 predict_energy 保持一致，约 1–3 分钟）做真实仿真，并以仿真结果为准作答。
- 门控判定为 "trust" 时：禁止再调用 run_energyplus_simulation，必须直接采用 predict_energy 的代理结果作答；
  只有 gating.verdict == "fallback" 或 predict_energy 工具失败时才允许回退仿真。
- 方案比较/节能量：对改动前后（或方案 A/B）分别调用 predict_energy 再相减；若某一侧门控为 fallback，
  先对该侧调用 run_energyplus_simulation 再相减。operation_savings 与 estimate_extreme 只适用于
  "基准办公楼"，本任务中不要使用；optimize_pareto 与 explain_features 与本任务无关，不要使用。
- 最终数值来自"门控通过的代理预测"或"EnergyPlus 仿真"时 confidence 用 confident；
  若工具失败且无法获得可靠数值，诚实选择 cannot_answer。"""

PAG_SYSTEM_PROMPT = build_system_prompt(C_AUGMENT)

# ---------------- E+ 回退缓存（monkeypatch，不动生产代码） ----------------

CACHE_PATH = CACHE_DIR / "eplus_fallback_cache.json"
_orig_run_fallback = eplus_fallback.run_fallback
_cache: dict[str, dict] = {}
if CACHE_PATH.exists():
    _cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))


def _design_key(design: dict | None) -> str:
    from backend import app
    d = design or {}
    full = {k: round(float(d.get(k, app.BASE_DESIGN[k]) or app.BASE_DESIGN[k]), 6)
            for k in app.STATIC_COLUMNS}
    scenario = str(d.get("climate_scenario") or "base")
    return hashlib.md5(json.dumps([full, scenario], sort_keys=True).encode()).hexdigest()


def _cached_run_fallback(design: dict | None) -> dict:
    key = _design_key(design)
    hit = _cache.get(key)
    if hit is not None:
        result = dict(hit["result"])
        result["cached"] = True
        result["firstRuntimeSeconds"] = hit["first_runtime_s"]
        return result
    result = _orig_run_fallback(design)
    if isinstance(result, dict) and "error" not in result:
        slim = {k: v for k, v in result.items() if k != "chart"}
        _cache[key] = {"result": slim, "first_runtime_s": result.get("runtimeSeconds"), "ts": now_ts()}
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(json.dumps(_cache, ensure_ascii=False, indent=1), encoding="utf-8")
    return result


eplus_fallback.run_fallback = _cached_run_fallback

# ---------------- headless 驱动 ----------------


def run_question(q: dict) -> dict:
    """驱动一次编排器完整回合，收集最终文本 + 工具轨迹 + 门控诊断。"""
    text_parts: list[str] = []
    trace: list[dict] = []
    t0 = time.time()
    for ev in orchestrator.run_agent_stream(
        [{"role": "user", "content": q["question_zh"]}],
        system_prompt=PAG_SYSTEM_PROMPT,
        temperature=0.0,
    ):
        et = ev.get("type")
        if et == "delta":
            text_parts.append(ev["content"])
        elif et == "tool_call":
            trace.append({"event": "tool_call", "name": ev["name"], "arguments": ev["arguments"]})
        elif et == "tool_result":
            result = ev["result"]
            if isinstance(result, dict):
                result = {k: v for k, v in result.items() if k != "chart"}
            trace.append({"event": "tool_result", "name": ev["name"], "result": result})
        elif et == "progress":
            print(f"      … EnergyPlus 回退仿真进行中 {ev['elapsedS']}s", flush=True)
    wall = time.time() - t0

    gate_verdicts, fallback_calls = [], []
    for e in trace:
        if e["event"] != "tool_result" or not isinstance(e.get("result"), dict):
            continue
        if e["name"] == "predict_energy" and "gating" in e["result"]:
            g = e["result"]["gating"]
            gate_verdicts.append({"verdict": g.get("verdict"), "sigma": g.get("sigmaEnergyTotalKwh"),
                                  "ratio": g.get("ratio")})
        if e["name"] == "run_energyplus_simulation":
            r = e["result"]
            fallback_calls.append({"cached": bool(r.get("cached")),
                                   "runtime_s": r.get("firstRuntimeSeconds") or r.get("runtimeSeconds"),
                                   "scenario": r.get("climateScenario"), "error": r.get("error")})

    return {
        "response_text": "".join(text_parts),
        "trace": trace,
        "wall_s": round(wall, 2),
        "gate_verdicts": gate_verdicts,
        "fallback_used": bool(fallback_calls),
        "fallback_calls": fallback_calls,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="QA benchmark runner C: PAG (headless agent)")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--ids", type=str, default="")
    ap.add_argument("--out", type=str, default=str(RESULTS_RAW / "pag.jsonl"))
    ap.add_argument("--max-retries", type=int, default=3)
    args = ap.parse_args()

    out = Path(args.out)
    questions = load_questions(args.ids.split(",") if args.ids else None)
    done = load_done_keys(out)
    todo = [(q, rep) for rep in range(1, args.reps + 1) for q in questions if (q["id"], rep) not in done]
    print(f"C/PAG: {len(questions)} 题 × {args.reps} 次 | 已完成 {len(done)} | 待跑 {len(todo)} | "
          f"E+ 缓存 {len(_cache)} 条", flush=True)

    for n, (q, rep) in enumerate(todo, 1):
        last_exc = None
        for attempt in range(args.max_retries):
            try:
                r = run_question(q)
                break
            except Exception as exc:  # noqa: BLE001 API 抖动整题重试（E+ 结果已缓存，重试廉价）
                last_exc = exc
                wait = min(2 ** attempt * 5, 60)
                print(f"    整题失败（{type(exc).__name__}: {exc}），{wait}s 后重试", flush=True)
                time.sleep(wait)
        else:
            raise RuntimeError(f"{q['id']} rep{rep} 连续失败：{last_exc}")

        parsed = parse_answer(r["response_text"])
        append_jsonl(out, {
            "framework": "pag", "id": q["id"], "tier": q["tier"], "qtype": q["qtype"],
            "rep": rep, "ts": now_ts(), "wall_s": r["wall_s"],
            "system_prompt": PAG_SYSTEM_PROMPT, "question": q["question_zh"],
            "expected_gate": q.get("expected_gate"),
            "gate_verdicts": r["gate_verdicts"], "fallback_used": r["fallback_used"],
            "fallback_calls": r["fallback_calls"], "trace": r["trace"],
            "response_text": r["response_text"], "parsed": parsed,
        })
        val = parsed["answer_value"]
        gates = "/".join(g["verdict"] or "?" for g in r["gate_verdicts"]) or "-"
        print(f"[{n}/{len(todo)}] {q['id']} rep{rep} -> {parsed['confidence']} "
              f"{'' if val is None else f'{val:,.0f} kWh'} | 门控 {gates} | "
              f"回退 {'是' if r['fallback_used'] else '否'} ({r['wall_s']}s)", flush=True)

    print(f"done -> {out}", flush=True)


if __name__ == "__main__":
    main()
