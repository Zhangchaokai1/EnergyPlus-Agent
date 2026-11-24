# -*- coding: utf-8 -*-
"""框架 A：LLM-only。同一 LLM 直接答题，无任何增强源（允许诚实拒答）。

用法：python run_llm_only.py [--reps 1] [--ids T1-P01,T3-X01] [--out ../results/raw/llm_only.jsonl]
断点续跑：输出文件中已有的 (题号, rep) 自动跳过。
"""
from __future__ import annotations

import argparse
from pathlib import Path

from common import (RESULTS_RAW, append_jsonl, build_system_prompt, chat_once,
                    load_done_keys, load_questions, now_ts, parse_answer)

A_AUGMENT = "你没有任何外部工具、资料或数据库可用，只能依靠自身知识与推理进行估算。"


def main() -> None:
    ap = argparse.ArgumentParser(description="QA benchmark runner A: LLM-only")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--ids", type=str, default="")
    ap.add_argument("--out", type=str, default=str(RESULTS_RAW / "llm_only.jsonl"))
    args = ap.parse_args()

    out = Path(args.out)
    questions = load_questions(args.ids.split(",") if args.ids else None)
    done = load_done_keys(out)
    system = build_system_prompt(A_AUGMENT)

    todo = [(q, rep) for rep in range(1, args.reps + 1) for q in questions if (q["id"], rep) not in done]
    print(f"A/LLM-only: {len(questions)} 题 × {args.reps} 次 | 已完成 {len(done)} | 待跑 {len(todo)}", flush=True)

    for n, (q, rep) in enumerate(todo, 1):
        resp = chat_once(system, q["question_zh"], temperature=0.0)
        parsed = parse_answer(resp["text"])
        append_jsonl(out, {
            "framework": "llm_only", "id": q["id"], "tier": q["tier"], "qtype": q["qtype"],
            "rep": rep, "ts": now_ts(), "model": resp["model"], "wall_s": resp["wall_s"],
            "usage": resp["usage"], "system_prompt": system, "question": q["question_zh"],
            "response_text": resp["text"], "parsed": parsed,
        })
        val = parsed["answer_value"]
        print(f"[{n}/{len(todo)}] {q['id']} rep{rep} -> {parsed['confidence']} "
              f"{'' if val is None else f'{val:,.0f} kWh'} ({resp['wall_s']}s)", flush=True)

    print(f"done -> {out}", flush=True)


if __name__ == "__main__":
    main()
