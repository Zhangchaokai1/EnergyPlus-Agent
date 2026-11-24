# -*- coding: utf-8 -*-
"""框架 B：LLM-RAG。同一 LLM + 检索增强（训练分割 716 条仿真样本的 k=5 近邻 + 全局统计）。

检索查询直接使用题面的结构化参数（B 免于语言解析，这是给足 B 的优势，判读时如实披露）。
用法：python run_llm_rag.py [--reps 1] [--ids ...] [--k 5] [--out ../results/raw/llm_rag.jsonl]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import rag_corpus
from common import (RESULTS_RAW, append_jsonl, build_system_prompt, chat_once,
                    load_done_keys, load_questions, now_ts, parse_answer)

B_AUGMENT_TEMPLATE = """你可以使用下面的检索资料（除此之外没有其他工具）。资料检索自一个 EnergyPlus 仿真样本库：
与题目同一栋南京 19440 m² 高层办公原型在典型气象年(TMY)下参数化仿真得到的 {n} 条样本，
按 10 维设计参数的归一化欧氏距离取最近邻 k={k} 条（距离越小越相似，0=参数完全相同）。
表中能耗为全年输入电耗（kWh）；"年空调总电耗"=年供暖输入+年供冷输入。
你可以基于这些样本读表、插值、外推，并结合自身知识推理。

【检索资料】
{block}"""


def main() -> None:
    ap = argparse.ArgumentParser(description="QA benchmark runner B: LLM-RAG")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--ids", type=str, default="")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", type=str, default=str(RESULTS_RAW / "llm_rag.jsonl"))
    args = ap.parse_args()

    out = Path(args.out)
    questions = load_questions(args.ids.split(",") if args.ids else None)
    done = load_done_keys(out)
    n_corpus = rag_corpus._load()[1]["n_samples"]

    todo = [(q, rep) for rep in range(1, args.reps + 1) for q in questions if (q["id"], rep) not in done]
    print(f"B/LLM-RAG: {len(questions)} 题 × {args.reps} 次 | 语料 {n_corpus} 条 | "
          f"已完成 {len(done)} | 待跑 {len(todo)}", flush=True)

    for n, (q, rep) in enumerate(todo, 1):
        block = rag_corpus.format_retrieval_block(q, k=args.k)
        system = build_system_prompt(B_AUGMENT_TEMPLATE.format(n=n_corpus, k=args.k, block=block))
        resp = chat_once(system, q["question_zh"], temperature=0.0)
        parsed = parse_answer(resp["text"])
        append_jsonl(out, {
            "framework": "llm_rag", "id": q["id"], "tier": q["tier"], "qtype": q["qtype"],
            "rep": rep, "ts": now_ts(), "model": resp["model"], "wall_s": resp["wall_s"],
            "usage": resp["usage"], "k": args.k, "system_prompt": system,
            "question": q["question_zh"], "response_text": resp["text"], "parsed": parsed,
        })
        val = parsed["answer_value"]
        print(f"[{n}/{len(todo)}] {q['id']} rep{rep} -> {parsed['confidence']} "
              f"{'' if val is None else f'{val:,.0f} kWh'} ({resp['wall_s']}s)", flush=True)

    print(f"done -> {out}", flush=True)


if __name__ == "__main__":
    main()
