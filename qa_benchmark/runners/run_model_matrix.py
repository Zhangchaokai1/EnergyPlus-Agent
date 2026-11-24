# -*- coding: utf-8 -*-
"""Run QA benchmark model sweeps without changing EnergyPlus-Agent .env.

This wrapper reads the local API credential files, injects LLM_* environment
variables into isolated runner subprocesses, and writes outputs under:

  results/model_sweep/raw/{model_alias}/{framework}.jsonl

Secrets are never printed or written to JSONL.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent
QA_ROOT = HERE.parent
REPO_ROOT = QA_ROOT.parent
MODEL_SWEEP_ROOT = QA_ROOT / "results" / "model_sweep"
RAW_ROOT = MODEL_SWEEP_ROOT / "raw"

PILOT_IDS = "T1-P01,T1-C01,T1-S01,T2-P05,T3-P01,T3-X01"
FRAMEWORK_SCRIPTS = {
    "llm_only": HERE / "run_llm_only.py",
    "llm_rag": HERE / "run_llm_rag.py",
    "pag": HERE / "run_pag.py",
}


def _read_qwen_csv() -> dict[str, str]:
    path = REPO_ROOT / "千问API.csv"
    rows = list(csv.reader(path.open(encoding="utf-8-sig")))
    return {r[0].strip(): r[1].strip() for r in rows if len(r) >= 2}


def _read_proxy_sections() -> dict[str, dict]:
    path = REPO_ROOT / "代理平台API.txt"
    text = path.read_text(encoding="utf-8-sig")
    sections: dict[str, dict] = {}
    for block in re.split(r"\n\s*\n", text):
        label_m = re.search(r"^\s*([^:\r\n]+):", block)
        json_m = re.search(r"\{.*\}", block, re.S)
        if not label_m or not json_m:
            continue
        sections[label_m.group(1).strip()] = json.loads(json_m.group(0))
    return sections


def load_model_specs() -> dict[str, dict[str, str]]:
    """Return the fixed model sweep set, including secrets kept in memory only."""
    qwen = _read_qwen_csv()
    proxy = _read_proxy_sections()

    def proxy_base(label: str) -> str:
        url = proxy[label]["url"].rstrip("/")
        return url if url.endswith("/v1") else f"{url}/v1"

    return {
        "qwen3_7_plus": {
            "provider_label": "qwen_csv",
            "model": "qwen3.7-plus",
            "base_url": qwen["openAiCompatible"].rstrip("/"),
            "api_key": qwen["apiKey"],
            "extra_body": {"enable_thinking": False},
            "max_tokens": "4096",
            "timeout_s": "120",
            "system_suffix": (
                "【模型输出约束】本轮只允许输出最终 JSON 对象，不要输出分析、推导、Markdown 或代码块。"
                "JSON 必须可被 json.loads 直接解析；若无法可靠作答，answer_value 填 null 且 confidence 填 cannot_answer。"
                "方案比较题必须包含 choice。JSON 后不要继续输出任何内容。"
            ),
        },
        "gpt_5_5": {
            "provider_label": "GPT5.5",
            "model": "gpt-5.5",
            "base_url": proxy_base("GPT5.5"),
            "api_key": proxy["GPT5.5"]["key"],
        },
        "gemini_3_5_flash": {
            "provider_label": "Gemini",
            "model": "gemini-3.5-flash",
            "base_url": proxy_base("Gemini"),
            "api_key": proxy["Gemini"]["key"],
        },
    }


def base_host(base_url: str) -> str:
    return urlparse(base_url).netloc


def _request_json(method: str, url: str, api_key: str, payload: dict | None = None,
                  timeout: int = 90) -> tuple[int | str, dict, float]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method=method)
    req.add_header("Authorization", f"Bearer {api_key}")
    if body is not None:
        req.add_header("Content-Type", "application/json")
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read(65536).decode("utf-8", "replace")
            return resp.status, json.loads(raw), time.time() - t0
    except urllib.error.HTTPError as exc:
        raw = exc.read(4096).decode("utf-8", "replace")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {"raw": raw[:500]}
        return exc.code, data, time.time() - t0
    except Exception as exc:  # noqa: BLE001
        return "EXC", {"error": type(exc).__name__, "message": str(exc)[:500]}, time.time() - t0


def _stream_tool_probe(spec: dict[str, str]) -> dict:
    tool_spec = [{
        "type": "function",
        "function": {
            "name": "add_numbers",
            "description": "Add two integers.",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a", "b"],
            },
        },
    }]
    payload = {
        "model": spec["model"],
        "messages": [{"role": "user", "content": "Use the tool to add 2 and 3. Do not answer directly."}],
        "tools": tool_spec,
        "tool_choice": "auto",
        "temperature": 0,
        "stream": True,
    }
    payload.update(spec.get("extra_body") or {})
    req = urllib.request.Request(
        f"{spec['base_url']}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
    )
    req.add_header("Authorization", f"Bearer {spec['api_key']}")
    req.add_header("Content-Type", "application/json")
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            fragments: list[dict] = []
            finish: list[str] = []
            lines = 0
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                chunk = line[5:].strip()
                if chunk == "[DONE]":
                    break
                try:
                    data = json.loads(chunk)
                except json.JSONDecodeError:
                    continue
                lines += 1
                choice = (data.get("choices") or [{}])[0]
                if choice.get("finish_reason"):
                    finish.append(choice["finish_reason"])
                delta = choice.get("delta") or {}
                for tc in delta.get("tool_calls") or []:
                    fn = tc.get("function") or {}
                    fragments.append({
                        "id": tc.get("id"),
                        "name": fn.get("name"),
                        "args": fn.get("arguments"),
                        "index": tc.get("index"),
                    })
            return {
                "status": resp.status,
                "elapsed_s": round(time.time() - t0, 2),
                "sse_lines": lines,
                "finish": finish,
                "tool_call_ok": any((f.get("name") == "add_numbers") or f.get("args") for f in fragments),
            }
    except urllib.error.HTTPError as exc:
        raw = exc.read(4096).decode("utf-8", "replace")
        try:
            err = json.loads(raw)
        except json.JSONDecodeError:
            err = {"raw": raw[:500]}
        return {"status": exc.code, "elapsed_s": round(time.time() - t0, 2), "error": err}
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "EXC",
            "elapsed_s": round(time.time() - t0, 2),
            "error": type(exc).__name__,
            "message": str(exc)[:500],
        }


def preflight(models: dict[str, dict[str, str]], aliases: list[str]) -> bool:
    ok_all = True
    for alias in aliases:
        spec = models[alias]
        print(f"\n## {alias} | {spec['model']} | {base_host(spec['base_url'])}", flush=True)

        status, data, elapsed = _request_json("GET", f"{spec['base_url']}/models", spec["api_key"], timeout=30)
        model_ids = [m.get("id") for m in data.get("data", []) if isinstance(m, dict) and m.get("id")]
        has_model = spec["model"] in model_ids
        print(f"models: status={status} elapsed={elapsed:.2f}s has_model={has_model}", flush=True)

        payload = {
            "model": spec["model"],
            "messages": [{"role": "user", "content": 'Return compact JSON only: {"ok":true,"answer":42}'}],
            "temperature": 0,
            "max_tokens": 512,
            "stream": False,
        }
        payload.update(spec.get("extra_body") or {})
        status, data, elapsed = _request_json(
            "POST", f"{spec['base_url']}/chat/completions", spec["api_key"], payload=payload
        )
        choice = (data.get("choices") or [{}])[0] if isinstance(data, dict) else {}
        content = ((choice.get("message") or {}).get("content") or "")[:120]
        json_ok = status == 200 and '"answer":42' in content.replace(" ", "")
        print(
            f"json_chat: status={status} elapsed={elapsed:.2f}s ok={json_ok} "
            f"finish={choice.get('finish_reason')} content={content!r}",
            flush=True,
        )

        tool = _stream_tool_probe(spec)
        tool_ok = tool.get("status") == 200 and tool.get("tool_call_ok")
        print(
            f"stream_tool: status={tool.get('status')} elapsed={tool.get('elapsed_s')}s "
            f"ok={tool_ok} finish={tool.get('finish')}",
            flush=True,
        )
        ok_all = ok_all and status == 200 and json_ok and tool_ok
    return ok_all


def annotate_jsonl(path: Path, alias: str, spec: dict[str, str], run_group: str) -> None:
    if not path.exists():
        return
    lines = []
    changed = False
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        rec = json.loads(raw)
        metadata = {
            "model_alias": alias,
            "provider_label": spec["provider_label"],
            "model": spec["model"],
            "base_host": base_host(spec["base_url"]),
            "run_group": run_group,
        }
        for key, value in metadata.items():
            if rec.get(key) != value:
                rec[key] = value
                changed = True
        lines.append(json.dumps(rec, ensure_ascii=False))
    if changed:
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def run_one(alias: str, spec: dict[str, str], framework: str, reps: int,
            ids: str, run_group: str) -> None:
    out = RAW_ROOT / alias / f"{framework}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-u", str(FRAMEWORK_SCRIPTS[framework]), "--reps", str(reps), "--out", str(out)]
    if ids:
        cmd.extend(["--ids", ids])

    env = os.environ.copy()
    env.update({
        "LLM_BASE_URL": spec["base_url"],
        "LLM_MODEL": spec["model"],
        "LLM_API_KEY": spec["api_key"],
    })
    if spec.get("extra_body"):
        env["LLM_EXTRA_BODY_JSON"] = json.dumps(spec["extra_body"], ensure_ascii=False)
    if spec.get("max_tokens"):
        env["LLM_MAX_TOKENS"] = str(spec["max_tokens"])
    if spec.get("timeout_s"):
        env["LLM_TIMEOUT_S"] = str(spec["timeout_s"])
    if spec.get("system_suffix"):
        env["LLM_SYSTEM_SUFFIX"] = str(spec["system_suffix"])
    print(f"\n=== {alias} / {framework} / reps={reps} / out={out} ===", flush=True)
    proc = subprocess.run(cmd, cwd=str(HERE), env=env, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"{alias}/{framework} failed with exit code {proc.returncode}")
    annotate_jsonl(out, alias, spec, run_group)


def parse_csv_arg(value: str, allowed_order: list[str]) -> list[str]:
    if value.lower() == "all":
        return list(allowed_order)
    items = [x.strip() for x in value.split(",") if x.strip()]
    allowed = set(allowed_order)
    bad = sorted(set(items) - allowed)
    if bad:
        raise SystemExit(f"Unknown values: {bad}; allowed={allowed_order}")
    return items


def main() -> None:
    ap = argparse.ArgumentParser(description="Run multi-model QA benchmark matrix")
    ap.add_argument("--stage", choices=["preflight", "pilot", "rep1", "stability"], default="preflight")
    ap.add_argument("--models", default="all", help="Comma-separated aliases or all")
    ap.add_argument("--frameworks", default="all", help="Comma-separated framework names or all")
    ap.add_argument("--ids", default="", help="Override question ids. Defaults depend on stage.")
    ap.add_argument("--reps", type=int, default=0, help="Override repetitions. Defaults depend on stage.")
    ap.add_argument("--skip-preflight", action="store_true")
    args = ap.parse_args()

    models = load_model_specs()
    aliases = parse_csv_arg(args.models, list(models))
    frameworks = parse_csv_arg(args.frameworks, list(FRAMEWORK_SCRIPTS))

    if args.stage == "preflight":
        ok = preflight(models, aliases)
        raise SystemExit(0 if ok else 1)

    if not args.skip_preflight:
        ok = preflight(models, aliases)
        if not ok:
            raise SystemExit("Preflight failed; rerun with --skip-preflight only after manual review.")

    ids = args.ids
    reps = args.reps
    if args.stage == "pilot":
        ids = ids or PILOT_IDS
        reps = reps or 1
    elif args.stage == "rep1":
        reps = reps or 1
    elif args.stage == "stability":
        reps = reps or 3

    for alias in aliases:
        for framework in frameworks:
            run_one(alias, models[alias], framework, reps, ids, args.stage)

    print(f"\nDone. Raw outputs: {RAW_ROOT}", flush=True)


if __name__ == "__main__":
    main()
