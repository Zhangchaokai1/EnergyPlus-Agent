"""大模型 Provider 配置。

从环境变量 / backend/.env 读取 provider / base_url / model / api_key。
内置零依赖的 .env 解析器（不强制 python-dotenv）。Provider 做了抽象，
所有预设都使用 OpenAI 兼容协议，便于后续把 DeepSeek 切换为阿里百炼(DashScope)
或其它 OpenAI 兼容端点，而无需改动调用方。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"


def _load_env_file(path: Path) -> None:
    """极简 .env 加载器：KEY=VALUE，忽略空行/注释；不覆盖已存在的环境变量。"""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_env_file(_ENV_PATH)


# provider -> 默认端点/模型/密钥环境变量名（均为 OpenAI 兼容协议）
PROVIDER_PRESETS: dict[str, dict[str, str]] = {
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
        "api_key_env": "DEEPSEEK_API_KEY",
    },
    "dashscope": {  # 阿里百炼 OpenAI 兼容模式（赛事备用）
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
        "api_key_env": "DASHSCOPE_API_KEY",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "api_key_env": "OPENAI_API_KEY",
    },
}


def get_llm_config() -> dict:
    """返回当前生效的大模型配置。

    优先级：显式 LLM_BASE_URL/LLM_MODEL/LLM_API_KEY 环境变量 > provider 预设。
    """
    provider = os.environ.get("LLM_PROVIDER", "deepseek").lower()
    preset = PROVIDER_PRESETS.get(provider, PROVIDER_PRESETS["deepseek"])
    base_url = os.environ.get("LLM_BASE_URL", preset["base_url"])
    model = os.environ.get("LLM_MODEL", preset["model"])
    api_key = (
        os.environ.get("LLM_API_KEY")
        or os.environ.get(preset["api_key_env"])
        or ""
    )
    return {
        "provider": provider,
        "base_url": base_url,
        "model": model,
        "api_key": api_key,
    }


def get_llm_request_options() -> dict:
    """Return optional chat.completions kwargs from environment.

    This is mainly used by benchmark sweeps for OpenAI-compatible providers
    that expose model-specific knobs such as Qwen's enable_thinking flag.
    """
    opts: dict = {}

    extra_raw = os.environ.get("LLM_EXTRA_BODY_JSON", "").strip()
    if extra_raw:
        extra_body = json.loads(extra_raw)
        if not isinstance(extra_body, dict):
            raise RuntimeError("LLM_EXTRA_BODY_JSON 必须是 JSON 对象")
        opts["extra_body"] = extra_body

    max_tokens = os.environ.get("LLM_MAX_TOKENS", "").strip()
    if max_tokens:
        opts["max_tokens"] = int(max_tokens)

    timeout_s = os.environ.get("LLM_TIMEOUT_S", "").strip()
    if timeout_s:
        opts["timeout"] = float(timeout_s)

    return opts
