"""大模型流式对话封装（默认 DeepSeek，OpenAI 兼容）。

骨架阶段：仅做纯流式对话，尚未接入工具调用。
后续 agent/orchestrator.py 将在此基础上加入 Function Calling 循环、
σ_E 不确定性门控与 EnergyPlus 回退仿真。
"""
from __future__ import annotations

from typing import Iterable, Iterator

from .config import get_llm_config

# 系统提示：确立智能体身份与"物理增强生成(PAG) + 不确定性门控"的叙事。
SYSTEM_PROMPT = """你是 EnergyPlus-Agent，一个面向建筑能耗与节能分析的物理增强智能体。

你背后连接着：
1) 一个用 EnergyPlus 仿真数据训练的 HC-DeepONet 深度集成代理模型（5 成员），输入 10 维建筑/系统
   参数与 20 维逐日气候特征，输出冷热负荷、等效不舒适小时，并给出预测均值 μ_E 与集成分歧 σ_E；
2) 高保真 EnergyPlus 参数化仿真器。

核心理念是"物理增强生成(PAG) + 不确定性门控"：能算的交给物理代理算（毫秒级）；当 σ_E 偏高
（外推/极端工况，例如异常天气或超出训练范围的围护参数）时，回退到 EnergyPlus 仿真，做到
"知之为知之，不知则仿真"。数值推理应交给物理模型，而不是你来心算热力学。

服务对象包括业主、物业、社区与设计院人员，主题围绕"双碳与城市更新 / 可信物理 AI / 建筑物理民主化"。
典型场景：① 设计/改造期的成本-能耗-舒适三目标权衡；② 运维期空调设定点与节能；③ 极端天气能耗峰值估计。

当前为大模型联调骨架阶段：工具调用尚未接入。请先用简体中文、简洁专业地回答用户问题；当问题需要定量
结果时，说明你随后将调用代理模型/仿真来给出具体数值与不确定性，而不要凭空捏造精确数字。"""


def _client():
    cfg = get_llm_config()
    if not cfg["api_key"]:
        raise RuntimeError(
            "未配置大模型 API Key：请在 EnergyPlus-Agent/backend/.env 中设置 DEEPSEEK_API_KEY。"
        )
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "缺少 openai 依赖：请先在 backend 环境执行 pip install openai。"
        ) from exc
    return OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"]), cfg["model"]


def stream_chat(messages: Iterable[dict], temperature: float = 0.6) -> Iterator[str]:
    """逐段产出 assistant 文本增量（自动在最前面拼接系统提示）。

    messages: [{"role": "user"|"assistant", "content": str}, ...]
    """
    client, model = _client()
    payload = [{"role": "system", "content": SYSTEM_PROMPT}, *messages]
    stream = client.chat.completions.create(
        model=model,
        messages=payload,
        stream=True,
        temperature=temperature,
    )
    for chunk in stream:
        if not getattr(chunk, "choices", None):
            continue
        delta = chunk.choices[0].delta
        content = getattr(delta, "content", None)
        if content:
            yield content
