"""智能体后端路由（骨架阶段：流式对话 + 健康检查）。

挂载在主应用 app.py 上：app.include_router(agent_router)。
后续阶段会在这里扩展工具调用、门控状态与 EnergyPlus 回退任务查询等端点。
"""
from __future__ import annotations

import json
from typing import Literal

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/agent", tags=["agent"])


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"] = "user"
    content: str


class AgentChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(default_factory=list)


@router.get("/health")
def agent_health():
    """返回当前大模型 provider/model 及是否已配置密钥（不回传密钥本身）。"""
    from .config import get_llm_config

    cfg = get_llm_config()
    return {
        "provider": cfg["provider"],
        "model": cfg["model"],
        "baseUrl": cfg["base_url"],
        "configured": bool(cfg["api_key"]),
    }


@router.post("/chat")
def agent_chat(request: AgentChatRequest):
    """SSE 流式：事件 {type: delta|tool_call|tool_result|done|error}。"""
    from .orchestrator import run_agent_stream

    messages = [m.model_dump() for m in request.messages]

    def event_stream():
        try:
            for event in run_agent_stream(messages):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as exc:  # noqa: BLE001 - 把错误透传到前端气泡
            yield f"data: {json.dumps({'type': 'error', 'error': str(exc)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
