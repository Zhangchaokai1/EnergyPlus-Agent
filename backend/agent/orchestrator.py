"""智能体编排器：流式 Function Calling 循环。

每一步用 stream=True 调用大模型并累积增量：
  - 文本增量 → 产出 {"type":"delta","content":...}
  - 工具调用增量 → 累积后执行工具，产出 {"type":"tool_call"} 与 {"type":"tool_result"}
若本轮 finish_reason 为 tool_calls，则把工具结果回灌到对话并继续下一轮；否则结束。

这样既能 token 级流式输出最终回答，又能把"调用了哪个工具 / 门控判定"实时展示到前端。
"""
from __future__ import annotations

import json
from typing import Iterable, Iterator

from .config import get_llm_request_options
from .llm import _client
from .tools import TOOL_SPECS, dispatch_tool
from .pareto_presentation import PRESENTATION_SPEC, render_pareto_comparison, normalize_priorities
from . import eplus_fallback

MAX_TOOL_STEPS = 6
OUTPUT_LANGUAGE = "zh"

SYSTEM_PROMPT = """你是 EnergyPlus-Agent，面向建筑能耗与改造决策的物理增强智能体。默认模型是基于 EnergyPlus 仿真训练的南京高层办公楼五成员 HC-DeepONet 集成；也支持模型库中用户训练的建筑专属模型。

定量依据来自工具：predict_energy 提供设计预测和不确定性门控；optimize_pareto 提供成本—用电—舒适方案；operation_savings 提供设定点节能；estimate_extreme 提供异常天气情景；explain_features 提供 SHAP；list_available_models 提供模型信息；run_energyplus_simulation 提供高保真核验。门控建议回退或用户要求核验时，告知用户后运行 EnergyPlus，并对比代理与仿真结果。说明建筑、模型与天气依据，不把默认模型几何套用于其他建筑。集成分歧较小本身不证明准确。

解释规则：
- Pareto 排序与极值采用 comparisonSummary，并明确范围仅为展示方案。展示子集按成本升序选择，不是按综合平衡评分排序。候选集内的非支配关系与用户偏好分开；推荐说明优先目标。未计算唯一综合最优、单位投资性价比或普遍成本收益规律。
- 区分平均预测用电与含不确定性惩罚的稳健目标，节能率沿用工具的计算口径。引用工具给出的数值与单位，不编造工具未计算的评分、比值或定量比较；用对应行的数据说明取舍。
- SHAP 使用 directionCode 及证据。正、负贡献分别相对归因参考点提高、降低指标；重要度百分比为平均绝对 SHAP 的归一化占比，不是能耗占比或节能率。相关系数与四分位组模式描述关联，不证明因果或全局单调性；全局归因不能解释某个方案的具体成因。
- 混合 SHAP 表示没有单一方向满足显示规则；用户未追问原因时简洁说明即可。解释标签时必须逐变量核对其自身的相关系数、四分位组均值与同号比例，不把某个变量未满足的判据套用于另一个变量。没有分析支持时不推断交互或物理机制；用户明确要求假设时才单独标注假设。
- 归类变量时，供暖/供冷设定温度属于运行控制，COP 属于设备效率，SHGC/WWR/窗 U 值属于围护结构参数。SHAP 方向相同不代表物理类别相同：供暖设定温度应与围护结构及太阳得热变量分开表述。

用简体中文简洁、专业地回答，以紧凑表格和基于明确偏好的取舍建议为主。直接使用已计算的比较摘要，不重新推断极值。不提出缺乏依据的设计或执行过程描述。标题前留空行。"""


def _present_events(pareto_result, shap_result, priorities, language, *, selection_source):
    priorities = normalize_priorities(priorities)
    yield {"type": "tool_call", "name": "present_pareto_comparison",
           "arguments": {"priority_order": priorities}}
    text = render_pareto_comparison(pareto_result, shap_result,
                                   priority_order=priorities, language=language)
    yield {"type": "tool_result", "name": "present_pareto_comparison", "result": {
        "source": "computed_pareto_presentation",
        "displayedCount": len(pareto_result["topSolutions"]),
        "priorityOrder": priorities,
        "prioritySelectionSource": selection_source,
        "basis": "Computed tool rows and rankings; no free-form rewrite of numerical comparisons",
    }}
    yield {"type": "delta", "content": "\n\n" + text}
    yield {"type": "done"}


def run_agent_stream(messages: Iterable[dict], *, system_prompt: str | None = None,
                     temperature: float = 0.3) -> Iterator[dict]:
    """Stream normal tools; present Pareto facts through a bounded tool renderer."""
    client, model = _client()
    request_options = get_llm_request_options()
    messages = list(messages)
    convo = [{"role": "system", "content": system_prompt or SYSTEM_PROMPT}, *messages]
    has_tool_results = False
    pareto_result = None
    shap_result = None

    for _ in range(MAX_TOOL_STEPS):
        specs = TOOL_SPECS + ([PRESENTATION_SPEC] if pareto_result is not None else [])
        # If a Pareto result exists, final factual prose is emitted by the
        # presentation tool. Other task types retain their existing streaming.
        bounded_output = pareto_result is not None
        stream = client.chat.completions.create(
            model=model, messages=convo, tools=specs, tool_choice="auto",
            stream=True, temperature=temperature, **request_options,
        )
        content_parts: list[str] = []
        tool_calls: dict[int, dict] = {}
        finish_reason = None
        for chunk in stream:
            if not getattr(chunk, "choices", None):
                continue
            choice = chunk.choices[0]
            delta = choice.delta
            if getattr(delta, "content", None):
                content_parts.append(delta.content)
                if not bounded_output:
                    yield {"type": "delta", "content": delta.content}
            for tcd in (getattr(delta, "tool_calls", None) or []):
                slot = tool_calls.setdefault(tcd.index, {"id": "", "name": "", "args": ""})
                if tcd.id:
                    slot["id"] = tcd.id
                fn = getattr(tcd, "function", None)
                if fn is not None:
                    if getattr(fn, "name", None):
                        slot["name"] = fn.name
                    if getattr(fn, "arguments", None):
                        slot["args"] += fn.arguments
            if choice.finish_reason:
                finish_reason = choice.finish_reason

        if finish_reason == "tool_calls" and tool_calls:
            convo.append({
                "role": "assistant", "content": "".join(content_parts) or None,
                "tool_calls": [
                    {"id": tc["id"] or f"call_{idx}", "type": "function",
                     "function": {"name": tc["name"], "arguments": tc["args"] or "{}"}}
                    for idx, tc in sorted(tool_calls.items())
                ],
            })
            presentation_args = None
            for idx, tc in sorted(tool_calls.items()):
                name = tc["name"]
                try:
                    args = json.loads(tc["args"]) if tc["args"] else {}
                    if not isinstance(args, dict):
                        args = {}
                except json.JSONDecodeError:
                    args = {}
                if name == "present_pareto_comparison" and pareto_result is not None:
                    # Execute other tools in this round first (e.g. SHAP).
                    presentation_args = args
                    continue
                yield {"type": "tool_call", "name": name, "arguments": args}
                if name == "run_energyplus_simulation":
                    result = {"error": "Simulation did not run"}
                    for ev in eplus_fallback.run_with_progress(args):
                        if "result" in ev:
                            result = ev["result"]
                        else:
                            yield {"type": "progress", "name": name, "elapsedS": ev["elapsedS"]}
                else:
                    result = dispatch_tool(name, args)
                yield {"type": "tool_result", "name": name, "result": result}
                if name == "optimize_pareto" and isinstance(result, dict) and "topSolutions" in result:
                    pareto_result = result
                if name == "explain_features" and isinstance(result, dict) and "topFeatures" in result:
                    shap_result = result
                llm_result = ({k: v for k, v in result.items() if k != "chart"}
                              if isinstance(result, dict) else result)
                convo.append({"role": "tool", "tool_call_id": tc["id"] or f"call_{idx}",
                              "content": json.dumps(llm_result, ensure_ascii=False)})
                has_tool_results = True
            if presentation_args is not None:
                yield from _present_events(pareto_result, shap_result,
                                           presentation_args.get("priority_order"), OUTPUT_LANGUAGE,
                                           selection_source="llm_tool_call")
                return
            if pareto_result is not None:
                convo.append({"role": "system", "content":
                    "For this Pareto answer, obtain any other requested tool evidence first, then finish with present_pareto_comparison. Its table and trade-offs are displayed directly; do not write a separate numerical comparison. Choose discussion priority_order from the user request, not optimization weights."})
            continue

        if pareto_result is not None:
            # Some providers try prose instead of calling the presentation tool.
            # Request only its small structured priority selection, never ask for
            # a second prose repair or silently overwrite a saved conversation.
            final = client.chat.completions.create(
                model=model, messages=convo, tools=[PRESENTATION_SPEC],
                tool_choice={"type": "function", "function": {"name": "present_pareto_comparison"}},
                stream=False, temperature=temperature, **request_options,
            )
            selected = None
            for call in (getattr(final.choices[0].message, "tool_calls", None) or []):
                if call.function.name == "present_pareto_comparison":
                    try:
                        value = json.loads(call.function.arguments or "{}")
                        selected = value.get("priority_order") if isinstance(value, dict) else None
                    except json.JSONDecodeError:
                        pass
            yield from _present_events(pareto_result, shap_result, selected, OUTPUT_LANGUAGE,
                                       selection_source="llm_tool_call" if selected else "default_all_priorities")
            return

        if has_tool_results and not "".join(content_parts).strip():
            final = client.chat.completions.create(
                model=model, messages=convo, tools=TOOL_SPECS, tool_choice="none",
                stream=False, temperature=temperature, **request_options,
            )
            text = final.choices[0].message.content or ""
            if text:
                yield {"type": "delta", "content": text}
        yield {"type": "done"}
        return

    if pareto_result is not None:
        yield from _present_events(pareto_result, shap_result, None, OUTPUT_LANGUAGE,
                                   selection_source="step_limit_default_all_priorities")
    else:
        yield {"type": "done"}
