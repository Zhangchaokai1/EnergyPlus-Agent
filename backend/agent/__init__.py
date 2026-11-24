"""EnergyPlus-Agent 智能体后端模块。

骨架阶段：仅包含大模型连接（config + llm + routes）。
后续阶段将加入：tools（Function Calling 工具层）、gating（σ_E 不确定性门控）、
orchestrator（工具调用循环）、eplus_fallback（真实 EnergyPlus 回退仿真）。
"""
