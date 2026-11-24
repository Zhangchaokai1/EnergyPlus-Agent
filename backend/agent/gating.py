"""σ_E 不确定性门控（框架的灵魂：知之为知之，不知则仿真）。

加载验证集 p95 阈值（models/.../evaluation/ood_gate_recall.csv），对单个设计的
energy_total 集成分歧 σ_E 给出"信任代理 / 建议回退 EnergyPlus 仿真"判定。

energy_total 的 σ_E = 5 个集成成员各自 (年供暖负荷 + 年供冷负荷) 的标准差
（与 evaluate_office_surrogate.py 中门控设计口径一致；不是按目标分别求 σ 再合成）。

在该 p95 阈值下：OOD 召回率 100%，分布内误报约 5.2%（见 ood_gate_recall.csv）。
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]  # 仓库根（agent 比 app.py 深一层）
GATE_RECALL_CSV = ROOT / "models" / "office_hc_deeponet_ensemble" / "evaluation" / "ood_gate_recall.csv"

# 兜底阈值（CSV 缺失时使用）：各目标验证集 p95 in-dist σ_E
_FALLBACK_TAU = {
    "energy_total": 21163.715,
    "heating": 3292.299,
    "cooling": 21536.025,
    "discomfort": 11.890,
}

# 目标列顺序：[heating, cooling, discomfort]；energy_total = heating + cooling
ENERGY_IDX = (0, 1)


@lru_cache(maxsize=1)
def get_thresholds() -> dict:
    """读取验证集 p95 门控阈值；失败则用兜底常量。"""
    try:
        df = pd.read_csv(GATE_RECALL_CSV)
        return {str(r["target"]): float(r["tau_sigmaE_p95_indist"]) for _, r in df.iterrows()}
    except Exception:
        return dict(_FALLBACK_TAU)


def energy_total_sigma(member_annual: np.ndarray) -> float:
    """member_annual: (M, N, targets) 或 (M, targets)。返回第 1 个样本的 energy_total σ_E。"""
    arr = np.asarray(member_annual)
    idx = list(ENERGY_IDX)
    if arr.ndim == 3:
        per_member = arr[:, 0, idx].sum(axis=1)  # (M,)
    else:
        per_member = arr[:, idx].sum(axis=1)
    return float(per_member.std())


def gate_verdict(sigma_energy_total: float) -> dict:
    """给出门控判定。sigma 单位为 kWh（负荷尺度）。"""
    tau = get_thresholds().get("energy_total", _FALLBACK_TAU["energy_total"])
    trust = sigma_energy_total <= tau
    ratio = sigma_energy_total / tau if tau else float("inf")
    return {
        "sigmaEnergyTotalKwh": round(sigma_energy_total, 1),
        "thresholdP95Kwh": round(tau, 1),
        "ratio": round(ratio, 3),
        "trustSurrogate": bool(trust),
        "verdict": "trust" if trust else "fallback",
        "message": (
            "σ_E 低于验证集 p95 阈值，处于训练分布内，可直接信任代理模型结果（毫秒级）。"
            if trust else
            "σ_E 超过验证集 p95 阈值，疑似外推/OOD 工况，建议回退 EnergyPlus 高保真仿真复核。"
        ),
    }
