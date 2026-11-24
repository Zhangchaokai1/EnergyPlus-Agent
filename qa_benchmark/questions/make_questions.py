# -*- coding: utf-8 -*-
"""构建建筑能耗问答基准题集 v1（45 题，三档难度，可复现）。

选点策略与出题纪律见 schema.md 与 ../基准实验规划.md。
固定随机种子（SEED=20260702），任何人重跑得到同一题集。

产出（写入本目录 / ../truth_runs）：
  questions_v1.yaml      45 题（新点/对照变体题在 extra_truth.csv 就绪前 truth.status=pending）
  selection_report.md    选点统计与题集总表（汇报点②材料）
  ../truth_runs/extra_jobs.csv   需补跑的 EnergyPlus 任务清单（14 个）

用法：python make_questions.py（任意 cwd；补跑完成后重跑本脚本即自动绑定真值）
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # D:\pyproject\EnergyPlus\7
sys.path.insert(0, str(ROOT))

from office_sampling import BASELINE, VAR_ORDER, VARIABLE_BOUNDS, VARIABLE_STEPS, quantize  # noqa: E402
from weather_ood_stress_test import generate_indist_buildings  # noqa: E402

SEED = 20260702
SUMMARY_CSV = ROOT / "datasets" / "office_sample_summary.csv"
SPLIT_CSV = ROOT / "models" / "office_hc_deeponet_ensemble" / "split_sample_ids.csv"
OOD_CSV = ROOT / "datasets" / "office_ood_dataset.csv"
WEATHER_TRUTH_CSV = ROOT / "datasets" / "office_weather_ood_truth.csv"
ANOM_STATS_CSV = ROOT / "models" / "office_hc_deeponet_ensemble" / "evaluation" / "weather_ood_anom_years.csv"
EXTRA_JOBS_CSV = HERE.parent / "truth_runs" / "extra_jobs.csv"
EXTRA_TRUTH_CSV = HERE.parent / "truth_runs" / "extra_truth.csv"

# bounds 键名 → summary CSV 列名（仅 window_u 不同）
def col(k: str) -> str:
    return "window_u_w_m2k" if k == "window_u" else k

PARAM_COLS = [col(k) for k in VAR_ORDER]
# 注意：数据集的 annual_total_input_kwh 是全楼电耗（含照明+设备），而题目与 PAG 工具口径都是
# "空调总电耗 = 供暖输入 + 供冷输入"。total_input 一律现算，不用现成总列。
TARGET_COL = {
    "cooling_input": "annual_cooling_input_kwh",
    "heating_input": "annual_heating_input_kwh",
}


def hvac_total(row) -> float:
    return float(row["annual_heating_input_kwh"]) + float(row["annual_cooling_input_kwh"])

# ---------------------------------------------------------------- 几何/度量工具

def norm_vec(p: dict) -> np.ndarray:
    out = []
    for k in VAR_ORDER:
        lo, hi = VARIABLE_BOUNDS[k]
        out.append((float(p[col(k)]) - lo) / (hi - lo))
    return np.asarray(out)


def bound_margins(p: dict) -> np.ndarray:
    """每维到最近边界的归一化距离（界内为正，界外为负）。"""
    out = []
    for k in VAR_ORDER:
        lo, hi = VARIABLE_BOUNDS[k]
        v = float(p[col(k)])
        out.append(min(v - lo, hi - v) / (hi - lo))
    return np.asarray(out)


def greedy_maximin(pool: pd.DataFrame, n: int, anchor: np.ndarray | None = None) -> list[int]:
    """在 pool（含 _nv 归一化向量列）里贪心选 n 个彼此尽量分散的行，返回 pool.index 列表。"""
    idxs = list(pool.index)
    vecs = np.stack(pool["_nv"].to_list())
    chosen: list[int] = []
    chosen_vecs: list[np.ndarray] = [] if anchor is None else [anchor]
    for _ in range(n):
        if not chosen_vecs:
            pick = 0  # 第一个取池中距池心最远者
            center = vecs.mean(axis=0)
            pick = int(np.argmax(np.linalg.norm(vecs - center, axis=1)))
        else:
            d = np.min(np.stack([np.linalg.norm(vecs - cv, axis=1) for cv in chosen_vecs]), axis=0)
            d[[idxs.index(c) for c in chosen]] = -1.0
            pick = int(np.argmax(d))
        chosen.append(idxs[pick])
        chosen_vecs.append(vecs[pick])
    return chosen


def params_of(row: pd.Series) -> dict:
    return {c: round(float(row[c]), 4) for c in PARAM_COLS}


# ---------------------------------------------------------------- 题面渲染

OPENERS = [
    "南京一栋 15 层高层办公楼（总建筑面积约 19440 m²，标准层 36 m × 36 m），按南京典型气象年评估。",
    "我们在南京运营一栋约 19440 m² 的高层办公楼（15 层），按当地典型气象年考虑。",
    "某设计院正在评估南京一栋高层办公方案（15 层、约 19440 m²），气象按南京典型年。",
    "位于南京的一栋约 19440 m² 高层办公建筑（15 层），按典型气象年估算。",
]

TARGET_PHRASE = {
    "total_input": "全年空调总电耗（供冷+供暖输入电耗合计，kWh）大约是多少？",
    "cooling_input": "全年供冷输入电耗（kWh）大约是多少？",
    "heating_input": "全年供暖输入电耗（kWh）大约是多少？",
}


def desc_design(p: dict, skip: set[str] | None = None) -> str:
    skip = skip or set()
    parts = []
    if "wall_insulation_m" not in skip:
        parts.append(f"外墙保温 {p['wall_insulation_m'] * 1000:.0f} mm")
    if "roof_insulation_m" not in skip:
        parts.append(f"屋面保温 {p['roof_insulation_m'] * 1000:.0f} mm")
    if "window_u_w_m2k" not in skip:
        parts.append(f"外窗传热系数 U={p['window_u_w_m2k']:g} W/(m²·K)")
    if "window_shgc" not in skip:
        parts.append(f"太阳得热系数 SHGC={p['window_shgc']:g}")
    if "infiltration_ach" not in skip:
        parts.append(f"渗透率 {p['infiltration_ach']:g} ACH")
    if "window_wall_ratio" not in skip:
        parts.append(f"窗墙比 {p['window_wall_ratio']:g}")
    parts.append(f"空调供冷/供暖设定点 {p['cooling_setpoint_c']:g}/{p['heating_setpoint_c']:g} °C")
    parts.append(f"供冷系统 COP {p['cooling_cop']:g}、供暖系统 COP {p['heating_cop']:g}")
    return "，".join(parts)


def q_point(i: int, p: dict, target: str) -> str:
    return f"{OPENERS[i % len(OPENERS)]}围护与系统参数：{desc_design(p)}。请问{TARGET_PHRASE[target]}"


def q_compare(i: int, pa: dict, pb: dict) -> str:
    return (
        f"{OPENERS[i % len(OPENERS)]}现有两个候选方案。\n"
        f"方案 A：{desc_design(pa)}。\n方案 B：{desc_design(pb)}。\n"
        f"哪个方案全年空调总电耗（供冷+供暖输入电耗）更低？大约低多少 kWh？"
    )


def q_saving(i: int, p: dict, param: str, v_from: float, v_to: float) -> str:
    if param == "cooling_setpoint_c":
        act = f"把供冷设定点从 {v_from:g} °C 提高到 {v_to:g} °C"
        tgt = "全年供冷输入电耗大约能节省多少 kWh？"
    else:
        act = f"把供暖设定点从 {v_from:g} °C 降低到 {v_to:g} °C"
        tgt = "全年供暖输入电耗大约能节省多少 kWh？"
    return f"{OPENERS[i % len(OPENERS)]}当前参数：{desc_design(p)}。若{act}（其余参数不变），{tgt}"


# ---------------------------------------------------------------- 主流程

def main() -> None:
    rng = np.random.default_rng(SEED)

    summary = pd.read_csv(SUMMARY_CSV, encoding="utf-8-sig")
    summary = summary[summary["simulation_status"] == "ok"].copy()
    split = pd.read_csv(SPLIT_CSV, encoding="utf-8-sig")
    train_ids = set(split.loc[split["split"] == "train", "sample_id"])
    val_ids = set(split.loc[split["split"] == "val", "sample_id"])

    val = summary[summary["sample_id"].isin(val_ids)].copy()
    train = summary[summary["sample_id"].isin(train_ids)].copy()
    for df in (val, train):
        df["_nv"] = df.apply(lambda r: norm_vec(r), axis=1)
        df["_margins"] = df.apply(lambda r: bound_margins(r), axis=1)

    # OOD 设计点（每 sample 取首行，annual_* 各行相同）
    ood = pd.read_csv(OOD_CSV, encoding="utf-8-sig")
    ood = ood[ood["simulation_status"] == "ok"].groupby("sample_id", as_index=False).first()

    def violation(row) -> float:
        out = []
        for k in VAR_ORDER:
            lo, hi = VARIABLE_BOUNDS[k]
            v = float(row[col(k)])
            out.append(max(lo - v, v - hi, 0.0) / (hi - lo))
        return float(np.mean(out))

    ood["_viol"] = ood.apply(violation, axis=1)
    ood["_nv"] = ood.apply(lambda r: norm_vec(r), axis=1)

    # 气候异常年真值 + 20 栋测试建筑（确定性复原）
    wt = pd.read_csv(WEATHER_TRUTH_CSV, encoding="utf-8-sig")
    anom_stats = pd.read_csv(ANOM_STATS_CSV, encoding="utf-8-sig").set_index("scenario")
    buildings = {i: b.to_record() for i, b in enumerate(generate_indist_buildings(20, 909))}

    used_val: set[int] = set()  # 已用 val sample_id，保证题间不重复

    def take(pool: pd.DataFrame, n: int) -> pd.DataFrame:
        pool = pool[~pool["sample_id"].isin(used_val)]
        picked = pool.loc[greedy_maximin(pool, n)]
        used_val.update(picked["sample_id"].tolist())
        return picked

    interior = val[val["_margins"].apply(lambda m: bool((m >= 0.05).all()))]
    # 近边样本：自适应阈值，保证池子够大
    for thr, ndim in [(0.02, 2), (0.03, 2), (0.02, 1)]:
        near_edge = val[val["_margins"].apply(lambda m: int((m <= thr).sum()) >= ndim)]
        if len(near_edge) >= 18:
            edge_rule = f"≥{ndim} 维距边 ≤{thr:g} 量程"
            break

    questions: list[dict] = []
    jobs: list[dict] = []
    job_seq = [0]

    def new_job(kind: str, qid: str, p: dict, note: str) -> str:
        job_seq[0] += 1
        jid = f"EX{job_seq[0]:02d}"
        jobs.append({"job_id": jid, "qid": qid, "kind": kind, "note": note, **p})
        return jid

    def add_q(qid, tier, qtype, ood_axis, text, params, truth, scoring, gate, **extra):
        q = {"id": qid, "tier": tier, "qtype": qtype, "ood_axis": ood_axis, "question_zh": text,
             "params": params, "truth": truth, "scoring": scoring, "expected_gate": gate}
        q.update(extra)
        questions.append(q)

    def truth_from_row(row, target: str) -> dict:
        src_file = "office_ood_dataset.csv" if int(row["sample_id"]) >= 100000 else "office_sample_summary.csv"
        v = hvac_total(row) if target == "total_input" else float(row[TARGET_COL[target]])
        return {"value": round(v, 1), "unit": "kWh",
                "source": f"{src_file}#sample_id={int(row['sample_id'])}", "status": "ok"}

    def scoring_of(target: str) -> dict:
        sc = {"target": target, "correct_pct": 10, "acceptable_pct": 25}
        if target.endswith("_saving"):
            # 节能量真值可能很小（如供暖侧数百 kWh），纯相对误差病态；
            # 增加绝对容差层：|err|<=floor 亦判 correct/acceptable（奖励答对量级）。
            sc["abs_floor_correct_kwh"] = 1000
            sc["abs_floor_acceptable_kwh"] = 2500
        return sc

    qn = [0]

    def nq() -> int:
        qn[0] += 1
        return qn[0]

    # ---------------- T1 分布内（20）----------------
    # P×9：留出集内部点
    t1_targets = ["total_input", "total_input", "total_input", "cooling_input", "heating_input",
                  "total_input", "cooling_input", "total_input", "heating_input"]
    for j, (_, row) in enumerate(take(interior, 9).iterrows()):
        t = t1_targets[j]
        add_q(f"T1-P{j+1:02d}", "in_dist", "point", None, q_point(nq(), params_of(row), t),
              params_of(row), truth_from_row(row, t), scoring_of(t), "trust")

    # P×3：远离语料的新采样点（补跑）
    existing_keys = {tuple(round(float(r[c]), 4) for c in PARAM_COLS) for _, r in summary.iterrows()}
    all_vecs = np.stack(pd.concat([train, val])["_nv"].to_list())
    cand_rows = []
    while len(cand_rows) < 4096:
        raw = {k: float(rng.uniform(*VARIABLE_BOUNDS[k])) for k in VAR_ORDER}
        p = {col(k): quantize(raw[k], *VARIABLE_BOUNDS[k], VARIABLE_STEPS[k]) for k in VAR_ORDER}
        m = bound_margins(p)
        if (m >= 0.05).all() and tuple(round(p[c], 4) for c in PARAM_COLS) not in existing_keys:
            cand_rows.append(p)
    cand_vec = np.stack([norm_vec(p) for p in cand_rows])
    picked_new: list[dict] = []
    dmin = np.min(np.linalg.norm(cand_vec[:, None, :] - all_vecs[None, :, :], axis=2), axis=1)
    for j in range(3):
        k = int(np.argmax(dmin))
        picked_new.append(cand_rows[k])
        dmin = np.minimum(dmin, np.linalg.norm(cand_vec - cand_vec[k], axis=1))
    for j, p in enumerate(picked_new):
        qid = f"T1-P{10+j:02d}"
        jid = new_job("new_point", qid, p, "远离语料的分布内新点")
        add_q(qid, "in_dist", "point", None, q_point(nq(), p, "total_input"), p,
              {"value": None, "unit": "kWh", "source": f"truth_runs/extra_truth.csv#job_id={jid}",
               "status": "pending"}, scoring_of("total_input"), "trust")

    # C×4：留出集方案比较（2 对差异大 + 2 对差异小）
    cmp_pool = take(interior, 8)
    rows = list(cmp_pool.iterrows())
    pairs, used_i = [], set()
    rel = lambda a, b: abs(hvac_total(a) - hvac_total(b)) / min(hvac_total(a), hvac_total(b))
    cand_pairs = sorted(((rel(a, b), ia, ib) for ia, (_, a) in enumerate(rows) for ib, (_, b) in enumerate(rows)
                         if ia < ib for a, b in [(a, b)]), reverse=True)
    for r, ia, ib in cand_pairs:  # 先大
        if len(pairs) >= 2 or ia in used_i or ib in used_i:
            continue
        pairs.append((ia, ib)); used_i.update((ia, ib))
    for r, ia, ib in reversed(cand_pairs):  # 再小
        if len(pairs) >= 4 or ia in used_i or ib in used_i:
            continue
        pairs.append((ia, ib)); used_i.update((ia, ib))
    for j, (ia, ib) in enumerate(pairs):
        a, b = rows[ia][1], rows[ib][1]
        ta, tb = hvac_total(a), hvac_total(b)
        add_q(f"T1-C{j+1:02d}", "in_dist", "compare", None, q_compare(nq(), params_of(a), params_of(b)),
              None,
              {"value": round(abs(ta - tb), 1), "unit": "kWh", "choice": "A" if ta < tb else "B",
               "source": f"office_sample_summary.csv#sample_id={int(a['sample_id'])},{int(b['sample_id'])}",
               "status": "ok"},
              scoring_of("input_diff"), "trust",
              params_a=params_of(a), params_b=params_of(b))

    # S×4：设定点节能（基准=留出样本，变体补跑）
    sv_cool = take(interior[interior["cooling_setpoint_c"] <= 27.0], 3)
    sv_heat = take(interior[interior["heating_setpoint_c"] >= 17.0], 1)
    for j, (_, row) in enumerate(pd.concat([sv_cool, sv_heat]).iterrows()):
        is_cool = j < 3
        param = "cooling_setpoint_c" if is_cool else "heating_setpoint_c"
        v0 = float(row[param]); v1 = v0 + 1.0 if is_cool else v0 - 1.0
        vp = params_of(row); vp[param] = v1
        qid = f"T1-S{j+1:02d}"
        jid = new_job("variant", qid, vp, f"节能对照变体：{param} {v0:g}->{v1:g}，基准=sample_id {int(row['sample_id'])}")
        chan = "cooling_input" if is_cool else "heating_input"
        add_q(qid, "in_dist", "saving", None, q_saving(nq(), params_of(row), param, v0, v1),
              params_of(row),
              {"value": None, "unit": "kWh", "status": "pending",
               "source": f"office_sample_summary.csv#sample_id={int(row['sample_id'])} − truth_runs/extra_truth.csv#job_id={jid}",
               "base_sample_id": int(row["sample_id"]), "base_channel": TARGET_COL[chan]},
              scoring_of(f"{chan}_saving"), "trust",
              change={"param": param, "from": v0, "to": v1})

    # ---------------- T2 分布边缘（15）----------------
    # P×4：界内贴边
    t2_targets = ["total_input", "total_input", "cooling_input", "heating_input"]
    for j, (_, row) in enumerate(take(near_edge, 4).iterrows()):
        t = t2_targets[j]
        add_q(f"T2-P{j+1:02d}", "edge", "point", None, q_point(nq(), params_of(row), t),
              params_of(row), truth_from_row(row, t), scoring_of(t), "trust")

    # P×4：单维轻度越界（定制，补跑）
    mild_overrides = [("window_u_w_m2k", 6.00, "窗 U 越上界 5.8"),
                      ("infiltration_ach", 0.90, "渗透越上界 0.8"),
                      ("window_wall_ratio", 0.64, "窗墙比越上界 0.6"),
                      ("window_shgc", 0.58, "SHGC 越上界 0.55")]
    mild_bases = take(interior, 4)
    for j, ((_, row), (pcol, v, note)) in enumerate(zip(mild_bases.iterrows(), mild_overrides)):
        p = params_of(row); p[pcol] = v
        qid = f"T2-P{5+j:02d}"
        jid = new_job("mild_ood", qid, p, f"单维轻度越界：{note}")
        add_q(qid, "edge", "point", "design_mild", q_point(nq(), p, "total_input"), p,
              {"value": None, "unit": "kWh", "source": f"truth_runs/extra_truth.csv#job_id={jid}",
               "status": "pending"}, scoring_of("total_input"), "transition")

    # C×4：贴边方案比较
    cmp2 = take(near_edge, 8)
    rows2 = list(cmp2.iterrows())
    cand2 = sorted(((rel(a, b), ia, ib) for ia, (_, a) in enumerate(rows2) for ib, (_, b) in enumerate(rows2)
                    if ia < ib for a, b in [(a, b)]), reverse=True)
    pairs2, used2 = [], set()
    for r, ia, ib in cand2:
        if len(pairs2) >= 2 or ia in used2 or ib in used2:
            continue
        pairs2.append((ia, ib)); used2.update((ia, ib))
    for r, ia, ib in reversed(cand2):
        if len(pairs2) >= 4 or ia in used2 or ib in used2:
            continue
        pairs2.append((ia, ib)); used2.update((ia, ib))
    for j, (ia, ib) in enumerate(pairs2):
        a, b = rows2[ia][1], rows2[ib][1]
        ta, tb = hvac_total(a), hvac_total(b)
        add_q(f"T2-C{j+1:02d}", "edge", "compare", None, q_compare(nq(), params_of(a), params_of(b)),
              None,
              {"value": round(abs(ta - tb), 1), "unit": "kWh", "choice": "A" if ta < tb else "B",
               "source": f"office_sample_summary.csv#sample_id={int(a['sample_id'])},{int(b['sample_id'])}",
               "status": "ok"},
              scoring_of("input_diff"), "trust",
              params_a=params_of(a), params_b=params_of(b))

    # S×2：贴边设定点节能（变体保持界内，补跑）
    sv2 = take(near_edge[near_edge["cooling_setpoint_c"] <= 27.0], 2)
    for j, (_, row) in enumerate(sv2.iterrows()):
        v0 = float(row["cooling_setpoint_c"]); v1 = v0 + 1.0
        vp = params_of(row); vp["cooling_setpoint_c"] = v1
        qid = f"T2-S{j+1:02d}"
        jid = new_job("variant", qid, vp, f"节能对照变体（贴边）：cooling {v0:g}->{v1:g}，基准=sample_id {int(row['sample_id'])}")
        add_q(qid, "edge", "saving", None, q_saving(nq(), params_of(row), "cooling_setpoint_c", v0, v1),
              params_of(row),
              {"value": None, "unit": "kWh", "status": "pending",
               "source": f"office_sample_summary.csv#sample_id={int(row['sample_id'])} − truth_runs/extra_truth.csv#job_id={jid}",
               "base_sample_id": int(row["sample_id"]), "base_channel": TARGET_COL["cooling_input"]},
              scoring_of("cooling_input_saving"), "trust",
              change={"param": "cooling_setpoint_c", "from": v0, "to": v1})

    # X×1：老化情景（单维轻度越界，补跑）
    age_base = take(interior, 1).iloc[0]
    p = params_of(age_base); p["infiltration_ach"] = 0.95
    jid = new_job("mild_ood", "T2-X01", p, "老化情景：渗透 0.95 ACH（越上界 0.8）")
    add_q("T2-X01", "edge", "scenario", "design_mild",
          f"南京一栋 15 层高层办公楼（约 19440 m²）建成已约 20 年，近期实测围护气密性明显退化，渗透率约 0.95 ACH。"
          f"其余参数：{desc_design(p, skip={'infiltration_ach'})}。按南京典型气象年，这栋楼目前全年空调总电耗"
          f"（供冷+供暖输入电耗合计，kWh）大约是多少？",
          p,
          {"value": None, "unit": "kWh", "source": f"truth_runs/extra_truth.csv#job_id={jid}", "status": "pending"},
          scoring_of("total_input"), "transition")

    # ---------------- T3 OOD（10）----------------
    # P×5：重度设计越界（真值现成）
    severe_pool = ood.sort_values("_viol", ascending=False).head(20).copy()
    sev = severe_pool.loc[greedy_maximin(severe_pool, 5)]
    t3_targets = ["total_input", "total_input", "heating_input", "cooling_input", "total_input"]
    for j, (_, row) in enumerate(sev.iterrows()):
        t = t3_targets[j]
        add_q(f"T3-P{j+1:02d}", "ood", "point", "design_severe", q_point(nq(), params_of(row), t),
              params_of(row), truth_from_row(row, t), scoring_of(t), "fallback")

    # C×2：重度越界方案 vs 界内方案
    sev2 = severe_pool.loc[greedy_maximin(severe_pool[~severe_pool["sample_id"].isin(sev["sample_id"])], 2)]
    ref2 = take(interior, 2)
    for j, ((_, a), (_, b)) in enumerate(zip(sev2.iterrows(), ref2.iterrows())):
        ta, tb = hvac_total(a), hvac_total(b)
        add_q(f"T3-C{j+1:02d}", "ood", "compare", "design_severe",
              q_compare(nq(), params_of(a), params_of(b)), None,
              {"value": round(abs(ta - tb), 1), "unit": "kWh", "choice": "A" if ta < tb else "B",
               "source": f"office_ood_dataset.csv#sample_id={int(a['sample_id'])} vs office_sample_summary.csv#sample_id={int(b['sample_id'])}",
               "status": "ok"},
              scoring_of("input_diff"), "fallback",
              params_a=params_of(a), params_b=params_of(b))

    # X×3：异常年情景（真值=负荷/COP）
    anom_plan = [("anom_balanced", 3, "cooling"), ("anom_warm", 11, "cooling"), ("anom_cold", 17, "heating")]
    for j, (scen, b_idx, chan) in enumerate(anom_plan):
        st = anom_stats.loc[scen]
        bp = buildings[b_idx]
        p = {col(k): round(float(bp[col(k)] if col(k) in bp else bp[k]), 4) for k in VAR_ORDER}
        row = wt[(wt["scenario"] == scen) & (wt["building_idx"] == b_idx)].iloc[0]
        cop = p["cooling_cop"] if chan == "cooling" else p["heating_cop"]
        load = float(row[f"annual_{chan}_load_kwh"])
        mean_dt = float(st["annual_mean_dT"])
        mean_desc = f"高约 {mean_dt:.1f} °C" if mean_dt > 0.5 else (f"低约 {abs(mean_dt):.1f} °C" if mean_dt < -0.5 else f"基本持平（约 {mean_dt:+.2f} °C）")
        text = (
            f"南京一栋 15 层高层办公楼（约 19440 m²），参数：{desc_design(p)}。"
            f"假设明年是一个异常气候年：年平均气温较典型年{mean_desc}，"
            f"但夏季出现峰值约 +{float(st['peak_hot_dT']):.1f} °C 的持续热浪、冬季出现约 {float(st['peak_cold_dT']):.1f} °C 的强冷潮"
            f"（全年约 {int(st['days_above_base_max'])} 天日均温超出典型年最高日均、约 {int(st['days_below_base_min'])} 天低于最低日均）。"
            f"这种年份下，这栋楼{'全年供冷' if chan == 'cooling' else '全年供暖'}输入电耗（kWh）大约是多少？"
        )
        add_q(f"T3-X{j+1:02d}", "ood", "scenario", "anom_year", text, p,
              {"value": round(load / cop, 1), "unit": "kWh",
               "source": f"office_weather_ood_truth.csv#scenario={scen},building_idx={b_idx}（负荷/COP={cop:g}）",
               "status": "ok"},
              scoring_of(f"{chan}_input"), "fallback",
              scenario=scen)

    # ---------------- 绑定补跑真值（extra_truth.csv 就绪时）----------------
    extra = None
    if EXTRA_TRUTH_CSV.exists():
        extra = pd.read_csv(EXTRA_TRUTH_CSV, encoding="utf-8-sig").set_index("job_id")
        sid_map = summary.set_index("sample_id")
        for q in questions:
            tr = q["truth"]
            if tr.get("status") != "pending":
                continue
            jid = next(j["job_id"] for j in jobs if j["qid"] == q["id"])
            if jid not in extra.index or extra.loc[jid, "simulation_status"] != "ok":
                continue
            er = extra.loc[jid]
            if q["qtype"] == "saving":
                base_v = float(sid_map.loc[tr["base_sample_id"], tr["base_channel"]])
                tr["value"] = round(base_v - float(er[tr["base_channel"]]), 1)
            else:
                tgt = q["scoring"]["target"]
                v = (float(er["annual_heating_input_kwh"]) + float(er["annual_cooling_input_kwh"])
                     ) if tgt == "total_input" else float(er[TARGET_COL[tgt]])
                tr["value"] = round(v, 1)
            tr["status"] = "ok"

    # ---------------- 校验与输出 ----------------
    assert len(questions) == 45, f"题数 {len(questions)} != 45"
    assert len(jobs) == 14, f"补跑任务 {len(jobs)} != 14"
    tiers = pd.Series([q["tier"] for q in questions]).value_counts().to_dict()
    assert tiers == {"in_dist": 20, "edge": 15, "ood": 10}, tiers

    EXTRA_JOBS_CSV.parent.mkdir(exist_ok=True)
    pd.DataFrame(jobs).to_csv(EXTRA_JOBS_CSV, index=False, encoding="utf-8-sig")
    with open(HERE / "questions_v1.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(questions, f, allow_unicode=True, sort_keys=False, width=110)

    # 选点报告
    pend = sum(1 for q in questions if q["truth"].get("status") == "pending")
    lines = [
        "# 题集 v1 选点报告（自动生成，勿手改）", "",
        f"- 生成种子：{SEED}；生成时间见 git/文件时间戳",
        f"- 池子：留出集 {len(val)}（内部 {len(interior)}，贴边 {len(near_edge)}，贴边判据：{edge_rule}）；"
        f"设计 OOD {len(ood)}；异常年真值 {len(wt)} 行",
        f"- 题数：45（T1×20 / T2×15 / T3×10）；真值待补跑：{pend} 题（{len(jobs)} 个 EnergyPlus 任务）",
        f"- 已用留出样本 {len(used_val)} 个（题间不重复）", "",
        "| id | tier | qtype | ood_axis | 判分目标 | 真值状态 | expected_gate |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for q in questions:
        lines.append(f"| {q['id']} | {q['tier']} | {q['qtype']} | {q.get('ood_axis') or '-'} | "
                     f"{q['scoring']['target']} | {q['truth'].get('status')} | {q['expected_gate']} |")
    (HERE / "selection_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"questions_v1.yaml: 45 题（pending {pend}）")
    print(f"extra_jobs.csv: {len(jobs)} 个补跑任务")
    if extra is not None:
        print("已绑定补跑真值。")


if __name__ == "__main__":
    main()
