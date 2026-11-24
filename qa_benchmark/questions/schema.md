# 题目 Schema 与出题纪律

## 字段定义（questions_v1.yaml）

```yaml
- id: T1-P01              # 档-题型-序号；P=点预测 C=方案比较 S=节能量 X=情景估计
  tier: in_dist           # in_dist | edge | ood
  qtype: point            # point | compare | saving | scenario
  ood_axis: null          # null | design_mild | design_severe | anom_year
  question_zh: "……"       # 完整题面（自然语言，含全部 10 维参数）
  params: {...}           # 方案参数（10 维；compare 题为 params_a / params_b）
  change: {...}           # 仅 saving 题：{param, from, to}
  scenario: null          # 仅 anom 题：anom_warm | anom_balanced | anom_cold
  truth:
    value: 123456.7       # 真值（pending 时为 null）
    unit: kWh
    source: "office_sample_summary.csv#sample_id=812"
    status: ok            # ok | pending（等待补跑）
    choice: null          # 仅 compare 题：A | B（更省电的一方）
  scoring:
    target: total_input   # total_input | cooling_input | heating_input | input_diff | *_saving
    correct_pct: 10
    acceptable_pct: 25
    abs_floor_correct_kwh: 1000     # 仅 saving 题：|err|≤floor 亦判 correct（防小真值病态）
    abs_floor_acceptable_kwh: 2500  # 仅 saving 题：|err|≤floor 亦判 acceptable
  expected_gate: trust    # trust | transition | fallback（对 C 框架门控行为的预期，仅用于事后核对，不参与判分）
```

## 出题纪律

1. **题面不可泄露分档**：不出现"训练范围 / OOD / 边缘 / 极端参数"等提示词；三档题措辞风格一致（同一组模板轮换），防止 LLM 靠措辞猜"该不该拒答"。
2. **参数必须齐全**：题面写全 10 维设计参数（外墙/屋面保温、窗 U、SHGC、渗透、窗墙比、冷/热设定点、冷/热 COP），避免默认值歧义。
3. **判分目标只用能耗（输入电耗）**：total / cooling / heating input 及其差值。不舒适小时不入基准（好方案真值近零，相对误差病态），留给场景演示。
4. **真值一律 EnergyPlus**：来源三类——`office_sample_summary.csv`（留出集现成）、`office_ood_dataset.csv`（93 个设计 OOD 点现成）、`office_weather_ood_truth.csv`（异常年现成，负荷→电耗按该楼 COP 换算）；新点/对照变体由 `../truth_runs/run_extra_truth.py` 补跑。
5. **compare 题**问"哪个方案全年空调总电耗更低？低多少 kWh？"——判分先看方向（choice 错且 confident 直接判幻觉），再看差值误差。
6. **saving 题**基准侧用留出集样本（真值现成），只补跑变体侧；真值 = 对应通道输入电耗之差。
7. **anom 题**用气候统计描述情景（年均偏移、热浪/冷潮峰值、超阈天数），不出现内部情景代号；建筑取自 `generate_indist_buildings(20, seed=909)` 的确定性复原。
8. **可复现**：`make_questions.py` 固定随机种子（20260702），任何人重跑得到同一题集。
9. **节能量题的绝对容差层**：节能量真值可能天然很小（南京办公供暖侧 −1 °C 仅省数百 kWh），纯相对误差判分病态；故 saving 题在相对阈值外增加绝对容差（correct ≤1000 kWh / acceptable ≤2500 kWh，约为典型 HVAC 年电耗的 0.3%/0.8%）——这也让"省得很少"这类正确的量级判断得到分数。
