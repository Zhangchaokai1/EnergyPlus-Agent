import { useEffect, useMemo, useState } from 'react';
import {
  ArrowLeft,
  BarChart3,
  Bot,
  BrainCircuit,
  Building2,
  CloudSun,
  Fan,
  FileText,
  Gauge,
  Loader2,
  Settings2,
  SlidersHorizontal,
  Sparkles,
  ThermometerSun,
  Zap,
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import BuildingScene from './components/BuildingScene.jsx';
import AgentPage from './AgentPage.jsx';
import TrainingPage from './TrainingPage.jsx';
import { api } from './api/client.js';

const COOLING_SETPOINT_MIN = 24;
const COOLING_SETPOINT_MAX = 28;
const HEATING_SETPOINT_MIN = 16;
const HEATING_SETPOINT_MAX = 20;

const defaultInputs = {
  wall_insulation_m: 0.02,
  roof_insulation_m: 0.02,
  window_u_w_m2k: 3.5,
  window_shgc: 0.5,
  infiltration_ach: 0.8,
  window_wall_ratio: 0.4,
  cooling_setpoint_c: 26,
  heating_setpoint_c: 20,
  cooling_cop: 3,
  heating_cop: 1.8,
};

const designControls = [
  ['wall_insulation_m', '外墙保温', 'm', 0.02, 0.12, 0.005],
  ['roof_insulation_m', '屋面保温', 'm', 0.02, 0.15, 0.005],
  ['window_u_w_m2k', '窗 U 值', 'W/m2K', 1.4, 5.8, 0.05],
  ['window_shgc', 'SHGC', '', 0.3, 0.55, 0.01],
  ['infiltration_ach', '渗透 ACH', '1/h', 0.3, 0.8, 0.02],
  ['window_wall_ratio', '窗墙比', '', 0.3, 0.6, 0.01],
  ['cooling_setpoint_c', '供冷设定温度', '°C', COOLING_SETPOINT_MIN, COOLING_SETPOINT_MAX, 0.5],
  ['heating_setpoint_c', '供暖设定温度', '°C', HEATING_SETPOINT_MIN, HEATING_SETPOINT_MAX, 0.5],
  ['cooling_cop', '制冷 COP', '', 3, 6, 0.1],
  ['heating_cop', '制热 COP', '', 1.8, 4, 0.1],
];

const SU7_PRO_BATTERY_KWH = 94.3;
const SU7_PRO_CLTC_KM = 830;
const NANJING_BEIJING_ONE_WAY_KM = 1019.25;

const scenes = [
  {
    id: 'design',
    icon: Building2,
    title: '设计/建设期',
    subtitle: '快速获得年冷热负荷、等效不舒适小时与三目标 Pareto 前沿。',
    stats: '围护结构 · 设定点 · 系统效率',
  },
  {
    id: 'operation',
    icon: Fan,
    title: '运维期',
    subtitle: '用于调整空调设定点，评估全年曲线和节能效果。',
    stats: '设定温度 · 调控策略 · 节能量',
  },
  {
    id: 'extreme',
    icon: CloudSun,
    title: '极端天气峰值',
    subtitle: '对比正常气候日平均能耗与异常年 OOD 情景下的日峰值。',
    stats: '异常暖年 · 均衡异常年 · 异常冷年',
  },
  {
    id: 'features',
    icon: FileText,
    title: '模型输入特征',
    subtitle: '展示代理模型使用的气候与建筑输入变量，说明各特征含义、范围和可视化趋势。',
    stats: '气候特征 · 建筑参数 · 输入范围',
  },
];

const principleEntry = {
  id: 'principles',
  title: '原理文档',
  subtitle: '热平衡法 · PMV · COP',
};

function fmtKwh(value) {
  if (!Number.isFinite(value)) return '-';
  if (Math.abs(value) >= 10000) return `${(value / 10000).toFixed(1)} 万kWh`;
  return `${value.toFixed(0)} kWh`;
}

function fmtMoney(value) {
  if (!Number.isFinite(value)) return '-';
  if (Math.abs(value) >= 10000) return `${(value / 10000).toFixed(1)} 万元`;
  return `${value.toFixed(0)} 元`;
}

function fmtHours(value) {
  if (!Number.isFinite(value)) return '-';
  return `${value.toFixed(1)} h`;
}

function chartDaily(data = []) {
  return data.map((day) => ({
    label: day.label,
    month: day.month,
    heating: Number(day.heatingInputKwh.toFixed(1)),
    cooling: Number(day.coolingInputKwh.toFixed(1)),
    total: Number(day.totalInputKwh.toFixed(1)),
    discomfort: Number(day.discomfortHours.toFixed(2)),
    drybulb: Number(day.drybulbMeanC.toFixed(1)),
  }));
}

function discomfortColor(hours) {
  if (!Number.isFinite(hours)) return 'rgba(30,64,175,.28)';
  if (hours < 20) return '#2563eb';
  if (hours < 100) return '#f59e0b';
  return '#dc2626';
}

function Metric({ icon: Icon, label, value, tone = 'neutral' }) {
  return (
    <div className={`metric ${tone}`}>
      <Icon size={18} />
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function Slider({ item, value, onChange }) {
  const [key, label, unit, min, max, step] = item;
  return (
    <label className="control-row">
      <span>
        <b>{label}</b>
        <em>{value}{unit ? ` ${unit}` : ''}</em>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(key, Number(event.target.value))}
      />
    </label>
  );
}

function StatusPill({ health, loading }) {
  return (
    <span className={`status-pill ${health?.status === 'ok' ? 'ok' : ''}`}>
      {loading ? <Loader2 size={14} className="spin" /> : <Zap size={14} />}
      {health?.status === 'ok' ? `${health.members} 成员 · ${health.device}` : '本地代理模型'}
    </span>
  );
}

function Home({ onEnter, health, loading }) {
  return (
    <main className="home">
      <section className="home-copy">
        <h1>物理增强Agent：用于建筑能耗预测与方案优化</h1>
        <p className="home-tagline">一个物理可信、会自省的建筑能耗决策智能体（以南京高层办公楼为例）</p>
        <p>
          核心方法：RAG → PAG（Physics-Augmented Generation，物理增强生成）。基于真实物理仿真结果进行方案优化与决策建议。
        </p>
      </section>
      <section className="scene-grid">
        <button className="agent-hero" onClick={() => onEnter('agent')}>
          <span className="agent-hero-icon"><Bot size={28} /></span>
          <div>
            <strong>EnergyPlus-Agent 智能问答</strong>
            <p>用自然语言提问，智能体调用物理代理模型与不确定性门控，必要时回退 EnergyPlus 仿真。</p>
            <em><Sparkles size={14} /> 物理增强生成 · 不确定性门控 · 知之为知之，不知则仿真</em>
          </div>
        </button>
        <button className="training-hero" onClick={() => onEnter('training')}>
          <span className="training-hero-icon"><BrainCircuit size={28} /></span>
          <div>
            <strong>自然语言训练工作台</strong>
            <p>上传建筑图片，确认矩形多层建筑参数，自动生成 IDF、启动 EnergyPlus 数据集与专属代理模型训练。</p>
            <em><Sparkles size={14} /> 图片识别 · 参数确认 · 模型库切换</em>
          </div>
        </button>
        {scenes.map((scene) => {
          const Icon = scene.icon;
          return (
            <button key={scene.id} className={`scene-card ${scene.id}`} onClick={() => onEnter(scene.id)}>
              <span className="scene-icon"><Icon size={28} /></span>
              <strong>{scene.title}</strong>
              <p>{scene.subtitle}</p>
              <em>{scene.stats}</em>
            </button>
          );
        })}
        <button className="principle-entry" onClick={() => onEnter(principleEntry.id)}>
          <FileText size={20} />
          <span>
            <strong>{principleEntry.title}</strong>
            <em>{principleEntry.subtitle}</em>
          </span>
        </button>
      </section>
    </main>
  );
}

function Shell({ mode, title, subtitle, onBack, children, right }) {
  return (
    <main className={`workspace ${mode}`}>
      <header className="topbar">
        <button className="icon-button" onClick={onBack} aria-label="返回">
          <ArrowLeft size={20} />
        </button>
        <div>
          <h1>{title}</h1>
          <p>{subtitle}</p>
        </div>
      </header>
      <section className="workspace-layout">
        <aside className="panel">{children}</aside>
        <section className="model-panel">
          <BuildingScene mode={mode} />
        </section>
        <section className="analysis-panel">{right}</section>
      </section>
    </main>
  );
}

function DesignPage({ onBack }) {
  const [inputs, setInputs] = useState(defaultInputs);
  const [prediction, setPrediction] = useState(null);
  const [pareto, setPareto] = useState(null);
  const [candidateCount, setCandidateCount] = useState(120);
  const [loading, setLoading] = useState(false);
  const [paretoLoading, setParetoLoading] = useState(false);
  const [error, setError] = useState('');

  const runPrediction = async (payload = inputs) => {
    setLoading(true);
    setError('');
    try {
      setPrediction(await api.designPredict(payload));
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const runPareto = async () => {
    setParetoLoading(true);
    setError('');
    try {
      setPareto(await api.designPareto({ count: candidateCount, seed: 2026, lock_setpoints: false }));
    } catch (err) {
      setError(err.message);
    } finally {
      setParetoLoading(false);
    }
  };

  useEffect(() => {
    runPrediction(defaultInputs);
    runPareto();
  }, []);

  const setValue = (key, value) => {
    const next = { ...inputs, [key]: value };
    setInputs(next);
  };

  const daily = useMemo(() => chartDaily(prediction?.daily || []), [prediction]);
  const scatter = useMemo(() => (pareto?.points || []).map((point) => ({
    ...point,
    x: point.retrofitCostCny / 10000,
    y: point.annualRobustTotalInputKwh / 10000,
    z: point.annualDiscomfortHours,
  })), [pareto]);

  return (
    <Shell
      mode="design"
      title="工程师设计工作台"
      subtitle="南京气候固定，输入建筑参数，由代理模型输出年能耗、等效不舒适小时与三目标 Pareto 前沿。"
      onBack={onBack}
      right={(
        <>
          <div className="metrics-row">
            <Metric icon={ThermometerSun} label="年供暖输入" value={fmtKwh(prediction?.annual?.heatingInputKwh)} tone="warm" />
            <Metric icon={Gauge} label="年供冷输入" value={fmtKwh(prediction?.annual?.coolingInputKwh)} tone="cool" />
            <Metric icon={Zap} label="冷热总输入" value={fmtKwh(prediction?.annual?.totalInputKwh)} tone="energy" />
            <Metric icon={Fan} label="年等效不舒适" value={fmtHours(prediction?.annual?.discomfortHours)} tone="comfort" />
          </div>
          <div className="chart-box tall">
            <h2>全年冷热能耗曲线</h2>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={daily} margin={{ top: 10, right: 20, left: 0, bottom: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(30,41,59,.15)" />
                <XAxis dataKey="label" minTickGap={28} />
                <YAxis />
                <Tooltip formatter={(v) => [`${v} kWh`, '']} />
                <Legend />
                <Line type="monotone" dataKey="heating" name="供暖输入" stroke="#d97706" dot={false} strokeWidth={2} />
                <Line type="monotone" dataKey="cooling" name="供冷输入" stroke="#0284c7" dot={false} strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <div className="chart-box">
            <div className="chart-title-row">
              <h2>三目标 Pareto 前沿</h2>
              <span>成本 · 稳健冷热输入 · 等效不舒适 | {pareto?.paretoCount || 0}/{pareto?.count || 0} 个非支配方案</span>
            </div>
            <div className="comfort-legend">
              <span><i className="low" />舒适风险低</span>
              <span><i className="mid" />中等</span>
              <span><i className="high" />较高</span>
            </div>
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 14, right: 20, bottom: 14, left: 34 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(30,41,59,.15)" />
                <XAxis type="number" dataKey="x" name="改造成本" unit="万元" />
                <YAxis type="number" dataKey="y" name="稳健冷热输入" unit="万kWh" width={96} />
                <Tooltip
                  cursor={{ strokeDasharray: '3 3' }}
                  formatter={(value, name) => [Number(value).toFixed(1), name]}
                  content={({ active, payload }) => {
                    if (!active || !payload?.length) return null;
                    const p = payload[0].payload;
                    return (
                      <div className="tooltip">
                        <b>{p.pareto ? 'Pareto 方案' : '候选方案'} #{p.id}</b>
                        <span>稳健冷热输入 {fmtKwh(p.annualRobustTotalInputKwh)}</span>
                        <span>模型 σ {fmtKwh(p.sigmaTotalInputKwh)}</span>
                        <span>等效不舒适 {fmtHours(p.annualDiscomfortHours)}</span>
                        <span>相对基线节能 {p.savingsVsBaselinePct.toFixed(2)}%</span>
                        <span>成本 {fmtMoney(p.retrofitCostCny)}</span>
                        <span>设定 {p.inputs.cooling_setpoint_c.toFixed(1)} / {p.inputs.heating_setpoint_c.toFixed(1)} °C</span>
                        <span>WWR {p.inputs.window_wall_ratio.toFixed(2)} · COP {p.inputs.cooling_cop.toFixed(1)}</span>
                      </div>
                    );
                  }}
                />
                <Scatter data={scatter} name="候选方案">
                  {scatter.map((entry) => (
                    <Cell
                      key={entry.id}
                      fill={discomfortColor(entry.annualDiscomfortHours)}
                      fillOpacity={entry.pareto ? 0.95 : 0.28}
                      stroke={entry.pareto ? '#0f172a' : 'transparent'}
                      strokeWidth={entry.pareto ? 1.2 : 0}
                    />
                  ))}
                </Scatter>
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </>
      )}
    >
      <div className="panel-heading">
        <Settings2 size={20} />
        <div>
          <h2>建筑参数</h2>
          <p>气候数据为南京市EPW</p>
        </div>
      </div>
      <div className="control-list">
        {designControls.map((item) => (
          <Slider key={item[0]} item={item} value={inputs[item[0]]} onChange={setValue} />
        ))}
      </div>
      <button className="primary-button" onClick={() => runPrediction()} disabled={loading}>
        {loading ? <Loader2 className="spin" size={18} /> : <BarChart3 size={18} />}
        更新能耗预测
      </button>
      <div className="pareto-tools">
        <label>
          候选方案数量
          <input type="number" min="20" max="800" value={candidateCount} onChange={(e) => setCandidateCount(Number(e.target.value))} />
        </label>
        <button className="secondary-button" onClick={runPareto} disabled={paretoLoading}>
          {paretoLoading ? <Loader2 className="spin" size={18} /> : <SlidersHorizontal size={18} />}
          重新优化
        </button>
      </div>
      {error && <p className="error-text">{error}</p>}
    </Shell>
  );
}

function OperationPage({ onBack }) {
  const [inputs, setInputs] = useState({ cooling_setpoint_c: 26, heating_setpoint_c: 20 });
  const [prediction, setPrediction] = useState(null);
  const [saving, setSaving] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const run = async (payload = inputs) => {
    setLoading(true);
    setError('');
    try {
      const [pred, saved] = await Promise.all([api.operationPredict(payload), api.operationSaving(payload)]);
      setPrediction(pred);
      setSaving(saved.saving);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    run(inputs);
  }, []);

  const daily = useMemo(() => chartDaily(prediction?.daily || []), [prediction]);
  const mobility = useMemo(() => {
    const savedKwh = saving?.summerCoolingInputSavedKwh;
    if (!Number.isFinite(savedKwh)) {
      return { fullCharges: null, roundTrips: null };
    }
    const fullCharges = savedKwh / SU7_PRO_BATTERY_KWH;
    const equivalentKm = fullCharges * SU7_PRO_CLTC_KM;
    const roundTrips = equivalentKm / (NANJING_BEIJING_ONE_WAY_KM * 2);
    return { fullCharges, roundTrips };
  }, [saving]);

  return (
    <Shell
      mode="operation"
      title="运维调控驾驶舱"
      subtitle="建筑实体固定，仅调整新模型训练范围内的空调设定温度，评估全年冷热输入和升温一度节能收益。"
      onBack={onBack}
      right={(
        <>
          <div className="metrics-row">
            <Metric icon={Gauge} label="供冷输入" value={fmtKwh(prediction?.annual?.coolingInputKwh)} tone="cool" />
            <Metric icon={ThermometerSun} label="供暖输入" value={fmtKwh(prediction?.annual?.heatingInputKwh)} tone="warm" />
            <Metric icon={Zap} label="夏季升温一度可省" value={fmtKwh(saving?.summerCoolingInputSavedKwh)} tone="energy" />
          </div>
          <div className="chart-box tall">
            <h2>全年运维能耗曲线</h2>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={daily} margin={{ top: 10, right: 20, left: 0, bottom: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(30,41,59,.15)" />
                <XAxis dataKey="label" minTickGap={28} />
                <YAxis />
                <Tooltip formatter={(v) => [`${v} kWh`, '']} />
                <Legend />
                <Line type="monotone" dataKey="heating" name="供暖输入" stroke="#d97706" dot={false} strokeWidth={2} />
                <Line type="monotone" dataKey="cooling" name="供冷输入" stroke="#059669" dot={false} strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <div className="compare-box">
            <div>
              <span>{inputs.cooling_setpoint_c >= COOLING_SETPOINT_MAX ? '供冷设定值已到训练上限' : '把夏季供冷设定值提高 1°C'}</span>
              <strong>{inputs.cooling_setpoint_c >= COOLING_SETPOINT_MAX ? '上限' : `${saving?.summerCoolingSavingPct?.toFixed(2) || '-'}%`}</strong>
            </div>
            <p>
              {inputs.cooling_setpoint_c >= COOLING_SETPOINT_MAX
                ? `当前供冷设定温度已经达到 ${COOLING_SETPOINT_MAX} °C，展示端不再外推到训练范围之外。`
                : `按当前建筑与南京气候估算，夏季供冷输入电耗减少 ${fmtKwh(saving?.summerCoolingInputSavedKwh)}，全年供冷输入电耗减少 ${fmtKwh(saving?.annualCoolingInputSavedKwh)}。`}
            </p>
          </div>
          <div className="mobility-card">
            <div className="mobility-visual">
              <img src="/assets/su7_19.png" alt="小米 SU7 Pro" />
            </div>
            <div className="mobility-copy">
              <span>这些电量相当于</span>
              <div className="mobility-stats">
                <div>
                  <strong>{mobility.fullCharges === null ? '-' : mobility.fullCharges.toFixed(1)}</strong>
                  <em>约可充满 SU7 Pro</em>
                </div>
                <div>
                  <strong>{mobility.roundTrips === null ? '-' : mobility.roundTrips.toFixed(1)}</strong>
                  <em>约等于南京-北京往返</em>
                </div>
              </div>
              <p>按 SU7 Pro 94.3 kWh 电池、CLTC 830 km 续航、南京到北京单程约 1019 km 估算。</p>
            </div>
          </div>
        </>
      )}
    >
      <div className="panel-heading">
        <Fan size={20} />
        <div>
          <h2>空调设定</h2>
          <p>围护结构和系统效率按基准办公楼固定。</p>
        </div>
      </div>
      <Slider
        item={['cooling_setpoint_c', '供冷设定温度', '°C', COOLING_SETPOINT_MIN, COOLING_SETPOINT_MAX, 0.5]}
        value={inputs.cooling_setpoint_c}
        onChange={(key, value) => setInputs({ ...inputs, [key]: value })}
      />
      <Slider
        item={['heating_setpoint_c', '供暖设定温度', '°C', 16, 20, 0.5]}
        value={inputs.heating_setpoint_c}
        onChange={(key, value) => setInputs({ ...inputs, [key]: value })}
      />
      <button className="primary-button green" onClick={() => run()} disabled={loading}>
        {loading ? <Loader2 className="spin" size={18} /> : <Gauge size={18} />}
        更新运维预测
      </button>
      {error && <p className="error-text">{error}</p>}
    </Shell>
  );
}

function ExtremePage({ onBack, metadata }) {
  const scenarioOptions = (metadata?.weatherScenarios || ['base', 'anom_warm']).filter((s) => s !== 'base');
  const [payload, setPayload] = useState({ scenario: scenarioOptions[0] || 'anom_warm', cooling_setpoint_c: 26, heating_setpoint_c: 20 });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const run = async (next = payload) => {
    setLoading(true);
    setError('');
    try {
      setResult(await api.extremeEstimate(next));
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (payload.scenario) run(payload);
  }, [payload.scenario]);

  const bars = useMemo(() => {
    if (!result) return [];
    return [
      { name: '正常气候日均', value: Math.round(result.normal.averageDailyTotalInputKwh), fill: '#64748b' },
      { name: '极端日峰值', value: Math.round(result.extreme.peakDay.totalInputKwh), fill: '#d9480f' },
    ];
  }, [result]);

  return (
    <Shell
      mode="extreme"
      title="异常年 OOD 能耗峰值估计"
      subtitle="左侧基准是正常气候日平均能耗，右侧选择异常年 OOD 情景并估计日峰值增长。"
      onBack={onBack}
      right={(
        <>
          <div className="metrics-row">
            <Metric icon={Gauge} label="正常气候日均" value={fmtKwh(result?.normal?.averageDailyTotalInputKwh)} />
            <Metric icon={ThermometerSun} label="极端峰值日" value={fmtKwh(result?.extreme?.peakDay?.totalInputKwh)} tone="warm" />
            <Metric icon={Zap} label="峰值增长" value={`${result?.extreme?.increasePctVsNormalAverage?.toFixed(1) || '-'}%`} tone="energy" />
          </div>
          <div className="chart-box tall">
            <h2>正常日均 vs 极端天气日峰值</h2>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={bars} margin={{ top: 20, right: 24, bottom: 20, left: 8 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(30,41,59,.15)" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip formatter={(v) => [`${v} kWh`, '日总输入']} />
                <Bar dataKey="value" radius={[6, 6, 0, 0]}>
                  {bars.map((entry) => <Cell key={entry.name} fill={entry.fill} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="compare-box extreme-text">
            <div>
              <span>{result?.scenario || payload.scenario}</span>
              <strong>{result?.extreme?.peakDay?.label || '-'}</strong>
            </div>
            <p>该异常年 OOD 情景下，峰值日相比正常气候日平均冷热输入电耗增加 {result?.extreme?.increasePctVsNormalAverage?.toFixed(1) || '-'}%。异常年全年日均变化为 {result?.extreme?.averageIncreasePct?.toFixed(1) || '-'}%。</p>
          </div>
        </>
      )}
    >
      <div className="panel-heading">
        <CloudSun size={20} />
        <div>
          <h2>极端气候参数</h2>
          <p>选择现有异常年 OOD EPW 情景，后端实时提取 HDH16 等逐日天气特征。</p>
        </div>
      </div>
      <label className="select-row">
        <span>天气情景</span>
        <select
          value={payload.scenario}
          onChange={(event) => setPayload({ ...payload, scenario: event.target.value })}
        >
          {scenarioOptions.map((item) => <option key={item} value={item}>{item}</option>)}
        </select>
      </label>
      <Slider
        item={['cooling_setpoint_c', '供冷设定温度', '°C', COOLING_SETPOINT_MIN, COOLING_SETPOINT_MAX, 0.5]}
        value={payload.cooling_setpoint_c}
        onChange={(key, value) => setPayload({ ...payload, [key]: value })}
      />
      <Slider
        item={['heating_setpoint_c', '供暖设定温度', '°C', 16, 20, 0.5]}
        value={payload.heating_setpoint_c}
        onChange={(key, value) => setPayload({ ...payload, [key]: value })}
      />
      <button className="primary-button hot" onClick={() => run()} disabled={loading}>
        {loading ? <Loader2 className="spin" size={18} /> : <ThermometerSun size={18} />}
        估计极端峰值
      </button>
      {error && <p className="error-text">{error}</p>}
    </Shell>
  );
}

function FeaturePage({ onBack }) {
  const [features, setFeatures] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    setLoading(true);
    api.inputFeatures()
      .then(setFeatures)
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, []);

  const climate = useMemo(() => (features?.climateDaily || []).map((day) => ({
    ...day,
    drybulbMeanC: Number(day.drybulbMeanC.toFixed(1)),
    drybulbMinC: Number(day.drybulbMinC.toFixed(1)),
    drybulbMaxC: Number(day.drybulbMaxC.toFixed(1)),
    rhMeanPct: Number(day.rhMeanPct.toFixed(1)),
    windMeanMS: Number(day.windMeanMS.toFixed(2)),
    ghiKwhM2: Number(day.ghiKwhM2.toFixed(2)),
    hdh16CH: Number(day.hdh16CH.toFixed(1)),
    cdh26CH: Number(day.cdh26CH.toFixed(1)),
  })), [features]);

  return (
    <main className="feature-page">
      <header className="feature-hero">
        <button className="icon-button" onClick={onBack} aria-label="返回">
          <ArrowLeft size={20} />
        </button>
        <div>
          <span>HC-DeepONet 输入变量说明</span>
          <h1>模型输入特征</h1>
          <p>展示代理模型使用的南京基准气候物理特征与建筑/系统参数范围，帮助理解预测结果背后的输入变量。</p>
        </div>
      </header>

      {error && <p className="error-text">{error}</p>}
      {loading && (
        <div className="feature-loading">
          <Loader2 className="spin" size={22} />
          <span>正在加载输入特征...</span>
        </div>
      )}

      {features && (
        <>
          <section className="feature-summary">
            <Metric icon={CloudSun} label="南京逐日气候" value={`${features.summary.climateDays} 天`} tone="cool" />
            <Metric icon={Building2} label="建筑/系统输入" value={`${features.summary.buildingFeatureCount} 项`} tone="energy" />
            <Metric icon={Settings2} label="模型动态输入" value={`${features.summary.dailyWeatherFeatureCount} 维/日`} tone="comfort" />
          </section>

          <section className="feature-section">
            <div className="feature-section-title">
              <h2>气候特征可视化</h2>
              <p>以下图表展示用户容易理解的物理气候特征；模型内部还使用日周期编码和 3 日滚动特征。</p>
            </div>
            <div className="feature-chart-grid">
              <div className="feature-chart wide">
                <h3>日干球温度范围</h3>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={climate} margin={{ top: 12, right: 24, bottom: 12, left: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(30,41,59,.14)" />
                    <XAxis dataKey="label" minTickGap={28} />
                    <YAxis unit="°C" />
                    <Tooltip formatter={(v) => [`${v} °C`, '']} />
                    <Legend />
                    <Line type="monotone" dataKey="drybulbMaxC" name="最高温" stroke="#d9480f" dot={false} strokeWidth={2} />
                    <Line type="monotone" dataKey="drybulbMeanC" name="日均温" stroke="#0f6b8f" dot={false} strokeWidth={2.4} />
                    <Line type="monotone" dataKey="drybulbMinC" name="最低温" stroke="#2563eb" dot={false} strokeWidth={2} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
              <div className="feature-chart">
                <h3>太阳辐射与风速</h3>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={climate} margin={{ top: 12, right: 24, bottom: 12, left: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(30,41,59,.14)" />
                    <XAxis dataKey="label" minTickGap={40} />
                    <YAxis yAxisId="left" />
                    <YAxis yAxisId="right" orientation="right" />
                    <Tooltip />
                    <Legend />
                    <Bar yAxisId="left" dataKey="ghiKwhM2" name="GHI kWh/m2" fill="#f59e0b" radius={[4, 4, 0, 0]} />
                    <Line yAxisId="right" type="monotone" dataKey="windMeanMS" name="风速 m/s" stroke="#0f766e" dot={false} strokeWidth={2} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <div className="feature-chart">
                <h3>冷热度时强度</h3>
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={climate} margin={{ top: 12, right: 24, bottom: 12, left: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(30,41,59,.14)" />
                    <XAxis dataKey="label" minTickGap={40} />
                    <YAxis />
                    <Tooltip formatter={(v) => [`${v} °C·h`, '']} />
                    <Legend />
                    <Area type="monotone" dataKey="hdh16CH" name="HDH16" stroke="#2563eb" fill="#bfdbfe" fillOpacity={0.75} />
                    <Area type="monotone" dataKey="cdh26CH" name="CDH26" stroke="#d9480f" fill="#fed7aa" fillOpacity={0.72} />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>
            <div className="climate-feature-list">
              {features.climateFeatures.map((item) => (
                <article key={item.name}>
                  <strong>{item.name}</strong>
                  <span>{item.unit}</span>
                  <p>{item.description}</p>
                </article>
              ))}
            </div>
            <p className="feature-note">{features.summary.engineeringNote}</p>
          </section>

          <section className="feature-section">
            <div className="feature-section-title">
              <h2>建筑特征参数介绍</h2>
              <p>这些参数构成模型的 10 维静态输入，范围来自代理模型训练数据。</p>
            </div>
            <div className="building-feature-grid">
              {features.buildingFeatures.map((item) => (
                <article key={item.key} className="building-feature-card">
                  <div>
                    <span>{item.key}</span>
                    <strong>{item.label}</strong>
                  </div>
                  <p>{item.description}</p>
                  <dl>
                    <div>
                      <dt>训练范围</dt>
                      <dd>{item.min} - {item.max}{item.unit === '-' ? '' : ` ${item.unit}`}</dd>
                    </div>
                    <div>
                      <dt>默认值</dt>
                      <dd>{item.default}{item.unit === '-' ? '' : ` ${item.unit}`}</dd>
                    </div>
                  </dl>
                </article>
              ))}
            </div>
          </section>
        </>
      )}
    </main>
  );
}

const DOC_REMARK = [remarkGfm, remarkMath];
const DOC_REHYPE = [[rehypeKatex, { throwOnError: false, strict: false }]];

function DocBlock({ block }) {
  if (block.type === 'heading') {
    const Tag = block.level === 1 ? 'h1' : block.level === 2 ? 'h2' : 'h3';
    return <Tag className={`doc-heading level-${block.level}`}>{block.text}</Tag>;
  }
  if (block.type === 'paragraph') {
    return <p className="doc-paragraph">{block.text}</p>;
  }
  if (block.type === 'formula' || block.type === 'code') {
    return <pre className={`doc-${block.type}`}>{block.text}</pre>;
  }
  if (block.type === 'list') {
    const Tag = block.ordered ? 'ol' : 'ul';
    return (
      <Tag className="doc-list">
        {block.items.map((item, index) => <li key={`${item}-${index}`}>{item}</li>)}
      </Tag>
    );
  }
  if (block.type === 'table') {
    return (
      <div className="doc-table-wrap">
        <table className="doc-table">
          <thead>
            <tr>{block.header.map((cell, index) => <th key={`${cell}-${index}`}>{cell}</th>)}</tr>
          </thead>
          <tbody>
            {block.rows.map((row, rowIndex) => (
              <tr key={rowIndex}>
                {row.map((cell, cellIndex) => <td key={`${cell}-${cellIndex}`}>{cell}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  }
  return null;
}

function PrinciplesPage({ onBack }) {
  const [docs, setDocs] = useState([]);
  const [activeId, setActiveId] = useState('heat-balance');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    setLoading(true);
    api.principleDocs()
      .then((data) => {
        setDocs(data.docs || []);
        if (data.docs?.[0]?.id) setActiveId(data.docs[0].id);
      })
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, []);

  const activeDoc = docs.find((doc) => doc.id === activeId) || docs[0];

  return (
    <main className="principles-page">
      <header className="principles-hero">
        <button className="icon-button" onClick={onBack} aria-label="返回">
          <ArrowLeft size={20} />
        </button>
        <div>
          <span>EnergyPlus / PMV / COP</span>
          <h1>原理文档</h1>
          <p>把三篇介绍性说明放在同一个阅读页中，帮助理解能耗模拟、热舒适评价和设备效率换算的底层逻辑。</p>
        </div>
      </header>

      {error && <p className="error-text">{error}</p>}
      {loading && (
        <div className="feature-loading">
          <Loader2 className="spin" size={22} />
          <span>正在加载原理文档...</span>
        </div>
      )}

      {activeDoc && (
        <section className="principles-shell">
          <nav className="doc-tabs" aria-label="原理文档切换">
            {docs.map((doc) => (
              <button
                key={doc.id}
                className={doc.id === activeDoc.id ? 'active' : ''}
                onClick={() => setActiveId(doc.id)}
              >
                <strong>{doc.title.replace(/^#\s*/, '')}</strong>
                <span>{doc.summary}</span>
              </button>
            ))}
          </nav>

          <article className="doc-reader">
            <div className="doc-meta">
              <span>{activeDoc.sourceName}</span>
              <p>{activeDoc.summary}</p>
            </div>
            {activeDoc.markdown
              ? (
                <div className="doc-md">
                  <ReactMarkdown remarkPlugins={DOC_REMARK} rehypePlugins={DOC_REHYPE}>
                    {activeDoc.markdown}
                  </ReactMarkdown>
                </div>
              )
              : activeDoc.blocks.map((block, index) => (
                <DocBlock key={`${block.type}-${index}`} block={block} />
              ))}
          </article>
        </section>
      )}
    </main>
  );
}

export default function App() {
  const [scene, setScene] = useState('home');
  const [metadata, setMetadata] = useState(null);
  const [health, setHealth] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([api.health(), api.metadata()])
      .then(([h, m]) => {
        setHealth(h);
        setMetadata(m);
      })
      .catch(() => {
        setHealth(null);
      })
      .finally(() => setLoading(false));
  }, []);

  if (scene === 'design') return <DesignPage onBack={() => setScene('home')} />;
  if (scene === 'operation') return <OperationPage onBack={() => setScene('home')} />;
  if (scene === 'extreme') return <ExtremePage onBack={() => setScene('home')} metadata={metadata} />;
  if (scene === 'features') return <FeaturePage onBack={() => setScene('home')} />;
  if (scene === 'principles') return <PrinciplesPage onBack={() => setScene('home')} />;
  if (scene === 'agent') return <AgentPage onBack={() => setScene('home')} />;
  if (scene === 'training') return <TrainingPage onBack={() => setScene('home')} />;
  return <Home onEnter={setScene} health={health} loading={loading} />;
}
