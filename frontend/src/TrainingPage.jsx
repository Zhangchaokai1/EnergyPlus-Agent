import { useEffect, useMemo, useRef, useState } from 'react';
import {
  ArrowLeft,
  Bot,
  Building2,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  ImageUp,
  Loader2,
  Play,
  Save,
  Send,
  ServerCog,
  User,
} from 'lucide-react';
import { api } from './api/client.js';
import BuildingScene from './components/BuildingScene.jsx';
import './training.css';

const PLAN_SHAPE_OPTIONS = [
  { value: 'rectangle', label: '矩形', hint: '单一体量，东南西北周边区 + 核心区' },
  { value: 'l_shape', label: 'L 形', hint: '两段体量拼接，从包络矩形一角切除缺角' },
];

const NOTCH_CORNER_OPTIONS = [
  { value: 'nw', label: '西北角', row: 0, col: 0 },
  { value: 'ne', label: '东北角', row: 0, col: 1 },
  { value: 'sw', label: '西南角', row: 1, col: 0 },
  { value: 'se', label: '东南角', row: 1, col: 1 },
];

const FIELD_GROUPS = [
  ['building_identity', [
    ['building_name', '建筑名称', 'text'],
    ['building_type', '建筑类型', 'text'],
    ['city', '城市', 'text'],
    ['weather_file', '天气文件说明', 'text'],
  ]],
  ['geometry', [
    ['plan_shape', '平面形状', 'select', PLAN_SHAPE_OPTIONS],
    ['length_m', '长度 m（L 形为包络长度）', 'number'],
    ['width_m', '宽度 m（L 形为包络宽度）', 'number'],
    ['floor_count', '层数', 'number'],
    ['ground_floor_height_m', '首层层高 m', 'number'],
    ['typical_floor_height_m', '标准层层高 m', 'number'],
    ['orientation_deg', '朝向 deg', 'number'],
    ['perimeter_depth_m', '周边区深度 m', 'number'],
    ['zoning_mode', '分区模式', 'text'],
    ['notch_corner', '缺角位置（仅 L 形）', 'select', NOTCH_CORNER_OPTIONS],
    ['notch_length_m', '缺角长度 m（仅 L 形）', 'number'],
    ['notch_width_m', '缺角宽度 m（仅 L 形）', 'number'],
  ]],
  ['envelope', [
    ['wall_insulation_m', '外墙保温 m', 'number'],
    ['roof_insulation_m', '屋面保温 m', 'number'],
    ['window_u_w_m2k', '窗 U 值 W/m2K', 'number'],
    ['window_shgc', 'SHGC', 'number'],
    ['window_wall_ratio', '窗墙比', 'number'],
    ['infiltration_ach', '渗透 ACH', 'number'],
  ]],
  ['operation', [
    ['cooling_setpoint_c', '供冷设定 °C', 'number'],
    ['heating_setpoint_c', '供暖设定 °C', 'number'],
    ['cooling_cop', '制冷 COP', 'number'],
    ['heating_cop', '制热 COP', 'number'],
    ['schedule_template', '运行模板', 'text'],
  ]],
  ['dataset_training', [
    ['sample_count', '样本数', 'number'],
    ['sampling_method', '采样方法', 'text'],
    ['seed', '随机种子', 'number'],
    ['train_ensemble_members', '集成成员数', 'number'],
  ]],
];

const SECTION_LABELS = {
  building_identity: '建筑身份',
  geometry: '几何参数',
  envelope: '围护结构',
  operation: '运行参数',
  dataset_training: '数据集与训练',
};

const SOURCE_LABELS = {
  vision_estimated: '视觉估计',
  default_assumption: '默认假设',
  user_confirmed: '用户确认',
};

function setNested(spec, section, key, value, type) {
  const parsed = type === 'number' ? Number(value) : value;
  return {
    ...spec,
    [section]: {
      ...(spec[section] || {}),
      [key]: parsed,
    },
    parameter_sources: {
      ...(spec.parameter_sources || {}),
      [`${section}.${key}`]: 'user_confirmed',
    },
  };
}

// L 形约束：缺角长/宽都要严格大于 2×周边区深度，且小于 包络边长-2×周边区深度，
// 否则某一翼的进深放不下"周边区+核心区"。与 building_project.validate_spec 的规则保持一致，
// 用于前端即时提示；真正的强校验仍在后端 confirm-spec 完成。
function notchConstraintWarnings(spec) {
  const g = spec?.geometry;
  if (!g || g.plan_shape !== 'l_shape') return [];
  const d = Number(g.perimeter_depth_m) || 0;
  const warnings = [];
  const lengthLo = 2 * d;
  const lengthHi = Number(g.length_m) - 2 * d;
  const widthLo = 2 * d;
  const widthHi = Number(g.width_m) - 2 * d;
  const nl = Number(g.notch_length_m);
  const nw = Number(g.notch_width_m);
  if (!(nl > lengthLo && nl < lengthHi)) {
    warnings.push(`缺角长度建议在 ${lengthLo.toFixed(1)} ~ ${lengthHi.toFixed(1)} m 之间（当前 ${fmtNum(nl)} m）`);
  }
  if (!(nw > widthLo && nw < widthHi)) {
    warnings.push(`缺角宽度建议在 ${widthLo.toFixed(1)} ~ ${widthHi.toFixed(1)} m 之间（当前 ${fmtNum(nw)} m）`);
  }
  return warnings;
}

function fmtNum(value) {
  return Number.isFinite(value) ? value.toFixed(1) : '-';
}

function applyQuickTextPatch(spec, text) {
  if (!spec) return spec;
  let next = spec;
  const update = (section, key, value) => {
    next = setNested(next, section, key, value, 'number');
  };
  const updateText = (section, key, value) => {
    next = setNested(next, section, key, value, 'text');
  };
  if (/L\s*[形型]|拐角|转角|退台|两段体量/i.test(text)) {
    updateText('geometry', 'plan_shape', 'l_shape');
  } else if (/矩形|长方形/.test(text) && !/L\s*[形型]/i.test(text)) {
    updateText('geometry', 'plan_shape', 'rectangle');
  }
  const cornerMatch = text.match(/(东北|西北|东南|西南)角?/);
  if (cornerMatch) {
    const map = { 东北: 'ne', 西北: 'nw', 东南: 'se', 西南: 'sw' };
    updateText('geometry', 'notch_corner', map[cornerMatch[1]]);
  }
  const patterns = [
    [/长(?:度)?(?:改为|为|是|=)?\s*(\d+(?:\.\d+)?)\s*m?/i, 'geometry', 'length_m'],
    [/宽(?:度)?(?:改为|为|是|=)?\s*(\d+(?:\.\d+)?)\s*m?/i, 'geometry', 'width_m'],
    [/(\d+(?:\.\d+)?)\s*m?\s*[x×*]\s*(\d+(?:\.\d+)?)\s*m?/i, 'geometry_pair', 'length_width'],
    [/层(?:数)?(?:改为|为|是|=)?\s*(\d+)/i, 'geometry', 'floor_count'],
    [/(\d+)\s*层/i, 'geometry', 'floor_count'],
    [/(?:首层|底层|一层|大堂)层?高(?:度)?(?:改为|为|是|=)?\s*(\d+(?:\.\d+)?)\s*m?/i, 'geometry', 'ground_floor_height_m'],
    [/(?:标准层)层?高(?:度)?(?:改为|为|是|=)?\s*(\d+(?:\.\d+)?)\s*m?/i, 'geometry', 'typical_floor_height_m'],
    [/(?:楼层高度|每层高度|单层高度)(?:改为|为|是|=)?\s*(\d+(?:\.\d+)?)\s*m?/i, 'geometry', 'typical_floor_height_m'],
    [/缺角长度(?:改为|为|是|=)?\s*(\d+(?:\.\d+)?)\s*m?/i, 'geometry', 'notch_length_m'],
    [/缺角宽度(?:改为|为|是|=)?\s*(\d+(?:\.\d+)?)\s*m?/i, 'geometry', 'notch_width_m'],
    [/周边区(?:深度)?(?:改为|为|是|=)?\s*(\d+(?:\.\d+)?)\s*m?/i, 'geometry', 'perimeter_depth_m'],
    [/(?:周边|外围)深度(?:改为|为|是|=)?\s*(\d+(?:\.\d+)?)\s*m?/i, 'geometry', 'perimeter_depth_m'],
    [/朝向(?:改为|为|是|=)?\s*(-?\d+(?:\.\d+)?)\s*(?:度|deg)?/i, 'geometry', 'orientation_deg'],
    [/窗墙比(?:改为|为|是|=)?\s*(0?\.\d+|\d+(?:\.\d+)?%?)/i, 'envelope', 'window_wall_ratio'],
    [/(?:外墙保温|墙体保温)(?:厚度)?(?:改为|为|是|=)?\s*(\d+(?:\.\d+)?)\s*(?:m|米)?/i, 'envelope', 'wall_insulation_m'],
    [/(?:屋面保温|屋顶保温)(?:厚度)?(?:改为|为|是|=)?\s*(\d+(?:\.\d+)?)\s*(?:m|米)?/i, 'envelope', 'roof_insulation_m'],
    [/(?:渗透|换气次数|ACH)(?:改为|为|是|=)?\s*(\d+(?:\.\d+)?)/i, 'envelope', 'infiltration_ach'],
    [/(?:供冷|制冷|冷房)(?:设定)?温度(?:改为|为|是|=)?\s*(\d+(?:\.\d+)?)\s*(?:°?C|度)?/i, 'operation', 'cooling_setpoint_c'],
    [/(?:供暖|采暖|制热)(?:设定)?温度(?:改为|为|是|=)?\s*(\d+(?:\.\d+)?)\s*(?:°?C|度)?/i, 'operation', 'heating_setpoint_c'],
    [/样本数(?:改为|为|是|=)?\s*(\d+)/i, 'dataset_training', 'sample_count'],
  ];
  patterns.forEach(([regex, section, key]) => {
    const match = text.match(regex);
    if (!match) return;
    if (section === 'geometry_pair') {
      update('geometry', 'length_m', Number(match[1]));
      update('geometry', 'width_m', Number(match[2]));
      return;
    }
    let value = match[1];
    if (key === 'window_wall_ratio' && String(value).endsWith('%')) {
      value = Number(String(value).slice(0, -1)) / 100;
    }
    update(section, key, Number(value));
  });
  return next;
}

function sourceOf(spec, section, key) {
  return spec?.parameter_sources?.[`${section}.${key}`] || 'default_assumption';
}

function fmt(value, suffix = '') {
  if (value === null || value === undefined || value === '') return '-';
  return `${value}${suffix}`;
}

function parseJobSamples(job, fallbackTotal) {
  const totalFromJob = Number(job?.sample_total ?? job?.sample_count);
  const doneFromJob = Number(job?.sample_completed ?? job?.completed_samples);
  let total = Number.isFinite(totalFromJob) && totalFromJob > 0 ? totalFromJob : Number(fallbackTotal);
  let completed = Number.isFinite(doneFromJob) && doneFromJob >= 0 ? doneFromJob : null;
  const step = String(job?.current_step || '');
  const match = step.match(/(?:样本|sample)\s*(\d+)\s*\/\s*(\d+)/i)
    || step.match(/completed\s+(\d+)\s*\/\s*(\d+)\s+samples/i);
  if (match) {
    completed = Number(match[1]);
    total = Number(match[2]);
  }
  if (!Number.isFinite(total) || total <= 0) total = Number(fallbackTotal) || 0;
  if (completed === null) {
    if (job?.status === 'completed') completed = total;
    else if (!job) completed = 0;
    else completed = 0;
  }
  completed = Math.max(0, Math.min(Number(completed) || 0, total || Number.MAX_SAFE_INTEGER));
  return { completed, total };
}

function TrainingProgress({ job, sampleTotal }) {
  const { completed, total } = parseJobSamples(job, sampleTotal);
  const pct = job ? Math.max(0, Math.min(100, Number(job.progress_pct) || 0)) : 0;
  const samplePct = total > 0 ? Math.max(0, Math.min(100, (completed / total) * 100)) : pct;
  const displayPct = job?.status === 'simulating' ? samplePct : pct;
  const isActive = job && !['completed', 'failed'].includes(job.status);
  const statusText = job ? `${job.status} · ${Math.round(pct)}%` : '尚未启动训练';
  const sampleText = total > 0 ? `${completed}/${total}` : '-/-';

  return (
    <div className={`job-box progress-card ${isActive ? 'active' : ''} ${job?.status || 'idle'}`}>
      <div className="progress-card-head">
        <span>{statusText}</span>
        <strong>{sampleText}</strong>
      </div>
      <div className="sample-progress-track" aria-label="训练样本进度">
        <div className="sample-progress-fill" style={{ width: `${displayPct}%` }} />
        <div className="sample-progress-glow" style={{ left: `${displayPct}%` }} />
      </div>
      <div className="sample-progress-meta">
        <span>已完成样本 {completed}</span>
        <span>总样本 {total || '-'}</span>
      </div>
      <p>{job?.current_step || '确认参数后，系统会生成 IDF，再异步运行 EnergyPlus 数据集和代理模型训练。'}</p>
      {job?.error && <em>{job.error}</em>}
    </div>
  );
}

function SpecMetric({ label, value, source }) {
  return (
    <div className="spec-metric">
      <span>{label}</span>
      <strong>{value}</strong>
      <em className={`source-badge ${source}`}>{SOURCE_LABELS[source] || '默认假设'}</em>
    </div>
  );
}

export default function TrainingPage({ onBack }) {
  const [spec, setSpec] = useState(null);
  const [file, setFile] = useState(null);
  const [note, setNote] = useState('');
  const [analysis, setAnalysis] = useState(null);
  const [confirmed, setConfirmed] = useState(null);
  const [job, setJob] = useState(null);
  const [models, setModels] = useState(null);
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: '请上传建筑图片，或者直接描述建筑：例如"南京办公楼，15层，矩形平面约36m×36m"，或"L 形平面，东北角缺角，主体40m×30m，首层层高4.5m、标准层3.9m"。我会把信息整理到右侧参数草案中，形状、层高等也可以在右侧面板直接调整。',
    },
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState('');
  const [error, setError] = useState('');
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const scrollRef = useRef(null);

  useEffect(() => {
    api.trainingTemplate()
      .then((data) => setSpec(data.spec))
      .catch((err) => setError(err.message));
    refreshModels();
  }, []);

  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages, loading]);

  useEffect(() => {
    if (!job || ['completed', 'failed'].includes(job.status)) return undefined;
    const timer = setInterval(async () => {
      try {
        const next = await api.trainingJob(job.job_id);
        setJob(next);
        if (next.status === 'completed') refreshModels();
      } catch (err) {
        setError(err.message);
      }
    }, 2500);
    return () => clearInterval(timer);
  }, [job]);

  const refreshModels = async () => {
    try {
      setModels(await api.models());
    } catch {
      setModels(null);
    }
  };

  const sendChat = async (preset) => {
    const content = (preset ?? input).trim();
    if (!content || !spec || loading) return;
    const nextMessages = [...messages, { role: 'user', content }];
    const quickSpec = applyQuickTextPatch(spec, content);
    setMessages(nextMessages);
    setSpec(quickSpec);
    setInput('');
    setError('');
    setLoading('chat');
    try {
      const result = await api.trainingChat({
        messages: nextMessages.map(({ role, content: text }) => ({ role, content: text })),
        spec: quickSpec,
        vision_analysis: analysis?.analysis || null,
      });
      setSpec(result.spec);
      setMessages([...nextMessages, { role: 'assistant', content: result.reply }]);
      if (result.ready_to_confirm) {
        await confirm(result.spec);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading('');
    }
  };

  const analyze = async () => {
    if (!file) {
      setError('请先选择建筑图片');
      return;
    }
    setLoading('analyze');
    setError('');
    try {
      const result = await api.analyzeImage(file, note);
      setAnalysis(result);
      setSpec(result.draft_spec);
      const visible = result.analysis?.visible_description || result.raw_text || '视觉模型已返回建筑描述。';
      setMessages((prev) => ([
        ...prev,
        { role: 'user', content: `我上传了一张建筑图片。${note ? `补充说明：${note}` : ''}` },
        { role: 'assistant', content: `我已经读取图片并更新了右侧参数草案。视觉描述：${visible}\n\n请继续确认长度、宽度、层数、层高、朝向、窗墙比、围护结构和训练样本数。` },
      ]));
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading('');
    }
  };

  const confirm = async (targetSpec = spec) => {
    setLoading('confirm');
    setError('');
    try {
      const result = await api.confirmTrainingSpec(targetSpec);
      setConfirmed(result);
      setMessages((prev) => ([
        ...prev,
        { role: 'assistant', content: `参数已确认，已生成基准 IDF：${result.building_id}。右侧可以启动 EnergyPlus 数据集生成和代理模型训练。` },
      ]));
      return result;
    } catch (err) {
      setError(err.message);
      return null;
    } finally {
      setLoading('');
    }
  };

  const start = async () => {
    if (!confirmed?.building_id) return;
    setLoading('start');
    setError('');
    try {
      const result = await api.startTraining(confirmed.building_id);
      setJob(result);
      setMessages((prev) => ([
        ...prev,
        { role: 'assistant', content: `训练任务已启动：${result.job_id}。我会在右侧显示仿真和训练进度。` },
      ]));
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading('');
    }
  };

  const selectModel = async (buildingId, modelId) => {
    setLoading(`${buildingId}:${modelId}`);
    setError('');
    try {
      await api.selectModel(buildingId, modelId);
      await refreshModels();
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading('');
    }
  };

  const onKeyDown = (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      sendChat();
    }
  };

  const geometry = spec?.geometry || {};
  const envelope = spec?.envelope || {};
  const operation = spec?.operation || {};
  const training = spec?.dataset_training || {};
  const buildingName = spec?.building_identity?.building_name || '用户建筑';
  const isLShape = geometry.plan_shape === 'l_shape';
  const notchWarnings = useMemo(() => notchConstraintWarnings(spec), [spec]);

  const quickSummary = useMemo(() => {
    if (!spec) return [];
    const rows = [
      ['形状', isLShape ? 'L 形' : '矩形', sourceOf(spec, 'geometry', 'plan_shape')],
      ['长度', fmt(geometry.length_m, ' m'), sourceOf(spec, 'geometry', 'length_m')],
      ['宽度', fmt(geometry.width_m, ' m'), sourceOf(spec, 'geometry', 'width_m')],
      ['层数', fmt(geometry.floor_count, ' 层'), sourceOf(spec, 'geometry', 'floor_count')],
      ['首层层高', fmt(geometry.ground_floor_height_m, ' m'), sourceOf(spec, 'geometry', 'ground_floor_height_m')],
      ['标准层层高', fmt(geometry.typical_floor_height_m, ' m'), sourceOf(spec, 'geometry', 'typical_floor_height_m')],
      ['窗墙比', fmt(envelope.window_wall_ratio), sourceOf(spec, 'envelope', 'window_wall_ratio')],
      ['样本数', fmt(training.sample_count), sourceOf(spec, 'dataset_training', 'sample_count')],
    ];
    if (isLShape) {
      const cornerLabel = NOTCH_CORNER_OPTIONS.find((o) => o.value === geometry.notch_corner)?.label || '-';
      rows.splice(1, 0, ['缺角位置', cornerLabel, sourceOf(spec, 'geometry', 'notch_corner')]);
      rows.push(['缺角长度', fmt(geometry.notch_length_m, ' m'), sourceOf(spec, 'geometry', 'notch_length_m')]);
      rows.push(['缺角宽度', fmt(geometry.notch_width_m, ' m'), sourceOf(spec, 'geometry', 'notch_width_m')]);
    }
    return rows;
  }, [spec, geometry, envelope, training, isLShape]);

  return (
    <main className="training-chat-page">
      <header className="training-chat-topbar">
        <button className="icon-button" onClick={onBack} aria-label="返回">
          <ArrowLeft size={20} />
        </button>
        <div>
          <h1>自然语言代理模型训练</h1>
          <p>左侧用 DeepSeek 对话确认参数，右侧实时查看建筑模型、参数草案和训练状态。</p>
        </div>
      </header>

      {error && <div className="training-error">{typeof error === 'string' ? error : JSON.stringify(error)}</div>}

      <section className="training-chat-shell">
        <section className="training-chat-left">
          <div className="chat-scroll training-scroll" ref={scrollRef}>
            {messages.map((message, index) => (
              <div key={`${message.role}-${index}`} className={`chat-msg ${message.role}`}>
                <span className="chat-avatar">
                  {message.role === 'user' ? <User size={17} /> : <Bot size={17} />}
                </span>
                <div className="chat-main">
                  <div className="chat-bubble">{message.content}</div>
                </div>
              </div>
            ))}
            {loading === 'chat' && (
              <div className="chat-msg assistant">
                <span className="chat-avatar"><Bot size={17} /></span>
                <div className="chat-bubble"><Loader2 className="spin" size={16} /></div>
              </div>
            )}
          </div>

          <div className="image-chat-card">
            <label>
              <ImageUp size={16} />
              <span>{file ? file.name : '上传建筑图片'}</span>
              <input type="file" accept="image/*" onChange={(e) => setFile(e.target.files?.[0] || null)} />
            </label>
            <input
              value={note}
              onChange={(e) => setNote(e.target.value)}
              placeholder="可选补充：城市、层数、用途、尺寸估计"
            />
            <button onClick={analyze} disabled={loading === 'analyze'}>
              {loading === 'analyze' ? <Loader2 className="spin" size={16} /> : <ImageUp size={16} />}
              分析图片
            </button>
          </div>

          <div className="chat-input-bar training-input">
            <textarea
              value={input}
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={onKeyDown}
              rows={1}
              placeholder="告诉我需要修改或确认的参数，例如：改成L形，东南角缺角；首层层高4.5m，标准层3.9m；样本数先用64。"
              disabled={!!loading || !spec}
            />
            <button className="chat-send" onClick={() => sendChat()} disabled={!!loading || !input.trim() || !spec}>
              {loading === 'chat' ? <Loader2 className="spin" size={18} /> : <Send size={18} />}
            </button>
          </div>
        </section>

        <aside className="training-chat-right">
          <section className="building-preview-panel">
            <div className="right-panel-head">
              <Building2 size={18} />
              <div>
                <strong>{buildingName}</strong>
                <span>{fmt(geometry.length_m, 'm')} × {fmt(geometry.width_m, 'm')} · {fmt(geometry.floor_count, '层')}{isLShape ? ' · L 形' : ''}</span>
              </div>
            </div>
            <div className="training-model-stage">
              <BuildingScene mode="design" intensity={0.7} spec={spec} />
            </div>
          </section>

          <section className="shape-panel">
            <div className="right-panel-head">
              <ServerCog size={18} />
              <div>
                <strong>平面形状与层高</strong>
                <span>形状变化仅影响几何生成，围护/运行参数不受影响</span>
              </div>
            </div>
            <div className="shape-toggle">
              {PLAN_SHAPE_OPTIONS.map((opt) => (
                <button
                  key={opt.value}
                  className={geometry.plan_shape === opt.value ? 'active' : ''}
                  onClick={() => spec && setSpec(setNested(spec, 'geometry', 'plan_shape', opt.value, 'text'))}
                  disabled={!spec}
                >
                  <strong>{opt.label}</strong>
                  <em>{opt.hint}</em>
                </button>
              ))}
            </div>
            {isLShape && (
              <div className="notch-editor">
                <div className="notch-corner-grid">
                  {NOTCH_CORNER_OPTIONS.map((opt) => (
                    <button
                      key={opt.value}
                      className={`notch-corner-cell r${opt.row}c${opt.col} ${geometry.notch_corner === opt.value ? 'active' : ''}`}
                      onClick={() => spec && setSpec(setNested(spec, 'geometry', 'notch_corner', opt.value, 'text'))}
                      title={`缺角在${opt.label}`}
                    >
                      {opt.label}
                    </button>
                  ))}
                </div>
                <div className="notch-dims">
                  <label>
                    <span>缺角长度 m</span>
                    <input
                      type="number"
                      step="0.5"
                      value={geometry.notch_length_m ?? ''}
                      onChange={(e) => setSpec(setNested(spec, 'geometry', 'notch_length_m', e.target.value, 'number'))}
                    />
                  </label>
                  <label>
                    <span>缺角宽度 m</span>
                    <input
                      type="number"
                      step="0.5"
                      value={geometry.notch_width_m ?? ''}
                      onChange={(e) => setSpec(setNested(spec, 'geometry', 'notch_width_m', e.target.value, 'number'))}
                    />
                  </label>
                </div>
                {notchWarnings.length > 0 && (
                  <ul className="notch-warnings">
                    {notchWarnings.map((w) => <li key={w}>{w}</li>)}
                  </ul>
                )}
              </div>
            )}
            <div className="floor-height-row">
              <label>
                <span>首层层高 m</span>
                <input
                  type="number"
                  step="0.1"
                  value={geometry.ground_floor_height_m ?? ''}
                  onChange={(e) => setSpec(setNested(spec, 'geometry', 'ground_floor_height_m', e.target.value, 'number'))}
                />
              </label>
              <label>
                <span>标准层层高 m</span>
                <input
                  type="number"
                  step="0.1"
                  value={geometry.typical_floor_height_m ?? ''}
                  onChange={(e) => setSpec(setNested(spec, 'geometry', 'typical_floor_height_m', e.target.value, 'number'))}
                />
              </label>
            </div>
          </section>

          <section className="spec-summary-panel">
            <div className="right-panel-head">
              <ServerCog size={18} />
              <div>
                <strong>参数草案</strong>
                <span>LLM 只维护参数，IDF 由程序生成</span>
              </div>
            </div>
            <div className="spec-metric-grid">
              {quickSummary.map(([label, value, source]) => (
                <SpecMetric key={label} label={label} value={value} source={source} />
              ))}
            </div>
            <dl className="spec-small-list">
              <div><dt>供冷/供暖设定</dt><dd>{fmt(operation.cooling_setpoint_c, '°C')} / {fmt(operation.heating_setpoint_c, '°C')}</dd></div>
              <div><dt>COP</dt><dd>{fmt(operation.cooling_cop)} / {fmt(operation.heating_cop)}</dd></div>
              <div><dt>外墙/屋面保温</dt><dd>{fmt(envelope.wall_insulation_m, 'm')} / {fmt(envelope.roof_insulation_m, 'm')}</dd></div>
              <div><dt>窗口性能</dt><dd>U={fmt(envelope.window_u_w_m2k)} · SHGC={fmt(envelope.window_shgc)}</dd></div>
            </dl>
          </section>

          <section className="training-actions-panel">
            <button className="primary-button green" onClick={() => confirm()} disabled={!spec || loading === 'confirm'}>
              {loading === 'confirm' ? <Loader2 className="spin" size={16} /> : <Save size={16} />}
              确认参数并生成 IDF
            </button>
            <button className="primary-button hot" onClick={start} disabled={!confirmed || loading === 'start'}>
              {loading === 'start' ? <Loader2 className="spin" size={16} /> : <Play size={16} />}
              启动仿真与训练
            </button>
            {confirmed && (
              <div className="confirm-box">
                <CheckCircle2 size={18} />
                <span>{confirmed.building_id}</span>
              </div>
            )}
            <TrainingProgress job={job} sampleTotal={training.sample_count} />
          </section>

          <section className="advanced-spec-panel">
            <button className="advanced-toggle" onClick={() => setAdvancedOpen((v) => !v)}>
              <span>高级参数编辑</span>
              {advancedOpen ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
            </button>
            {advancedOpen && spec && FIELD_GROUPS.map(([section, fields]) => (
              <fieldset key={section}>
                <legend>{SECTION_LABELS[section]}</legend>
                <div className="spec-fields compact">
                  {fields
                    .filter(([key]) => isLShape || !['notch_corner', 'notch_length_m', 'notch_width_m'].includes(key))
                    .map(([key, label, type, options]) => (
                    <label key={`${section}.${key}`}>
                      <span>
                        {label}
                        <em className={`source-badge ${sourceOf(spec, section, key)}`}>
                          {SOURCE_LABELS[sourceOf(spec, section, key)] || '默认假设'}
                        </em>
                      </span>
                      {type === 'select' ? (
                        <select
                          value={spec?.[section]?.[key] ?? ''}
                          onChange={(e) => setSpec(setNested(spec, section, key, e.target.value, 'text'))}
                        >
                          {options.map((opt) => <option key={opt.value} value={opt.value}>{opt.label}</option>)}
                        </select>
                      ) : (
                        <input
                          type={type}
                          step={type === 'number' ? 'any' : undefined}
                          value={spec?.[section]?.[key] ?? ''}
                          onChange={(e) => setSpec(setNested(spec, section, key, e.target.value, type))}
                        />
                      )}
                    </label>
                  ))}
                </div>
              </fieldset>
            ))}
          </section>


          <section className="model-library compact-library">
            <div className="right-panel-head">
              <ServerCog size={18} />
              <div>
                <strong>模型库</strong>
                <span>训练完成后可切换到新模型</span>
              </div>
            </div>
            <div className="model-list">
              {(models?.models || []).map((model) => {
                const selected = models?.selected?.building_id === model.building_id
                  && models?.selected?.model_id === model.model_id;
                return (
                  <article key={`${model.building_id}:${model.model_id}`} className={selected ? 'selected' : ''}>
                    <strong>{model.building_name || model.name}</strong>
                    <span>{model.building_id} / {model.model_id}</span>
                    <p>{model.is_default ? '默认模型' : `样本数 ${model.dataset?.sample_count ?? '-'} · ${model.model?.type || 'CHU-DeepONet'}`}</p>
                    <button
                      className="secondary-button"
                      onClick={() => selectModel(model.building_id, model.model_id)}
                      disabled={selected || loading === `${model.building_id}:${model.model_id}`}
                    >
                      {selected ? '当前模型' : '选择模型'}
                    </button>
                  </article>
                );
              })}
            </div>
          </section>
        </aside>
      </section>
    </main>
  );
}
