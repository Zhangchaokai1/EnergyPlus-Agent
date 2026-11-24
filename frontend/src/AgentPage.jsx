import { useEffect, useRef, useState } from 'react';
import { ArrowLeft, Bot, Check, DatabaseZap, FileDown, Loader2, Send, Sparkles, User, Wrench } from 'lucide-react';
import { api } from './api/client.js';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import ToolChart from './ChatCharts.jsx';
import './agent.css';

const SUGGESTIONS = [
  '把外墙保温加到 8cm、窗户换成 U=2.0 的 Low-E，相比现状能省多少冷热电耗？',
  '夏天供冷设定从 26°C 调到 27°C，全年大概能省多少电？舒适度会差多少？',
  '如果遇到异常高温年，这栋楼夏季能耗峰值会涨多少？',
  '在成本、能耗、舒适三个目标之间，帮我推荐几个高性价比的改造方案。',
];

const TOOL_LABELS = {
  present_pareto_comparison: '计算比较与偏好取舍',
  predict_energy: '代理模型能耗预测',
  optimize_pareto: '三目标 Pareto 优化',
  operation_savings: '运维设定点节能',
  estimate_extreme: '极端天气能耗峰值',
  explain_features: 'SHAP 特征归因',
  run_energyplus_simulation: 'EnergyPlus 高保真仿真',
};

function fmtNum(value) {
  return typeof value === 'number' ? value.toLocaleString('zh-CN') : value;
}

const REMARK_PLUGINS = [remarkGfm];

// 助手回复按 Markdown 渲染（加粗 / 表格 / 多级标题 / 引用 / 列表 / 代码）
function Markdown({ text }) {
  return (
    <div className="chat-md">
      <ReactMarkdown remarkPlugins={REMARK_PLUGINS}>{text}</ReactMarkdown>
    </div>
  );
}

function ToolSummary({ tool }) {
  const r = tool.result || {};
  if (tool.name === 'predict_energy' || tool.name === 'run_energyplus_simulation') {
    const a = r.annual || {};
    return (
      <ul className="tool-kv">
        <li>年冷热总输入：<b>{fmtNum(a.totalInputKwh)} kWh</b></li>
        <li>供暖 / 供冷输入：{fmtNum(a.heatingInputKwh)} / {fmtNum(a.coolingInputKwh)} kWh</li>
        <li>年等效不舒适：{a.discomfortHours} h</li>
        {r.vsBaseline && <li>相对基线节能：{r.vsBaseline.savingsPct}%</li>}
        {r.runtimeSeconds != null && <li>EnergyPlus 仿真耗时：{r.runtimeSeconds}s</li>}
      </ul>
    );
  }
  if (tool.name === 'optimize_pareto') {
    return (
      <ul className="tool-kv">
        <li>候选 {fmtNum(r.candidateCount)} 个 · 非支配 {r.paretoCount} 个</li>
        {r.comparisonSummary && <li>以下比较仅针对按成本升序选出的 {r.comparisonSummary.displayedCount} 个展示方案：</li>}
        {r.comparisonSummary ? r.comparisonSummary.highlights.map((text) => (
          <li key={text}>{text}</li>
        )) : (r.topSolutions || []).slice(0, 3).map((s) => (
          <li key={s.id}>
            #{s.id}：成本 {fmtNum(s.retrofitCostCny)} 元 · 节能 {s.savingsVsBaselinePct}% · 不适 {s.annualDiscomfortHours} h
          </li>
        ))}
        {r.comparisonSummary && <li>节能率按含不确定性惩罚的稳健用电目标相对基线计算。</li>}
      </ul>
    );
  }
  if (tool.name === 'operation_savings') {
    const a = r.annual || {};
    const s = r.raiseCoolingBy1C || {};
    return (
      <ul className="tool-kv">
        <li>年供冷 / 供暖输入：{fmtNum(a.coolingInputKwh)} / {fmtNum(a.heatingInputKwh)} kWh</li>
        <li>升温 1°C 夏季可省：{fmtNum(s.summerCoolingSavedKwh)} kWh（{s.summerCoolingSavingPct}%）</li>
      </ul>
    );
  }
  if (tool.name === 'estimate_extreme') {
    return (
      <ul className="tool-kv">
        <li>情景：{r.scenario}</li>
        <li>正常日均：{fmtNum(r.normalAvgDailyTotalInputKwh)} kWh</li>
        <li>极端峰值日（{r.peakDayLabel}）：{fmtNum(r.extremePeakDayKwh)} kWh（+{r.peakIncreasePctVsNormalAvg}%）</li>
      </ul>
    );
  }
  if (tool.name === 'explain_features') {
    return (
      <ul className="tool-kv">
        {(r.topFeatures || []).map((f) => (
          <li key={f.feature}>{f.label || f.feature}：{f.importancePct}%（{f.direction}）</li>
        ))}
      </ul>
    );
  }
  return null;
}

function ToolCard({ tool }) {
  const gate = tool.result && tool.result.gating;
  const source = tool.result && (tool.result.source === 'computed_pareto_presentation'
    ? '计算结果比较' : tool.result.source);
  const err = tool.result && tool.result.error;
  return (
    <div className="tool-card">
      <div className="tool-head">
        <Wrench size={14} />
        <strong>{TOOL_LABELS[tool.name] || tool.name}</strong>
        {tool.status === 'running'
          ? (
            <span className="tool-running">
              <Loader2 className="spin" size={14} />
              {tool.progressS != null ? `仿真中 ${tool.progressS}s` : ''}
            </span>
          )
          : <Check className="done-ic" size={14} />}
      </div>
      {gate && (
        <span className={`gate-badge ${gate.trustSurrogate ? 'trust' : 'fallback'}`}>
          {gate.trustSurrogate ? '🟢 信任代理' : '🔴 建议回退仿真'} · σ_E {fmtNum(gate.sigmaEnergyTotalKwh)} / {fmtNum(gate.thresholdP95Kwh)} kWh
        </span>
      )}
      {source && (
        <span className="gate-badge sim">
          ⚙ {source}{tool.result.runtimeSeconds != null ? ` · ${tool.result.runtimeSeconds}s` : ''}
        </span>
      )}
      {tool.result && !err && <ToolSummary tool={tool} />}
      {tool.result && tool.result.chart && <ToolChart chart={tool.result.chart} />}
      {err && <div className="tool-error">⚠ {err}</div>}
    </div>
  );
}

export default function AgentPage({ onBack }) {
  const [messages, setMessages] = useState([]); // {role, content, tools?}
  const [input, setInput] = useState('');
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState('');
  const [models, setModels] = useState(null);
  const [modelLoading, setModelLoading] = useState(false);
  const [modelsLoading, setModelsLoading] = useState(true);
  const [modelsError, setModelsError] = useState('');
  const [reportState, setReportState] = useState({ status: 'idle', result: null, error: '' });
  const scrollRef = useRef(null);

  const refreshModels = async () => {
    setModelsLoading(true);
    setModelsError('');
    try {
      setModels(await api.models());
    } catch (err) {
      setModels(null);
      setModelsError(err.message || String(err));
    } finally {
      setModelsLoading(false);
    }
  };

  useEffect(() => {
    refreshModels();
  }, []);

  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages, streaming]);

  const selectedKey = models?.selected
    ? `${models.selected.building_id}:${models.selected.model_id}`
    : '';
  const selectedModel = (models?.models || []).find(
    (model) => `${model.building_id}:${model.model_id}` === selectedKey,
  );
  const hasModels = (models?.models || []).length > 0;
  const modelMetaText = (() => {
    if (modelLoading) return '切换中...';
    if (modelsLoading) return '模型库加载中...';
    if (modelsError) return `模型接口请求失败：${modelsError}`;
    if (models && !hasModels) return '模型列表为空';
    if (selectedModel) {
      return `${selectedModel.is_default ? '默认模型' : `样本数 ${selectedModel.dataset?.sample_count ?? '-'}`} · ${selectedModel.model?.type || 'CHU-DeepONet'}`;
    }
    return '未选中模型';
  })();

  const onModelChange = async (event) => {
    const [buildingId, modelId] = event.target.value.split(':');
    if (!buildingId || !modelId) return;
    setModelLoading(true);
    setError('');
    setReportState({ status: 'idle', result: null, error: '' });
    try {
      await api.selectModel(buildingId, modelId);
      await refreshModels();
    } catch (err) {
      setError(err.message || String(err));
    } finally {
      setModelLoading(false);
    }
  };

  const generateReport = async () => {
    if (!models?.selected || reportState.status === 'loading') return;
    setError('');
    setReportState({ status: 'loading', result: null, error: '' });
    try {
      const result = await api.generateEnergyComfortReport({
        building_id: models.selected.building_id,
        model_id: models.selected.model_id,
      });
      setReportState({ status: 'ready', result, error: '' });
      const url = api.reportDownloadUrl(result.download_url);
      window.open(url, '_blank', 'noopener,noreferrer');
    } catch (err) {
      setReportState({ status: 'error', result: null, error: err.message || String(err) });
    }
  };

  const send = async (preset) => {
    const content = (preset ?? input).trim();
    if (!content || streaming) return;
    setError('');
    setInput('');
    const base = [...messages, { role: 'user', content }];
    setMessages([...base, { role: 'assistant', content: '', tools: [] }]);
    setStreaming(true);
    const wire = base
      .filter((m) => m.role === 'user' || (m.role === 'assistant' && m.content))
      .map((m) => ({ role: m.role, content: m.content }));
    try {
      await api.agentChat(wire, (ev) => {
        setMessages((prev) => {
          const next = prev.slice();
          const last = { ...next[next.length - 1] };
          last.tools = last.tools ? last.tools.slice() : [];
          if (ev.type === 'delta') {
            last.content += ev.content || '';
          } else if (ev.type === 'progress') {
            for (let i = last.tools.length - 1; i >= 0; i -= 1) {
              if (last.tools[i].name === ev.name && last.tools[i].status === 'running') {
                last.tools[i] = { ...last.tools[i], progressS: ev.elapsedS };
                break;
              }
            }
          } else if (ev.type === 'tool_call') {
            last.tools.push({ name: ev.name, arguments: ev.arguments, status: 'running', result: null });
          } else if (ev.type === 'tool_result') {
            for (let i = last.tools.length - 1; i >= 0; i -= 1) {
              if (last.tools[i].name === ev.name && last.tools[i].status === 'running') {
                last.tools[i] = { ...last.tools[i], status: 'done', result: ev.result };
                break;
              }
            }
          }
          next[next.length - 1] = last;
          return next;
        });
      });
    } catch (err) {
      setError(err.message || String(err));
      setMessages((prev) => {
        const next = prev.slice();
        const last = next[next.length - 1];
        if (last && last.role === 'assistant' && !last.content && (!last.tools || !last.tools.length)) {
          next.pop();
        }
        return next;
      });
    } finally {
      setStreaming(false);
    }
  };

  const onKeyDown = (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      send();
    }
  };

  const empty = messages.length === 0;

  return (
    <main className="agent-page">
      <header className="topbar">
        <button className="icon-button" onClick={onBack} aria-label="返回">
          <ArrowLeft size={20} />
        </button>
        <div>
          <h1>EnergyPlus-Agent 智能问答</h1>
          <p>物理增强生成（PAG）+ 不确定性门控 · 工具：代理预测 / Pareto / 运维 / 极端 / SHAP</p>
        </div>
      </header>

      <section className="agent-model-bar">
        <DatabaseZap size={18} />
        <label>
          <span>当前代理模型</span>
          <select value={selectedKey} onChange={onModelChange} disabled={streaming || modelLoading || modelsLoading || !hasModels}>
            {(models?.models || []).map((model) => (
              <option key={`${model.building_id}:${model.model_id}`} value={`${model.building_id}:${model.model_id}`}>
                {model.building_name || model.name || model.model_id}
              </option>
            ))}
          </select>
        </label>
        <div className={`agent-model-meta ${modelsError ? 'error' : ''}`}>
          {modelMetaText}
        </div>
        {(modelsError || (models && !hasModels)) && (
          <button
            type="button"
            className="model-retry-button"
            onClick={refreshModels}
            disabled={streaming || modelLoading || modelsLoading}
          >
            {modelsLoading ? <Loader2 className="spin" size={14} /> : null}
            重试加载
          </button>
        )}
        <div className="agent-report-actions">
          <button
            className="report-generate-button"
            onClick={generateReport}
            disabled={streaming || modelLoading || !models?.selected || reportState.status === 'loading'}
          >
            {reportState.status === 'loading' ? <Loader2 className="spin" size={16} /> : <FileDown size={16} />}
            建筑能耗与舒适度优化报告
          </button>
          {reportState.status === 'ready' && reportState.result && (
            <a
              className="report-download-link"
              href={api.reportDownloadUrl(reportState.result.download_url)}
              target="_blank"
              rel="noreferrer"
            >
              下载 PDF
              <span>{reportState.result.ai_mode === 'llm' ? 'AI 分析' : '模板兜底'}</span>
            </a>
          )}
          {reportState.status === 'error' && (
            <span className="report-error">{reportState.error}</span>
          )}
        </div>
      </section>

      <section className="agent-shell">
        <div className="chat-scroll" ref={scrollRef}>
          {empty && (
            <div className="chat-empty">
              <span className="chat-empty-icon"><Bot size={30} /></span>
              <h2>你好，我是 EnergyPlus-Agent</h2>
              <p>
                用自然语言描述你的建筑与目标，我会调用物理代理模型给出定量分析，并显示不确定性门控判定；
                遇到外推 / 极端工况会建议回退 EnergyPlus 仿真。可以先试试：
              </p>
              <div className="chat-suggestions">
                {SUGGESTIONS.map((item) => (
                  <button key={item} onClick={() => send(item)} disabled={streaming}>
                    <Sparkles size={15} />
                    <span>{item}</span>
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((message, index) => {
            const isLast = index === messages.length - 1;
            const hasTools = message.tools && message.tools.length > 0;
            const showBubble = message.content
              || (message.role === 'assistant' && streaming && isLast && !hasTools);
            return (
              <div key={index} className={`chat-msg ${message.role}`}>
                <span className="chat-avatar">
                  {message.role === 'user' ? <User size={17} /> : <Bot size={17} />}
                </span>
                <div className="chat-main">
                  {message.role === 'assistant' && hasTools && (
                    <div className="tool-stack">
                      {message.tools.map((tool, ti) => <ToolCard key={ti} tool={tool} />)}
                    </div>
                  )}
                  {showBubble && (
                    <div className="chat-bubble">
                      {message.content
                        ? (message.role === 'assistant'
                          ? <Markdown text={message.content} />
                          : message.content)
                        : <Loader2 className="spin" size={16} />}
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>

        {error && <p className="error-text agent-error">{error}</p>}

        <div className="chat-input-bar">
          <textarea
            value={input}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={onKeyDown}
            rows={1}
            placeholder="描述你的建筑 / 问题，回车发送，Shift+回车换行…"
            disabled={streaming}
          />
          <button className="chat-send" onClick={() => send()} disabled={streaming || !input.trim()}>
            {streaming ? <Loader2 className="spin" size={18} /> : <Send size={18} />}
          </button>
        </div>
      </section>
    </main>
  );
}
