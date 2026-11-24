import {
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

const AXIS = { fontSize: 11, fill: '#64748b' };

// 把工具返回的 chart 负载渲染为对应图表（复用 showcase 的 Recharts 组件）。
export default function ToolChart({ chart }) {
  if (!chart || !chart.kind) return null;

  if (chart.kind === 'dailyCurve') {
    return (
      <div className="tool-chart">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chart.daily} margin={{ top: 8, right: 14, left: 0, bottom: 4 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(30,41,59,.1)" />
            <XAxis dataKey="label" minTickGap={48} tick={AXIS} />
            <YAxis tick={AXIS} width={44} />
            <Tooltip formatter={(v) => [`${v} kWh`, '']} />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            <Line type="monotone" dataKey="heating" name="供暖输入" stroke="#d97706" dot={false} strokeWidth={1.6} />
            <Line type="monotone" dataKey="cooling" name="供冷输入" stroke="#0284c7" dot={false} strokeWidth={1.6} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    );
  }

  if (chart.kind === 'pareto') {
    return (
      <div className="tool-chart">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 10, right: 16, bottom: 6, left: 6 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(30,41,59,.1)" />
            <XAxis type="number" dataKey="x" name="成本" unit="万元" tick={AXIS} />
            <YAxis type="number" dataKey="y" name="输入" unit="万kWh" tick={AXIS} width={56} />
            <Tooltip cursor={{ strokeDasharray: '3 3' }} formatter={(v, n) => [Number(v).toFixed(1), n]} />
            <Scatter data={chart.points}>
              {chart.points.map((p, i) => (
                <Cell key={i} fill={p.pareto ? '#4f46e5' : '#94a3b8'} fillOpacity={p.pareto ? 0.95 : 0.32} />
              ))}
            </Scatter>
          </ScatterChart>
        </ResponsiveContainer>
        <div className="chart-note">实心=Pareto 非支配 · X 成本(万元) · Y 稳健冷热输入(万kWh)</div>
      </div>
    );
  }

  if (chart.kind === 'bars') {
    return (
      <div className="tool-chart">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chart.bars} margin={{ top: 12, right: 16, bottom: 6, left: 6 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(30,41,59,.1)" />
            <XAxis dataKey="name" tick={AXIS} />
            <YAxis tick={AXIS} width={56} />
            <Tooltip formatter={(v) => [`${Number(v).toLocaleString('zh-CN')} ${chart.unit || ''}`, '']} />
            <Bar dataKey="value" radius={[6, 6, 0, 0]}>
              {chart.bars.map((b, i) => <Cell key={i} fill={i === 0 ? '#64748b' : '#d9480f'} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    );
  }

  if (chart.kind === 'featureBars') {
    return (
      <div className="tool-chart">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chart.features} layout="vertical" margin={{ top: 6, right: 18, bottom: 6, left: 6 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(30,41,59,.1)" />
            <XAxis type="number" tick={AXIS} unit="%" />
            <YAxis type="category" dataKey="label" tick={AXIS} width={92} />
            <Tooltip formatter={(v) => [`${v}%`, '重要性']} />
            <Bar dataKey="importancePct" fill="#4f46e5" radius={[0, 5, 5, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    );
  }

  return null;
}
