const API_BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8010';

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    ...options,
  });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const data = await response.json();
      detail = data.detail || detail;
    } catch {
      // keep default error text
    }
    throw new Error(detail);
  }
  return response.json();
}

async function multipartRequest(path, formData) {
  const response = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    body: formData,
  });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const data = await response.json();
      detail = data.detail || detail;
    } catch {
      // keep default error text
    }
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
  }
  return response.json();
}

// 智能体流式对话：解析 SSE，对每个事件调用 onEvent({type, ...})。
// 事件类型：delta（文本增量）/ tool_call / tool_result / done / error。
async function agentChatStream(messages, onEvent, signal) {
  const response = await fetch(`${API_BASE}/api/agent/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ messages }),
    signal,
  });
  if (!response.ok || !response.body) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const data = await response.json();
      detail = data.detail || detail;
    } catch {
      // keep default error text
    }
    throw new Error(detail);
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let sep;
    while ((sep = buffer.indexOf('\n\n')) >= 0) {
      const chunk = buffer.slice(0, sep).trim();
      buffer = buffer.slice(sep + 2);
      if (!chunk.startsWith('data:')) continue;
      const payload = chunk.slice(5).trim();
      if (!payload) continue;
      let event;
      try {
        event = JSON.parse(payload);
      } catch {
        continue;
      }
      if (event.type === 'error' || event.error) {
        throw new Error(event.error || '智能体服务出错');
      }
      onEvent(event);
      if (event.type === 'done' || event.done) return;
    }
  }
}

export const api = {
  agentChat: (messages, onEvent, signal) => agentChatStream(messages, onEvent, signal),
  health: () => request('/api/health'),
  metadata: () => request('/api/metadata'),
  inputFeatures: () => request('/api/input-features'),
  principleDocs: () => request('/api/principle-docs'),
  designPredict: (inputs) => request('/api/design/predict', {
    method: 'POST',
    body: JSON.stringify({ inputs }),
  }),
  designPareto: (payload) => request('/api/design/pareto', {
    method: 'POST',
    body: JSON.stringify(payload),
  }),
  operationPredict: (inputs) => request('/api/operation/predict', {
    method: 'POST',
    body: JSON.stringify({ inputs }),
  }),
  operationSaving: (inputs) => request('/api/operation/saving', {
    method: 'POST',
    body: JSON.stringify({ inputs }),
  }),
  extremeEstimate: (payload) => request('/api/extreme/estimate', {
    method: 'POST',
    body: JSON.stringify(payload),
  }),
  trainingTemplate: () => request('/api/training/template'),
  trainingChat: (payload) => request('/api/training/chat', {
    method: 'POST',
    body: JSON.stringify(payload),
  }),
  analyzeImage: (file, note = '') => {
    const data = new FormData();
    data.append('file', file);
    return multipartRequest(`/api/training/analyze-image?note=${encodeURIComponent(note)}`, data);
  },
  confirmTrainingSpec: (spec) => request('/api/training/confirm-spec', {
    method: 'POST',
    body: JSON.stringify({ spec }),
  }),
  startTraining: (buildingId) => request('/api/training/start', {
    method: 'POST',
    body: JSON.stringify({ building_id: buildingId }),
  }),
  trainingJob: (jobId) => request(`/api/training/jobs/${encodeURIComponent(jobId)}`),
  models: () => request('/api/models'),
  selectModel: (buildingId, modelId) => request('/api/models/select', {
    method: 'POST',
    body: JSON.stringify({ building_id: buildingId, model_id: modelId }),
  }),
  generateEnergyComfortReport: (payload = {}) => request('/api/reports/energy-comfort', {
    method: 'POST',
    body: JSON.stringify(payload),
  }),
  reportDownloadUrl: (downloadUrl) => `${API_BASE}${downloadUrl}`,
};

export { API_BASE };
