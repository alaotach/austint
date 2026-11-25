/**
 * Intelligent OSINT - UI Controls, SSE Stream Client & Inspector Panel
 */

let currentJobId = null;
let eventSource = null;
let currentTab = 'structured';

document.addEventListener('DOMContentLoaded', () => {
  registerCustomG6Elements();
  initGraph();
  checkApiHealth();
});

async function checkApiHealth() {
  try {
    const res = await fetch('/api/health');
    const data = await res.json();
    if (data.status === 'online') {
      document.getElementById('system-status-dot').className = 'status-dot';
      document.getElementById('system-status-text').textContent = `Ready (${data.tools_count} Tools)`;
    }
  } catch (err) {
    document.getElementById('system-status-dot').className = 'status-dot idle';
    document.getElementById('system-status-text').textContent = 'Server Disconnected';
  }
}

function setTab(tab) {
  currentTab = tab;
  const tabStructured = document.getElementById('tab-structured');
  const tabFreetext = document.getElementById('tab-freetext');
  const formStructured = document.getElementById('form-structured');
  const formFreetext = document.getElementById('form-freetext');

  if (tab === 'structured') {
    tabStructured.classList.add('active');
    tabFreetext.classList.remove('active');
    formStructured.style.display = 'block';
    formFreetext.style.display = 'none';
  } else {
    tabFreetext.classList.add('active');
    tabStructured.classList.remove('active');
    formStructured.style.display = 'none';
    formFreetext.style.display = 'block';
  }
}

function toggleParams() {
  const content = document.getElementById('params-content');
  const arrow = document.getElementById('params-toggle-arrow');
  if (content.classList.contains('open')) {
    content.classList.remove('open');
    arrow.textContent = '▼';
  } else {
    content.classList.add('open');
    arrow.textContent = '▲';
  }
}

async function submitInvestigation() {
  const consentConfirmed = document.getElementById('consent-checkbox').checked;
  if (!consentConfirmed) {
    alert('Execution Refused: You must check the Scope Authorization Gate before running reconnaissance.');
    return;
  }

  const btn = document.getElementById('btn-dispatch');
  const btnText = document.getElementById('btn-dispatch-text');
  btn.disabled = true;
  btnText.textContent = 'Running Investigation...';

  document.getElementById('system-status-dot').className = 'status-dot running';
  document.getElementById('system-status-text').textContent = 'Agent Executing';

  const payload = {
    consent_confirmed: true,
    max_depth: parseInt(document.getElementById('param-depth').value) || 2,
    max_tool_calls: parseInt(document.getElementById('param-calls').value) || 15,
    confidence_decay: parseFloat(document.getElementById('param-decay').value) || 0.8,
    accept_threshold: parseFloat(document.getElementById('param-threshold').value) || 0.5,
    resolver: document.getElementById('param-resolver').value,
    planner: document.getElementById('param-planner').value,
    llm_provider: document.getElementById('param-llm-provider').value,
  };

  if (currentTab === 'structured') {
    payload.kind = document.getElementById('target-kind').value;
    payload.value = document.getElementById('target-value').value.trim();
  } else {
    payload.free_text = document.getElementById('target-freetext').value.trim();
  }

  try {
    const res = await fetch('/api/investigate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (!res.ok) {
      const errData = await res.json();
      throw new Error(errData.detail || 'Investigation failed to dispatch');
    }

    const data = await res.json();
    currentJobId = data.job_id;
    document.getElementById('header-job-id').textContent = currentJobId;

    appendLog('INIT', `Dispatched investigation job ${currentJobId}`, 'badge-info');

    connectEventStream(currentJobId);
    pollJobStatus(currentJobId);

  } catch (err) {
    alert(`Error: ${err.message}`);
    btn.disabled = false;
    btnText.textContent = 'Start Investigation';
    document.getElementById('system-status-dot').className = 'status-dot';
    document.getElementById('system-status-text').textContent = 'Error';
  }
}

function connectEventStream(jobId) {
  if (eventSource) {
    eventSource.close();
  }

  eventSource = new EventSource(`/api/investigate/${jobId}/stream`);

  eventSource.onmessage = function (event) {
    try {
      const evt = JSON.parse(event.data);
      handleLiveEvent(evt);
    } catch (e) {}
  };

  eventSource.onerror = function () {
    if (eventSource) {
      eventSource.close();
    }
  };
}

function handleLiveEvent(evt) {
  const name = evt.event;
  const data = evt.data || {};

  if (name === 'investigation_started') {
    liveNodeCount = 0;
    liveEdgeCount = 0;
    if (graph) graph.clear();
    const seedId = `${data.target_kind}:${data.target_value}`;
    addLiveNodeAndEdge(seedId, data.target_value, data.target_kind, 0, null, null, 'seed');
  } else if (name === 'tool_call_started') {
    appendLog('TOOL', `Dispatched ${data.tool} on ${data.entity_id}`, 'badge-info');
  } else if (name === 'profile_discovered') {
    appendLog('FOUND', `[${data.site}] ${data.url}`, 'badge-success');
    const nodeId = `profile:${data.url}`;
    const parentId = `username:${data.username}`;
    addLiveNodeAndEdge(nodeId, data.site || data.url, 'profile', 1, parentId, 'account_on', 'maigret');
  } else if (name === 'resolver_decision') {
    if (data.accepted) {
      appendLog('ACCEPT', `Admitted ${data.hit_value} (confidence: ${data.score})`, 'badge-success');
    } else {
      appendLog('REJECT', `Filtered ${data.hit_value} (score: ${data.score})`, 'badge-warning');
    }
  } else if (name === 'planner_decision') {
    const chosen = (data.chosen_tools || []).join(', ') || 'none';
    appendLog('PLAN', `Planner selected tools: [${chosen}]`, 'badge-info');
  } else if (name === 'job_completed') {
    appendLog('COMPLETE', `Finished: ${data.nodes_count} nodes, ${data.edges_count} edges`, 'badge-success');
    loadGraphData(currentJobId);
  } else if (name === 'job_failed' || name === 'job_refused') {
    appendLog('ALERT', `${data.error || 'Job failed'}`, 'badge-danger');
  }
}

async function pollJobStatus(jobId) {
  const interval = setInterval(async () => {
    try {
      const res = await fetch(`/api/investigate/${jobId}`);
      if (!res.ok) return;
      const data = await res.json();

      if (data.status === 'completed' || data.status === 'failed' || data.status === 'refused') {
        clearInterval(interval);
        const btn = document.getElementById('btn-dispatch');
        const btnText = document.getElementById('btn-dispatch-text');
        btn.disabled = false;
        btnText.textContent = 'Start Investigation';

        document.getElementById('system-status-dot').className = 'status-dot';
        document.getElementById('system-status-text').textContent = 'Ready';

        loadGraphData(jobId);
      }
    } catch (err) {}
  }, 1000);
}

function exportGraphJson() {
  if (!rawGraphData || !rawGraphData.elements) {
    alert('No graph data to export.');
    return;
  }
  const blob = new Blob([JSON.stringify(rawGraphData, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `osint_graph_${currentJobId || 'export'}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

function displayNodeDetails(data) {
  const panel = document.getElementById('inspector-content');
  const typeClass = (data.entityType || '').toLowerCase();
  const prov = (data.provenance || []).join(', ') || 'seed';
  const attrs = data.attributes || {};

  let extraFields = '';
  const displayKeys = ['name', 'bio', 'location', 'company', 'website', 'email', 'followers', 'karma', 'joined'];
  for (const k of displayKeys) {
    if (attrs[k] !== undefined && attrs[k] !== '' && attrs[k] !== null) {
      extraFields += `
        <div class="inspector-field">
          <span class="inspector-key">${escapeHtml(k)}</span>
          <span class="inspector-val" style="font-size: 12px;">${escapeHtml(String(attrs[k]))}</span>
        </div>
      `;
    }
  }
  if (Array.isArray(attrs.linked_urls) && attrs.linked_urls.length > 0) {
    extraFields += `
      <div class="inspector-field">
        <span class="inspector-key">Linked Profiles / URLs</span>
        <span class="inspector-val" style="font-size: 11px;">${attrs.linked_urls.map(u => escapeHtml(u)).join('<br>')}</span>
      </div>
    `;
  }

  panel.innerHTML = `
    <div class="inspector-field">
      <span class="inspector-key">Entity ID</span>
      <span class="inspector-val">${escapeHtml(data.id)}</span>
    </div>
    <div class="inspector-field">
      <span class="inspector-key">Value</span>
      <span class="inspector-val">${escapeHtml(data.labelFull || data.label)}</span>
    </div>
    <div class="inspector-field">
      <span class="inspector-key">Entity Type</span>
      <span class="tag ${typeClass}">${escapeHtml(data.entityType)}</span>
    </div>
    <div class="inspector-field">
      <span class="inspector-key">Confidence Score</span>
      <span class="inspector-val" style="color: var(--color-success); font-weight: 700;">${data.confidence}</span>
    </div>
    <div class="inspector-field">
      <span class="inspector-key">Chain Depth</span>
      <span class="inspector-val">${data.depth}</span>
    </div>
    ${extraFields}
    <div class="inspector-field">
      <span class="inspector-key">Provenance</span>
      <span class="inspector-val" style="font-size: 11px;">${escapeHtml(prov)}</span>
    </div>
  `;
}

function displayEdgeDetails(data) {
  const panel = document.getElementById('inspector-content');
  panel.innerHTML = `
    <div class="inspector-field">
      <span class="inspector-key">Relationship</span>
      <span class="inspector-val" style="color: var(--color-primary); font-weight: 700;">${escapeHtml(data.relation)}</span>
    </div>
    <div class="inspector-field">
      <span class="inspector-key">Origin Source Tool</span>
      <span class="inspector-val">${escapeHtml(data.sourceTool)}</span>
    </div>
    <div class="inspector-field">
      <span class="inspector-key">Edge Score</span>
      <span class="inspector-val">${data.score}</span>
    </div>
    <div class="inspector-field">
      <span class="inspector-key">Source ➔ Target</span>
      <span class="inspector-val" style="font-size: 11px;">${escapeHtml(data.source)}<br>➔ ${escapeHtml(data.target)}</span>
    </div>
  `;
}

function resetInspector() {
  const panel = document.getElementById('inspector-content');
  panel.innerHTML = `
    <p style="color: var(--text-dim); font-size: 12px; font-style: italic;">
      Click any node or edge in the knowledge graph to view detailed confidence scores, attributes, and origin tools.
    </p>
  `;
}

function appendLog(badge, msg, badgeClass = 'badge-info') {
  const logs = document.getElementById('stream-logs');
  const now = new Date().toTimeString().split(' ')[0];

  const entry = document.createElement('div');
  entry.className = 'log-entry';
  entry.innerHTML = `
    <span class="log-time">${now}</span>
    <span class="log-badge ${badgeClass}">${badge}</span>
    <span class="log-msg">${escapeHtml(msg)}</span>
  `;
  logs.appendChild(entry);
  logs.scrollTop = logs.scrollHeight;

  const countElem = document.getElementById('stream-count');
  const count = logs.querySelectorAll('.log-entry').length;
  countElem.textContent = `${count} Events`;
}

function escapeHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}
