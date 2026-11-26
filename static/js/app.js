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
  document.getElementById('tab-structured').classList.toggle('active', tab === 'structured');
  document.getElementById('tab-freetext').classList.toggle('active', tab === 'freetext');
  document.getElementById('tab-image').classList.toggle('active', tab === 'image');

  document.getElementById('form-structured').style.display = tab === 'structured' ? 'block' : 'none';
  document.getElementById('form-freetext').style.display = tab === 'freetext' ? 'block' : 'none';
  document.getElementById('form-image').style.display = tab === 'image' ? 'block' : 'none';
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

  if (currentTab === 'image') {
    const fileInput = document.getElementById('target-image-file');
    if (!fileInput.files || !fileInput.files[0]) {
      alert('Please select an image file first.');
      btn.disabled = false;
      btnText.textContent = 'Start Investigation';
      return;
    }
    try {
      const formData = new FormData();
      formData.append('file', fileInput.files[0]);
      formData.append('consent_confirmed', 'true');
      const res = await fetch('/api/analyze-image', { method: 'POST', body: formData });
      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || 'Image analysis failed');
      }
      const data = await res.json();
      currentJobId = data.job_id;
      document.getElementById('header-job-id').textContent = currentJobId;
      appendLog('IMAGE', `Extracted EXIF & Geo for ${fileInput.files[0].name} (${data.stats.nodes_count} nodes)`, 'badge-success');
      await loadGraphData(currentJobId);
    } catch (err) {
      alert(`Error: ${err.message}`);
    } finally {
      btn.disabled = false;
      btnText.textContent = 'Start Investigation';
      document.getElementById('system-status-dot').className = 'status-dot';
      document.getElementById('system-status-text').textContent = 'Ready';
    }
    return;
  }

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

    const stopBtn = document.getElementById('btn-force-stop');
    if (stopBtn) {
      stopBtn.style.display = 'block';
      stopBtn.disabled = false;
    }

    appendLog('INIT', `Dispatched investigation job ${currentJobId}`, 'badge-info');

    connectEventStream(currentJobId);
    pollJobStatus(currentJobId);

  } catch (err) {
    alert(`Error: ${err.message}`);
    btn.disabled = false;
    btnText.textContent = 'Start Investigation';
    const stopBtn = document.getElementById('btn-force-stop');
    if (stopBtn) stopBtn.style.display = 'none';
    document.getElementById('system-status-dot').className = 'status-dot';
    document.getElementById('system-status-text').textContent = 'Error';
  }
}

async function forceStopInvestigation() {
  if (!currentJobId) return;
  const stopBtn = document.getElementById('btn-force-stop');
  if (stopBtn) stopBtn.disabled = true;

  try {
    const res = await fetch(`/api/investigate/${currentJobId}/stop`, { method: 'POST' });
    if (res.ok) {
      appendLog('STOP', `Force stopped investigation job ${currentJobId}`, 'badge-warning');
    }
  } catch (err) {
    appendLog('ALERT', `Force stop request failed: ${err.message}`, 'badge-danger');
  } finally {
    if (eventSource) {
      eventSource.close();
      eventSource = null;
    }
    const btn = document.getElementById('btn-dispatch');
    const btnText = document.getElementById('btn-dispatch-text');
    btn.disabled = false;
    btnText.textContent = 'Start Investigation';
    if (stopBtn) stopBtn.style.display = 'none';
    document.getElementById('system-status-dot').className = 'status-dot idle';
    document.getElementById('system-status-text').textContent = 'Stopped';
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
    const parentPrefix = (data.username || '').includes('@') ? 'email' : 'username';
    const parentId = `${parentPrefix}:${data.username}`;
    const relation = parentPrefix === 'email' ? 'registered_on' : 'account_on';
    const toolName = parentPrefix === 'email' ? 'email_intel' : 'maigret';
    addLiveNodeAndEdge(nodeId, data.site || data.url, 'profile', 1, parentId, relation, toolName);
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
  } else if (name === 'job_stopped') {
    appendLog('STOP', `${data.reason || 'Investigation force stopped'}`, 'badge-warning');
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

      if (data.status === 'completed' || data.status === 'failed' || data.status === 'refused' || data.status === 'stopped') {
        clearInterval(interval);
        const btn = document.getElementById('btn-dispatch');
        const btnText = document.getElementById('btn-dispatch-text');
        const stopBtn = document.getElementById('btn-force-stop');
        btn.disabled = false;
        btnText.textContent = 'Start Investigation';
        if (stopBtn) stopBtn.style.display = 'none';

        document.getElementById('system-status-dot').className = data.status === 'stopped' ? 'status-dot idle' : 'status-dot';
        document.getElementById('system-status-text').textContent = data.status === 'stopped' ? 'Stopped' : 'Ready';

        if (data.status !== 'stopped') {
          loadGraphData(jobId);
        }
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

function formatInspectorValue(val) {
  if (Array.isArray(val)) {
    return val.map(v => escapeHtml(typeof v === 'object' ? JSON.stringify(v) : String(v))).join(', ');
  }
  if (typeof val === 'object' && val !== null) {
    return escapeHtml(JSON.stringify(val));
  }
  const str = String(val);
  if (/^https?:\/\//i.test(str)) {
    return `<a href="${escapeHtml(str)}" target="_blank" rel="noopener noreferrer" style="color: var(--color-primary); word-break: break-all;">${escapeHtml(str)}</a>`;
  }
  return escapeHtml(str);
}

function displayNodeDetails(data) {
  const panel = document.getElementById('inspector-content');
  const typeClass = (data.entityType || '').toLowerCase();
  const prov = (data.provenance || []).join(', ') || 'seed';
  const attrs = data.attributes || {};

  // 1. Render all discovered profile/app images (avatar_url + images array)
  const imgList = [];
  if (attrs.avatar_url && typeof attrs.avatar_url === 'string' && /^https?:\/\//i.test(attrs.avatar_url)) {
    imgList.push(attrs.avatar_url);
  }
  if (Array.isArray(attrs.images)) {
    for (const im of attrs.images) {
      if (typeof im === 'string' && /^https?:\/\//i.test(im) && !imgList.includes(im)) {
        imgList.push(im);
      }
    }
  }
  let imagesHtml = '';
  if (imgList.length > 0) {
    imagesHtml = `
      <div class="inspector-field">
        <span class="inspector-key">Images (${imgList.length})</span>
        <div style="display: flex; flex-wrap: wrap; gap: 6px; margin-top: 6px;">
          ${imgList.slice(0, 8).map(src => `
            <a href="${escapeHtml(src)}" target="_blank" rel="noopener noreferrer" title="Open full image">
              <img src="${escapeHtml(src)}" alt="Profile media"
                   onerror="this.parentElement.style.display='none'"
                   style="width: 56px; height: 56px; object-fit: cover; border-radius: 6px; border: 1px solid var(--border-color, #2d3748);" />
            </a>
          `).join('')}
        </div>
      </div>
    `;
  }

  // 2. Render priority fields first, then dynamically render ALL remaining attributes + extra_fields
  let extraFields = '';
  const priorityKeys = [
    'name', 'legal_name', 'aliases', 'job_title', 'company', 'organizations',
    'bio', 'llm_summary', 'location', 'office_address',
    'email', 'contact_emails', 'support_email', 'developer_email', 'phone_numbers', 'support_phone',
    'website', 'privacy_policy_url', 'skills', 'stories', 'projects_or_stories',
    'ml_keywords', 'ml_cluster_id', 'followers', 'karma', 'joined',
    'risk_label', 'risk_score', 'exposed_data_types', 'password_exposure_note',
    'computer_name', 'operating_system', 'stealer_type', 'date_compromised', 'total_user_services'
  ];
  const skipKeys = new Set(['avatar_url', 'images', 'linked_urls', 'extra_fields', 'url', 'site']);
  const renderedKeys = new Set();

  const appendField = (label, rawVal) => {
    if (rawVal === undefined || rawVal === null || rawVal === '') return;
    if (Array.isArray(rawVal) && rawVal.length === 0) return;
    if (typeof rawVal === 'object' && !Array.isArray(rawVal) && Object.keys(rawVal).length === 0) return;
    extraFields += `
      <div class="inspector-field">
        <span class="inspector-key">${escapeHtml(label)}</span>
        <span class="inspector-val" style="font-size: 12px;">${formatInspectorValue(rawVal)}</span>
      </div>
    `;
  };

  for (const k of priorityKeys) {
    if (attrs[k] !== undefined) {
      appendField(k, attrs[k]);
      renderedKeys.add(k);
    }
  }

  // Dynamic open-ended keys on attrs
  for (const [k, v] of Object.entries(attrs)) {
    if (skipKeys.has(k) || renderedKeys.has(k)) continue;
    appendField(k, v);
    renderedKeys.add(k);
  }

  // Dynamic open-ended extra_fields dict (any newly discovered fields from scraper or LLM)
  if (attrs.extra_fields && typeof attrs.extra_fields === 'object' && !Array.isArray(attrs.extra_fields)) {
    for (const [ek, ev] of Object.entries(attrs.extra_fields)) {
      if (renderedKeys.has(ek)) continue;
      appendField(ek, ev);
      renderedKeys.add(ek);
    }
  }

  if (Array.isArray(attrs.linked_urls) && attrs.linked_urls.length > 0) {
    extraFields += `
      <div class="inspector-field">
        <span class="inspector-key">Linked Profiles / URLs</span>
        <span class="inspector-val" style="font-size: 11px;">${attrs.linked_urls.map(u => `<a href="${escapeHtml(u)}" target="_blank" rel="noopener noreferrer" style="color: var(--color-primary); word-break: break-all;">${escapeHtml(u)}</a>`).join('<br>')}</span>
      </div>
    `;
  }

  panel.innerHTML = `
    ${imagesHtml}
    <div class="inspector-field">
      <span class="inspector-key">Entity ID</span>
      <span class="inspector-val">${escapeHtml(data.id)}</span>
    </div>
    <div class="inspector-field">
      <span class="inspector-key">Value</span>
      <span class="inspector-val">${formatInspectorValue(data.labelFull || data.label)}</span>
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
