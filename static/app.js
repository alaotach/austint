/**
 * Intelligent OSINT - Visually Rich Animated Knowledge Graph (AntV G6)
 */

let graph = null;
let currentJobId = null;
let eventSource = null;
let currentTab = 'structured';
let rawGraphData = { elements: { nodes: [], edges: [] } };

const TYPE_COLORS = {
  domain:   { fill: '#0c4a6e', stroke: '#38bdf8', glow: 'rgba(56, 189, 248, 0.55)' },
  username: { fill: '#4c1d95', stroke: '#a78bfa', glow: 'rgba(167, 139, 250, 0.55)' },
  profile:  { fill: '#881337', stroke: '#f43f5e', glow: 'rgba(244, 63, 94, 0.55)' },
  email:    { fill: '#78350f', stroke: '#fbbf24', glow: 'rgba(251, 191, 36, 0.55)' },
  org:      { fill: '#064e3b', stroke: '#34d399', glow: 'rgba(52, 211, 153, 0.55)' },
  place:    { fill: '#134e4a', stroke: '#2dd4bf', glow: 'rgba(45, 212, 191, 0.55)' },
  default:  { fill: '#1e293b', stroke: '#94a3b8', glow: 'rgba(148, 163, 184, 0.4)' },
};

document.addEventListener('DOMContentLoaded', () => {
  registerCustomG6Elements();
  initGraph();
  checkApiHealth();
});

function registerCustomG6Elements() {
  // 1. Pulsing animated halo for Seed Target node (depth = 0)
  G6.registerNode(
    'seed-pulse-node',
    {
      afterDraw(cfg, group) {
        const r = (cfg.size || 40) / 2;
        const color = (cfg.style && cfg.style.stroke) || '#38bdf8';

        const halo1 = group.addShape('circle', {
          zIndex: -3,
          attrs: {
            x: 0,
            y: 0,
            r: r,
            fill: color,
            opacity: 0.45,
          },
          name: 'halo-shape-1',
        });

        const halo2 = group.addShape('circle', {
          zIndex: -2,
          attrs: {
            x: 0,
            y: 0,
            r: r,
            fill: color,
            opacity: 0.3,
          },
          name: 'halo-shape-2',
        });

        group.sort();

        halo1.animate(
          (ratio) => ({
            r: r + ratio * 20,
            opacity: 0.45 * (1 - ratio),
          }),
          {
            repeat: true,
            duration: 2400,
            easing: 'easeCubic',
          }
        );

        halo2.animate(
          (ratio) => ({
            r: r + ratio * 14,
            opacity: 0.3 * (1 - ratio),
          }),
          {
            repeat: true,
            duration: 2400,
            delay: 900,
            easing: 'easeCubic',
          }
        );
      },
    },
    'circle'
  );

  // 2. Animated flowing dashed line on directed discovery edges
  G6.registerEdge(
    'flowing-edge',
    {
      afterDraw(cfg, group) {
        const shape = group.get('children')[0];
        if (!shape) return;
        let index = 0;
        shape.animate(
          () => {
            index++;
            if (index > 18) index = 0;
            return {
              lineDash: [4, 4],
              lineDashOffset: -index,
            };
          },
          {
            repeat: true,
            duration: 1800,
          }
        );
      },
    },
    'quadratic'
  );
}

function initGraph() {
  const container = document.getElementById('cy');
  const width = container.scrollWidth || 800;
  const height = container.scrollHeight || 600;

  const tooltip = new G6.Tooltip({
    offsetX: 12,
    offsetY: 12,
    itemTypes: ['node', 'edge'],
    getContent: (e) => {
      const model = e.item.getModel();
      const outDiv = document.createElement('div');
      if (e.item.getType() === 'node') {
        const conf = Math.round((model.confidence || 1.0) * 100);
        outDiv.innerHTML = `
          <div style="font-weight: 700; color: #f8fafc; margin-bottom: 4px;">${escapeHtml(model.labelFull || model.label)}</div>
          <div style="color: #94a3b8; font-size: 11px;">
            Type: <span style="color: #38bdf8; text-transform: uppercase; font-weight: 600;">${escapeHtml(model.entityType)}</span><br/>
            Confidence: <span style="color: #10b981; font-weight: 600;">${conf}%</span> &bull; Depth: ${model.depth}
          </div>
        `;
      } else {
        outDiv.innerHTML = `
          <div style="font-weight: 700; color: #38bdf8; margin-bottom: 2px;">${escapeHtml(model.relation)}</div>
          <div style="color: #94a3b8; font-size: 11px;">
            Tool: ${escapeHtml(model.sourceTool)} &bull; Score: ${model.score}
          </div>
        `;
      }
      return outDiv;
    },
  });

  const minimap = new G6.Minimap({
    size: [150, 100],
    className: 'g6-minimap',
    type: 'delegate',
  });

  graph = new G6.Graph({
    container: 'cy',
    width,
    height,
    fitView: true,
    fitViewPadding: [60, 60, 60, 60],
    animate: true,
    animateCfg: {
      duration: 500,
      easing: 'easeQuadOut',
    },
    plugins: [tooltip, minimap],
    modes: {
      default: [
        'drag-canvas',
        'zoom-canvas',
        'drag-node',
        {
          type: 'activate-relations',
          trigger: 'mouseenter',
          activeState: 'active',
          inactiveState: 'inactive',
          resetSelected: false,
        },
      ],
    },
    layout: getG6LayoutConfig('radial'),
    defaultNode: {
      type: 'circle',
      size: 32,
      style: {
        lineWidth: 2,
        cursor: 'pointer',
      },
      labelCfg: {
        position: 'bottom',
        offset: 10,
        style: {
          fill: '#f1f5f9',
          fontSize: 11,
          fontWeight: 500,
          fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
          background: {
            fill: '#0b1120',
            stroke: '#24344d',
            padding: [3, 7, 3, 7],
            radius: 4,
          },
        },
      },
    },
    defaultEdge: {
      type: 'flowing-edge',
      style: {
        stroke: '#334155',
        lineWidth: 1.6,
        endArrow: {
          path: G6.Arrow.triangle(7, 8, 2),
          d: 2,
          fill: '#475569',
        },
        cursor: 'pointer',
      },
      labelCfg: {
        autoRotate: true,
        style: {
          fill: '#94a3b8',
          fontSize: 9,
          fontWeight: 600,
          background: {
            fill: '#070b14',
            stroke: '#1e293b',
            padding: [2, 5, 2, 5],
            radius: 3,
          },
        },
      },
    },
    nodeStateStyles: {
      active: {
        lineWidth: 3,
        stroke: '#ffffff',
        shadowBlur: 18,
        shadowColor: '#38bdf8',
      },
      inactive: {
        opacity: 0.25,
      },
      selected: {
        lineWidth: 4,
        stroke: '#ffffff',
        shadowBlur: 20,
        shadowColor: '#ffffff',
      },
    },
    edgeStateStyles: {
      active: {
        stroke: '#38bdf8',
        lineWidth: 2.5,
        endArrow: {
          path: G6.Arrow.triangle(8, 9, 2),
          d: 2,
          fill: '#38bdf8',
        },
      },
      inactive: {
        opacity: 0.15,
      },
      selected: {
        stroke: '#38bdf8',
        lineWidth: 3,
      },
    },
  });

  // Click events for Inspector
  graph.on('node:click', (evt) => {
    clearSelectedStates();
    graph.setItemState(evt.item, 'selected', true);
    displayNodeDetails(evt.item.getModel());
  });

  graph.on('edge:click', (evt) => {
    clearSelectedStates();
    graph.setItemState(evt.item, 'selected', true);
    displayEdgeDetails(evt.item.getModel());
  });

  graph.on('canvas:click', () => {
    clearSelectedStates();
    resetInspector();
  });

  // Responsive resize
  window.addEventListener('resize', () => {
    if (!graph || graph.get('destroyed')) return;
    if (!container || !container.scrollWidth || !container.scrollHeight) return;
    graph.changeSize(container.scrollWidth, container.scrollHeight);
    graph.fitView(50);
  });
}

function clearSelectedStates() {
  if (!graph) return;
  graph.findAllByState('node', 'selected').forEach((n) => graph.setItemState(n, 'selected', false));
  graph.findAllByState('edge', 'selected').forEach((e) => graph.setItemState(e, 'selected', false));
}

function getG6LayoutConfig(name) {
  if (name === 'force') {
    return {
      type: 'force',
      preventOverlap: true,
      nodeSize: 130,
      nodeSpacing: 60,
      linkDistance: 220,
      nodeStrength: -450,
      edgeStrength: 0.35,
      collideStrength: 0.9,
      alphaDecay: 0.025,
    };
  }
  if (name === 'dagre') {
    return {
      type: 'dagre',
      rankdir: 'TB',
      nodesep: 75,
      ranksep: 100,
      controlPoints: true,
    };
  }
  // Default: Radial layout with zero label overlap
  return {
    type: 'radial',
    unitRadius: 210,
    preventOverlap: true,
    nodeSize: 140,
    nodeSpacing: 60,
    strictRadial: false,
    linkDistance: 200,
  };
}

function runGraphLayout(name) {
  if (!graph) return;
  const config = getG6LayoutConfig(name);
  graph.updateLayout(config);
  setTimeout(() => {
    if (graph && !graph.get('destroyed')) {
      graph.fitView(60, undefined, true);
    }
  }, 550);
}

function fitGraph() {
  if (graph) {
    graph.fitView(60, undefined, true);
  }
}

function clearGraph() {
  if (graph) {
    graph.clear();
  }
  rawGraphData = { elements: { nodes: [], edges: [] } };
  document.getElementById('header-node-count').textContent = '0';
  document.getElementById('header-edge-count').textContent = '0';
  resetInspector();
}

function wrapLabel(text, maxLen = 22) {
  if (!text || text.length <= maxLen) return text;
  if (text.length > 36) {
    return text.slice(0, 33) + '...';
  }
  return text;
}

function convertToG6Data(cytoscapeElements) {
  const nodes = (cytoscapeElements.nodes || []).map((n) => {
    const d = n.data;
    const etype = (d.type || 'default').toLowerCase();
    const palette = TYPE_COLORS[etype] || TYPE_COLORS.default;
    const isSeed = d.depth === 0;

    return {
      id: d.id,
      label: wrapLabel(d.label),
      labelFull: d.label,
      entityType: etype,
      confidence: d.confidence,
      depth: d.depth,
      provenance: d.provenance || [],
      type: isSeed ? 'seed-pulse-node' : 'circle',
      size: isSeed ? 42 : 30,
      style: {
        fill: palette.fill,
        stroke: isSeed ? '#ffffff' : palette.stroke,
        lineWidth: isSeed ? 3 : 2,
        shadowColor: palette.glow,
        shadowBlur: isSeed ? 22 : 12,
      },
      labelCfg: {
        position: 'bottom',
        offset: isSeed ? 12 : 9,
        style: {
          fontWeight: isSeed ? 700 : 500,
          fontSize: isSeed ? 12 : 11,
          background: {
            fill: '#0b1120',
            stroke: isSeed ? palette.stroke : '#24344d',
            padding: [3, 7, 3, 7],
            radius: 4,
          },
        },
      },
    };
  });

  const edges = (cytoscapeElements.edges || []).map((e) => {
    const d = e.data;
    return {
      id: d.id,
      source: d.source,
      target: d.target,
      label: d.relation,
      relation: d.relation,
      score: d.score,
      sourceTool: d.source_tool,
    };
  });

  return { nodes, edges };
}

async function loadGraphData(jobId) {
  try {
    const res = await fetch(`/api/graph/${jobId}`);
    if (!res.ok) return;
    const json = await res.json();
    rawGraphData = json.graph;

    document.getElementById('header-node-count').textContent = rawGraphData.stats.nodes_count || 0;
    document.getElementById('header-edge-count').textContent = rawGraphData.stats.edges_count || 0;

    const g6Data = convertToG6Data(rawGraphData.elements);
    graph.data(g6Data);
    graph.render();

    setTimeout(() => {
      if (graph && !graph.get('destroyed')) {
        graph.fitView(60, undefined, true);
      }
    }, 300);
  } catch (err) {
    console.error('Error loading graph:', err);
  }
}

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

  if (name === 'tool_call_started') {
    appendLog('TOOL', `Dispatched ${data.tool} on ${data.entity_id}`, 'badge-info');
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
