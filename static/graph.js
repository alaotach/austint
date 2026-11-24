/**
 * Intelligent OSINT - AntV G6 Knowledge Graph Visualizer & Live Node Streamer
 */

let graph = null;
let rawGraphData = { elements: { nodes: [], edges: [] } };
let liveNodeCount = 0;
let liveEdgeCount = 0;

const TYPE_COLORS = {
  domain:   { fill: '#0c4a6e', stroke: '#38bdf8', glow: 'rgba(56, 189, 248, 0.55)' },
  username: { fill: '#4c1d95', stroke: '#a78bfa', glow: 'rgba(167, 139, 250, 0.55)' },
  profile:  { fill: '#881337', stroke: '#f43f5e', glow: 'rgba(244, 63, 94, 0.55)' },
  email:    { fill: '#78350f', stroke: '#fbbf24', glow: 'rgba(251, 191, 36, 0.55)' },
  org:      { fill: '#064e3b', stroke: '#34d399', glow: 'rgba(52, 211, 153, 0.55)' },
  place:    { fill: '#134e4a', stroke: '#2dd4bf', glow: 'rgba(45, 212, 191, 0.55)' },
  default:  { fill: '#1e293b', stroke: '#94a3b8', glow: 'rgba(148, 163, 184, 0.4)' },
};

function registerCustomG6Elements() {
  // Pulsing halo for root Seed Target node (depth = 0)
  G6.registerNode(
    'seed-pulse-node',
    {
      afterDraw(cfg, group) {
        const r = (cfg.size || 40) / 2;
        const color = (cfg.style && cfg.style.stroke) || '#38bdf8';

        const halo1 = group.addShape('circle', {
          zIndex: -3,
          attrs: { x: 0, y: 0, r, fill: color, opacity: 0.45 },
          name: 'halo-shape-1',
        });

        const halo2 = group.addShape('circle', {
          zIndex: -2,
          attrs: { x: 0, y: 0, r, fill: color, opacity: 0.3 },
          name: 'halo-shape-2',
        });

        group.sort();

        halo1.animate(
          (ratio) => ({ r: r + ratio * 20, opacity: 0.45 * (1 - ratio) }),
          { repeat: true, duration: 2400, easing: 'easeCubic' }
        );

        halo2.animate(
          (ratio) => ({ r: r + ratio * 14, opacity: 0.3 * (1 - ratio) }),
          { repeat: true, duration: 2400, delay: 900, easing: 'easeCubic' }
        );
      },
    },
    'circle'
  );

  // Flowing dashed animation along directed discovery edges
  G6.registerEdge(
    'flowing-edge',
    {
      afterDraw(cfg, group) {
        const shape = group.get('children')[0];
        if (!shape) return;
        let index = 0;
        shape.animate(
          () => {
            index = (index + 1) % 18;
            return { lineDash: [4, 4], lineDashOffset: -index };
          },
          { repeat: true, duration: 1800 }
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
    animateCfg: { duration: 500, easing: 'easeQuadOut' },
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
      style: { lineWidth: 2, cursor: 'pointer' },
      labelCfg: {
        position: 'bottom',
        offset: 10,
        style: {
          fill: '#f1f5f9',
          fontSize: 11,
          fontWeight: 500,
          fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
          background: { fill: '#0b1120', stroke: '#24344d', padding: [3, 7, 3, 7], radius: 4 },
        },
      },
    },
    defaultEdge: {
      type: 'flowing-edge',
      style: {
        stroke: '#334155',
        lineWidth: 1.6,
        endArrow: { path: G6.Arrow.triangle(7, 8, 2), d: 2, fill: '#475569' },
        cursor: 'pointer',
      },
      labelCfg: {
        autoRotate: true,
        style: {
          fill: '#94a3b8',
          fontSize: 9,
          fontWeight: 600,
          background: { fill: '#070b14', stroke: '#1e293b', padding: [2, 5, 2, 5], radius: 3 },
        },
      },
    },
    nodeStateStyles: {
      active: { lineWidth: 3, stroke: '#ffffff', shadowBlur: 18, shadowColor: '#38bdf8' },
      inactive: { opacity: 0.25 },
      selected: { lineWidth: 4, stroke: '#ffffff', shadowBlur: 20, shadowColor: '#ffffff' },
    },
    edgeStateStyles: {
      active: { stroke: '#38bdf8', lineWidth: 2.5, endArrow: { path: G6.Arrow.triangle(8, 9, 2), d: 2, fill: '#38bdf8' } },
      inactive: { opacity: 0.15 },
      selected: { stroke: '#38bdf8', lineWidth: 3 },
    },
  });

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
  graph.updateLayout(getG6LayoutConfig(name));
  setTimeout(() => {
    if (graph && !graph.get('destroyed')) {
      graph.fitView(60, undefined, true);
    }
  }, 550);
}

function fitGraph() {
  if (graph) graph.fitView(60, undefined, true);
}

function clearGraph() {
  if (graph) graph.clear();
  rawGraphData = { elements: { nodes: [], edges: [] } };
  liveNodeCount = 0;
  liveEdgeCount = 0;
  document.getElementById('header-node-count').textContent = '0';
  document.getElementById('header-edge-count').textContent = '0';
  resetInspector();
}

function wrapLabel(text, maxLen = 22) {
  if (!text || text.length <= maxLen) return text;
  return text.length > 36 ? text.slice(0, 33) + '...' : text;
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

  const edges = (cytoscapeElements.edges || []).map((e) => ({
    id: e.data.id,
    source: e.data.source,
    target: e.data.target,
    label: e.data.relation,
    relation: e.data.relation,
    score: e.data.score,
    sourceTool: e.data.source_tool,
  }));

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

    graph.data(convertToG6Data(rawGraphData.elements));
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

function addLiveNodeAndEdge(nodeId, label, etype, depth, parentId, relation, sourceTool) {
  if (!graph || graph.get('destroyed')) return;
  if (graph.findById(nodeId)) return;

  const palette = TYPE_COLORS[etype] || TYPE_COLORS.default;
  const isSeed = depth === 0;

  const container = document.getElementById('cy');
  const cx = (container.scrollWidth || 800) / 2;
  const cy = (container.scrollHeight || 600) / 2;

  let x = cx;
  let y = cy;
  if (!isSeed) {
    const angle = (liveNodeCount * 0.61803398875) * Math.PI * 2;
    const radius = 160 + (liveNodeCount % 3) * 70;
    x = cx + Math.cos(angle) * radius;
    y = cy + Math.sin(angle) * radius;
  }

  graph.addItem('node', {
    id: nodeId,
    x,
    y,
    label: wrapLabel(label),
    labelFull: label,
    entityType: etype,
    confidence: 1.0,
    depth,
    provenance: [sourceTool || 'stream'],
    type: isSeed ? 'seed-pulse-node' : 'circle',
    size: isSeed ? 42 : 28,
    style: {
      fill: palette.fill,
      stroke: isSeed ? '#ffffff' : palette.stroke,
      lineWidth: isSeed ? 3 : 2,
      shadowColor: palette.glow,
      shadowBlur: isSeed ? 22 : 12,
    },
    labelCfg: {
      position: 'bottom',
      offset: isSeed ? 12 : 8,
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
  });

  liveNodeCount++;
  document.getElementById('header-node-count').textContent = liveNodeCount;

  if (parentId && graph.findById(parentId)) {
    const edgeId = `live_edge_${liveEdgeCount++}`;
    graph.addItem('edge', {
      id: edgeId,
      source: parentId,
      target: nodeId,
      label: relation || 'account_on',
      relation: relation || 'account_on',
      score: 1.0,
      sourceTool: sourceTool || 'maigret',
    });
    document.getElementById('header-edge-count').textContent = liveEdgeCount;
  }

  if (liveNodeCount === 1 || liveNodeCount % 8 === 0) {
    graph.fitView(60);
  }
}
