/* M5: Cytoscape.js Knowledge Graph View */

(function () {
  "use strict";

  let cyInstance = null;
  let graphTypeFilter = "all";
  let graphMinConf = 0;
  let graphMaxDepth = 4;
  let graphLayout = "cose";

  const TYPE_COLORS = {
    profile: "#38bdf8",
    username: "#a78bfa",
    email: "#f59e0b",
    domain: "#22c55e",
    org: "#f472b6",
    place: "#14b8a6",
    public_figure: "#06b6d4",
    consenting_person: "#60a5fa",
  };

  function esc(str) {
    return String(str ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function renderGraphView() {
    const container = document.getElementById("graph-container");
    if (!container) return;

    const allNodes = window.OSINT_STATE.nodes || [];
    const allEdges = window.OSINT_STATE.edges || [];
    const types = ["all", ...Array.from(new Set(allNodes.map((n) => n.data?.type).filter(Boolean)))];

    container.innerHTML = `
      <div class="card" style="display:flex; flex-wrap:wrap; align-items:center; justify-content:space-between; gap:var(--space-2);">
        <div style="display:flex; flex-wrap:wrap; align-items:center; gap:var(--space-2);">
          <select class="select" id="graph-filter-type" style="width:auto;">
            ${types.map((t) => `<option value="${esc(t)}" ${graphTypeFilter === t ? "selected" : ""}>Type: ${esc(t)}</option>`).join("")}
          </select>
          <select class="select" id="graph-filter-depth" style="width:auto;">
            ${[1, 2, 3, 4].map((d) => `<option value="${d}" ${graphMaxDepth === d ? "selected" : ""}>Max Depth: ${d}</option>`).join("")}
          </select>
          <label class="mono" style="font-size:11px; color:var(--text-muted);" for="graph-filter-conf">Conf >= ${Math.round(graphMinConf * 100)}%</label>
          <input type="range" id="graph-filter-conf" min="0" max="0.9" step="0.1" value="${graphMinConf}">
        </div>
        <div style="display:flex; flex-wrap:wrap; gap:var(--space-2);">
          <select class="select" id="graph-layout-select" style="width:auto;">
            ${["cose", "breadthfirst", "concentric", "circle"].map((l) => `<option value="${l}" ${graphLayout === l ? "selected" : ""}>Layout: ${l}</option>`).join("")}
          </select>
          <button type="button" class="btn btn-sm" id="btn-graph-fit">Fit</button>
          <button type="button" class="btn btn-sm" id="btn-graph-png">Export PNG</button>
          <button type="button" class="btn btn-sm" id="btn-graph-json">Export JSON</button>
        </div>
      </div>

      ${allNodes.length ? `
        <div class="card" style="padding:0; overflow:hidden;">
          <div id="cy-canvas" style="width:100%; height:500px; background:var(--bg-canvas);"></div>
        </div>
      ` : `<div class="state-panel">No graph nodes yet. Run a scan or load a recorded run from Launch.</div>`}
    `;

    document.getElementById("graph-filter-type")?.addEventListener("change", (e) => {
      graphTypeFilter = e.target.value;
      renderGraphView();
    });
    document.getElementById("graph-filter-depth")?.addEventListener("change", (e) => {
      graphMaxDepth = Number(e.target.value);
      renderGraphView();
    });
    document.getElementById("graph-filter-conf")?.addEventListener("input", (e) => {
      graphMinConf = Number(e.target.value);
      renderGraphView();
    });
    document.getElementById("graph-layout-select")?.addEventListener("change", (e) => {
      graphLayout = e.target.value;
      renderGraphView();
    });

    const cyEl = document.getElementById("cy-canvas");
    if (!cyEl || !allNodes.length) return;

    const filteredNodes = allNodes.filter((n) => {
      const d = n.data || {};
      if (graphTypeFilter !== "all" && d.type !== graphTypeFilter) return false;
      if (Number(d.confidence ?? 0) < graphMinConf) return false;
      if (Number(d.depth ?? 0) > graphMaxDepth) return false;
      return true;
    });
    const keepIds = new Set(filteredNodes.map((n) => n.data?.id));
    const filteredEdges = allEdges.filter((e) => keepIds.has(e.data?.source) && keepIds.has(e.data?.target));

    if (typeof window.cytoscape !== "function") {
      cyEl.innerHTML = `<div class="state-panel" style="margin:16px;">Cytoscape.js unavailable. ${filteredNodes.length} nodes, ${filteredEdges.length} edges loaded.</div>`;
      return;
    }

    const cyNodes = filteredNodes.map((n) => {
      const d = n.data || {};
      const raw = String(d.label || "");
      const safeLabel = /^(breach|breach-analytics|infostealer):/i.test(raw) ? `[Sensitive]` : raw;
      return {
        data: {
          ...d,
          displayLabel: safeLabel.length > 24 ? safeLabel.slice(0, 22) + ".." : safeLabel,
          color: TYPE_COLORS[d.type] || "#94a3b8",
          size: Math.round(22 + Number(d.confidence ?? 0.5) * 26),
        },
      };
    });

    const cyEdges = filteredEdges.map((e) => ({
      data: {
        ...e.data,
        lineStyle: Number(e.data?.score ?? 1) < 0.5 ? "dashed" : "solid",
      },
    }));

    cyInstance = window.cytoscape({
      container: cyEl,
      elements: { nodes: cyNodes, edges: cyEdges },
      style: [
        {
          selector: "node",
          style: {
            "background-color": "data(color)",
            "label": "data(displayLabel)",
            "width": "data(size)",
            "height": "data(size)",
            "color": "#e2e8f0",
            "font-size": "10px",
            "font-family": "monospace",
            "text-valign": "bottom",
            "text-margin-y": 4,
            "border-width": 1.5,
            "border-color": "#0f172a",
          },
        },
        {
          selector: "edge",
          style: {
            "width": 1.5,
            "line-color": "#475569",
            "target-arrow-color": "#475569",
            "target-arrow-shape": "triangle",
            "curve-style": "bezier",
            "line-style": "data(lineStyle)",
          },
        },
      ],
      layout: { name: graphLayout, animate: false },
    });

    cyInstance.on("tap", "node", (evt) => {
      window.OSINT_UI.openInspector(evt.target.id());
    });

    document.getElementById("btn-graph-fit")?.addEventListener("click", () => {
      if (cyInstance) cyInstance.fit(undefined, 24);
    });
    document.getElementById("btn-graph-png")?.addEventListener("click", () => {
      if (!cyInstance) return;
      const url = cyInstance.png({ full: true, bg: "#090d14" });
      const a = document.createElement("a");
      a.href = url;
      a.download = `osint_graph_${window.OSINT_STATE.jobId || "export"}.png`;
      a.click();
    });
    document.getElementById("btn-graph-json")?.addEventListener("click", () => {
      const blob = new Blob([JSON.stringify({ nodes: filteredNodes, edges: filteredEdges }, null, 2)], { type: "application/json" });
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = `osint_graph_${window.OSINT_STATE.jobId || "export"}.json`;
      a.click();
    });
  }

  window.renderGraphView = renderGraphView;
  const prevHook = window.onGraphUpdatedHook;
  window.onGraphUpdatedHook = function () {
    if (typeof prevHook === "function") prevHook();
    if (window.OSINT_STATE.activeView === "graph") renderGraphView();
  };
  window.onViewActivated = function (view) {
    if (view === "graph") renderGraphView();
    if (view === "timeline" && typeof window.renderTimelineMapView === "function") window.renderTimelineMapView();
    if (view === "audit" && typeof window.renderAuditView === "function") window.renderAuditView();
    if (view === "report" && typeof window.renderReportView === "function") window.renderReportView();
  };
  document.addEventListener("DOMContentLoaded", renderGraphView);
})();
