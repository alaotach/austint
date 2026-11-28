/* M5: Cytoscape.js Knowledge Graph View (Legend, LOD Labels, Neighborhood Hover, Detail Panel) */

(function () {
  "use strict";

  let cyInstance = null;
  let graphTypeFilter = "all";
  let graphMinConf = 0;
  let graphMaxDepth = 4;
  let graphLayout = "cose";
  let selectedGraphNodeId = null;
  let hasAnimatedInitial = false;

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

  const esc = (str) => (window.OSINT_UI?.esc ? window.OSINT_UI.esc(str) : String(str ?? "")
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;"));

  function renderDetailPanel(node) {
    const panel = document.getElementById("graph-detail-panel");
    if (!panel) return;
    if (!node) {
      panel.innerHTML = `
        <div class="field-label">Node Detail</div>
        <div class="state-panel" style="padding:14px;">Click any node in the graph to inspect its summary and neighbors.</div>
      `;
      return;
    }
    const d = node.data();
    const neighbors = node.neighborhood("node").slice(0, 8);
    const conf = Math.round(Number(d.confidence ?? 0.5) * 100);
    panel.innerHTML = `
      <div style="display:flex; justify-content:space-between; align-items:center; gap:8px;">
        <span class="badge badge-accent">${esc(d.type)}</span>
        <span class="badge ${conf >= 50 ? "badge-done" : "badge-unavailable"}">${conf}% conf</span>
      </div>
      <div style="font-weight:700; font-size:14px; word-break:break-word;">${esc(d.fullLabel || d.label)}</div>
      <div class="mono" style="font-size:11px; color:var(--text-muted);">depth=${esc(d.depth ?? 0)} &middot; degree=${node.degree()}</div>
      <button type="button" class="btn btn-sm btn-primary" id="btn-graph-open-inspector" style="width:100%;">Open Full Inspector &rarr;</button>
      <div style="border-top:1px solid var(--border-subtle); padding-top:10px; display:flex; flex-direction:column; gap:6px;">
        <div class="field-label" style="margin:0;">Connected (${node.neighborhood("node").length})</div>
        ${neighbors.length ? neighbors.map((nb) => {
          const nd = nb.data();
          return `<button type="button" class="btn btn-xs" style="justify-content:space-between; width:100%;" data-graph-focus="${esc(nd.id)}">
            <span style="overflow:hidden; text-overflow:ellipsis;">${esc(nd.displayLabel || nd.label)}</span>
            <span style="color:var(--text-muted);">${esc(nd.type)}</span>
          </button>`;
        }).join("") : `<div class="mono" style="font-size:11px; color:var(--text-muted);">No neighbors</div>`}
      </div>
    `;
    document.getElementById("btn-graph-open-inspector")?.addEventListener("click", () => {
      window.OSINT_UI?.openInspector(d.id);
    });
    panel.querySelectorAll("[data-graph-focus]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const target = cyInstance?.getElementById(btn.dataset.graphFocus);
        if (target && target.length) {
          cyInstance.animate({ center: { eles: target }, zoom: Math.max(cyInstance.zoom(), 1.2) }, { duration: 180 });
          target.select();
          selectedGraphNodeId = target.id();
          renderDetailPanel(target);
        }
      });
    });
  }

  function renderGraphView() {
    const container = document.getElementById("graph-container");
    if (!container) return;

    const allNodes = window.OSINT_STATE.nodes || [];
    const allEdges = window.OSINT_STATE.edges || [];
    const types = ["all", ...Array.from(new Set(allNodes.map((n) => n.data?.type).filter(Boolean)))];

    container.innerHTML = `
      <div class="card" style="padding:10px 14px; display:flex; flex-wrap:wrap; align-items:center; justify-content:space-between; gap:var(--space-2);">
        <div style="display:flex; flex-wrap:wrap; align-items:center; gap:var(--space-2);">
          <select class="select" id="graph-filter-type" style="width:auto; padding:5px 10px; font-size:12px;">
            ${types.map((t) => `<option value="${esc(t)}" ${graphTypeFilter === t ? "selected" : ""}>Type: ${esc(t)}</option>`).join("")}
          </select>
          <select class="select" id="graph-filter-depth" style="width:auto; padding:5px 10px; font-size:12px;">
            ${[1, 2, 3, 4].map((d) => `<option value="${d}" ${graphMaxDepth === d ? "selected" : ""}>Depth &le; ${d}</option>`).join("")}
          </select>
          <label class="mono" style="font-size:11px; color:var(--text-muted);" for="graph-filter-conf">Conf &ge; ${Math.round(graphMinConf * 100)}%</label>
          <input type="range" id="graph-filter-conf" min="0" max="0.9" step="0.1" value="${graphMinConf}" style="width:96px;">
        </div>
        <div style="display:flex; flex-wrap:wrap; gap:var(--space-2);">
          <select class="select" id="graph-layout-select" style="width:auto; padding:5px 10px; font-size:12px;">
            ${["cose", "breadthfirst", "concentric", "circle"].map((l) => `<option value="${l}" ${graphLayout === l ? "selected" : ""}>Layout: ${l}</option>`).join("")}
          </select>
          <button type="button" class="btn btn-sm" id="btn-graph-fit">Fit</button>
          <button type="button" class="btn btn-sm" id="btn-graph-png">PNG</button>
          <button type="button" class="btn btn-sm" id="btn-graph-json">JSON</button>
        </div>
      </div>

      ${allNodes.length ? `
        <div class="graph-layout">
          <div class="card" style="padding:0; overflow:hidden; display:flex; flex-direction:column;">
            <div id="cy-canvas" style="width:100%; height:calc(100vh - 260px); min-height:480px; background:var(--bg-canvas);"></div>
            <div class="graph-legend">
              ${Object.entries(TYPE_COLORS).map(([t, col]) => `
                <span class="graph-legend-item"><span class="graph-legend-dot" style="background:${col};"></span>${esc(t)}</span>
              `).join("")}
              <span style="margin-left:auto; color:var(--text-muted);">Size = confidence &middot; Hover = neighborhood</span>
            </div>
          </div>
          <aside class="card" id="graph-detail-panel" style="padding:16px; display:flex; flex-direction:column; gap:10px; overflow-y:auto; max-height:calc(100vh - 210px);">
          </aside>
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

    const isLight = window.OSINT_STATE.theme === "light";
    const labelCol = isLight ? "#09090b" : "#e4e4e7";
    const borderCol = isLight ? "#ffffff" : "#09090b";

    const cyNodes = filteredNodes.map((n) => {
      const d = n.data || {};
      const raw = String(d.label || "");
      const safeLabel = /^(breach|breach-analytics|infostealer):/i.test(raw) ? "[Sensitive]" : raw;
      return {
        data: {
          ...d,
          fullLabel: safeLabel,
          displayLabel: safeLabel.length > 22 ? safeLabel.slice(0, 20) + ".." : safeLabel,
          color: TYPE_COLORS[d.type] || "#94a3b8",
          size: Math.round(18 + Number(d.confidence ?? 0.5) * 24),
        },
      };
    });

    const cyEdges = filteredEdges.map((e) => ({
      data: {
        ...e.data,
        lineStyle: Number(e.data?.score ?? 1) < 0.5 ? "dashed" : "solid",
      },
    }));

    const animateOnce = !hasAnimatedInitial;
    hasAnimatedInitial = true;

    cyInstance = window.cytoscape({
      container: cyEl,
      elements: { nodes: cyNodes, edges: cyEdges },
      style: [
        {
          selector: "node",
          style: {
            "background-color": "data(color)",
            "label": "",
            "width": "data(size)",
            "height": "data(size)",
            "color": labelCol,
            "font-size": "10px",
            "font-family": "JetBrains Mono, monospace",
            "text-valign": "bottom",
            "text-margin-y": 4,
            "border-width": 1.5,
            "border-color": borderCol,
            "transition-property": "opacity, border-width, border-color",
            "transition-duration": "150ms",
          },
        },
        {
          selector: "node.lod-label, node:selected, node.hovered",
          style: {
            "label": "data(displayLabel)",
          },
        },
        {
          selector: "node:selected",
          style: {
            "border-width": 3,
            "border-color": "#38bdf8",
          },
        },
        {
          selector: "edge",
          style: {
            "width": 1.3,
            "line-color": isLight ? "#94a3b8" : "#475569",
            "target-arrow-color": isLight ? "#94a3b8" : "#475569",
            "target-arrow-shape": "triangle",
            "curve-style": "bezier",
            "line-style": "data(lineStyle)",
            "opacity": 0.65,
            "transition-property": "opacity, line-color",
            "transition-duration": "150ms",
          },
        },
        {
          selector: ".dimmed",
          style: {
            "opacity": 0.14,
          },
        },
      ],
      layout: {
        name: graphLayout,
        animate: animateOnce,
        animationDuration: animateOnce ? 360 : 0,
        animationEasing: "ease-out",
        padding: 48,
      },
    });

    const updateLodLabels = () => {
      if (!cyInstance) return;
      const zoom = cyInstance.zoom();
      cyInstance.nodes().forEach((n) => {
        if (zoom >= 1.15 || n.degree() >= 3) n.addClass("lod-label");
        else n.removeClass("lod-label");
      });
    };
    cyInstance.one("layoutstop", () => {
      cyInstance.fit(undefined, 48);
      updateLodLabels();
    });
    cyInstance.on("zoom", updateLodLabels);

    cyInstance.on("mouseover", "node", (evt) => {
      const node = evt.target;
      const hood = node.closedNeighborhood();
      cyInstance.elements().not(hood).addClass("dimmed");
      node.addClass("hovered");
    });
    cyInstance.on("mouseout", "node", (evt) => {
      cyInstance.elements().removeClass("dimmed");
      evt.target.removeClass("hovered");
    });

    cyInstance.on("tap", "node", (evt) => {
      selectedGraphNodeId = evt.target.id();
      renderDetailPanel(evt.target);
    });

    const initialSel = selectedGraphNodeId ? cyInstance.getElementById(selectedGraphNodeId) : null;
    renderDetailPanel(initialSel && initialSel.length ? initialSel : null);

    document.getElementById("btn-graph-fit")?.addEventListener("click", () => {
      if (cyInstance) cyInstance.fit(undefined, 48);
    });
    document.getElementById("btn-graph-png")?.addEventListener("click", () => {
      if (!cyInstance) return;
      const url = cyInstance.png({ full: true, bg: isLight ? "#f8f9fb" : "#09090b" });
      const a = document.createElement("a");
      a.href = url;
      a.download = `osint_graph_${window.OSINT_STATE.jobId || "export"}.png`;
      a.click();
      window.OSINT_UI?.toast("Exported graph PNG", "success");
    });
    document.getElementById("btn-graph-json")?.addEventListener("click", () => {
      const blob = new Blob([JSON.stringify({ nodes: filteredNodes, edges: filteredEdges }, null, 2)], { type: "application/json" });
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = `osint_graph_${window.OSINT_STATE.jobId || "export"}.json`;
      a.click();
      window.OSINT_UI?.toast("Exported graph JSON", "success");
    });
  }

  window.focusGraphSearchNode = function (query) {
    if (!cyInstance || !query) return;
    const match = cyInstance.nodes().filter((n) => {
      const lbl = String(n.data("fullLabel") || n.data("label") || "").toLowerCase();
      return lbl.includes(query);
    }).first();
    if (match && match.length) {
      cyInstance.nodes().unselect();
      match.select();
      cyInstance.animate({ center: { eles: match }, zoom: 1.35 }, { duration: 200 });
      selectedGraphNodeId = match.id();
      renderDetailPanel(match);
    }
  };

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
