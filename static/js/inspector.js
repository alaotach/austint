/* M4: Inspector Drawer (Overview, Evidence, ML, Related, Raw JSON) */

(function () {
  "use strict";

  let activeTab = "overview";

  const esc = (str) => (window.OSINT_UI?.esc ? window.OSINT_UI.esc(str) : String(str ?? "")
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;"));

  function findNode(nodeId) {
    return (window.OSINT_STATE.nodes || []).find((n) => n.data?.id === nodeId) || null;
  }

  function sanitizeNodeForDisplay(nodeData) {
    const clone = JSON.parse(JSON.stringify(nodeData || {}));
    const attrs = clone.attributes || {};
    ["phone", "phones", "breach", "stealer_family", "top_passwords"].forEach((k) => {
      if (k in attrs) attrs[k] = "[HIDDEN - SENSITIVE]";
    });
    if (/^(breach|breach-analytics|infostealer):/i.test(String(clone.label || ""))) {
      clone.label = "[Sensitive Finding]";
    }
    return clone;
  }

  function renderInspector(nodeId) {
    const targetId = nodeId || window.OSINT_STATE.selectedNodeId;
    const node = findNode(targetId);
    const titleEl = document.getElementById("inspector-title");
    const typeEl = document.getElementById("inspector-type");
    const bodyEl = document.getElementById("inspector-body");
    if (!bodyEl) return;

    if (!node) {
      if (titleEl) titleEl.textContent = "Select a node";
      if (typeEl) typeEl.textContent = "NODE INSPECTOR";
      bodyEl.innerHTML = `<div class="state-panel">Click any result card or graph node to inspect evidence, ML scores, and neighbors.</div>`;
      return;
    }

    const d = node.data || {};
    const attrs = d.attributes || {};
    const isSensitive = /^(breach|breach-analytics|infostealer):/i.test(String(d.label || ""));
    const displayTitle = isSensitive ? `[Sensitive Finding: ${String(d.label).split(":")[0]}]` : d.label;

    if (titleEl) titleEl.textContent = displayTitle || d.id;
    if (typeEl) typeEl.textContent = `${String(d.type || "entity").toUpperCase()} // DEPTH ${d.depth ?? 0}`;

    const tabs = [
      { id: "overview", label: "Overview" },
      { id: "evidence", label: "Evidence" },
      { id: "ml", label: "ML" },
      { id: "related", label: "Related" },
      { id: "raw", label: "Raw JSON" },
    ];

    let tabContent = "";
    if (activeTab === "overview") {
      const conf = Number(d.confidence ?? 0.5);
      const cov = d.field_coverage || { extracted_count: 0, expected_count: 0, missing_fields: [] };
      const kvRows = Object.entries(attrs)
        .filter(([k]) => !["resolver_features", "hand_features", "field_sources", "field_confidence", "ml_keywords", "ml_summary", "ml_cluster_id", "phone", "phones", "breach", "stealer_family", "images", "avatar_url", "avatar", "banner_url", "banner"].includes(k))
        .slice(0, 14);
      const imgs = Array.from(new Set([
        ...(attrs.avatar_url ? [attrs.avatar_url] : []),
        ...(attrs.avatar ? [attrs.avatar] : []),
        ...(Array.isArray(attrs.images) ? attrs.images : []),
      ].filter((u) => typeof u === "string" && u.startsWith("http"))));
      tabContent = `
        <div class="grid-2" style="margin-bottom: var(--space-4);">
          <div class="card">
            <div class="kpi-label">Calibrated Confidence</div>
            <div class="kpi-value">${Math.round(conf * 100)}%</div>
          </div>
          <div class="card">
            <div class="kpi-label">Field Coverage</div>
            <div class="kpi-value">${cov.extracted_count ?? 0} / ${cov.expected_count ?? 0}</div>
          </div>
        </div>
        ${imgs.length ? `
          <div class="card" style="margin-bottom: var(--space-3);">
            <div class="field-label" style="margin-bottom:8px;">Images &amp; Media (${imgs.length})</div>
            <div style="display:flex; flex-wrap:wrap; gap:8px;">
              ${imgs.slice(0, 10).map((u) => `
                <a href="${esc(u)}" target="_blank" rel="noopener noreferrer" title="${esc(u)}" style="display:inline-block; border:1px solid var(--border-strong); border-radius:6px; overflow:hidden; background:var(--bg-elevated);">
                  <img src="${esc(u)}" alt="" loading="lazy" onerror="this.parentElement.style.display='none'" style="width:64px; height:64px; object-fit:cover; display:block;">
                </a>
              `).join("")}
            </div>
          </div>
        ` : ""}
        <div class="card" style="margin-bottom: var(--space-3);">
          <div class="field-label" style="margin-bottom:8px;">Extracted Public Fields</div>
          ${kvRows.length ? kvRows.map(([k, v]) => {
            const valHtml = typeof v === "string" && /^https?:\/\//i.test(v)
              ? `<a href="${esc(v)}" target="_blank" rel="noopener noreferrer" style="color:var(--accent);">${esc(v)}</a>`
              : esc(typeof v === "object" ? JSON.stringify(v) : v);
            return `
            <div class="mono" style="font-size:12px; display:flex; justify-content:space-between; gap:8px; padding:4px 0; border-bottom:1px solid var(--border-subtle);">
              <span style="color:var(--text-muted);">${esc(k)}</span>
              <span style="text-align:right; word-break:break-word; max-width:65%;">${valHtml}</span>
            </div>`;
          }).join("") : `<div class="state-panel">Sparse node: no additional public fields.</div>`}
        </div>
      `;
    } else if (activeTab === "evidence") {
      const prov = d.provenance || [];
      tabContent = `
        <div class="card" style="margin-bottom: var(--space-3);">
          <div class="field-label" style="margin-bottom:8px;">Provenance Chain</div>
          ${prov.length ? `
            <div style="display:flex; flex-direction:column; gap:6px;">
              ${prov.map((step, i) => `
                <div class="mono" style="font-size:12px; padding:6px 8px; background:var(--bg-elevated); border-radius:4px;">
                  [${i + 1}] ${esc(step)}
                </div>
              `).join("")}
            </div>
          ` : `<div class="state-panel">Seed node (depth 0).</div>`}
        </div>
        <div class="card">
          <div class="field-label" style="margin-bottom:8px;">Source Collector Metadata</div>
          <div class="mono" style="font-size:12px;">source_tool: ${esc(attrs.source_tool || "seed")}</div>
          <div class="mono" style="font-size:12px;">site: ${esc(attrs.site || "--")}</div>
        </div>
      `;
    } else if (activeTab === "ml") {
      const feats = attrs.resolver_features || attrs.hand_features || {};
      const pairwise = d.phase_c_score ?? attrs.resolver_score ?? d.confidence;
      const gnn = d.gnn_confidence;
      const chain = d.chain_confidence;
      const keywords = attrs.keywords || attrs.skills || [];
      const entities = attrs.entities || [];
      tabContent = `
        <div class="grid-3" style="margin-bottom: var(--space-3);">
          <div class="card">
            <div class="kpi-label">Pairwise Score</div>
            <div class="kpi-value">${pairwise != null ? Number(pairwise).toFixed(3) : "--"}</div>
          </div>
          <div class="card">
            <div class="kpi-label">GNN Score</div>
            <div class="kpi-value">${gnn != null ? Number(gnn).toFixed(3) : "--"}</div>
          </div>
          <div class="card">
            <div class="kpi-label">Chain Score</div>
            <div class="kpi-value">${chain != null ? Number(chain).toFixed(3) : "--"}</div>
          </div>
        </div>
        <div class="card" style="margin-bottom: var(--space-3);">
          <div class="field-label" style="margin-bottom:8px;">Resolver Feature Contributions</div>
          ${Object.keys(feats).length ? Object.entries(feats).map(([k, v]) => {
            const pct = Math.min(100, Math.max(0, Math.round(Number(v) * 100)));
            return `
              <div style="margin-bottom:8px;">
                <div class="mono" style="font-size:11px; display:flex; justify-content:space-between; margin-bottom:2px;">
                  <span>${esc(k)}</span><span>${Number(v).toFixed(3)}</span>
                </div>
                <div class="progress-track"><div class="progress-fill" style="--progress:${pct / 100};"></div></div>
              </div>
            `;
          }).join("") : `<div class="state-panel">No per-feature vector recorded for this node.</div>`}
        </div>
        <div class="card">
          <div class="field-label" style="margin-bottom:6px;">ML Summary, Keywords &amp; Cluster</div>
          <div class="mono" style="font-size:12px; margin-bottom:6px;">cluster_id: ${esc(attrs.cluster_id ?? "unassigned")}</div>
          ${attrs.ml_summary ? `<p style="font-size:12px; margin:0 0 8px 0;">${esc(attrs.ml_summary)}</p>` : ""}
          <div style="display:flex; flex-wrap:wrap; gap:4px;">
            ${keywords.map((k) => `<span class="chip badge-accent">${esc(k)}</span>`).join("")}
            ${entities.map((e) => `<span class="chip">${esc(typeof e === "object" ? e.text || JSON.stringify(e) : e)}</span>`).join("")}
          </div>
        </div>
      `;
    } else if (activeTab === "related") {
      const edges = (window.OSINT_STATE.edges || []).filter(
        (e) => e.data?.source === d.id || e.data?.target === d.id
      );
      tabContent = `
        <div class="card">
          <div class="field-label" style="margin-bottom:8px;">Connected Neighbors (${edges.length})</div>
          ${edges.length ? `
            <div style="display:flex; flex-direction:column; gap:6px;">
              ${edges.map((e) => {
                const ed = e.data || {};
                const otherId = ed.source === d.id ? ed.target : ed.source;
                const dir = ed.source === d.id ? "->" : "<-";
                return `
                  <button type="button" class="btn" style="justify-content:space-between; width:100%;" data-jump-node="${esc(otherId)}">
                    <span style="overflow:hidden; text-overflow:ellipsis;">${dir} ${esc(otherId)}</span>
                    <span class="badge">${esc(ed.relation)} (${Number(ed.score ?? 1).toFixed(2)})</span>
                  </button>
                `;
              }).join("")}
            </div>
          ` : `<div class="state-panel">No connected neighbor edges.</div>`}
        </div>
      `;
    } else if (activeTab === "raw") {
      const safeObj = sanitizeNodeForDisplay(d);
      tabContent = `
        <div class="card">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
            <span class="field-label" style="margin:0;">Raw Node JSON</span>
            <button type="button" class="btn btn-sm" id="btn-copy-raw-json">Copy JSON</button>
          </div>
          <pre class="json-pretty" style="max-height:460px;">${esc(JSON.stringify(safeObj, null, 2))}</pre>
        </div>
      `;
    }

    bodyEl.innerHTML = `
      <div class="tab-bar" role="tablist" style="margin-bottom: var(--space-4);">
        ${tabs.map((t) => `<button type="button" class="tab-btn ${activeTab === t.id ? "active" : ""}" data-itab="${t.id}">${esc(t.label)}</button>`).join("")}
      </div>
      ${tabContent}
    `;

    bodyEl.querySelectorAll("[data-itab]").forEach((btn) => {
      btn.addEventListener("click", () => {
        activeTab = btn.dataset.itab;
        renderInspector(d.id);
      });
    });

    bodyEl.querySelectorAll("[data-jump-node]").forEach((btn) => {
      btn.addEventListener("click", () => {
        window.OSINT_UI.openInspector(btn.dataset.jumpNode);
      });
    });

    const copyBtn = document.getElementById("btn-copy-raw-json");
    if (copyBtn) {
      copyBtn.addEventListener("click", () => {
        if (navigator.clipboard) {
          navigator.clipboard.writeText(JSON.stringify(sanitizeNodeForDisplay(d), null, 2));
          copyBtn.textContent = "Copied";
          window.OSINT_UI?.toast("Copied node JSON", "success");
          setTimeout(() => { copyBtn.textContent = "Copy JSON"; }, 1400);
        }
      });
    }
  }

  window.renderInspector = renderInspector;
})();
