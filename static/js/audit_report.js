/* M6: Audit & Planner View + Report / Export View */

(function () {
  "use strict";

  let auditFilter = "all";

  const esc = (str) => (window.OSINT_UI?.esc ? window.OSINT_UI.esc(str) : String(str ?? "")
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;"));

  function syntaxHighlightJson(obj) {
    const raw = esc(JSON.stringify(obj, null, 2));
    return raw.replace(
      /(&quot;(?:\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\&])*&quot;)(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?/g,
      (match, str, colon, boolVal) => {
        if (str) {
          if (colon) return `<span class="json-key">${str}</span>:`;
          return `<span class="json-str">${str}</span>`;
        }
        if (boolVal) return `<span class="json-bool">${match}</span>`;
        return `<span class="json-num">${match}</span>`;
      }
    );
  }

  function summarizeAuditEvent(e) {
    const d = e.details || e.data || {};
    const actor = d.tool || d.planner || "--";
    let summary = "";
    if (e.event === "planner_decision") {
      summary = `tools=[${(d.chosen_tools || []).join(", ")}] fallback=${Boolean(d.fallback_used)} tokens=${d.token_usage || 0}`;
    } else if (e.event === "hit_evaluated") {
      const val = /^(breach|breach-analytics|infostealer):/i.test(String(d.discovered_value || "")) ? "[Sensitive]" : d.discovered_value;
      summary = `${d.discovered_type}: ${val} (score=${Number(d.resolver_score || 0).toFixed(2)}, accepted=${Boolean(d.accepted)})`;
    } else {
      summary = d.reason || d.error || (d.hits_returned != null ? `hits=${d.hits_returned}` : JSON.stringify(d));
    }
    return { actor: String(actor), summary: String(summary), details: d };
  }

  function renderAuditView() {
    const container = document.getElementById("audit-container");
    if (!container) return;

    const events = window.OSINT_STATE.auditEvents || [];
    const plannerEvents = events.filter((e) => e.event === "planner_decision");
    const toolEvents = events.filter((e) => e.event === "tool_call_started" || e.event === "tool_call_completed");
    const unavailable = events.filter((e) => e.event === "tool_unavailable" || e.event === "tool_call_error");
    const hitEvents = events.filter((e) => e.event === "hit_evaluated" || e.event === "gnn_graph_refined");
    const fallbacks = plannerEvents.filter((e) => Boolean((e.details || e.data)?.fallback_used)).length;
    const tokens = plannerEvents.reduce((acc, e) => acc + Number((e.details || e.data)?.token_usage || 0), 0);

    const filtered = events.filter((e) => {
      if (auditFilter === "planner") return e.event === "planner_decision";
      if (auditFilter === "tools") return e.event === "tool_call_started" || e.event === "tool_call_completed";
      if (auditFilter === "unavailable") return e.event === "tool_unavailable" || e.event === "tool_call_error";
      if (auditFilter === "hits") return e.event === "hit_evaluated" || e.event === "gnn_graph_refined";
      return true;
    });

    // Group consecutive identical events into expandable rows (xN)
    const slice = filtered.slice(-200).reverse();
    const grouped = [];
    for (const ev of slice) {
      const info = summarizeAuditEvent(ev);
      const sig = `${ev.event}|${info.actor}|${info.summary}`;
      const prev = grouped[grouped.length - 1];
      if (prev && prev.sig === sig) {
        prev.count += 1;
        prev.items.push(ev);
      } else {
        grouped.push({ sig, first: ev, info, count: 1, items: [ev] });
      }
    }

    container.innerHTML = `
      <div class="grid-4">
        <div class="card"><div class="kpi-label">Total Audit Events</div><div class="kpi-value">${events.length}</div></div>
        <div class="card"><div class="kpi-label">Planner Decisions</div><div class="kpi-value">${plannerEvents.length}</div></div>
        <div class="card"><div class="kpi-label">Fallbacks / Tokens</div><div class="kpi-value">${fallbacks} / ${tokens}</div></div>
        <div class="card"><div class="kpi-label">Skipped / Errors</div><div class="kpi-value">${unavailable.length}</div></div>
      </div>

      <div class="card" style="display:flex; flex-direction:column; gap:var(--space-3);">
        <div class="tab-bar" role="tablist">
          ${[
            ["all", "All", events.length],
            ["planner", "Planner Decisions", plannerEvents.length],
            ["tools", "Tool Calls", toolEvents.length],
            ["unavailable", "Skipped / Errors", unavailable.length],
            ["hits", "Hit Evaluations", hitEvents.length],
          ].map(([k, lbl, cnt]) => `<button type="button" class="tab-btn ${auditFilter === k ? "active" : ""}" role="tab" aria-selected="${auditFilter === k}" data-afilter="${k}">${lbl} (${cnt})</button>`).join("")}
        </div>

        ${grouped.length ? `
          <div class="data-table-wrap">
            <table class="data-table">
              <thead>
                <tr>
                  <th style="width:92px;">Time</th>
                  <th style="width:160px;">Event</th>
                  <th style="width:140px;">Tool / Planner</th>
                  <th>Details / Reason</th>
                </tr>
              </thead>
              <tbody>
                ${grouped.map((g, idx) => {
                  const ev = g.first;
                  const ts = String(ev.timestamp || "").slice(11, 19) || "--";
                  const badgeCls = ev.event.includes("error") || ev.event.includes("unavailable")
                    ? "badge-error"
                    : ev.event === "planner_decision"
                    ? "badge-accent"
                    : ev.event === "hit_evaluated"
                    ? "badge-done"
                    : "";
                  const jsonPayload = g.count > 1 ? g.items : g.info.details;
                  return `
                    <tr>
                      <td style="white-space:nowrap; color:var(--text-muted);">${esc(ts)}</td>
                      <td>
                        <span class="badge ${badgeCls}">${esc(ev.event)}</span>
                        ${g.count > 1 ? `<span class="badge badge-accent" style="margin-left:4px;">&times;${g.count}</span>` : ""}
                      </td>
                      <td>${esc(g.info.actor)}</td>
                      <td style="word-break:break-word;">
                        <details>
                          <summary style="cursor:pointer; list-style:none; display:flex; align-items:center; justify-content:space-between; gap:8px;">
                            <span>${esc(g.info.summary)}</span>
                            <span class="mono" style="font-size:10px; color:var(--accent); flex-shrink:0;">JSON &#9662;</span>
                          </summary>
                          <div style="margin-top:8px; position:relative;">
                            <button type="button" class="btn btn-xs" data-copy-audit="${idx}" style="position:absolute; top:6px; right:6px;">Copy</button>
                            <pre class="json-pretty">${syntaxHighlightJson(jsonPayload)}</pre>
                          </div>
                        </details>
                      </td>
                    </tr>
                  `;
                }).join("")}
              </tbody>
            </table>
          </div>
        ` : `<div class="state-panel">No audit events match this filter.</div>`}
      </div>
    `;

    container.querySelectorAll("[data-afilter]").forEach((btn) => {
      btn.addEventListener("click", () => {
        auditFilter = btn.dataset.afilter;
        renderAuditView();
      });
    });

    container.querySelectorAll("[data-copy-audit]").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        e.stopPropagation();
        const g = grouped[Number(btn.dataset.copyAudit)];
        if (!g || !navigator.clipboard) return;
        const payload = g.count > 1 ? g.items : g.info.details;
        navigator.clipboard.writeText(JSON.stringify(payload, null, 2));
        btn.textContent = "Copied";
        window.OSINT_UI?.toast("Copied audit JSON", "success");
        setTimeout(() => { btn.textContent = "Copy"; }, 1400);
      });
    });
  }

  function formatProvenanceCrumbs(prov) {
    const arr = Array.isArray(prov) ? prov : [];
    if (!arr.length) return `<span class="mono" style="color:var(--text-muted);">seed</span>`;
    const full = arr.join(" -> ");
    if (arr.length <= 2) {
      return `<span class="mono" data-tooltip="${esc(full)}">${arr.map((s) => esc(s)).join(" &rsaquo; ")}</span>`;
    }
    return `<span class="mono" data-tooltip="${esc(full)}">${esc(arr[0])} &rsaquo; <span style="color:var(--text-muted);">&hellip; (${arr.length - 2})</span> &rsaquo; ${esc(arr[arr.length - 1])}</span>`;
  }

  function renderReportView() {
    const container = document.getElementById("report-container");
    if (!container) return;

    const nodes = window.OSINT_STATE.nodes || [];
    const edges = window.OSINT_STATE.edges || [];
    const events = window.OSINT_STATE.auditEvents || [];

    const byType = {};
    let confSum = 0;
    let confirmed = 0;
    let lowConf = 0;
    nodes.forEach((n) => {
      const t = n.data?.type || "unknown";
      byType[t] = (byType[t] || 0) + 1;
      const c = Number(n.data?.confidence ?? 0.5);
      confSum += c;
      if (c >= 0.5) confirmed += 1;
      else lowConf += 1;
    });
    const avgConf = nodes.length ? (confSum / nodes.length).toFixed(3) : "0.000";

    container.innerHTML = `
      <div class="card" style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:var(--space-2);">
        <div class="mono" style="font-size:12px;">
          Job: <strong>${esc(window.OSINT_STATE.jobId || "none")}</strong>
          &middot; Nodes: <strong>${nodes.length}</strong>
          &middot; Edges: <strong>${edges.length}</strong>
        </div>
        <div class="no-print" style="display:flex; gap:var(--space-2);">
          <button type="button" class="btn btn-sm" id="btn-export-report-json">Export JSON</button>
          <button type="button" class="btn btn-sm btn-primary" id="btn-print-report">Print / PDF</button>
        </div>
      </div>

      ${nodes.length || events.length ? `
        <div class="grid-4">
          <div class="card"><div class="kpi-label">Total Nodes</div><div class="kpi-value">${nodes.length}</div></div>
          <div class="card"><div class="kpi-label">Confirmed (&ge;50%)</div><div class="kpi-value">${confirmed}</div></div>
          <div class="card"><div class="kpi-label">Low-Conf (&lt;50%)</div><div class="kpi-value">${lowConf}</div></div>
          <div class="card"><div class="kpi-label">Mean Confidence</div><div class="kpi-value">${avgConf}</div></div>
        </div>

        <div class="card">
          <div class="field-label" style="margin-bottom:10px;">Entity Breakdown (${confirmed} + ${lowConf} = ${nodes.length})</div>
          <div style="display:flex; flex-wrap:wrap; gap:8px;">
            ${Object.entries(byType).map(([k, v]) => `<span class="badge badge-accent">${esc(k)}: ${v}</span>`).join("")}
          </div>
        </div>

        <div class="card">
          <div class="field-label" style="margin-bottom:10px;">Top Confirmed Findings</div>
          <div class="data-table-wrap">
            <table class="data-table">
              <thead><tr><th>Type</th><th>Value</th><th>Confidence</th><th>Provenance</th></tr></thead>
              <tbody>
                ${nodes.slice().sort((a, b) => Number(b.data?.confidence || 0) - Number(a.data?.confidence || 0)).slice(0, 20).map((n) => {
                  const d = n.data || {};
                  const val = /^(breach|breach-analytics|infostealer):/i.test(String(d.label || "")) ? "[Sensitive Finding]" : d.label;
                  const pct = Math.round(Number(d.confidence || 0) * 100);
                  return `<tr>
                    <td><span class="badge">${esc(d.type)}</span></td>
                    <td style="word-break:break-word;">${esc(val)}</td>
                    <td><span class="badge ${pct >= 50 ? "badge-done" : "badge-unavailable"}">${pct}%</span></td>
                    <td>${formatProvenanceCrumbs(d.provenance)}</td>
                  </tr>`;
                }).join("")}
              </tbody>
            </table>
          </div>
        </div>
      ` : `<div class="state-panel">Run an investigation or load a recorded run to generate the summary report.</div>`}
    `;

    document.getElementById("btn-print-report")?.addEventListener("click", () => window.print());
    document.getElementById("btn-export-report-json")?.addEventListener("click", () => {
      const payload = {
        job_id: window.OSINT_STATE.jobId,
        nodes,
        edges,
        events_count: events.length,
      };
      const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = `osint_report_${window.OSINT_STATE.jobId || "export"}.json`;
      a.click();
      window.OSINT_UI?.toast("Exported report JSON", "success");
    });
  }

  window.renderAuditView = renderAuditView;
  window.renderReportView = renderReportView;
  window.onAuditUpdated = function () {
    if (window.OSINT_STATE.activeView === "audit") renderAuditView();
    if (window.OSINT_STATE.activeView === "report") renderReportView();
  };
  document.addEventListener("DOMContentLoaded", () => {
    renderAuditView();
    renderReportView();
  });
})();
