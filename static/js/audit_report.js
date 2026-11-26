/* M6: Audit & Planner View + Report / Export View */

(function () {
  "use strict";

  let auditFilter = "all";

  function esc(str) {
    return String(str ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function renderAuditView() {
    const container = document.getElementById("audit-container");
    if (!container) return;

    const events = window.OSINT_STATE.auditEvents || [];
    const plannerEvents = events.filter((e) => e.event === "planner_decision");
    const unavailable = events.filter((e) => e.event === "tool_unavailable" || e.event === "tool_call_error");
    const fallbacks = plannerEvents.filter((e) => Boolean(e.details?.fallback_used)).length;
    const tokens = plannerEvents.reduce((acc, e) => acc + Number(e.details?.token_usage || 0), 0);

    const filtered = events.filter((e) => {
      if (auditFilter === "planner") return e.event === "planner_decision";
      if (auditFilter === "tools") return e.event === "tool_call_started" || e.event === "tool_call_completed";
      if (auditFilter === "unavailable") return e.event === "tool_unavailable" || e.event === "tool_call_error";
      if (auditFilter === "hits") return e.event === "hit_evaluated" || e.event === "gnn_graph_refined";
      return true;
    });

    container.innerHTML = `
      <div class="grid-4">
        <div class="card"><div class="kpi-label">Total Audit Events</div><div class="kpi-value">${events.length}</div></div>
        <div class="card"><div class="kpi-label">Planner Decisions</div><div class="kpi-value">${plannerEvents.length}</div></div>
        <div class="card"><div class="kpi-label">Fallbacks / Tokens</div><div class="kpi-value">${fallbacks} / ${tokens}</div></div>
        <div class="card"><div class="kpi-label">Skipped / Errors</div><div class="kpi-value">${unavailable.length}</div></div>
      </div>

      <div class="card">
        <div class="tab-bar" style="margin-bottom: var(--space-3);">
          ${[
            ["all", "All"],
            ["planner", "Planner Decisions"],
            ["tools", "Tool Calls"],
            ["unavailable", "Skipped / Errors"],
            ["hits", "Hit Evaluations"],
          ].map(([k, lbl]) => `<button type="button" class="tab-btn ${auditFilter === k ? "active" : ""}" data-afilter="${k}">${lbl}</button>`).join("")}
        </div>

        ${filtered.length ? `
          <div class="data-table-wrap">
            <table class="data-table">
              <thead>
                <tr>
                  <th>Time</th>
                  <th>Event</th>
                  <th>Tool / Planner</th>
                  <th>Details / Reason</th>
                </tr>
              </thead>
              <tbody>
                ${filtered.slice(-150).reverse().map((e) => {
                  const d = e.details || {};
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
                  return `
                    <tr>
                      <td style="white-space:nowrap; color:var(--text-muted);">${esc(String(e.timestamp || "").slice(11, 19) || "--")}</td>
                      <td><span class="badge">${esc(e.event)}</span></td>
                      <td>${esc(actor)}</td>
                      <td style="word-break:break-word;">${esc(summary)}</td>
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
    let rejected = 0;
    nodes.forEach((n) => {
      const t = n.data?.type || "unknown";
      byType[t] = (byType[t] || 0) + 1;
      const c = Number(n.data?.confidence ?? 0.5);
      confSum += c;
      if (c >= 0.5) confirmed += 1;
      else rejected += 1;
    });
    const avgConf = nodes.length ? (confSum / nodes.length).toFixed(3) : "0.000";

    container.innerHTML = `
      <div class="card" style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:var(--space-2);">
        <div class="mono" style="font-size:12px;">Job: <strong>${esc(window.OSINT_STATE.jobId || "none")}</strong></div>
        <div style="display:flex; gap:var(--space-2);">
          <button type="button" class="btn btn-sm" id="btn-export-report-json">Export JSON</button>
          <button type="button" class="btn btn-sm btn-primary" id="btn-print-report">Print / PDF</button>
        </div>
      </div>

      ${nodes.length || events.length ? `
        <div class="grid-4">
          <div class="card"><div class="kpi-label">Total Entities</div><div class="kpi-value">${nodes.length}</div></div>
          <div class="card"><div class="kpi-label">Confirmed (&gt;=0.5)</div><div class="kpi-value">${confirmed}</div></div>
          <div class="card"><div class="kpi-label">Low-Conf (&lt;0.5)</div><div class="kpi-value">${rejected}</div></div>
          <div class="card"><div class="kpi-label">Mean Confidence</div><div class="kpi-value">${avgConf}</div></div>
        </div>

        <div class="card">
          <div class="field-label" style="margin-bottom:8px;">Entity Breakdown</div>
          <div style="display:flex; flex-wrap:wrap; gap:8px;">
            ${Object.entries(byType).map(([k, v]) => `<span class="badge badge-accent">${esc(k)}: ${v}</span>`).join("")}
          </div>
        </div>

        <div class="card">
          <div class="field-label" style="margin-bottom:8px;">Top Confirmed Findings</div>
          <div class="data-table-wrap">
            <table class="data-table">
              <thead><tr><th>Type</th><th>Value</th><th>Confidence</th><th>Provenance</th></tr></thead>
              <tbody>
                ${nodes.slice().sort((a, b) => Number(b.data?.confidence || 0) - Number(a.data?.confidence || 0)).slice(0, 20).map((n) => {
                  const d = n.data || {};
                  const val = /^(breach|breach-analytics|infostealer):/i.test(String(d.label || "")) ? "[Sensitive Finding]" : d.label;
                  return `<tr><td>${esc(d.type)}</td><td>${esc(val)}</td><td>${Math.round(Number(d.confidence || 0) * 100)}%</td><td>${esc((d.provenance || []).join(" -> "))}</td></tr>`;
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
