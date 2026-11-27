/* M2: Launch Screen & Live Scan View (per-collector progress, dot-grid, SSE streaming) */

(function () {
  "use strict";

  let collectorsCatalog = [];
  let collectorStatus = {}; // name -> {status, hits, reason, category}
  let activeEventSource = null;
  let elapsedTimer = null;
  let scanStartMs = 0;
  let scanBudget = { maxTools: 15, maxLlm: 10, toolsUsed: 0, llmUsed: 0 };
  let launchMode = "structured";
  let parsedSeeds = [];

  function esc(str) {
    return String(str ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  async function fetchCollectors() {
    try {
      const resp = await fetch("/api/collectors");
      if (!resp.ok) return;
      const data = await resp.json();
      collectorsCatalog = data.collectors || [];
      window.OSINT_COLLECTORS = collectorsCatalog;
      resetCollectorStatus();
      renderScanView();
    } catch (_) {
      // ignore offline
    }
  }

  function resetCollectorStatus() {
    collectorStatus = {};
    collectorsCatalog.forEach((c) => {
      collectorStatus[c.name] = {
        name: c.name,
        category: c.category || "General",
        status: "waiting",
        hits: 0,
        reason: "",
      };
    });
  }

  function renderLaunchView() {
    const container = document.getElementById("launch-container");
    if (!container) return;

    container.innerHTML = `
      <div style="display:flex; flex-direction:column; gap: var(--space-4);">
        <div class="tab-bar" role="tablist" aria-label="Launch Input Mode">
          <button type="button" class="tab-btn ${launchMode === "structured" ? "active" : ""}" data-lmode="structured">Structured Seed</button>
          <button type="button" class="tab-btn ${launchMode === "freetext" ? "active" : ""}" data-lmode="freetext">Free-Text Seeds</button>
          <button type="button" class="tab-btn ${launchMode === "image" ? "active" : ""}" data-lmode="image">Image / EXIF / GPS</button>
        </div>

        <div id="launch-panel-structured" style="display:${launchMode === "structured" ? "grid" : "none"}; gap: var(--space-3);" class="grid-2">
          <div>
            <label class="field-label" for="target-kind">Target Kind (Allowed Scope)</label>
            <select class="select" id="target-kind">
              <option value="domain">domain (Public Domain)</option>
              <option value="org">org (Public Organization)</option>
              <option value="place">place (Public Location)</option>
              <option value="public_figure">public_figure (Public Figure)</option>
              <option value="consenting_person">consenting_person (Consenting Person)</option>
              <option value="username">username (Public Handle)</option>
              <option value="email">email (Public Email)</option>
            </select>
          </div>
          <div>
            <label class="field-label" for="target-value">Seed Value</label>
            <input type="text" class="input" id="target-value" placeholder="e.g. example.com or torvalds" value="example.com">
          </div>
        </div>

        <div id="launch-panel-freetext" style="display:${launchMode === "freetext" ? "flex" : "none"}; flex-direction:column; gap: var(--space-2);">
          <label class="field-label" for="target-freetext">Natural Language Target Brief</label>
          <textarea class="textarea" id="target-freetext" placeholder="Paste public bio, organization description, or domain notes to extract seeds..."></textarea>
          <div style="display:flex; align-items:center; gap: var(--space-2); flex-wrap:wrap;">
            <button type="button" class="btn btn-sm" id="btn-parse-seeds">Extract Seeds</button>
            <span class="mono" style="font-size:12px; color: var(--text-muted);" id="parse-seeds-status"></span>
          </div>
          <div id="parsed-seeds-preview" style="display:flex; flex-wrap:wrap; gap: var(--space-2);"></div>
        </div>

        <div id="launch-panel-image" style="display:${launchMode === "image" ? "flex" : "none"}; flex-direction:column; gap: var(--space-3);">
          <label class="field-label" for="target-image-file">Drop or Select Image (EXIF, GPS, Reverse Search)</label>
          <div id="image-dropzone" class="state-panel" style="padding: var(--space-4); cursor:pointer;">
            <input type="file" id="target-image-file" accept="image/*" style="max-width:100%;">
            <div style="margin-top: var(--space-2); font-size:12px;">Extracts EXIF metadata, GPS coordinates, and public web matches.</div>
          </div>
          <div id="image-preview-result"></div>
        </div>

        <div class="grid-3">
          <div>
            <label class="field-label" for="param-depth">Max Depth: <strong id="val-depth">2</strong></label>
            <input type="range" id="param-depth" min="1" max="4" value="2" style="width:100%;">
          </div>
          <div>
            <label class="field-label" for="param-calls">Max Tool Calls: <strong id="val-calls">15</strong></label>
            <input type="range" id="param-calls" min="1" max="40" value="15" style="width:100%;">
          </div>
          <div>
            <label class="field-label" for="param-llm">Max LLM Calls: <strong id="val-llm">10</strong></label>
            <input type="range" id="param-llm" min="0" max="30" value="10" style="width:100%;">
          </div>
        </div>

        <div class="grid-2">
          <div>
            <label class="field-label" for="param-resolver">Entity Resolver</label>
            <select class="select" id="param-resolver">
              <option value="baseline">baseline (Weighted Heuristics)</option>
              <option value="learned">learned (Monotonic LogReg)</option>
              <option value="neural">neural (Multimodal CharCNN+CLIP)</option>
              <option value="neural_gnn">neural_gnn (Collective GNN)</option>
            </select>
          </div>
          <div>
            <label class="field-label" for="param-planner">Expansion Planner</label>
            <select class="select" id="param-planner">
              <option value="rule">rule (RuleBasedPlanner)</option>
              <option value="llm">llm (LLMPlanner)</option>
              <option value="bandit">bandit (LinUCB BanditPlanner)</option>
            </select>
          </div>
        </div>

        <div class="card" style="background: var(--bg-elevated); border-color: var(--border-strong);">
          <label style="display:flex; align-items:flex-start; gap: var(--space-3); cursor:pointer;">
            <input type="checkbox" id="consent-checkbox" style="margin-top: 3px;">
            <span style="font-size:13px;">
              <strong>Mandatory Authorization Gate:</strong> I confirm authorized, public-data-only reconnaissance for this target scope.
            </span>
          </label>
        </div>

        <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap: var(--space-3);">
          <div style="display:flex; gap: var(--space-2); flex-wrap:wrap;">
            <button type="button" class="btn btn-primary" id="btn-launch-scan" disabled>Start Investigation</button>
            <button type="button" class="btn" id="btn-load-demo">Load Recorded Run</button>
          </div>
          <span class="mono" style="font-size:12px; color: var(--rejected);" id="launch-error"></span>
        </div>
      </div>
    `;

    // Wire launch mode tabs
    container.querySelectorAll("[data-lmode]").forEach((btn) => {
      btn.addEventListener("click", () => {
        launchMode = btn.dataset.lmode;
        renderLaunchView();
      });
    });

    // Wire sliders
    const bindSlider = (id, outId) => {
      const el = document.getElementById(id);
      const out = document.getElementById(outId);
      if (el && out) {
        el.addEventListener("input", () => {
          out.textContent = el.value;
        });
      }
    };
    bindSlider("param-depth", "val-depth");
    bindSlider("param-calls", "val-calls");
    bindSlider("param-llm", "val-llm");

    // Consent gate blocks submit
    const consentBox = document.getElementById("consent-checkbox");
    const launchBtn = document.getElementById("btn-launch-scan");
    if (consentBox && launchBtn) {
      consentBox.addEventListener("change", () => {
        launchBtn.disabled = !consentBox.checked;
      });
    }

    // Parse seeds
    const parseBtn = document.getElementById("btn-parse-seeds");
    if (parseBtn) {
      parseBtn.addEventListener("click", async () => {
        const text = (document.getElementById("target-freetext")?.value || "").trim();
        const statusEl = document.getElementById("parse-seeds-status");
        const prevEl = document.getElementById("parsed-seeds-preview");
        if (!text) return;
        statusEl.textContent = "Parsing...";
        try {
          const r = await fetch("/api/parse-seeds", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text }),
          });
          const data = await r.json();
          parsedSeeds = data.seeds || [];
          statusEl.textContent = `Extracted ${parsedSeeds.length} seed(s)`;
          prevEl.innerHTML = parsedSeeds
            .map((s) => `<span class="chip badge-accent">${esc(s.kind)}: ${esc(s.value)}</span>`)
            .join("");
        } catch (err) {
          statusEl.textContent = `Error: ${err.message}`;
        }
      });
    }

    // Image upload
    const imgInput = document.getElementById("target-image-file");
    if (imgInput) {
      imgInput.addEventListener("change", async () => {
        const file = imgInput.files?.[0];
        const prev = document.getElementById("image-preview-result");
        if (!file || !prev) return;
        prev.innerHTML = `<div class="state-panel">Analyzing ${esc(file.name)}...</div>`;
        const fd = new FormData();
        fd.append("file", file);
        try {
          const r = await fetch("/api/analyze-image", { method: "POST", body: fd });
          const data = await r.json();
          prev.innerHTML = `<pre class="card mono" style="font-size:11px; overflow:auto; max-height:180px;">${esc(JSON.stringify(data, null, 2))}</pre>`;
        } catch (err) {
          prev.innerHTML = `<div class="state-panel state-panel-error">${esc(err.message)}</div>`;
        }
      });
    }

    // Launch scan
    if (launchBtn) {
      launchBtn.addEventListener("click", startInvestigation);
    }

    // Load recorded run
    const demoBtn = document.getElementById("btn-load-demo");
    if (demoBtn) {
      demoBtn.addEventListener("click", () => loadRecordedDemoRun(true));
    }
  }

  async function startInvestigation() {
    const errEl = document.getElementById("launch-error");
    const consent = Boolean(document.getElementById("consent-checkbox")?.checked);
    if (!consent) {
      if (errEl) errEl.textContent = "Consent checkbox is required before launching.";
      return;
    }
    if (errEl) errEl.textContent = "";

    const depth = Number(document.getElementById("param-depth")?.value || 2);
    const maxCalls = Number(document.getElementById("param-calls")?.value || 15);
    const maxLlm = Number(document.getElementById("param-llm")?.value || 10);
    const resolver = document.getElementById("param-resolver")?.value || "baseline";
    const planner = document.getElementById("param-planner")?.value || "rule";

    const payload = {
      consent_confirmed: true,
      max_depth: depth,
      max_tool_calls: maxCalls,
      max_llm_calls: maxLlm,
      resolver,
      planner,
    };

    if (launchMode === "freetext") {
      payload.free_text = document.getElementById("target-freetext")?.value || "";
      if (parsedSeeds.length) payload.seeds = parsedSeeds;
    } else {
      payload.target_kind = document.getElementById("target-kind")?.value || "domain";
      payload.target_value = (document.getElementById("target-value")?.value || "").trim();
    }

    scanBudget = { maxTools: maxCalls, maxLlm: maxLlm, toolsUsed: 0, llmUsed: 0 };
    resetCollectorStatus();
    window.OSINT_STATE.isRecordedRun = false;
    window.OSINT_STATE.nodes = [];
    window.OSINT_STATE.edges = [];
    window.OSINT_STATE.auditEvents = [];

    try {
      const resp = await fetch("/api/investigate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!resp.ok) {
        const msg = await resp.text();
        if (errEl) errEl.textContent = msg;
        return;
      }
      const data = await resp.json();
      window.OSINT_STATE.jobId = data.job_id;
      document.getElementById("header-job-id").textContent = data.job_id;
      setSystemStatus("running", "Scanning");
      document.getElementById("btn-stop-scan").disabled = false;
      window.OSINT_UI.switchView("scan");
      startElapsedTimer();
      connectSSE(data.job_id);
    } catch (err) {
      if (errEl) errEl.textContent = err.message;
    }
  }

  function setSystemStatus(kind, label) {
    const dot = document.getElementById("system-status-dot");
    const txt = document.getElementById("system-status-text");
    if (dot) dot.className = "status-dot " + kind;
    if (txt) txt.textContent = label;
  }

  function startElapsedTimer() {
    scanStartMs = Date.now();
    if (elapsedTimer) clearInterval(elapsedTimer);
    elapsedTimer = setInterval(() => {
      const el = document.getElementById("kpi-elapsed");
      if (el) el.textContent = ((Date.now() - scanStartMs) / 1000).toFixed(1) + "s";
    }, 250);
  }

  function stopElapsedTimer() {
    if (elapsedTimer) {
      clearInterval(elapsedTimer);
      elapsedTimer = null;
    }
  }

  function handleEvent(ev) {
    const st = window.OSINT_STATE;
    st.auditEvents.push(ev);
    const navAudit = document.getElementById("nav-badge-audit");
    if (navAudit) navAudit.textContent = String(st.auditEvents.length);

    const type = ev.event;
    const d = ev.details || {};

    if (type === "tool_call_started" && d.tool) {
      scanBudget.toolsUsed += 1;
      if (collectorStatus[d.tool]) collectorStatus[d.tool].status = "running";
    } else if (type === "tool_call_completed" && d.tool) {
      if (collectorStatus[d.tool]) {
        const hits = Number(d.hits_returned || 0);
        collectorStatus[d.tool].hits += hits;
        collectorStatus[d.tool].status = hits > 0 ? "found" : "not_found";
      }
      if (st.jobId && !st.isRecordedRun) refreshGraph(st.jobId);
    } else if (type === "tool_call_error" && d.tool) {
      if (collectorStatus[d.tool]) {
        collectorStatus[d.tool].status = "error";
        collectorStatus[d.tool].reason = d.error || "error";
      }
    } else if (type === "tool_unavailable" && d.tool) {
      if (collectorStatus[d.tool]) {
        collectorStatus[d.tool].status = "unavailable";
        collectorStatus[d.tool].reason = d.reason || "unavailable";
      }
    } else if (type === "planner_decision" && Number(d.token_usage || 0) > 0) {
      scanBudget.llmUsed += 1;
    } else if (type === "job_completed" || type === "job_failed" || type === "job_refused") {
      stopElapsedTimer();
      document.getElementById("btn-stop-scan").disabled = true;
      setSystemStatus(type === "job_completed" ? "completed" : "failed", type.replace("job_", ""));
      if (st.jobId) refreshGraph(st.jobId);
    }

    renderScanView();
    if (typeof window.onAuditUpdated === "function") window.onAuditUpdated();
  }

  function connectSSE(jobId) {
    if (activeEventSource) activeEventSource.close();
    activeEventSource = new EventSource(`/api/investigate/${encodeURIComponent(jobId)}/stream`);
    activeEventSource.onmessage = (msg) => {
      try {
        const ev = JSON.parse(msg.data);
        handleEvent(ev);
      } catch (_) {}
    };
    activeEventSource.onerror = () => {
      if (activeEventSource) {
        activeEventSource.close();
        activeEventSource = null;
      }
    };
  }

  async function refreshGraph(jobId) {
    try {
      const r = await fetch(`/api/graph/${encodeURIComponent(jobId)}`);
      if (!r.ok) return;
      const data = await r.json();
      const elems = data.graph?.elements || {};
      window.OSINT_STATE.nodes = elems.nodes || [];
      window.OSINT_STATE.edges = elems.edges || [];
      document.getElementById("header-node-count").textContent = String(window.OSINT_STATE.nodes.length);
      document.getElementById("header-edge-count").textContent = String(window.OSINT_STATE.edges.length);
      document.getElementById("nav-badge-results").textContent = String(window.OSINT_STATE.nodes.length);
      if (typeof window.onGraphUpdated === "function") window.onGraphUpdated();
    } catch (_) {}
  }

  async function loadRecordedDemoRun(switchToScan) {
    try {
      const r = await fetch("/api/demo-runs");
      if (!r.ok) return;
      const data = await r.json();
      const run = (data.runs || [])[0];
      if (!run) return;
      resetCollectorStatus();
      window.OSINT_STATE.isRecordedRun = true;
      window.OSINT_STATE.jobId = run.file.replace("audit_", "").replace(".jsonl", "");
      window.OSINT_STATE.auditEvents = [];
      const nodesMap = {};
      const edgesList = [];
      document.getElementById("header-job-id").textContent = window.OSINT_STATE.jobId;
      (run.events || []).forEach((ev) => {
        handleEvent(ev);
        if (ev.event === "hit_evaluated" && ev.details) {
          const d = ev.details;
          const id = `${d.discovered_type || "node"}:${d.discovered_value}`;
          const meta = d.metadata || {};
          if (!nodesMap[id]) {
            nodesMap[id] = {
              data: {
                id,
                label: d.discovered_value,
                type: d.discovered_type || "profile",
                confidence: Number(d.child_confidence ?? d.resolver_score ?? 0.5),
                depth: 1,
                provenance: [`${d.tool || "collector"}:discovered`],
                attributes: { ...meta, source_tool: d.tool || "", resolver_score: d.resolver_score },
              },
            };
          }
          edgesList.push({
            data: {
              id: `e_${edgesList.length}`,
              source: d.parent_id || "seed:target",
              target: id,
              relation: d.accepted ? "confirmed" : "rejected",
              score: Number(d.resolver_score ?? 0.5),
              source_tool: d.tool || "unknown",
            },
          });
        } else if (ev.event === "profile_discovered" && ev.details && ev.details.url) {
          const pd = ev.details;
          const id = `profile:${pd.url}`;
          if (nodesMap[id]) {
            Object.assign(nodesMap[id].data.attributes, {
              site: pd.site || nodesMap[id].data.attributes.site,
              url: pd.url,
              username: pd.username || nodesMap[id].data.attributes.username,
            });
          }
        }
      });
      window.OSINT_STATE.nodes = Object.values(nodesMap);
      window.OSINT_STATE.edges = edgesList;
      document.getElementById("header-node-count").textContent = String(window.OSINT_STATE.nodes.length);
      document.getElementById("header-edge-count").textContent = String(window.OSINT_STATE.edges.length);
      document.getElementById("nav-badge-results").textContent = String(window.OSINT_STATE.nodes.length);
      if (typeof window.onGraphUpdated === "function") window.onGraphUpdated();
      setSystemStatus("completed", "Recorded Run");
      if (switchToScan) window.OSINT_UI.switchView("scan");
    } catch (_) {}
  }
  window.loadRecordedDemoRun = loadRecordedDemoRun;

  function renderScanView() {
    const container = document.getElementById("scan-container");
    if (!container) return;

    const list = Object.values(collectorStatus);
    const total = list.length;
    const checked = list.filter((c) => c.status !== "waiting" && c.status !== "running").length;
    const matches = list.filter((c) => c.status === "found").length;
    const totalHits = list.reduce((acc, c) => acc + c.hits, 0);
    const elapsedSec = scanStartMs ? Math.max((Date.now() - scanStartMs) / 1000, 0.1) : 0;
    const rate = elapsedSec > 0 && checked > 0 ? (checked / elapsedSec).toFixed(2) : "0.00";

    const scanBadge = document.getElementById("nav-badge-scan");
    if (scanBadge) scanBadge.textContent = `${checked}/${total}`;

    // Category aggregation
    const cats = {};
    list.forEach((c) => {
      if (!cats[c.category]) cats[c.category] = { total: 0, done: 0, found: 0 };
      cats[c.category].total += 1;
      if (c.status !== "waiting" && c.status !== "running") cats[c.category].done += 1;
      if (c.status === "found") cats[c.category].found += 1;
    });

    const dotColor = (st) => {
      if (st === "found") return "var(--confirmed)";
      if (st === "running") return "var(--accent)";
      if (st === "error") return "var(--rejected)";
      if (st === "unavailable") return "var(--uncertain)";
      if (st === "not_found") return "var(--text-muted)";
      return "var(--border-strong)";
    };

    const statusBadgeClass = (st) => {
      if (st === "found") return "badge-confirmed";
      if (st === "running") return "badge-accent";
      if (st === "error") return "badge-error";
      if (st === "unavailable") return "badge-uncertain";
      return "";
    };

    const recentHits = (window.OSINT_STATE.auditEvents || [])
      .filter((e) => e.event === "hit_evaluated")
      .slice(-12)
      .reverse();

    container.innerHTML = `
      <div class="grid-4">
        <div class="card">
          <div class="kpi-label">Collectors Checked</div>
          <div class="kpi-value">${checked} / ${total}</div>
          <div class="mono" style="font-size:11px; color: var(--text-muted);">${rate} answers/s</div>
        </div>
        <div class="card">
          <div class="kpi-label">Matches / Hits</div>
          <div class="kpi-value">${matches} <span style="font-size:14px; color:var(--text-muted);">(${totalHits} hits)</span></div>
          <div class="mono" style="font-size:11px; color: var(--text-muted);">Elapsed: <span id="kpi-elapsed">${elapsedSec ? elapsedSec.toFixed(1) + "s" : "0.0s"}</span></div>
        </div>
        <div class="card">
          <div class="kpi-label">Tool Budget</div>
          <div class="kpi-value">${scanBudget.toolsUsed} / ${scanBudget.maxTools}</div>
          <div class="progress-track" style="margin-top:6px;"><div class="progress-fill" style="width:${Math.min(100, (scanBudget.toolsUsed / Math.max(1, scanBudget.maxTools)) * 100)}%;"></div></div>
        </div>
        <div class="card">
          <div class="kpi-label">LLM Budget</div>
          <div class="kpi-value">${scanBudget.llmUsed} / ${scanBudget.maxLlm}</div>
          <div class="progress-track" style="margin-top:6px;"><div class="progress-fill" style="width:${Math.min(100, (scanBudget.llmUsed / Math.max(1, scanBudget.maxLlm)) * 100)}%;"></div></div>
        </div>
      </div>

      <div class="grid-2">
        <div class="card">
          <div class="field-label" style="margin-bottom: var(--space-3);">Category Progress</div>
          <div style="display:flex; flex-direction:column; gap: var(--space-3);">
            ${Object.entries(cats).map(([cname, cdata]) => {
              const pct = Math.round((cdata.done / Math.max(1, cdata.total)) * 100);
              return `
                <div>
                  <div style="display:flex; justify-content:space-between; font-family:var(--font-mono); font-size:12px; margin-bottom:4px;">
                    <span>${esc(cname)}</span>
                    <span>${cdata.done}/${cdata.total} (${cdata.found} hit)</span>
                  </div>
                  <div class="progress-track"><div class="progress-fill" style="width:${pct}%;"></div></div>
                </div>
              `;
            }).join("")}
          </div>
        </div>

        <div class="card">
          <div class="field-label" style="margin-bottom: var(--space-3);">Collector Status Dot-Grid</div>
          <div style="display:flex; flex-wrap:wrap; gap: 6px; margin-bottom: var(--space-3);">
            ${list.map((c) => `
              <span title="${esc(c.name)}: ${esc(c.status)} (${c.hits} hits)" style="display:inline-flex; align-items:center; gap:5px; padding:3px 7px; border:1px solid var(--border-subtle); border-radius:4px; font-family:var(--font-mono); font-size:11px;">
                <span style="width:8px; height:8px; border-radius:50%; background:${dotColor(c.status)}; display:inline-block;"></span>
                ${esc(c.name)}
              </span>
            `).join("")}
          </div>
          <div style="display:flex; flex-wrap:wrap; gap:10px; font-family:var(--font-mono); font-size:11px; color:var(--text-muted);">
            <span><span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--confirmed);"></span> found</span>
            <span><span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--text-muted);"></span> not found</span>
            <span><span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--border-strong);"></span> waiting</span>
            <span><span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--uncertain);"></span> unavailable</span>
          </div>
        </div>
      </div>

      <div class="card">
        <div class="field-label" style="margin-bottom: var(--space-3);">Per-Collector Telemetry (${total} Collectors)</div>
        <div class="grid-3">
          ${list.map((c) => `
            <div style="padding: var(--space-2) var(--space-3); border:1px solid var(--border-subtle); border-radius:var(--radius-sm); display:flex; justify-content:space-between; align-items:center; gap:var(--space-2);">
              <div style="min-width:0;">
                <div class="mono" style="font-size:12px; font-weight:600; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">${esc(c.name)}</div>
                <div class="mono" style="font-size:11px; color:var(--text-muted);">${esc(c.category)}</div>
              </div>
              <span class="badge ${statusBadgeClass(c.status)}">${esc(c.status)}${c.hits > 0 ? " (" + c.hits + ")" : ""}</span>
            </div>
          `).join("")}
        </div>
      </div>

      <div class="card">
        <div class="field-label" style="margin-bottom: var(--space-2);">Live SSE Hit Stream</div>
        ${recentHits.length ? `
          <div style="display:flex; flex-direction:column; gap:6px;">
            ${recentHits.map((e) => {
              const dt = e.details || {};
              const acc = dt.accepted ? "badge-confirmed" : "badge-rejected";
              return `
                <div class="mono" style="font-size:12px; display:flex; justify-content:space-between; flex-wrap:wrap; gap:8px; padding:6px 8px; background:var(--bg-elevated); border-radius:4px;">
                  <span><strong>${esc(dt.tool)}</strong> -> ${esc(dt.discovered_type)}: ${esc(dt.discovered_value)}</span>
                  <span class="badge ${acc}">score=${Number(dt.resolver_score || 0).toFixed(2)} | conf=${Number(dt.child_confidence || 0).toFixed(2)}</span>
                </div>
              `;
            }).join("")}
          </div>
        ` : `<div class="state-panel">Waiting for streaming hit events...</div>`}
      </div>
    `;
  }

  document.addEventListener("DOMContentLoaded", () => {
    renderLaunchView();
    fetchCollectors();
    const stopBtn = document.getElementById("btn-stop-scan");
    if (stopBtn) {
      stopBtn.addEventListener("click", async () => {
        const jid = window.OSINT_STATE.jobId;
        if (!jid) return;
        stopBtn.disabled = true;
        try {
          await fetch(`/api/investigate/${encodeURIComponent(jid)}/stop`, { method: "POST" });
          setSystemStatus("failed", "Stopped");
          stopElapsedTimer();
        } catch (_) {}
      });
    }
  });
})();
