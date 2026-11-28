/* M2: Launch Screen & Live Scan View (Run State Machine, Unified Matrix, Throttled SSE Stream) */

(function () {
  "use strict";

  let collectorsCatalog = [];
  let collectorStatus = {}; // name -> {status, hits, reason, category}
  let activeEventSource = null;
  let elapsedTimer = null;
  let scanStartMs = 0;
  let finalElapsedSec = null;
  let scanBudget = { maxTools: 15, maxLlm: 10, toolsUsed: 0, llmUsed: 0 };
  let launchMode = "structured";
  let parsedSeeds = [];
  const pendingDeletes = new Map(); // id -> timeoutId

  // SSE Hit Stream controls
  let streamPaused = false;
  let streamCollectorFilter = "all";
  let unseenHitsCount = 0;
  let lastHitAnimMs = 0;
  let recentHitTimestamps = [];
  let lastAnimatedHitKey = "";

  const esc = (str) => (window.OSINT_UI?.esc ? window.OSINT_UI.esc(str) : String(str ?? "")
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;"));

  function normalizeCollectorStatus(raw) {
    const s = String(raw || "waiting").toLowerCase();
    if (s === "found" || s === "done" || s === "completed" || s === "ok") return "done";
    if (s === "running" || s === "active") return "running";
    if (s === "error" || s === "failed") return "error";
    if (s === "unavailable" || s === "skipped") return "unavailable";
    if (s === "not_found" || s === "empty") return "not_found";
    return "waiting";
  }

  function statusCue(norm) {
    if (norm === "done") return "✓";
    if (norm === "running") return "●";
    if (norm === "error") return "✕";
    if (norm === "unavailable") return "–";
    if (norm === "not_found") return "·";
    return "○";
  }

  function statusBadgeClass(norm) {
    if (norm === "done") return "badge-done";
    if (norm === "running") return "badge-running";
    if (norm === "error") return "badge-error";
    if (norm === "unavailable") return "badge-unavailable";
    return "badge-waiting";
  }

  function statusSortRank(norm) {
    const order = { running: 0, error: 1, done: 2, not_found: 3, waiting: 4, unavailable: 5 };
    return order[norm] ?? 6;
  }

  function formatRelativeTime(isoStr) {
    if (!isoStr) return "--";
    const t = Date.parse(isoStr);
    if (Number.isNaN(t)) return String(isoStr).slice(0, 16).replace("T", " ");
    const diffSec = Math.max(0, Math.floor((Date.now() - t) / 1000));
    if (diffSec < 60) return `${diffSec}s ago`;
    const mins = Math.floor(diffSec / 60);
    if (mins < 60) return `${mins}m ago`;
    const hrs = Math.floor(mins / 60);
    if (hrs < 24) return `${hrs}h ago`;
    const days = Math.floor(hrs / 24);
    if (days < 30) return `${days}d ago`;
    return new Date(t).toISOString().slice(0, 10);
  }

  function computeElapsedFromEvents(events) {
    if (!Array.isArray(events) || events.length < 2) return null;
    const times = events
      .map((e) => Date.parse(e.timestamp || ""))
      .filter((t) => Number.isFinite(t));
    if (times.length < 2) return null;
    return Math.max(0.1, (Math.max(...times) - Math.min(...times)) / 1000);
  }

  function safeBudgetPair(used, max) {
    const u = Number.isFinite(Number(used)) ? Number(used) : null;
    const m = Number.isFinite(Number(max)) && Number(max) > 0 ? Number(max) : null;
    return {
      text: `${u !== null ? u : "--"} / ${m !== null ? m : "--"}`,
      ratio: u !== null && m !== null ? Math.min(1, Math.max(0, u / m)) : 0,
    };
  }

  function deriveRunStateFromCollectors() {
    const list = Object.values(collectorStatus);
    if (!list.length) return window.OSINT_STATE.nodes?.length ? "completed" : "idle";
    const anyRunning = list.some((c) => normalizeCollectorStatus(c.status) === "running");
    if (anyRunning) return "running";
    const anyFinished = list.some((c) => ["done", "not_found", "error", "unavailable"].includes(normalizeCollectorStatus(c.status)));
    if (!anyFinished && !(window.OSINT_STATE.nodes?.length)) return "idle";
    const anyIncomplete = list.some((c) => ["waiting", "error"].includes(normalizeCollectorStatus(c.status)));
    return anyIncomplete ? "partial" : "completed";
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
      <div class="launch-split">
        <section class="card" style="padding: var(--space-6); display:flex; flex-direction:column; gap: var(--space-5);">
          <div class="form-section">
            <div class="form-section-title">1. Seed Target</div>
            <div>
              <div class="segmented-bar" role="tablist" aria-label="Launch Input Mode">
                <button type="button" class="tab-btn ${launchMode === "structured" ? "active" : ""}" role="tab" aria-selected="${launchMode === "structured"}" data-lmode="structured">Structured Seed</button>
                <button type="button" class="tab-btn ${launchMode === "freetext" ? "active" : ""}" role="tab" aria-selected="${launchMode === "freetext"}" data-lmode="freetext">Free-Text Brief</button>
                <button type="button" class="tab-btn ${launchMode === "image" ? "active" : ""}" role="tab" aria-selected="${launchMode === "image"}" data-lmode="image">Image / EXIF</button>
              </div>
            </div>

            <div style="min-height: 118px;">
              <div id="launch-panel-structured" class="grid-2" style="display:${launchMode === "structured" ? "grid" : "none"}; gap: var(--space-4);">
                <div>
                  <label class="field-label" for="target-kind">Target Kind</label>
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

              <div id="launch-panel-freetext" style="display:${launchMode === "freetext" ? "flex" : "none"}; flex-direction:column; gap: var(--space-3);">
                <label class="field-label" for="target-freetext">Natural Language Target Brief</label>
                <textarea class="textarea" id="target-freetext" placeholder="Paste public bio, faculty profile, organization description, or domain notes to extract seeds..."></textarea>
                <div style="display:flex; align-items:center; gap: var(--space-3); flex-wrap:wrap;">
                  <button type="button" class="btn btn-sm" id="btn-parse-seeds">Extract Seeds</button>
                  <span class="mono" style="font-size:12px; color: var(--text-muted);" id="parse-seeds-status"></span>
                </div>
                <div id="parsed-seeds-preview" style="display:flex; flex-wrap:wrap; gap: var(--space-2);"></div>
              </div>

              <div id="launch-panel-image" style="display:${launchMode === "image" ? "flex" : "none"}; flex-direction:column; gap: var(--space-3);">
                <label class="field-label" for="target-image-file">Upload Image (EXIF, GPS &amp; Reverse Search)</label>
                <div id="image-dropzone" class="state-panel" style="padding: var(--space-4); cursor:pointer;">
                  <input type="file" id="target-image-file" accept="image/*" style="max-width:100%;">
                  <div style="margin-top: var(--space-2); font-size:12px;">Extracts EXIF metadata, GPS coordinates, and public web matches.</div>
                </div>
                <div id="image-preview-result"></div>
              </div>
            </div>
          </div>

          <div class="form-section">
            <div class="form-section-title">2. Budget &amp; Scope</div>
            <div class="grid-3">
              <div>
                <label class="field-label" for="param-depth">Max Depth: <strong id="val-depth" style="color:var(--text-primary);">2</strong></label>
                <input type="range" id="param-depth" min="1" max="4" value="2" style="width:100%;">
                <div class="mono" id="hint-depth" style="font-size:11px; color:var(--text-muted); margin-top:4px;">Seed + 2 expansion hops</div>
              </div>
              <div>
                <label class="field-label" for="param-calls">Tool Calls: <strong id="val-calls" style="color:var(--text-primary);">15</strong></label>
                <input type="range" id="param-calls" min="1" max="40" value="15" style="width:100%;">
                <div class="mono" id="hint-calls" style="font-size:11px; color:var(--text-muted); margin-top:4px;">~15 collector queries (~25s)</div>
              </div>
              <div>
                <label class="field-label" for="param-llm">LLM Calls: <strong id="val-llm" style="color:var(--text-primary);">10</strong></label>
                <input type="range" id="param-llm" min="0" max="30" value="10" style="width:100%;">
                <div class="mono" id="hint-llm" style="font-size:11px; color:var(--text-muted); margin-top:4px;">~10 planner / synthesis calls</div>
              </div>
            </div>
          </div>

          <div class="form-section">
            <div class="form-section-title">3. Engines</div>
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
          </div>

          <div class="form-section">
            <div class="form-section-title">4. Authorize</div>
            <div style="padding: var(--space-4); border-radius: var(--radius-md); background: var(--bg-elevated); border: 1px solid var(--border-subtle);">
              <label style="display:flex; align-items:flex-start; gap: var(--space-3); cursor:pointer;">
                <input type="checkbox" id="consent-checkbox" style="margin-top: 2px; flex-shrink:0;">
                <span style="font-size:13px; color:var(--text-secondary); line-height:1.45;">
                  <strong style="color:var(--text-primary);">Mandatory Authorization Gate:</strong> I confirm authorized, public-data-only reconnaissance for this target scope.
                </span>
              </label>
            </div>

            <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap: var(--space-3); margin-top: var(--space-1);">
              <div style="display:flex; align-items:center; gap: var(--space-3); flex-wrap:wrap;">
                <button type="button" class="btn btn-primary" id="btn-launch-scan" disabled>Start Investigation</button>
                <button type="button" class="btn" id="btn-load-demo">Load Recorded Run</button>
                <span class="mono" id="auth-helper-text" style="font-size:11px; color:var(--text-muted);">Confirm authorization to continue</span>
              </div>
              <span class="mono" style="font-size:12px; color: var(--status-error);" id="launch-error"></span>
            </div>
          </div>
        </section>

        <aside class="card" style="padding: var(--space-5); display:flex; flex-direction:column; gap: var(--space-4);">
          <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:var(--space-2);">
            <div>
              <div class="field-label">Saved Snapshots</div>
              <div style="font-size:12px; color:var(--text-muted); line-height:1.45;">Reload saved investigations without re-running collectors.</div>
            </div>
            <button type="button" class="btn btn-sm" id="btn-refresh-saved">Refresh</button>
          </div>
          <div id="saved-runs-list" class="mono" style="font-size:12px; color:var(--text-muted); display:flex; flex-direction:column; gap:8px;">Loading saved snapshots...</div>
        </aside>
      </div>
    `;

    // Swap seed tabs in-place without layout jump or wiping inputs
    container.querySelectorAll("[data-lmode]").forEach((btn) => {
      btn.addEventListener("click", () => {
        launchMode = btn.dataset.lmode;
        container.querySelectorAll("[data-lmode]").forEach((b) => {
          const active = b.dataset.lmode === launchMode;
          b.classList.toggle("active", active);
          b.setAttribute("aria-selected", String(active));
        });
        document.getElementById("launch-panel-structured").style.display = launchMode === "structured" ? "grid" : "none";
        document.getElementById("launch-panel-freetext").style.display = launchMode === "freetext" ? "flex" : "none";
        document.getElementById("launch-panel-image").style.display = launchMode === "image" ? "flex" : "none";
      });
    });

    const bindSlider = (id, outId, hintId, fmtHint) => {
      const el = document.getElementById(id);
      const out = document.getElementById(outId);
      const hint = document.getElementById(hintId);
      if (el && out) {
        el.addEventListener("input", () => {
          out.textContent = el.value;
          if (hint && fmtHint) hint.textContent = fmtHint(Number(el.value));
        });
      }
    };
    bindSlider("param-depth", "val-depth", "hint-depth", (v) => `Seed + ${v} expansion hop${v === 1 ? "" : "s"}`);
    bindSlider("param-calls", "val-calls", "hint-calls", (v) => `~${v} collector queries (~${Math.round(v * 1.6)}s)`);
    bindSlider("param-llm", "val-llm", "hint-llm", (v) => v === 0 ? "Rule-only (0 LLM calls)" : `~${v} planner / synthesis calls`);

    const consentBox = document.getElementById("consent-checkbox");
    const launchBtn = document.getElementById("btn-launch-scan");
    const helperTxt = document.getElementById("auth-helper-text");
    if (consentBox && launchBtn) {
      consentBox.addEventListener("change", () => {
        const ok = consentBox.checked;
        launchBtn.disabled = !ok;
        if (helperTxt) helperTxt.style.display = ok ? "none" : "inline";
      });
    }

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
          parsedSeeds = data.entities || data.seeds || [];
          statusEl.textContent = `Extracted ${parsedSeeds.length} seed(s)`;
          prevEl.innerHTML = parsedSeeds
            .map((s) => `<span class="chip badge-accent">${esc(s.type || s.kind)}: ${esc(s.value)}</span>`)
            .join("");
        } catch (err) {
          statusEl.textContent = `Error: ${err.message}`;
        }
      });
    }

    const imgInput = document.getElementById("target-image-file");
    if (imgInput) {
      imgInput.addEventListener("change", async () => {
        const file = imgInput.files?.[0];
        const prev = document.getElementById("image-preview-result");
        if (!file || !prev) return;
        const consent = Boolean(document.getElementById("consent-checkbox")?.checked);
        if (!consent) {
          prev.innerHTML = `<div class="state-panel state-panel-error">Please check the authorization consent box below before analyzing an image.</div>`;
          imgInput.value = "";
          return;
        }
        prev.innerHTML = `<div class="state-panel">Analyzing ${esc(file.name)}...</div>`;
        const fd = new FormData();
        fd.append("file", file);
        fd.append("consent_confirmed", "true");
        try {
          const r = await fetch("/api/analyze-image", { method: "POST", body: fd });
          const data = await r.json();
          prev.innerHTML = `<pre class="json-pretty" style="max-height:180px;">${esc(JSON.stringify(data, null, 2))}</pre>`;
        } catch (err) {
          prev.innerHTML = `<div class="state-panel state-panel-error">${esc(err.message)}</div>`;
        }
      });
    }

    launchBtn?.addEventListener("click", startInvestigation);
    document.getElementById("btn-load-demo")?.addEventListener("click", () => loadRecordedDemoRun(true));
    document.getElementById("btn-refresh-saved")?.addEventListener("click", refreshSavedRuns);
    refreshSavedRuns();
  }

  async function refreshSavedRuns() {
    const el = document.getElementById("saved-runs-list");
    if (!el) return;
    try {
      const r = await fetch("/api/saved-runs");
      if (!r.ok) {
        el.textContent = "Could not load saved snapshots.";
        return;
      }
      const data = await r.json();
      const runs = (data.runs || []).filter((run) => !pendingDeletes.has(run.id));
      if (!runs.length) {
        el.textContent = "No saved snapshots yet. Run a scan and click \"Save Snapshot\" on the Results tab.";
        return;
      }
      el.innerHTML = runs.map((run) => {
        const absTime = String(run.created_at || "").slice(0, 19).replace("T", " ");
        const relTime = formatRelativeTime(run.created_at);
        return `
          <div data-saved-row="${esc(run.id)}" style="display:flex; align-items:center; justify-content:space-between; gap:10px; padding:10px 12px; border:1px solid var(--border-subtle); border-radius:var(--radius-md); background:var(--bg-elevated);">
            <div style="min-width:0;">
              <div data-tooltip="${esc(run.name)}" style="font-family:var(--font-sans); font-size:13px; font-weight:600; color:var(--text-primary); overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">${esc(run.name)}</div>
              <div style="margin-top:2px; font-size:11px; color:var(--text-muted);">${run.nodes_count} nodes &middot; ${run.edges_count} edges &middot; <span data-tooltip="${esc(absTime || "Unknown timestamp")}">${esc(relTime)}</span></div>
            </div>
            <div style="display:flex; gap:6px; flex-shrink:0;">
              <button type="button" class="btn btn-sm btn-primary" data-load-saved="${esc(run.id)}">Load</button>
              <button type="button" class="btn btn-sm btn-ghost" data-del-saved="${esc(run.id)}" data-del-name="${esc(run.name)}">Delete</button>
            </div>
          </div>
        `;
      }).join("");

      el.querySelectorAll("[data-load-saved]").forEach((btn) => {
        btn.addEventListener("click", async () => {
          const rid = btn.dataset.loadSaved;
          const resp = await fetch(`/api/saved-runs/${encodeURIComponent(rid)}`);
          if (!resp.ok) {
            window.OSINT_UI?.toast("Failed to load snapshot", "error");
            return;
          }
          const snap = await resp.json();
          resetCollectorStatus();
          stopElapsedTimer();
          scanStartMs = 0;
          window.OSINT_STATE.isRecordedRun = true;
          window.OSINT_STATE.jobId = snap.job_id || snap.id;
          window.OSINT_STATE.nodes = snap.nodes || [];
          window.OSINT_STATE.edges = snap.edges || [];
          window.OSINT_STATE.auditEvents = [];

          const tel = snap.telemetry || {};
          const sb = tel.scanBudget || {};
          scanBudget = {
            maxTools: Number.isFinite(Number(sb.maxTools)) ? Number(sb.maxTools) : 15,
            maxLlm: Number.isFinite(Number(sb.maxLlm)) ? Number(sb.maxLlm) : 10,
            toolsUsed: Number.isFinite(Number(sb.toolsUsed)) ? Number(sb.toolsUsed) : 0,
            llmUsed: Number.isFinite(Number(sb.llmUsed)) ? Number(sb.llmUsed) : 0,
          };

          if (tel.collectorStatus) {
            Object.entries(tel.collectorStatus).forEach(([k, v]) => {
              if (collectorStatus[k] && v) {
                collectorStatus[k] = { ...collectorStatus[k], ...v, status: normalizeCollectorStatus(v.status) };
              }
            });
            window.OSINT_STATE.auditEvents = snap.audit_events || [];
          } else {
            (snap.audit_events || []).forEach((ev) => handleEvent(ev, true));
          }

          // Reconcile collector hits/budget from audit events if missing
          const evs = window.OSINT_STATE.auditEvents;
          if (!scanBudget.toolsUsed && evs.length) {
            scanBudget.toolsUsed = evs.filter((e) => e.event === "tool_call_started").length;
            scanBudget.llmUsed = evs.filter((e) => e.event === "planner_decision" && Number((e.details || e.data)?.token_usage || 0) > 0).length;
          }
          finalElapsedSec = computeElapsedFromEvents(evs);

          const nextState = deriveRunStateFromCollectors();
          window.OSINT_UI?.setRunState(nextState === "idle" ? "completed" : nextState);

          document.getElementById("header-job-id").textContent = window.OSINT_STATE.jobId || "--";
          document.getElementById("header-node-count").textContent = String(window.OSINT_STATE.nodes.length);
          document.getElementById("header-edge-count").textContent = String(window.OSINT_STATE.edges.length);
          document.getElementById("nav-badge-results").textContent = String(window.OSINT_STATE.nodes.length);
          document.getElementById("nav-badge-audit").textContent = String(evs.length);
          renderScanView();
          if (typeof window.onGraphUpdated === "function") window.onGraphUpdated();
          if (typeof window.onAuditUpdated === "function") window.onAuditUpdated();
          window.OSINT_UI?.toast(`Loaded "${snap.name || rid}"`, "success");
          window.OSINT_UI?.switchView("results");
        });
      });

      el.querySelectorAll("[data-del-saved]").forEach((btn) => {
        btn.addEventListener("click", () => {
          const rid = btn.dataset.delSaved;
          const name = btn.dataset.delName || rid;
          const row = el.querySelector(`[data-saved-row="${CSS.escape(rid)}"]`);
          if (row) row.style.display = "none";

          const timerId = setTimeout(async () => {
            pendingDeletes.delete(rid);
            await fetch(`/api/saved-runs/${encodeURIComponent(rid)}`, { method: "DELETE" });
            refreshSavedRuns();
          }, 6000);
          pendingDeletes.set(rid, timerId);

          window.OSINT_UI?.toast(`Deleted "${name}"`, "info", {
            duration: 6000,
            actionLabel: "Undo",
            onAction: () => {
              const t = pendingDeletes.get(rid);
              if (t) clearTimeout(t);
              pendingDeletes.delete(rid);
              if (row) row.style.display = "flex";
              window.OSINT_UI?.toast(`Restored "${name}"`, "success");
            },
          });
        });
      });
    } catch (_) {
      el.textContent = "Could not load saved snapshots.";
    }
  }

  window.refreshSavedRuns = refreshSavedRuns;
  window.getScanTelemetry = function () {
    return {
      collectorStatus: JSON.parse(JSON.stringify(collectorStatus)),
      scanBudget: { ...scanBudget },
    };
  };

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
      payload.free_text = (document.getElementById("target-freetext")?.value || "").trim();
      if (parsedSeeds.length) payload.seeds = parsedSeeds;
    } else {
      payload.kind = document.getElementById("target-kind")?.value || "domain";
      payload.value = (document.getElementById("target-value")?.value || "").trim();
    }

    scanBudget = { maxTools: maxCalls, maxLlm: maxLlm, toolsUsed: 0, llmUsed: 0 };
    finalElapsedSec = null;
    unseenHitsCount = 0;
    streamPaused = false;
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
        window.OSINT_UI?.toast("Failed to start investigation", "error");
        return;
      }
      const data = await resp.json();
      window.OSINT_STATE.jobId = data.job_id;
      document.getElementById("header-job-id").textContent = data.job_id;
      window.OSINT_UI?.setRunState("running", "Scanning");
      window.OSINT_UI?.switchView("scan");
      startElapsedTimer();
      connectSSE(data.job_id);
    } catch (err) {
      if (errEl) errEl.textContent = err.message;
    }
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
    if (scanStartMs) {
      finalElapsedSec = Math.max(0.1, (Date.now() - scanStartMs) / 1000);
    }
  }

  function handleEvent(ev, suppressRender = false) {
    const st = window.OSINT_STATE;
    st.auditEvents.push(ev);
    const navAudit = document.getElementById("nav-badge-audit");
    if (navAudit) navAudit.textContent = String(st.auditEvents.length);

    const type = ev.event;
    const d = ev.details || ev.data || {};

    if (type === "tool_call_started" && d.tool) {
      scanBudget.toolsUsed += 1;
      if (collectorStatus[d.tool]) collectorStatus[d.tool].status = "running";
    } else if (type === "tool_call_completed" && d.tool) {
      if (collectorStatus[d.tool]) {
        const hits = Number(d.hits_returned || 0);
        collectorStatus[d.tool].hits += hits;
        collectorStatus[d.tool].status = hits > 0 ? "done" : "not_found";
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
    } else if (type === "hit_evaluated" && d.discovered_value) {
      recentHitTimestamps.push(Date.now());
      recentHitTimestamps = recentHitTimestamps.filter((t) => Date.now() - t < 1000);
      lastAnimatedHitKey = `${d.tool || ""}|${d.discovered_type || ""}|${d.discovered_value || ""}`;

      const streamEl = document.getElementById("sse-hit-stream");
      if (streamEl && streamEl.scrollTop > 20) {
        unseenHitsCount += 1;
      }

      const id = `${d.discovered_type || "node"}:${d.discovered_value}`;
      const meta = d.metadata || {};
      if (!st.nodes.some((n) => n.data?.id === id)) {
        st.nodes.push({
          data: {
            id,
            label: d.discovered_value,
            type: d.discovered_type || "profile",
            confidence: Number(d.child_confidence ?? d.resolver_score ?? 0.5),
            depth: 1,
            provenance: [`${d.tool || "collector"}:discovered`],
            attributes: { ...meta, source_tool: d.tool || "", resolver_score: d.resolver_score },
          },
        });
        const navRes = document.getElementById("nav-badge-results");
        if (navRes) navRes.textContent = String(st.nodes.length);
        const hdrNodes = document.getElementById("header-node-count");
        if (hdrNodes) hdrNodes.textContent = String(st.nodes.length);
        if (typeof window.onGraphUpdated === "function") window.onGraphUpdated();
      }
    } else if (type === "planner_decision" && Number(d.token_usage || 0) > 0) {
      scanBudget.llmUsed += 1;
    } else if (type === "job_completed" || type === "job_failed" || type === "job_refused") {
      stopElapsedTimer();
      window.OSINT_UI?.setRunState(type === "job_completed" ? "completed" : "error", type.replace("job_", ""));
      if (st.jobId) refreshGraph(st.jobId);
    }

    if (!suppressRender) {
      renderScanView();
      if (typeof window.onAuditUpdated === "function") window.onAuditUpdated();
    }
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
    activeEventSource.onerror = async () => {
      if (activeEventSource) {
        activeEventSource.close();
        activeEventSource = null;
      }
      const jid = window.OSINT_STATE.jobId;
      if (jid) {
        try {
          const sr = await fetch(`/api/investigate/${encodeURIComponent(jid)}`);
          if (sr.ok) {
            const st = await sr.json();
            if (["completed", "failed", "refused", "stopped"].includes(st.status)) {
              stopElapsedTimer();
              const mapped = st.status === "completed" ? "completed" : st.status === "stopped" ? "stopped" : "error";
              window.OSINT_UI?.setRunState(mapped);
              await refreshGraph(jid);
              renderScanView();
            }
          }
        } catch (_) {}
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
      stopElapsedTimer();
      scanStartMs = 0;
      scanBudget = { maxTools: 15, maxLlm: 10, toolsUsed: 0, llmUsed: 0 };
      window.OSINT_STATE.isRecordedRun = true;
      window.OSINT_STATE.jobId = run.file.replace("audit_", "").replace(".jsonl", "");
      window.OSINT_STATE.auditEvents = [];
      const nodesMap = {};
      const discoveredMeta = {};
      const edgesList = [];
      document.getElementById("header-job-id").textContent = window.OSINT_STATE.jobId;
      (run.events || []).forEach((ev) => {
        handleEvent(ev, true);
        const evDetails = ev.details || ev.data;
        if (ev.event === "profile_discovered" && evDetails && evDetails.url) {
          const pd = evDetails;
          discoveredMeta[pd.url] = { site: pd.site || "", url: pd.url, username: pd.username || "" };
          const id = `profile:${pd.url}`;
          if (nodesMap[id]) Object.assign(nodesMap[id].data.attributes, discoveredMeta[pd.url]);
        } else if (ev.event === "hit_evaluated" && evDetails) {
          const d = evDetails;
          const id = `${d.discovered_type || "node"}:${d.discovered_value}`;
          const meta = d.metadata || {};
          const disc = discoveredMeta[d.discovered_value] || {};
          if (!nodesMap[id]) {
            nodesMap[id] = {
              data: {
                id,
                label: d.discovered_value,
                type: d.discovered_type || "profile",
                confidence: Number(d.child_confidence ?? d.resolver_score ?? 0.5),
                depth: 1,
                provenance: [`${d.tool || "collector"}:discovered`],
                attributes: { ...disc, ...meta, source_tool: d.tool || "", resolver_score: d.resolver_score },
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
        }
      });
      finalElapsedSec = computeElapsedFromEvents(run.events || []);
      window.OSINT_STATE.nodes = Object.values(nodesMap);
      window.OSINT_STATE.edges = edgesList;
      document.getElementById("header-node-count").textContent = String(window.OSINT_STATE.nodes.length);
      document.getElementById("header-edge-count").textContent = String(window.OSINT_STATE.edges.length);
      document.getElementById("nav-badge-results").textContent = String(window.OSINT_STATE.nodes.length);
      window.OSINT_UI?.setRunState(deriveRunStateFromCollectors());
      renderScanView();
      if (typeof window.onGraphUpdated === "function") window.onGraphUpdated();
      if (typeof window.onAuditUpdated === "function") window.onAuditUpdated();
      if (switchToScan) window.OSINT_UI?.switchView("scan");
    } catch (_) {}
  }
  window.loadRecordedDemoRun = loadRecordedDemoRun;

  function renderScanView() {
    const container = document.getElementById("scan-container");
    if (!container) return;

    const prevStreamEl = document.getElementById("sse-hit-stream");
    const prevScrollTop = prevStreamEl ? prevStreamEl.scrollTop : 0;

    const runState = window.OSINT_STATE.runState || "idle";
    const isRunning = runState === "running";
    const list = Object.values(collectorStatus).map((c) => ({
      ...c,
      normStatus: normalizeCollectorStatus(c.status),
    }));
    list.sort((a, b) => statusSortRank(a.normStatus) - statusSortRank(b.normStatus) || a.name.localeCompare(b.name));

    const total = list.length;
    const checked = list.filter((c) => c.normStatus !== "waiting" && c.normStatus !== "running").length;
    const collectorsWithHits = list.filter((c) => Number(c.hits) > 0 || c.normStatus === "done").length;
    const totalHits = list.reduce((acc, c) => acc + Number(c.hits || 0), 0);
    const liveElapsed = scanStartMs ? Math.max((Date.now() - scanStartMs) / 1000, 0.1) : null;
    const displayElapsed = isRunning ? liveElapsed : finalElapsedSec;
    const rate = isRunning && liveElapsed > 0 && checked > 0 ? (checked / liveElapsed).toFixed(2) : null;

    const scanBadge = document.getElementById("nav-badge-scan");
    if (scanBadge) scanBadge.textContent = `${checked}/${total}`;

    const toolB = safeBudgetPair(scanBudget.toolsUsed, scanBudget.maxTools);
    const llmB = safeBudgetPair(scanBudget.llmUsed, scanBudget.maxLlm);

    // Category aggregation
    const cats = {};
    list.forEach((c) => {
      if (!cats[c.category]) cats[c.category] = { total: 0, done: 0, hits: 0 };
      cats[c.category].total += 1;
      if (c.normStatus !== "waiting" && c.normStatus !== "running") cats[c.category].done += 1;
      if (Number(c.hits) > 0 || c.normStatus === "done") cats[c.category].hits += 1;
    });

    const dotColor = (norm) => {
      if (norm === "done") return "var(--status-done)";
      if (norm === "running") return "var(--status-running)";
      if (norm === "error") return "var(--status-error)";
      if (norm === "unavailable") return "var(--status-unavailable)";
      if (norm === "not_found") return "var(--text-muted)";
      return "var(--border-strong)";
    };

    const allHits = (window.OSINT_STATE.auditEvents || []).filter((e) => e.event === "hit_evaluated");
    const hitTools = ["all", ...Array.from(new Set(allHits.map((e) => (e.details || e.data)?.tool).filter(Boolean)))];
    const filteredHits = allHits
      .filter((e) => streamCollectorFilter === "all" || (e.details || e.data)?.tool === streamCollectorFilter)
      .slice(-30)
      .reverse();

    const shouldAnimateNewest = isRunning && recentHitTimestamps.length <= 5 && (Date.now() - lastHitAnimMs >= 100);
    if (shouldAnimateNewest) lastHitAnimMs = Date.now();

    container.innerHTML = `
      <div class="card" style="padding:14px 18px; display:flex; flex-wrap:wrap; align-items:center; justify-content:space-between; gap:12px;">
        <div style="display:flex; align-items:center; gap:12px; flex-wrap:wrap;">
          <span class="badge ${runState === "running" ? "badge-running" : runState === "completed" ? "badge-done" : runState === "partial" ? "badge-unavailable" : runState === "idle" ? "badge-waiting" : "badge-error"}">
            <span class="status-dot ${esc(runState)}"></span>
            ${esc(runState.toUpperCase())}
          </span>
          <span class="mono" style="font-size:12px; color:var(--text-secondary);">Collectors: <strong style="color:var(--text-primary);">${checked}/${total}</strong></span>
          <span class="mono" style="font-size:12px; color:var(--text-secondary);">Elapsed: <strong id="kpi-elapsed" style="color:var(--text-primary);">${displayElapsed != null ? displayElapsed.toFixed(1) + "s" : "--"}</strong></span>
          <span class="mono" style="font-size:12px; color:var(--text-secondary);">Total Hits: <strong style="color:var(--text-primary);">${totalHits || allHits.length}</strong></span>
        </div>
        ${rate ? `<span class="badge badge-accent">${rate} collectors/s</span>` : `<span class="mono" style="font-size:11px; color:var(--text-muted);">${window.OSINT_STATE.isRecordedRun ? "Recorded Snapshot" : "Ready"}</span>`}
      </div>

      <div class="grid-4">
        <div class="card">
          <div class="kpi-label">Collectors Checked</div>
          <div class="kpi-value">${checked} / ${total}</div>
          <div class="mono" style="font-size:11px; color:var(--text-muted);">${isRunning && rate ? `${rate} answers/s` : `${total - checked} remaining`}</div>
        </div>
        <div class="card">
          <div class="kpi-label">Collectors w/ Hits</div>
          <div class="kpi-value">${collectorsWithHits} <span style="font-size:13px; color:var(--text-muted);">(${totalHits || allHits.length} hits)</span></div>
          <div class="mono" style="font-size:11px; color:var(--text-muted);">Entities: ${(window.OSINT_STATE.nodes || []).length}</div>
        </div>
        <div class="card">
          <div class="kpi-label">Tool Budget</div>
          <div class="kpi-value">${esc(toolB.text)}</div>
          <div class="progress-track" style="margin-top:8px;"><div class="progress-fill" style="--progress:${toolB.ratio};"></div></div>
        </div>
        <div class="card">
          <div class="kpi-label">LLM Budget</div>
          <div class="kpi-value">${esc(llmB.text)}</div>
          <div class="progress-track" style="margin-top:8px;"><div class="progress-fill" style="--progress:${llmB.ratio};"></div></div>
        </div>
      </div>

      <div class="card" style="display:flex; flex-direction:column; gap:var(--space-5);">
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
          <div class="field-label" style="margin:0;">Collector Matrix &amp; Category Progress</div>
          <div style="display:flex; flex-wrap:wrap; gap:10px; font-family:var(--font-mono); font-size:11px; color:var(--text-muted);">
            <span>&#10003; done</span>
            <span>&#9679; running</span>
            <span>&middot; not found</span>
            <span>&#9675; waiting</span>
            <span>&ndash; unavailable</span>
            <span>&#10005; error</span>
          </div>
        </div>

        <div class="grid-2" style="gap:var(--space-5); align-items:start;">
          <div style="display:flex; flex-direction:column; gap:var(--space-3);">
            ${Object.entries(cats).map(([cname, cdata]) => {
              const ratio = cdata.done / Math.max(1, cdata.total);
              return `
                <div>
                  <div style="display:flex; justify-content:space-between; font-family:var(--font-mono); font-size:12px; margin-bottom:4px;">
                    <span>${esc(cname)}</span>
                    <span style="color:var(--text-secondary);">${cdata.done}/${cdata.total} (${cdata.hits} active)</span>
                  </div>
                  <div class="progress-track"><div class="progress-fill" style="--progress:${ratio};"></div></div>
                </div>
              `;
            }).join("")}
          </div>

          <div style="display:flex; flex-wrap:wrap; gap:6px;">
            ${list.map((c) => `
              <span data-tooltip="${esc(c.name)}: ${esc(c.normStatus)} (${c.hits} hits)" style="display:inline-flex; align-items:center; gap:5px; padding:4px 8px; border:1px solid var(--border-subtle); border-radius:4px; background:var(--bg-elevated); font-family:var(--font-mono); font-size:11px;">
                <span style="color:${dotColor(c.normStatus)}; font-weight:700;">${statusCue(c.normStatus)}</span>
                <span>${esc(c.name)}</span>
              </span>
            `).join("")}
          </div>
        </div>
      </div>

      <div class="card">
        <div class="field-label" style="margin-bottom: var(--space-3);">Per-Collector Telemetry (${total} Collectors, sorted by status)</div>
        <div class="grid-3">
          ${list.map((c) => `
            <div style="padding: var(--space-2) var(--space-3); border:1px solid var(--border-subtle); border-radius:var(--radius-sm); background:var(--bg-elevated); display:flex; justify-content:space-between; align-items:center; gap:var(--space-2);">
              <div style="min-width:0;">
                <div class="mono" style="font-size:12px; font-weight:600; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">${esc(c.name)}</div>
                <div class="mono" style="font-size:11px; color:var(--text-muted);">${esc(c.category)}</div>
              </div>
              <span class="badge ${statusBadgeClass(c.normStatus)}">${statusCue(c.normStatus)} ${esc(c.normStatus)}${c.hits > 0 ? " (" + c.hits + ")" : ""}</span>
            </div>
          `).join("")}
        </div>
      </div>

      <div class="card" style="display:flex; flex-direction:column; gap:var(--space-3);">
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
          <div style="display:flex; align-items:center; gap:10px;">
            <span class="field-label" style="margin:0;">Live SSE Hit Stream (${allHits.length})</span>
            ${unseenHitsCount > 0 ? `<button type="button" class="btn btn-xs btn-primary" id="btn-sse-jump-top">${unseenHitsCount} new hits &uarr;</button>` : ""}
          </div>
          <div style="display:flex; align-items:center; gap:8px; flex-wrap:wrap;">
            <select class="select" id="sse-collector-filter" style="width:auto; padding:4px 8px; font-size:11px;">
              ${hitTools.map((t) => `<option value="${esc(t)}" ${streamCollectorFilter === t ? "selected" : ""}>Tool: ${esc(t)}</option>`).join("")}
            </select>
            <button type="button" class="btn btn-sm" id="btn-sse-pause">${streamPaused ? "Resume Stream" : "Pause Stream"}</button>
          </div>
        </div>

        ${streamPaused ? `<div class="state-panel" style="padding:10px;">Stream display paused. New hits continue indexing in background.</div>` : ""}

        ${filteredHits.length && !streamPaused ? `
          <div id="sse-hit-stream" style="display:flex; flex-direction:column; gap:6px; max-height:320px; overflow-y:auto;">
            ${filteredHits.map((e, idx) => {
              const dt = e.details || e.data || {};
              const acc = dt.accepted ? "badge-done" : "badge-error";
              const key = `${dt.tool || ""}|${dt.discovered_type || ""}|${dt.discovered_value || ""}`;
              const animCls = idx === 0 && shouldAnimateNewest && key === lastAnimatedHitKey ? "sse-hit-enter" : "";
              return `
                <div class="sse-hit-row ${animCls}">
                  <span style="min-width:0; word-break:break-word;"><strong>${esc(dt.tool)}</strong> &rarr; ${esc(dt.discovered_type)}: ${esc(dt.discovered_value)}</span>
                  <span class="badge ${acc}">score=${Number(dt.resolver_score || 0).toFixed(2)} | conf=${Number(dt.child_confidence || 0).toFixed(2)}</span>
                </div>
              `;
            }).join("")}
          </div>
        ` : (!streamPaused ? `<div class="state-panel">Waiting for streaming hit events...</div>` : "")}
      </div>
    `;

    const streamEl = document.getElementById("sse-hit-stream");
    if (streamEl) {
      if (prevScrollTop > 20) streamEl.scrollTop = prevScrollTop;
      streamEl.addEventListener("scroll", () => {
        if (streamEl.scrollTop <= 8 && unseenHitsCount > 0) {
          unseenHitsCount = 0;
          document.getElementById("btn-sse-jump-top")?.remove();
        }
      });
      const enterEl = streamEl.querySelector(".sse-hit-enter");
      if (enterEl) {
        requestAnimationFrame(() => enterEl.classList.add("is-shown"));
      }
    }

    document.getElementById("btn-sse-jump-top")?.addEventListener("click", () => {
      unseenHitsCount = 0;
      const el = document.getElementById("sse-hit-stream");
      if (el) el.scrollTop = 0;
      renderScanView();
    });

    document.getElementById("btn-sse-pause")?.addEventListener("click", () => {
      streamPaused = !streamPaused;
      renderScanView();
    });

    document.getElementById("sse-collector-filter")?.addEventListener("change", (e) => {
      streamCollectorFilter = e.target.value;
      renderScanView();
    });
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
          window.OSINT_UI?.setRunState("stopped", "Stopped");
          stopElapsedTimer();
          renderScanView();
          window.OSINT_UI?.toast("Scan stopped", "info");
        } catch (_) {}
      });
    }
  });
})();
