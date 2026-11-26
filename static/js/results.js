/* M3: Result Cards (all entity variants, confidence ring, why-matched chips, field coverage, sparse states) */

(function () {
  "use strict";

  let typeFilter = "all";
  let minConfFilter = 0;

  function esc(str) {
    return String(str ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function confColor(conf) {
    if (conf >= 0.75) return "var(--confirmed)";
    if (conf >= 0.5) return "var(--accent)";
    if (conf >= 0.3) return "var(--uncertain)";
    return "var(--rejected)";
  }

  function computeCoverageClient(nodeData) {
    if (nodeData.field_coverage) return nodeData.field_coverage;
    const attrs = nodeData.attributes || {};
    const expected = nodeData.type === "profile"
      ? ["username", "name", "bio", "location", "joined", "followers", "links", "avatar", "banner", "orgs", "activity_dates"]
      : nodeData.type === "domain"
      ? ["domain", "registrar", "created", "nameservers", "ip"]
      : nodeData.type === "email"
      ? ["email", "username", "name", "links"]
      : ["name", "url", "description", "location"];
    const present = expected.filter((f) => {
      const v = attrs[f] ?? attrs[f + "_url"] ?? (f === "links" ? attrs.linked_urls || attrs.website : null);
      return v !== undefined && v !== null && v !== "" && (!Array.isArray(v) || v.length > 0);
    });
    const missing = expected.filter((f) => !present.includes(f));
    return {
      extracted_count: present.length,
      expected_count: expected.length,
      ratio: expected.length ? present.length / expected.length : 0,
      present_fields: present,
      missing_fields: missing,
      is_partial: present.length < expected.length,
      is_sparse: present.length <= 2,
    };
  }

  function buildWhyMatchedChips(d) {
    const attrs = d.attributes || {};
    const chips = [];
    const feats = attrs.resolver_features || attrs.hand_features || {};
    Object.entries(feats).forEach(([k, v]) => {
      if (Number(v) > 0.25) chips.push(`${k}: ${Number(v).toFixed(2)}`);
    });
    if (!chips.length && Array.isArray(d.provenance)) {
      d.provenance.slice(-2).forEach((p) => chips.push(p));
    }
    if (d.gnn_confidence != null) {
      chips.push(`GNN: ${Number(d.gnn_confidence).toFixed(2)}`);
    }
    return chips.slice(0, 4);
  }

  function renderEntityFields(d) {
    const attrs = d.attributes || {};
    const rows = [];
    const add = (label, val) => {
      if (val !== undefined && val !== null && val !== "" && (!Array.isArray(val) || val.length > 0)) {
        const text = Array.isArray(val) ? val.slice(0, 3).join(", ") : String(val);
        rows.push(`<div class="mono" style="font-size:11px; display:flex; justify-content:space-between; gap:8px;"><span style="color:var(--text-muted);">${esc(label)}</span><span style="text-align:right; overflow:hidden; text-overflow:ellipsis; max-width:68%;">${esc(text)}</span></div>`);
      }
    };

    if (d.type === "profile") {
      add("name", attrs.name || attrs.display_name);
      add("location", attrs.location);
      add("joined", attrs.joined || attrs.created_at);
      add("active", attrs.activity_dates || attrs.updated_at);
      add("followers", attrs.followers);
      add("languages", attrs.languages);
      add("orgs", attrs.orgs);
    } else if (d.type === "domain") {
      add("registrar", attrs.registrar);
      add("ip", attrs.ip);
      add("created", attrs.created);
      add("subdomains", attrs.subdomains);
    } else if (d.type === "email") {
      add("username", attrs.username);
      add("name", attrs.name);
      add("pgp", attrs.pgp_key ? "present" : "");
    } else if (d.type === "place") {
      add("lat/lon", attrs.lat != null && attrs.lon != null ? `${attrs.lat}, ${attrs.lon}` : "");
      add("country", attrs.country);
    } else {
      add("site", attrs.site);
      add("summary", attrs.bio || attrs.description || attrs.summary);
      add("date", attrs.date || attrs.published_date);
    }

    // Sensitive fields gated behind explicit reveal
    const hasSensitive = Boolean(attrs.phone || attrs.phones || attrs.breach || attrs.stealer_family);
    let sensitiveHtml = "";
    if (hasSensitive) {
      sensitiveHtml = `
        <div style="margin-top:6px;">
          <button type="button" class="btn btn-sm btn-danger" onclick="event.stopPropagation(); this.nextElementSibling.style.display='block'; this.style.display='none';">
            Sensitive (Authorized Use Only) - Reveal
          </button>
          <div class="mono" style="display:none; font-size:11px; color:var(--rejected); margin-top:4px;">
            ${esc(JSON.stringify({ phone: attrs.phone || attrs.phones, breach: attrs.breach || attrs.stealer_family }))}
          </div>
        </div>
      `;
    }

    if (!rows.length && !sensitiveHtml) {
      return `<div class="state-panel" style="padding: var(--space-2); font-size:11px;">Found, little public data exposed by source.</div>`;
    }
    return `<div style="display:flex; flex-direction:column; gap:4px;">${rows.join("")}${sensitiveHtml}</div>`;
  }

  function renderResultCard(node, idx) {
    const d = node.data || {};
    const attrs = d.attributes || {};
    const conf = Number(d.confidence ?? 0.5);
    const pct = Math.round(conf * 100);
    const ringCol = confColor(conf);
    const cov = computeCoverageClient(d);
    const chips = buildWhyMatchedChips(d);
    const sourceTool = attrs.source_tool || (Array.isArray(d.provenance) && d.provenance.length ? d.provenance[d.provenance.length - 1].split(":")[0] : "seed");
    const linkOut = attrs.url || attrs.website || (String(d.label || "").startsWith("http") ? d.label : "");
    const selected = window.OSINT_STATE.selectedNodeId === d.id ? "is-selected" : "";
    const rawLabel = String(d.label || "");
    const isSensitiveRecord = /^(breach|breach-analytics|infostealer):/i.test(rawLabel) || Boolean(attrs.phone || attrs.phones || attrs.breach || attrs.stealer_family);
    const displayLabel = isSensitiveRecord ? `[Sensitive Finding: ${rawLabel.split(":")[0]}]` : rawLabel;

    const bannerHtml = attrs.banner || attrs.banner_url
      ? `<div style="height:44px; border-radius:4px; background:url('${esc(attrs.banner || attrs.banner_url)}') center/cover no-repeat; margin-bottom:8px; border:1px solid var(--border-subtle);"></div>`
      : "";
    const avatarHtml = attrs.avatar || attrs.avatar_url
      ? `<img src="${esc(attrs.avatar || attrs.avatar_url)}" alt="" width="32" height="32" style="border-radius:50%; object-fit:cover; border:1px solid var(--border-strong);">`
      : "";

    return `
      <article
        class="card card-interactive result-card ${selected}"
        tabindex="0"
        data-node-id="${esc(d.id)}"
        data-card-index="${idx}"
        aria-label="${esc(d.type)}: ${esc(d.label)}"
      >
        ${bannerHtml}
        <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:var(--space-3); margin-bottom:var(--space-2);">
          <div style="display:flex; align-items:center; gap:var(--space-2); min-width:0;">
            ${avatarHtml}
            <div style="min-width:0;">
              <div style="display:flex; align-items:center; gap:6px; flex-wrap:wrap;">
                <span class="badge badge-accent">${esc(d.type)}</span>
                <span class="badge">${esc(attrs.site || sourceTool)}</span>
                <span class="badge">d=${esc(d.depth ?? 0)}</span>
              </div>
              <h3 style="margin:4px 0 0 0; font-size:14px; font-weight:700; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">${esc(displayLabel)}</h3>
            </div>
          </div>

          <div
            title="Calibrated confidence: ${pct}%"
            style="width:42px; height:42px; flex-shrink:0; border-radius:50%; background:conic-gradient(${ringCol} ${pct * 3.6}deg, var(--bg-elevated) 0deg); display:flex; align-items:center; justify-content:center;"
          >
            <span class="mono" style="width:32px; height:32px; border-radius:50%; background:var(--bg-surface); display:flex; align-items:center; justify-content:center; font-size:11px; font-weight:700;">${pct}</span>
          </div>
        </div>

        ${attrs.bio || attrs.description ? `<p style="margin:0 0 8px 0; font-size:12px; color:var(--text-secondary); display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden;">${esc(attrs.bio || attrs.description)}</p>` : ""}

        <div style="display:flex; flex-wrap:wrap; gap:4px; margin-bottom:8px;">
          ${chips.map((c) => `<span class="chip">${esc(c)}</span>`).join("")}
        </div>

        ${renderEntityFields(d)}

        <div style="margin-top:10px; padding-top:8px; border-top:1px solid var(--border-subtle); display:flex; flex-direction:column; gap:6px;">
          <div style="display:flex; justify-content:space-between; align-items:center; gap:8px; font-family:var(--font-mono); font-size:11px;">
            <span>${cov.extracted_count} of ${cov.expected_count} fields</span>
            ${cov.is_partial ? `<span class="badge badge-uncertain" title="Missing: ${esc((cov.missing_fields || []).join(', '))}">Partial data</span>` : `<span class="badge badge-confirmed">Complete</span>`}
          </div>
          <div class="progress-track"><div class="progress-fill" style="width:${Math.round(cov.ratio * 100)}%;"></div></div>
          ${cov.is_partial && cov.missing_fields?.length ? `<div class="mono" style="font-size:10px; color:var(--text-muted); overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">Missing: ${esc(cov.missing_fields.join(", "))}</div>` : ""}
          <div style="display:flex; justify-content:space-between; align-items:center; gap:8px; margin-top:2px;">
            ${linkOut ? `<a href="${esc(linkOut)}" target="_blank" rel="noopener noreferrer" class="btn btn-sm" onclick="event.stopPropagation();">Open Link</a>` : `<span></span>`}
            <button type="button" class="btn btn-sm btn-primary" data-inspect-id="${esc(d.id)}">Inspect -&gt;</button>
          </div>
        </div>
      </article>
    `;
  }

  function renderResultsView() {
    const container = document.getElementById("results-container");
    if (!container) return;

    const allNodes = window.OSINT_STATE.nodes || [];
    const q = window.OSINT_STATE.searchQuery || "";

    const types = ["all", ...Array.from(new Set(allNodes.map((n) => n.data?.type).filter(Boolean)))];
    const filtered = allNodes.filter((n) => {
      const d = n.data || {};
      if (typeFilter !== "all" && d.type !== typeFilter) return false;
      if (Number(d.confidence ?? 0) < minConfFilter) return false;
      if (q) {
        const hay = `${d.label || ""} ${d.type || ""} ${JSON.stringify(d.attributes || {})}`.toLowerCase();
        if (!hay.includes(q)) return false;
      }
      return true;
    });

    const visible = filtered.slice(0, 200);

    container.innerHTML = `
      <div class="card" style="display:flex; flex-wrap:wrap; align-items:center; justify-content:space-between; gap:var(--space-3);">
        <div style="display:flex; flex-wrap:wrap; gap:6px;">
          ${types.map((t) => `<button type="button" class="tab-btn ${typeFilter === t ? "active" : ""}" data-rtype="${esc(t)}">${esc(t)}</button>`).join("")}
        </div>
        <div style="display:flex; align-items:center; gap:8px;">
          <label class="mono" style="font-size:11px; color:var(--text-muted);" for="filter-min-conf">Min Conf: <strong id="lbl-min-conf">${Math.round(minConfFilter * 100)}%</strong></label>
          <input type="range" id="filter-min-conf" min="0" max="0.9" step="0.1" value="${minConfFilter}">
          <span class="badge">${filtered.length} shown</span>
        </div>
      </div>

      ${visible.length ? `
        <div class="grid-3" id="results-cards-grid">
          ${visible.map((n, i) => renderResultCard(n, i)).join("")}
        </div>
        ${filtered.length > 200 ? `<div class="state-panel">Showing top 200 of ${filtered.length} cards (use search or filters to narrow).</div>` : ""}
      ` : `<div class="state-panel">No entities match the current filter. Run a scan or click "Load Recorded Run" on Launch.</div>`}
    `;

    container.querySelectorAll("[data-rtype]").forEach((btn) => {
      btn.addEventListener("click", () => {
        typeFilter = btn.dataset.rtype;
        renderResultsView();
      });
    });

    const confSlider = document.getElementById("filter-min-conf");
    if (confSlider) {
      confSlider.addEventListener("input", () => {
        minConfFilter = Number(confSlider.value);
        renderResultsView();
      });
    }

    container.querySelectorAll(".result-card").forEach((card) => {
      card.addEventListener("click", () => {
        window.OSINT_UI.openInspector(card.dataset.nodeId);
      });
      card.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          window.OSINT_UI.openInspector(card.dataset.nodeId);
        } else if (e.key === "ArrowRight" || e.key === "ArrowDown") {
          e.preventDefault();
          const next = card.nextElementSibling;
          if (next) next.focus();
        } else if (e.key === "ArrowLeft" || e.key === "ArrowUp") {
          e.preventDefault();
          const prev = card.previousElementSibling;
          if (prev) prev.focus();
        }
      });
    });
  }

  window.renderResultsView = renderResultsView;
  window.onGraphUpdated = function () {
    renderResultsView();
    if (typeof window.onGraphUpdatedHook === "function") window.onGraphUpdatedHook();
  };
  window.onSearchChanged = function () {
    renderResultsView();
  };
  document.addEventListener("DOMContentLoaded", renderResultsView);
})();
