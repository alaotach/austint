/* M3: Result Cards (all entity variants, confidence ring, why-matched chips, field coverage, sparse states) */

(function () {
  "use strict";

  let typeFilter = "all";
  let minConfFilter = 0;
  let visibleLimit = 200;

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
    const attrs = { ...(nodeData.attributes || {}) };
    if (String(nodeData.label || "").startsWith("http") && !attrs.url) {
      attrs.url = nodeData.label;
    }
    if (nodeData.type === "username" && nodeData.label && !attrs.username) {
      attrs.username = nodeData.label;
    }
    const srcTool = attrs.source_tool || "";
    const collectorMeta = (window.OSINT_COLLECTORS || []).find((c) => c.name === srcTool);
    const expected = collectorMeta?.expected_fields?.length
      ? collectorMeta.expected_fields
      : nodeData.type === "profile"
      ? ["username", "name", "bio", "location", "joined", "followers", "links", "avatar", "banner", "orgs", "activity_dates"]
      : nodeData.type === "domain"
      ? ["domain", "registrar", "created", "nameservers", "ip"]
      : nodeData.type === "email"
      ? ["email", "username", "name", "links"]
      : ["name", "url", "description", "location"];
    const aliasMap = {
      name: [attrs.name, attrs.display_name, attrs.title],
      links: [attrs.links, attrs.linked_urls, attrs.website, attrs.url],
      avatar: [attrs.avatar, attrs.avatar_url],
      banner: [attrs.banner, attrs.banner_url],
      joined: [attrs.joined, attrs.created_at, attrs.created],
      activity_dates: [attrs.activity_dates, attrs.updated_at, attrs.last_active, attrs.published_date],
    };
    const present = expected.filter((f) => {
      const cands = aliasMap[f] || [attrs[f]];
      return cands.some((v) => v !== undefined && v !== null && v !== "" && (!Array.isArray(v) || v.length > 0));
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
    const seenKeys = new Set();
    const skipKeys = new Set([
      "url", "website", "title", "site", "bio", "description",
      "avatar", "avatar_url", "banner", "banner_url", "images",
      "source_tool", "resolver_features", "hand_features",
      "field_sources", "field_confidence", "ml_keywords", "ml_summary",
      "limited_public_data", "status_note", "extra_fields",
      "phone", "phones", "breach", "stealer_family",
    ]);

    const formatVal = (val) => {
      if (val === undefined || val === null || val === "") return "";
      if (Array.isArray(val)) {
        if (!val.length) return "";
        return val.map((v) => (typeof v === "object" && v !== null ? JSON.stringify(v) : String(v))).join(", ");
      }
      if (typeof val === "object") {
        const entries = Object.entries(val).filter(([, v]) => v !== undefined && v !== null && v !== "");
        if (!entries.length) return "";
        return entries.map(([k, v]) => `${k}: ${Array.isArray(v) ? v.join(", ") : v}`).join(" · ");
      }
      if (typeof val === "boolean") return val ? "yes" : "no";
      return String(val);
    };

    const add = (rawKey, val) => {
      const normKey = String(rawKey).toLowerCase();
      if (skipKeys.has(normKey) || seenKeys.has(normKey)) return;
      const text = formatVal(val).trim();
      if (!text) return;
      seenKeys.add(normKey);
      const label = normKey.replace(/_/g, " ");
      if (text.length > 260) {
        rows.push(`
          <details class="mono" style="font-size:11px; margin-top:2px;" onclick="event.stopPropagation();">
            <summary style="cursor:pointer; color:var(--text-muted);">${esc(label)} <span style="color:var(--accent);">(${text.length} chars)</span></summary>
            <div style="margin-top:4px; padding:6px; border-radius:4px; background:var(--bg-elevated); white-space:pre-wrap; word-break:break-word; max-height:180px; overflow:auto;">${esc(text)}</div>
          </details>
        `);
      } else {
        rows.push(`<div class="mono" style="font-size:11px; display:flex; justify-content:space-between; gap:8px; align-items:flex-start;"><span style="color:var(--text-muted); flex-shrink:0;">${esc(label)}</span><span style="text-align:right; word-break:break-word; max-width:72%;">${esc(text)}</span></div>`);
      }
    };

    // Priority order for common high-signal fields, then dynamically render EVERYTHING else
    const priorityOrder = [
      "name", "display_name", "username", "job_title", "headline",
      "organizations", "orgs", "company", "location", "country",
      "dob", "email", "contact_emails", "current_research_area",
      "education", "experience", "guide_name", "skills",
      "projects_or_stories", "languages", "followers", "post_count",
      "total_views", "joined", "created_at", "activity_dates", "updated_at",
      "registrar", "ip", "created", "subdomains", "llm_summary",
    ];
    const ef = (attrs.extra_fields && typeof attrs.extra_fields === "object") ? attrs.extra_fields : {};
    const merged = { ...attrs, ...ef };
    priorityOrder.forEach((k) => {
      if (k in merged) add(k, merged[k]);
    });
    Object.entries(merged).forEach(([k, v]) => add(k, v));

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
      const note = attrs.status_note || (attrs.limited_public_data ? "Exists, limited public data (login-walled)." : "Found, little public data exposed by source.");
      return `<div class="state-panel" style="padding: var(--space-2); font-size:11px;">${esc(note)}</div>`;
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
    const cleanUrlTitle = (urlStr) => {
      let host = "";
      try {
        host = new URL(urlStr).hostname.replace(/^www\./i, "");
      } catch {}
      const siteName = (typeof attrs.site === "string" && attrs.site.trim()) ? attrs.site.trim() : host;
      const primary = [attrs.title, attrs.name, attrs.display_name, attrs.username].find(
        (v) => typeof v === "string" && v.trim() && !/^https?:\/\//i.test(v.trim())
      )?.trim() || "";
      if (siteName && primary) {
        if (primary.toLowerCase().includes(siteName.toLowerCase())) return primary;
        return `${siteName} · ${primary}`;
      }
      return primary || siteName || urlStr;
    };
    const displayLabel = isSensitiveRecord
      ? `[Sensitive Finding: ${rawLabel.split(":")[0]}]`
      : (/^https?:\/\//i.test(rawLabel.trim()) ? cleanUrlTitle(rawLabel.trim()) : rawLabel);

    const bannerHtml = attrs.banner || attrs.banner_url
      ? `<div style="height:44px; border-radius:4px; background:url('${esc(attrs.banner || attrs.banner_url)}') center/cover no-repeat; margin-bottom:8px; border:1px solid var(--border-subtle);"></div>`
      : "";
    const avatarHtml = attrs.avatar || attrs.avatar_url
      ? `<img src="${esc(attrs.avatar || attrs.avatar_url)}" alt="" width="32" height="32" onerror="this.style.display='none'" style="border-radius:50%; object-fit:cover; border:1px solid var(--border-strong);">`
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
              <h3 title="${esc(rawLabel)}" style="margin:4px 0 0 0; font-size:14px; font-weight:700; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">${esc(displayLabel)}</h3>
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

  function buildDossierHtml(allNodes) {
    const names = new Set();
    const emails = new Set();
    const phones = new Set();
    const locations = new Set();
    const orgs = new Set();
    const websites = new Set();
    const roles = new Set();
    const dobs = new Set();
    const education = new Set();
    const skills = new Set();
    const projects = new Set();
    const summaries = new Set();
    const photos = new Set();

    const photoItems = [];
    const seenPhotoUrls = new Set();
    const pushPhoto = (u, srcLabel, linkUrl) => {
      if (typeof u === "string" && u.startsWith("http") && !seenPhotoUrls.has(u)) {
        seenPhotoUrls.add(u);
        photos.add(u);
        photoItems.push({ url: u, source: srcLabel || "", pageUrl: linkUrl || u });
      }
    };
    const addStr = (set, v, maxLen = 260) => {
      const s = String(v || "").trim();
      if (s && s.length < maxLen) set.add(s);
    };
    const isLikelyDomain = (v) => {
      const s = String(v || "").trim().toLowerCase();
      if (!s || /^(s\.no|m\.tech|b\.tech|b\.e|r\.k\..*)$/.test(s)) return false;
      if (/^[a-z]\.[a-z]\./.test(s)) return false;
      return true;
    };

    allNodes.forEach((n) => {
      const d = n.data || {};
      const a = d.attributes || {};
      const ef = a.extra_fields || {};
      const srcLabel = a.site || a.source_tool || d.type || "";
      const pageUrl = a.url || (String(d.label || "").startsWith("http") ? d.label : "");
      if (d.type === "email") addStr(emails, d.label);
      if (d.type === "org") addStr(orgs, d.label);
      if (d.type === "domain" && isLikelyDomain(d.label)) addStr(websites, d.label);
      [a.name, a.display_name, ef.legal_name].forEach((x) => {
        if (x && String(x).trim().toUpperCase() !== "WATCH") addStr(names, x);
      });
      [a.email, a.developer_email, a.support_email, ef.developer_email, ef.support_email].forEach((x) => addStr(emails, x));
      (a.contact_emails || []).forEach((x) => addStr(emails, x));
      (a.phone_numbers || []).forEach((x) => addStr(phones, x));
      [a.location, a.office_address, ef.office_address, a.country].forEach((x) => addStr(locations, x));
      [a.company, a.developer_name, ef.developer_name].forEach((x) => addStr(orgs, x));
      (a.orgs || []).forEach((x) => addStr(orgs, x));
      (a.organizations || []).forEach((x) => addStr(orgs, x));
      [a.job_title, a.headline, ef.headline].forEach((x) => addStr(roles, x));
      [a.dob, ef.dob].forEach((x) => addStr(dobs, x));
      [a.education, ef.education, a.guide_name ? `Guide: ${a.guide_name}` : "", ef.guide_name ? `Guide: ${ef.guide_name}` : ""].forEach((x) => addStr(education, x, 350));
      (a.skills || []).forEach((x) => addStr(skills, x));
      [a.current_research_area, ef.current_research_area].forEach((x) => addStr(skills, x, 300));
      (a.projects_or_stories || []).forEach((x) => addStr(projects, x, 300));
      if (a.llm_summary) addStr(summaries, a.llm_summary, 400);
      if (typeof a.website === "string" && a.website.startsWith("http")) addStr(websites, a.website);
      [a.avatar_url, a.avatar].forEach((u) => pushPhoto(u, srcLabel, pageUrl));
      (a.images || []).forEach((u) => pushPhoto(u, srcLabel, pageUrl));
    });

    window.__OSINT_DOSSIER_PHOTOS = photoItems;

    const photoList = Array.from(photos);
    const primaryPhoto = photoList[0] || "";
    const row = (label, set, isLink) => {
      const arr = Array.from(set).slice(0, 6);
      if (!arr.length) return "";
      const vals = arr.map((v) => isLink
        ? `<a href="${esc(v.startsWith("http") ? v : "https://" + v)}" target="_blank" rel="noopener noreferrer" style="color:var(--accent);">${esc(v)}</a>`
        : esc(v)
      ).join("<br>");
      return `
        <div style="padding:8px 0; border-bottom:1px solid var(--border-subtle);">
          <div class="field-label" style="font-size:10px; margin-bottom:2px;">${esc(label)}</div>
          <div class="mono" style="font-size:12px; word-break:break-word;">${vals}</div>
        </div>
      `;
    };

    const hasAny = names.size || emails.size || phones.size || locations.size || orgs.size || websites.size || photoList.length;
    if (!hasAny) return "";

    return `
      <section class="card" style="padding:var(--space-4);">
        <div style="display:flex; flex-wrap:wrap; gap:var(--space-4); align-items:flex-start;">
          <div style="display:flex; align-items:center; gap:12px; min-width:220px;">
            ${primaryPhoto ? `<a href="${esc(primaryPhoto)}" target="_blank" rel="noopener noreferrer"><img src="${esc(primaryPhoto)}" alt="" onerror="this.style.display='none'" style="width:56px; height:56px; border-radius:50%; object-fit:cover; border:1px solid var(--border-strong);"></a>` : ""}
            <div>
              <div class="field-label">Subject Identity Dossier</div>
              <div style="font-weight:700; font-size:16px;">${esc(Array.from(names)[0] || "Aggregated Target Intel")}</div>
              ${photoList.length ? `
                <div style="display:flex; flex-wrap:wrap; align-items:center; gap:4px; margin-top:6px;">
                  ${photoList.slice(0, 6).map((u) => `
                    <a href="${esc(u)}" class="dossier-photo-trigger" title="Open Gallery">
                      <img src="${esc(u)}" alt="" loading="lazy" onerror="this.parentElement.style.display='none'" style="width:30px; height:30px; border-radius:4px; object-fit:cover; border:1px solid var(--border-subtle);">
                    </a>
                  `).join("")}
                  <button type="button" class="btn btn-sm" id="btn-open-photo-gallery" style="font-size:11px; padding:3px 8px;">Gallery (${photoList.length})</button>
                </div>
              ` : ""}
            </div>
          </div>
          <div style="flex:1; min-width:280px; display:grid; grid-template-columns:repeat(auto-fit, minmax(190px, 1fr)); gap:0 var(--space-4);">
            ${row("Names & Aliases", names, false)}
            ${row("Roles & Titles", roles, false)}
            ${row("Emails", emails, false)}
            ${row("Phone Numbers", phones, false)}
            ${row("Date of Birth", dobs, false)}
            ${row("Locations & Addresses", locations, false)}
            ${row("Organizations", orgs, false)}
            ${row("Education & Guide", education, false)}
            ${row("Skills & Research", skills, false)}
            ${row("Projects & Publications", projects, false)}
            ${row("Domains & Websites", websites, true)}
            ${row("Profile Summary", summaries, false)}
          </div>
        </div>
      </section>
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

    filtered.sort((a, b) => {
      const da = a.data || {};
      const db = b.data || {};
      const fa = computeCoverageClient(da).extracted_count || 0;
      const fb = computeCoverageClient(db).extracted_count || 0;
      if (fb !== fa) return fb - fa;
      const ca = Number(da.confidence ?? 0);
      const cb = Number(db.confidence ?? 0);
      if (Math.abs(cb - ca) > 1e-6) return cb - ca;
      const pa = da.type === "profile" ? 1 : 0;
      const pb = db.type === "profile" ? 1 : 0;
      return pb - pa;
    });

    const normEntityKey = (d) => {
      const rawUrl = d.attributes?.url || (/^https?:\/\//i.test(String(d.label || "").trim()) ? d.label : "");
      if (rawUrl) {
        return "url:" + String(rawUrl).trim().toLowerCase().replace(/^https?:\/\/(www\.)?/, "").replace(/\/+$/, "");
      }
      return `${d.type || ""}:${String(d.label || "").trim().toLowerCase()}`;
    };
    const seenKeys = new Set();
    const deduped = [];
    for (const n of filtered) {
      const key = normEntityKey(n.data || {});
      if (seenKeys.has(key)) continue;
      seenKeys.add(key);
      deduped.push(n);
    }
    filtered.splice(0, filtered.length, ...deduped);

    const visible = filtered.slice(0, visibleLimit);
    const dossierHtml = buildDossierHtml(allNodes);

    container.innerHTML = `
      <div class="card" style="display:flex; flex-wrap:wrap; align-items:center; justify-content:space-between; gap:var(--space-3);">
        <div style="display:flex; flex-wrap:wrap; gap:6px;">
          ${types.map((t) => `<button type="button" class="tab-btn ${typeFilter === t ? "active" : ""}" data-rtype="${esc(t)}">${esc(t)}</button>`).join("")}
        </div>
        <div style="display:flex; align-items:center; gap:8px; flex-wrap:wrap;">
          <label class="mono" style="font-size:11px; color:var(--text-muted);" for="filter-min-conf">Min Conf: <strong id="lbl-min-conf">${Math.round(minConfFilter * 100)}%</strong></label>
          <input type="range" id="filter-min-conf" min="0" max="0.9" step="0.1" value="${minConfFilter}">
          <span class="badge">${visible.length} / ${filtered.length}</span>
          ${allNodes.length ? `<button type="button" class="btn btn-sm btn-primary" id="btn-save-snapshot">Save Snapshot</button>` : ""}
        </div>
      </div>

      ${dossierHtml}

      <div>
        ${visible.length ? `
          <div id="results-cards-grid" style="display:grid; grid-template-columns:repeat(auto-fill, minmax(280px, 1fr)); gap:var(--space-3);">
            ${visible.map((n, i) => renderResultCard(n, i)).join("")}
          </div>
          ${filtered.length > visibleLimit ? `
            <div class="state-panel" style="display:flex; align-items:center; justify-content:center; gap:12px; flex-wrap:wrap;">
              <span>Windowed DOM (${visible.length} of ${filtered.length} cards rendered for 60fps performance)</span>
              <button type="button" class="btn btn-sm btn-primary" id="btn-load-more-cards">Load +100 Cards</button>
            </div>
          ` : ""}
        ` : `<div class="state-panel">No entities match the current filter. Run a scan or click "Load Recorded Run" on Launch.</div>`}
      </div>
    `;

    document.getElementById("btn-save-snapshot")?.addEventListener("click", async () => {
      const btn = document.getElementById("btn-save-snapshot");
      const defaultName = window.OSINT_STATE.jobId || (allNodes[0]?.data?.label) || "osint_snapshot";
      const name = window.prompt("Save snapshot as:", defaultName);
      if (!name || !name.trim()) return;
      if (btn) btn.disabled = true;
      try {
        const resp = await fetch("/api/saved-runs", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            name: name.trim(),
            job_id: window.OSINT_STATE.jobId || null,
            nodes: window.OSINT_STATE.nodes || [],
            edges: window.OSINT_STATE.edges || [],
            audit_events: window.OSINT_STATE.auditEvents || [],
            telemetry: typeof window.getScanTelemetry === "function" ? window.getScanTelemetry() : {},
          }),
        });
        if (resp.ok && btn) {
          btn.textContent = "Saved!";
          if (typeof window.refreshSavedRuns === "function") window.refreshSavedRuns();
          setTimeout(() => {
            btn.textContent = "Save Snapshot";
            btn.disabled = false;
          }, 1500);
        } else if (btn) {
          btn.disabled = false;
        }
      } catch (_) {
        if (btn) btn.disabled = false;
      }
    });

    document.getElementById("btn-open-photo-gallery")?.addEventListener("click", openPhotoGalleryModal);
    container.querySelectorAll(".dossier-photo-trigger").forEach((el) => {
      el.addEventListener("click", (e) => {
        e.preventDefault();
        openPhotoGalleryModal();
      });
    });

    document.getElementById("btn-load-more-cards")?.addEventListener("click", () => {
      visibleLimit += 100;
      renderResultsView();
    });

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

  function openPhotoGalleryModal() {
    const items = window.__OSINT_DOSSIER_PHOTOS || [];
    let modal = document.getElementById("osint-photo-gallery-modal");
    if (modal) modal.remove();
    modal = document.createElement("div");
    modal.id = "osint-photo-gallery-modal";
    modal.style.cssText = "position:fixed; inset:0; z-index:9999; background:rgba(8,12,20,0.88); backdrop-filter:blur(4px); display:flex; align-items:center; justify-content:center; padding:24px;";
    modal.innerHTML = `
      <div class="card" style="max-width:1040px; width:100%; max-height:88vh; display:flex; flex-direction:column; gap:12px; padding:20px;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <div>
            <div class="field-label">Collected Photos &amp; Media</div>
            <div style="font-weight:700; font-size:16px;">${items.length} Images</div>
          </div>
          <button type="button" class="btn btn-sm" id="btn-close-photo-gallery">Close (Esc)</button>
        </div>
        <div style="overflow-y:auto; display:grid; grid-template-columns:repeat(auto-fill, minmax(150px, 1fr)); gap:12px; padding-right:4px;">
          ${items.map((it) => `
            <div class="card" style="padding:8px; display:flex; flex-direction:column; gap:6px;">
              <a href="${esc(it.url)}" target="_blank" rel="noopener noreferrer" style="display:block; aspect-ratio:1/1; overflow:hidden; border-radius:4px; background:var(--bg-elevated);">
                <img src="${esc(it.url)}" alt="" loading="lazy" onerror="this.closest('.card').style.display='none'" style="width:100%; height:100%; object-fit:cover; display:block;">
              </a>
              <div class="mono" style="font-size:10px; color:var(--text-muted); overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">${esc(it.source || "image")}</div>
              <a href="${esc(it.url)}" target="_blank" rel="noopener noreferrer" class="btn btn-sm" style="font-size:10px; text-align:center;">Open Original</a>
            </div>
          `).join("")}
        </div>
      </div>
    `;
    document.body.appendChild(modal);
    const close = () => {
      modal.remove();
      document.removeEventListener("keydown", onKey);
    };
    const onKey = (e) => {
      if (e.key === "Escape") close();
    };
    document.getElementById("btn-close-photo-gallery")?.addEventListener("click", close);
    modal.addEventListener("click", (e) => {
      if (e.target === modal) close();
    });
    document.addEventListener("keydown", onKey);
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
