/* M3: Result Cards & Source-Cited Dossier (Confidence Ranking, Sanitizer, Pagination, Citations) */

(function () {
  "use strict";

  let typeFilter = "all";
  let minConfFilter = 0;
  let cardViewMode = "compact"; // "compact" (default) | "detailed"
  let currentPage = 1;
  const PAGE_SIZE = 36;
  const dismissedFacts = new Set();

  const esc = (str) => (window.OSINT_UI?.esc ? window.OSINT_UI.esc(str) : String(str ?? "")
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;"));

  function confColor(conf) {
    if (conf >= 0.75) return "var(--status-done)";
    if (conf >= 0.5) return "var(--accent)";
    if (conf >= 0.3) return "var(--status-unavailable)";
    return "var(--status-error)";
  }

  function isJunkText(val, fieldKind) {
    const s = String(val || "").trim();
    if (!s) return true;
    const lower = s.toLowerCase();
    if (/cookie|privacy policy|terms of service|accept all|we use cookies|gdpr|javascript is disabled|enable javascript|access denied|cloudflare|thank you|shared your comment|before you go/i.test(lower)) {
      return true;
    }
    if (fieldKind === "name") {
      if (/^(watch|who is email|login|sign in|sign up|home|profile|404|not found|discord|kick|reddit|github|twitter|x|instagram|facebook|youtube|linkedin|wordpress|medium|substack|gravatar)$/i.test(lower)) {
        return true;
      }
      if (s.length > 72 || /[.!?]\s+\w+/.test(s)) return true;
    }
    if (fieldKind === "role") {
      if (s.length > 96 || (s.split(/\s+/).length > 14)) return true;
    }
    return false;
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
    return chips.slice(0, 3);
  }

  function renderEntityFields(d) {
    const attrs = d.attributes || {};
    const rows = [];
    const seenKeys = new Set();
    const skipKeys = new Set([
      "url", "website", "title", "site", "bio", "description",
      "avatar", "avatar_url", "banner", "banner_url", "images",
      "source_tool", "resolver_features", "hand_features",
      "field_sources", "field_confidence", "ml_keywords", "ml_summary", "ml_cluster_id",
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
      if (text.length > 220) {
        rows.push(`
          <details class="result-kv-row" style="display:block;" onclick="event.stopPropagation();">
            <summary style="cursor:pointer; color:var(--text-muted); list-style:none;">${esc(label)} <span style="color:var(--accent);">(${text.length} chars)</span></summary>
            <div style="margin-top:6px; padding:8px 10px; border-radius:var(--radius-sm); background:var(--bg-elevated); white-space:pre-wrap; word-break:break-word; max-height:160px; overflow:auto;">${esc(text)}</div>
          </details>
        `);
      } else {
        rows.push(`<div class="result-kv-row"><span style="color:var(--text-muted); flex-shrink:0;">${esc(label)}</span><span style="text-align:right; word-break:break-word; color:var(--text-primary); max-width:68%;">${esc(text)}</span></div>`);
      }
    };

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

    const hasSensitive = Boolean(attrs.phone || attrs.phones || attrs.breach || attrs.stealer_family);
    let sensitiveHtml = "";
    if (hasSensitive) {
      sensitiveHtml = `
        <div style="margin-top:6px;">
          <button type="button" class="btn btn-sm btn-danger" onclick="event.stopPropagation(); this.nextElementSibling.style.display='block'; this.style.display='none';">
            Sensitive (Authorized Use Only) - Reveal
          </button>
          <div class="mono" style="display:none; font-size:11px; color:var(--status-error); margin-top:4px;">
            ${esc(JSON.stringify({ phone: attrs.phone || attrs.phones, breach: attrs.breach || attrs.stealer_family }))}
          </div>
        </div>
      `;
    }

    if (!rows.length && !sensitiveHtml) {
      const note = attrs.status_note || (attrs.limited_public_data ? "Exists, limited public data (login-walled)." : "Found, little public data exposed by source.");
      return `<div class="state-panel" style="padding: var(--space-3); font-size:11px;">${esc(note)}</div>`;
    }

    const headRows = rows.slice(0, 6);
    const tailRows = rows.slice(6);
    const tailHtml = tailRows.length ? `
      <div class="card-extra-fields" style="display:none;">${tailRows.join("")}</div>
      <button type="button" class="btn btn-xs btn-ghost" style="align-self:flex-start; margin-top:4px;" onclick="event.stopPropagation(); const ex=this.previousElementSibling; const open=ex.style.display==='block'; ex.style.display=open?'none':'block'; this.textContent=open?'Show all (${rows.length})':'Show less';">Show all (${rows.length})</button>
    ` : "";

    return `<div class="result-kv-list">${headRows.join("")}${tailHtml}${sensitiveHtml}</div>`;
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

    const avatarHtml = attrs.avatar || attrs.avatar_url
      ? `<img src="${esc(attrs.avatar || attrs.avatar_url)}" alt="" width="34" height="34" loading="lazy" onerror="this.style.display='none'" style="border-radius:50%; object-fit:cover; border:1px solid var(--border-strong); flex-shrink:0;">`
      : "";

    if (cardViewMode === "compact") {
      const mainSnippet = [attrs.job_title, attrs.headline, attrs.location, attrs.email, attrs.bio, attrs.description]
        .find((v) => typeof v === "string" && v.trim()) || "";
      return `
        <article
          id="card-${esc(d.id)}"
          class="card card-interactive result-card ${selected}"
          tabindex="0"
          data-node-id="${esc(d.id)}"
          data-card-index="${idx}"
          aria-label="${esc(d.type)}: ${esc(displayLabel)}"
        >
          <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:var(--space-3);">
            <div style="display:flex; align-items:flex-start; gap:10px; min-width:0;">
              ${avatarHtml}
              <div style="min-width:0;">
                <div style="display:flex; align-items:center; gap:5px; flex-wrap:wrap;">
                  <span class="badge badge-accent">${esc(d.type)}</span>
                  <span class="badge">${esc(attrs.site || sourceTool)}</span>
                  <span class="badge">d=${esc(d.depth ?? 0)}</span>
                </div>
                <h3 title="${esc(rawLabel)}" style="margin:6px 0 0 0; font-size:15px; font-weight:600; letter-spacing:-0.01em; line-height:1.35; display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden;">${esc(displayLabel)}</h3>
              </div>
            </div>

            <div
              data-tooltip="Calibrated confidence: ${pct}%"
              style="width:40px; height:40px; flex-shrink:0; border-radius:50%; background:conic-gradient(${ringCol} ${pct * 3.6}deg, var(--bg-elevated) 0deg); display:flex; align-items:center; justify-content:center;"
            >
              <span class="mono" style="width:30px; height:30px; border-radius:50%; background:var(--bg-surface); display:flex; align-items:center; justify-content:center; font-size:11px; font-weight:700;">${pct}</span>
            </div>
          </div>

          ${mainSnippet ? `<p style="margin:0; font-size:12.5px; line-height:1.45; color:var(--text-secondary); display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden;">${esc(mainSnippet)}</p>` : ""}

          <div style="margin-top:auto; padding-top:10px; border-top:1px solid var(--border-subtle); display:flex; justify-content:space-between; align-items:center; gap:8px;">
            <span class="mono" style="font-size:11px; color:var(--text-muted);">${cov.extracted_count}/${cov.expected_count} fields</span>
            <div style="display:flex; align-items:center; gap:6px;">
              ${linkOut ? `<a href="${esc(linkOut)}" target="_blank" rel="noopener noreferrer" class="btn btn-xs" onclick="event.stopPropagation();">Open &#8599;</a>` : ""}
              <button type="button" class="btn btn-xs btn-primary" data-inspect-id="${esc(d.id)}">Details &rarr;</button>
            </div>
          </div>
        </article>
      `;
    }

    return `
      <article
        id="card-${esc(d.id)}"
        class="card card-interactive result-card ${selected}"
        tabindex="0"
        data-node-id="${esc(d.id)}"
        data-card-index="${idx}"
        aria-label="${esc(d.type)}: ${esc(displayLabel)}"
      >
        <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:var(--space-3);">
          <div style="display:flex; align-items:flex-start; gap:10px; min-width:0;">
            ${avatarHtml}
            <div style="min-width:0;">
              <div style="display:flex; align-items:center; gap:5px; flex-wrap:wrap;">
                <span class="badge badge-accent">${esc(d.type)}</span>
                <span class="badge">${esc(attrs.site || sourceTool)}</span>
                <span class="badge">d=${esc(d.depth ?? 0)}</span>
              </div>
              <h3 title="${esc(rawLabel)}" style="margin:6px 0 0 0; font-size:15px; font-weight:600; letter-spacing:-0.01em; line-height:1.35; display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden;">${esc(displayLabel)}</h3>
            </div>
          </div>

          <div
            data-tooltip="Calibrated confidence: ${pct}%"
            style="width:40px; height:40px; flex-shrink:0; border-radius:50%; background:conic-gradient(${ringCol} ${pct * 3.6}deg, var(--bg-elevated) 0deg); display:flex; align-items:center; justify-content:center;"
          >
            <span class="mono" style="width:30px; height:30px; border-radius:50%; background:var(--bg-surface); display:flex; align-items:center; justify-content:center; font-size:11px; font-weight:700;">${pct}</span>
          </div>
        </div>

        ${attrs.bio || attrs.description ? `<p style="margin:0; font-size:12.5px; line-height:1.5; color:var(--text-secondary); display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden;">${esc(attrs.bio || attrs.description)}</p>` : ""}

        ${chips.length ? `<div style="display:flex; flex-wrap:wrap; gap:5px;">${chips.map((c) => `<span class="chip">${esc(c)}</span>`).join("")}</div>` : ""}

        ${renderEntityFields(d)}

        <div style="margin-top:auto; padding-top:10px; border-top:1px solid var(--border-subtle); display:flex; flex-direction:column; gap:8px;">
          <div style="display:flex; justify-content:space-between; align-items:center; gap:8px; font-family:var(--font-mono); font-size:11px; color:var(--text-muted);">
            <span>${cov.extracted_count} / ${cov.expected_count} fields</span>
            ${cov.is_partial ? `<span class="badge badge-uncertain" data-tooltip="Missing: ${esc((cov.missing_fields || []).join(', '))}">Partial</span>` : `<span class="badge badge-confirmed">Complete</span>`}
          </div>
          <div class="progress-track"><div class="progress-fill" style="--progress:${cov.ratio};"></div></div>
          <div style="display:flex; justify-content:space-between; align-items:center; gap:8px; margin-top:2px;">
            ${linkOut ? `<a href="${esc(linkOut)}" target="_blank" rel="noopener noreferrer" class="btn btn-sm" onclick="event.stopPropagation();">Open Link &#8599;</a>` : `<span></span>`}
            <button type="button" class="btn btn-sm btn-primary" data-inspect-id="${esc(d.id)}">Inspect &rarr;</button>
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

    const factSources = new Map();
    const factConf = new Map();
    const citationByNode = new Map();
    const citations = [];
    const registerCite = (d) => {
      if (!d || !d.id) return null;
      if (citationByNode.has(d.id)) return citationByNode.get(d.id);
      const a = d.attributes || {};
      const title = a.title || a.name || a.display_name || a.site || d.label || d.id;
      const site = a.site || (typeof d.label === "string" && d.label.startsWith("http") ? (() => { try { return new URL(d.label).hostname.replace(/^www\./, ""); } catch { return ""; } })() : d.type);
      const url = a.url || (typeof d.label === "string" && d.label.startsWith("http") ? d.label : "");
      const rec = { num: citations.length + 1, nodeId: d.id, title: String(title).slice(0, 72), site: String(site || ""), url };
      citations.push(rec);
      citationByNode.set(d.id, rec);
      return rec;
    };
    const supHtml = (val) => {
      const nodeSet = factSources.get(val);
      if (!nodeSet || !nodeSet.size) return "";
      return Array.from(nodeSet).slice(0, 3).map((nid) => {
        const c = citationByNode.get(nid);
        if (!c) return "";
        return `<sup class="cite-ref"><a href="#card-${esc(nid)}" class="cite-link" data-cite-node="${esc(nid)}" data-tooltip="Source [${c.num}]: ${esc(c.site ? c.site + ' — ' + c.title : c.title)}">[${c.num}]</a></sup>`;
      }).join("");
    };

    const photoItems = [];
    const seenPhotoUrls = new Set();
    const pushPhoto = (u, srcLabel, linkUrl, conf) => {
      if (typeof u === "string" && u.startsWith("http") && !seenPhotoUrls.has(u)) {
        seenPhotoUrls.add(u);
        photoItems.push({ url: u, source: srcLabel || "source", pageUrl: linkUrl || u, conf: Number(conf ?? 0.5) });
      }
    };
    const addStr = (set, v, maxLen = 260, srcData = null, fieldKind = "") => {
      const s = String(v || "").trim();
      if (!s || s.length >= maxLen) return;
      if (fieldKind && isJunkText(s, fieldKind)) return;
      if (dismissedFacts.has(`${fieldKind}|${s}`)) return;
      set.add(s);
      const c = Number(srcData?.confidence ?? 0.5);
      factConf.set(s, Math.max(factConf.get(s) ?? 0, c));
      if (srcData && srcData.id) {
        registerCite(srcData);
        if (!factSources.has(s)) factSources.set(s, new Set());
        factSources.get(s).add(srcData.id);
      }
    };
    const isLikelyDomain = (v) => {
      const s = String(v || "").trim().toLowerCase();
      if (!s || /^(s\.no|m\.tech|b\.tech|b\.e|r\.k\..*|vision\.https|gmail\.co)$/.test(s)) return false;
      if (/^[a-z]\.[a-z]\./.test(s)) return false;
      if (Array.from(emails).some((em) => em.toLowerCase().startsWith(s + "@"))) return false;
      const tld = s.split(".").pop() || "";
      return tld.length >= 2 && tld.length <= 6;
    };
    const isJunkEmail = (v) => {
      const s = String(v || "").trim().toLowerCase();
      return !s || /^[0-9]{1,4}@/.test(s) || s.endsWith("@yopmail.com") || s.endsWith("@example.com");
    };

    const dossierNodes = allNodes.filter((n) => {
      const d = n.data || {};
      const lbl = String(d.label || "").toLowerCase();
      return !/pastebin\.com|eromexxx\.com|complaintwire\.org|site-bewertungen\.com/.test(lbl);
    });
    const sourceNodes = dossierNodes.length ? dossierNodes : allNodes;

    sourceNodes.forEach((n) => {
      const d = n.data || {};
      const a = d.attributes || {};
      const ef = a.extra_fields || {};
      const srcLabel = a.site || a.source_tool || d.type || "";
      const pageUrl = a.url || (String(d.label || "").startsWith("http") ? d.label : "");
      if (d.type === "email" && !isJunkEmail(d.label)) addStr(emails, d.label, 260, d, "email");
      if (d.type === "org") addStr(orgs, d.label, 260, d, "org");
      if (d.type === "domain" && isLikelyDomain(d.label)) addStr(websites, d.label, 260, d, "domain");
      [a.name, a.display_name, ef.legal_name].forEach((x) => addStr(names, x, 120, d, "name"));
      [a.email, a.developer_email, a.support_email, ef.developer_email, ef.support_email].forEach((x) => {
        if (x && !isJunkEmail(x)) addStr(emails, x, 260, d, "email");
      });
      (a.contact_emails || []).forEach((x) => {
        if (x && !isJunkEmail(x)) addStr(emails, x, 260, d, "email");
      });
      (a.phone_numbers || []).forEach((x) => addStr(phones, x, 260, d, "phone"));
      [a.location, a.office_address, ef.office_address, a.country].forEach((x) => addStr(locations, x, 120, d, "location"));
      [a.company, a.developer_name, ef.developer_name].forEach((x) => addStr(orgs, x, 160, d, "org"));
      (a.orgs || []).forEach((x) => addStr(orgs, x, 160, d, "org"));
      (a.organizations || []).forEach((x) => addStr(orgs, x, 160, d, "org"));
      [a.job_title, a.headline, ef.headline].forEach((x) => addStr(roles, x, 120, d, "role"));
      [a.dob, ef.dob].forEach((x) => addStr(dobs, x, 80, d, "dob"));
      [a.education, ef.education, a.guide_name ? `Guide: ${a.guide_name}` : "", ef.guide_name ? `Guide: ${ef.guide_name}` : ""].forEach((x) => addStr(education, x, 350, d, "edu"));
      (a.skills || []).forEach((x) => addStr(skills, x, 160, d, "skill"));
      [a.current_research_area, ef.current_research_area].forEach((x) => addStr(skills, x, 300, d, "skill"));
      (a.projects_or_stories || []).forEach((x) => addStr(projects, x, 300, d, "project"));
      if (a.llm_summary) addStr(summaries, a.llm_summary, 420, d, "summary");
      if (typeof a.website === "string" && a.website.startsWith("http")) addStr(websites, a.website, 260, d, "domain");
      [a.avatar_url, a.avatar].forEach((u) => pushPhoto(u, srcLabel, pageUrl, d.confidence));
      (a.images || []).forEach((u) => pushPhoto(u, srcLabel, pageUrl, d.confidence));
    });

    photoItems.sort((a, b) => b.conf - a.conf);
    window.__OSINT_DOSSIER_PHOTOS = photoItems;
    const highConfPhotos = photoItems.filter((p) => p.conf >= 0.5);
    const previewPhotos = highConfPhotos.length ? highConfPhotos : photoItems;
    const primaryPhoto = previewPhotos[0]?.url || "";

    const renderFactCell = (label, set, fieldKind, isLink, fullWidth = false, conflictCheck = false) => {
      const sorted = Array.from(set).sort((a, b) => (factConf.get(b) ?? 0.5) - (factConf.get(a) ?? 0.5));
      if (!sorted.length) return "";
      const highVals = sorted.filter((v) => (factConf.get(v) ?? 0.5) >= 0.45);
      const lowVals = sorted.filter((v) => (factConf.get(v) ?? 0.5) < 0.45);
      const showConflict = conflictCheck && highVals.length > 1;

      const renderVal = (v, isLow) => {
        const cite = supHtml(v);
        const pct = Math.round((factConf.get(v) ?? 0.5) * 100);
        const core = isLink
          ? `<a href="${esc(v.startsWith("http") ? v : "https://" + v)}" target="_blank" rel="noopener noreferrer" style="color:var(--accent);">${esc(v)}</a>`
          : esc(v);
        if (fullWidth) {
          return `<div style="padding:8px 12px; border-left:2px solid var(--border-strong); background:var(--bg-elevated); border-radius:0 6px 6px 0; font-family:var(--font-sans); font-size:12.5px; line-height:1.5;">${core}${cite}</div>`;
        }
        return `
          <div class="dossier-val-row ${isLow ? "is-low-conf" : ""}">
            <span style="min-width:0; word-break:break-word;">${core}${cite} <span style="font-size:10px; color:var(--text-muted);">(${pct}%)</span></span>
            <button type="button" class="dismiss-btn" data-dismiss-fact="${esc(fieldKind)}|${esc(v)}" title="Dismiss noisy value" aria-label="Dismiss value">&times;</button>
          </div>
        `;
      };

      const mainList = (highVals.length ? highVals : sorted.slice(0, 3)).slice(0, fullWidth ? 4 : 5);
      const hiddenLow = highVals.length ? lowVals : sorted.slice(3);

      return `
        <div class="dossier-fact-cell" ${fullWidth ? 'style="grid-column: 1 / -1;"' : ""}>
          <div style="display:flex; align-items:center; justify-content:space-between; gap:6px; margin-bottom:6px;">
            <span class="field-label" style="font-size:10px; margin-bottom:0;">${esc(label)}</span>
            ${showConflict ? `<span class="badge badge-unavailable" style="padding:1px 6px; font-size:10px;" data-tooltip="Multiple conflicting values across sources; ranked by confidence">Conflict (${highVals.length})</span>` : ""}
          </div>
          <div class="mono" style="font-size:12px; line-height:1.55; color:var(--text-primary); display:flex; flex-direction:column; gap:3px;">
            ${mainList.map((v) => renderVal(v, (factConf.get(v) ?? 0.5) < 0.45)).join("")}
            ${hiddenLow.length ? `
              <details style="margin-top:4px;">
                <summary style="cursor:pointer; font-size:11px; color:var(--text-muted);">+ ${hiddenLow.length} low-conf</summary>
                <div style="margin-top:4px; display:flex; flex-direction:column; gap:3px;">
                  ${hiddenLow.slice(0, 6).map((v) => renderVal(v, true)).join("")}
                </div>
              </details>
            ` : ""}
          </div>
        </div>
      `;
    };

    const hasAny = names.size || emails.size || phones.size || locations.size || orgs.size || websites.size || photoItems.length;
    if (!hasAny) return "";

    const refsHtml = citations.length ? `
      <details style="margin-top:var(--space-4); padding-top:var(--space-3); border-top:1px solid var(--border-subtle);">
        <summary class="field-label" style="cursor:pointer; margin-bottom:0;">References &amp; Sources (${citations.length})</summary>
        <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(240px, 1fr)); gap:8px 20px; margin-top:10px;">
          ${citations.map((c) => `
            <div class="mono" style="font-size:11px; display:flex; align-items:baseline; gap:6px; overflow:hidden;">
              <a href="#card-${esc(c.nodeId)}" class="cite-link" data-cite-node="${esc(c.nodeId)}" style="flex-shrink:0;">[${c.num}]</a>
              <a href="#card-${esc(c.nodeId)}" data-cite-node="${esc(c.nodeId)}" style="color:var(--text-secondary); text-decoration:none; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;" title="Jump to card: ${esc(c.title)}">${esc(c.site ? `${c.site} · ${c.title}` : c.title)}</a>
            </div>
          `).join("")}
        </div>
      </details>
    ` : "";

    const rankedNames = Array.from(names).sort((a, b) => (factConf.get(b) ?? 0.5) - (factConf.get(a) ?? 0.5));
    const rankedRoles = Array.from(roles).sort((a, b) => (factConf.get(b) ?? 0.5) - (factConf.get(a) ?? 0.5));

    return `
      <section class="card dossier-card">
        <div class="dossier-layout">
          <div class="dossier-profile-col">
            <div style="display:flex; align-items:center; gap:14px;">
              ${primaryPhoto ? `<a href="${esc(primaryPhoto)}" target="_blank" rel="noopener noreferrer" style="flex-shrink:0;"><img src="${esc(primaryPhoto)}" alt="" class="dossier-avatar" onerror="this.style.display='none'"></a>` : ""}
              <div style="min-width:0;">
                <div class="field-label" style="margin-bottom:2px;">Subject Identity Dossier</div>
                <div style="font-weight:700; font-size:19px; letter-spacing:-0.02em; line-height:1.2; word-break:break-word;">${esc(rankedNames[0] || "Aggregated Target Intel")}</div>
                ${rankedRoles[0] ? `<div style="margin-top:4px; font-size:12px; color:var(--text-secondary);">${esc(rankedRoles[0])}</div>` : ""}
              </div>
            </div>
            ${photoItems.length ? `
              <div style="display:flex; flex-direction:column; gap:8px; margin-top:4px;">
                <div class="mono" style="font-size:10px; color:var(--text-muted);">Collected Media (Unverified)</div>
                <div style="display:flex; flex-wrap:wrap; align-items:center; gap:6px;">
                  ${previewPhotos.slice(0, 5).map((p) => `
                    <a href="${esc(p.url)}" class="dossier-photo-trigger" data-tooltip="${esc(p.source)} (${Math.round(p.conf * 100)}%)">
                      <img src="${esc(p.url)}" alt="" loading="lazy" onerror="this.parentElement.style.display='none'" style="width:38px; height:38px; border-radius:6px; object-fit:cover; border:1px solid var(--border-subtle);">
                    </a>
                  `).join("")}
                </div>
                <button type="button" class="btn btn-sm" id="btn-open-photo-gallery" style="align-self:flex-start;">Open Gallery (${highConfPhotos.length} high-conf / ${photoItems.length} total)</button>
              </div>
            ` : ""}
          </div>
          <div class="dossier-facts-grid">
            ${renderFactCell("Names & Aliases", names, "name", false)}
            ${renderFactCell("Roles & Titles", roles, "role", false)}
            ${renderFactCell("Emails", emails, "email", false)}
            ${renderFactCell("Phone Numbers", phones, "phone", false)}
            ${renderFactCell("Date of Birth", dobs, "dob", false, false, true)}
            ${renderFactCell("Locations & Addresses", locations, "location", false, false, true)}
            ${renderFactCell("Organizations", orgs, "org", false)}
            ${renderFactCell("Education & Guide", education, "edu", false)}
            ${renderFactCell("Skills & Research", skills, "skill", false)}
            ${renderFactCell("Projects & Publications", projects, "project", false)}
            ${renderFactCell("Domains & Websites", websites, "domain", true)}
            ${renderFactCell("Profile Summary", summaries, "summary", false, true)}
          </div>
        </div>
        ${refsHtml}
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
      return (db.type === "profile" ? 1 : 0) - (da.type === "profile" ? 1 : 0);
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

    const totalPages = Math.max(1, Math.ceil(deduped.length / PAGE_SIZE));
    if (currentPage > totalPages) currentPage = totalPages;
    const startIdx = (currentPage - 1) * PAGE_SIZE;
    const visible = deduped.slice(startIdx, startIdx + PAGE_SIZE);
    const dossierHtml = buildDossierHtml(allNodes);

    container.innerHTML = `
      ${dossierHtml}

      <div class="sticky-filter-bar" style="display:flex; flex-wrap:wrap; align-items:center; justify-content:space-between; gap:var(--space-3);">
        <div class="segmented-bar">
          ${types.map((t) => `<button type="button" class="tab-btn ${typeFilter === t ? "active" : ""}" data-rtype="${esc(t)}">${esc(t)}</button>`).join("")}
        </div>
        <div style="display:flex; align-items:center; gap:10px; flex-wrap:wrap;">
          <div class="segmented-bar" role="group" aria-label="Card Layout Mode">
            <button type="button" class="tab-btn ${cardViewMode === "compact" ? "active" : ""}" data-cview="compact">Compact</button>
            <button type="button" class="tab-btn ${cardViewMode === "detailed" ? "active" : ""}" data-cview="detailed">Detailed</button>
          </div>
          <label class="mono" style="font-size:11px; color:var(--text-muted);" for="filter-min-conf">Min Conf: <strong id="lbl-min-conf" style="color:var(--text-primary);">${Math.round(minConfFilter * 100)}%</strong></label>
          <input type="range" id="filter-min-conf" min="0" max="0.9" step="0.1" value="${minConfFilter}">
          <span class="badge" data-tooltip="Deduplicated unique entities vs total raw graph nodes">${deduped.length} unique (${allNodes.length} nodes)</span>
          ${allNodes.length ? `<button type="button" class="btn btn-sm btn-primary" id="btn-save-snapshot">Save Snapshot</button>` : ""}
        </div>
      </div>

      <div style="display:flex; flex-direction:column; gap:var(--space-4);">
        ${visible.length ? `
          <div id="results-cards-grid" style="display:grid; grid-template-columns:repeat(auto-fill, minmax(310px, 1fr)); gap:16px;">
            ${visible.map((n, i) => renderResultCard(n, startIdx + i)).join("")}
          </div>
          ${totalPages > 1 ? `
            <div class="card" style="padding:12px 16px; display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:10px;">
              <span class="mono" style="font-size:12px; color:var(--text-muted);">Showing ${startIdx + 1}–${startIdx + visible.length} of ${deduped.length} unique entities</span>
              <div style="display:flex; align-items:center; gap:8px;">
                <button type="button" class="btn btn-sm" id="btn-page-prev" ${currentPage <= 1 ? "disabled" : ""}>&larr; Prev</button>
                <span class="mono" style="font-size:12px;">Page ${currentPage} / ${totalPages}</span>
                <button type="button" class="btn btn-sm" id="btn-page-next" ${currentPage >= totalPages ? "disabled" : ""}>Next &rarr;</button>
              </div>
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
          window.OSINT_UI?.toast(`Saved snapshot "${name.trim()}"`, "success");
          if (typeof window.refreshSavedRuns === "function") window.refreshSavedRuns();
          setTimeout(() => {
            btn.textContent = "Save Snapshot";
            btn.disabled = false;
          }, 1500);
        } else if (btn) {
          window.OSINT_UI?.toast("Failed to save snapshot", "error");
          btn.disabled = false;
        }
      } catch (_) {
        if (btn) btn.disabled = false;
      }
    });

    container.querySelectorAll("[data-dismiss-fact]").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        e.stopPropagation();
        dismissedFacts.add(btn.dataset.dismissFact);
        window.OSINT_UI?.toast("Dismissed noisy dossier value", "info");
        renderResultsView();
      });
    });

    document.getElementById("btn-open-photo-gallery")?.addEventListener("click", openPhotoGalleryModal);
    container.querySelectorAll(".dossier-photo-trigger").forEach((el) => {
      el.addEventListener("click", (e) => {
        e.preventDefault();
        openPhotoGalleryModal();
      });
    });

    container.querySelectorAll("[data-cite-node]").forEach((el) => {
      el.addEventListener("click", (e) => {
        e.preventDefault();
        e.stopPropagation();
        jumpToCitedCard(el.dataset.citeNode, deduped);
      });
    });

    document.getElementById("btn-page-prev")?.addEventListener("click", () => {
      if (currentPage > 1) { currentPage -= 1; renderResultsView(); window.scrollTo({ top: 0, behavior: "smooth" }); }
    });
    document.getElementById("btn-page-next")?.addEventListener("click", () => {
      if (currentPage < totalPages) { currentPage += 1; renderResultsView(); window.scrollTo({ top: 0, behavior: "smooth" }); }
    });

    container.querySelectorAll("[data-rtype]").forEach((btn) => {
      btn.addEventListener("click", () => {
        typeFilter = btn.dataset.rtype;
        currentPage = 1;
        renderResultsView();
      });
    });

    container.querySelectorAll("[data-cview]").forEach((btn) => {
      btn.addEventListener("click", () => {
        cardViewMode = btn.dataset.cview === "detailed" ? "detailed" : "compact";
        renderResultsView();
      });
    });

    const confSlider = document.getElementById("filter-min-conf");
    if (confSlider) {
      confSlider.addEventListener("input", () => {
        minConfFilter = Number(confSlider.value);
        currentPage = 1;
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
          card.nextElementSibling?.focus();
        } else if (e.key === "ArrowLeft" || e.key === "ArrowUp") {
          e.preventDefault();
          card.previousElementSibling?.focus();
        }
      });
    });
  }

  function jumpToCitedCard(nodeId, dedupedList) {
    if (!nodeId) return;
    window.OSINT_UI?.switchView("results");
    const container = document.getElementById("results-container");
    if (!container) return;
    let card = Array.from(container.querySelectorAll(".result-card")).find((c) => c.dataset.nodeId === nodeId);
    if (!card) {
      if (Array.isArray(dedupedList)) {
        const idx = dedupedList.findIndex((n) => n.data?.id === nodeId);
        if (idx >= 0) {
          currentPage = Math.floor(idx / PAGE_SIZE) + 1;
        } else {
          typeFilter = "all";
          minConfFilter = 0;
          currentPage = 1;
        }
      } else {
        typeFilter = "all";
        minConfFilter = 0;
        currentPage = 1;
      }
      renderResultsView();
      card = Array.from(container.querySelectorAll(".result-card")).find((c) => c.dataset.nodeId === nodeId);
    }
    if (card) {
      card.scrollIntoView({ behavior: "smooth", block: "center" });
      card.classList.remove("cite-target-pulse");
      void card.offsetWidth;
      card.classList.add("cite-target-pulse");
      card.focus({ preventScroll: true });
      setTimeout(() => card.classList.remove("cite-target-pulse"), 1800);
    }
  }
  window.OSINT_JUMP_TO_CARD = jumpToCitedCard;

  function openPhotoGalleryModal() {
    const allItems = window.__OSINT_DOSSIER_PHOTOS || [];
    let showLowConf = false;
    const triggerEl = document.activeElement;
    let modal = document.getElementById("osint-photo-gallery-modal");
    if (modal) modal.remove();
    modal = document.createElement("div");
    modal.id = "osint-photo-gallery-modal";
    modal.className = "modal-backdrop";
    modal.setAttribute("role", "dialog");
    modal.setAttribute("aria-modal", "true");
    modal.setAttribute("aria-label", "Collected Media Gallery");

    const close = () => {
      modal.classList.remove("open");
      document.removeEventListener("keydown", onKey);
      setTimeout(() => modal.remove(), 200);
      if (triggerEl && typeof triggerEl.focus === "function") triggerEl.focus({ preventScroll: true });
    };
    const onKey = (e) => { if (e.key === "Escape") close(); };

    const renderGalleryBody = () => {
      const items = showLowConf ? allItems : allItems.filter((it) => it.conf >= 0.5);
      const list = items.length ? items : allItems;
      modal.innerHTML = `
        <div class="card modal-dialog" style="gap:12px; padding:20px;">
          <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;">
            <div>
              <div class="field-label">Collected Media (Unverified — Multiple Sources)</div>
              <div style="font-weight:700; font-size:16px;">${list.length} of ${allItems.length} Images</div>
            </div>
            <div style="display:flex; align-items:center; gap:10px;">
              <label class="mono" style="font-size:11px; display:flex; align-items:center; gap:6px; cursor:pointer;">
                <input type="checkbox" id="chk-gallery-lowconf" ${showLowConf ? "checked" : ""}>
                Show low-confidence (&lt;50%)
              </label>
              <button type="button" class="btn btn-sm" id="btn-close-photo-gallery">Close (Esc)</button>
            </div>
          </div>
          <div style="overflow-y:auto; display:grid; grid-template-columns:repeat(auto-fill, minmax(156px, 1fr)); align-content:start; gap:12px; padding-right:4px;">
            ${list.map((it) => `
              <div class="card" style="padding:8px; display:flex; flex-direction:column; gap:8px; overflow:visible;">
                <a href="${esc(it.url)}" target="_blank" rel="noopener noreferrer" style="display:block; width:100%; height:136px; flex-shrink:0; overflow:hidden; border-radius:4px; background:var(--bg-elevated);">
                  <img src="${esc(it.url)}" alt="" loading="lazy" onerror="this.closest('.card').style.display='none'" style="width:100%; height:100%; object-fit:cover; display:block;">
                </a>
                <div style="display:flex; justify-content:space-between; align-items:center; gap:4px;">
                  <span class="mono" style="font-size:10px; color:var(--text-muted); overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">${esc(it.source || "image")}</span>
                  <span class="badge ${it.conf >= 0.5 ? "badge-done" : "badge-unavailable"}" style="padding:1px 5px; font-size:10px; flex-shrink:0;">${Math.round(it.conf * 100)}%</span>
                </div>
                <a href="${esc(it.url)}" target="_blank" rel="noopener noreferrer" class="btn btn-xs" style="text-align:center;">Open Original</a>
              </div>
            `).join("")}
          </div>
        </div>
      `;
      document.getElementById("btn-close-photo-gallery")?.addEventListener("click", close);
      document.getElementById("chk-gallery-lowconf")?.addEventListener("change", (e) => {
        showLowConf = e.target.checked;
        renderGalleryBody();
      });
    };

    document.body.appendChild(modal);
    renderGalleryBody();
    requestAnimationFrame(() => modal.classList.add("open"));
    document.getElementById("btn-close-photo-gallery")?.focus({ preventScroll: true });
    modal.addEventListener("click", (e) => { if (e.target === modal) close(); });
    document.addEventListener("keydown", onKey);
  }

  window.renderResultsView = renderResultsView;
  window.onGraphUpdated = function () {
    renderResultsView();
    if (typeof window.onGraphUpdatedHook === "function") window.onGraphUpdatedHook();
  };
  window.onSearchChanged = function () {
    currentPage = 1;
    renderResultsView();
  };
  document.addEventListener("DOMContentLoaded", renderResultsView);
})();
