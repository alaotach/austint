/* M5: Timeline & Leaflet OSM Map View (Normalized, Year-Grouped, Conditional Map Space) */

(function () {
  "use strict";

  let mapInstance = null;

  const esc = (str) => (window.OSINT_UI?.esc ? window.OSINT_UI.esc(str) : String(str ?? "")
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;"));

  function normalizeLabel(raw) {
    let s = String(raw || "").trim();
    if (/^https?:\/\//i.test(s)) {
      s = s.replace(/^https?:\/\/(www\.)?/i, "").replace(/\/+$/, "");
    }
    return s.replace(/\s+/g, " ");
  }

  function extractYear(dateStr) {
    const m = String(dateStr || "").match(/\b((?:19|20)\d{2})\b/);
    return m ? m[1] : "Undated / Milestone";
  }

  function renderTimelineMapView() {
    const container = document.getElementById("timeline-container");
    if (!container) return;

    const nodes = window.OSINT_STATE.nodes || [];
    const dated = [];
    const geoPoints = [];

    const seenTl = new Set();
    const pushDated = (id, type, label, dateStr) => {
      const ds = String(dateStr || "").trim();
      if (!ds) return;
      const normLbl = normalizeLabel(label);
      const year = extractYear(ds);
      const dedupeKey = `${year}|${String(type || "").toLowerCase()}|${normLbl.toLowerCase()}`;
      if (seenTl.has(dedupeKey)) return;
      seenTl.add(dedupeKey);
      dated.push({ id, label: normLbl, rawLabel: String(label || ""), type, date: ds, year });
    };

    nodes.forEach((n) => {
      const d = n.data || {};
      const attrs = d.attributes || {};
      const ef = attrs.extra_fields || {};
      const dt = attrs.date || attrs.published_date || attrs.joined || attrs.created || attrs.created_at || attrs.updated_at;
      if (dt) pushDated(d.id, d.type, d.label, String(dt));

      const dob = attrs.dob || ef.dob;
      if (dob) pushDated(d.id, "DOB", `${attrs.name || d.label}`, String(dob));

      const careerArr = Array.isArray(attrs.career_timeline) ? attrs.career_timeline : (Array.isArray(ef.career_timeline) ? ef.career_timeline : []);
      careerArr.forEach((item) => {
        const s = String(item || "").trim();
        const m = s.match(/^([^:]+):\s*(.+)$/);
        if (m) pushDated(d.id, "Career", m[2], m[1]);
        else if (s) pushDated(d.id, "Career", s, "Milestone");
      });

      const textBlob = [attrs.experience, ef.experience, attrs.education, ef.education, attrs.bio].filter(Boolean).join(" | ");
      if (textBlob) {
        const reMil = /(?:Assistant Professor|Associate Professor|Professor|Ph\.?D\.?|M\.?Tech|B\.?E\.?)[^|.\n]{0,90}?\b((?:19|20)\d{2})\b/gi;
        let m;
        while ((m = reMil.exec(textBlob)) !== null) {
          pushDated(d.id, "Career", m[0].replace(/\s+/g, " ").trim(), m[1]);
        }
      }

      const lat = attrs.lat ?? attrs.latitude;
      const lon = attrs.lon ?? attrs.longitude;
      if (lat != null && lon != null && !Number.isNaN(Number(lat)) && !Number.isNaN(Number(lon))) {
        geoPoints.push({ id: d.id, label: d.label, type: d.type, lat: Number(lat), lon: Number(lon) });
      }
    });

    dated.sort((a, b) => a.date.localeCompare(b.date));

    // Group by year
    const byYear = new Map();
    dated.forEach((item) => {
      if (!byYear.has(item.year)) byYear.set(item.year, []);
      byYear.get(item.year).push(item);
    });

    const hasGeo = geoPoints.length > 0;

    container.innerHTML = `
      ${!hasGeo ? `
        <div class="card" style="padding:12px 16px; display:flex; align-items:center; justify-content:space-between; gap:12px;">
          <span class="field-label" style="margin:0;">Geolocated Places &amp; EXIF GPS (0)</span>
          <span class="mono" style="font-size:12px; color:var(--text-muted);">No coordinates (places or EXIF GPS) in the current run.</span>
        </div>
      ` : ""}

      <div class="${hasGeo ? "grid-2" : ""}">
        <div class="card" style="display:flex; flex-direction:column; gap:12px;">
          <div class="field-label" style="margin:0;">Dated Events Timeline (${dated.length} deduplicated)</div>
          ${dated.length ? `
            <div style="display:flex; flex-direction:column; gap:14px;">
              ${Array.from(byYear.entries()).map(([yr, items]) => `
                <div class="tl-year-group">
                  <div class="tl-year-header">${esc(yr)} &middot; ${items.length} event${items.length === 1 ? "" : "s"}</div>
                  ${items.map((item) => `
                    <button type="button" class="tl-row" data-tl-node="${esc(item.id)}" data-tooltip="${esc(item.type + ": " + item.label)}">
                      <span class="badge badge-accent" style="flex-shrink:0; min-width:92px; justify-content:center;">${esc(item.date.slice(0, 10))}</span>
                      <span class="badge" style="flex-shrink:0;">${esc(item.type)}</span>
                      <span style="min-width:0; flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; font-size:13px;">${esc(item.label)}</span>
                    </button>
                  `).join("")}
                </div>
              `).join("")}
            </div>
          ` : `<div class="state-panel">No dated events extracted in the current run.</div>`}
        </div>

        ${hasGeo ? `
          <div class="card" style="display:flex; flex-direction:column; gap:12px;">
            <div class="field-label" style="margin:0;">Geolocated Places &amp; EXIF GPS (${geoPoints.length})</div>
            <div id="leaflet-map" style="width:100%; height:440px; border-radius:var(--radius-sm); border:1px solid var(--border-subtle);"></div>
          </div>
        ` : ""}
      </div>
    `;

    container.querySelectorAll("[data-tl-node]").forEach((btn) => {
      btn.addEventListener("click", () => {
        window.OSINT_UI?.openInspector(btn.dataset.tlNode);
      });
    });

    const mapEl = document.getElementById("leaflet-map");
    if (mapEl && hasGeo && typeof window.L !== "undefined") {
      if (mapInstance) {
        mapInstance.remove();
        mapInstance = null;
      }
      mapInstance = window.L.map(mapEl).setView([geoPoints[0].lat, geoPoints[0].lon], 4);
      window.L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
        maxZoom: 18,
        attribution: "&copy; OpenStreetMap contributors",
      }).addTo(mapInstance);
      geoPoints.forEach((pt) => {
        const marker = window.L.marker([pt.lat, pt.lon]).addTo(mapInstance);
        marker.bindPopup(`<strong>${esc(pt.type)}</strong><br>${esc(pt.label)}`);
        marker.on("click", () => window.OSINT_UI?.openInspector(pt.id));
      });
    }
  }

  window.renderTimelineMapView = renderTimelineMapView;
  document.addEventListener("DOMContentLoaded", renderTimelineMapView);
})();
