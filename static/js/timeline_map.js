/* M5: Timeline & Leaflet OSM Map View */

(function () {
  "use strict";

  let mapInstance = null;

  function esc(str) {
    return String(str ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
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
      const key = `${id}|${ds}|${label}`;
      if (seenTl.has(key)) return;
      seenTl.add(key);
      dated.push({ id, label, type, date: ds });
    };

    nodes.forEach((n) => {
      const d = n.data || {};
      const attrs = d.attributes || {};
      const ef = attrs.extra_fields || {};
      const dt = attrs.date || attrs.published_date || attrs.joined || attrs.created || attrs.created_at || attrs.updated_at;
      if (dt) {
        pushDated(d.id, d.type, d.label, String(dt));
      }
      const dob = attrs.dob || ef.dob;
      if (dob) {
        pushDated(d.id, "DOB", `${attrs.name || d.label}`, String(dob));
      }
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

    container.innerHTML = `
      <div class="grid-2">
        <div class="card">
          <div class="field-label" style="margin-bottom: var(--space-3);">Dated Events Timeline (${dated.length})</div>
          ${dated.length ? `
            <div style="display:flex; flex-direction:column; gap:8px; max-height:420px; overflow-y:auto;">
              ${dated.map((item) => `
                <button type="button" class="btn" style="justify-content:space-between; width:100%;" data-tl-node="${esc(item.id)}">
                  <span class="badge badge-accent">${esc(item.date)}</span>
                  <span style="overflow:hidden; text-overflow:ellipsis;">${esc(item.type)}: ${esc(item.label)}</span>
                </button>
              `).join("")}
            </div>
          ` : `<div class="state-panel">No dated events extracted in the current run.</div>`}
        </div>

        <div class="card">
          <div class="field-label" style="margin-bottom: var(--space-3);">Geolocated Places &amp; EXIF GPS (${geoPoints.length})</div>
          ${geoPoints.length ? `
            <div id="leaflet-map" style="width:100%; height:380px; border-radius:var(--radius-sm); border:1px solid var(--border-subtle);"></div>
          ` : `<div class="state-panel" style="height:260px; display:flex; align-items:center; justify-content:center;">No coordinates (places or EXIF GPS) in the current run.</div>`}
        </div>
      </div>
    `;

    container.querySelectorAll("[data-tl-node]").forEach((btn) => {
      btn.addEventListener("click", () => {
        window.OSINT_UI.openInspector(btn.dataset.tlNode);
      });
    });

    const mapEl = document.getElementById("leaflet-map");
    if (mapEl && geoPoints.length && typeof window.L !== "undefined") {
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
        marker.on("click", () => window.OSINT_UI.openInspector(pt.id));
      });
    }
  }

  window.renderTimelineMapView = renderTimelineMapView;
  document.addEventListener("DOMContentLoaded", renderTimelineMapView);
})();
