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

    nodes.forEach((n) => {
      const d = n.data || {};
      const attrs = d.attributes || {};
      const dt = attrs.date || attrs.published_date || attrs.joined || attrs.created || attrs.created_at || attrs.updated_at;
      if (dt) {
        dated.push({ id: d.id, label: d.label, type: d.type, date: String(dt) });
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
