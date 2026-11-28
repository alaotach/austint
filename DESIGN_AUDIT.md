# Design Engineering Audit — Intelligent OSINT (`AUSTINT`)

**Methodology**: Audited against Emil Kowalski's `emil-design-eng` principles (animation frequency vs. purpose, custom easing curves, press feedback, origin-aware overlays, `@media (hover: hover) and (pointer: fine)` touch safety, interruptible transitions, WCAG AA accessibility & focus management).

---

## 1. Project Architecture Map

| Layer | Technology / Files |
| --- | --- |
| **Framework & Routing** | Vanilla JS multi-view SPA served by FastAPI (`static/index.html`, `static/js/app.js` view switching via `.view-section.active`) |
| **Styling Approach** | Custom CSS Design Tokens (`static/css/tokens.css`), Component Library (`static/css/components.css`), App Shell & Responsive Grid (`static/css/app.css`) |
| **Animation Stack** | Native CSS transitions & keyframes (no external motion library needed) |
| **Visualization Libraries** | Cytoscape.js 3.30.2 (`graph_view.js`), Leaflet 1.9.4 (`timeline_map.js`) |
| **Screens / Views** | `1. Launch` (`launch_scan.js`), `2. Live Scan` (`launch_scan.js`), `3. Results` (`results.js`), `4. Graph` (`graph_view.js`), `5. Timeline / Map` (`timeline_map.js`), `6. Audit & Planner` (`audit_report.js`), `7. Report` (`audit_report.js`) |
| **Shared Overlays** | Inspector Slide-over Drawer (`#inspector-drawer` / `inspector.js`), Photo Gallery Modal (`#photo-gallery-modal` / `results.js`) |

---

## 2. Interactive Element & Motion Inventory

| Element / Component | Screen(s) | Frequency | Input Modality | Current Motion / Feedback |
| --- | --- | --- | --- | --- |
| Primary Nav Rail (`.nav-item`) | Global Shell | Constantly | Mouse / Touch / Keyboard | Instant class toggle; no press scale; hover not gated by pointer type |
| Global Search (`#global-search`) | Header | Constantly | Keyboard (`/`) / Mouse | Instant focus ring; no clear/feedback micro-state |
| Theme Toggle (`#theme-toggle`) | Header | Rare / Occasional | Mouse / Touch | Instant attribute swap; `.btn` press scale |
| Mode & Filter Tabs (`.tab-btn`) | Launch, Results, Inspector, Audit | Constantly | Mouse / Touch | Instant class toggle; lacks `:active` press scale |
| Buttons (`.btn`, `.btn-primary`, `.btn-danger`) | All Views | Constantly | Mouse / Touch / Keyboard | `scale(0.97)` on `:active`, `140ms` |
| Form Inputs, Selects, Textareas | Launch, Filters | Constantly | Keyboard / Mouse / Touch | Static border; no transition on focus/hover; touch targets `< 44px` on mobile |
| Result Cards (`.result-card`) | Results | Constantly | Mouse / Touch / Keyboard | Border color transition; clickable to open drawer |
| Superscript Citations (`.cite-link`) | Results Dossier | Occasional | Mouse / Keyboard | Smooth scroll + `.cite-target-pulse` keyframe on target card |
| Inspector Drawer (`#inspector-drawer`) | Global Overlay | Occasional | Mouse / Touch / Keyboard (`Esc`) | `translateX(100%) -> 0` over `320ms --ease-drawer`; **missing** backdrop scrim, focus trap, focus restoration, and swipe-to-dismiss |
| Photo Gallery Modal (`#photo-gallery-modal`) | Results | Rare / Occasional | Mouse / Touch / Keyboard | Appears abruptly with no `scale(0.96) + opacity 0 -> 1` entrance, no `Esc` key handler, no focus restoration |
| Toast / Action Feedback | Launch / Save Snapshot | Occasional | Automated / Action | Inline button text swap (`"Saved!"`) only; no unified stackable toast feedback for errors/copy/saves |
| `<details>` Accordions | Results Cards, Dossier | Occasional | Mouse / Touch | Native instant snap; no hover affordance |
| Live Scan Status Dot (`.status-dot.running`) | Header, Live Scan | Occasional | Automated | Static dot without active pulse indicator |

---

## 3. Severity-Grouped Findings

### Broken Feel
1. **Inspector Drawer lacks backdrop, focus trap, focus restoration, and swipe-to-dismiss**: Opening `#inspector-drawer` leaves background controls tabbable, `Esc` doesn't return focus to the triggering card/button, and mobile users cannot swipe right to dismiss.
2. **Photo Gallery Modal snaps in/out without entrance/exit, `Esc` close, or focus management**: Created with raw inline styles (`position:fixed`), appears from nothing, and lacks keyboard `Escape` support.
3. **Touch targets on mobile (`<= 600px`) are undersized**: `.btn-sm`, `.tab-btn`, and `.cite-link` drop below `44x44px` on coarse-pointer touch screens.

### Inconsistent
4. **Hover effects apply on touch devices**: `.btn:hover`, `.nav-item:hover`, `.tab-btn:hover`, and `.card-interactive:hover` are not wrapped in `@media (hover: hover) and (pointer: fine)`, causing sticky hover states after taps on mobile.
5. **Inline styles in JS bypass design tokens**: `results.js`, `inspector.js`, `launch_scan.js`, and `timeline_map.js` mix hardcoded pixel radii (`border-radius:4px`), ad-hoc font sizes, and inline transitions instead of shared token classes.
6. **Custom `select` / `input` focus & hover states**: Inputs lack smooth border/ring transitions and consistent hover feedback.

### Missing Feedback
7. **Tabs (`.tab-btn`), Nav items (`.nav-item`), and Cards (`.card-interactive`) lack `:active` press feedback**: Only `.btn` had `scale(0.97)` on `:active`.
8. **No unified Toast notification system**: Actions like saving a snapshot, deleting a saved run, copying/exporting JSON, or encountering API errors rely on silent or inconsistent inline text changes instead of Sonner-style stackable, swipe-dismissible toasts.
9. **Running scan status dot (`.status-dot.running`) is static**: Gives no subtle visual pulse to distinguish an active scan from an idle one.
10. **Photo thumbnails & Gallery triggers**: Lack subtle loading/hover affordance and keyboard focus styles.

### Over-Animated / Mis-Timed
11. **None currently on view switches** (already made instant in prior pass), **but** `.cite-target-pulse` uses a `1.8s` keyframe where a shorter `1.2s` highlight with `--ease-out` feels crisper and less distracting.
