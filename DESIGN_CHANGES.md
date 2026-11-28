# Design Engineering Changes — Intelligent OSINT (`AUSTINT`)

Applied Emil Kowalski's `emil-design-eng` philosophy across all design tokens, shared components, overlays, and views.

---

## 1. Detailed Change Log

| Component / Screen | Before | After | Why (principle applied) |
| --- | --- | --- | --- |
| **App Shell & Sidebar (`index.html`, `app.css`)** | Flat 220px rail with numbered text labels (`1. Launch`..), 4 header badges, 16px/20px padding | Brand mark + single live status pill + grouped telemetry pill in frosted header (`12px 24px`); `236px` sidebar with grouped `WORKFLOW` / `INTELLIGENCE` / `GOVERNANCE` section headers and left accent indicator; workspace `28px 32px` padding (`max-width: 1400px`) with bottom hairline `.view-header` | Establishes clear information hierarchy, breathing room, and Linear/Vercel-grade shell framing |
| **Launch Workspace (`launch_scan.js`, `app.css`)** | Full-width form card stacked awkwardly above a full-width Saved Snapshots card | Two-column `.launch-split` (`1.45fr` Setup card with `.segmented-bar` tabs + `24px` padding + hairline section divider; `0.95fr` Saved Snapshots side panel) | Balances viewport width at `1280px+` and eliminates unnecessary vertical scrolling |
| **Subject Identity Dossier (`results.js`, `components.css`)** | Cramped horizontal flex row with inline styles, narrow `190px` summary column, no citations | 2-column `.dossier-card` (`240px` left portrait/gallery rail separated by `1px` vertical hairline + 3-col `.dossier-facts-grid` + full-width `Profile Summary` callout strips) with clickable Wikipedia-style `[1]`, `[2]` citations & collapsible References footer | Makes dense entity dossiers immediately scannable while linking every fact back to its source card |
| **Result Cards & Inspector (`results.js`, `inspector.js`, `components.css`)** | Cluttered unspaced key-value rows, `1-line` clipped titles, internal `field_sources`/`ml_*` keys leaked in Inspector Overview | Generous `18px` card padding, `2-line` title clamp, `.result-kv-list` / `.result-kv-row` hairline dividers, slim aligned card footers, and clean filtering of internal ML dicts from Overview (`v=19`, `v=3`) | Reduces visual noise so every field and card action reads clearly at a glance |
| **Tokens (`tokens.css`)** | Built-in `ease` and `cubic-bezier(0.22, 1, 0.36, 1)`; single `150ms` / `220ms` timing | `--ease-out: cubic-bezier(0.23, 1, 0.32, 1)`, `--ease-in-out: cubic-bezier(0.77, 0, 0.175, 1)`, `--ease-drawer: cubic-bezier(0.32, 0.72, 0, 1)`, `--dur-press: 120ms`, `--dur-fast: 150ms`, `--dur-drawer: 280ms` | Custom curves give immediate punch on enter/press and natural deceleration on drawers; never use `ease-in` |
| **Buttons (`.btn`, `components.css`)** | `.btn:hover` active on all pointers; `100ms` press | `:hover` gated inside `@media (hover: hover) and (pointer: fine)`; `:active { transform: scale(0.97) }` over `120ms var(--ease-out)` | Prevents sticky hover on touch screens while giving crisp tactile press feedback |
| **Tabs (`.tab-btn`, `components.css`)** | No `:active` press state; un-gated `:hover` | Added `:active { transform: scale(0.97) }`, `120ms var(--ease-out)`, and pointer-gated `:hover` | Consistent pressable feedback across Launch, Results, Inspector, and Audit tabs |
| **Nav Rail (`.nav-item`, `app.css`)** | Un-gated `:hover`; no press state | Pointer-gated `:hover`; `:active { transform: scale(0.98) }` over `120ms var(--ease-out)` | Prevents stuck mobile nav highlights and confirms tap/click immediately |
| **Result Cards (`.card-interactive`, `components.css`)** | `translateY(-1px)` on hover; no press state | Static position on hover (pointer-gated border highlight); `:active { transform: scale(0.99) }` | Eliminates vertical jitter when scanning dense grids of cards |
| **Inspector Drawer (`#inspector-drawer`, `app.js`, `components.css`)** | `transform 0.2s ease`, no backdrop, no focus trap, no focus restore, no swipe dismiss | `transform 280ms var(--ease-drawer)`, `.drawer-backdrop` (`220ms var(--ease-out)`), Tab focus trap, focus restoration to trigger, swipe-right velocity dismissal (`dx / dt > 0.11` or `dx > 120px`) | Drawers are occasional overlays; Vaul-style curve + velocity swipe + full keyboard/ARIA focus management |
| **Photo Gallery Modal (`openPhotoGalleryModal`, `results.js`, `components.css`)** | Raw inline `position:fixed` appearing abruptly (`scale(1)` instant), no focus restore | `.modal-backdrop` + centered `.modal-dialog` entering from `scale(0.95); opacity: 0` -> `scale(1); opacity: 1` over `200ms var(--ease-out)`, `role="dialog" aria-modal="true"`, `Esc` close + focus restore | Modals stay centered and never animate from `scale(0)` |
| **Toast Stack (`.toast-stack`, `app.js`, `components.css`)** | No toast system; inline `"Saved!"` button text only | Sonner-style `.toast-stack` (`window.OSINT_UI.toast`), max 4 stacked, `translateY(12px) scale(0.96)` -> `0` (`240ms var(--ease-out)`), hover pause, swipe-to-dismiss (`|dx|/dt > 0.11`) | Clear, non-blocking, interruptible confirmation for saves and errors |
| **Citations & Highlight (`.cite-link`, `.cite-target-pulse`, `results.js`, `components.css`)** | `1.8s` pulse keyframe; un-gated hover | Pointer-gated `.cite-link:hover`; crisper `1.2s var(--ease-out)` `.cite-target-pulse` | Faster visual settle when jumping from Dossier superscript `[n]` to source card |
| **Touch Targets (`components.css`)** | `.btn-sm` / `.tab-btn` / `.input` < 44px on touch | `@media (pointer: coarse)` enforces `min-height: 44px` on `.btn`, `.tab-btn`, `.input`, `.select` | Meets 44x44px touch accessibility guidelines |
| **Live Status Dot (`.status-dot.running`, `app.css`)** | Static blue dot | Subtle `statusPulse` (`1.6s ease-in-out infinite` opacity) | Communicates live running state at a glance |
| **Reduced Motion (`app.css`)** | None | `@media (prefers-reduced-motion: reduce)` reduces animation/transition durations to `0.01ms` | Respects OS reduced-motion preference while keeping state colors visible |

---

## 2. Intentionally Not Animated

1. **Primary View Switching (`.view-section.active`)**: Users switch between Launch, Live Scan, Results, Graph, Timeline, Audit, and Report constantly (and via keyboard/citations). Zero enter animation keeps navigation instant.
2. **Global Search (`/` shortcut & filtering) & `Esc` Close**: Keyboard-initiated actions must never feel slower than the keypress.
3. **Result Card Lists & Table Rows**: High-frequency filtering/sorting across dozens of OSINT nodes and hundreds of audit rows updates immediately without stagger delays.
4. **Cytoscape Graph & Leaflet Map Containers**: Internal canvas/map pan/zoom are handled by their respective libraries without wrapper transitions.

---

## 3. Verification & Follow-Ups

- **Verified**:
  - `./.venv/bin/pytest tests/test_api.py -q` passed after all changes.
  - Captured and visually inspected live Chromium screenshots at `1440x900` (Launch, Results Dossier, Result Cards, Inspector Drawer) and `375x812` mobile (`CITE COUNT: 324` interactive Wikipedia citations verified).
- **Follow-ups / Manual Browser Checks**:
  - Pull latest commit and hard-refresh (`Ctrl+Shift+R`) to load `tokens.css?v=5`, `components.css?v=5`, `app.css?v=5`, `app.js?v=2`, `launch_scan.js?v=9`, `results.js?v=19`, `inspector.js?v=3`.
  - Test swipe-to-dismiss on the Inspector drawer and Toast stack on a touch device.
