# UI Redesign Audit Log (`ui-redesign`)

## Milestone 1: Design Tokens + Component Library + App Shell & Navigation
- **Iteration 1**:
  - Defect found at `375px`: `.nav-rail` stretched vertically inside `.app-shell` CSS Grid (`launch_375.png`).
  - Fix: Added `grid-template-rows: auto minmax(0, 1fr); align-content: start;` and `flex-shrink: 0` on `.nav-item` under `@media (max-width: 960px)` in `static/css/app.css`.
- **Iteration 2** (Verified at `375`, `768`, `1440`):
  - Hierarchy & scannability: **5/5**, Information density: **5/5**, Consistency: **5/5**, State coverage: **4/5**, Responsiveness: **5/5**, Accessibility: **5/5**, Console/network: **5/5** (0 errors).

## Milestone 2: Launch Screen + Live Scan View
- **Iteration 1**:
  - Defect found: `/api/collectors` returned HTTP 500 due to `tool.supported_types` instead of `tool.target_types`. Also 7 collectors fell into `"General"` category.
  - Fix: Fixed `tool.target_types` in `osint_agent/api/server.py` and mapped all 14 exact `TOOL_REGISTRY` names in `COLLECTOR_METADATA`.
- **Iteration 2** (Verified at `375`, `768`, `1440` with recorded run):
  - Hierarchy & scannability: **5/5**, Information density: **5/5**, Consistency: **5/5**, State coverage: **5/5**, Responsiveness: **5/5**, Accessibility: **5/5**, Console/network: **5/5** (0 errors).

## Milestone 3: Result Cards + Completeness Meters + Partial States
- **Iteration 1**:
  - Defect found at `375px`: `white-space: nowrap` on `Missing: ...` caused `article.result-card` to overflow the CSS Grid track and clip the confidence ring. Also raw `breach:`/`infostealer:` labels were visible in recorded runs.
  - Fix: Added `min-width: 0; max-width: 100%; overflow: hidden;` on `.card` in `static/css/components.css` and masked sensitive `breach:`/`infostealer:` labels in `static/js/results.js`.
- **Iteration 2** (Verified at `375`, `768`, `1440`):
  - Hierarchy & scannability: **5/5**, Information density: **5/5**, Consistency: **5/5**, State coverage: **5/5**, Responsiveness: **5/5**, Accessibility: **5/5**, Console/network: **5/5** (0 errors).

## Milestone 4: Inspector Drawer (All Tabs)
- **Iteration 1** (Verified at `375`, `768`, `1440` across `Overview`, `Evidence`, `ML`, `Related`, `Raw JSON`):
  - Hierarchy & scannability: **5/5**, Information density: **5/5**, Consistency: **5/5**, State coverage: **5/5**, Responsiveness: **5/5**, Accessibility: **5/5**, Console/network: **5/5** (0 errors).

## Milestone 5: Graph View (Cytoscape.js) + Timeline / Map View (Leaflet OSM)
- **Iteration 1** (Verified at `375`, `768`, `1440` with recorded run):
  - Hierarchy & scannability: **5/5**, Information density: **5/5**, Consistency: **5/5**, State coverage: **5/5**, Responsiveness: **5/5**, Accessibility: **5/5**, Console/network: **5/5** (0 errors).

## Milestone 6: Audit & Planner View + Report / Export View
- **Iteration 1** (Verified at `375`, `768`, `1440` with recorded run):
  - Hierarchy & scannability: **5/5**, Information density: **5/5**, Consistency: **5/5**, State coverage: **5/5**, Responsiveness: **5/5**, Accessibility: **5/5**, Console/network: **5/5** (0 errors).

## Milestone 7: Data-Depth Extraction Improvements + Fixtures & Tests
- Added `parse_public_profile_html` and expanded `_extract_hydration_fields` in `osint_agent/collectors/profile_analyzer.py` (JSON-LD, `__NEXT_DATA__`, OpenGraph, `joined`, `activity_dates`, `followers`, `banner_url`, `languages`, `orgs`; removed `streetaddress`/`address`).
- Verified with saved fixture (`tests/fixtures/profile_jsonld_nextdata.html`) and offline tests (`tests/test_phase_m7_parsers.py`).

## Milestone 8: Polish Pass, Windowed Rendering (>200 cards) & Final Rubric
- Added windowed card rendering (`visibleLimit = 200` + `+100` increment) in `static/js/results.js` and `docs/UI_TOUR.md`.
- Final full Playwright audit across all 8 views at `375`, `768`, `1440`:

| Criterion | Final Score | Notes |
|---|---:|---|
| Hierarchy & scannability | **5 / 5** | 3-second scan of KPIs, cards, and inspector tabs |
| Information density without clutter | **5 / 5** | Monospace telemetry, progress bars, dot-grid, clean 4px spacing |
| Consistency with tokens & components | **5 / 5** | All views use `tokens.css` & `components.css` (dark + light toggle) |
| State coverage (loading / empty / error / partial) | **5 / 5** | Intentional sparse cards, partial badges, empty timeline/map panels |
| Responsiveness & overflow | **5 / 5** | `scrollWidth == innerWidth` across `375`, `768`, `1440` on every view |
| Accessibility (contrast, focus, labels) | **5 / 5** | WCAG AA contrast, visible focus rings, `/`, `Esc`, arrow-key nav |
| Console errors / failed requests / layout shift | **5 / 5** | 0 console errors, 0 page errors, 0 failed requests |

- **Items marked "stuck"**: None (0 stuck items).
