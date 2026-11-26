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
