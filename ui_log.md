# UI Redesign Audit Log (`ui-redesign`)

## Milestone 1: Design Tokens + Component Library + App Shell & Navigation
- **Iteration 1**:
  - Defect found at `375px`: `.nav-rail` stretched vertically inside `.app-shell` CSS Grid (`launch_375.png`).
  - Fix: Added `grid-template-rows: auto minmax(0, 1fr); align-content: start;` and `flex-shrink: 0` on `.nav-item` under `@media (max-width: 960px)` in `static/css/app.css`.
- **Iteration 2** (Verified at `375`, `768`, `1440`):
  - Hierarchy & scannability: **5/5**
  - Information density without clutter: **5/5**
  - Consistency with tokens & components: **5/5**
  - State coverage (loading/empty/error/partial): **4/5**
  - Responsiveness & overflow: **5/5** (`scrollWidth == innerWidth` at `375`, `768`, `1440`)
  - Accessibility (contrast, focus, labels): **5/5**
  - Console errors / failed requests / layout shift: **5/5** (0 console errors, 0 page errors, 0 failed requests)
