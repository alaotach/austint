# Design Changes (`DESIGN_CHANGES.md`)

All changes follow [Emil Kowalski's `emil-design-eng` skill](.agents/skills/emil-design-eng/SKILL.md) with custom easing `--ease-out: cubic-bezier(0.23, 1, 0.32, 1)`, `--ease-in-out: cubic-bezier(0.77, 0, 0.175, 1)`, `--ease-drawer: cubic-bezier(0.32, 0.72, 0, 1)`, and `tabular-nums` across all numeric readouts.

## Global & Design System

| Component | Before | After | Why (principle applied) |
| --- | --- | --- | --- |
| Motion tokens | Built-in `ease`, `ease-in`, durations >300ms | `--ease-out: cubic-bezier(0.23, 1, 0.32, 1)`, `--dur-press: 120ms`, `--dur-fast: 160ms`, `--dur-drawer: 280ms` | Never use `ease-in` (feels sluggish); custom ease-out gives immediate response under 300ms |
| Muted text contrast | `--text-muted: #71717a` (~4.0:1 on surface) | `#9494a0` (dark) / `#52525b` (light) | Meets WCAG AA >= 4.5:1 contrast for mono micro-labels in both themes |
| Progress bars | Animated `width` property | GPU-only `transform: scaleX(var(--progress, 0))` with `transform-origin: left` (`220ms var(--ease-out)`) | Only animate `transform` and `opacity`; avoids layout/paint thrashing |
| Tooltips | Native browser `title` attributes | Shared origin-aware `.tooltip-popover` (`140ms var(--ease-out)`, `scale(0.95)` -> `1`) with `data-instant="true"` on subsequent hovers | Origin-aware popovers; skip delay & animation once a tooltip is open |
| Run state machine | Header showed `"Idle"` while Live Scan showed `11/14` | Single `window.OSINT_UI.setRunState()` (`idle`, `running`, `completed`, `stopped`, `partial`, `error`) driving pill, dot, and Stop button | Single source of truth for status across header, sidebar, and Live Scan |
| View wrappers | `#*-container` lacked flex gap, causing cards to touch | `display: flex; flex-direction: column; gap: 20px; min-width: 0` on all view containers | Consistent vertical rhythm and zero horizontal overflow |
| Reduced motion | `transition-duration: 0.01ms !important` killed all fades | Drops `transform` & `animation` movement while keeping `opacity` & `color` transitions | Preserves state visibility without motion sickness |
| Touch & hover | Hover styles fired on touch taps | Hover gated in `@media (hover: hover) and (pointer: fine)`; `min-height: 44px` on `pointer: coarse` | Prevents sticky hover on touch and meets 44x44px touch targets |

## Launch View

| Component | Before | After | Why (principle applied) |
| --- | --- | --- | --- |
| Form structure | Flat ungrouped controls | 4 clear sections: `1. Seed Target`, `2. Budget & Scope`, `3. Engines`, `4. Authorize` | Scannable hierarchy |
| Seed mode tabs | Re-rendered entire Launch view on click (layout jump) | Toggles panels in-place inside `min-height: 118px` container | Zero layout shift on tab switch |
| Start button & gate | Disabled button looked active with no hint | Muted disabled styling + `"Confirm authorization to continue"` helper text (`160ms` color/opacity transition on enable) | Clear disabled state without hiding the mandatory gate |
| Sliders | Raw range input with number only | Live value + one-line scope/cost hint + `:active` thumb `scale(1.22)` (`120ms var(--ease-out)`) | Tactile drag feedback and immediate cost context |
| Snapshot delete | Immediate `DELETE` on click | Soft-delete row hide + 6s Sonner toast with `Undo` button (swipe-to-dismiss) | Forgiving, non-blocking destructive action |

## Live Scan View

| Component | Before | After | Why (principle applied) |
| --- | --- | --- | --- |
| Budget KPIs | Rendered `"undefined / undefined"` on partial snapshots | `safeBudgetPair()` with `"--"` fallbacks and `scaleX` progress bar | Never render literal `"undefined"` |
| Hero & rate metrics | `"Matches 0 (135 hits)"` & `"0.00 answers/s"` on finished runs | Hero row (state, collectors `x/y`, final elapsed, total hits); hides `answers/s` when not running | Accurate finished-run telemetry |
| Collector matrix | Grey dots even when cards said `"done"`, separate from categories | Merged `Collector Matrix & Category Progress` with status normalization + shape cues (`✓`, `●`, `·`, `○`, `–`, `✕`), sorted by status | Color is never the sole status signal; unified component |
| SSE hit stream | Unthrottled re-render | Pause/Resume, tool filter, newest-row `opacity` + `4px translateY` (`150ms var(--ease-out)`), skipped above 5 hits/sec, `"N new hits"` pill when scrolled | Preserves scroll position and prevents animation spam |

## Results & Dossier View

| Component | Before | After | Why (principle applied) |
| --- | --- | --- | --- |
| Filter bar | Scrolled away | `.sticky-filter-bar` with frosted glass + explicit `157 unique (160 nodes)` count | Reconciles deduplicated vs raw node counts |
| Dossier quality | Cookie-banner/platform junk, equal-weight conflicting values | Junk text sanitizer, per-value confidence %, `<45%` collapse, `×` dismiss, `Conflict (N)` badge | Analyst-grade signal-to-noise ratio |
| Citations | Tiny inline `[69]` | Accessible `.cite-link` (`min-width: 20px`) jumping to page/card with highlight pulse | Real hit area and cross-page citation navigation |
| Source cards | Uneven heights, `&amp;` entities, 136+ DOM nodes | `decodeEntities()`, 6-field collapse (`Show all`), equal-height cards, 36/page pagination, `rel="noopener noreferrer"` | Clean typography and 60fps DOM |
| Media gallery | Mixed uncalibrated photos | `Collected Media (Unverified)` with per-image source + confidence badge, defaults to `>=50%` | Never implies a single verified identity |

## Knowledge Graph View

| Component | Before | After | Why (principle applied) |
| --- | --- | --- | --- |
| Layout & fit | Hairball in 1/3 canvas, overlapping labels, no legend | `padding: 48`, initial `360ms` ease-out layout once, LOD labels (`degree >= 3` or `zoom >= 1.15`), type legend | Readable overview and crisp zoom detail |
| Hover & selection | No hover feedback or side detail | `150ms` neighborhood highlight (non-neighbors `opacity: 0.14`), right-hand detail panel, `/` search focus | Preserves spatial context during exploration |

## Timeline & Map View

| Component | Before | After | Why (principle applied) |
| --- | --- | --- | --- |
| Timeline rows | Duplicate URLs/years, misaligned buttons | URL/whitespace normalization, year-grouped sticky headers, left-aligned rows with tooltips | Eliminates duplicates and nested scroll clutter |
| Empty map | Large 260px empty box taking 50% width | Compact 1-line strip when 0 coordinates; full 2-column map only when geo data exists | Gives timeline full width when no coordinates exist |

## Audit & Planner / Report Views

| Component | Before | After | Why (principle applied) |
| --- | --- | --- | --- |
| Audit table | Dozens of repeated rows, raw JSON strings | Consecutive identical rows grouped (`×N`), sticky `<thead>`, tab counts, syntax-colored JSON + `Copy` | Surfaces signal without burying it |
| Report | Touching cards, raw `->` provenance, no print CSS | `20px` vertical rhythm, `seed › … › match` breadcrumbs with tooltip, `@media print` light stylesheet | Clean print/PDF output |

---

## Intentionally Not Animated

1. **Screen navigation (`launch` .. `report`)**: Instant `display` switch (0ms). Users switch views dozens of times per session; transitions add perceived latency.
2. **`/` global search focus & `1-7` keyboard shortcuts**: Zero animation (`0ms`). Keyboard-initiated actions must never feel slower than the keypress.
3. **Dense table & result filter/slider re-renders**: Instant DOM update (`0ms`). Animating 36+ cards or 200 table rows on every slider tick causes frame drops.
4. **Cytoscape re-layouts & filter updates**: `animate: false` after initial load. Only the first graph render animates (`360ms`); subsequent filtering is instant.
5. **Inspector tab switching**: Instant content swap (`0ms`). Rapidly inspecting Evidence -> ML -> Related should feel instantaneous.
