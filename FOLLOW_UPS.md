# Backend & Pipeline Follow-Ups (`FOLLOW_UPS.md`)

Frontend UI/UX safeguards (junk filtering, confidence collapsing, entity decoding, and count labeling) are in place, but the following root causes live in the Python collection/extraction/resolution pipeline and should be addressed in a backend pass:

1. **Junk Role & Name Extraction (`osint_agent/extractors/`, `osint_agent/collectors/`)**
   - Cookie-banner / boilerplate text (`"We use cookies..."`, `"Privacy Policy"`, `"Sign in"`) and platform names (`"Discord"`, `"KICK"`) are occasionally scraped into `name` or `job_title`/`headline` fields.
   - **Recommended fix**: Add a shared HTML boilerplate/nav/banner stripper before regex/LLM field extraction, and reject platform-brand tokens from `name` slots.

2. **Conflicting Single-Valued Entities (`Location`, `DOB`, `Role`)**
   - Unrelated profiles with colliding usernames contribute conflicting locations (e.g. `IN`, `Melbourne`, `Peru`, `San Jose`) at similar confidence levels.
   - **Recommended fix**: Keep `field_confidence` and `field_sources` calibrated per-source in the entity resolver and flag mutually exclusive locations/DOBs in the backend graph summary rather than flattening them.

3. **Count Semantics (`nodes` vs `unique_entities` vs `confirmed`)**
   - Raw graph nodes (e.g. `160`) can contain duplicate normalized URLs (`https://github.com/x` vs `https://www.github.com/x/`), causing raw node count (`160`) to differ from deduplicated entity count (`157`).
   - **Recommended fix**: Canonicalize URLs (`strip www.`, strip trailing `/`, lowercase host) before node insertion in `osint_agent/graph/` so node IDs and counts match 1:1.

4. **Snapshot Telemetry Persistence (`osint_agent/api/server.py`)**
   - Older saved snapshots lack `telemetry.scanBudget` and `telemetry.collectorStatus`, requiring client-side fallback from `audit_events`.
   - **Recommended fix**: Always persist `scanBudget`, `elapsed_seconds`, and `collectorStatus` alongside `nodes`, `edges`, and `audit_events` on job completion.

5. **Role Extraction on Academic Faculty Pages (e.g. JNU)**
   - Target faculty roles such as `"Professor"` on university directory pages are not consistently captured into `job_title` / `headline` when the page layout uses non-standard headings or tables.
   - **Recommended fix**: Add a targeted academic title extractor (`Professor`, `Associate Professor`, `Assistant Professor`) for `.ac.in` / `.edu` faculty pages.
