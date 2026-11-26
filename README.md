# Intelligent OSINT: Agentic Knowledge Graph Constructor

A modular, ML-assisted Open-Source Intelligence (OSINT) knowledge graph constructor. Given a seed entity (`email`, `username`, `domain`, `org`, `place`, `public_figure`, `consenting_person`), free-text prompt, or image (EXIF/GPS), the agent expands across **14 public OSINT collectors**, filters catch-all/soft-404 false positives, deeply scrapes profiles & code repositories, runs **local ML enrichment + clustering**, optionally synthesizes unstructured pages with a **multimodal LLM**, and visualizes the resulting confidence-scored knowledge graph in a real-time web UI.

---

## Architecture & Pipeline

```
Seed Input (Structured / Free-Text / Image EXIF)
   │
   ▼
Consent Gate (Scope.consent_confirmed == True)
   │
   ▼
Priority Queue + Planner (RuleBasedPlanner / LLMPlanner)
   │
   ▼
14 Collector Tools (email_intel, sherlock, maigret, github_code_intel,
                    playstore_intel, breach_intel, exa_search, crtsh,
                    domain_intel, web_archive_crawl, threat_intel,
                    open_data_intel, nominatim, wikidata)
   │
   ├──► False-Positive & Soft-404 Filter (stream_checker.py + false_positive_sites.json)
   │
   ▼
Deep Profile & Repo Analyzer (profile_analyzer.py)
   ├── Platform APIs (GitHub User/Repo/Trees/Manifests, Wattpad, Play Store App & Developer,
   │                  Reddit, Bluesky, Mastodon, Dev.to, HuggingFace, Telegram)
   ├── Generic HTML (JSON-LD, Next.js __NEXT_DATA__, <dt>/<dd>, mailto:/tel:, og:image/images)
   ├── 2nd-Hop Follow-Up (auto-scrapes Play Store apps / LinkedIn / dev links found in repos/bios)
   │
   ├──► Local ML Enrichment & Clustering (profile_matcher.py + embeddings.py)
   │      • TF-IDF keyword extraction + extractive ml_summary + tech skill matching + spaCy NER
   │      • SentenceTransformer (all-MiniLM-L6-v2) + char n-gram + shared-signal AgglomerativeClustering
   │
   └──► Selective Multimodal LLM Enrichment (llm/client.py — OpenRouter / Anthropic / Gemini)
          • Only unstructured hits not already covered by APIs/ML (1 batch max, 429 circuit breaker)
          • Open-ended `additional_facts` merged into `extra_fields`
          • Cluster-guided cross-profile relation synthesis (`synthesize_profile_relations`)
   │
   ▼
Swappable Entity Resolver (BaselineWeightedResolver / LearnedResolver ML)
   │
   ▼
Lead Extractor (extractor.py) ──► Next-Depth Entities ──► NetworkX Graph + Cytoscape UI
```

---

## Implemented Modules & Features

### 1. 14 Collector Tools (`osint_agent/collectors/tools.py`)
| Tool | Target Types | What It Gathers |
|---|---|---|
| `email_intel` | `email` | GitHub commits/users, Gravatar JSON + avatar, Duolingo, OpenPGP/Ubuntu PGP keyservers, Holehe CLI registrations, **XposedOrNot** breaches + risk/password-strength analytics, **Hudson Rock Cavalier** infostealer metadata |
| `sherlock` | `username`, `public_figure`, `consenting_person` | Multi-site username checks with live baseline-probe verification against catch-all 200s |
| `maigret` | `username`, `public_figure`, `consenting_person` | Maigret CLI or native async `stream_checker.py` across 10,000+ site rules |
| `github_code_intel` | `username`, `org` | Public repos, commit author emails, orgs, and deep code/manifest inspection |
| `playstore_intel` | `username`, `public_figure`, `org`, `consenting_person` | Google Play Store developer pages & strict handle-matched app search |
| `breach_intel` | `username`, `public_figure`, `consenting_person` | Hudson Rock Cavalier infostealer compromise metadata (computer, OS, stealer type, date, service count, redacted login/password patterns) |
| `exa_search` | `username`, `email`, `public_figure`, `consenting_person`, `org`, `domain` | Exa AI web search (`/search` via Hack Club proxy or `EXA_API_KEY`) with zero-retry `402`/`429` circuit breakers |
| `crtsh` | `domain` | Certificate Transparency subdomains (crt.sh + HackerTarget fallback) |
| `domain_intel` | `domain` | RDAP registrar/nameservers, DNS A-records, HTTP headers, page metadata |
| `web_archive_crawl` | `domain` | Wayback Machine CDX snapshots, `robots.txt`, `sitemap.xml`, outbound links |
| `threat_intel` | `domain` | URLScan.io public scans + optional VirusTotal, AlienVault OTX, AbuseIPDB |
| `open_data_intel` | `org`, `public_figure`, `place` | Wikipedia summaries, GDELT news, OpenSky Network flights, optional OpenSanctions/AIS |
| `nominatim` | `place` | OpenStreetMap Nominatim geocoding & administrative divisions |
| `wikidata` | `org`, `public_figure`, `consenting_person`, `place` | Wikidata entity search |

### 2. Catch-All & Soft-404 Filtering (`stream_checker.py` + `false_positive_sites.json`)
- **Curated Blocklist (`false_positive_sites.json`)**: Pre-filters 35 confirmed catch-all / `202` / bot-wall hosts (`last.fm`, `www.7cups.com`, `www.memrise.com`, `houzz.com`, `www.gaiaonline.com`, `likee.video`, `www.freelancer.com`, etc.).
- **Live Baseline Probing**: Probes matched sites with `_BASELINE_PROBE_USERNAME` (`zzqwx991827364nonexistent`) and auto-persists newly detected catch-all hosts into `.osint_dead_hosts.json`.
- **Deep Soft-404 / Redirect Guard (`_is_bot_or_soft404_response`)**: Rejects `202 Accepted`, scans 64 KB of HTML across all check types, rejects redirects to `/notfound`, `/dashboard`, `/signup`, `/login`, `/verify-human`, or root `/`, and checks whitespace-normalized empty JSON (`{"users": {}}`).

### 3. Deep Profile & Repository Scraping (`profile_analyzer.py`)
- **GitHub Repos (`_analyze_github_repo`)**: Fetches repo metadata, README, commits, recursive file tree, and prioritizes manifests (`build.gradle`, `build.gradle.kts`, `AndroidManifest.xml`, `package.json`, `pyproject.toml`, `CNAME`) while skipping `.idea/`, `node_modules/`, `venv/`, and `gradle/wrapper/`. Extracts emails, URLs, and Android `applicationId`s (auto-linking to Google Play Store).
- **Play Store (`_parse_playstore_app_html`, `_analyze_playstore`)**: Parses app title, description, icon + screenshots (`images`), developer page link, **About the developer** (legal name, developer email, office address, phone), and **App support** (support email, support phone, website, privacy policy URL), chaining developer pages to their top apps.
- **Universal HTML (`_analyze_generic_html`)**: Strips `<script>`, `<style>`, `<nav>`, `<header>`, `<footer>`, `<aside>` before link extraction, filters CDN/JS/schema/platform-brand noise via `_is_useful_profile_link()`, and extracts JSON-LD, Next.js `__NEXT_DATA__`, `<dl><dt>/<dd>`, `mailto:`, `tel:`, and profile images into structured + open-ended `extra_fields`.
- **2nd-Hop Follow-Up (`agent.py`)**: Automatically promotes newly discovered high-value profile/app URLs (e.g., Play Store apps found in GitHub `build.gradle.kts`) into `PROFILE` hits and feeds discovered emails/websites into `extract_entities()`.

### 4. Local ML Pipeline (`osint_agent/ml/`)
- **Profile Enrichment (`profile_matcher.py`)**:
  - `extract_profile_keywords()`: TF-IDF 1–2 gram keyword extraction per profile.
  - `compress_profile_text_ml()`: Sentence-level TF-IDF + OSINT contact/project signal scoring to strip boilerplate.
  - `enrich_profiles_ml()`: Runs **locally before any LLM call** — generates extractive `ml_summary`, matches tech/domain `skills`, and runs spaCy NER when installed.
  - `rank_profiles_for_llm()`: Skips profiles already structured by APIs/Play Store/GitHub/ML and selects at most 6 unstructured profiles per batch.
  - `discover_ml_profile_relations()`: Multi-signal similarity (`all-MiniLM-L6-v2` embeddings, char n-gram handle similarity, shared link Jaccard, location, email) + `AgglomerativeClustering` assigning `ml_cluster_id` and `ml_similar_profile` edges.
- **Entity Resolution (`resolver.py`, `features.py`, `dataset.py`)**: `BaselineWeightedResolver` and `LearnedResolver` (`LogisticRegression` on 9-feature vectors).

### 5. Resilient LLM Client (`osint_agent/llm/client.py`, `seeds.py`, `planner.py`)
- Supports **OpenRouter / Hack Club proxy (`openrouter/free`)**, **Anthropic**, **Gemini**, and **FakeClient**.
- **Reasoning-Model Support**: Floor `max_tokens = 8192` (auto-bumping to `16384` on `finish_reason="length"`), truncated-JSON auto-repair (`_repair_truncated_json()`), and automatic `400` multimodal → text-only fallback.
- **429 Circuit Breaker**: Trips a shared `90s` cooldown immediately on `429` (`LLMRateLimitError`) with zero retry spam.
- **Free-Text Seed Parsing (`seeds.py`)**: Deterministic `_fallback_parse_seeds()` handles emails, URLs, handle cues, bare usernames, and person/org phrases even without an LLM or `spaCy`.

### 6. Web Dashboard & REST API (`osint_agent/api/server.py`, `static/`)
- **Inputs**: Structured Target, Free-Text Natural Language (`openrouter` supported in UI), and Image Upload (EXIF/GPS + reverse geocoding).
- **Live Controls**: Real-time SSE log stream, **Force Stop** button (`POST /api/investigate/{job_id}/stop` cancelling the active `asyncio.Task`), and JSON graph export.
- **Dynamic Inspector (`static/js/app.js`)**: Renders clickable profile/app image galleries (`avatar_url` + `images`), phone numbers, breach/infostealer metadata, ML clusters, and **all** dynamic keys in `attributes` and `extra_fields`.

---

## Setup & Usage

### Prerequisites & Install
```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

Optional `.env` keys:
```env
OPENROUTER_API_KEY=sk-hc-v1-...   # Used for OpenRouter/HackClub LLM & Exa proxy
LLM_PROVIDER=openrouter
```

### Run the Web Dashboard & API
```bash
uvicorn osint_agent.api.server:app --reload --port 8000
```
Open `http://127.0.0.1:8000` (API docs at `http://127.0.0.1:8000/docs`).

#### Key API Endpoints
- `GET /api/health` — System status and all 14 registered collector tools.
- `POST /api/parse-seeds` — Free-text seed extraction with safety filtering.
- `POST /api/investigate` — Dispatch an investigation (`consent_confirmed=true` required).
- `POST /api/investigate/{job_id}/stop` — Force-stop a running investigation.
- `GET /api/investigate/{job_id}` / `/stream` — Job status and real-time SSE event stream.
- `GET /api/graph/{job_id}` — Cytoscape-formatted graph JSON.
- `POST /api/analyze-image` — EXIF/camera/GPS extraction and reverse geocoding.

### Run via CLI
```bash
python -m osint_agent.cli --kind email --value user@example.com --consent
python -m osint_agent.cli --kind username --value octocat --consent --resolver learned --model-path model.joblib
```

---

## Running the Test Suite

```bash
./.venv/bin/pytest -q
```
All **98 unit & integration tests** are mocked/deterministic and pass with zero live external calls.
