# Intelligent OSINT: Agentic Knowledge Graph Constructor

A modular, agentic Open-Source Intelligence (OSINT) knowledge graph constructor built for college ML research. The system takes seed information about a target (organization, domain, place, public figure, consenting person, or username) and expands outward across public data sources to construct a confidence-scored knowledge graph.

## Architecture & Core Design

```
Seed Info -> Typed Entities -> Tool Registry -> Async Collectors -> Normalized Hits
                                                                           │
                                                                           ▼
                                                                  Profile Enrichment
                                                             (HTML Metadata, SSRF Guard)
                                                                           │
                                                                           ▼
                                                                Swappable Entity Resolver
                                                          (Baseline Heuristics vs Learned ML)
                                                                           │
                                                                           ▼
Lead Extractor ──► Priority Queue ◄── Confirmed Entities (Decayed Confidence)
 (Emails, Domains, Handles)
       │
       ▼
Graph Nodes & Edges (Deduplicated with Merged Provenance)
```

1. **Seeds & Scope**: Typed entity seeds (`org`, `domain`, `place`, `public_figure`, `consenting_person`, `username`). Refuses to run unless `Scope.consent_confirmed` is explicitly True.
2. **Collector Tool Registry (`tools.py`)**:
   - `crtsh`: Certificate Transparency log parser for subdomains (`domain`).
   - `nominatim`: OpenStreetMap geocoder with polite rate limiting (`place`).
   - `wikidata`: Entity search API for public entities and figures (`org`, `place`, `public_figure`, `consenting_person`).
   - `sherlock`: CLI subprocess executing Sherlock in isolated temporary directories for social profiles (`username`, `public_figure`, `consenting_person`).
   - `maigret`: CLI subprocess executing Maigret for social accounts and profile dossiers (`username`, `public_figure`, `consenting_person`).
   - *Graceful degradation*: If Sherlock or Maigret binaries are missing, the agent logs an audit event (`tool_unavailable`) and continues without crashing.
3. **Profile Enrichment (`enrich.py`)**:
   - Asynchronously parses `og:title`, `og:description`, `meta[name="description"]`, and `<title>`.
   - **SSRF Guard**: Prohibits requests resolving to loopback (`127.0.0.1`, `localhost`, `::1`) or private IPv4/IPv6 ranges (`10.x`, `172.16-31.x`, `192.168.x`, `169.254.x`).
   - **Size Limit**: Enforces a strict 200 KB response cap using streaming bytes.
   - **Polite Crawling**: Respects `robots.txt` per domain and throttles to a 1-second per-domain delay with a global concurrency limit of 5. Enriches at most 15 profiles per username.
4. **Lead & Entity Extraction (`extractor.py`)**:
   - Analyzes confirmed profile bios and text to surface new leads:
     - Emails (`name@domain.com`)
     - Domains (excluding common social platforms and link shorteners)
     - Social handles (`@username`)
     - Organizations & Places (via `spaCy` NER when installed; regex fallback)
   - **Sanitization Guard**: Prohibits phone numbers, street addresses, and sensitive personal identifiers from ever being extracted.
5. **Deduplication & Provenance**:
   - Nodes discovered via multiple expansion paths retain the highest confidence and merge provenance chains without duplication.
6. **Swappable Entity Resolvers (`resolver.py`)**:
   - `BaselineWeightedResolver`: Hand-weighted multi-feature heuristic scoring across active attributes.
   - `LearnedResolver`: Supervised `LogisticRegression` classifier trained on extracted 9-feature vectors.
7. **Dataset & Training (`dataset.py`, `scripts/train_eval.py`)**:
   - Synthetic generator with label noise (~5%), hard negative collisions (same handle and city with divergent bios), and overlapping feature distributions.
   - Leakage check benchmarking in-distribution vs. out-of-distribution performance.
8. **Human-in-the-Loop Annotation (`label_cli.py`)**:
   - Interactive terminal tool to inspect candidate pairs from `graph.json` or JSONL and generate ground truth training datasets.

---

## Installation & Setup

### Prerequisites
- Python 3.11 or 3.12
- Recreate environment using `uv` or `venv`:

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

---

## Usage

### 1. Training & Evaluating the Resolver (Leakage Check)
Train `LearnedResolver` on 500 synthetic pairs and benchmark on both in-distribution and out-of-distribution held-out splits:
```bash
python scripts/train_eval.py --pairs 500
```

### 2. Running the Username Expansion Demo
Run the end-to-end username expansion demo (Sherlock account discovery -> ML resolver gating -> HTML enrichment -> lead entity extraction):
```bash
python scripts/demo_username.py
```

### 3. Running the Agent via CLI
Run with the baseline resolver:
```bash
python -m osint_agent.cli --kind username --value octocat --consent
```

Run with the trained ML resolver:
```bash
python -m osint_agent.cli --kind username --value octocat --consent --resolver learned --model-path model.joblib
```

### 4. Running the REST API Server
Start the FastAPI server with live event streaming and Cytoscape.js graph output:
```bash
uvicorn osint_agent.server:app --reload --port 8000
```
Interactive API docs available at: `http://localhost:8000/docs`

#### Key API Endpoints:
- `GET /api/health`: System health and catalog of available collector tools.
- `POST /api/parse-seeds`: Parse unstructured input into candidate seeds using LLM with regex fallback.
- `POST /api/investigate`: Submit an investigation job (`consent_confirmed=true` required).
- `GET /api/investigate/{job_id}`: Check investigation job status and graph statistics.
- `GET /api/investigate/{job_id}/stream`: Server-Sent Events (SSE) streaming real-time decisions and discovered nodes.
- `GET /api/graph/{job_id}`: Cytoscape.js formatted graph nodes and edges for visualization.
- `GET /api/audit/{job_id}`: Full audit trail for the investigation.

### 5. Running the Planner Comparison Benchmark
Compare `RuleBasedPlanner` vs `LLMPlanner` on representative seeds:
```bash
python scripts/compare_planners.py
```

### 6. Annotating Pairs Interactively
Annotate candidate pairs extracted from a graph run:
```bash
python -m osint_agent.label_cli --input graph.json --output pairs.jsonl
```

---

## Running the Test Suite

Run all 60 unit tests (all mocked, zero external network calls, zero model weight downloads):
```bash
pytest -v
```
