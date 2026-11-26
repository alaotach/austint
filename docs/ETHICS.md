# Dataset & Research Ethics (`OSINT-Link`)

This document defines the ethical, legal, and privacy boundaries governing the
**OSINT-Link** dataset and all experiments in *"Learning to Investigate:
Graph-Based Collective Identity Resolution and Budget-Aware Expansion for
Agentic OSINT"*.

## 1. Public Data & Consent Scope
- **Public data only**: All profiles are collected exclusively from publicly
  accessible endpoints and pages (no authentication, no login walls, no
  private messages, no paywalled APIs).
- **Eligible subjects**: Collection is restricted to public figures,
  organizations, open-source maintainers/contributors, and consenting
  individuals.
- **Consent Gate & Audit Trail**: `OSINTAgent` continues to enforce
  `Scope.consent_confirmed == True` and full JSONL audit logging.

## 2. Weak-Supervision Label Provenance
- Positive identity pairs (`label = 1`) are constructed **only** from
  **self-declared** cross-platform links:
  - GitHub `social_accounts` API and user `blog` / homepage fields.
  - Mastodon `fields` with server-verified `verified_at` (`rel="me"`).
  - Personal websites / link-in-bio pages containing explicit `<a rel="me">`
    or bidirectional backlinks to the seed profile.
- We never infer positive labels from unverified name collisions.

## 3. Privacy & Data Minimization in Exported Artifacts
- **No raw bios or raw photos**: Exported dataset files (`train.jsonl`,
  `val.jsonl`, `test.jsonl`, `ood_platform_test.jsonl`) store **only** derived
  numerical/structural features (text embeddings, perceptual image hashes
  `pHash`, CLIP embeddings, link-overlap statistics, and 9 hand-engineered
  similarity features) plus salted SHA-256 `person_id` hashes.
- **No sensitive attributes**: Phone numbers, residential street addresses,
  government IDs, financial records, health/biometric data, and private
  credentials are never collected or exported.
- **Crawling etiquette**: Every HTTP fetch respects `robots.txt`
  (`can_fetch_robots`), enforces SSRF protection (`is_private_or_loopback_host`),
  caps response sizes at 200 KB, and rate-limits per domain (`>= 1.0s`).

## 4. Evaluation Integrity & Limitations
- **Person-level group splits**: `train`, `val`, and `test` are split by
  hashed `person_id` so **no person appears in more than one split**.
- **Out-of-distribution (OOD) split**: Specific platforms (`mastodon`, `devto`)
  are held out from training/validation to measure zero-shot platform transfer.
- **Label noise**: Weak supervision introduces noise (e.g., abandoned links,
  shared org accounts, bots). A manual audit sample of 100 pairs is used to
  estimate and report the empirical label-noise rate.
