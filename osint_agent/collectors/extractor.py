"""Lead and entity extractor analyzing bio/description text for emails, domains, handles, and orgs."""

from __future__ import annotations

import logging
import re
from typing import Any

from osint_agent.core.schema import (
    SENSITIVE_ATTRIBUTE_PATTERNS,
    Entity,
    EntityType,
)

logger = logging.getLogger(__name__)

# Single log flag for spaCy availability
_SPACY_LOGGED = False
_SPACY_NLP: Any | None = None

# Social network, media, search, CDN, schema, and URL shortener domains to filter out
EXCLUDED_DOMAINS = {
    "twitter.com",
    "x.com",
    "instagram.com",
    "github.com",
    "gitlab.com",
    "linkedin.com",
    "facebook.com",
    "bit.ly",
    "t.co",
    "tinyurl.com",
    "youtu.be",
    "youtube.com",
    "reddit.com",
    "tiktok.com",
    "medium.com",
    "miro.medium.com",
    "google.com",
    "support.google.com",
    "play.google.com",
    "googleusercontent.com",
    "gstatic.com",
    "googleapis.com",
    "wikipedia.org",
    "wikimedia.org",
    "gravatar.com",
    "t.me",
    "discord.gg",
    "pinterest.com",
    "pinimg.com",
    "threads.net",
    "substack.com",
    "patreon.com",
    "schemas.android.com",
    "w3.org",
    "purl.org",
    "xmlns.com",
    "wordpress.org",
    "w.org",
    "tenor.com",
    "epicgames.com",
    "unrealengine.com",
    "artstation.com",
    "geeksforgeeks.org",
    "wattpad.com",
    "duolingo.com",
    "cloudflare.com",
    "jsdelivr.net",
    "unpkg.com",
    "githubusercontent.com",
    "bsky.app",
    "mastodon.world",
}

_NON_DOMAIN_EXTENSIONS = {
    "py", "js", "mjs", "cjs", "ts", "tsx", "jsx", "kt", "kts", "java",
    "gradle", "json", "xml", "yaml", "yml", "toml", "md", "txt", "html",
    "htm", "css", "scss", "png", "jpg", "jpeg", "gif", "svg", "webp",
    "ico", "woff", "woff2", "ttf", "eot", "pdf", "zip", "tar", "gz",
    "apk", "aab", "so", "dll", "exe", "sh", "lock", "log", "csv",
    "sql", "db", "env", "properties", "pro", "plist", "pbxproj",
}

_ANDROID_PACKAGE_RE = re.compile(r"^(?:com|org|net|io|dev|app|in)\.[a-z0-9_]+(?:\.[a-z0-9_]+)+$")

# Regex for forbidden sensitive items (phones, residential street addresses)
PHONE_REGEX = re.compile(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}")
STREET_ADDRESS_REGEX = re.compile(
    r"\b\d{1,5}\s+[A-Za-z0-9\s.,]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Way|Court|Ct|Circle|Cir)\b",
    re.IGNORECASE,
)

EMAIL_REGEX = re.compile(r"\b[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+\b")
DOMAIN_REGEX = re.compile(r"\b(?:https?://)?(?:www\.)?([a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)\b")
HANDLE_REGEX = re.compile(r"(?:^|(?<=\s))@([a-zA-Z0-9_]{3,30})\b")


def get_spacy_nlp() -> Any | None:
    """Safely obtain spaCy en_core_web_sm pipeline or log single warning on absence."""
    global _SPACY_NLP, _SPACY_LOGGED
    if _SPACY_NLP is not None:
        return _SPACY_NLP

    try:
        import spacy  # type: ignore

        _SPACY_NLP = spacy.load("en_core_web_sm")
        return _SPACY_NLP
    except Exception as exc:
        if not _SPACY_LOGGED:
            logger.info("spaCy (en_core_web_sm) not available (%s); falling back to regex extraction.", exc)
            _SPACY_LOGGED = True
        return None


def is_sensitive(candidate: str) -> bool:
    """Verify if string contains phone numbers, street addresses, or sensitive keywords."""
    if PHONE_REGEX.search(candidate):
        return True
    if STREET_ADDRESS_REGEX.search(candidate):
        return True
    if any(pat.search(candidate) for pat in SENSITIVE_ATTRIBUTE_PATTERNS):
        return True
    return False


def extract_entities(text: str) -> list[Entity]:
    """Extract lead entities (emails, domains, handles, organizations, places) while scrubbing sensitive info."""
    clean_text = text.strip()
    if not clean_text:
        return []

    entities: list[Entity] = []
    seen_ids: set[str] = set()

    # 1. Extract Emails
    for match in EMAIL_REGEX.finditer(clean_text):
        email_val = match.group(0).lower().strip()
        if not is_sensitive(email_val):
            ent = Entity.create(
                entity_type=EntityType.EMAIL,
                value=email_val,
                confidence=0.85,
                provenance=["extractor:email"],
            )
            if ent.id not in seen_ids:
                seen_ids.add(ent.id)
                entities.append(ent)

    # 2. Extract Domains (excluding social platforms, CDNs, filenames, and Android package IDs)
    for match in DOMAIN_REGEX.finditer(clean_text):
        full_match = match.group(0).lower()
        raw_domain = match.group(1).lower().strip(".,;:?!'\"")
        # Ensure it has a valid TLD
        if "." not in raw_domain:
            continue
        # Strip trailing path if caught
        domain_val = raw_domain.split("/")[0]
        tld = domain_val.rsplit(".", 1)[-1]
        # Real TLDs are pure ASCII letters (2-15 chars) and not file extensions
        if not tld.isalpha() or not (2 <= len(tld) <= 15) or tld in _NON_DOMAIN_EXTENSIONS:
            continue
        # Reject reverse-DNS Android/Java package names like com.alaotach.app when not an explicit URL
        if not full_match.startswith(("http://", "https://")) and _ANDROID_PACKAGE_RE.match(domain_val):
            continue

        if domain_val in EXCLUDED_DOMAINS or any(domain_val.endswith(f".{x}") for x in EXCLUDED_DOMAINS):
            continue
        if is_sensitive(domain_val):
            continue

        ent = Entity.create(
            entity_type=EntityType.DOMAIN,
            value=domain_val,
            confidence=0.80,
            provenance=["extractor:domain"],
        )
        if ent.id not in seen_ids:
            seen_ids.add(ent.id)
            entities.append(ent)

    # 3. Extract @handles
    for match in HANDLE_REGEX.finditer(clean_text):
        handle_val = match.group(1).strip()
        if not is_sensitive(handle_val):
            ent = Entity.create(
                entity_type=EntityType.USERNAME,
                value=handle_val,
                confidence=0.80,
                provenance=["extractor:handle"],
            )
            if ent.id not in seen_ids:
                seen_ids.add(ent.id)
                entities.append(ent)

    # 4. Extract ORG and GPE (Places) via spaCy when available
    nlp = get_spacy_nlp()
    if nlp is not None:
        try:
            doc = nlp(clean_text)
            for ent in doc.ents:
                ent_text = ent.text.strip()
                if len(ent_text) < 2 or is_sensitive(ent_text):
                    continue

                if ent.label_ == "ORG":
                    cand = Entity.create(
                        entity_type=EntityType.ORG,
                        value=ent_text,
                        confidence=0.75,
                        provenance=["extractor:spacy_org"],
                    )
                    if cand.id not in seen_ids:
                        seen_ids.add(cand.id)
                        entities.append(cand)

                elif ent.label_ == "GPE":
                    cand = Entity.create(
                        entity_type=EntityType.PLACE,
                        value=ent_text,
                        confidence=0.75,
                        provenance=["extractor:spacy_gpe"],
                    )
                    if cand.id not in seen_ids:
                        seen_ids.add(cand.id)
                        entities.append(cand)
        except Exception as err:
            logger.debug("spaCy entity extraction error: %s", err)

    return entities
