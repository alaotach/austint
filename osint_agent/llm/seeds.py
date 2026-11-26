"""LLM-assisted free-text seed entity parsing with Pydantic validation and sensitive filtering."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field

from osint_agent.core.audit import write_audit_event as _log_audit_event
from osint_agent.collectors.extractor import extract_entities, is_sensitive
from osint_agent.llm.client import LLMClient
from osint_agent.core.schema import Entity, EntityType

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an OSINT reconnaissance seed parser.
Your task is to identify and extract target entities from free-text user input.
Return a single JSON object with the following schema:
{
  "target_kind": "org | domain | place | public_figure | consenting_person | username",
  "entities": [
    {
      "type": "org | domain | place | public_figure | consenting_person | username | email",
      "value": "<entity value>"
    }
  ]
}

STRICT SAFETY AND PRIVACY RULES:
1. ONLY return allowed entity types: org, domain, place, public_figure, consenting_person, username, email.
2. NEVER extract or output sensitive attributes such as phone numbers, residential street addresses, government IDs, SSNs, financial records, or medical details.
3. You do NOT have authority to grant or confirm legal target consent.
4. Output valid JSON only, without markdown explanation.
"""


class ParsedEntityItem(BaseModel):
    """Pydantic model validating extracted seed candidate entity."""
    type: str = Field(..., description="Entity type identifier")
    value: str = Field(..., description="Entity value string")


class SeedParseResult(BaseModel):
    """Pydantic model validating LLM seed parser output structure."""
    target_kind: str | None = None
    entities: list[ParsedEntityItem] = Field(default_factory=list)


_URL_SEED_RE = re.compile(r"https?://[^\s<>\"'\]\)]+", re.IGNORECASE)
_HANDLE_CUE_RE = re.compile(
    r"\b(?:username|user|handle|github|ig|instagram|twitter|x|wattpad|linkedin)\s*(?:is|:|=|/|at)?\s*@?([a-zA-Z0-9._-]{2,39})\b",
    re.IGNORECASE,
)


def _fallback_parse_seeds(clean_text: str) -> list[Entity]:
    """Deterministic seed extraction fallback for emails, domains, URLs, usernames, and names."""
    entities = extract_entities(clean_text)
    seen = {e.id for e in entities}

    # 1. Explicit profile URLs (e.g. https://github.com/alaotach, https://play.google.com/...)
    for m in _URL_SEED_RE.finditer(clean_text):
        url_val = m.group(0).rstrip(".,;:?!")
        if not is_sensitive(url_val):
            ent = Entity.create(
                entity_type=EntityType.PROFILE,
                value=url_val,
                confidence=0.9,
                provenance=["extractor:url"],
            )
            if ent.id not in seen:
                seen.add(ent.id)
                entities.append(ent)

    # 2. Explicit username/handle cues (e.g. "username alaotach", "github: alaotach")
    for m in _HANDLE_CUE_RE.finditer(clean_text):
        uname = m.group(1).strip(".,;:?!")
        if uname and not is_sensitive(uname) and uname.lower() not in {"is", "the", "for", "and"}:
            ent = Entity.create(
                entity_type=EntityType.USERNAME,
                value=uname,
                confidence=0.85,
                provenance=["extractor:handle_cue"],
            )
            if ent.id not in seen:
                seen.add(ent.id)
                entities.append(ent)

    if entities:
        return entities

    # 3. Single bare handle (no whitespace, no @, no .) -> USERNAME
    if re.fullmatch(r"[a-zA-Z0-9._-]{2,40}", clean_text) and not is_sensitive(clean_text):
        return [
            Entity.create(
                entity_type=EntityType.USERNAME,
                value=clean_text,
                confidence=0.85,
                provenance=["extractor:bare_username"],
            )
        ]

    # 4. Short natural-language name / org query (e.g. "Aryan Mishra" or "EryzaLabs")
    cleaned_phrase = re.sub(
        r"^(?:investigate|search|find|lookup|look up|check|scan|profile|person)\s+",
        "",
        clean_text,
        flags=re.IGNORECASE,
    ).strip(" .,:;!?")
    if cleaned_phrase and len(cleaned_phrase) <= 80 and not is_sensitive(cleaned_phrase):
        etype = EntityType.USERNAME if " " not in cleaned_phrase and len(cleaned_phrase) <= 30 else EntityType.CONSENTING_PERSON
        return [
            Entity.create(
                entity_type=etype,
                value=cleaned_phrase,
                confidence=0.75,
                provenance=["extractor:freetext_phrase"],
            )
        ]

    return []


def parse_seeds(
    text: str,
    client: LLMClient | None = None,
    audit_path: str | Path | None = None,
    audit_callback: Callable[[str, dict[str, Any]], None] | None = None,
) -> list[Entity]:
    """Parse free-text input into validated seed Entity objects using LLM with regex fallback.

    Enforces strict Pydantic validation, drops unknown entity types, filters out sensitive data,
    and preserves the consent gate (never sets consent_confirmed).
    """
    clean_text = text.strip()
    if not clean_text:
        return []

    # If no LLM client provided, fallback immediately to deterministic extractor
    if client is None:
        logger.debug("No LLM client supplied for seed parsing; using deterministic fallback.")
        return _fallback_parse_seeds(clean_text)

    # Attempt LLM extraction
    try:
        user_prompt = f"Parse the following target description into seed entities:\n\n{clean_text}"
        raw_res = client.complete_json(system=SYSTEM_PROMPT, user=user_prompt, max_tokens=600)
        parsed_res = SeedParseResult.model_validate(raw_res)
    except Exception as exc:
        logger.warning("LLM seed parsing failed (%s); falling back to deterministic extractor.", exc)
        _log_audit_event(
            "llm_seed_parse_fallback",
            {"reason": str(exc), "input_preview": clean_text[:64]},
            audit_path=audit_path,
            callback=audit_callback,
        )
        return _fallback_parse_seeds(clean_text)

    entities: list[Entity] = []
    seen: set[str] = set()

    for item in parsed_res.entities:
        raw_type = item.type.strip().lower()
        raw_val = item.value.strip()

        if not raw_val:
            continue

        # Sensitive type defense: check explicit sensitive type names or values
        if raw_type in {"phone", "phone_number", "address", "street_address", "ssn", "gov_id"} or is_sensitive(raw_val):
            redacted = raw_val[:3] + "***" if len(raw_val) > 3 else "***"
            logger.warning("Dropping sensitive seed candidate: type=%s, value=%s", raw_type, redacted)
            _log_audit_event(
                "sensitive_seed_dropped",
                {"type": raw_type, "value_redacted": redacted, "reason": "sensitive attribute policy"},
                audit_path=audit_path,
                callback=audit_callback,
            )
            continue

        # Type validation: ensure string maps to known EntityType
        try:
            etype = EntityType(raw_type)
        except ValueError:
            logger.debug("Dropping seed entity with unsupported type: %s", raw_type)
            _log_audit_event(
                "unsupported_seed_type_dropped",
                {"type": raw_type, "value": raw_val[:32]},
                audit_path=audit_path,
                callback=audit_callback,
            )
            continue

        entity = Entity.create(entity_type=etype, value=raw_val, confidence=1.0, depth=0)
        if entity.id not in seen:
            seen.add(entity.id)
            entities.append(entity)

    # If LLM returned zero entities, fallback to deterministic extractor
    if not entities:
        logger.debug("LLM returned 0 valid entities; running deterministic fallback.")
        return _fallback_parse_seeds(clean_text)

    _log_audit_event(
        "llm_seeds_parsed",
        {
            "count": len(entities),
            "suggested_target_kind": parsed_res.target_kind,
            "entities": [{"type": e.entity_type.value, "value": e.value} for e in entities],
        },
        audit_path=audit_path,
        callback=audit_callback,
    )
    return entities
