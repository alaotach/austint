"""Schema definitions for Intelligent OSINT entities, hits, and scopes."""

from __future__ import annotations

import re
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field, field_validator


class EntityType(str, Enum):
    """Permitted target and discovered entity kinds."""
    ORG = "org"
    DOMAIN = "domain"
    PLACE = "place"
    PUBLIC_FIGURE = "public_figure"
    CONSENTING_PERSON = "consenting_person"
    USERNAME = "username"
    PROFILE = "profile"
    EMAIL = "email"


SENSITIVE_ATTRIBUTE_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"home_?address",
        r"residential_?address",
        r"street_?address",
        r"street",
        r"house_?number",
        r"postal_?address",
        r"home_?phone",
        r"private_?phone",
        r"telephone",
        r"mobile_?phone",
        r"cell_?phone",
        r"family",
        r"spouse",
        r"child(ren)?",
        r"relative",
        r"health",
        r"medical",
        r"diagnosis",
        r"ssn",
        r"social_?security",
        r"passport",
        r"credit_?card",
        r"financial_?record",
        r"personal_?phone",
        r"religion",
        r"sexual_?orientation",
        r"biometric",
    ]
]


def sanitize_attributes(attrs: dict[str, Any]) -> dict[str, Any]:
    """Strip any sensitive attributes strictly violating OSINT ethics rules."""
    cleaned: dict[str, Any] = {}
    for key, value in attrs.items():
        if any(pat.search(key) for pat in SENSITIVE_ATTRIBUTE_PATTERNS):
            continue
        if isinstance(value, dict):
            cleaned[key] = sanitize_attributes(value)
        else:
            cleaned[key] = value
    return cleaned


class Scope(BaseModel):
    """Investigation scope and operational boundary constraints."""
    target_kind: EntityType
    target_value: str
    consent_confirmed: bool = False
    max_depth: int = 2
    max_tool_calls: int = 15
    confidence_decay: float = 0.8
    min_confidence: float = 0.2
    accept_threshold: float = 0.5

    @field_validator("target_value")
    @classmethod
    def validate_target_value(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Target value must not be empty.")
        return stripped

    def ensure_consent(self) -> None:
        """Enforce strict consent confirmation."""
        if not self.consent_confirmed:
            raise PermissionError(
                f"Agent refused to run: Scope.consent_confirmed must be True for {self.target_kind.value} target '{self.target_value}'."
            )


class Hit(BaseModel):
    """A normalized candidate discovery returned by a collector tool."""
    target_entity_id: str
    discovered_type: EntityType
    discovered_value: str
    relation: str
    source_tool: str
    is_trusted: bool = False
    raw_data: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("raw_data", "metadata")
    @classmethod
    def sanitize_hit_data(cls, v: dict[str, Any]) -> dict[str, Any]:
        return sanitize_attributes(v)

    @field_validator("discovered_value")
    @classmethod
    def validate_discovered_value(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Discovered value cannot be empty.")
        return stripped


class Entity(BaseModel):
    """A verified or seed entity within the knowledge graph."""
    id: str
    entity_type: EntityType
    value: str
    confidence: float = 1.0
    depth: int = 0
    attributes: dict[str, Any] = Field(default_factory=dict)
    provenance: list[str] = Field(default_factory=list)

    @field_validator("attributes")
    @classmethod
    def sanitize_entity_attributes(cls, v: dict[str, Any]) -> dict[str, Any]:
        return sanitize_attributes(v)

    @classmethod
    def make_id(cls, entity_type: EntityType, value: str) -> str:
        """Create a deterministic unique entity ID."""
        clean_val = value.strip().lower()
        return f"{entity_type.value}:{clean_val}"

    @classmethod
    def create(
        cls,
        entity_type: EntityType,
        value: str,
        confidence: float = 1.0,
        depth: int = 0,
        attributes: dict[str, Any] | None = None,
        provenance: list[str] | None = None,
    ) -> Entity:
        """Factory constructor ensuring ID generation and sanitization."""
        clean_val = value.strip()
        entity_id = cls.make_id(entity_type, clean_val)
        return cls(
            id=entity_id,
            entity_type=entity_type,
            value=clean_val,
            confidence=max(0.0, min(1.0, confidence)),
            depth=depth,
            attributes=attributes or {},
            provenance=provenance or [f"seed:{entity_id}"],
        )
