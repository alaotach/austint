"""Feature extraction and TargetContext representation for OSINT entity resolution."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from rapidfuzz import fuzz

from osint_agent.embeddings import compute_text_similarity
from osint_agent.schema import Entity, EntityType, Hit

FEATURE_NAMES: list[str] = [
    "name_sim",
    "username_sim",
    "text_sim",
    "loc_match",
    "domain_link",
    "uname_len",
    "token_overlap",
    "has_text",
    "exact_name_match",
]


@dataclass
class TargetContext:
    """Consolidated profile context describing the target during resolution."""

    names: list[str] = field(default_factory=list)
    usernames: list[str] = field(default_factory=list)
    locations: list[str] = field(default_factory=list)
    domains: list[str] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)

    def _add_unique(self, target_list: list[str], items: list[str] | str | None) -> None:
        if not items:
            return
        candidates = [items] if isinstance(items, str) else items
        for item in candidates:
            if not isinstance(item, str):
                continue
            clean = item.strip()
            if clean and clean.lower() not in [x.lower() for x in target_list]:
                target_list.append(clean)

    def absorb(self, item: Entity | Hit | None = None, **kwargs: Any) -> None:
        """Absorb contextual signals from an Entity, a Hit, or explicit keyword values."""
        if isinstance(item, Entity):
            if item.entity_type == EntityType.DOMAIN:
                self._add_unique(self.domains, item.value)
            elif item.entity_type == EntityType.PLACE:
                self._add_unique(self.locations, item.value)
            elif item.entity_type == EntityType.USERNAME:
                self._add_unique(self.usernames, item.value)
            elif item.entity_type in (
                EntityType.ORG,
                EntityType.PUBLIC_FIGURE,
                EntityType.CONSENTING_PERSON,
            ):
                self._add_unique(self.names, item.value)

            # Inspect entity attributes
            attrs = item.attributes or {}
            self._add_unique(self.usernames, attrs.get("username") or attrs.get("usernames"))
            self._add_unique(self.locations, attrs.get("location") or attrs.get("locations"))
            self._add_unique(self.domains, attrs.get("domain") or attrs.get("domains"))
            self._add_unique(
                self.texts,
                attrs.get("text") or attrs.get("description") or attrs.get("bio"),
            )

        elif isinstance(item, Hit):
            if item.discovered_type == EntityType.DOMAIN:
                self._add_unique(self.domains, item.discovered_value)
            elif item.discovered_type == EntityType.PLACE:
                self._add_unique(self.locations, item.discovered_value)
            elif item.discovered_type == EntityType.USERNAME:
                self._add_unique(self.usernames, item.discovered_value)
            elif item.discovered_type in (
                EntityType.ORG,
                EntityType.PUBLIC_FIGURE,
                EntityType.CONSENTING_PERSON,
            ):
                self._add_unique(self.names, item.discovered_value)

            merged_meta = {**item.raw_data, **item.metadata}
            self._add_unique(
                self.usernames, merged_meta.get("username") or merged_meta.get("usernames")
            )
            self._add_unique(
                self.locations, merged_meta.get("location") or merged_meta.get("locations")
            )
            self._add_unique(
                self.domains, merged_meta.get("domain") or merged_meta.get("parent_domain")
            )
            self._add_unique(
                self.texts,
                merged_meta.get("text") or merged_meta.get("description") or merged_meta.get("bio"),
            )

        # Handle explicit kwargs
        if "names" in kwargs:
            self._add_unique(self.names, kwargs["names"])
        if "usernames" in kwargs:
            self._add_unique(self.usernames, kwargs["usernames"])
        if "locations" in kwargs:
            self._add_unique(self.locations, kwargs["locations"])
        if "domains" in kwargs:
            self._add_unique(self.domains, kwargs["domains"])
        if "texts" in kwargs:
            self._add_unique(self.texts, kwargs["texts"])

    @classmethod
    def from_entity(cls, entity: Entity) -> TargetContext:
        """Construct a TargetContext seeded from an existing Entity."""
        ctx = cls()
        ctx.absorb(entity)
        return ctx

    def to_dict(self) -> dict[str, list[str]]:
        """Serialize context to dictionary for persistence in datasets."""
        return {
            "names": list(self.names),
            "usernames": list(self.usernames),
            "locations": list(self.locations),
            "domains": list(self.domains),
            "texts": list(self.texts),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TargetContext:
        """Instantiate TargetContext from dictionary."""
        return cls(
            names=list(data.get("names", [])),
            usernames=list(data.get("usernames", [])),
            locations=list(data.get("locations", [])),
            domains=list(data.get("domains", [])),
            texts=list(data.get("texts", [])),
        )


def _tokenize(text: str) -> set[str]:
    """Tokenize text into lowercase alphanumeric word tokens."""
    return set(re.findall(r"\w+", text.lower()))


def extract_features(ctx: TargetContext, hit: Hit) -> dict[str, float]:
    """Extract ordered numerical feature vector describing candidate hit alignment with context."""
    cand_val = hit.discovered_value.strip()
    merged_data = {**hit.raw_data, **hit.metadata}

    # 1. exact_name_match (0/1)
    exact_match = 1.0 if any(cand_val.lower() == n.lower() for n in ctx.names) else 0.0

    # 2. name_sim (rapidfuzz WRatio / 100.0)
    if ctx.names:
        name_sim = max(fuzz.WRatio(cand_val.lower(), n.lower()) / 100.0 for n in ctx.names)
    else:
        name_sim = 0.0

    # 3. username_sim (rapidfuzz WRatio / 100.0)
    cand_uname = merged_data.get("username") or cand_val
    if ctx.usernames and cand_uname:
        username_sim = max(fuzz.WRatio(cand_uname.lower(), u.lower()) / 100.0 for u in ctx.usernames)
    else:
        username_sim = 0.0

    # 4. text extraction & has_text (0/1)
    cand_text = (
        merged_data.get("description")
        or merged_data.get("bio")
        or merged_data.get("text")
        or ""
    ).strip()
    has_text = 1.0 if len(cand_text) > 0 else 0.0

    # 5. text_sim (embedding cosine similarity)
    if cand_text and ctx.texts:
        text_sim = max(compute_text_similarity(cand_text, t) for t in ctx.texts)
    else:
        text_sim = 0.0

    # 6. loc_match (location similarity)
    cand_loc = (
        cand_val if hit.discovered_type == EntityType.PLACE else (merged_data.get("location") or "")
    ).strip()
    if cand_loc and ctx.locations:
        loc_scores = []
        for loc in ctx.locations:
            if cand_loc.lower() in loc.lower() or loc.lower() in cand_loc.lower():
                loc_scores.append(1.0)
            else:
                loc_scores.append(fuzz.WRatio(cand_loc, loc) / 100.0)
        loc_match = max(loc_scores) if loc_scores else 0.0
    else:
        loc_match = 0.0

    # 7. domain_link (0/1 or partial match)
    cand_domain = (
        cand_val if hit.discovered_type == EntityType.DOMAIN else (merged_data.get("domain") or "")
    ).strip().lower()

    domain_link = 0.0
    if ctx.domains:
        for d in ctx.domains:
            d_clean = d.strip().lower()
            if not d_clean:
                continue
            if cand_domain and (
                cand_domain == d_clean
                or cand_domain.endswith(f".{d_clean}")
                or d_clean.endswith(f".{cand_domain}")
            ):
                domain_link = 1.0
                break
            if cand_text and d_clean in cand_text.lower():
                domain_link = 1.0
                break

    # 8. uname_len (length of candidate identifier)
    uname_len = float(len(cand_uname))

    # 9. token_overlap (Jaccard similarity between word tokens)
    hit_tokens = _tokenize(f"{cand_val} {cand_text}")
    ctx_text_blob = " ".join(ctx.texts + ctx.names + ctx.usernames)
    ctx_tokens = _tokenize(ctx_text_blob)

    if hit_tokens and ctx_tokens:
        union = hit_tokens | ctx_tokens
        intersection = hit_tokens & ctx_tokens
        token_overlap = round(len(intersection) / len(union), 4) if union else 0.0
    else:
        token_overlap = 0.0

    return {
        "name_sim": round(name_sim, 4),
        "username_sim": round(username_sim, 4),
        "text_sim": round(text_sim, 4),
        "loc_match": round(loc_match, 4),
        "domain_link": round(domain_link, 4),
        "uname_len": uname_len,
        "token_overlap": token_overlap,
        "has_text": has_text,
        "exact_name_match": exact_match,
    }
