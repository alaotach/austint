"""Dataset utilities for OSINT entity resolution training, evaluation, and synthetic generation."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from osint_agent.ml.features import FEATURE_NAMES, TargetContext, extract_features
from osint_agent.core.schema import EntityType, Hit


@dataclass
class LabeledPair:
    """A pair composed of a target profile context, candidate Hit, label (0/1), and notes."""

    target_context: dict[str, Any]
    hit: dict[str, Any]
    label: int
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LabeledPair:
        return cls(
            target_context=dict(data.get("target_context", {})),
            hit=dict(data.get("hit", {})),
            label=int(data.get("label", 0)),
            notes=str(data.get("notes", "")),
        )


def save_pairs(pairs: list[LabeledPair], path: str | Path) -> None:
    """Serialize a list of LabeledPairs into JSONL format."""
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair.to_dict(), default=str) + "\n")


def load_pairs(path: str | Path) -> list[LabeledPair]:
    """Read LabeledPairs from a JSONL file."""
    pairs: list[LabeledPair] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            clean = line.strip()
            if not clean:
                continue
            pairs.append(LabeledPair.from_dict(json.loads(clean)))
    return pairs


def build_xy(
    pairs: list[LabeledPair],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Convert a sequence of labeled pairs into feature matrix X and label vector y."""
    from osint_agent.ml.embeddings import warm_embedding_cache

    # Collect texts to pre-warm embedding cache in a vectorized pass
    all_texts: list[str] = []
    for pair in pairs:
        all_texts.extend(pair.target_context.get("texts", []))
        hit_meta = {**pair.hit.get("raw_data", {}), **pair.hit.get("metadata", {})}
        t = hit_meta.get("description") or hit_meta.get("bio") or hit_meta.get("text")
        if t:
            all_texts.append(str(t))

    warm_embedding_cache(all_texts)

    X_rows: list[list[float]] = []
    y_vals: list[int] = []

    for pair in pairs:
        ctx = TargetContext.from_dict(pair.target_context)
        hit = Hit.model_validate(pair.hit)
        feat_dict = extract_features(ctx, hit)

        row = [float(feat_dict[col]) for col in FEATURE_NAMES]
        X_rows.append(row)
        y_vals.append(int(pair.label))

    X = np.asarray(X_rows, dtype=np.float32)
    y = np.asarray(y_vals, dtype=np.int64)
    return X, y, list(FEATURE_NAMES)


def generate_synthetic_pairs(
    n: int = 500,
    seed: int = 42,
    ood: bool = False,
    noise_rate: float = 0.0,
) -> list[LabeledPair]:
    """Generate realistic synthetic pairs with hard collisions, overlapping distributions, and label noise."""
    rng = random.Random(seed)
    pairs: list[LabeledPair] = []

    if not ood:
        names_pool = [
            ("Sarah Connor", "Cyberdyne Systems Security Lead", "Los Angeles"),
            ("Alexander Wright", "Distributed Systems Engineer and Cloud Architect", "Seattle"),
            ("Elena Rostova", "Bioinformatics Researcher and Genomic Data Specialist", "Zurich"),
            ("Marcus Vance", "Fintech Quant Developer and Algorithmic Trader", "New York"),
            ("Priya Sharma", "AI Ethics Researcher and NLP Specialist", "London"),
            ("Liam O'Connor", "Renewable Energy Infrastructure Engineer", "Dublin"),
            ("Chen Wei", "Semiconductor VLSI Designer and Hardware Hacker", "Taipei"),
            ("Amira Mansour", "Civil Liberties Attorney and Digital Rights Advocate", "Berlin"),
        ]
        usernames_pool = [
            "vance_quant", "cyber_sarah", "alex_dev99", "elena_dna",
            "priya_nlp", "liam_green", "wei_silicon", "amira_lex",
            "pixel_hunter", "shadow_root", "code_ninja42", "data_wolf"
        ]
        domains_pool = [
            ("cyberdyne.org", "Cyberdyne Systems", "Autonomous robotics and artificial intelligence"),
            ("vancefintech.io", "Vance Financial Systems", "Low-latency order routing and quantitative finance"),
            ("helios-energy.ie", "Helios Clean Energy", "Offshore wind power and battery grid storage"),
            ("gene-syn.ch", "GeneSyn Laboratories", "Next generation sequencing and structural biology"),
        ]
        unrelated_bios = [
            "Artisan baker passionate about sourdough fermentation and local grains.",
            "Vintage motorcycle restoration enthusiast and mechanic based in Melbourne.",
            "Middle school history teacher and historical miniature painter.",
            "Dog agility trainer and certified canine behavioralist.",
            "Landscape watercolor artist exhibiting across regional galleries.",
            "Competitive fly fisherman and outdoor survival gear reviewer.",
        ]
        cities_pool = [
            "Los Angeles", "Seattle", "Zurich", "New York", "London", "Dublin", "Taipei", "Berlin",
            "Omaha", "Calgary", "Perth", "Kyoto", "Mendoza", "Oslo", "Reykjavik", "Tbilisi"
        ]
    else:
        # Out-of-Distribution evaluation pool: distinct templates, domains, and regions
        names_pool = [
            ("Kavita Nair", "Quantum Computing Researcher and Cryogenic Hardware Physicist", "Bengaluru"),
            ("Lucas Moreau", "Autonomous Maritime Navigation and Vessel Routing Specialist", "Marseille"),
            ("Tariq Al-Mansoor", "Desalination Infrastructure Architect and Smart Water Systems", "Abu Dhabi"),
            ("Sven Lindqvist", "Arctic Climate Observation and Cryosphere Remote Sensing", "Stockholm"),
            ("Mateo Rossi", "AgTech Drone Imagery and Precision Viticulture Engineer", "Florence"),
        ]
        usernames_pool = [
            "kavita_quant", "lucas_sea", "tariq_hydro", "sven_ice", "mateo_vine",
            "blue_orbit", "terra_probe", "nano_wave", "quantum_pulse"
        ]
        domains_pool = [
            ("atlantis-ocean.fr", "Atlantis Marine Dynamics", "Autonomous ocean gliders and telemetry"),
            ("nordic-cryo.se", "Nordic Cryo Observation", "High-latitude radar and climate sensors"),
            ("oasis-desal.ae", "Oasis Water Engineering", "Solar membrane distillation and resource tech"),
        ]
        unrelated_bios = [
            "Ceramics instructor specializing in wood-fired stoneware and raku pottery.",
            "Independent bookstore manager and 20th-century poetry collector.",
            "Jazz trumpeter and modular synthesizer acoustic designer.",
            "Alpine mountaineering guide and avalanche safety instructor.",
        ]
        cities_pool = [
            "Bengaluru", "Marseille", "Abu Dhabi", "Stockholm", "Florence",
            "Zagreb", "Valparaiso", "Wellington", "Helsinki", "Bergen"
        ]

    n_pos = n // 2
    n_neg = n - n_pos

    # 1. Generate realistic positive pairs (with varied text lengths and missing bios)
    for i in range(n_pos):
        scenario = i % 4
        if scenario == 0:
            # Person / Org identity match
            base_name, base_bio, base_city = rng.choice(names_pool)
            
            # Vary bio availability: 25% empty/missing bio, 25% short snippet, 50% full bio
            bio_rand = rng.random()
            if bio_rand < 0.25:
                cand_bio = ""
            elif bio_rand < 0.50:
                cand_bio = base_bio.split()[0]  # Very short snippet
            else:
                cand_bio = f"{base_bio} at Institute"

            ctx = TargetContext(
                names=[base_name],
                locations=[base_city],
                texts=[base_bio] if cand_bio else [],
                domains=[f"{base_name.lower().replace(' ', '')}.me"],
            )

            cand_name = base_name if rng.random() < 0.6 else f"{base_name} PhD"
            hit = Hit(
                target_entity_id=f"org:{base_name.lower().replace(' ', '')}",
                discovered_type=EntityType.PUBLIC_FIGURE,
                discovered_value=cand_name,
                relation="same_person",
                source_tool="wikidata",
                is_trusted=False,
                raw_data={"description": cand_bio} if cand_bio else {},
                metadata={"location": base_city, "name": cand_name, "bio": cand_bio},
            )
            pairs.append(
                LabeledPair(
                    target_context=ctx.to_dict(),
                    hit=hit.model_dump(),
                    label=1,
                    notes="[synthetic] [not for the report] positive: person name match with variable bio",
                )
            )

        elif scenario == 1:
            # Domain / Subdomain expansion
            domain, company, desc = rng.choice(domains_pool)
            has_desc = rng.random() > 0.35  # Sometimes subdomains have no bio
            ctx = TargetContext(
                names=[company],
                domains=[domain],
                texts=[desc] if has_desc else [],
            )
            sub = rng.choice(["api", "portal", "cloud", "auth", "dev", "status"])
            cand_domain = f"{sub}.{domain}"
            hit = Hit(
                target_entity_id=f"domain:{domain}",
                discovered_type=EntityType.DOMAIN,
                discovered_value=cand_domain,
                relation="subdomain_of",
                source_tool="crtsh",
                is_trusted=True,
                raw_data={"issuer_name": "DigiCert Global Root CA"},
                metadata={"domain": cand_domain, "parent_domain": domain},
            )
            pairs.append(
                LabeledPair(
                    target_context=ctx.to_dict(),
                    hit=hit.model_dump(),
                    label=1,
                    notes="[synthetic] [not for the report] positive: verified subdomain with domain link",
                )
            )

        elif scenario == 2:
            # Username match: 50% with bio alignment, 50% from bare username seeds
            uname = rng.choice(usernames_pool)
            base_name, base_bio, base_city = rng.choice(names_pool)
            bare_seed = rng.random() < 0.50

            if bare_seed:
                ctx = TargetContext(usernames=[uname])
                bio_text = f"Developer and open source enthusiast: {uname}"
            else:
                ctx = TargetContext(
                    names=[base_name],
                    usernames=[uname],
                    locations=[base_city],
                    texts=[base_bio],
                )
                short_bio = rng.random() < 0.30
                bio_text = base_bio.split(" ")[0] if short_bio else f"Tech enthusiast: {base_bio}"

            hit = Hit(
                target_entity_id=f"person:{uname}",
                discovered_type=EntityType.PROFILE,
                discovered_value=f"https://github.com/{uname}",
                relation="account_on",
                source_tool="sherlock",
                is_trusted=False,
                raw_data={"username": uname, "description": bio_text},
                metadata={"username": uname, "location": base_city if not bare_seed else "", "name": base_name if not bare_seed else uname},
            )
            pairs.append(
                LabeledPair(
                    target_context=ctx.to_dict(),
                    hit=hit.model_dump(),
                    label=1,
                    notes="[synthetic] [not for the report] positive: matching handle with bio overlap",
                )
            )

        else:
            # Organization entity match
            domain, company, desc = rng.choice(domains_pool)
            has_text = rng.random() > 0.20
            ctx = TargetContext(
                names=[company],
                domains=[domain],
                texts=[desc] if has_text else [],
            )
            hit = Hit(
                target_entity_id=f"org:{company.lower().replace(' ', '')}",
                discovered_type=EntityType.ORG,
                discovered_value=company,
                relation="wikidata_reference",
                source_tool="wikidata",
                is_trusted=False,
                raw_data={"description": desc if has_text else ""},
                metadata={"domain": domain, "name": company},
            )
            pairs.append(
                LabeledPair(
                    target_context=ctx.to_dict(),
                    hit=hit.model_dump(),
                    label=1,
                    notes="[synthetic] [not for the report] positive: organization record match",
                )
            )

    # 2. Generate challenging hard negatives (overlapping locations, handles, and partial names)
    for i in range(n_neg):
        scenario = i % 4
        if scenario == 0:
            # Hard Negative: Same common username AND SAME CITY, but divergent profession/bio
            uname = rng.choice(usernames_pool)
            base_name, base_bio, base_city = rng.choice(names_pool)
            diff_bio = rng.choice(unrelated_bios)
            bare_neg = rng.random() < 0.30

            if bare_neg:
                # Target is username, but candidate handle is completely different
                diff_uname = rng.choice(usernames_pool)
                while diff_uname == uname:
                    diff_uname = rng.choice(usernames_pool)
                ctx = TargetContext(usernames=[uname])
                cand_uname = diff_uname
            else:
                ctx = TargetContext(
                    names=[base_name],
                    usernames=[uname],
                    locations=[base_city],
                    texts=[base_bio],
                )
                cand_uname = uname

            hit = Hit(
                target_entity_id=f"person:{cand_uname}",
                discovered_type=EntityType.PROFILE,
                discovered_value=f"https://someplatform.com/{cand_uname}",
                relation="account_on",
                source_tool="sherlock",
                is_trusted=False,
                raw_data={"username": cand_uname, "description": diff_bio},
                metadata={"username": cand_uname, "location": base_city},  # Shared city collision!
            )
            pairs.append(
                LabeledPair(
                    target_context=ctx.to_dict(),
                    hit=hit.model_dump(),
                    label=0,
                    notes="[synthetic] [not for the report] hard negative: handle collision sharing same city",
                )
            )

        elif scenario == 1:
            # Hard Negative: Same name, divergent city/profession
            base_name, base_bio, base_city = rng.choice(names_pool)
            diff_bio = rng.choice(unrelated_bios)
            diff_city = rng.choice(cities_pool)

            has_bio = rng.random() > 0.25
            ctx = TargetContext(
                names=[base_name],
                locations=[base_city],
                texts=[base_bio],
            )
            hit = Hit(
                target_entity_id=f"person:{base_name.lower().replace(' ', '')}",
                discovered_type=EntityType.PUBLIC_FIGURE,
                discovered_value=base_name,
                relation="candidate_name",
                source_tool="wikidata",
                is_trusted=False,
                raw_data={"description": diff_bio if has_bio else ""},
                metadata={"location": diff_city, "name": base_name},
            )
            pairs.append(
                LabeledPair(
                    target_context=ctx.to_dict(),
                    hit=hit.model_dump(),
                    label=0,
                    notes="[synthetic] [not for the report] hard negative: identical name homonym in different city",
                )
            )

        elif scenario == 2:
            # Hard Negative: Lookalike domain with unrelated org and text
            domain, company, desc = rng.choice(domains_pool)
            fake_tld = rng.choice([".biz", ".info", "-review.com", "-scam.org"])
            spoofed_domain = domain.split(".")[0] + fake_tld
            diff_bio = rng.choice(unrelated_bios)
            has_text = rng.random() > 0.30

            ctx = TargetContext(
                names=[company],
                domains=[domain],
                texts=[desc],
            )
            hit = Hit(
                target_entity_id=f"domain:{domain}",
                discovered_type=EntityType.DOMAIN,
                discovered_value=spoofed_domain,
                relation="unrelated_web",
                source_tool="crawler",
                is_trusted=False,
                raw_data={"description": diff_bio if has_text else ""},
                metadata={"domain": spoofed_domain},
            )
            pairs.append(
                LabeledPair(
                    target_context=ctx.to_dict(),
                    hit=hit.model_dump(),
                    label=0,
                    notes="[synthetic] [not for the report] hard negative: spoofed domain with zero entity affiliation",
                )
            )

        else:
            # Hard Negative: Distinct individual in same city with partial token overlap
            base_name, base_bio, base_city = rng.choice(names_pool)
            diff_name, diff_bio, _ = rng.choice(names_pool)
            while diff_name == base_name:
                diff_name, diff_bio, _ = rng.choice(names_pool)

            ctx = TargetContext(
                names=[base_name],
                locations=[base_city],
                texts=[base_bio],
            )
            hit = Hit(
                target_entity_id=f"person:{diff_name.lower().replace(' ', '')}",
                discovered_type=EntityType.PUBLIC_FIGURE,
                discovered_value=diff_name,
                relation="nearby_candidate",
                source_tool="wikidata",
                is_trusted=False,
                raw_data={"description": diff_bio},
                metadata={"location": base_city, "name": diff_name},
            )
            pairs.append(
                LabeledPair(
                    target_context=ctx.to_dict(),
                    hit=hit.model_dump(),
                    label=0,
                    notes="[synthetic] [not for the report] hard negative: distinct individual in same city",
                )
            )

    # 3. Add realistic label noise (~5% of labels flipped)
    for p in pairs:
        if rng.random() < noise_rate:
            p.label = 1 - p.label
            p.notes += " (label_noise_flipped)"

    rng.shuffle(pairs)
    return pairs

    rng.shuffle(pairs)
    return pairs
