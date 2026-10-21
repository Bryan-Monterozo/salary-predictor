from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
import unicodedata
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from difflib import SequenceMatcher
from pathlib import Path

from salary_predictor.contracts import ContractError, UserProfile

from .psoc_workbook import load_psoc_groups

SOURCE_KINDS = frozenset({"occupation", "skill", "region", "education"})
ISCO_BROWSE_GROUPS = {
    "0": "Armed forces", "1": "Managers", "2": "Professionals",
    "3": "Technicians", "4": "Clerical support", "5": "Services and sales",
    "6": "Agriculture and fisheries", "7": "Craft and trades",
    "8": "Plant operators and drivers", "9": "Elementary occupations",
}


def key(value: str) -> str:
    return " ".join(
        "".join(
            c for c in unicodedata.normalize("NFKC", value).casefold()
            if not unicodedata.combining(c)
        ).split()
    )


@dataclass(frozen=True)
class Entry:
    id: str
    label: str
    source: str
    kind: str
    aliases: tuple[str, ...] = ()
    parent_code: str | None = None
    browse_group: str | None = None


@dataclass(frozen=True)
class Candidate:
    id: str
    label: str
    source: str
    score: float
    evidence: str
    parent_code: str | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id, "label": self.label, "source": self.source,
            "score": self.score, "evidence": self.evidence,
            "parent_code": self.parent_code,
        }


@dataclass(frozen=True)
class Match:
    raw_input: str
    status: str
    chosen: Candidate | None
    alternatives: tuple[Candidate, ...] = ()

    def to_dict(self) -> dict:
        return {
            "raw_input": self.raw_input,
            "status": self.status,
            "chosen": self.chosen.to_dict() if self.chosen else None,
            "alternatives": [candidate.to_dict() for candidate in self.alternatives],
        }


@dataclass(frozen=True)
class Experience:
    raw_input: str
    status: str
    years: Decimal | None
    explanation: str

    def to_dict(self) -> dict:
        return {
            "raw_input": self.raw_input,
            "status": self.status,
            "years": str(self.years) if self.years is not None else None,
            "explanation": self.explanation,
        }


def normalize_experience(value: str) -> Experience:
    if not isinstance(value, str):
        raise ContractError("experience", "expected text")
    phrase = key(value)
    if not phrase or phrase in {"unknown", "not sure", "n/a"}:
        return Experience(value, "unknown", None, "Experience not provided")
    match = re.fullmatch(r"(\d+(?:\.\d{1,2})?)\s*(years?|yrs?|months?|mos?)?", phrase)
    if match is None:
        return Experience(value, "ambiguous", None, "Needs manual review")
    try:
        years = Decimal(match.group(1))
        if match.group(2) and match.group(2).startswith(("month", "mo")):
            years = (years / 12).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    except InvalidOperation:
        return Experience(value, "ambiguous", None, "Invalid duration")
    if not 0 <= years <= 70:
        return Experience(value, "ambiguous", None, "Outside 0–70 years")
    return Experience(value, "matched", years, "Explicit duration, rounded to two decimals")


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != 1:
        raise ValueError(f"Unsupported taxonomy file version: {path.name}")
    return value


def _esco_rows(path: Path) -> dict[str, dict]:
    csv.field_size_limit(sys.maxsize)
    unique: dict[str, dict] = {}
    with path.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            uri = row.get("conceptUri")
            if not uri:
                raise ValueError(f"ESCO concept missing URI: {path.name}")
            previous = unique.get(uri)
            if previous is not None:
                fields = set(row) - {"modifiedDate"}
                if any(row[field] != previous[field] for field in fields):
                    raise ValueError(f"Conflicting ESCO concept URI: {uri}")
                if row.get("modifiedDate", "") <= previous.get("modifiedDate", ""):
                    continue
            unique[uri] = row
    return unique


class Taxonomy:
    def __init__(self, root: Path) -> None:
        self.root = root
        config = root / "configs" / "taxonomy"
        self.regions = _load_json(config / "psgc_regions_v1.json")
        self.psoc = _load_json(config / "psoc_reviewed_v1.json")
        self.education = _load_json(config / "education_v1.json")
        self.crosswalk = _load_json(config / "esco_psoc_crosswalk_v1.json")

        inventory = _load_json(root / "configs" / "local_raw_file_manifest.json")
        psoc_path = "data/raw/psoc/2022-psoc.xlsx"
        source_file = next(
            (item for item in inventory["files"] if item["path"] == psoc_path), None
        )
        if source_file is None:
            raise ValueError("PSOC workbook is not in the reviewed source manifest")
        self.psoc_groups = load_psoc_groups(root / psoc_path, source_file["sha256"])
        self._psoc_by_code = {group.code: group for group in self.psoc_groups}

        self._entries: dict[str, list[Entry]] = {name: [] for name in SOURCE_KINDS}
        self._add_reference_rows()
        raw = root / "data" / "raw" / "esco"
        manifest = _load_json(root / "configs" / "local_raw_file_manifest.json")
        expected = {Path(item["path"]).name: item["sha256"] for item in manifest["files"]}
        for filename in ("occupations_en.csv", "skills_en.csv", "occupationSkillRelations_en.csv"):
            with (raw / filename).open("rb") as file:
                checksum = hashlib.file_digest(file, "sha256").hexdigest()
            if expected.get(filename) != checksum:
                raise ValueError(f"Changed ESCO source: {filename}")
        if expected["occupations_en.csv"] != self.crosswalk["source_esco_sha256"]:
            raise ValueError("Crosswalk was reviewed against a different ESCO file")
        self._occupation_rows = _esco_rows(raw / "occupations_en.csv")
        self._skill_rows = _esco_rows(raw / "skills_en.csv")
        self._add_esco_rows()
        self._related_skills: dict[str, list[dict]] | None = None

        self._by_id: dict[str, dict[tuple[str, str], Entry]] = {}
        self._by_name: dict[str, dict[str, dict[tuple[str, str], tuple[Entry, str]]]] = {}
        for kind, entries in self._entries.items():
            by_id: dict[tuple[str, str], Entry] = {}
            by_name: dict[str, dict[tuple[str, str], tuple[Entry, str]]] = {}
            for entry in entries:
                id_key = (entry.source, entry.id)
                if id_key in by_id:
                    raise ValueError(f"Duplicate {kind} identifier: {id_key}")
                by_id[id_key] = entry
                for label, evidence in [(entry.label, "official_or_source_label")] + [
                    (alias, "source_or_reviewed_alias") for alias in entry.aliases
                ]:
                    label_key = key(label)
                    if label_key:
                        matches = by_name.setdefault(label_key, {})
                        if id_key not in matches or evidence == "official_or_source_label":
                            matches[id_key] = (entry, evidence)
            self._by_id[kind] = by_id
            self._by_name[kind] = by_name

    def _add_reference_rows(self) -> None:
        if len(self.regions["regions"]) != 18:
            raise ValueError("Expected 18 dated PSGC regions")
        for row in self.regions["regions"]:
            self._entries["region"].append(Entry(
                id=row["code"], label=row["label"], source="psgc_2025_07", kind="region",
                aliases=tuple(row["aliases"]),
            ))
        reviewed_aliases = {row["code"]: row["aliases"] for row in self.psoc["codes"]}
        for group in self.psoc_groups:
            if group.level != "unit":
                continue
            self._entries["occupation"].append(Entry(
                id=group.code,
                label=group.label,
                source="psoc_2022_workbook",
                kind="occupation",
                aliases=tuple(reviewed_aliases.get(group.code, ())),
                parent_code=group.parent_code,
                browse_group=group.major_label,
            ))
        for row in self.education["levels"]:
            self._entries["education"].append(Entry(
                id=row["id"], label=row["label"], source="education_ux_v1",
                kind="education", aliases=tuple(row["aliases"]),
            ))

    def _add_esco_rows(self) -> None:
        for row in self._occupation_rows.values():
            isco = row["iscoGroup"]
            self._entries["occupation"].append(Entry(
                id=row["conceptUri"], label=row["preferredLabel"], source="esco_local",
                kind="occupation",
                aliases=tuple(name.strip() for name in row["altLabels"].splitlines() if name.strip()),
                parent_code=isco or None,
                browse_group=ISCO_BROWSE_GROUPS.get(isco[:1], "Other"),
            ))
        for row in self._skill_rows.values():
            self._entries["skill"].append(Entry(
                id=row["conceptUri"], label=row["preferredLabel"], source="esco_local",
                kind="skill",
                aliases=tuple(name.strip() for name in row["altLabels"].splitlines() if name.strip()),
            ))

    def browse(self, kind: str, query: str = "", *, limit: int = 20) -> list[dict]:
        if kind not in SOURCE_KINDS:
            raise ValueError("Unsupported lookup kind")
        if not 1 <= limit <= 100:
            raise ValueError("Limit must be between 1 and 100")
        phrase = key(query)
        matches = []
        for entry in self._entries[kind]:
            if phrase and phrase not in key(entry.label) and not any(
                phrase in key(alias) for alias in entry.aliases
            ):
                continue
            matches.append(entry)
        matches.sort(key=lambda e: (
            0 if key(e.label) == phrase else 1 if key(e.label).startswith(phrase) else 2,
            key(e.label), e.source, e.id
        ))
        return [
            {
                "id": e.id, "label": e.label, "source": e.source,
                "kind": e.kind, "parent_code": e.parent_code,
                "browse_group": e.browse_group,
            }
            for e in matches[:limit]
        ]

    def resolve(self, kind: str, raw_input: str, *, limit: int = 5) -> Match:
        if kind not in SOURCE_KINDS:
            raise ValueError("Unsupported lookup kind")
        if not isinstance(raw_input, str):
            raise ValueError("Expected lookup text")
        if not 1 <= limit <= 20:
            raise ValueError("Limit must be between 1 and 20")
        normalized = key(raw_input)
        if not normalized:
            return Match(raw_input, "unknown", None)
        name_hits = self._by_name[kind].get(normalized, {})
        id_hits = {
            item: (entry, "exact_identifier")
            for item, entry in self._by_id[kind].items()
            if key(entry.id) == normalized
        }
        hits = id_hits or name_hits
        if hits:
            choices = sorted(
                (Candidate(e.id, e.label, e.source, 1.0, reason, e.parent_code)
                 for e, reason in hits.values()),
                key=lambda item: (item.source, item.label, item.id),
            )
            if len(choices) == 1:
                return Match(raw_input, "matched", choices[0])
            return Match(raw_input, "ambiguous", None, tuple(choices[:limit]))
        approximate = []
        for label_key, labeled in self._by_name[kind].items():
            if abs(len(label_key) - len(normalized)) > 5:
                continue
            score = SequenceMatcher(None, normalized, label_key).ratio()
            if score < 0.88:
                continue
            for entry, _ in labeled.values():
                approximate.append(Candidate(
                    entry.id, entry.label, entry.source, round(score, 3),
                    "possible_spelling_only", entry.parent_code,
                ))
        best: dict[tuple[str, str], Candidate] = {}
        for item in approximate:
            identity = (item.source, item.id)
            if identity not in best or item.score > best[identity].score:
                best[identity] = item
        choices = sorted(best.values(), key=lambda c: (-c.score, c.label, c.id))
        return Match(
            raw_input, "suggested" if choices else "unknown", None, tuple(choices[:limit])
        )

    def psoc_hierarchy(self, code: str) -> list[dict]:
        group = self._psoc_by_code.get(code)
        if group is None:
            raise ValueError("Unknown PSOC classification code")
        major = group.major_group
        sequence = [{
            "code": major,
            "label": group.major_label,
            "level": "major",
            "parent_code": None,
            "source": "psoc_2022_workbook",
        }]
        for length in (2, 3, 4):
            if len(code) < length:
                break
            part = self._psoc_by_code[code[:length]]
            sequence.append({
                "code": part.code,
                "label": part.label,
                "level": part.level,
                "parent_code": part.parent_code,
                "source": "psoc_2022_workbook",
                "source_sheet": part.source_sheet,
                "source_row": part.source_row,
                "psoc_1992_relation": part.psoc_1992_relation,
                "isco_2008_relation": part.isco_2008_relation,
            })
        return sequence

    def crosswalk_for_esco(self, esco_uri: str) -> dict | None:
        if esco_uri not in self._occupation_rows:
            return None
        records = [
            row for row in self.crosswalk["mappings"]
            if row["esco_uri"] == esco_uri and row["review_status"] == "source_reviewed"
        ]
        if len(records) != 1:
            return None
        row = records[0]
        target = self._psoc_by_code.get(row["psoc_code"])
        if target is None or target.level != "unit":
            raise ValueError("Crosswalk target is not a PSOC unit group")
        if row["esco_label"] != self._occupation_rows[esco_uri]["preferredLabel"]:
            raise ValueError("Crosswalk ESCO label changed")
        if row["auto_assign"] is not False:
            raise ValueError("Crosswalk cannot auto-assign PH codes")
        return dict(row)

    def reviewed_profile(
        self, *, occupation_psoc_code: str | None, region_psgc_code: str | None,
        education_level: str | None, years_experience: str | None,
        normalized_skills: list[str]
    ) -> UserProfile:
        if occupation_psoc_code is not None and (
            occupation_psoc_code not in self._psoc_by_code
            or self._psoc_by_code[occupation_psoc_code].level != "unit"
        ):
            raise ContractError("occupation_psoc_code", "not a known PSOC unit group")
        if region_psgc_code is not None and (
            "psgc_2025_07", region_psgc_code
        ) not in self._by_id["region"]:
            raise ContractError("region_psgc_code", "unknown dated region code")
        for skill in normalized_skills:
            if ("esco_local", skill) not in self._by_id["skill"]:
                raise ContractError("normalized_skills", "unknown ESCO identifier")
        value = UserProfile.from_dict({
            "feature_schema_version": 1,
            "occupation_psoc_code": occupation_psoc_code,
            "region_psgc_code": region_psgc_code,
            "years_experience": years_experience,
            "education_level": education_level,
            "normalized_skills": normalized_skills,
        })
        return value


    def skills_for_esco_occupation(self, occupation_uri: str, *, limit: int = 100) -> list[dict]:
        if occupation_uri not in self._occupation_rows:
            raise ValueError("Unknown ESCO occupation URI")
        if not 1 <= limit <= 1000:
            raise ValueError("Limit must be between 1 and 1000")
        if self._related_skills is None:
            connections: dict[str, dict[tuple[str, str], dict]] = {}
            path = self.root / "data" / "raw" / "esco" / "occupationSkillRelations_en.csv"
            with path.open(encoding="utf-8-sig", newline="") as file:
                for row in csv.DictReader(file):
                    occupation = row["occupationUri"]
                    skill = row["skillUri"]
                    relation = row["relationType"]
                    if (
                        occupation not in self._occupation_rows
                        or skill not in self._skill_rows
                        or relation not in {"essential", "optional"}
                    ):
                        raise ValueError("Broken ESCO occupation-skill relationship")
                    connection = {
                        "skill_uri": skill,
                        "skill_label": self._skill_rows[skill]["preferredLabel"],
                        "relation_type": relation,
                        "source": "esco_local",
                    }
                    connections.setdefault(occupation, {})[(skill, relation)] = connection
            self._related_skills = {
                occupation: sorted(
                    items.values(),
                    key=lambda item: (
                        0 if item["relation_type"] == "essential" else 1,
                        key(item["skill_label"]), item["skill_uri"],
                    ),
                )
                for occupation, items in connections.items()
            }
        return self._related_skills.get(occupation_uri, [])[:limit]
