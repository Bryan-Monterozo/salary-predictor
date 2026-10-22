from __future__ import annotations

from dataclasses import dataclass

from salary_predictor.taxonomy import Taxonomy

ALLOWED_ORIGINS = frozenset({
    "user_explicit", "resume_explicit_reviewed", "job_ad_explicit_reviewed",
})


@dataclass(frozen=True, slots=True)
class ObservedSkillEncoding:
    observation_origin: str
    observed: bool
    matched_uris: tuple[str, ...]
    unresolved_text: tuple[str, ...]
    ambiguous_text: tuple[str, ...]
    duplicate_matches: int

    def to_dict(self) -> dict:
        return {
            "observation_origin": self.observation_origin,
            "observed": self.observed,
            "matched_uris": list(self.matched_uris),
            "unresolved_text": list(self.unresolved_text),
            "ambiguous_text": list(self.ambiguous_text),
            "duplicate_matches": self.duplicate_matches,
            "salary_feature_enabled": False,
        }


def encode_explicit_skills(
    taxonomy: Taxonomy,
    raw_skills: list[str] | None,
    *,
    origin: str,
    max_skills: int = 100,
) -> ObservedSkillEncoding:
    if origin not in ALLOWED_ORIGINS:
        raise ValueError("Only explicit, reviewed skill observations are accepted")
    if raw_skills is None:
        return ObservedSkillEncoding(origin, False, (), (), (), 0)
    if (
        not isinstance(raw_skills, list)
        or isinstance(max_skills, bool)
        or not 1 <= max_skills <= 100
        or len(raw_skills) > max_skills
        or any(not isinstance(item, str) or len(item) > 300 for item in raw_skills)
    ):
        raise ValueError("Invalid observed skill input")
    seen: set[str] = set()
    unknown: set[str] = set()
    ambiguous: set[str] = set()
    duplicates = 0
    for raw in raw_skills:
        phrase = raw.strip()
        if not phrase:
            continue
        result = taxonomy.resolve("skill", phrase)
        if result.status == "matched" and result.chosen is not None:
            if result.chosen.source != "esco_local":
                raise ValueError("Unsupported skill source")
            if result.chosen.id in seen:
                duplicates += 1
            seen.add(result.chosen.id)
        elif result.status == "ambiguous":
            ambiguous.add(phrase)
        else:
            # Suggested spellings need review; never turn into observed skills.
            unknown.add(phrase)
    return ObservedSkillEncoding(
        observation_origin=origin,
        observed=True,
        matched_uris=tuple(sorted(seen)),
        unresolved_text=tuple(sorted(unknown)),
        ambiguous_text=tuple(sorted(ambiguous)),
        duplicate_matches=duplicates,
    )


def binary_skill_features(
    encoding: ObservedSkillEncoding,
    *,
    frozen_training_vocabulary: tuple[str, ...],
) -> dict[str, int | None]:
    if tuple(sorted(set(frozen_training_vocabulary))) != frozen_training_vocabulary:
        raise ValueError("Frozen skill vocabulary must be sorted and unique")
    if any(
        not isinstance(uri, str)
        or not uri.startswith("http://data.europa.eu/esco/skill/")
        for uri in frozen_training_vocabulary
    ):
        raise ValueError("Feature vocabulary must contain ESCO skill URIs")
    if not encoding.observed:
        return {uri: None for uri in frozen_training_vocabulary}
    present = set(encoding.matched_uris)
    return {uri: int(uri in present) for uri in frozen_training_vocabulary}
