from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from salary_predictor.contracts import ContractError, UserProfile
from salary_predictor.taxonomy import Taxonomy

FIELDS = (
    "job_title", "occupation_psoc_code", "region_psgc_code",
    "years_experience", "education_level", "skills", "employment_context",
)
EMPTY = dict.fromkeys(FIELDS, "")
OCCUPATION_CHOICE = re.compile(r"^(\d{4})\s+[—–-]\s+(.+)$")


@lru_cache(maxsize=4)
def get_catalog(root: str) -> Taxonomy | None:
    base = Path(root)
    if not (base / "data/raw/psoc/2022-psoc.xlsx").is_file():
        return None
    return Taxonomy(base)


def reference_choices(catalog: Taxonomy | None) -> tuple[list[dict], list[dict]]:
    if catalog is None:
        return [], []
    occupations = [
        {"code": group.code, "label": group.label}
        for group in catalog.psoc_groups if group.level == "unit"
    ]
    occupations.sort(key=lambda row: (row["label"].casefold(), row["code"]))
    regions = [
        {"code": row["code"], "label": row["label"]}
        for row in catalog.regions["regions"]
    ]
    regions.sort(key=lambda row: row["label"].casefold())
    return occupations, regions


def draft_from(form) -> dict[str, str]:
    return {field: str(form.get(field, "")).strip() for field in FIELDS}


def occupation_code(value: str, catalog: Taxonomy | None) -> str | None:
    if not value:
        return None
    if catalog is None:
        raise ContractError("occupation_psoc_code", "occupation lookup is unavailable")
    option = OCCUPATION_CHOICE.fullmatch(value)
    code = option.group(1) if option else value
    match = catalog.resolve("occupation", code)
    if match.status != "matched" or match.chosen.source != "psoc_2022_workbook":
        raise ContractError("occupation_psoc_code", "choose a listed PSOC occupation")
    if option and option.group(2) != match.chosen.label:
        raise ContractError("occupation_psoc_code", "choose a listed PSOC occupation")
    return match.chosen.id


def validate_profile(
    raw: dict[str, str], catalog: Taxonomy | None
) -> tuple[UserProfile | None, dict, dict[str, str]]:
    errors: dict[str, str] = {}
    if not 2 <= len(raw["job_title"]) <= 150:
        errors["job_title"] = "Enter a job title (2–150 characters)."
    if len(raw["skills"]) > 500:
        errors["skills"] = "Skills must fit within 500 characters."
    skills = [word.strip() for word in raw["skills"].split(",") if word.strip()]
    if len(skills) > 30:
        errors["skills"] = "Enter no more than 30 skills."
    if any(len(word) > 100 for word in skills):
        errors["skills"] = "Each skill must be 100 characters or fewer."
    if len({word.casefold() for word in skills}) != len(skills):
        errors["skills"] = "List each skill only once."

    code = None
    try:
        code = occupation_code(raw["occupation_psoc_code"], catalog)
    except ContractError as exc:
        errors[exc.field] = exc.reason

    region_code = raw["region_psgc_code"] or None
    if region_code and (catalog is None or not any(
        row["code"] == region_code for row in catalog.regions["regions"]
    )):
        errors["region_psgc_code"] = "Choose a listed Philippine region."

    skill_details = []
    normalized_skills = []
    for skill in skills:
        # Do not auto-treat a role-linked skill as one the user knows.
        match = catalog.resolve("skill", skill) if catalog else None
        if match and match.status == "matched" and match.chosen.source == "esco_local":
            identifier = match.chosen.id
            skill_details.append({
                "label": match.chosen.label, "value": identifier, "mapped": True
            })
        else:
            identifier = skill
            skill_details.append({"label": skill, "value": skill, "mapped": False})
        normalized_skills.append(identifier)
    if len({item.casefold() for item in normalized_skills}) != len(normalized_skills):
        errors["skills"] = "Some skills refer to the same concept; remove duplicates."

    profile = None
    try:
        profile = UserProfile.from_dict({
            "feature_schema_version": 1,
            "occupation_psoc_code": code,
            "region_psgc_code": region_code,
            "years_experience": raw["years_experience"] or None,
            "education_level": raw["education_level"] or None,
            "normalized_skills": normalized_skills,
            "employment_context": raw["employment_context"] or None,
        })
    except ContractError as exc:
        errors.setdefault(exc.field, exc.reason)
    except (ValueError, TypeError):
        errors["profile"] = "Check the entered profile values."

    overview = {
        "occupation": None,
        "region": None,
        "skills": skill_details,
    }
    if catalog is not None and code:
        selected = catalog.resolve("occupation", code).chosen
        overview["occupation"] = selected.label if selected else None
    if catalog is not None and region_code and "region_psgc_code" not in errors:
        selected = catalog.resolve("region", region_code).chosen
        overview["region"] = selected.label if selected else None
    return (profile if not errors else None), overview, errors
