from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path

MISSING_CODES = frozenset({"", "-99", "-88", "-77", "-66", "-44"})
FIELDS = (
    "earnings_h", "l_hrearn", "employed", "wage_worker", "informal_wage",
    "w1_psuwt", "w2_finhhwt", "w3_finspwt",
    "occupation", "years_educ", "region_code", "hours", "total_hours_week",
)


def _positive(value: str) -> bool:
    if value.strip() in MISSING_CODES:
        return False
    try:
        x = Decimal(value)
        return x.is_finite() and x > 0
    except InvalidOperation:
        return False


def inspect_step(root: Path, manifest: dict) -> dict:
    raw = "data/raw/step/step_ph_weights_merged_public.csv"
    source = root / raw
    evidence = next((f for f in manifest["files"] if f["path"] == raw), None)
    if evidence is None:
        raise ValueError("STEP source must be registered before inspection")
    with source.open("rb") as file:
        sha = hashlib.file_digest(file, "sha256").hexdigest()
    if sha != evidence["sha256"]:
        raise ValueError("STEP checksum changed")
    csv.field_size_limit(2**24)
    counts = Counter()
    occupations = Counter()
    regions = Counter()
    with source.open(encoding="cp1252", errors="strict", newline="") as file:
        reader = csv.DictReader(file)
        columns = reader.fieldnames or []
        if len(columns) != 1276 or len(columns) != len(set(columns)):
            raise ValueError("Unexpected STEP field names")
        if not set(FIELDS).issubset(columns):
            raise ValueError("STEP audited variables absent")
        replicate_household = [c for c in columns if c.startswith("hh_rpl")]
        replicate_person = [c for c in columns if c.startswith("sp_rpl")]
        if len(replicate_household) != 30 or len(replicate_person) != 30:
            raise ValueError("Expected STEP replicate weight columns")
        for row in reader:
            if None in row:
                raise ValueError("STEP row has extra columns")
            counts["respondents"] += 1
            for name in FIELDS:
                if row[name].strip() not in MISSING_CODES:
                    counts["present_" + name] += 1
            hourly = _positive(row["earnings_h"])
            if hourly:
                counts["positive_hourly_earnings"] += 1
                if row["occupation"].strip() not in MISSING_CODES:
                    counts["positive_hourly_with_occupation"] += 1
                if row["wage_worker"].strip() == "1":
                    counts["wage_worker_positive_hourly"] += 1
                    if row["occupation"].strip() not in MISSING_CODES:
                        counts["wage_worker_hourly_with_occupation"] += 1
                    if row["years_educ"].strip() not in MISSING_CODES:
                        counts["wage_worker_hourly_with_education"] += 1
            if _positive(row["w3_finspwt"]):
                counts["positive_person_weights"] += 1
            if row["wage_worker"].strip() == "1":
                counts["wage_worker_respondents"] += 1
            if row["employed"].strip() == "1":
                counts["employed_respondents"] += 1
            if row["occupation"].strip() not in MISSING_CODES:
                occupations[row["occupation"].strip()] += 1
            if row["region_code"].strip() not in MISSING_CODES:
                regions[row["region_code"].strip()] += 1
    if counts["respondents"] != evidence["rows"]:
        raise ValueError("STEP row count mismatch")
    return {
        "source_id": "world_bank_step_ph_2015",
        "source_sha256": sha,
        "source_file": raw,
        "encoding": "cp1252",
        "respondents": counts["respondents"],
        "columns": len(columns),
        "counts": dict(sorted(counts.items())),
        "hourly_label_blank": counts["respondents"] - counts["present_earnings_h"],
        "hourly_label_present_but_not_positive": (
            counts["present_earnings_h"] - counts["positive_hourly_earnings"]
        ),
        "potential_wage_worker_hourly_candidates_accepted": (
            counts["wage_worker_hourly_with_occupation"]
        ),
        "respondents_not_qualified_as_wage_worker_hourly_candidates": (
            counts["respondents"] - counts["wage_worker_hourly_with_occupation"]
        ),
        "occupation_major_group_counts": dict(sorted(occupations.items())),
        "distinct_survey_regions": len(regions),
        "person_weight_variable": "w3_finspwt",
        "sampling_cluster_variable": "psu",
        "replicate_weight_columns": {
            "hh": len(replicate_household), "sp": len(replicate_person)
        },
        "observed_label": "earnings_h: published derived hourly labor earnings; positive is a candidate, NOT approved monthly pay",
        "salary_label_candidate_wage_workers": counts["wage_worker_hourly_with_occupation"],
        "monthly_php_label_count": None,
        "monthly_conversion": "BLOCKED: no reviewed paid-hours/month and gross/net equivalence",
        "training_eligible": False,
        "survey_inference": "BLOCKED pending sampling weights, stratification and replicate-weight validation",
        "occupation_code_level": "one-digit ISCO major category, not four-digit PSOC",
        "rights": "Public-use terms reviewed; ML deployment/reuse permission not yet independently approved",
        "contains_personal_identifiers": True,
        "output_privacy": "Aggregate counts only; no respondent-level values, IDs or location details",
    }


def inspect_step_questionnaire(root: Path, manifest: dict) -> dict:
    from pypdf import PdfReader

    path = "data/raw/step/2.0_step_eec_questionnaire_withoutvar_en_20150817.pdf"
    info = next((x for x in manifest["files"] if x["path"] == path), None)
    if info is None:
        raise ValueError("Questionnaire not registered")
    source = root / path
    with source.open("rb") as file:
        actual = hashlib.file_digest(file, "sha256").hexdigest()
    if actual != info["sha256"]:
        raise ValueError("Questionnaire checksum changed")
    reader = PdfReader(source)
    if len(reader.pages) != 60:
        raise ValueError("Unexpected questionnaire version")
    pages = (37, 38, 39, 42)
    texts = {str(n): reader.pages[n - 1].extract_text() for n in pages}
    if "take-home pay/net wage" not in texts["39"] or "MODULE 6: WORK SKILLS" not in texts["42"]:
        raise ValueError("Questionnaire wage or work-skills sections changed")
    return {
        "source_sha256": actual,
        "pages": len(reader.pages),
        "question_references": {
            "employee_net_last_pay_period": "page 39",
            "self_employed_amount_and_period": "page 37",
            "occupational_skills_questions": "page 42 onward",
        },
        "warning": "Survey questions alone do not prove derived earnings_h processing or monthly basis",
    }


def write_aggregate_report(root: Path, result: dict) -> Path:
    payload = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    fingerprint = hashlib.sha256(payload).hexdigest()[:16]
    path = root / f"data/audit/step_jobicy_feasibility-{fingerprint}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != payload:
        raise ValueError("Existing feasibility report differs; review before replacing")
    if not path.exists():
        path.write_bytes(payload)
    return path
