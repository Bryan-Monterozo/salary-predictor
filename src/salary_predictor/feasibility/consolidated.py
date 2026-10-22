from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path

from salary_predictor.feasibility.jobicy import _salary, _text
from salary_predictor.source_inventory import verify_raw_files
from salary_predictor.taxonomy import Taxonomy
from salary_predictor.trainer.baseline import BenchmarkOnlyBaseline

POLICY_PATH = "configs/feasibility/consolidation_policy.json"
SOURCE = "jobicy_public_remote_jobs"


def _json_bytes(value: dict) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def _hash(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _policy(root: Path) -> dict:
    policy = json.loads((root / POLICY_PATH).read_text(encoding="utf-8"))
    if (
        policy.get("schema_version") != 1
        or policy.get("policy_id") != "philippines_first_job_evidence_v1"
        or policy.get("training_enabled") is not False
        or policy.get("salary_training_enabled") is not False
        or policy.get("forecast_enabled") is not False
        or policy.get("ranking_only") is not True
        or policy.get("minimum_role_support_for_path_suggestion", 0) < 2
    ):
        raise ValueError("Unreviewed consolidation or training configuration")
    for value in policy["weights"].values():
        try:
            weight = Decimal(value)
        except (InvalidOperation, TypeError) as exc:
            raise ValueError("Invalid geographic relevance weight") from exc
        if not weight.is_finite() or not Decimal(0) < weight <= 5:
            raise ValueError("Unsafe geographic relevance weight")
    return policy


def relevance(job_geo: str, policy: dict) -> dict:
    segments = {part.casefold().strip() for part in job_geo.split(",") if part.strip()}
    regional = {part.casefold() for part in policy["regional_peer_tokens"]}
    apac = {part.casefold() for part in policy["other_apac_tokens"]}
    if "philippines" in segments:
        bucket = "PH_ELIGIBLE"
        key = "philippines_eligible"
    elif segments & regional:
        bucket = "REGIONAL_PEER_ELIGIBLE"
        key = "regional_peer_eligible"
    elif segments & apac:
        bucket = "OTHER_APAC_ELIGIBLE"
        key = "other_apac_eligible"
    else:
        bucket = "GLOBAL_OR_UNKNOWN"
        key = "global_or_unresolved"
    return {
        "bucket": bucket,
        "ranking_weight": policy["weights"][key],
        "geography_confidence": "remote_eligibility_only",
        "model_sample_weight_applied": False,
    }


def _archive_jobs(root: Path) -> tuple[dict[str, tuple[dict, dict]], Counter]:
    folder = root / "data/raw/jobicy"
    saved: dict[str, tuple[dict, dict]] = {}
    by_url: dict[str, str] = {}
    counts: Counter = Counter()
    for meta_path in sorted(folder.glob("*.meta.json"), reverse=True):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("archive_type") != "api":
            continue
        file_name = meta["raw_filename"]
        if Path(file_name).name != file_name:
            raise ValueError("Unsafe Jobicy archive path")
        raw = (folder / file_name).read_bytes()
        if _hash(raw) != meta["response_sha256"]:
            raise ValueError("Source archive checksum changed")
        counts["api_snapshots"] += 1
        payload = json.loads(raw)
        if payload.get("success") is False:
            counts["api_error_snapshots"] += 1
            continue
        if not isinstance(payload.get("jobs"), list):
            raise ValueError("Malformed Jobicy jobs archive")
        for item in payload["jobs"]:
            counts["api_rows_seen"] += 1
            identity = str(item.get("id") or "").strip()
            link = str(item.get("url") or "").split("?")[0].rstrip("/")
            if not identity:
                counts["missing_identity"] += 1
                continue
            canonical = identity if identity in saved else by_url.get(link) if link else None
            if canonical:
                counts["duplicate_snapshot_rows"] += 1
                old_item, _ = saved[canonical]
                if _salary(item) and not _salary(old_item):
                    saved[canonical] = (item, meta)
                continue
            saved[identity] = (item, meta)
            if link:
                by_url[link] = identity
    return saved, counts


def _skill_patterns(taxonomy: Taxonomy, policy: dict) -> list[tuple[str, str | None, re.Pattern]]:
    patterns = []
    for phrase in policy["skill_text_probes"]:
        match = taxonomy.resolve("skill", phrase)
        uri = match.chosen.id if match.status == "matched" and match.chosen else None
        regex = re.compile(r"(?<![\w])" + re.escape(phrase) + r"(?![\w])", re.I)
        patterns.append((phrase, uri, regex))
    return patterns


def _job_record(identity: str, item: dict, meta: dict, policy: dict, patterns: list) -> dict:
    title = str(item.get("jobTitle") or "").strip()
    if not title:
        raise ValueError("Jobicy listing missing title; cannot consolidate")
    plain = _text(str(item.get("jobDescription") or ""))
    skills = [
        {"phrase": term, "esco_uri": uri, "evidence": "unreviewed_description_mention"}
        for term, uri, pattern in patterns
        if pattern.search(plain)
    ]
    salary = None
    unit = _salary(item)
    if unit:
        salary = {
            "amount_min": str(item["salaryMin"]),
            "amount_max": str(item["salaryMax"]),
            "currency": unit[0],
            "period": unit[1],
            "basis": "advertised_offer_range",
            "converted_to_php": False,
        }
    job_geo = str(item.get("jobGeo") or "Unknown").strip()
    return {
        "record_id": "jobicy:" + identity,
        "source_id": SOURCE,
        "cohort": "advertised_salary",
        "title": title,
        "title_search_key": " ".join(title.casefold().split()),
        "job_geo": job_geo,
        "relevance": relevance(job_geo, policy),
        "seniority_label": str(item.get("jobLevel") or "") or None,
        "experience_years_observed": None,
        "listed_date": str(item.get("pubDate") or "") or None,
        "source_url": str(item.get("url") or "").split("?")[0].rstrip("/"),
        "source_response_sha256": meta["response_sha256"],
        "source_retrieved_utc": meta["retrieved_utc"],
        "description_skill_mentions": skills,
        "salary": salary,
        "salary_label_present": salary is not None,
        "monthly_php_training_label": None,
        "approved_for_training": False,
        "estimated_missing_salary": None,
    }


def consolidate(root: Path, *, as_of: date) -> tuple[dict, list[dict], list[dict]]:
    root = root.resolve(strict=True)
    policy = _policy(root)
    raw_manifest = json.loads(
        (root / "configs/local_raw_file_manifest.json").read_text(encoding="utf-8")
    )
    failures = verify_raw_files(root, raw_manifest)
    if failures:
        raise ValueError("Raw manifest verification failed: " + "; ".join(failures[:3]))
    taxonomy = Taxonomy(root)
    patterns = _skill_patterns(taxonomy, policy)
    found, counts = _archive_jobs(root)
    rows = [
        _job_record(identity, item, meta, policy, patterns)
        for identity, (item, meta) in sorted(found.items())
    ]
    queue = [
        {
            "record_id": item["record_id"],
            "job_title": item["title"],
            "source_url": item["source_url"],
            "source_response_sha256": item["source_response_sha256"],
            "status": "NEEDS_VERIFIABLE_OFFER_SALARY_SOURCE",
            "do_not_impute": True,
        }
        for item in rows if item["salary"] is None
    ]
    salaries = [item for item in rows if item["salary"] is not None]
    buckets = Counter(item["relevance"]["bucket"] for item in rows)
    currencies = Counter(item["salary"]["currency"] for item in salaries)
    periods = Counter(item["salary"]["period"] for item in salaries)
    baseline = BenchmarkOnlyBaseline(root, as_of=as_of).reproducible_report()
    report = {
        "schema_version": 1,
        "status": "CONSOLIDATED_RESEARCH_EVIDENCE_NOT_TRAINING",
        "as_of": as_of.isoformat(),
        "policy_sha256": _hash((root / POLICY_PATH).read_bytes()),
        "raw_manifest_sha256": _hash(
            (root / "configs/local_raw_file_manifest.json").read_bytes()
        ),
        "jobicy_archived_snapshots": counts["api_snapshots"],
        "jobicy_api_rows_seen_in_snapshots": counts["api_rows_seen"],
        "jobicy_snapshot_duplicates_excluded": counts["duplicate_snapshot_rows"],
        "unique_job_postings": len(rows),
        "valid_disclosed_salary_ranges": len(salaries),
        "missing_salary_enrichment_candidates": len(queue),
        "relevance_bucket_distribution": dict(sorted(buckets.items())),
        "salary_currency_distribution": dict(sorted(currencies.items())),
        "salary_period_distribution": dict(sorted(periods.items())),
        "postings_with_skill_mentions": sum(bool(r["description_skill_mentions"]) for r in rows),
        "skills_are_unreviewed_mentions": True,
        "psa_ows": {
            "separate_cohort": "aggregate_benchmark",
            "published_cells": baseline["published_cells"],
            "reference_date": baseline["source_reference_date"],
            "snapshot_sha256": baseline["dataset_snapshot_sha256"],
            "joined_into_job_offer_labels": False,
        },
        "step": {
            "separate_cohort": "historical_individual_hourly_earnings",
            "verified_wage_worker_hourly_occupation_candidates": 987,
            "joined_into_job_offer_labels": False,
        },
        "salary_forecast_available": False,
        "model_trained": False,
        "approved_training_rows": 0,
        "ml_rights_review_required": True,
        "no_generated_salary_labels": True,
    }
    return report, rows, queue


def preview_career_match(
    rows: list[dict], *, job_title: str, skills: list[str],
    years_experience: Decimal | None, current_salary_php: Decimal | None,
    min_support: int,
) -> dict:
    normalized = " ".join(job_title.casefold().split())
    if not normalized or len(normalized) > 150:
        raise ValueError("Provide a job title")
    if len(skills) > 100 or any(not isinstance(x, str) for x in skills):
        raise ValueError("Invalid user skills")
    if years_experience is not None and (
        not years_experience.is_finite() or not 0 <= years_experience <= 70
    ):
        raise ValueError("Invalid experience")
    if current_salary_php is not None and (
        not current_salary_php.is_finite() or current_salary_php <= 0
    ):
        raise ValueError("Invalid current salary")
    matched = [
        item for item in rows
        if normalized in item["title_search_key"]
    ]
    matched.sort(
        key=lambda job: (
            -Decimal(job["relevance"]["ranking_weight"]),
            not job["salary_label_present"],
            job["title_search_key"],
            job["record_id"],
        )
    )
    user_skills = {value.casefold().strip() for value in skills}
    by_skill = Counter(
        mention["phrase"]
        for item in matched
        for mention in item["description_skill_mentions"]
    )
    supported = [
        {"skill": term, "listing_mentions": count, "type": "potential_skill_to_review"}
        for term, count in by_skill.most_common()
        if count >= min_support and term.casefold() not in user_skills
    ]
    role_counts = Counter((item["title"], item["seniority_label"]) for item in matched)
    related_roles = [
        {
            "job_title": title,
            "advertised_level": level,
            "example_postings": number,
            "role_path_claim": "related_vacancy_not_verified_promotion",
        }
        for (title, level), number in role_counts.most_common(10)
    ]
    return {
        "status": "JOB_MATCH_RESEARCH_NOT_SALARY_PREDICTION",
        "matched_titles_count": len(matched),
        "salary_disclosed_in_matched_ads": sum(
            item["salary_label_present"] for item in matched
        ),
        "matched_job_examples": [
            {
                "record_id": item["record_id"],
                "title": item["title"],
                "job_geo": item["job_geo"],
                "currency_and_period": (
                    {"currency": item["salary"]["currency"], "period": item["salary"]["period"]}
                    if item["salary"] else None
                ),
                "salary": item["salary"],
                "source_url": item["source_url"],
            }
            for item in matched[:10]
        ],
        "career_skill_research_suggestions": supported[:10],
        "related_role_path_options": related_roles,
        "roles_are_not_guaranteed_promotions": True,
        "suggestions_are_not_skill_salary_premiums": True,
        "current_salary_reference_php": (
            str(current_salary_php) if current_salary_php is not None else None
        ),
        "current_salary_is_not_guaranteed_offer_floor": True,
        "years_experience_received_not_model_adjusted": (
            str(years_experience) if years_experience is not None else None
        ),
        "predicted_salary_php_monthly": None,
        "forecast_salary_php_monthly": None,
        "forecast_status": "BLOCKED_NO_COMPARABLE_PERMISSIONED_PHP_LABELS",
        "user_data_recorded_as_training_label": False,
    }


def export_consolidated(root: Path, report: dict, rows: list[dict], queue: list[dict]) -> dict:
    root = root.resolve(strict=True)
    directory = root / "data/audit"
    if directory.is_symlink():
        raise ValueError("Unsafe audit output directory")
    directory.mkdir(parents=True, exist_ok=True)
    serialized = (
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n"
    ).encode()
    job_sha = _hash(serialized)
    files = {
        f"job-evidence-{job_sha[:16]}.jsonl": serialized,
        f"missing-job-salary-queue-{job_sha[:16]}.json": _json_bytes({
            "status": "RESEARCH_QUEUE_NO_IMPUTED_SALARIES", "items": queue,
        }),
        f"job-evidence-report-{job_sha[:16]}.json": _json_bytes({
            **report, "consolidated_rows_sha256": job_sha,
        }),
    }
    for name, body in files.items():
        dest = directory / name
        try:
            with dest.open("xb") as handle:
                handle.write(body)
        except FileExistsError:
            if dest.read_bytes() != body:
                raise ValueError("Existing evidence export changed")
    return {
        "files": {name: _hash(body) for name, body in files.items()},
        "unique_postings": len(rows),
        "training_enabled": False,
    }
