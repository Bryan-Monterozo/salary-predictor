from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import date
from decimal import Decimal
from pathlib import Path

from salary_predictor.contracts import BenchmarkObservation

EXPECTED_TABLE_DIMENSIONS = {
    "Table 1": {("occupation", "industry")},
    "Table 3": {("occupation",), ("occupation", "region")},
}


def load_policy(root: Path) -> dict:
    path = root / "configs/release_policy.yaml"
    # JSON is also valid YAML 1.2. Keep this dependency-free for the local auditor.
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("policy_version") != 1 or config.get("evidence", {}).get(
        "supervised_training_enabled"
    ) is not False:
        raise ValueError("Unreviewed release policy cannot be used")
    return config


def load_staged_benchmarks(root: Path, relative_path: str) -> tuple[list[dict], dict]:
    root = root.resolve()
    staged_dir = (root / "data/staged").resolve()
    path = (root / relative_path).resolve()
    if not path.is_relative_to(staged_dir) or path.suffix != ".jsonl" or not path.is_file():
        raise ValueError("Only existing staged JSONL inside data/staged is auditable")

    stem = path.name[:-6]
    metadata_path = path.with_name(stem + ".lineage.json")
    report_path = path.with_name(stem + ".report.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    ingest_report = json.loads(report_path.read_text(encoding="utf-8"))
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    if (
        metadata.get("record_fingerprint") != checksum
        or ingest_report.get("record_fingerprint") != checksum
        or metadata.get("source_id") != ingest_report.get("source_id")
        or metadata.get("source_sha256") != ingest_report.get("raw_sha256")
        or metadata.get("release_status") != "not_approved_for_runtime"
        or metadata.get("records_path") != path.relative_to(root).as_posix()
        or ingest_report.get("adapter") != "psa_ows_2022_pdf"
    ):
        raise ValueError("Staged records, provenance or release status failed verification")
    if ingest_report["rejected_count"] or ingest_report["accepted_count"] != metadata["accepted_count"]:
        raise ValueError("Stage failed the source rejection/count gate")

    rows = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        item = json.loads(line)
        record = BenchmarkObservation.from_dict(item["record"])
        lineage = item["lineage"]
        if (
            record.source_id != metadata["source_id"]
            or record.source_version != metadata["source_version"]
            or lineage["source_id"] != record.source_id
            or lineage["source_version"] != record.source_version
            or lineage["raw_sha256"] != metadata["source_sha256"]
            or lineage.get("adapter") != "psa_ows_2022_pdf"
            or lineage.get("rights_status") != "verified_cc_by_public_material"
        ):
            raise ValueError(f"Staged lineage inconsistent at row {index}")
        if tuple(record.published_dimensions) not in EXPECTED_TABLE_DIMENSIONS.get(
            record.source_table, set()
        ):
            raise ValueError(f"Unsupported source table/dimensions at row {index}")
        if record.reference_date != "2022-08" or record.metric != "mean_monthly_wage":
            raise ValueError(f"Unexpected survey vintage or wage metric at row {index}")
        if record.occupation_psoc_code is not None or record.region_psgc_code is not None:
            raise ValueError("Staged code must not claim unreviewed historical mapping")
        rows.append(item)
    if len(rows) != ingest_report["accepted_count"]:
        raise ValueError("Staged record count mismatch")
    return rows, {"lineage": metadata, "ingest": ingest_report, "fingerprint": checksum}


def _quantile(values: list[Decimal], fraction: Decimal) -> Decimal:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot profile empty published values")
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _value(value: Decimal) -> str:
    return format(value.quantize(Decimal("0.01")), ".2f")


def months_older(reference: str, as_of: date) -> int:
    year, month = map(int, reference[:7].split("-"))
    return max(0, (as_of.year - year) * 12 + as_of.month - month)


def _identity(item: dict) -> tuple:
    r = item["record"]
    return (
        r["source_id"], r["source_version"], r["source_table"],
        r["reference_date"], r["occupation_label"].casefold(),
        item["lineage"].get("source_occupation_code"),
        r["industry_code"], r["region_label"], r["metric"],
    )


def _coverage_key(item: dict) -> dict:
    r = item["record"]
    return {
        "source_table": r["source_table"],
        "occupation_label": r["occupation_label"],
        "occupation_source_code": item["lineage"].get("source_occupation_code"),
        "industry_code": r["industry_code"],
        "region_label": r["region_label"],
        "amount_php_monthly": r["amount_php_monthly"],
        "source_locator": item["lineage"]["source_locator"],
        "observation_id": r["observation_id"],
    }


def exact_coverage(
    coverage: dict,
    occupation: str,
    *,
    industry_code: str | None = None,
    region_label: str | None = None,
) -> dict:
    if not isinstance(occupation, str) or not occupation.strip():
        return {"status": "UNAVAILABLE", "reason": "Missing occupation"}
    if industry_code and region_label:
        return {
            "status": "UNAVAILABLE",
            "reason": "No published joint occupation + industry + region table",
        }
    if industry_code:
        table = "Table 1"
    elif region_label:
        table = "Table 3"
    else:
        table = "Table 3"
    matches = [
        row for row in coverage["published_cells"]
        if row["source_table"] == table
        and row["occupation_label"].casefold() == occupation.casefold()
        and row["industry_code"] == industry_code
        and row["region_label"] == region_label
    ]
    if len(matches) != 1:
        return {
            "status": "UNAVAILABLE",
            "reason": (
                "Ambiguous published cells" if len(matches) > 1
                else "No exact published occupation, industry and geographic scope"
            ),
        }
    return {"status": "PUBLISHED_CELL_PRESENT_NOT_RELEASED", "cell": matches[0]}


def audit_benchmarks(rows: list[dict], provenance: dict, policy: dict, as_of: date) -> dict:
    if not rows:
        raise ValueError("No benchmark cells to audit")
    if provenance["ingest"]["adapter"] != "psa_ows_2022_pdf":
        raise ValueError("Only the published benchmark cohort is supported")
    if policy["benchmark"]["allow_combining_marginal_tables"]:
        raise ValueError("Policy would allow fabrication of joint dimensions")
    tables = Counter()
    scopes = Counter()
    distinct_occupations = defaultdict(set)
    distinct_industries = defaultdict(set)
    distinct_regions = defaultdict(set)
    observed_dates = Counter()
    seen_ids = Counter()
    natural_keys = Counter()
    missing = Counter()
    lineage_codes = Counter()
    wages: list[Decimal] = []
    issues = []
    for item in rows:
        r = item["record"]
        table = r["source_table"]
        tables[table] += 1
        scopes["+".join(r["published_dimensions"])] += 1
        distinct_occupations[table].add(r["occupation_label"])
        if r["industry_code"] is not None:
            distinct_industries[table].add(r["industry_code"])
        if r["region_label"] is not None:
            distinct_regions[table].add(r["region_label"])
        observed_dates[r["reference_date"]] += 1
        seen_ids[r["observation_id"]] += 1
        natural_keys[_identity(item)] += 1
        for field in ("occupation_psoc_code", "region_psgc_code"):
            if r[field] is None:
                missing[field] += 1
        original_code = item["lineage"].get("source_occupation_code")
        if original_code:
            lineage_codes[original_code] += 1
        wages.append(Decimal(r["amount_php_monthly"]))
    if len(set(observed_dates)) != 1:
        issues.append("Multiple survey vintages mixed")
    repeated_ids = sum(n - 1 for n in seen_ids.values() if n > 1)
    repeated_cells = sum(n - 1 for n in natural_keys.values() if n > 1)
    if repeated_ids or repeated_cells:
        issues.append("Duplicate published observation ID or natural cell key")
    q1 = _quantile(wages, Decimal("0.25"))
    median = _quantile(wages, Decimal("0.5"))
    q3 = _quantile(wages, Decimal("0.75"))
    fence = q3 + Decimal("1.5") * (q3 - q1)
    lower_fence = q1 - Decimal("1.5") * (q3 - q1)
    outliers = [
        {"observation_id": x["record"]["observation_id"], "amount_php_monthly": x["record"]["amount_php_monthly"],
         "source_table": x["record"]["source_table"]}
        for x in rows if not lower_fence <= Decimal(x["record"]["amount_php_monthly"]) <= fence
    ]
    outliers.sort(key=lambda item: item["observation_id"])
    reference = next(iter(observed_dates))
    stale_months = months_older(reference, as_of)
    published_cells = [_coverage_key(item) for item in rows]
    published_cells.sort(key=lambda r: (
        r["source_table"], r["industry_code"] or "",
        r["region_label"] or "", r["occupation_label"], r["observation_id"],
    ))
    coverage = {
        "source_id": provenance["lineage"]["source_id"],
        "source_version": provenance["lineage"]["source_version"],
        "snapshot_sha256": provenance["fingerprint"],
        "source_type": "published_aggregate_benchmark_not_individual_salary",
        "selection": "exact source table dimensions only; no hierarchy or marginal synthesis",
        "published_cells": published_cells,
    }
    analysis = {
        "audit_schema_version": 1,
        "snapshot_sha256": provenance["fingerprint"],
        "source_id": provenance["lineage"]["source_id"],
        "as_of": as_of.isoformat(),
        "observation_unit": "one published aggregate wage cell, NOT one worker",
        "accepted_cells": len(rows),
        "tables": dict(sorted(tables.items())),
        "published_dimensions": dict(sorted(scopes.items())),
        "distinct_occupation_labels_by_table": {
            k: len(v) for k, v in sorted(distinct_occupations.items())
        },
        "distinct_industry_codes_by_table": {
            k: len(v) for k, v in sorted(distinct_industries.items())
        },
        "distinct_published_region_labels_by_table": {
            k: len(v) for k, v in sorted(distinct_regions.items())
        },
        "survey_reference_dates": dict(sorted(observed_dates.items())),
        "latest_vintage_months_old": stale_months,
        "staleness_warning": stale_months > policy["benchmark"]["data_stale_after_months"],
        "current_region_dictionary_must_not_backfill_2022": True,
        "missing_per_field": dict(sorted(missing.items())),
        "unmeasured_personal_predictors": [
            "years_experience", "education_level", "normalized_skills", "individual_region"
        ],
        "skills_labeled_ratio": None,
        "skills_labeled_ratio_reason": "No worker-level salary observations; denominator unavailable",
        "worker_sample_size": None,
        "worker_sample_size_reason": "Published cells do not supply underlying worker counts",
        "source_composition": [{
            "source_id": provenance["lineage"]["source_id"],
            "role": "aggregate_benchmark",
            "cells": len(rows),
            "raw_sha256": provenance["lineage"]["source_sha256"],
        }],
        "original_occupation_codes_in_lineage": len(lineage_codes),
        "ows_only_code_cells": sum(v for code, v in lineage_codes.items() if code.endswith("b")),
        "wage_distribution_php_monthly": {
            "minimum": _value(min(wages)),
            "q1": _value(q1),
            "median_of_published_cells": _value(median),
            "q3": _value(q3),
            "maximum": _value(max(wages)),
            "mean_of_published_cells_not_worker_mean": _value(
                sum(wages) / len(wages)
            ),
        },
        "outlier_rule": "Tukey 1.5xIQR across published wage cells; review only, no automatic exclusions",
        "outliers_flagged_count": len(outliers),
        "outlier_examples": outliers[:10],
        "duplicate_observation_ids": repeated_ids,
        "duplicate_natural_cells": repeated_cells,
        "quality_issues": issues,
        "quality_status": "PASS_BENCHMARK_ONLY" if not issues else "REVIEW_REQUIRED",
        "training_eligible": False,
        "salary_model_metrics": None,
        "salary_model_metrics_reason": "No permitted individual or advertised salary labels",
    }
    return {"audit": analysis, "coverage": coverage}


def blocked_split_manifest(snapshot: str, policy: dict) -> dict:
    return {
        "protocol_version": 1,
        "status": "BLOCKED_NO_ELIGIBLE_LABELS",
        "dataset_snapshot_sha256": snapshot,
        "split_frozen": False,
        "partition_counts": {"train": 0, "validation": 0, "test": 0},
        "train_observation_ids": [],
        "test_observation_ids": [],
        "holdout_protocol": policy["holdout"],
        "supervised_draft_thresholds": policy["supervised_draft_thresholds"],
        "reason": "Only published aggregate cells are available; no eligible person-level or job-ad labels",
        "actual_evaluation_scores": None,
    }


def leakage_checklist(snapshot: str, audit: dict, policy: dict) -> dict:
    return {
        "snapshot_sha256": snapshot,
        "cohort": "published_aggregate_benchmark",
        "checks": {
            "source_and_record_lineage": "PASS",
            "duplicate_published_cells": (
                "PASS" if audit["duplicate_natural_cells"] == 0 else "FAIL"
            ),
            "published_dimensions_not_joined": "PASS",
            "individual_target_labels_observed": "BLOCKED_NONE",
            "repeat_worker_and_employer_linkage": "NOT_APPLICABLE_NO_PERSON_RECORDS",
            "source_group_disjointness": "NOT_RUN_NO_LABELED_SPLIT",
            "time_heldout_test": "NOT_RUN_NO_LABELED_SPLIT",
            "encoders_and_imputers_fit_train_only": "REQUIRED_BEFORE_ML",
            "outcome_derived_predictors_excluded": "POLICY_DEFINED_NOT_USED",
            "supervised_model_metrics": "NOT_AVAILABLE",
        },
        "prohibited_predictor_fields": policy["holdout"]["never_use_as_feature"],
        "release": "NOT_APPROVED_FOR_RUNTIME",
    }
