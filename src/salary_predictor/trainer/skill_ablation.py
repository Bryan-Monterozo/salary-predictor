from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

from salary_predictor.taxonomy import Taxonomy
from salary_predictor.trainer.experiment import readiness_report

from .skills import encode_explicit_skills

CONFIG = "configs/experiments/part09_skill_ablation.json"


def load_skill_plan(root: Path) -> dict:
    plan = json.loads((root / CONFIG).read_text(encoding="utf-8"))
    if (
        plan.get("schema_version") != 1
        or plan.get("experiment_id") != "part09_observed_skill_ablation_v1"
        or plan.get("status") != "DEFERRED_NO_APPROVED_PAIRED_SKILL_AND_SALARY_LABELS"
        or plan.get("skills_enabled_for_salary_inference") is not False
        or plan.get("salary_training_enabled") is not False
        or plan.get("skills_for_search_and_extraction") is not True
        or plan.get("benchmark_baseline_unchanged") is not True
        or plan.get("release", {}).get("actual_training") is not False
        or plan.get("release", {}).get("numeric_skill_premium_allowed") is not False
    ):
        raise ValueError("Skill salary feature is not approved; expected disabled-only plan")
    if set(plan["source_evidence_allowed"]) != {
        "job_ad_explicit_reviewed", "resume_explicit_reviewed", "user_explicit"
    }:
        raise ValueError("Unexpected skill observation origins")
    if plan["encoder"]["representation"] != "binary_ESCO_URI_presence":
        raise ValueError("Unapproved skill representation")
    return plan


def ensure_same_ablation_partition(
    without: dict, with_skills: dict,
) -> None:
    required = (
        "source_snapshot_sha256", "cohort_id", "target_definition",
        "label_fingerprint", "cutoff_month", "partition_fingerprint",
        "train_ids", "validation_ids", "test_ids",
    )
    for field in required:
        if field not in without or field not in with_skills:
            raise ValueError(f"Missing frozen ablation field: {field}")
        if without[field] != with_skills[field]:
            raise ValueError(f"With/without skills ablation differs in {field}")
    for field in ("train_ids", "validation_ids", "test_ids"):
        values = without[field]
        if (
            not isinstance(values, list) or not values
            or not all(isinstance(item, str) and item for item in values)
            or len(set(values)) != len(values)
        ):
            raise ValueError(f"Invalid {field} partition")
    for a, b in (
        ("train_ids", "validation_ids"),
        ("train_ids", "test_ids"),
        ("validation_ids", "test_ids"),
    ):
        if set(without[a]) & set(without[b]):
            raise ValueError("Overlapping ablation training/evaluation identities")
    if without["target_definition"] != "approved_monthly_PHP_salary":
        raise ValueError("Only comparable monthly PHP salary experiments are supported")


def readiness(root: Path, *, as_of: date) -> dict:
    root = root.resolve(strict=True)
    load_skill_plan(root)
    part8 = readiness_report(root, as_of=as_of)
    if (
        part8["status"] != "BLOCKED_NO_APPROVED_COMPARABLE_LABELED_HOLDOUT"
        or part8["model_trained"]
        or part8["eligible_supervised_training_rows"] != 0
    ):
        raise ValueError("Part 08 readiness changed; re-review before skill experiments")
    return {
        "schema_version": 1,
        "part": "09",
        "status": "DEFERRED_NO_PAIRED_OBSERVED_SKILLS_AND_APPROVED_SALARIES",
        "as_of": as_of.isoformat(),
        "configuration_sha256": hashlib.sha256((root / CONFIG).read_bytes()).hexdigest(),
        "part08_configuration_sha256": part8["experiment_config_sha256"],
        "benchmark_snapshot_sha256": part8["benchmark_snapshot_sha256"],
        "raw_files_verified": part8["raw_files_verified"],
        "esco_skills": "READ_ONLY_VERSIONED_LOCAL_TAXONOMY",
        "source_cohorts": {
            "STEP_hourly_wage_worker_candidates": part8["cohort_inspection"]["STEP"][
                "observed_hourly_wage_worker_candidates"
            ],
            "Jobicy_global_advertised_salary_ranges": part8["cohort_inspection"]["Jobicy"][
                "valid_advertised_salary_ranges"
            ],
            "PSA_OWS_published_aggregate_cells": part8["cohort_inspection"]["PSA_OWS"][
                "published_aggregate_cells"
            ],
        },
        "approved_salary_plus_observed_skill_training_rows": 0,
        "with_skills_model": "NOT_RUN",
        "without_skills_model": "NOT_RUN",
        "shared_frozen_partition": None,
        "with_skills_holdout_mae_php": None,
        "without_skills_holdout_mae_php": None,
        "delta_mae_php": None,
        "per_occupation_region_error": None,
        "missing_skill_robustness": "NOT_MEASURED",
        "feature_importance": None,
        "skill_premiums_php": None,
        "skill_salary_effect_causal": False,
        "skill_salary_feature_enabled": False,
        "skill_search_and_extraction_enabled": True,
        "runtime_recommendation": "KEEP_SKILLS_EXTRACTOR_ONLY_AND_PART07_RUNTIME_UNAVAILABLE",
        "future_validation": [
            "Verify same-record explicitly observed skill evidence and salary labels",
            "Confirm model training and reuse permissions for each source cohort",
            "Freeze identical train/validation/test IDs and group/time cutoff",
            "Fit any top-k skill vocabulary on training only",
            "Compare same-split with/without skills against Part 07 baseline",
            "Report MAE delta, support, variance, subgroup and missing-skill robustness",
            "Release only if validated improvements justify complexity, without causal premiums",
        ],
        "source_gates": {
            "STEP": "HOURLY_NOT_APPROVED_OR_MONTHLY_PHP_NORMALIZED",
            "Jobicy": "REMOTE_ADVERTISED_NOT_APPROVED_FOR_TRAINING",
            "PSA": "AGGREGATES_NOT_EMPLOYEE_SKILL_LABELS",
        },
    }


def local_skills_preview(root: Path, observed: list[str] | None, *, origin: str) -> dict:
    taxonomy = Taxonomy(root)
    return encode_explicit_skills(taxonomy, observed, origin=origin).to_dict()


def immutable_report(root: Path, result: dict) -> dict:
    root = root.resolve(strict=True)
    data = (json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    digest = hashlib.sha256(data).hexdigest()
    folder = root / "data/audit"
    if folder.is_symlink():
        raise ValueError("Audit directory cannot be a symlink")
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"part09-skills-readiness-{digest[:16]}.json"
    try:
        with path.open("xb") as file:
            file.write(data)
    except FileExistsError:
        if path.read_bytes() != data:
            raise ValueError("Readiness report already exists with different bytes")
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": digest,
        "status": result["status"],
    }
