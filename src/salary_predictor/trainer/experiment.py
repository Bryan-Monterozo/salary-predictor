from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

from salary_predictor.audit.quality import load_policy
from salary_predictor.contracts.domain import PROFILE_FIELDS
from salary_predictor.feasibility.jobicy import inspect_archive
from salary_predictor.feasibility.step import inspect_step
from salary_predictor.source_inventory import verify_raw_files
from salary_predictor.source_registry import load_source_register, validate_source_register

from .baseline import BenchmarkOnlyBaseline

ALLOWED_PROFILE_FIELDS = (PROFILE_FIELDS - {"feature_schema_version"}) | {
    "industry_code", "employment_context"
}
EXPERIMENT_CONFIG = "configs/experiments/part08_comparison.json"


def _read_plan(root: Path) -> dict:
    path = root / EXPERIMENT_CONFIG
    plan = json.loads(path.read_text(encoding="utf-8"))
    if (
        plan.get("schema_version") != 1
        or plan.get("experiment_id") != "part08_candidate_comparison_v1"
        or plan.get("training_enabled") is not False
        or plan.get("status") != "BLOCKED_PENDING_PERMISSIONED_COMPARABLE_LABELS"
        or plan.get("never_mix_cohorts") is not True
        or set(plan.get("candidate_features", [])) - ALLOWED_PROFILE_FIELDS
        or not plan.get("excluded_features")
    ):
        raise ValueError("Part 08 experiment plan is not an approved blocked-only plan")
    names = {x["id"] for x in plan["comparison"]["candidates"]}
    if names != {"ridge_one_hot", "random_forest"}:
        raise ValueError("Unreviewed candidate model definitions")
    if any(x["status"] != "NOT_RUN" for x in plan["comparison"]["candidates"]):
        raise ValueError("Candidate configurations cannot claim fitting or scores")
    if plan["regional_priority"]["use_in_training"]:
        raise ValueError("Draft geographic sample preference is not training authorization")
    return plan


def readiness_report(root: Path, *, as_of: date) -> dict:
    root = root.resolve(strict=True)
    proposal = _read_plan(root)
    policy = load_policy(root)
    if policy["evidence"]["supervised_training_enabled"] is not False:
        raise ValueError("Part 08 runner does not perform authorized training")
    register = load_source_register(root / "configs/data_sources.json")
    issues = validate_source_register(register)
    if issues:
        raise ValueError("Source register invalid: " + "; ".join(issues))
    manifest = json.loads(
        (root / "configs/local_raw_file_manifest.json").read_text(encoding="utf-8")
    )
    errors = verify_raw_files(root, manifest)
    if errors:
        raise ValueError("Raw source integrity failed: " + "; ".join(errors[:5]))

    indexed = {source["id"]: source for source in register["sources"]}
    required = (
        ("world_bank_step_ph_2015", "individual_earnings"),
        ("jobicy_public_remote_jobs", "advertised_salary"),
    )
    for id_, expected_role in required:
        source = indexed.get(id_)
        if source is None or source["dataset_role"] != expected_role:
            raise ValueError("Expected salary evidence cohort missing from register")
        if source["enabled_for_import"] or "train_on_labeled_salary" in source["approved_use"]:
            raise ValueError("Part 08 preflight cannot accept promoted source permissions")
    step = inspect_step(root, manifest)
    jobicy = inspect_archive(root, salary_preview_limit=0)
    if step["training_eligible"] or jobicy["training_or_deployment_approved"]:
        raise ValueError("Source inspectors contradict blocked source review")
    previous = BenchmarkOnlyBaseline(root, as_of=as_of).reproducible_report()
    if previous["supervised_status"] != "BLOCKED_NO_ELIGIBLE_LABELS":
        raise ValueError("Part 07 supervised baseline state changed")

    candidates = [
        {
            "id": item["id"],
            "family": item["family"],
            "status": "NOT_RUN_BLOCKED_DATA_RIGHTS_AND_TARGET_COMPARABILITY",
            "mae_php": None,
            "median_absolute_error_php": None,
            "error_quantiles_php": None,
            "per_occupation_error": None,
            "per_region_error": None,
            "selected_hyperparameters": None,
        }
        for item in proposal["comparison"]["candidates"]
    ]
    return {
        "schema_version": 1,
        "part": "08",
        "as_of": as_of.isoformat(),
        "experiment_id": proposal["experiment_id"],
        "experiment_config_sha256": hashlib.sha256(
            (root / EXPERIMENT_CONFIG).read_bytes()
        ).hexdigest(),
        "release_policy_status": policy["status"],
        "status": "BLOCKED_NO_APPROVED_COMPARABLE_LABELED_HOLDOUT",
        "benchmark_path": "PART_07_SOURCE_PREVIEW_REMAINS_AVAILABLE",
        "benchmark_snapshot_sha256": previous["dataset_snapshot_sha256"],
        "benchmark_published_cells": previous["published_cells"],
        "source_integrity": "PASS_RAW_AND_STAGE_SHA_CHECKS",
        "raw_files_verified": len(manifest["files"]),
        "cohort_inspection": {
            "STEP": {
                "source_id": step["source_id"],
                "observed_hourly_wage_worker_candidates": (
                    step["salary_label_candidate_wage_workers"]
                ),
                "approved_monthly_php_training_rows": 0,
                "blocked_reasons": [
                    "Derived historical hourly earnings not approved as monthly gross PHP",
                    "Survey weighting and complex-sample evaluation not cleared",
                    "Public-use training/deployment permissions unresolved",
                    "One-digit ISCO occupation cannot be silently mapped to detailed PSOC",
                ],
            },
            "Jobicy": {
                "source_id": jobicy["source_id"],
                "unique_remote_job_listings": jobicy["distinct_job_ids"],
                "valid_advertised_salary_ranges": jobicy["salary_disclosed_valid_range"],
                "php_denominated_advertised_ranges": jobicy["currency_counts"].get(
                    "PHP", 0
                ),
                "apac_eligible_disclosed_ranges": jobicy["geo_priority_counts"].get(
                    "salary:APAC_ELIGIBLE", 0
                ),
                "approved_monthly_php_training_rows": 0,
                "blocked_reasons": [
                    "Global remote jobGeo eligibility is not Philippine salary geography",
                    "Salary currencies, tax basis and annualization not normalized",
                    "Training and long-term reuse rights not confirmed",
                    "No reviewed equivalent occupation/employer/time holdout",
                ],
            },
            "PSA_OWS": {
                "source_id": previous["source_id"],
                "published_aggregate_cells": previous["published_cells"],
                "approved_individual_salary_training_rows": 0,
                "blocked_reasons": [
                    "Published occupation/industry/region wage cells are aggregates",
                ],
            },
        },
        "target_currency": "PHP",
        "target_period": "monthly",
        "eligible_supervised_training_rows": 0,
        "train_count": 0,
        "validation_count": 0,
        "test_count": 0,
        "frozen_group_time_holdout": None,
        "fit_transform_evidence": "NOT_RUN_NO_ELIGIBLE_PARTITIONS",
        "source_vs_recent_period_stress_test": "NOT_RUN",
        "required_predictors_from_profile": proposal["candidate_features"],
        "predictor_observation_gate": "BLOCKED_NO_LINKED_MONTHLY_PHP_LABELS",
        "comparison_baseline": {
            "id": "part07_permissioned_occupation_median",
            "status": "NOT_TRAINED",
            "naive_global_median_mae_php": None,
            "occupation_baseline_mae_php": None,
        },
        "candidates": candidates,
        "selection_decision": "NO_MODEL_SELECTED_RETAIN_PART07_BENCHMARK_ONLY",
        "model_trained": False,
        "model_artifact": None,
        "holdout_mae_php": None,
        "test_error_quantiles_php": None,
        "subgroup_regressions": None,
        "skill_salary_premium": None,
        "geo_weighting_applied": False,
        "feature_importance_causal_claim": False,
        "runtime_promotion": "BLOCKED",
        "next_requirements": proposal["source_minimum_approvals"],
    }


def immutable_report(root: Path, report: dict) -> dict:
    root = root.resolve(strict=True)
    data = (json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    digest = hashlib.sha256(data).hexdigest()
    folder = root / "data/audit"
    if folder.is_symlink():
        raise ValueError("Audit reports directory cannot be a symlink")
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"part08-readiness-{digest[:16]}.json"
    try:
        with target.open("xb") as file:
            file.write(data)
    except FileExistsError:
        if target.read_bytes() != data:
            raise ValueError("Existing immutable readiness report differs")
    return {
        "path": target.relative_to(root).as_posix(),
        "sha256": digest,
        "status": report["status"],
    }
