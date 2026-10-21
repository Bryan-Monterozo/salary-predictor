from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from salary_predictor.audit.__main__ import run_audit
from salary_predictor.audit.quality import exact_coverage, load_staged_benchmarks
from salary_predictor.contracts import BenchmarkObservation, PredictionResult
from salary_predictor.source_registry import load_source_register, validate_source_register

CORRECTED_PSA_STAGED = (
    "data/staged/psa_ows_2022_public_tables-25a47e4e13d22b33.jsonl"
)
CORRECTED_SHA256 = (
    "25a47e4e13d22b332c5b4116251a5269cf065bdf41fc7c53f3546a9fd1f25b9f"
)
BENCHMARK_ID = "psa_ows_2022_public_tables"


@dataclass(frozen=True, slots=True)
class PublishedCellPreview:
    status: str
    reason: str
    observation: BenchmarkObservation | None
    pdf_locator: str | None
    sample_size: int | None
    can_be_used_for_runtime: bool

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "reason": self.reason,
            "observation": (
                self.observation.to_dict() if self.observation is not None else None
            ),
            "pdf_locator": self.pdf_locator,
            "worker_sample_size": self.sample_size,
            "can_be_used_for_runtime": self.can_be_used_for_runtime,
        }


def _unavailable(reason: str) -> PredictionResult:
    return PredictionResult.from_dict({
        "contract_version": 1,
        "mode": "UNAVAILABLE",
        "central_php_monthly": None,
        "lower_php_monthly": None,
        "upper_php_monthly": None,
        "range_basis": None,
        "basis": "none",
        "data_as_of": None,
        "occupation_specificity": "unknown",
        "region_specificity": "unknown",
        "explanation": reason,
        "limitations": [
            "No approved runtime salary benchmark or evaluated model",
            "Published OWS wage cells are not individual salary predictions",
        ],
        "reliability_flags": ["runtime_source_not_approved"],
        "source_id": None,
        "source_version": None,
        "artifact_version": None,
        "currency": "PHP",
        "period": "monthly",
    })


class BenchmarkOnlyBaseline:
    def __init__(self, root: Path, *, as_of: date):
        root = root.resolve(strict=True)
        register = load_source_register(root / "configs/data_sources.json")
        issues = validate_source_register(register)
        if issues:
            raise ValueError("Source registry invalid; refusing benchmark baseline")
        approved = [
            source for source in register["sources"]
            if source["id"] == BENCHMARK_ID and source["dataset_role"] == "aggregate_benchmark"
        ]
        if len(approved) != 1 or not approved[0]["enabled_for_import"]:
            raise ValueError("PSA source is not approved for local ingestion")
        products = run_audit(root, CORRECTED_PSA_STAGED, as_of=as_of)
        audit = products["audit"]
        if (
            audit["snapshot_sha256"] != CORRECTED_SHA256
            or audit["source_id"] != BENCHMARK_ID
            or audit["training_eligible"]
            or audit["quality_status"] != "PASS_BENCHMARK_ONLY"
            or products["split"]["status"] != "BLOCKED_NO_ELIGIBLE_LABELS"
            or products["leakage"]["release"] != "NOT_APPROVED_FOR_RUNTIME"
        ):
            raise ValueError("Benchmark-only source or release controls failed")
        rows, _ = load_staged_benchmarks(root, CORRECTED_PSA_STAGED)
        self._records = {
            item["record"]["observation_id"]: (
                BenchmarkObservation.from_dict(item["record"]),
                item["lineage"]["source_locator"],
            )
            for item in rows
        }
        if len(self._records) != len(rows):
            raise ValueError("Duplicate source observation ID")
        self._coverage = products["coverage"]
        self._audit = audit
        self._date = as_of

    def lookup(
        self, occupation_label: str,
        *, industry_code: str | None = None, region_label: str | None = None,
    ) -> PublishedCellPreview:
        for value in (occupation_label, industry_code, region_label):
            if value is not None and not isinstance(value, str):
                raise ValueError("Use original source labels, not numeric PSOC/PSGC codes")
        if (
            industry_code is not None and not industry_code.strip()
            or region_label is not None and not region_label.strip()
        ):
            return PublishedCellPreview(
                "UNAVAILABLE", "Blank industry or region is not a published scope",
                None, None, None, False,
            )
        result = exact_coverage(
            self._coverage, occupation_label,
            industry_code=industry_code, region_label=region_label,
        )
        if result["status"] != "PUBLISHED_CELL_PRESENT_NOT_RELEASED":
            return PublishedCellPreview(
                "UNAVAILABLE", result["reason"], None, None, None, False
            )
        cell = result["cell"]
        observation, locator = self._records[cell["observation_id"]]
        return PublishedCellPreview(
            status="SOURCE_MATCH_NOT_RELEASED",
            reason=(
                "Exact PSA 2022 historical published average found for this table scope; "
                "source is approved for local staging only, not runtime inference"
            ),
            observation=observation,
            pdf_locator=locator,
            sample_size=None,
            can_be_used_for_runtime=False,
        )

    def runtime_result(
        self, occupation_label: str, *,
        industry_code: str | None = None, region_label: str | None = None,
    ) -> PredictionResult:
        preview = self.lookup(
            occupation_label, industry_code=industry_code, region_label=region_label
        )
        if preview.observation:
            return _unavailable("A matching historical wage cell exists, but its runtime release is not approved.")
        return _unavailable(f"Cannot provide a salary: {preview.reason}.")

    def reproducible_report(self) -> dict:
        return {
            "baseline_kind": "EXACT_PUBLISHED_BENCHMARK_LOOKUP_ONLY",
            "dataset_snapshot_sha256": CORRECTED_SHA256,
            "as_of": self._date.isoformat(),
            "source_id": BENCHMARK_ID,
            "source_reference_date": "2022-08",
            "published_cells": self._audit["accepted_cells"],
            "source_table_counts": self._audit["tables"],
            "published_dimensions": self._audit["published_dimensions"],
            "historical_source_warning": self._audit["staleness_warning"],
            "audit_status": self._audit["quality_status"],
            "runtime_release": "BLOCKED_NOT_APPROVED",
            "supervised_status": "BLOCKED_NO_ELIGIBLE_LABELS",
            "train_count": 0,
            "validation_count": 0,
            "test_count": 0,
            "model_trained": False,
            "model_artifact": None,
            "naive_global_median_monthly_php": None,
            "naive_global_median_mae_php": None,
            "occupation_baseline_mae_php": None,
            "occupation_baseline_median_absolute_error_php": None,
            "occupation_region_holdout_coverage": None,
            "per_occupation_test_error": None,
            "per_region_test_error": None,
            "model_confidence_interval": None,
            "skills_salary_effect": None,
            "evaluation_reason": (
                "Only published aggregate wage cells available. STEP/Jobicy supervised "
                "targets are not permissioned, normalized comparable Philippine "
                "monthly-salary cohorts. No train/test labels or model metrics."
            ),
            "supported_lookup": (
                "Use exact published source occupation label with Table 1 industry, "
                "or Table 3 published region, or a Table 3 national occupation cell"
            ),
            "not_supported": [
                "PSOC/PSGC code automatic mapping",
                "Occupation-industry-region joint wage synthesis",
                "Occupation hierarchy fallback, personal skill or experience pay effects",
                "Regional salary estimates for unlisted occupation groups",
                "Inference on this locally staged source before separate promotion",
            ],
        }


def encode_report(report: dict) -> bytes:
    return (json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
