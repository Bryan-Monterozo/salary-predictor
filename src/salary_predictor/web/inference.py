from __future__ import annotations

import hashlib
import json
import re
from datetime import date
from pathlib import Path

from salary_predictor.audit.quality import load_staged_benchmarks, months_older
from salary_predictor.contracts import BenchmarkObservation, PredictionResult, UserProfile

SOURCE = "psa_ows_2022_public_tables"
STAGED = "data/staged/psa_ows_2022_public_tables-25a47e4e13d22b33.jsonl"
STAGED_SHA = "25a47e4e13d22b332c5b4116251a5269cf065bdf41fc7c53f3546a9fd1f25b9f"
RUN_ID = re.compile(r"bench-[0-9a-f]{20}\Z")
TAXONOMY_FILES = {
    "configs/taxonomy/psoc_reviewed_v1.json",
    "configs/taxonomy/psgc_regions_v1.json",
    "configs/taxonomy/esco_psoc_crosswalk_v1.json",
}
CONFIG_FILES = {
    "source_registry": "configs/data_sources.json",
    "release_policy": "configs/release_policy.yaml",
    "result_policy": "configs/result_decision_v1.json",
    "raw_inventory": "configs/local_raw_file_manifest.json",
}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file(root: Path, relative: str) -> bytes:
    path = root
    for part in Path(relative).parts:
        path = path / part
        if path.is_symlink():
            raise ValueError("Release path is a symbolic link")
    if not path.is_file() or path.stat().st_size > 1024 * 1024 * 5:
        raise ValueError("Missing or oversized release file")
    return path.read_bytes()


def _json(root: Path, relative: str) -> dict:
    payload = json.loads(_file(root, relative))
    if not isinstance(payload, dict):
        raise ValueError("Invalid release object")
    return payload


def _approved_records(root: Path, today: date) -> list[BenchmarkObservation] | None:
    policy = _json(root, "configs/result_decision_v1.json")
    release = _json(root, "configs/release_policy.yaml")
    if (
        policy.get("schema_version") != 1
        or not policy.get("runtime_release_enabled")
        or not policy.get("benchmark_release_enabled")
        or policy.get("model_release_enabled")
        or policy.get("benchmark", {}).get("allow_region_industry_table_join") is not False
        or policy.get("benchmark", {}).get("allow_occupation_hierarchy_fallback") is not False
        or policy.get("benchmark", {}).get("allow_published_ranges") is not False
        or release.get("status") != "APPROVED_FOR_RUNTIME"
        or release.get("evidence", {}).get("supervised_training_enabled") is not False
    ):
        return None

    active_path = root / "data/artifacts/active.json"
    if not active_path.exists():
        return None
    active = _json(root, "data/artifacts/active.json")
    run_id = active.get("release_id")
    if not isinstance(run_id, str) or not RUN_ID.fullmatch(run_id):
        raise ValueError("Unrecognized active release ID")
    if active.get("scope") != "published_benchmark_only":
        raise ValueError("Unrecognized active release scope")
    base = f"data/artifacts/candidates/{run_id}"
    manifest_bytes = _file(root, base + "/manifest.json")
    if active.get("manifest_sha256") != _sha(manifest_bytes):
        raise ValueError("Active release manifest changed")
    manifest = json.loads(manifest_bytes)
    if (
        manifest.get("run_id") != run_id
        or manifest.get("artifact_schema_version") != 1
        or manifest.get("contract_version") != 1
        or manifest.get("feature_schema_version") != 1
        or manifest.get("status") != "CANDIDATE_NOT_APPROVED"
        or manifest.get("kind") != "published_benchmark_only"
        or manifest.get("benchmark_file") != "benchmark.json"
        or manifest.get("model_file") is not None
        or manifest.get("source_fingerprint") != STAGED_SHA
        or manifest.get("source_id") != SOURCE
        or manifest.get("permissions", {}).get("operator_runtime_release") is not False
    ):
        raise ValueError("Release schema or approved source mismatch")
    approval = _json(root, f"data/artifacts/approvals/{run_id}.json")
    if (
        approval.get("schema_version") != 1
        or approval.get("release_id") != run_id
        or approval.get("approved_for") != "published_benchmark_only"
        or approval.get("manifest_sha256") != _sha(manifest_bytes)
        or not all(approval.get(field) is True for field in (
            "operator_reviewed", "source_rights_reviewed", "tests_passed_reviewed",
            "release_policy_reviewed",
        ))
        or not isinstance(approval.get("approved_at_utc"), str)
        or not approval["approved_at_utc"].strip()
        or not isinstance(approval.get("operator_id"), str)
        or not re.fullmatch(r"[a-zA-Z0-9_.-]{3,64}", approval["operator_id"])
    ):
        raise ValueError("Release not approved by the operator")

    if set(manifest.get("taxonomy_sha256", {})) != TAXONOMY_FILES:
        raise ValueError("Taxonomy version mismatch")
    if set(manifest.get("dependencies_sha256", {})) != set(CONFIG_FILES):
        raise ValueError("Release configuration mismatch")
    for path, expected in manifest["taxonomy_sha256"].items():
        if _sha(_file(root, path)) != expected:
            raise ValueError("Taxonomy changed after approval")
    for key, path in CONFIG_FILES.items():
        if _sha(_file(root, path)) != manifest["dependencies_sha256"][key]:
            raise ValueError("Release policy changed after approval")

    bundle_bytes = _file(root, base + "/benchmark.json")
    if _sha(bundle_bytes) != manifest.get("benchmark_sha256"):
        raise ValueError("Benchmark content has changed")
    bundle = json.loads(bundle_bytes)
    if (
        bundle.get("schema_version") != 1
        or bundle.get("kind") != "published_benchmark_only"
        or bundle.get("source_id") != SOURCE
        or bundle.get("source_fingerprint") != STAGED_SHA
        or bundle.get("reference_date") != "2022-08"
        or bundle.get("runtime_release_approved") is not False
        or bundle.get("personal_model") is not None
        or bundle.get("source_table_count") != 560
        or not isinstance(bundle.get("records"), list)
        or len(bundle["records"]) != manifest.get("published_cell_count")
    ):
        raise ValueError("Published source bundle is not verified")
    max_age = policy["benchmark"]["max_reference_age_months"]
    if (
        not isinstance(max_age, int)
        or max_age > 36
        or months_older("2022-08", today) > max_age
    ):
        return None
    staged, provenance = load_staged_benchmarks(root, STAGED)
    if (
        provenance["fingerprint"] != STAGED_SHA
        or len(staged) != 560
        or [row["record"] for row in staged] != bundle["records"]
    ):
        raise ValueError("Bundle does not equal source evidence")
    return [BenchmarkObservation.from_dict(item) for item in bundle["records"]]


def _unavailable(reason: str) -> PredictionResult:
    return PredictionResult.from_dict({
        "contract_version": 1, "mode": "UNAVAILABLE",
        "central_php_monthly": None, "lower_php_monthly": None,
        "upper_php_monthly": None, "range_basis": None,
        "basis": "none", "data_as_of": None,
        "occupation_specificity": "unknown", "region_specificity": "unknown",
        "explanation": reason,
        "limitations": [
            "No approved comparable monthly Philippine salary release covers this profile.",
            "Experience, education and skills are not given unsupported pay premiums.",
        ],
        "reliability_flags": ["no_approved_matching_release"],
        "source_id": None, "source_version": None, "artifact_version": None,
        "currency": "PHP", "period": "monthly",
    })


def _match_published(
    profile: UserProfile, records: list[BenchmarkObservation]
) -> BenchmarkObservation | None:
    if not profile.occupation_psoc_code:
        return None
    # No fuzzy mapping between a modern PSOC code and a published occupation label.
    matches = [
        item for item in records
        if item.occupation_psoc_code == profile.occupation_psoc_code
        and item.source_table == "Table 3"
        and item.published_dimensions == ("occupation",)
        and item.metric == "mean_monthly_wage"
        and item.source_id == SOURCE
        and item.pay_component == "total"
        and item.tax_basis in ("gross", "unknown")
        and item.region_psgc_code is None
        and item.industry_code is None
    ]
    return matches[0] if len(matches) == 1 else None


def result_from_verified_records(
    profile: UserProfile, records: list[BenchmarkObservation]
) -> PredictionResult:
    item = _match_published(profile, records)
    if item is None:
        return _unavailable("No exact approved published occupation mapping is available.")
    return PredictionResult.from_dict({
        "contract_version": 1, "mode": "MARKET_BENCHMARK",
        "central_php_monthly": str(item.amount_php_monthly),
        "lower_php_monthly": None, "upper_php_monthly": None,
        "range_basis": None, "basis": "published_average_monthly_wage",
        "data_as_of": item.reference_date,
        "occupation_specificity": "exact",
        "region_specificity": "national",
        "explanation": (
            f"Historical {item.source_table} published average monthly wage for "
            f"{item.occupation_label} among {item.population}. "
            "National published occupation scope; not your personal salary estimate."
        ),
        "limitations": [
            "Your selected region, experience, education and skills do not adjust this figure.",
            "This published average may not match your employer or employment arrangement.",
            "No worker-level prediction interval or source-backed salary range is available.",
        ],
        "reliability_flags": [
            "published_average_not_personal_prediction", "no_personal_skill_premium",
        ],
        "source_id": item.source_id, "source_version": item.source_version,
        "artifact_version": None, "currency": "PHP", "period": "monthly",
    })


class ApprovedInferenceService:
    def __init__(self, root: Path) -> None:
        self.root = root

    def __call__(self, profile: UserProfile) -> PredictionResult:
        if not isinstance(profile, UserProfile):
            raise ValueError("Expected a reviewed UserProfile")
        try:
            verified = _approved_records(self.root, date.today())
        except (OSError, ValueError, KeyError, TypeError, OverflowError):
            verified = None
        if verified is None:
            return _unavailable(
                "No current, operator-approved salary artifact is active for this website."
            )
        return result_from_verified_records(profile, verified)
