from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path

from salary_predictor.audit.quality import months_older
from salary_predictor.contracts import BenchmarkObservation, PredictionResult, UserProfile
from salary_predictor.taxonomy import Taxonomy
from salary_predictor.trainer.baseline import BenchmarkOnlyBaseline

_CONFIG = "configs/result_decision_v1.json"


@dataclass(frozen=True)
class CalibratedRange:
    low: Decimal
    high: Decimal
    calibration_records: int
    empirical_coverage: Decimal
    method: str
    verified_heldout: bool


@dataclass(frozen=True)
class ModelCandidate:
    artifact_version: str
    reference_date: str
    predicted_php_monthly: Decimal
    occupation_support: int
    region_support: int
    holdout_records: int
    test_groups: int
    mapping_confidence: Decimal
    features_used: tuple[str, ...]
    salary_basis: str
    population_comparable: bool
    source_rights_verified: bool
    artifact_integrity_verified: bool
    model_release_verified: bool
    holdout_evaluation_verified: bool
    interval: CalibratedRange | None = None


@dataclass(frozen=True)
class BenchmarkCandidate:
    observation: BenchmarkObservation
    source_snapshot_verified: bool
    exact_published_scope: bool
    reviewed_profile_mapping: bool
    mapping_confidence: Decimal
    source_rights_verified: bool
    benchmark_release_verified: bool
    population_comparable: bool
    worker_sample_size: int | None = None
    published_range: tuple[Decimal, Decimal] | None = None
    published_range_verified: bool = False


@dataclass(frozen=True)
class Decision:
    result: PredictionResult
    route: str
    reason_codes: tuple[str, ...]
    used_features: tuple[str, ...]
    disclosed_worker_sample_size: int | None
    copy_key: str

    def to_dict(self) -> dict:
        return {
            "result": self.result.to_dict(),
            "route": self.route,
            "reason_codes": list(self.reason_codes),
            "used_features": list(self.used_features),
            "disclosed_worker_sample_size": self.disclosed_worker_sample_size,
            "copy_key": self.copy_key,
        }


def _positive(value: Decimal) -> bool:
    return isinstance(value, Decimal) and value.is_finite() and value > 0


def _confidence(value: Decimal, minimum: str) -> bool:
    try:
        return isinstance(value, Decimal) and value.is_finite() and Decimal(0) <= value <= 1 and value >= Decimal(minimum)
    except InvalidOperation:
        return False


def _result(
    mode: str, central: Decimal | None, *,
    lower: Decimal | None = None,
    upper: Decimal | None = None,
    range_basis: str | None = None,
    basis: str = "none",
    date_value: str | None = None,
    occupation_specificity: str = "unknown",
    region_specificity: str = "unknown",
    explanation: str,
    limitations: list[str],
    reliability_flags: list[str],
    source_id: str | None = None,
    source_version: str | None = None,
    artifact_version: str | None = None,
) -> PredictionResult:
    return PredictionResult.from_dict({
        "contract_version": 1,
        "mode": mode,
        "central_php_monthly": str(central) if central is not None else None,
        "lower_php_monthly": str(lower) if lower is not None else None,
        "upper_php_monthly": str(upper) if upper is not None else None,
        "range_basis": range_basis,
        "basis": basis,
        "data_as_of": date_value,
        "occupation_specificity": occupation_specificity,
        "region_specificity": region_specificity,
        "explanation": explanation,
        "limitations": limitations,
        "reliability_flags": sorted(set(reliability_flags)),
        "source_id": source_id,
        "source_version": source_version,
        "artifact_version": artifact_version,
        "currency": "PHP",
        "period": "monthly",
    })


def _unavailable(reasons: list[str]) -> Decision:
    codes = sorted(set(reasons)) or ["no_approved_provider"]
    prediction = _result(
        "UNAVAILABLE", None,
        explanation="No approved, comparable salary estimate or published benchmark is available for this profile.",
        limitations=[
            "No authorized salary source or evaluated model may be shown",
            "Skills, education and experience are not assigned unsupported wage increases",
        ],
        reliability_flags=codes,
    )
    return Decision(prediction, "UNAVAILABLE", tuple(codes), (), None, "unavailable")


def _checked_age(reference: str, as_of: date) -> int | None:
    try:
        ref = date.fromisoformat(reference + "-01" if len(reference) == 7 else reference)
    except (TypeError, ValueError):
        return None
    if ref > as_of:
        return None
    return months_older(reference, as_of)


def _model_rejection(
    candidate: ModelCandidate | None, profile: UserProfile, config: dict, as_of: date,
) -> list[str]:
    if candidate is None:
        return ["model_artifact_not_available"]
    p = config["model"]
    reasons = []
    if not all((
        candidate.model_release_verified,
        candidate.artifact_integrity_verified,
        candidate.source_rights_verified,
        candidate.holdout_evaluation_verified,
    )):
        reasons.append("model_release_not_verified")
    if candidate.salary_basis != "approved_comparable_monthly_PHP" or not candidate.population_comparable:
        reasons.append("model_salary_basis_mismatch")
    if not _positive(candidate.predicted_php_monthly):
        reasons.append("invalid_model_prediction")
    age = _checked_age(candidate.reference_date, as_of)
    if age is None or age > p["max_reference_age_months"]:
        reasons.append("model_vintage_unsupported")
    if not _confidence(candidate.mapping_confidence, config["mapping"]["minimum_source_reviewed_confidence"]):
        reasons.append("occupation_mapping_not_reviewed")
    for field, threshold in (
        ("occupation_support", p["min_occupation_support"]),
        ("region_support", p["min_region_support"]),
        ("holdout_records", p["min_holdout_records"]),
        ("test_groups", p["min_test_groups"]),
    ):
        observed = getattr(candidate, field)
        if type(observed) is not int or observed < threshold:
            reasons.append("insufficient_" + field)
    if not candidate.artifact_version or candidate.artifact_version != candidate.artifact_version.strip():
        reasons.append("model_artifact_version_missing")
    allowed = {
        "occupation_psoc_code", "region_psgc_code", "years_experience",
        "education_level", "normalized_skills", "industry_code", "employment_context",
    }
    if not candidate.features_used or set(candidate.features_used) - allowed:
        reasons.append("model_features_unrecognized")
    if "normalized_skills" in candidate.features_used and not p["allow_skills_as_salary_feature"]:
        reasons.append("skill_salary_feature_not_approved")
    for field in candidate.features_used:
        value = getattr(profile, field, None)
        if value is None or (field == "normalized_skills" and not value):
            reasons.append("required_model_feature_missing")
    if p["require_observed_occupation"] and profile.occupation_psoc_code is None:
        reasons.append("occupation_missing")
    return reasons


def _benchmark_rejection(
    candidate: BenchmarkCandidate | None, config: dict, as_of: date,
) -> list[str]:
    if candidate is None:
        return ["benchmark_not_available"]
    p = config["benchmark"]
    record = candidate.observation
    reasons = []
    if not all((
        candidate.benchmark_release_verified, candidate.source_snapshot_verified,
        candidate.source_rights_verified,
    )):
        reasons.append("benchmark_release_not_verified")
    if not candidate.exact_published_scope or not candidate.reviewed_profile_mapping:
        reasons.append("benchmark_exact_mapping_missing")
    if not _confidence(candidate.mapping_confidence, config["mapping"]["minimum_source_reviewed_confidence"]):
        reasons.append("occupation_mapping_not_reviewed")
    if record.source_id not in p["allowed_source_ids"] or record.metric not in p["allowed_metrics"]:
        reasons.append("unsupported_benchmark_measure")
    if record.pay_component != "total" or record.tax_basis not in {"gross", "unknown"}:
        reasons.append("benchmark_pay_basis_mismatch")
    if not candidate.population_comparable:
        reasons.append("benchmark_population_not_comparable")
    if not _positive(record.amount_php_monthly):
        reasons.append("invalid_benchmark_amount")
    age = _checked_age(record.reference_date, as_of)
    if age is None or age > p["max_reference_age_months"]:
        reasons.append("benchmark_stale_or_future_dated")
    expected = {
        "Table 1": {("occupation", "industry")},
        "Table 3": {("occupation",), ("occupation", "region")},
    }
    if record.published_dimensions not in expected.get(record.source_table, set()):
        reasons.append("unsupported_joint_table")
    if record.occupation_label is None:
        reasons.append("benchmark_occupation_missing")
    if candidate.worker_sample_size is not None:
        if type(candidate.worker_sample_size) is not int or candidate.worker_sample_size < config["reliability"]["suppress_small_groups_below"]:
            reasons.append("benchmark_insufficient_sample_support")
    elif not p["allow_unknown_worker_sample_size"]:
        reasons.append("benchmark_sample_support_unknown")
    if candidate.published_range is not None and (
        not candidate.published_range_verified or not p["allow_published_ranges"]
    ):
        reasons.append("benchmark_range_not_verified")
    return reasons


def _model_interval(candidate: ModelCandidate, config: dict) -> tuple[Decimal | None, Decimal | None, str | None, str]:
    interval = candidate.interval
    if interval is None:
        return None, None, None, "No calibrated personal prediction interval is available"
    p = config["uncertainty"]
    if (
        not interval.verified_heldout
        or not interval.method.strip()
        or type(interval.calibration_records) is not int
        or interval.calibration_records < p["min_calibration_records"]
        or not isinstance(interval.empirical_coverage, Decimal)
        or not interval.empirical_coverage.is_finite()
        or not Decimal(p["min_calibration_empirical_coverage"]) <= interval.empirical_coverage <= 1
        or not _positive(interval.low)
        or not _positive(interval.high)
        or not interval.low <= candidate.predicted_php_monthly <= interval.high
    ):
        return None, None, None, "Model range suppressed: held-out calibration evidence is insufficient"
    return interval.low, interval.high, "calibrated_model_interval", (
        f"Held-out empirical interval coverage: {interval.empirical_coverage}; "
        f"method: {interval.method}; calibration rows: {interval.calibration_records}"
    )


def route_hypothetical_verified_evidence(
    profile: UserProfile,
    *,
    model: ModelCandidate | None,
    benchmark: BenchmarkCandidate | None,
    policy: dict,
    as_of: date,
) -> Decision:
    """Policy matrix for reviewed future evidence; not a runtime release verifier."""
    if not isinstance(profile, UserProfile) or not isinstance(as_of, date):
        raise ValueError("Expected reviewed profile and as-of date")
    if policy.get("routing_order") != ["MODEL_ESTIMATE", "MARKET_BENCHMARK", "UNAVAILABLE"]:
        raise ValueError("Unreviewed routing order")
    rejected = []
    if policy.get("model_release_enabled"):
        failures = _model_rejection(model, profile, policy, as_of)
        if not failures and model is not None:
            lower, upper, range_basis, interval_copy = _model_interval(model, policy)
            flags = [] if lower is not None else ["no_calibrated_model_interval"]
            return Decision(
                _result(
                    "MODEL_ESTIMATE", model.predicted_php_monthly,
                    lower=lower, upper=upper, range_basis=range_basis,
                    basis="model_prediction",
                    date_value=model.reference_date,
                    occupation_specificity="exact",
                    region_specificity="exact" if "region_psgc_code" in model.features_used else "unknown",
                    explanation=(
                        "Model-based monthly PHP estimate using reviewed profile fields: "
                        + ", ".join(model.features_used) + ". " + interval_copy
                    ),
                    limitations=[
                        "Prediction is an association, not a guaranteed salary or causal skill premium",
                        "Error estimates apply to the evaluated cohort, not every employer",
                    ],
                    reliability_flags=flags,
                    artifact_version=model.artifact_version,
                ),
                "MODEL_ESTIMATE", tuple(flags), model.features_used, None, "model_estimate",
            )
        rejected.extend(failures)
    else:
        rejected.append("model_runtime_not_released")
    if policy.get("benchmark_release_enabled"):
        failures = _benchmark_rejection(benchmark, policy, as_of)
        if not failures and benchmark is not None:
            obs = benchmark.observation
            lower = upper = None
            if benchmark.published_range is not None and benchmark.published_range_verified:
                lower, upper = benchmark.published_range
                if not (
                    _positive(lower) and _positive(upper)
                    and lower <= obs.amount_php_monthly <= upper
                ):
                    return _unavailable(rejected + ["invalid_published_benchmark_range"])
            age = _checked_age(obs.reference_date, as_of)
            flags = ["published_average_not_personal_prediction"]
            if benchmark.worker_sample_size is None:
                flags.append("worker_sample_size_not_published")
            if age is not None and age > policy["reliability"]["source_age_warning_months"]:
                flags.append("historical_source_age_warning")
            count = (
                benchmark.worker_sample_size
                if policy["reliability"]["only_reveal_source_reported_support"] else None
            )
            return Decision(
                _result(
                    "MARKET_BENCHMARK", obs.amount_php_monthly,
                    lower=lower, upper=upper,
                    range_basis="published_source_range" if lower is not None else None,
                    basis="published_average_monthly_wage",
                    date_value=obs.reference_date,
                    occupation_specificity="exact",
                    region_specificity="exact" if "region" in obs.published_dimensions else "national",
                    explanation=(
                        f"Published {obs.metric} from {obs.source_id}, {obs.source_table}, "
                        f"reference {obs.reference_date}; scoped to {obs.population}. "
                        "This is a historical published average, not an individual prediction."
                    ),
                    limitations=[
                        "The source's population and pay basis may not match an individual worker",
                        "Personal experience, education and skills were not salary-adjusted",
                    ],
                    reliability_flags=flags,
                    source_id=obs.source_id,
                    source_version=obs.source_version,
                ),
                "MARKET_BENCHMARK", tuple(flags), (), count, "market_benchmark",
            )
        rejected.extend(failures)
    else:
        rejected.append("benchmark_runtime_not_released")
    return _unavailable(rejected)


def load_local_policy(root: Path) -> dict:
    policy = json.loads((root / _CONFIG).read_text(encoding="utf-8"))
    if (
        policy.get("schema_version") != 1
        or policy.get("policy_id") != "result_decision_part10_v1"
        or policy.get("status") != "DRAFT_NOT_APPROVED_FOR_RUNTIME"
        or policy.get("runtime_release_enabled") is not False
        or policy.get("model_release_enabled") is not False
        or policy.get("benchmark_release_enabled") is not False
        or policy.get("routing_order") != ["MODEL_ESTIMATE", "MARKET_BENCHMARK", "UNAVAILABLE"]
        or policy.get("benchmark", {}).get("allow_region_industry_table_join") is not False
        or policy.get("benchmark", {}).get("allow_occupation_hierarchy_fallback") is not False
        or policy.get("benchmark", {}).get("allow_published_ranges") is not False
        or policy.get("uncertainty", {}).get("allow_synthetic_percentage_bands") is not False
        or policy.get("model", {}).get("allow_skills_as_salary_feature") is not False
    ):
        raise ValueError("Result routing policy is not the approved local blocked-only draft")
    return policy


class LocalResultResolver:
    def __init__(self, root: Path, *, as_of: date):
        self.root = root.resolve(strict=True)
        self.as_of = as_of
        self.policy = load_local_policy(self.root)
        self.baseline = BenchmarkOnlyBaseline(self.root, as_of=as_of)
        self.taxonomy = Taxonomy(self.root)
        self._report = self.baseline.reproducible_report()
        if self._report["runtime_release"] != "BLOCKED_NOT_APPROVED":
            raise ValueError("Source status changed; part 10 resolver must be reviewed")

    def _known_occupation(self, code: str) -> bool:
        try:
            return self.taxonomy.psoc_hierarchy(code)[-1]["level"] == "unit"
        except ValueError:
            return False

    def _known_region(self, code: str) -> bool:
        match = self.taxonomy.resolve("region", code)
        return (
            match.status == "matched"
            and match.chosen is not None
            and match.chosen.id == code
        )

    def decide(
        self, profile: UserProfile, *,
        published_occupation_label: str | None = None,
        published_industry_code: str | None = None,
        published_region_label: str | None = None,
    ) -> Decision:
        if not isinstance(profile, UserProfile):
            raise ValueError("Expected typed reviewed UserProfile")
        reasons = ["model_runtime_not_released", "benchmark_runtime_not_released"]
        if profile.occupation_psoc_code is None:
            reasons.append("profile_occupation_missing")
        elif not self._known_occupation(profile.occupation_psoc_code):
            reasons.append("occupation_unrecognized_local_taxonomy")
        if profile.region_psgc_code is None:
            reasons.append("profile_region_missing")
        elif not self._known_region(profile.region_psgc_code):
            reasons.append("region_unrecognized_local_taxonomy")
        if published_occupation_label is not None:
            preview = self.baseline.lookup(
                published_occupation_label,
                industry_code=published_industry_code,
                region_label=published_region_label,
            )
            reasons.append(
                "matching_source_cell_not_released"
                if preview.observation else "published_scope_unavailable"
            )
        elif published_industry_code is not None or published_region_label is not None:
            reasons.append("original_source_occupation_label_missing")
        if _checked_age(self._report["source_reference_date"], self.as_of) is not None and (
            months_older(self._report["source_reference_date"], self.as_of)
            > self.policy["benchmark"]["max_reference_age_months"]
        ):
            reasons.append("historical_benchmark_vintage_stale_for_runtime")
        return _unavailable(reasons)

    def local_diagnostic(self, profile: UserProfile, **kwargs: str | None) -> dict:
        decision = self.decide(profile, **kwargs)
        return {
            **decision.to_dict(),
            "dataset_snapshot_sha256": self._report["dataset_snapshot_sha256"],
            "original_published_scope_preview": (
                "source_exists_but_runtime_not_released"
                if "matching_source_cell_not_released" in decision.reason_codes
                else "not_available_or_not_checked"
            ),
            "model_interval_calibration": "NOT_AVAILABLE",
            "worker_sample_size": None,
            "inference_runtime_release": "BLOCKED",
            "historical_source_2022": True,
        }

    def policy_audit(self) -> dict:
        return {
            "schema_version": 1,
            "part": "10",
            "as_of": self.as_of.isoformat(),
            "status": "PASS_FAIL_CLOSED_ROUTING_ONLY",
            "runtime_release": "BLOCKED",
            "result_mode": "UNAVAILABLE",
            "policy_sha256": hashlib.sha256(
                (self.root / _CONFIG).read_bytes()
            ).hexdigest(),
            "benchmark_snapshot_sha256": self._report["dataset_snapshot_sha256"],
            "baseline_quality": self._report["audit_status"],
            "source_reference_date": self._report["source_reference_date"],
            "historical_source_stale_for_runtime": (
                months_older(self._report["source_reference_date"], self.as_of)
                > self.policy["benchmark"]["max_reference_age_months"]
            ),
            "approved_runtime_model_artifacts": 0,
            "approved_runtime_benchmark_bundles": 0,
            "model_interval_coverage": None,
            "benchmark_worker_sample_size": None,
            "numeric_skill_salary_effect": None,
            "routing": ["MODEL_ESTIMATE", "MARKET_BENCHMARK", "UNAVAILABLE"],
            "actual_selected_mode": "UNAVAILABLE",
            "explanation": (
                "Routing branches are independently testable with fictional evidence; "
                "no runtime branch beyond UNAVAILABLE is released"
            ),
        }


def write_policy_audit(root: Path, audit: dict) -> dict:
    root = root.resolve(strict=True)
    payload = (
        json.dumps(audit, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    folder = root / "data/audit"
    if folder.is_symlink():
        raise ValueError("Audit directory cannot be a symlink")
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"part10-result-routing-{digest[:16]}.json"
    try:
        with path.open("xb") as out:
            out.write(payload)
    except FileExistsError:
        if path.read_bytes() != payload:
            raise ValueError("Existing decision report changed")
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": digest,
        "status": audit["status"],
    }
