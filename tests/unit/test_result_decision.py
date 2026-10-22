from __future__ import annotations

import copy
import json
from dataclasses import replace
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from salary_predictor.contracts import PredictionResult, UserProfile
from salary_predictor.trainer.baseline import BenchmarkOnlyBaseline
from salary_predictor.trainer.decide import main
from salary_predictor.trainer.decision import (
    BenchmarkCandidate,
    CalibratedRange,
    LocalResultResolver,
    ModelCandidate,
    load_local_policy,
    route_hypothetical_verified_evidence,
    write_policy_audit,
)

ROOT = Path(__file__).resolve().parents[2]


def profile(*, occupation: str | None = "2512", region: str | None = "1300000000") -> UserProfile:
    return UserProfile.from_dict({
        "feature_schema_version": 1,
        "occupation_psoc_code": occupation,
        "region_psgc_code": region,
        "years_experience": None,
        "education_level": None,
        "normalized_skills": [],
    })


def policy(*, model: bool = True, benchmark: bool = True) -> dict:
    plan = copy.deepcopy(load_local_policy(ROOT))
    # Hypothetical release gates are for pure policy tests, not the real runtime loader.
    plan["model_release_enabled"] = model
    plan["benchmark_release_enabled"] = benchmark
    return plan


def candidate_model() -> ModelCandidate:
    return ModelCandidate(
        artifact_version="fictional-test-only-v0",
        reference_date="2023-02",
        predicted_php_monthly=Decimal("55000"),
        occupation_support=80,
        region_support=70,
        holdout_records=150,
        test_groups=15,
        mapping_confidence=Decimal("0.99"),
        features_used=("occupation_psoc_code", "region_psgc_code"),
        salary_basis="approved_comparable_monthly_PHP",
        population_comparable=True,
        source_rights_verified=True,
        artifact_integrity_verified=True,
        model_release_verified=True,
        holdout_evaluation_verified=True,
    )


@pytest.fixture(scope="module")
def observation():
    baseline = BenchmarkOnlyBaseline(ROOT, as_of=date(2026, 10, 21))
    item = baseline.lookup("Software Developers", industry_code="J62")
    assert item.observation is not None
    return item.observation


def candidate_benchmark(observation) -> BenchmarkCandidate:
    return BenchmarkCandidate(
        observation=observation,
        source_snapshot_verified=True,
        exact_published_scope=True,
        reviewed_profile_mapping=True,
        mapping_confidence=Decimal("0.99"),
        source_rights_verified=True,
        benchmark_release_verified=True,
        population_comparable=True,
    )


def test_actual_local_policy_and_runtime_are_immutably_blocked() -> None:
    p = load_local_policy(ROOT)
    assert p["status"] == "DRAFT_NOT_APPROVED_FOR_RUNTIME"
    assert p["runtime_release_enabled"] is False
    assert p["model_release_enabled"] is False
    assert p["benchmark_release_enabled"] is False
    assert p["uncertainty"]["allow_synthetic_percentage_bands"] is False


@pytest.mark.parametrize("change", [
    ("runtime_release_enabled", True),
    ("model_release_enabled", True),
    ("benchmark_release_enabled", True),
    ("status", "RELEASED"),
])
def test_config_tampering_fails_closed(tmp_path: Path, change: tuple) -> None:
    name, value = change
    configs = tmp_path / "configs"
    configs.mkdir()
    data = load_local_policy(ROOT)
    data[name] = value
    (configs / "result_decision_v1.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="blocked-only"):
        load_local_policy(tmp_path)


@pytest.mark.parametrize(("code", "expected"), [
    (None, "profile_occupation_missing"),
    ("9999", "model_runtime_not_released"),
    ("2512", "model_runtime_not_released"),
])
def test_production_no_artifact_and_no_released_source(
    code: str | None, expected: str,
) -> None:
    resolver = LocalResultResolver(ROOT, as_of=date(2026, 10, 21))
    result = resolver.decide(profile(occupation=code))
    assert isinstance(result.result, PredictionResult)
    assert result.result.mode == result.route == "UNAVAILABLE"
    assert expected in result.reason_codes
    assert result.result.central_php_monthly is None
    assert result.result.lower_php_monthly is None
    assert result.result.upper_php_monthly is None
    assert result.result.range_basis is None
    assert result.result.data_as_of is None
    assert result.result.source_id is None
    assert result.result.artifact_version is None


def test_existing_unreleased_source_match_is_still_unavailable() -> None:
    resolver = LocalResultResolver(ROOT, as_of=date(2026, 10, 21))
    d = resolver.local_diagnostic(
        profile(),
        published_occupation_label="Software Developers",
        published_industry_code="J62",
    )
    assert d["original_published_scope_preview"] == "source_exists_but_runtime_not_released"
    assert d["result"]["mode"] == "UNAVAILABLE"
    assert d["result"]["central_php_monthly"] is None
    assert "matching_source_cell_not_released" in d["reason_codes"]
    assert d["worker_sample_size"] is None
    assert d["model_interval_calibration"] == "NOT_AVAILABLE"
    assert d["inference_runtime_release"] == "BLOCKED"


@pytest.mark.parametrize(("name", "industry", "region"), [
    ("unknown career", None, None),
    ("Software Developers", None, "National Capital Region"),
    ("Software Developers", "J62", "National Capital Region"),
    ("Software Developers", "Q99", None),
    (None, "J62", None),
])
def test_production_unsupported_table_and_missing_labels_do_not_fallback(
    name: str | None, industry: str | None, region: str | None,
) -> None:
    resolver = LocalResultResolver(ROOT, as_of=date(2026, 10, 21))
    decision = resolver.local_diagnostic(
        profile(),
        published_occupation_label=name,
        published_industry_code=industry,
        published_region_label=region,
    )
    assert decision["result"]["mode"] == "UNAVAILABLE"
    assert decision["original_published_scope_preview"] == "not_available_or_not_checked"
    assert decision["result"]["central_php_monthly"] is None


def test_hypothetical_policy_prefers_verified_model_and_never_invents_range(observation) -> None:
    decision = route_hypothetical_verified_evidence(
        profile(), model=candidate_model(),
        benchmark=candidate_benchmark(observation),
        policy=policy(), as_of=date(2023, 4, 21),
    )
    assert decision.route == "MODEL_ESTIMATE"
    assert decision.result.central_php_monthly == Decimal("55000")
    assert decision.result.range_basis is None
    assert decision.result.lower_php_monthly is None
    assert "no_calibrated_model_interval" in decision.reason_codes
    assert decision.used_features == ("occupation_psoc_code", "region_psgc_code")
    assert decision.result.source_id is None
    assert decision.result.artifact_version == "fictional-test-only-v0"
    assert decision.result.to_dict() == PredictionResult.from_dict(decision.result.to_dict()).to_dict()


def test_hypothetical_calibrated_interval_requires_heldout_evidence(observation) -> None:
    valid = CalibratedRange(
        Decimal("45000"), Decimal("68000"), 175, Decimal("0.86"),
        "fictional-heldout-quantile-test", True,
    )
    model = replace(candidate_model(), interval=valid)
    d = route_hypothetical_verified_evidence(
        profile(), model=model, benchmark=candidate_benchmark(observation),
        policy=policy(), as_of=date(2023, 4, 21),
    )
    assert d.result.range_basis == "calibrated_model_interval"
    assert d.result.lower_php_monthly == Decimal("45000")
    assert "0.86" in d.result.explanation
    for bad in (
        replace(valid, verified_heldout=False),
        replace(valid, empirical_coverage=Decimal("0.4")),
        replace(valid, calibration_records=2),
        replace(valid, low=Decimal("60000")),
    ):
        rejected = route_hypothetical_verified_evidence(
            profile(), model=replace(model, interval=bad),
            benchmark=None, policy=policy(), as_of=date(2023, 4, 21),
        )
        assert rejected.route == "MODEL_ESTIMATE"
        assert rejected.result.lower_php_monthly is None
        assert rejected.result.range_basis is None
        assert "no_calibrated_model_interval" in rejected.reason_codes


@pytest.mark.parametrize("alter", [
    {"occupation_support": 3},
    {"region_support": 2},
    {"holdout_records": 4},
    {"test_groups": 1},
    {"mapping_confidence": Decimal("0.4")},
    {"model_release_verified": False},
    {"artifact_integrity_verified": False},
    {"source_rights_verified": False},
    {"population_comparable": False},
    {"salary_basis": "advertised_USD_hourly"},
    {"reference_date": "2029-04"},
    {"features_used": ("occupation_psoc_code", "years_experience")},
    {"features_used": ("occupation_psoc_code", "normalized_skills")},
])
def test_invalid_model_hypothetically_falls_back_to_approved_exact_benchmark(
    observation, alter: dict,
) -> None:
    d = route_hypothetical_verified_evidence(
        profile(), model=replace(candidate_model(), **alter),
        benchmark=candidate_benchmark(observation),
        policy=policy(), as_of=date(2023, 4, 21),
    )
    assert d.route == "MARKET_BENCHMARK"
    assert d.result.central_php_monthly == observation.amount_php_monthly
    assert d.result.basis == "published_average_monthly_wage"
    assert d.result.lower_php_monthly is None
    assert d.result.range_basis is None
    assert "published_average_not_personal_prediction" in d.reason_codes
    assert d.disclosed_worker_sample_size is None


@pytest.mark.parametrize("change", [
    {"benchmark_release_verified": False},
    {"source_rights_verified": False},
    {"reviewed_profile_mapping": False},
    {"mapping_confidence": Decimal("0.6")},
    {"population_comparable": False},
    {"worker_sample_size": 5},
    {"published_range": (Decimal("40000"), Decimal("50000")),
     "published_range_verified": False},
    {"published_range": (Decimal("40000"), Decimal("50000")),
     "published_range_verified": True},
])
def test_hypothetical_invalid_benchmark_never_fabricates_salary(
    observation, change: dict,
) -> None:
    d = route_hypothetical_verified_evidence(
        profile(), model=None,
        benchmark=replace(candidate_benchmark(observation), **change),
        policy=policy(), as_of=date(2023, 4, 21),
    )
    assert d.route == "UNAVAILABLE"
    assert d.result.central_php_monthly is None


def test_old_benchmark_stale_and_date_invalid_fail_closed(observation) -> None:
    stale = route_hypothetical_verified_evidence(
        profile(), model=None, benchmark=candidate_benchmark(observation),
        policy=policy(), as_of=date(2026, 10, 21),
    )
    assert stale.route == "UNAVAILABLE"
    assert "benchmark_stale_or_future_dated" in stale.reason_codes
    future = route_hypothetical_verified_evidence(
        profile(), model=None, benchmark=candidate_benchmark(observation),
        policy=policy(), as_of=date(2021, 10, 21),
    )
    assert future.route == "UNAVAILABLE"


def test_unreleased_everything_yields_one_mode_without_salary(observation) -> None:
    d = route_hypothetical_verified_evidence(
        profile(), model=candidate_model(),
        benchmark=candidate_benchmark(observation),
        policy=policy(model=False, benchmark=False),
        as_of=date(2023, 4, 21),
    )
    assert d.route == d.result.mode == "UNAVAILABLE"
    assert d.result.central_php_monthly is None


def test_cli_no_release_and_copy_remains_unavailable(capsys) -> None:
    code = main([
        "--root", str(ROOT), "--as-of", "2026-10-21",
        "--psoc", "2512", "--psgc", "1300000000",
        "--published-occupation-label", "Software Developers",
        "--published-industry-code", "J62",
    ])
    assert code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["result"]["mode"] == "UNAVAILABLE"
    assert output["copy_key"] == "unavailable"
    assert output["result"]["central_php_monthly"] is None
    assert output["model_interval_calibration"] == "NOT_AVAILABLE"



def test_unrecognized_code_flags_never_convert_to_source_label() -> None:
    resolver = LocalResultResolver(ROOT, as_of=date(2026, 10, 21))
    result = resolver.decide(profile(occupation="9999", region="9999999999"))
    assert result.result.mode == "UNAVAILABLE"
    assert "occupation_unrecognized_local_taxonomy" in result.reason_codes
    assert "region_unrecognized_local_taxonomy" in result.reason_codes
    assert result.used_features == ()


def test_reproducible_policy_audit_contains_no_user_profile_or_predictions(
    tmp_path: Path,
) -> None:
    resolver = LocalResultResolver(ROOT, as_of=date(2026, 10, 21))
    policy_audit = resolver.policy_audit()
    assert policy_audit["status"] == "PASS_FAIL_CLOSED_ROUTING_ONLY"
    assert policy_audit["actual_selected_mode"] == "UNAVAILABLE"
    assert policy_audit["approved_runtime_model_artifacts"] == 0
    assert policy_audit["approved_runtime_benchmark_bundles"] == 0
    assert policy_audit["historical_source_stale_for_runtime"] is True
    assert policy_audit["model_interval_coverage"] is None
    assert policy_audit["benchmark_worker_sample_size"] is None
    assert "profile" not in json.dumps(policy_audit).lower()
    first = write_policy_audit(tmp_path, policy_audit)
    second = write_policy_audit(tmp_path, policy_audit)
    assert first == second
    assert json.loads((tmp_path / first["path"]).read_text()) == policy_audit


def test_policy_audit_is_independent_of_current_personal_inputs() -> None:
    resolver = LocalResultResolver(ROOT, as_of=date(2026, 10, 21))
    before = resolver.policy_audit()
    resolver.decide(profile(occupation="2512"), published_occupation_label="Software Developers",
                    published_industry_code="J62")
    resolver.decide(profile(occupation="9999", region=None))
    assert resolver.policy_audit() == before


def test_cli_immutable_policy_report_does_not_claim_release(capsys) -> None:
    assert main([
        "--root", str(ROOT), "--as-of", "2026-10-21",
        "--psoc", "2512", "--write-report",
    ]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["result"]["mode"] == "UNAVAILABLE"
    assert output["policy_audit_report"]["status"] == "PASS_FAIL_CLOSED_ROUTING_ONLY"
    assert output["result"]["data_as_of"] is None
