from __future__ import annotations

import copy
import hashlib
import json
from datetime import date
from pathlib import Path

import pytest

from salary_predictor.trainer.compare import main as compare_main
from salary_predictor.trainer.experiment import (
    _read_plan,
    immutable_report,
    readiness_report,
)
from salary_predictor.trainer.metrics import (
    absolute_error_metrics,
    subgroup_error_metrics,
)

ROOT = Path(__file__).resolve().parents[2]
AS_OF = date(2026, 10, 21)


@pytest.fixture(scope="module")
def report() -> dict:
    return readiness_report(ROOT, as_of=AS_OF)


def test_part08_preflight_has_actual_source_counts_and_no_training(report: dict) -> None:
    assert report["part"] == "08"
    assert report["status"] == "BLOCKED_NO_APPROVED_COMPARABLE_LABELED_HOLDOUT"
    assert report["source_integrity"] == "PASS_RAW_AND_STAGE_SHA_CHECKS"
    assert report["raw_files_verified"] == 37
    assert report["benchmark_published_cells"] == 560
    assert report["cohort_inspection"]["STEP"]["observed_hourly_wage_worker_candidates"] == 987
    jobicy = report["cohort_inspection"]["Jobicy"]
    assert jobicy["unique_remote_job_listings"] == 989
    assert jobicy["valid_advertised_salary_ranges"] == 513
    assert jobicy["php_denominated_advertised_ranges"] == 1
    assert jobicy["apac_eligible_disclosed_ranges"] == 9
    assert report["eligible_supervised_training_rows"] == 0
    assert report["train_count"] == 0
    assert report["validation_count"] == 0
    assert report["test_count"] == 0
    assert report["frozen_group_time_holdout"] is None


def test_candidates_are_specified_but_no_actual_error_claims(report: dict) -> None:
    candidates = report["candidates"]
    assert {x["id"] for x in candidates} == {"ridge_one_hot", "random_forest"}
    for item in candidates:
        assert item["status"].startswith("NOT_RUN")
        assert item["mae_php"] is None
        assert item["per_occupation_error"] is None
        assert item["per_region_error"] is None
        assert item["selected_hyperparameters"] is None
    assert report["model_trained"] is False
    assert report["model_artifact"] is None
    assert report["holdout_mae_php"] is None
    assert report["selection_decision"] == "NO_MODEL_SELECTED_RETAIN_PART07_BENCHMARK_ONLY"
    assert report["skill_salary_premium"] is None
    assert report["geo_weighting_applied"] is False
    assert report["runtime_promotion"] == "BLOCKED"


def test_candidate_schema_requires_only_real_profile_features() -> None:
    config = _read_plan(ROOT)
    assert config["comparison"]["test_set"] == "immutable_untouched_final_only"
    assert config["comparison"]["hyperparameter_selection"] == "training_validation_only"
    assert config["comparison"]["fit_transformers_on"] == "training_only"
    assert config["never_mix_cohorts"] is True
    assert "normalized_skills" not in config["candidate_features"]
    assert config["regional_priority"]["use_in_training"] is False
    assert "handle_unknown=ignore" in config["comparison"]["candidates"][0]["pipeline"][2]
    assert config["comparison"]["fixed_seed"] == 42


@pytest.mark.parametrize(
    ("alter", "expected"),
    [
        ({"training_enabled": True}, "not an approved blocked-only plan"),
        ({"never_mix_cohorts": False}, "not an approved blocked-only plan"),
        ({"candidate_features": ["earnings_h"]}, "not an approved blocked-only plan"),
    ],
)
def test_changed_experiment_plan_fails_closed(
    tmp_path: Path, alter: dict, expected: str,
) -> None:
    dest = tmp_path / "configs/experiments"
    dest.mkdir(parents=True)
    contents = json.loads(
        (ROOT / "configs/experiments/part08_comparison.json").read_text()
    )
    contents.update(alter)
    (dest / "part08_comparison.json").write_text(json.dumps(contents))
    with pytest.raises(ValueError, match=expected):
        _read_plan(tmp_path)


def test_candidate_change_requires_new_review(tmp_path: Path) -> None:
    dest = tmp_path / "configs/experiments"
    dest.mkdir(parents=True)
    contents = json.loads(
        (ROOT / "configs/experiments/part08_comparison.json").read_text()
    )
    contents["comparison"]["candidates"][0]["status"] = "FIT"
    (dest / "part08_comparison.json").write_text(json.dumps(contents))
    with pytest.raises(ValueError, match="cannot claim fitting"):
        _read_plan(tmp_path)


def test_source_promotion_cannot_bypass_read_only_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import salary_predictor.trainer.experiment as experiment
    from salary_predictor.source_registry import load_source_register

    original = load_source_register(ROOT / "configs/data_sources.json")
    changed = copy.deepcopy(original)
    for source in changed["sources"]:
        if source["id"] == "jobicy_public_remote_jobs":
            source["enabled_for_import"] = True
            source["approved_use"].append("train_on_labeled_salary")
    monkeypatch.setattr(experiment, "load_source_register", lambda path: changed)
    with pytest.raises(ValueError, match="(Source register invalid|cannot accept promoted)"):
        readiness_report(ROOT, as_of=AS_OF)


def test_source_checksum_failure_prevents_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import salary_predictor.trainer.experiment as experiment

    monkeypatch.setattr(experiment, "verify_raw_files", lambda root, manifest: [
        "Checksum changed: data/raw/step/step_ph_weights_merged_public.csv"
    ])
    with pytest.raises(ValueError, match="Raw source integrity failed"):
        readiness_report(ROOT, as_of=AS_OF)


def test_metric_arithmetic_on_fictional_inputs_only() -> None:
    result = absolute_error_metrics(
        ["100", "200", "300", "400"], ["90", "220", "280", "450"],
    )
    assert result["count"] == 4
    assert result["mae_php"] == "25.00"
    assert result["median_absolute_error_php"] == "20.00"
    assert result["absolute_error_q90_php"] == "41.00"
    assert result["absolute_error_q95_php"] == "45.50"
    assert absolute_error_metrics([100], [-5])["mae_php"] == "105.00"


@pytest.mark.parametrize(
    ("actual", "predicted"),
    [
        ([], []),
        ([100], []),
        (["NaN"], [100]),
        (["Infinity"], [100]),
        ([-1], [100]),
        ([100], ["NaN"]),
        ([100], ["Infinity"]),
    ],
)
def test_metrics_reject_missing_or_invalid_salary_pairs(
    actual: list, predicted: list,
) -> None:
    with pytest.raises(ValueError):
        absolute_error_metrics(actual, predicted)


def test_subgroups_suppress_insufficient_support_on_fixture() -> None:
    summary = subgroup_error_metrics(
        [100, 200, 300, 400],
        [90, 220, 280, 450],
        ["A", "A", "B", None],
        min_support=2,
    )
    assert summary["reported"]["A"]["count"] == 2
    assert summary["reported"]["A"]["mae_php"] == "15.00"
    assert summary["suppressed_group_counts"] == {"B": 1}
    assert summary["missing_group_count"] == 1


def test_content_addressed_preflight_is_repeatable(
    tmp_path: Path, report: dict,
) -> None:
    first = immutable_report(tmp_path, report)
    second = immutable_report(tmp_path, report)
    assert first == second
    path = tmp_path / first["path"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == first["sha256"]
    assert json.loads(path.read_text()) == report
    assert report == readiness_report(ROOT, as_of=AS_OF)


def test_cli_does_not_claim_metrics_or_fit(capsys: pytest.CaptureFixture[str]) -> None:
    result = compare_main(["--root", str(ROOT), "--as-of", "2026-10-21"])
    assert result == 0
    out = json.loads(capsys.readouterr().out)
    assert out["approved_training_rows"] == 0
    assert out["model_trained"] is False
    assert out["mae_php"] is None
    assert out["benchmark_path"] == "PART_07_SOURCE_PREVIEW_REMAINS_AVAILABLE"
