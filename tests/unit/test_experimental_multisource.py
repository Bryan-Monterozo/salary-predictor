from __future__ import annotations

import shutil
from datetime import date
from pathlib import Path

import pytest

from salary_predictor.feasibility.consolidated import consolidate
from salary_predictor.trainer.experimental_multisource import (
    _offered_monthly_usd,
    _partition,
    _training_records,
    train_experiment,
    verify_experiment,
)
from salary_predictor.trainer.experimental_predict import predict_experimental

ROOT = Path(__file__).resolve().parents[2]
AS_OF = date(2026, 10, 22)


@pytest.fixture(scope="module")
def trained() -> dict:
    return train_experiment(ROOT, as_of=AS_OF)


def test_salary_normalization_preserves_source_currency_and_assumptions() -> None:
    item = {"salary": {
        "amount_min": "120000", "amount_max": "120000",
        "currency": "USD", "period": "yearly",
    }}
    assert _offered_monthly_usd(item) == 10000
    item["salary"]["period"] = "monthly"
    assert _offered_monthly_usd(item) == 120000
    item["salary"]["period"] = "hourly"
    item["salary"]["amount_min"] = "50"
    item["salary"]["amount_max"] = "50"
    assert round(_offered_monthly_usd(item), 2) == 8666.67
    item["salary"]["currency"] = "CAD"
    assert _offered_monthly_usd(item) is None


def test_raw_consolidation_and_title_group_holdout_have_no_salary_fabrication() -> None:
    summary, rows, unresolved = consolidate(ROOT, as_of=AS_OF)
    paid = _training_records(rows)
    assert summary["unique_job_postings"] == 989
    assert summary["valid_disclosed_salary_ranges"] == 513
    assert len(unresolved) == 476
    assert len(paid) == 393
    assert all(x["target_usd_monthly"] > 0 for x in paid)
    split = _partition(paid)
    title_keys = {
        name: {paid[i]["title_group"] for i in ids} for name, ids in split.items()
    }
    assert not title_keys["train"] & title_keys["validation"]
    assert not title_keys["train"] & title_keys["test"]
    assert not title_keys["validation"] & title_keys["test"]


def test_saved_multisource_experiment_is_local_only(trained: dict) -> None:
    assert trained["run_id"].startswith("exp-")
    assert trained["training_rows"] == 393
    assert trained["production_release_approved"] is False
    report, folder = verify_experiment(ROOT, trained["run_id"])
    assert folder.is_dir()
    assert report["source_id"] == "jobicy_public_remote_jobs"
    assert report["psa_benchmarks_joined_to_ad_training"] is False
    assert report["step_hourly_earnings_joined_to_ad_training"] is False
    assert report["test_selected"]["count"] >= 20
    assert report["test_selected"]["philippines_eligible_count"] == 0
    assert report["status"] == "LOCAL_EXPERIMENT_NOT_APPROVED_FOR_DEPLOYMENT"


def test_full_prediction_returns_real_components_and_transparent_scenario(trained: dict) -> None:
    result = predict_experimental(
        ROOT, job_title="Software Developer",
        skills=["C#", "SQL"], years_experience=4.0,
        current_salary_php=45000, usd_to_php=58,
        isco_major="2", step_region_code="13", education_years=16,
        psa_occupation="Software Developers", psa_industry="J62",
        scenario_annual_growth=.03, run_id=trained["run_id"], as_of=AS_OF,
    )
    assert result["mode"] == "EXPERIMENTAL_LOCAL_ONLY"
    assert result["estimated_monthly_php_scenario"] > 0
    assert result["model_artifact"] == trained["run_id"]
    assert set(result["sources"]) == {
        "jobicy_remote_offer", "step_historical", "psa_published_2022"
    }
    assert result["sources"]["psa_published_2022"]["monthly_php_scenario"] == 45520
    assert result["sources"]["jobicy_remote_offer"]["rate_assumption_php_per_usd"] == 58
    assert result["forecast"]["method"].startswith("user_chosen")
    assert result["blend_weights_illustrative_not_learned"] == {
        "psa_published_2022": .65, "step_historical": .3, "jobicy_remote_offer": .05
    }
    assert result["current_salary_php_reference"] == 45000
    assert result["skill_salary_effect_not_validated"] is True
    assert result["esco_occupation_skill_gap_ideas"]
    assert result["calibrated_php_prediction_interval"] is None
    assert result["model_release_approved"] is False


def test_without_fx_or_domestic_evidence_never_invents_php(trained: dict) -> None:
    result = predict_experimental(
        ROOT, job_title="Senior Developer", skills=[], years_experience=None,
        current_salary_php=None, usd_to_php=None,
        run_id=trained["run_id"], as_of=AS_OF,
    )
    assert result["sources"]["jobicy_remote_offer"]["monthly_usd"] > 0
    assert result["estimated_monthly_php_scenario"] is None
    assert result["forecast"] is None


@pytest.mark.parametrize("bad", [
    {"usd_to_php": -2}, {"years_experience": 90}, {"current_salary_php": -1},
    {"education_years": 60}, {"isco_major": "2512"}, {"seniority": "CEO"},
])
def test_unreasonable_profile_is_rejected_before_loading_model(bad: dict) -> None:
    params = dict(
        job_title="Software Developer", skills=[], years_experience=None,
        current_salary_php=None, usd_to_php=None,
        as_of=AS_OF, run_id="exp-00000000000000000000",
    )
    params.update(bad)
    with pytest.raises(ValueError):
        predict_experimental(ROOT, **params)


def test_model_tampering_never_loads_an_unverified_pickle(tmp_path: Path, trained: dict) -> None:
    source = ROOT / trained["artifact"]
    target = tmp_path / "data/artifacts/experimental" / trained["run_id"]
    target.mkdir(parents=True)
    for name in ("manifest.json", "model.joblib", "evaluation.json"):
        shutil.copy2(source / name, target / name)
    config = tmp_path / "configs/feasibility"
    config.mkdir(parents=True)
    for src, dst in (
        (ROOT / "configs/local_raw_file_manifest.json", tmp_path / "configs/local_raw_file_manifest.json"),
        (ROOT / "configs/feasibility/consolidation_policy.json", config / "consolidation_policy.json"),
    ):
        shutil.copy2(src, dst)
    assert verify_experiment(tmp_path, trained["run_id"])[0]["selected"]
    (target / "model.joblib").write_bytes(b"malicious-not-loaded")
    with pytest.raises(ValueError, match="do not match"):
        verify_experiment(tmp_path, trained["run_id"])


def test_production_salary_release_stays_separate(trained: dict) -> None:
    assert not (ROOT / "data/artifacts/active.json").exists()
    evaluation, _ = verify_experiment(ROOT, trained["run_id"])
    assert evaluation["production_release_approved"] is False
    assert evaluation["rights_review_required_before_distribution"] is True
