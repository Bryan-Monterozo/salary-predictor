from __future__ import annotations

import copy
import hashlib
import json
from datetime import date
from pathlib import Path

import pytest

from salary_predictor.taxonomy import Taxonomy
from salary_predictor.trainer.baseline import BenchmarkOnlyBaseline
from salary_predictor.trainer.skill_ablation import (
    ensure_same_ablation_partition,
    immutable_report,
    load_skill_plan,
    readiness,
)
from salary_predictor.trainer.skill_compare import main
from salary_predictor.trainer.skills import binary_skill_features, encode_explicit_skills

ROOT = Path(__file__).resolve().parents[2]
AS_OF = date(2026, 10, 21)
PYTHON_URI = "http://data.europa.eu/esco/skill/ccd0a1d9-afda-43d9-b901-96344886e14d"


@pytest.fixture(scope="module")
def taxonomy() -> Taxonomy:
    return Taxonomy(ROOT)


@pytest.fixture(scope="module")
def evidence() -> dict:
    return readiness(ROOT, as_of=AS_OF)


def test_readiness_is_blocked_with_true_verified_source_counts(evidence: dict) -> None:
    assert evidence["part"] == "09"
    assert evidence["status"] == "DEFERRED_NO_PAIRED_OBSERVED_SKILLS_AND_APPROVED_SALARIES"
    assert evidence["source_cohorts"] == {
        "STEP_hourly_wage_worker_candidates": 987,
        "Jobicy_global_advertised_salary_ranges": 513,
        "PSA_OWS_published_aggregate_cells": 560,
    }
    assert evidence["raw_files_verified"] == 37
    assert evidence["approved_salary_plus_observed_skill_training_rows"] == 0
    assert evidence["skill_salary_feature_enabled"] is False
    assert evidence["skill_search_and_extraction_enabled"] is True
    assert evidence["with_skills_model"] == evidence["without_skills_model"] == "NOT_RUN"
    assert evidence["delta_mae_php"] is None
    assert evidence["skill_premiums_php"] is None
    assert evidence["shared_frozen_partition"] is None
    assert evidence["feature_importance"] is None
    assert evidence["runtime_recommendation"].endswith("RUNTIME_UNAVAILABLE")


def test_skill_exact_alias_uri_and_duplicate_synonyms_merge(taxonomy: Taxonomy) -> None:
    checked = encode_explicit_skills(
        taxonomy,
        ["Python", " Python (computer programming) ", PYTHON_URI],
        origin="user_explicit",
    )
    assert checked.observed is True
    assert checked.matched_uris == (PYTHON_URI,)
    assert checked.duplicate_matches == 2
    assert checked.ambiguous_text == checked.unresolved_text == ()
    assert checked.to_dict()["salary_feature_enabled"] is False


def test_unknown_and_ambiguous_are_not_auto_assigned(taxonomy: Taxonomy) -> None:
    checked = encode_explicit_skills(
        taxonomy, ["SQL", "made up unseen skill", "Python"],
        origin="resume_explicit_reviewed",
    )
    assert checked.matched_uris == (PYTHON_URI,)
    assert checked.ambiguous_text == ("SQL",)
    assert checked.unresolved_text == ("made up unseen skill",)
    assert checked.duplicate_matches == 0


def test_observed_absent_distinct_from_reviewed_empty(taxonomy: Taxonomy) -> None:
    unknown = encode_explicit_skills(taxonomy, None, origin="user_explicit")
    empty = encode_explicit_skills(taxonomy, [], origin="user_explicit")
    vocab = (PYTHON_URI,)
    assert unknown.observed is False
    assert empty.observed is True
    assert binary_skill_features(unknown, frozen_training_vocabulary=vocab) == {
        PYTHON_URI: None
    }
    assert binary_skill_features(empty, frozen_training_vocabulary=vocab) == {
        PYTHON_URI: 0
    }
    assert binary_skill_features(
        encode_explicit_skills(taxonomy, ["Python"], origin="user_explicit"),
        frozen_training_vocabulary=vocab,
    ) == {PYTHON_URI: 1}


@pytest.mark.parametrize(
    "origin",
    [
        "occupation_skill_relation",
        "occupation_required_or_optional",
        "taxonomy_suggestion",
        "job_title_inferred",
        "resume_guess",
    ],
)
def test_occupation_linked_skills_cannot_be_masqueraded_as_observed(
    taxonomy: Taxonomy, origin: str,
) -> None:
    with pytest.raises(ValueError, match="Only explicit"):
        encode_explicit_skills(taxonomy, ["Python"], origin=origin)


@pytest.mark.parametrize(
    "values",
    [123, ["Python", 12], ["Python"] * 101, ["x" * 301]],
)
def test_invalid_skill_observations_are_rejected(taxonomy: Taxonomy, values: object) -> None:
    with pytest.raises(ValueError, match="Invalid observed"):
        encode_explicit_skills(taxonomy, values, origin="user_explicit")


def test_unseen_skills_do_not_modify_frozen_feature_columns(taxonomy: Taxonomy) -> None:
    known = encode_explicit_skills(taxonomy, ["Python"], origin="user_explicit")
    more = encode_explicit_skills(
        taxonomy, ["Python", "unknown custom emerging skill"],
        origin="user_explicit",
    )
    vocab = (PYTHON_URI,)
    assert binary_skill_features(known, frozen_training_vocabulary=vocab) == (
        binary_skill_features(more, frozen_training_vocabulary=vocab)
    )
    with pytest.raises(ValueError, match="sorted and unique"):
        binary_skill_features(known, frozen_training_vocabulary=vocab * 2)


def _fixture_ablation() -> dict:
    return {
        "source_snapshot_sha256": "synthetic-fixture-sha",
        "cohort_id": "fictional-monthly-pay",
        "target_definition": "approved_monthly_PHP_salary",
        "label_fingerprint": "synthetic-identical-label-hash",
        "cutoff_month": "2026-01",
        "partition_fingerprint": "synthetic-split-hash",
        "train_ids": ["train1", "train2"],
        "validation_ids": ["validate1"],
        "test_ids": ["test1", "test2"],
    }


def test_ablation_partition_contract_requires_identical_observations() -> None:
    clean = _fixture_ablation()
    ensure_same_ablation_partition(clean, copy.deepcopy(clean))
    for key, changed_value in (
        ("test_ids", ["different"]),
        ("label_fingerprint", "different-labels"),
        ("source_snapshot_sha256", "another-source"),
        ("partition_fingerprint", "different-split"),
        ("target_definition", "advertised_hourly_USD"),
        ("cohort_id", "another-cohort"),
    ):
        mismatch = copy.deepcopy(clean)
        mismatch[key] = changed_value
        with pytest.raises(ValueError):
            ensure_same_ablation_partition(clean, mismatch)


def test_ablation_partition_rejects_overlapping_or_missing_data() -> None:
    clean = _fixture_ablation()
    overlap = copy.deepcopy(clean)
    overlap["test_ids"] = ["train1"]
    with pytest.raises(ValueError, match="Overlapping"):
        ensure_same_ablation_partition(overlap, copy.deepcopy(overlap))
    empty = copy.deepcopy(clean)
    empty["validation_ids"] = []
    with pytest.raises(ValueError, match="Invalid"):
        ensure_same_ablation_partition(empty, copy.deepcopy(empty))
    missing = copy.deepcopy(clean)
    del missing["label_fingerprint"]
    with pytest.raises(ValueError, match="Missing frozen"):
        ensure_same_ablation_partition(missing, copy.deepcopy(missing))


def test_plan_is_disabled_and_cannot_auto_promote(tmp_path: Path) -> None:
    plan = load_skill_plan(ROOT)
    assert plan["skills_enabled_for_salary_inference"] is False
    assert plan["salary_training_enabled"] is False
    assert plan["release"]["numeric_skill_premium_allowed"] is False
    assert plan["encoder"]["top_k_vocabulary_fit"] == "train_only_after_permissioned_holdout"
    directory = tmp_path / "configs/experiments"
    directory.mkdir(parents=True)
    for field, value in (
        ("skills_enabled_for_salary_inference", True),
        ("salary_training_enabled", True),
        ("status", "APPROVED"),
    ):
        updated = copy.deepcopy(plan)
        updated[field] = value
        (directory / "part09_skill_ablation.json").write_text(json.dumps(updated))
        with pytest.raises(ValueError, match="not approved"):
            load_skill_plan(tmp_path)


def test_disabled_skills_leave_part07_baseline_identical(taxonomy: Taxonomy) -> None:
    baseline = BenchmarkOnlyBaseline(ROOT, as_of=AS_OF)
    before = baseline.runtime_result("Software Developers", industry_code="J62")
    before_source = baseline.lookup("Software Developers", industry_code="J62")
    _ = encode_explicit_skills(
        taxonomy, ["Python", "Javascript"],
        origin="user_explicit",
    )
    after = baseline.runtime_result("Software Developers", industry_code="J62")
    after_source = baseline.lookup("Software Developers", industry_code="J62")
    assert before == after
    assert before_source == after_source
    assert after.mode == "UNAVAILABLE"


def test_immutable_report_repeats_and_no_model_scores(tmp_path: Path, evidence: dict) -> None:
    first = immutable_report(tmp_path, evidence)
    second = immutable_report(tmp_path, evidence)
    assert first == second
    target = tmp_path / first["path"]
    assert hashlib.sha256(target.read_bytes()).hexdigest() == first["sha256"]
    assert json.loads(target.read_text()) == evidence


def test_cli_can_preview_explicit_skill_without_salary_effect(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main([
        "--root", str(ROOT), "--as-of", "2026-10-21",
        "--skill", "Python", "--skill", "SQL",
    ]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "DEFERRED_NO_PAIRED_OBSERVED_SKILLS_AND_APPROVED_SALARIES"
    assert result["mae_delta_php"] is None
    assert result["explicit_skill_preview"]["matched_uris"] == [PYTHON_URI]
    assert result["explicit_skill_preview"]["ambiguous_text"] == ["SQL"]
    assert result["explicit_skill_preview"]["salary_feature_enabled"] is False
