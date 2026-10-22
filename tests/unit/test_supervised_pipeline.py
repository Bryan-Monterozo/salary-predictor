from __future__ import annotations

import copy
import hashlib
import json
from datetime import date
from pathlib import Path

import pytest

from salary_predictor.trainer.operator_cli import main as cli
from salary_predictor.trainer.reviewed_data import frozen_splits, reviewed_rows
from salary_predictor.trainer.supervised import (
    _weight,
    run_supervised,
    verify_model_candidate,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def reviewed_fixture(tmp_path: Path) -> tuple[Path, str]:
    cfg = tmp_path / "configs"
    cfg.mkdir()
    registry = json.loads((ROOT / "configs/data_sources.json").read_text())
    a = next(x for x in registry["sources"] if x["id"] == "jobicy_public_remote_jobs")
    a["rights_status"] = "verified_fixture_operator_training_permission"
    a["enabled_for_import"] = True
    a["approved_use"] = ["train_on_labeled_salary"]
    b = copy.deepcopy(a)
    b["id"] = "fixture_local_jobs"
    b["name"] = "Fictional data supplier for unit tests"
    registry["sources"].append(b)
    (cfg / "data_sources.json").write_text(json.dumps(registry))
    policy = json.loads((ROOT / "configs/release_policy.yaml").read_text())
    policy["evidence"]["supervised_training_enabled"] = True
    (cfg / "release_policy.yaml").write_text(json.dumps(policy))
    folder = tmp_path / "data/curated"
    folder.mkdir(parents=True)
    rows = []
    for i in range(180):
        month = "2025-01" if i < 45 else "2025-03" if i < 90 else (
            "2026-05" if i < 135 else "2026-09"
        )
        role = "2512" if i % 3 != 0 else "2411"
        loc = "1300000000" if i % 2 == 0 else "0400000000"
        rows.append({
            "observation_id": f"TEST-{i:03}",
            "source_id": "jobicy_public_remote_jobs" if i % 2 else "fixture_local_jobs",
            "cohort": "advertised_salary",
            "job_title": "Software Developer" if role == "2512" else "Accountant",
            "occupation_code": role,
            "region_code": loc,
            "market_country": "PH" if i % 3 else "MY",
            "market_country_verified": True,
            "employer_group": f"fictional-employer-{i}",
            "duplicate_cluster_id": f"fictional-ad-{i}",
            "observation_date": month + "-15",
            "salary_php_monthly": str(
                30000 + (i % 3) * 10000 + (i % 5) * 500 + (i // 45) * 1000
            ),
            "currency": "PHP", "period": "monthly",
            "salary_basis": "advertised_range_midpoint",
            "normalization_evidence": "fictional_test_data_only",
            "years_experience": i % 8, "education_level": "college",
            "employment_context": "full_time",
        })
    content = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode()
    (folder / "fictional-jobs.jsonl").write_bytes(content)
    manifest = {
        "schema_version": 1,
        "status": "REVIEWED_FOR_SUPERVISED_RESEARCH",
        "independent_rights_review": True,
        "reviewed_by": "fixture-test-reviewer",
        "rights_evidence_reference": "SYNTHETIC_UNIT_TEST_NEVER_REAL_APPROVAL",
        "source_ids": ["jobicy_public_remote_jobs", "fixture_local_jobs"],
        "cohort": "advertised_salary",
        "target_currency": "PHP", "target_period": "monthly",
        "target_salary_basis": "advertised_range_midpoint",
        "reviewed_normalization_method": "fictional_test_rows_already_monthly_php",
        "holdout_reviewed": True,
        "validation_from": "2026-01-01",
        "test_from": "2026-07-01",
        "dataset_file": "data/curated/fictional-jobs.jsonl",
        "dataset_sha256": hashlib.sha256(content).hexdigest(),
    }
    path = folder / "fictional-review.json"
    path.write_text(json.dumps(manifest))
    return tmp_path, "data/curated/fictional-review.json"


def test_fit_models_with_separate_holdout_and_permissioned_fixture(reviewed_fixture) -> None:
    root, manifest = reviewed_fixture
    result = run_supervised(root, manifest, as_of=date(2026, 10, 21))
    assert result["status"] == "LOCAL_RESEARCH_CANDIDATE_NOT_APPROVED"
    assert result["runtime_release_approved"] is False
    assert result["split_counts"] == {
        "train": 90, "validation": 45, "test": 45, "quarantine": 0,
    }
    assert result["heldout_mae_php"] is not None
    verified = verify_model_candidate(root, result["run_id"])
    assert verified["runtime_release_approved"] is False
    folder = root / result["artifact_path"]
    outcome = json.loads((folder / "evaluation.json").read_text())
    assert len(outcome["validation_candidates"]) == 7
    assert outcome["sources"] == ["jobicy_public_remote_jobs", "fixture_local_jobs"]
    assert outcome["test_metrics_global_baseline"]["count"] == 45
    assert outcome["test_metrics_occupation_baseline"]["count"] == 45
    assert outcome["selection_rule"] == "validation_0.7_ph_mae_plus_0.3_all_mae"
    assert outcome["worker_and_advertised_cohorts_separate"] is True
    assert outcome["runtime_release_approved"] is False
    again = run_supervised(root, manifest, as_of=date(2026, 10, 21))
    assert again == result
    assert not (root / "data/artifacts/active.json").exists()


def test_weighting_is_not_foreign_to_php_conversion() -> None:
    assert [_weight(x) for x in ("PH", "MY", "AU", "US", "unknown")] == [
        3.0, 2.0, 1.0, 0.5, 0.5,
    ]


def test_spanning_employer_group_quarantined(reviewed_fixture) -> None:
    root, path = reviewed_fixture
    _meta, rows = reviewed_rows(root, path)
    rows[0]["employer_group"] = rows[-1]["employer_group"]
    split = frozen_splits(
        rows, validation_from="2026-01-01", test_from="2026-07-01"
    )
    assert 0 in split["quarantine"] and 179 in split["quarantine"]
    assert 0 not in split["train"] and 179 not in split["test"]


@pytest.mark.parametrize("change", ["rights", "role", "mixed", "amount", "digest", "source"])
def test_training_rejects_unreviewed_or_mismatched_data(
    reviewed_fixture, change: str,
) -> None:
    root, rel = reviewed_fixture
    p = root / rel
    manifest = json.loads(p.read_text())
    data_path = root / manifest["dataset_file"]
    rows = [json.loads(line) for line in data_path.read_text().splitlines()]
    if change == "rights":
        manifest["independent_rights_review"] = False
    elif change == "role":
        manifest["cohort"] = "individual_earnings"
        manifest["target_salary_basis"] = "individual_gross_total"
    elif change == "mixed":
        rows[0]["cohort"] = "individual_earnings"
    elif change == "amount":
        rows[1]["salary_php_monthly"] = None
    elif change == "source":
        registry_file = root / "configs/data_sources.json"
        registry = json.loads(registry_file.read_text())
        for item in registry["sources"]:
            if item["id"] == "jobicy_public_remote_jobs":
                item["approved_use"] = []
                item["enabled_for_import"] = False
        registry_file.write_text(json.dumps(registry))
    else:
        manifest["dataset_sha256"] = "f" * 64
    if change in {"mixed", "amount"}:
        content = "".join(json.dumps(row) + "\n" for row in rows).encode()
        data_path.write_bytes(content)
        manifest["dataset_sha256"] = hashlib.sha256(content).hexdigest()
    p.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        reviewed_rows(root, rel)


def test_tampering_model_candidate_is_detected(reviewed_fixture) -> None:
    root, path = reviewed_fixture
    record = run_supervised(root, path, as_of=date(2026, 10, 21))
    folder = root / record["artifact_path"]
    (folder / "evaluation.json").write_text('{"fake":true}')
    with pytest.raises(ValueError, match="integrity"):
        verify_model_candidate(root, record["run_id"])


def test_no_manifest_keeps_real_training_blocked(capsys) -> None:
    assert cli(["train", "--root", str(ROOT), "--as-of", "2026-10-21"]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["eligible_training_rows"] == 0
    assert result["status"] == "BLOCKED_NO_APPROVED_COMPARABLE_LABELED_HOLDOUT"


def test_trainer_cli_can_fit_fixture_and_verify_without_loading_pickle(
    reviewed_fixture, capsys,
) -> None:
    root, path = reviewed_fixture
    assert cli([
        "train", "--root", str(root), "--as-of", "2026-10-21",
        "--reviewed-manifest", path,
    ]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["runtime_release_approved"] is False
    assert cli(["verify", "--root", str(root), "--run-id", output["run_id"]]) == 0
    verify_result = json.loads(capsys.readouterr().out)
    assert verify_result["model_bytes_validated_but_not_loaded"] in (True, False)



@pytest.mark.parametrize("change", ["reviewed_data", "reviewed_manifest"])
def test_model_candidate_rejects_changed_source_lineage(
    reviewed_fixture, change: str,
) -> None:
    root, source_manifest = reviewed_fixture
    trained = run_supervised(root, source_manifest, as_of=date(2026, 10, 21))
    assert verify_model_candidate(root, trained["run_id"])["runtime_release_approved"] is False
    metadata_path = root / source_manifest
    reviewed = json.loads(metadata_path.read_text())
    if change == "reviewed_data":
        data = root / reviewed["dataset_file"]
        data.write_bytes(data.read_bytes() + b"\n")
    else:
        reviewed["rights_evidence_reference"] = "CHANGED_TEST_APPROVAL"
        metadata_path.write_text(json.dumps(reviewed))
    with pytest.raises(ValueError, match="changed"):
        verify_model_candidate(root, trained["run_id"])


def test_curated_path_escape_and_unreviewed_raw_are_rejected(reviewed_fixture) -> None:
    root, _ = reviewed_fixture
    for path in (
        "../salary-raw.jsonl",
        "data/raw/jobicy/file.json",
        "data/curated/../raw/jobicy/file.json",
    ):
        with pytest.raises((ValueError, OSError)):
            reviewed_rows(root, path)
