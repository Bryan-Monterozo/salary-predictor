from __future__ import annotations

import hashlib
import json
import shutil
from datetime import date
from pathlib import Path

import pytest

from salary_predictor.trainer import release
from salary_predictor.trainer.operator_cli import main as operator_main

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def isolated_root(tmp_path: Path) -> Path:
    cfg = tmp_path / "configs"
    cfg.mkdir()
    for name in ("data_sources.json", "release_policy.yaml",
                 "result_decision_v1.json", "local_raw_file_manifest.json"):
        shutil.copy2(ROOT / "configs" / name, cfg / name)
    tax = cfg / "taxonomy"
    tax.mkdir()
    for name in ("psoc_reviewed_v1.json", "psgc_regions_v1.json",
                 "esco_psoc_crosswalk_v1.json"):
        shutil.copy2(ROOT / "configs/taxonomy" / name, tax / name)
    staged = tmp_path / "data/staged"
    staged.mkdir(parents=True)
    stem = "psa_ows_2022_public_tables-25a47e4e13d22b33"
    for ext in (".jsonl", ".lineage.json", ".report.json"):
        shutil.copy2(ROOT / "data/staged" / (stem + ext), staged / (stem + ext))
    return tmp_path


@pytest.fixture
def candidate(isolated_root: Path) -> tuple[Path, dict]:
    return isolated_root, release.build_benchmark_candidate(
        isolated_root, as_of=date(2026, 10, 21)
    )


def test_benchmark_candidate_integrity(candidate) -> None:
    root, result = candidate
    assert result["status"] == "CANDIDATE_NOT_APPROVED"
    assert result["published_cells"] == 560
    assert result["model_trained"] is False
    assert not result["runtime_approved"]
    manifest = release.verify_candidate(root, result["run_id"])
    assert manifest["training_records"] == 0
    assert manifest["model_file"] is None
    assert manifest["feature_schema_version"] == 1
    assert manifest["contract_version"] == 1
    assert manifest["heldout_mae_php"] is None
    folder = root / "data/artifacts/candidates" / result["run_id"]
    assert sorted(p.name for p in folder.iterdir()) == ["benchmark.json", "manifest.json"]


def test_reproducible_candidate_does_not_overwrite(candidate) -> None:
    root, result = candidate
    assert release.build_benchmark_candidate(root, as_of=date(2026, 10, 21)) == result
    assert hashlib.sha256((root / result["manifest_path"]).read_bytes()).hexdigest() == (
        result["manifest_sha256"]
    )
    other = release.build_benchmark_candidate(root, as_of=date(2026, 11, 21))
    assert other["run_id"] != result["run_id"]


@pytest.mark.parametrize("changed", ("benchmark", "schema", "rights", "taxonomy"))
def test_corrupt_or_incompatible_artifact_is_rejected(candidate, changed: str) -> None:
    root, result = candidate
    folder = root / "data/artifacts/candidates" / result["run_id"]
    if changed == "benchmark":
        (folder / "benchmark.json").write_text('{"invalid":true}')
    elif changed == "schema":
        path = folder / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["feature_schema_version"] = 2
        path.write_text(json.dumps(manifest))
    elif changed == "rights":
        (root / "configs/release_policy.yaml").write_text('{"changed":true}')
    else:
        (root / "configs/taxonomy/psoc_reviewed_v1.json").write_text('{"changed":true}')
    with pytest.raises((ValueError, KeyError)):
        release.verify_candidate(root, result["run_id"])


def test_training_gate_reports_no_labels(capsys: pytest.CaptureFixture[str]) -> None:
    for command in ("train", "evaluate"):
        assert operator_main([
            command, "--root", str(ROOT), "--as-of", "2026-10-21"
        ]) == 2
        result = json.loads(capsys.readouterr().out)
        assert result["eligible_training_rows"] == 0
        assert result["status"] == "BLOCKED_NO_APPROVED_COMPARABLE_LABELED_HOLDOUT"
        assert result["requested_operation"] == command


def test_real_promotion_is_blocked(candidate) -> None:
    root, result = candidate
    blockers = release.promotion_blockers(
        root, result["run_id"], today=date(2026, 10, 21)
    )
    assert any("age" in issue for issue in blockers)
    assert any("policy" in issue.lower() for issue in blockers)
    with pytest.raises(ValueError, match="confirm"):
        release.promote(root, result["run_id"], confirm=False)
    with pytest.raises(ValueError, match="Promotion blocked"):
        release.promote(root, result["run_id"], confirm=True)
    assert release.active_release(root)["status"] == "NO_APPROVED_ACTIVE_RELEASE"



def _approve_fixture(root: Path, item: dict) -> None:
    folder = root / "data/artifacts/approvals"
    folder.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": 1,
        "release_id": item["run_id"],
        "approved_for": "published_benchmark_only",
        "operator_reviewed": True,
        "source_rights_reviewed": True,
        "tests_passed_reviewed": True,
        "release_policy_reviewed": True,
        "manifest_sha256": item["manifest_sha256"],
        "operator_id": "fixture-reviewer",
        "approved_at_utc": "2023-01-01T00:00:00+00:00",
    }
    (folder / (item["run_id"] + ".json")).write_text(json.dumps(record))


def test_isolated_approved_fixture_promotion_and_rollback(
    isolated_root: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = isolated_root
    release_policy = root / "configs/release_policy.yaml"
    policy = json.loads(release_policy.read_text())
    policy["status"] = "APPROVED_FOR_RUNTIME"
    release_policy.write_text(json.dumps(policy))
    route_policy = root / "configs/result_decision_v1.json"
    config = json.loads(route_policy.read_text())
    config["runtime_release_enabled"] = True
    config["benchmark_release_enabled"] = True
    route_policy.write_text(json.dumps(config))
    first = release.build_benchmark_candidate(root, as_of=date(2023, 1, 21))
    second = release.build_benchmark_candidate(root, as_of=date(2023, 2, 21))
    _approve_fixture(root, first)
    _approve_fixture(root, second)

    class TestDate(date):
        @classmethod
        def today(cls):
            return date(2023, 3, 1)

    monkeypatch.setattr(release, "date", TestDate)
    assert release.promotion_blockers(
        root, first["run_id"], today=date(2023, 3, 1)
    ) == []
    assert release.promote(root, first["run_id"], confirm=True)["status"] == (
        "ACTIVE_BENCHMARK_ONLY"
    )
    assert release.active_release(root)["release_id"] == first["run_id"]
    release.promote(root, second["run_id"], confirm=True)
    assert release.active_release(root)["release_id"] == second["run_id"]
    rolled = release.rollback(root, first["run_id"], confirm=True)
    assert rolled["previous_release_id"] == second["run_id"]
    assert release.active_release(root)["release_id"] == first["run_id"]
    assert release.active_release(root)["runtime_prediction_enabled"] is False

    approval = root / "data/artifacts/approvals" / (first["run_id"] + ".json")
    data = json.loads(approval.read_text())
    data["operator_reviewed"] = False
    approval.write_text(json.dumps(data))
    assert release.active_release(root)["status"] == "BLOCKED_INVALID_ACTIVE_RELEASE"


def test_unsafe_release_ids_and_broken_benchmark_symlink(candidate) -> None:
    root, info = candidate
    for bad in ("../escape", "anything", "bench-xyz"):
        with pytest.raises(ValueError):
            release.verify_candidate(root, bad)
    file = root / info["bundle_path"]
    file.unlink()
    file.symlink_to(root / "configs/result_decision_v1.json")
    with pytest.raises(ValueError, match="unsafe"):
        release.verify_candidate(root, info["run_id"])



def test_cli_help_and_legacy_dispatch_do_not_fit_models(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from salary_predictor.trainer.__main__ import dispatch

    with pytest.raises(SystemExit) as help_status:
        dispatch(["--help"])
    assert help_status.value.code == 0
    usage = capsys.readouterr().out
    for command in (
        "sources", "ingest", "audit", "train", "evaluate", "prepare",
        "verify", "promote", "rollback", "active",
    ):
        assert command in usage
    assert dispatch(["--as-of", "2026-10-21"]) == 0
    legacy = json.loads(capsys.readouterr().out)
    assert legacy["model_trained"] is False
    assert legacy["published_cells"] == 560


def test_modified_benchmark_cannot_be_forged_by_updating_manifest_hash(candidate) -> None:
    root, result = candidate
    folder = root / "data/artifacts/candidates" / result["run_id"]
    benchmark_path = folder / "benchmark.json"
    manifest_path = folder / "manifest.json"
    data = json.loads(benchmark_path.read_text())
    data["records"][0]["amount_php_monthly"] = "9999999"
    body = release._json_bytes(data)
    benchmark_path.write_bytes(body)
    manifest = json.loads(manifest_path.read_text())
    manifest["benchmark_sha256"] = hashlib.sha256(body).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="original source records"):
        release.verify_candidate(root, result["run_id"])


def test_inference_web_module_import_has_no_trainer_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib

    monkeypatch.setattr(
        release, "training_gate",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("training called")),
    )
    imported = importlib.import_module("salary_predictor.web")
    assert imported.__name__ == "salary_predictor.web"
