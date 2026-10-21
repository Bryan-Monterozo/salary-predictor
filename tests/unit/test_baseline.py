from __future__ import annotations

import hashlib
import json
import shutil
from datetime import date
from pathlib import Path

import pytest

from salary_predictor.contracts import BenchmarkObservation, PredictionResult
from salary_predictor.trainer.__main__ import main, write_report
from salary_predictor.trainer.baseline import (
    BENCHMARK_ID,
    CORRECTED_PSA_STAGED,
    CORRECTED_SHA256,
    BenchmarkOnlyBaseline,
)

ROOT = Path(__file__).resolve().parents[2]
AS_OF = date(2026, 10, 21)


@pytest.fixture(scope="module")
def baseline() -> BenchmarkOnlyBaseline:
    return BenchmarkOnlyBaseline(ROOT, as_of=AS_OF)


def test_reproducible_report_has_no_fabricated_training_metrics(
    baseline: BenchmarkOnlyBaseline,
) -> None:
    report = baseline.reproducible_report()
    assert report == BenchmarkOnlyBaseline(ROOT, as_of=AS_OF).reproducible_report()
    assert report["dataset_snapshot_sha256"] == CORRECTED_SHA256
    assert report["source_id"] == BENCHMARK_ID
    assert report["published_cells"] == 560
    assert report["source_table_counts"] == {"Table 1": 524, "Table 3": 36}
    assert report["published_dimensions"] == {
        "occupation+industry": 524,
        "occupation+region": 34,
        "occupation": 2,
    }
    assert report["supervised_status"] == "BLOCKED_NO_ELIGIBLE_LABELS"
    assert report["train_count"] == report["test_count"] == report["validation_count"] == 0
    assert report["model_trained"] is False
    assert report["runtime_release"] == "BLOCKED_NOT_APPROVED"
    for field in (
        "naive_global_median_monthly_php",
        "naive_global_median_mae_php",
        "occupation_baseline_mae_php",
        "occupation_baseline_median_absolute_error_php",
        "occupation_region_holdout_coverage",
        "per_occupation_test_error",
        "per_region_test_error",
        "skills_salary_effect",
        "model_confidence_interval",
    ):
        assert report[field] is None, field


def test_exact_national_occupation_by_industry_is_published_mean_not_model(
    baseline: BenchmarkOnlyBaseline,
) -> None:
    result = baseline.lookup("Software Developers", industry_code="J62")
    assert result.status == "SOURCE_MATCH_NOT_RELEASED"
    assert result.can_be_used_for_runtime is False
    assert result.sample_size is None
    assert result.pdf_locator is not None
    assert result.observation is not None
    observation = BenchmarkObservation.from_dict(result.to_dict()["observation"])
    assert str(observation.amount_php_monthly) == "45520"
    assert observation.source_id == BENCHMARK_ID
    assert observation.source_table == "Table 1"
    assert observation.source_page == 10
    assert observation.metric == "mean_monthly_wage"
    assert observation.published_dimensions == ("occupation", "industry")
    assert observation.region_psgc_code is None
    assert observation.region_label is None
    assert observation.tax_basis == "unknown"
    assert observation.reference_date == "2022-08"
    assert "2022" in result.reason


def test_exact_accounting_regional_cell_and_national_reference(
    baseline: BenchmarkOnlyBaseline,
) -> None:
    region = baseline.lookup(
        "Accounting and Bookkeeping Clerks", region_label="National Capital Region"
    )
    assert region.status == "SOURCE_MATCH_NOT_RELEASED"
    assert region.observation is not None
    assert str(region.observation.amount_php_monthly) == "24530"
    assert region.observation.source_table == "Table 3"
    assert region.observation.source_page == 17
    assert region.observation.published_dimensions == ("occupation", "region")
    national = baseline.lookup("Accounting and Bookkeeping Clerks")
    assert national.status == "SOURCE_MATCH_NOT_RELEASED"
    assert national.observation is not None
    assert national.observation.published_dimensions == ("occupation",)
    assert national.observation.region_label is None


@pytest.mark.parametrize(
    ("occupation", "industry", "region", "why"),
    [
        ("Software Developers", None, "National Capital Region", "No exact published"),
        ("Software Developers", "J62", "National Capital Region", "No published joint"),
        ("Software Developers", None, None, "No exact published"),
        ("Unobserved Career", "J62", None, "No exact published"),
        ("2512", "J62", None, "No exact published"),
        ("Accounting and Bookkeeping Clerks", "Z99", None, "No exact published"),
        ("Accounting and Bookkeeping Clerks", "", None, "Blank"),
        ("Accounting and Bookkeeping Clerks", None, " ", "Blank"),
        ("", None, None, "Missing occupation"),
    ],
)
def test_no_unpublished_occupation_fallback_or_marginal_synthesis(
    baseline: BenchmarkOnlyBaseline,
    occupation: str,
    industry: str | None,
    region: str | None,
    why: str,
) -> None:
    preview = baseline.lookup(occupation, industry_code=industry, region_label=region)
    assert preview.status == "UNAVAILABLE"
    assert why in preview.reason
    assert preview.observation is None
    assert preview.pdf_locator is None
    assert preview.can_be_used_for_runtime is False


def test_ambiguous_untrusted_occupation_codes_are_not_labels(
    baseline: BenchmarkOnlyBaseline,
) -> None:
    with pytest.raises(ValueError, match="original source labels"):
        baseline.lookup("Software Developers", industry_code=62)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="original source labels"):
        baseline.lookup("Software Developers", region_label=1300000000)  # type: ignore[arg-type]


def test_runtime_keeps_unavailable_even_for_a_real_source_match(
    baseline: BenchmarkOnlyBaseline,
) -> None:
    for args in (
        ("Software Developers", {"industry_code": "J62"}),
        ("Software Developers", {"region_label": "National Capital Region"}),
        ("Accounting and Bookkeeping Clerks", {"region_label": "National Capital Region"}),
    ):
        result = baseline.runtime_result(args[0], **args[1])
        checked = PredictionResult.from_dict(result.to_dict())
        assert checked.mode == "UNAVAILABLE"
        assert checked.central_php_monthly is None
        assert checked.lower_php_monthly is None
        assert checked.upper_php_monthly is None
        assert checked.artifact_version is None
        assert checked.source_id is None
        assert checked.source_version is None
        assert checked.data_as_of is None
        assert checked.basis == "none"


def test_benchmark_metadata_distinguishes_similar_industry_scopes(
    baseline: BenchmarkOnlyBaseline,
) -> None:
    retail = baseline.lookup("Accounting and Bookkeeping Clerks", industry_code="G47")
    transport = baseline.lookup(
        "Accounting and Bookkeeping Clerks", industry_code="H49 excl. H492"
    )
    assert retail.observation is not None and transport.observation is not None
    assert str(retail.observation.amount_php_monthly) == "15491"
    assert str(transport.observation.amount_php_monthly) == "15998"
    assert retail.observation.industry_code != transport.observation.industry_code


def test_unapproved_or_superseded_snapshot_never_passes_gate(tmp_path: Path) -> None:
    root = tmp_path
    stage = root / "data/staged"
    stage.mkdir(parents=True)
    (root / "configs").mkdir()
    for source in ("data_sources.json", "release_policy.yaml"):
        shutil.copy(ROOT / "configs" / source, root / "configs" / source)
    original = ROOT / CORRECTED_PSA_STAGED
    for suffix in (".jsonl", ".lineage.json", ".report.json"):
        original_basename = original.name.removesuffix(".jsonl")
        old = original.with_name(original_basename + suffix)
        (stage / old.name).write_bytes(old.read_bytes())
    baseline = BenchmarkOnlyBaseline(root, as_of=AS_OF)
    assert baseline.reproducible_report()["published_cells"] == 560
    new_path = stage / original.name
    new_path.write_bytes(new_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="provenance or release status"):
        BenchmarkOnlyBaseline(root, as_of=AS_OF)


def test_local_registry_rights_or_import_change_blocks_even_preview(tmp_path: Path) -> None:
    root = tmp_path
    (root / "configs").mkdir()
    register = json.loads((ROOT / "configs/data_sources.json").read_text())
    for source in register["sources"]:
        if source["id"] == BENCHMARK_ID:
            source["enabled_for_import"] = False
    (root / "configs/data_sources.json").write_text(json.dumps(register))
    with pytest.raises(ValueError, match="not approved for local ingestion"):
        BenchmarkOnlyBaseline(root, as_of=AS_OF)


def test_report_write_is_immutable_and_content_addressed(
    tmp_path: Path, baseline: BenchmarkOnlyBaseline,
) -> None:
    report = baseline.reproducible_report()
    first = write_report(tmp_path, report)
    second = write_report(tmp_path, report)
    assert first == second
    assert first["report_status"] == "BLOCKED_NO_ELIGIBLE_LABELS"
    target = tmp_path / first["path"]
    assert target.is_file()
    assert hashlib.sha256(target.read_bytes()).hexdigest() == first["sha256"]
    assert json.loads(target.read_text()) == report


def test_cli_does_not_train_and_does_not_promote_benchmark(capsys) -> None:
    exit_code = main([
        "--root", str(ROOT), "--as-of", AS_OF.isoformat(),
        "--occupation", "Software Developers", "--industry", "J62"
    ])
    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["published_cell_preview"]["observation"]["amount_php_monthly"] == "45520"
    assert output["published_cell_preview"]["can_be_used_for_runtime"] is False
    assert output["runtime_result"]["mode"] == "UNAVAILABLE"
    assert output["runtime_result"]["central_php_monthly"] is None
    assert output["mae_php"] is None
    assert output["model_trained"] is False
