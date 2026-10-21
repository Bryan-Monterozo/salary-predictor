from __future__ import annotations

import copy
import hashlib
import json
from datetime import date
from pathlib import Path

import pytest

from salary_predictor.audit.__main__ import run_audit, write_audit
from salary_predictor.audit.quality import (
    audit_benchmarks,
    exact_coverage,
    load_policy,
    load_staged_benchmarks,
)
from salary_predictor.audit.splits import propose_group_time_holdout

ROOT = Path(__file__).resolve().parents[2]
STAGED = "data/staged/psa_ows_2022_public_tables-25a47e4e13d22b33.jsonl"
AS_OF = date(2026, 10, 21)


@pytest.fixture(scope="module")
def products() -> dict:
    return run_audit(ROOT, STAGED, as_of=AS_OF)


def test_benchmark_audit_counts_and_scopes(products: dict) -> None:
    audit = products["audit"]
    assert audit["quality_status"] == "PASS_BENCHMARK_ONLY"
    assert audit["accepted_cells"] == 560
    assert audit["tables"] == {"Table 1": 524, "Table 3": 36}
    assert audit["published_dimensions"] == {
        "occupation+industry": 524,
        "occupation": 2,
        "occupation+region": 34,
    }
    assert audit["distinct_occupation_labels_by_table"]["Table 1"] == 218
    assert audit["distinct_industry_codes_by_table"]["Table 1"] == 56
    assert audit["distinct_published_region_labels_by_table"]["Table 3"] == 17
    assert audit["survey_reference_dates"] == {"2022-08": 560}
    assert audit["duplicate_observation_ids"] == 0
    assert audit["duplicate_natural_cells"] == 0
    assert audit["outliers_flagged_count"] == 25
    assert audit["staleness_warning"] is True


def test_wage_distribution_is_aggregate_not_model_accuracy(products: dict) -> None:
    audit = products["audit"]
    distribution = audit["wage_distribution_php_monthly"]
    assert distribution["minimum"] == "9125.00"
    assert distribution["maximum"] == "135363.00"
    assert audit["observation_unit"].endswith("NOT one worker")
    assert audit["worker_sample_size"] is None
    assert audit["skills_labeled_ratio"] is None
    assert audit["salary_model_metrics"] is None
    assert audit["training_eligible"] is False


def test_exact_published_cell_coverage_and_impossible_pairs(products: dict) -> None:
    map_ = products["coverage"]
    accepted = exact_coverage(map_, "Software Developers", industry_code="J62")
    assert accepted["status"] == "PUBLISHED_CELL_PRESENT_NOT_RELEASED"
    assert accepted["cell"]["amount_php_monthly"] == "45520"
    accepted = exact_coverage(
        map_, "Accounting and Bookkeeping Clerks", region_label="National Capital Region"
    )
    assert accepted["status"] == "PUBLISHED_CELL_PRESENT_NOT_RELEASED"
    assert accepted["cell"]["amount_php_monthly"] == "24530"

    assert exact_coverage(
        map_, "Software Developers", region_label="National Capital Region"
    )["status"] == "UNAVAILABLE"
    assert exact_coverage(
        map_, "Software Developers", industry_code="J62",
        region_label="National Capital Region",
    )["status"] == "UNAVAILABLE"
    assert exact_coverage(map_, "Software Developers")["status"] == "UNAVAILABLE"


def test_exclusion_headings_have_unique_coverage_keys(products: dict) -> None:
    coverage = products["coverage"]
    retail = exact_coverage(coverage, "Accounting and Bookkeeping Clerks", industry_code="G47")
    road = exact_coverage(
        coverage, "Accounting and Bookkeeping Clerks", industry_code="H49 excl. H492"
    )
    assert retail["cell"]["amount_php_monthly"] == "15491"
    assert road["cell"]["amount_php_monthly"] == "15998"


def test_staged_snapshot_hash_is_enforced(tmp_path: Path) -> None:
    original = ROOT / STAGED
    target_dir = tmp_path / "data/staged"
    target_dir.mkdir(parents=True)
    for suffix in (".jsonl", ".lineage.json", ".report.json"):
        basename = original.name.removesuffix(".jsonl")
        source = original.with_name(basename + suffix)
        (target_dir / source.name).write_bytes(source.read_bytes())
    target = target_dir / original.name
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="provenance or release status"):
        load_staged_benchmarks(tmp_path, f"data/staged/{original.name}")


def test_conflicting_published_cell_is_quarantined_by_audit() -> None:
    rows, source = load_staged_benchmarks(ROOT, STAGED)
    modified = copy.deepcopy(rows)
    duplicate = copy.deepcopy(modified[0])
    duplicate["record"]["observation_id"] = "fake-duplicate-but-unique-id"
    duplicate["record"]["amount_php_monthly"] = "99999"
    modified.append(duplicate)
    report = audit_benchmarks(modified, source, load_policy(ROOT), AS_OF)["audit"]
    assert report["duplicate_natural_cells"] == 1
    assert report["quality_status"] == "REVIEW_REQUIRED"


def test_source_policy_blocks_supervised_models(products: dict) -> None:
    policy = load_policy(ROOT)
    assert policy["evidence"]["supervised_training_enabled"] is False
    assert policy["benchmark"]["allow_combining_marginal_tables"] is False
    assert policy["benchmark"]["allow_hierarchy_salary_fallback"] is False
    assert policy["holdout"]["encoders_and_imputers_fit"] == "train_only"
    split = products["split"]
    assert split["status"] == "BLOCKED_NO_ELIGIBLE_LABELS"
    assert split["partition_counts"] == {"train": 0, "validation": 0, "test": 0}
    assert split["actual_evaluation_scores"] is None
    assert products["leakage"]["checks"]["supervised_model_metrics"] == "NOT_AVAILABLE"


def _fixture_row(name: str, month: str, *, group: str, cluster: str | None = None) -> dict:
    return {
        "source_id": "fake-test-fixture",
        "observation_id": name,
        "record_type": "advertised_salary",
        "observation_date": month,
        "group_id": group,
        "duplicate_cluster_id": cluster or name,
        "predictor_fields": ["occupation_psoc_code", "region_psgc_code"],
    }


def _test_policy() -> dict:
    policy = load_policy(ROOT)
    policy["supervised_draft_thresholds"] = dict(
        min_train_distinct_groups=1,
        min_test_distinct_groups=1,
        min_training_time_periods=2,
        min_test_time_periods=1,
        min_occupation_labeled_records=1,
    )
    return policy


def test_group_and_time_split_cannot_leak_repeated_ads() -> None:
    rows = [
        _fixture_row("old-one", "2025-11", group="employer-1"),
        _fixture_row("old-two", "2025-12", group="employer-2"),
        _fixture_row("recent", "2026-01", group="employer-3", cluster="crosspost"),
        _fixture_row("old-crosspost", "2025-10", group="employer-4", cluster="crosspost"),
        _fixture_row("recent-same-group", "2026-02", group="employer-4"),
    ]
    split = propose_group_time_holdout(rows, cutoff_month="2026-01", policy=_test_policy())
    assert split["status"] == "PROPOSAL_ONLY"
    assert split["approval"] == "NOT_APPROVED_FOR_TRAINING"
    assert set(split["train_ids"]) == {"fake-test-fixture:old-one", "fake-test-fixture:old-two"}
    assert set(split["test_ids"]) == {
        "fake-test-fixture:recent", "fake-test-fixture:recent-same-group"
    }
    assert split["quarantined_ids"] == ["fake-test-fixture:old-crosspost"]
    assert set(split["train_ids"]).isdisjoint(split["test_ids"])
    assert split == propose_group_time_holdout(list(reversed(rows)), cutoff_month="2026-01", policy=_test_policy())
    assert split["frozen"] is False
    assert split["training_and_evaluation_performed"] is False


def test_no_labels_and_low_support_block_ml_protocol() -> None:
    assert propose_group_time_holdout([], cutoff_month="2026-01", policy=load_policy(ROOT))[
        "status"
    ] == "BLOCKED_NO_ELIGIBLE_LABELS"
    one = [_fixture_row("single", "2026-01", group="employer-1")]
    assert propose_group_time_holdout(one, cutoff_month="2026-01", policy=load_policy(ROOT))[
        "status"
    ] == "BLOCKED_INSUFFICIENT_SUPPORT"


def test_target_derived_features_and_mixed_cohorts_fail() -> None:
    row = _fixture_row("bad-feature", "2025-10", group="employer-1")
    row["predictor_fields"].append("monthly_php")
    with pytest.raises(ValueError, match="Outcome-derived"):
        propose_group_time_holdout([row], cutoff_month="2026-01", policy=_test_policy())
    row["predictor_fields"] = ["occupation_psoc_code"]
    other = _fixture_row("mixed", "2026-01", group="employer-2")
    other["record_type"] = "individual_earnings"
    with pytest.raises(ValueError, match="Cannot merge"):
        propose_group_time_holdout([row, other], cutoff_month="2026-01", policy=_test_policy())
    other["record_type"] = "aggregate_benchmark"
    with pytest.raises(ValueError, match="Aggregate benchmark"):
        propose_group_time_holdout([row, other], cutoff_month="2026-01", policy=_test_policy())


def test_duplicate_record_id_fails_before_split() -> None:
    row = _fixture_row("same", "2025-12", group="group-1")
    with pytest.raises(ValueError, match="Repeated labeled"):
        propose_group_time_holdout([row, row], cutoff_month="2026-01", policy=_test_policy())


def test_repeatable_local_audit_output(tmp_path: Path, products: dict) -> None:
    output_root = tmp_path
    first = write_audit(output_root, products)
    second = write_audit(output_root, products)
    assert first == second
    assert set(first) == {"audit", "coverage", "leakage", "split"}
    for file in first.values():
        path = output_root / file
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        saved = json.loads(path.read_text())
        assert saved.get("snapshot_sha256", saved.get("dataset_snapshot_sha256")) == (
            products["audit"]["snapshot_sha256"]
        )
