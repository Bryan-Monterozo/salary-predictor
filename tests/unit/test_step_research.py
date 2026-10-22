from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

from salary_predictor.trainer.operator_cli import main as trainer_cli
from salary_predictor.trainer.step_research import (
    load_research_sample,
    split_clusters,
    train_step_research,
)

ROOT = Path(__file__).resolve().parents[2]


def test_real_step_sample_is_hourly_and_weighted():
    rows, info = load_research_sample(ROOT)
    assert len(rows) == 987
    assert info["distinct_psus"] == 186
    assert info["raw_sha256"].startswith("f181d1d32")
    assert all(row["person_weight"] > 0 and row["hourly_php"] > 0 for row in rows)
    assert all(len(row["occupation"]) == 1 for row in rows)


def test_clusters_never_cross_holdout():
    rows, _ = load_research_sample(ROOT)
    parts = split_clusters(rows)
    assert parts == split_clusters(rows)
    assert [len(parts[k]) for k in ("train", "validation", "test")] == [644, 134, 209]
    for first, second in (("train", "validation"), ("train", "test"), ("validation", "test")):
        assert not {rows[i]["psu"] for i in parts[first]} & {
            rows[i]["psu"] for i in parts[second]
        }


def test_actual_research_evaluation_is_reproducible():
    a = train_step_research(ROOT, as_of=date(2026, 10, 21), write=False)
    b = train_step_research(ROOT, as_of=date(2026, 10, 21), write=False)
    assert a == b
    assert a["training_was_actual"] is True
    assert a["test_metrics_selected"]["count"] == 209
    assert a["cluster_disjointness_verified"] is True
    assert a["test_metrics_occupation_baseline"]["count"] == 209
    assert a["weighted_test_gain_over_baseline"] is False
    assert a["personal_monthly_php_model_available"] is False
    assert a["research_data_not_approved_for_model_deployment"] is True


def test_report_has_only_aggregate_evaluation():
    report = train_step_research(ROOT, as_of=date(2026, 10, 21))
    target = ROOT / report["report_path"]
    assert hashlib.sha256(target.read_bytes()).hexdigest() == report["report_sha256"]
    stored = json.loads(target.read_text(encoding="utf-8"))
    assert stored["source"]["eligible_hourly_research_rows"] == 987
    assert stored["test_metrics_selected"]["psu_groups"] == 37
    assert not list(target.parent.glob("*.joblib"))


def test_regular_product_training_still_blocked(capsys):
    assert trainer_cli(["train", "--root", str(ROOT), "--as-of", "2026-10-21"]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["eligible_training_rows"] == 0
