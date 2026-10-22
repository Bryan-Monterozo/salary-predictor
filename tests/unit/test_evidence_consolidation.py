from __future__ import annotations

import hashlib
import json
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from salary_predictor.feasibility.consolidate import main
from salary_predictor.feasibility.consolidated import (
    _archive_jobs,
    _job_record,
    _policy,
    consolidate,
    export_consolidated,
    preview_career_match,
    relevance,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def evidence() -> tuple[dict, list[dict], list[dict]]:
    return consolidate(ROOT, as_of=date(2026, 10, 21))


def test_consolidated_jobicy_is_deduplicated_with_actual_salary_provenance(evidence) -> None:
    report, rows, queue = evidence
    assert report["unique_job_postings"] >= 989
    assert report["valid_disclosed_salary_ranges"] >= 513
    assert report["missing_salary_enrichment_candidates"] == len(queue)
    assert report["valid_disclosed_salary_ranges"] + len(queue) == len(rows)
    assert report["jobicy_snapshot_duplicates_excluded"] >= 78
    assert len({r["record_id"] for r in rows}) == len(rows)
    urls = [r["source_url"] for r in rows if r["source_url"]]
    assert len(urls) == len(set(urls))
    assert report["model_trained"] is False
    assert report["approved_training_rows"] == 0
    assert report["no_generated_salary_labels"] is True
    assert all(r["monthly_php_training_label"] is None for r in rows)
    assert all(r["approved_for_training"] is False for r in rows)
    assert all(r["source_response_sha256"] for r in rows)


def test_no_fabricated_salary_or_illicit_target_merging(evidence) -> None:
    report, rows, queue = evidence
    assert report["psa_ows"]["published_cells"] == 560
    assert report["psa_ows"]["joined_into_job_offer_labels"] is False
    assert report["step"]["verified_wage_worker_hourly_occupation_candidates"] == 987
    assert report["step"]["joined_into_job_offer_labels"] is False
    for job in rows:
        assert job["cohort"] == "advertised_salary"
        if job["salary"] is None:
            assert job["estimated_missing_salary"] is None
        else:
            assert Decimal(job["salary"]["amount_min"]) > 0
            assert Decimal(job["salary"]["amount_max"]) >= Decimal(
                job["salary"]["amount_min"]
            )
            assert job["salary"]["converted_to_php"] is False
    assert all(job["do_not_impute"] is True for job in queue)
    assert all(job["source_url"].startswith("https://") for job in queue)


def test_geographic_priority_does_not_convert_or_train() -> None:
    p = _policy(ROOT)
    philippines = relevance("Philippines, Europe", p)
    nearby = relevance("India, Europe", p)
    other_apac = relevance("Australia", p)
    global_ = relevance("Anywhere", p)
    assert [Decimal(x["ranking_weight"]) for x in (
        philippines, nearby, other_apac, global_
    )] == [Decimal("3.0"), Decimal("2.0"), Decimal("1.0"), Decimal("0.5")]
    assert philippines["bucket"] == "PH_ELIGIBLE"
    assert nearby["bucket"] == "REGIONAL_PEER_ELIGIBLE"
    assert all(
        x["model_sample_weight_applied"] is False
        and x["geography_confidence"] == "remote_eligibility_only"
        for x in (philippines, nearby, other_apac, global_)
    )


def test_user_salary_and_experience_do_not_become_training_labels(evidence) -> None:
    _report, rows, _queue = evidence
    job_title = rows[0]["title"]
    response = preview_career_match(
        rows,
        job_title=job_title,
        skills=["Python"],
        years_experience=Decimal("4"),
        current_salary_php=Decimal("45000"),
        min_support=3,
    )
    assert response["matched_titles_count"] >= 1
    assert response["current_salary_reference_php"] == "45000"
    assert response["years_experience_received_not_model_adjusted"] == "4"
    assert response["predicted_salary_php_monthly"] is None
    assert response["forecast_salary_php_monthly"] is None
    assert response["user_data_recorded_as_training_label"] is False
    assert response["current_salary_is_not_guaranteed_offer_floor"] is True
    assert response["related_role_path_options"]
    assert response["roles_are_not_guaranteed_promotions"] is True
    assert all(
        option["role_path_claim"] == "related_vacancy_not_verified_promotion"
        for option in response["related_role_path_options"]
    )
    assert all(
        "salary_premium" not in item for item in response["career_skill_research_suggestions"]
    )
    with pytest.raises(ValueError, match="Invalid current salary"):
        preview_career_match(
            rows, job_title=job_title, skills=[],
            years_experience=None, current_salary_php=Decimal("-1"), min_support=3,
        )


def test_export_is_immutable_without_writing_original_raw_files(evidence, tmp_path) -> None:
    report, rows, queue = evidence
    first = export_consolidated(tmp_path, report, rows, queue)
    second = export_consolidated(tmp_path, report, rows, queue)
    assert first == second
    path = tmp_path / "data/audit"
    for name, digest in first["files"].items():
        assert hashlib.sha256((path / name).read_bytes()).hexdigest() == digest
    saved = sorted(path.glob("job-evidence-*.jsonl"))
    with saved[0].open(encoding="utf-8") as file:
        entries = [json.loads(line) for line in file]
    assert len(entries) == report["unique_job_postings"]
    assert not (tmp_path / "data/raw").exists()


def test_bad_policy_cannot_enable_training(tmp_path: Path) -> None:
    path = tmp_path / "configs/feasibility"
    path.mkdir(parents=True)
    original = json.loads(
        (ROOT / "configs/feasibility/consolidation_policy.json").read_text()
    )
    original["training_enabled"] = True
    (path / "consolidation_policy.json").write_text(json.dumps(original))
    with pytest.raises(ValueError, match="Unreviewed"):
        _policy(tmp_path)


def test_duplicate_api_pages_cannot_fabricate_distinct_postings(tmp_path: Path) -> None:
    raw = tmp_path / "data/raw/jobicy"
    raw.mkdir(parents=True)
    jobs = [
        {"id": 1, "url": "https://jobicy.com/jobs/example", "jobTitle": "Example"},
        {"id": 2, "url": "https://jobicy.com/jobs/example", "jobTitle": "Example"},
    ]
    for index, slice_ in enumerate((jobs, [jobs[0]])):
        data = json.dumps({"success": True, "jobs": slice_}).encode()
        name = f"test{index}.json"
        (raw / name).write_bytes(data)
        (raw / f"test{index}.meta.json").write_text(json.dumps({
            "archive_type": "api", "raw_filename": name,
            "response_sha256": hashlib.sha256(data).hexdigest(),
            "retrieved_utc": "2026-10-21T10:00:00Z",
        }))
    records, counts = _archive_jobs(tmp_path)
    assert len(records) == 1
    assert counts["duplicate_snapshot_rows"] == 2


def test_invalid_salary_from_job_is_not_imputed() -> None:
    data = _job_record(
        "1",
        {"jobTitle": "Engineer", "jobGeo": "Philippines",
         "salaryMin": None, "salaryMax": None, "jobDescription": "<p>Python</p>"},
        {"response_sha256": "a" * 64, "retrieved_utc": "2026-10-21T10:00:00Z"},
        _policy(ROOT),
        [("Python", "http://data.europa.eu/esco/skill/example",
          __import__("re").compile(r"(?<!\w)Python(?!\w)", __import__("re").I))],
    )
    assert data["salary"] is None
    assert data["salary_label_present"] is False
    assert data["monthly_php_training_label"] is None
    assert data["description_skill_mentions"][0]["evidence"] == "unreviewed_description_mention"


def test_cli_offline_summary_only_does_not_create_a_personal_salary(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["--root", str(ROOT), "--as-of", "2026-10-21"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["consolidation"]["model_trained"] is False
    assert result["consolidation"]["salary_forecast_available"] is False
