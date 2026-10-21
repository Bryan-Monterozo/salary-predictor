from __future__ import annotations

import json
from pathlib import Path

import pytest

from salary_predictor.feasibility import jobicy
from salary_predictor.feasibility.step import inspect_step, inspect_step_questionnaire
from salary_predictor.source_inventory import verify_raw_files
from salary_predictor.source_registry import load_source_register, validate_source_register

ROOT = Path(__file__).resolve().parents[2]


def manifest() -> dict:
    return json.loads((ROOT / "configs/local_raw_file_manifest.json").read_text())


def test_step_real_source_is_cp1252_and_has_genuine_hourly_candidates() -> None:
    report = inspect_step(ROOT, manifest())
    assert report["encoding"] == "cp1252"
    assert (report["respondents"], report["columns"]) == (3000, 1276)
    assert report["counts"]["positive_hourly_earnings"] == 1615
    assert report["salary_label_candidate_wage_workers"] == 987
    assert report["counts"]["wage_worker_respondents"] == 1025
    assert report["replicate_weight_columns"] == {"hh": 30, "sp": 30}
    assert report["person_weight_variable"] == "w3_finspwt"
    assert report["monthly_php_label_count"] is None
    assert report["training_eligible"] is False
    assert "uniqueid" not in json.dumps(report).lower()


def test_step_questionnaire_pay_basis_and_source_pin() -> None:
    report = inspect_step_questionnaire(ROOT, manifest())
    assert report["pages"] == 60
    assert report["question_references"]["employee_net_last_pay_period"] == "page 39"


def test_step_changed_checksum_blocks_inspection() -> None:
    data = manifest()
    entry = next(x for x in data["files"] if x["path"].endswith("step_ph_weights_merged_public.csv"))
    entry["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="STEP checksum changed"):
        inspect_step(ROOT, data)


def test_jobicy_valid_salary_requires_units_and_range() -> None:
    valid = {
        "salaryMin": 20000, "salaryMax": 30000, "salaryCurrency": "PHP",
        "salaryPeriod": "monthly"
    }
    assert jobicy._salary(valid) == ("PHP", "monthly")
    for changed in (
        {"salaryMax": 100},
        {"salaryMin": 0},
        {"salaryCurrency": ""},
        {"salaryCurrency": "INR extra"},
        {"salaryPeriod": "unknown"},
        {"salaryMax": None},
    ):
        assert jobicy._salary(dict(valid, **changed)) is None


def test_every_api_request_archives_raw_json_and_manifest_first(tmp_path: Path, monkeypatch) -> None:
    base = tmp_path / "configs"
    base.mkdir()
    (base / "local_raw_file_manifest.json").write_text('{"schema_version":1,"files":[]}')
    body = json.dumps({
        "success": True, "jobs": [{
            "id": 1234, "jobTitle": "Example",
            "jobDescription": "<p>Python and SQL</p>",
            "jobGeo": "Philippines", "jobLevel": "Junior",
            "salaryMin": 20000, "salaryMax": 30000,
            "salaryCurrency": "PHP", "salaryPeriod": "monthly",
        }], "nextCursor": None,
    }).encode()
    monkeypatch.setattr(jobicy, "_request", lambda url, accept: body)
    saved = jobicy.fetch_archived(tmp_path)
    assert (tmp_path / saved["body_path"]).read_bytes() == body
    assert (tmp_path / saved["metadata_path"]).exists()
    saved_manifest = json.loads((base / "local_raw_file_manifest.json").read_text())
    assert {file["path"] for file in saved_manifest["files"]} == {
        saved["body_path"], saved["metadata_path"],
    }
    assert verify_raw_files(tmp_path, saved_manifest) == []
    agg = jobicy.inspect_archive(tmp_path)
    assert agg["distinct_job_ids"] == agg["salary_disclosed_valid_range"] == 1
    assert agg["verified_esco_seed_skill_mentions"] == {
        "Python (computer programming)": 1, "SQL": 1
    }


def test_jobicy_reused_ids_and_missing_salary_do_not_fabricate_labels(
    tmp_path: Path, monkeypatch
) -> None:
    cfg = tmp_path / "configs"
    cfg.mkdir()
    (cfg / "local_raw_file_manifest.json").write_text('{"schema_version":1,"files":[]}')
    body = json.dumps({"success": True, "jobs": [
        {"id": 3, "jobDescription": "<p>Python</p>", "jobGeo": "Philippines"},
        {"id": 3, "jobDescription": "second listing with same id",
         "salaryMin": 1, "salaryMax": 10,
         "salaryPeriod": "yearly", "salaryCurrency": "USD"},
    ]}).encode()
    monkeypatch.setattr(jobicy, "_request", lambda url, accept: body)
    jobicy.fetch_archived(tmp_path)
    output = jobicy.inspect_archive(tmp_path)
    assert output["api_entries_seen"] == 2
    assert output["distinct_job_ids"] == 1
    assert output["duplicate_or_missing_ids"] == 1
    assert output["salary_disclosed_valid_range"] == 1
    assert output["salary_range_preview"][0]["job_id"] == "3"
    assert output["philippines_local_jobs_confirmed"] is None


def test_jobicy_xml_archives_without_pretending_its_a_json_api_page(
    tmp_path: Path, monkeypatch
) -> None:
    cfg = tmp_path / "configs"
    cfg.mkdir()
    (cfg / "local_raw_file_manifest.json").write_text('{"schema_version":1,"files":[]}')
    data = b'<rss><channel><item><title>Remote PH eligible</title></item></channel></rss>'
    monkeypatch.setattr(jobicy, "_request", lambda url, accept: data)
    archive = jobicy.fetch_archived(tmp_path, kind="rss")
    assert archive["items_on_page"] == 1
    assert jobicy.inspect_archive(tmp_path)["distinct_job_ids"] == 0
    assert verify_raw_files(
        tmp_path, json.loads((cfg / "local_raw_file_manifest.json").read_text())
    ) == []


def test_jobicy_request_rejects_unsafe_inputs() -> None:
    with pytest.raises(ValueError):
        jobicy.fetch_archived(ROOT, kind="api", count=0)
    with pytest.raises(ValueError):
        jobicy.fetch_archived(ROOT, kind="rss", cursor="bogus")


def test_registry_keeps_step_and_jobicy_disabled() -> None:
    source = load_source_register(ROOT / "configs/data_sources.json")
    assert not validate_source_register(source)
    by_id = {s["id"]: s for s in source["sources"]}
    for id_ in ("world_bank_step_ph_2015", "jobicy_public_remote_jobs"):
        assert by_id[id_]["enabled_for_import"] is False
        assert by_id[id_]["approved_use"] == []


def test_actual_jobicy_archive_is_checksum_verified_and_aggregate_only() -> None:
    data = jobicy.inspect_archive(ROOT)
    assert data["api_archive_count"] >= 1
    assert data["distinct_job_ids"] >= data["salary_disclosed_valid_range"]
    assert data["training_or_deployment_approved"] is False
    assert "jobDescription" not in json.dumps(data)
    assert "companyName" not in json.dumps(data)



def test_repeated_jobicy_poll_is_blocked_and_archived_first(tmp_path: Path, monkeypatch) -> None:
    cfg = tmp_path / "configs"
    cfg.mkdir()
    (cfg / "local_raw_file_manifest.json").write_text('{"schema_version":1,"files":[]}')
    payload = b'{"success":true,"jobs":[],"nextCursor":null}'
    monkeypatch.setattr(jobicy, "_request", lambda url, accept: payload)
    first = jobicy.fetch_archived(tmp_path)
    assert (tmp_path / first["body_path"]).read_bytes() == payload
    with pytest.raises(ValueError, match="once per hour"):
        jobicy.fetch_archived(tmp_path)
    assert jobicy.inspect_archive(tmp_path)["api_archive_count"] == 1


def test_jobicy_error_json_is_archived_before_being_rejected(
    tmp_path: Path, monkeypatch
) -> None:
    cfg = tmp_path / "configs"
    cfg.mkdir()
    (cfg / "local_raw_file_manifest.json").write_text('{"schema_version":1,"files":[]}')
    payload = b'{"success":false,"statusCode":429,"error":"rate limit"}'
    monkeypatch.setattr(jobicy, "_request", lambda url, accept: payload)
    with pytest.raises(ValueError, match="Archived Jobicy API error"):
        jobicy.fetch_archived(tmp_path)
    assert len(list((tmp_path / "data/raw/jobicy").glob("*.json"))) == 2
    assert jobicy.inspect_archive(tmp_path)["api_error_snapshots"] == 1
    assert verify_raw_files(
        tmp_path, json.loads((cfg / "local_raw_file_manifest.json").read_text())
    ) == []



def test_unfiltered_paginated_jobicy_collection_deduplicates_across_pages(
    tmp_path: Path, monkeypatch
) -> None:
    folder = tmp_path / "configs"
    folder.mkdir()
    (folder / "local_raw_file_manifest.json").write_text('{"schema_version":1,"files":[]}')
    bodies = [
        {"success": True, "nextCursor": "opaque-cursor", "jobs": [
            {"id": 101, "url": "https://jobicy.com/jobs/101",
             "salaryMin": 1000, "salaryMax": 2000,
             "salaryCurrency": "USD", "salaryPeriod": "monthly"},
            {"id": 102, "url": "https://jobicy.com/jobs/102"},
        ]},
        {"success": True, "nextCursor": None, "jobs": [
            {"id": 101, "url": "https://jobicy.com/jobs/101"},
            {"id": 103, "url": "https://jobicy.com/jobs/103",
             "salaryMin": 2000, "salaryMax": 3000,
             "salaryCurrency": "USD", "salaryPeriod": "monthly"},
        ]},
    ]
    urls: list[str] = []

    def fake_request(url: str, accept: str) -> bytes:
        urls.append(url)
        return json.dumps(bodies[len(urls) - 1]).encode()

    monkeypatch.setattr(jobicy, "_request", fake_request)
    result = jobicy.collect_available(tmp_path, target=5000, count=200)
    assert result["status"] == "SOURCE_EXHAUSTED"
    assert result["unique_job_ids_this_traversal"] == 3
    assert result["target_shortfall"] == 4997
    assert result["pages_archived_this_run"] == 2
    assert "geo=" not in urls[0]
    assert "cursor=opaque-cursor" in urls[1]
    assert jobicy.inspect_archive(tmp_path)["distinct_job_ids"] == 3
    exported = jobicy.export_unique_salary_ranges(tmp_path)
    assert exported["unique_valid_salary_ranges"] == 2
    records = json.loads((tmp_path / exported["path"]).read_text())["records"]
    assert [r["job_id"] for r in records] == ["101", "103"]
    assert jobicy.export_unique_salary_ranges(tmp_path)["sha256"] == exported["sha256"]
    assert verify_raw_files(
        tmp_path, json.loads((folder / "local_raw_file_manifest.json").read_text())
    ) == []


def test_geo_priority_is_proposed_only_and_does_not_change_salary() -> None:
    from salary_predictor.feasibility.geo import geo_priority, load_draft_weighting

    policy = load_draft_weighting(ROOT)
    assert policy["training_enabled"] is False
    assert geo_priority("APAC,  Canada,  Europe,  USA", policy)["proposed_sample_weight"] == "2.0"
    assert geo_priority("Philippines", policy)["bucket"] == "APAC_ELIGIBLE"
    assert geo_priority("Anywhere", policy)["proposed_sample_weight"] == "1.0"
    assert geo_priority("USA, Canada", policy)["bucket"] == "OTHER_OR_UNKNOWN"
    assert geo_priority("", policy)["proposed_sample_weight"] == "1.0"
    with pytest.raises(ValueError):
        geo_priority(None, policy)


def test_actual_all_geographies_have_no_duplicate_derived_salary_listings() -> None:
    result = jobicy.inspect_archive(ROOT, salary_preview_limit=None)
    rows = result["salary_range_preview"]
    assert len(rows) == result["salary_disclosed_valid_range"]
    ids = [item["job_id"] for item in rows]
    assert len(ids) == len(set(ids))
    links = [item["source_url"] for item in rows if item["source_url"]]
    assert len(links) == len(set(links))
    assert result["total_valid_salary_records"] == len(rows)



def test_duplicate_jobicy_urls_with_different_ids_are_excluded(tmp_path: Path, monkeypatch) -> None:
    cfg = tmp_path / "configs"
    cfg.mkdir()
    (cfg / "local_raw_file_manifest.json").write_text('{"schema_version":1,"files":[]}')
    body = json.dumps({"success": True, "nextCursor": None, "jobs": [
        {"id": 501, "url": "https://jobicy.com/jobs/same-job",
         "salaryMin": 12, "salaryMax": 20,
         "salaryCurrency": "USD", "salaryPeriod": "hourly"},
        {"id": 502, "url": "https://jobicy.com/jobs/same-job?utm_source=archive"},
    ]}).encode()
    monkeypatch.setattr(jobicy, "_request", lambda url, accept: body)
    jobicy.fetch_archived(tmp_path)
    result = jobicy.inspect_archive(tmp_path)
    assert result["distinct_job_ids"] == 1
    assert result["duplicate_or_missing_ids"] == 1
    assert result["salary_disclosed_valid_range"] == 1


def test_collection_throttle_applies_even_when_changing_geo(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "configs"
    config.mkdir()
    (config / "local_raw_file_manifest.json").write_text('{"schema_version":1,"files":[]}')
    monkeypatch.setattr(
        jobicy, "_request",
        lambda url, accept: b'{"success":true,"nextCursor":null,"jobs":[]}',
    )
    jobicy.fetch_archived(tmp_path, geo="philippines")
    with pytest.raises(ValueError, match="once per hour"):
        jobicy.fetch_archived(tmp_path, geo=None)
