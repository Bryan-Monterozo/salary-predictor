from __future__ import annotations

import copy
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest

from salary_predictor.ingest.pipeline import inspect, stage

ROOT = Path(__file__).resolve().parents[2]
PDF_MANIFEST = Path("configs/ingest/psa_ows_2022.json")
OCC_MANIFEST = Path("configs/ingest/esco_occupations.json")
PSOC_MANIFEST = Path("configs/ingest/psoc_workbook.json")

KEYS = [
    "observation_id", "source_id", "record_type", "currency",
    "original_amount_min", "original_amount_max", "original_period",
    "target_scope", "pay_component", "tax_basis", "thirteenth_month_included",
    "allowances_included", "observation_date", "occupation_psoc_code",
    "region_psgc_code", "derivation_notes",
]
KEYS += ["observed_" + name for name in (
    "occupation_psoc_code", "region_psgc_code", "years_experience",
    "education_level", "normalized_skills",
)]


def salary_workspace(tmp_path: Path, *, postings: bool = False) -> Path:
    config = tmp_path / "configs" / "ingest"
    raw = tmp_path / "data" / "raw" / "salaries"
    config.mkdir(parents=True)
    raw.mkdir(parents=True)
    fixture = "synthetic_job_ad_rows.csv" if postings else "synthetic_salary_rows.csv"
    target = raw / fixture
    shutil.copyfile(ROOT / "tests" / "fixtures" / "ingest" / fixture, target)
    sha = hashlib.sha256(target.read_bytes()).hexdigest()
    original = json.loads((ROOT / "configs" / "data_sources.json").read_text())
    source = copy.deepcopy(original["sources"][0])
    source.update({
        "id": "fixture_salary_cohort",
        "name": "Synthetic test-only salary rows",
        "dataset_role": "advertised_salary" if postings else "individual_earnings",
        "local_path": "data/raw/salaries/" + fixture,
        "rights_status": "verified_permissioned_fixture",
        "approved_use": ["train_on_labeled_salary"],
        "available_features": ["salary", "occupation_psoc_code"],
        "public_benchmark_eligible": False,
        "enabled_for_import": True,
        "format": "csv",
        "sha256": sha,
        "reference_date": "2026-01",
    })
    original["sources"] = [source]
    (tmp_path / "configs" / "data_sources.json").write_text(json.dumps(original))
    data = {
        "contract_version": 1,
        "source_id": source["id"],
        "source_version": "synthetic-test-only-v1",
        "source_url": "https://example.org/synthetic-test-only",
        "local_path": source["local_path"],
        "file_sha256": sha,
        "dataset_role": source["dataset_role"],
        "file_format": "csv",
        "rights_status": source["rights_status"],
        "approved_uses": source["approved_use"],
        "reference_date": "2026-01",
        "observation_unit": "One fictional worker or job posting",
        "population": "Synthetic fixtures: never use as training data",
        "compensation_basis": "Fictional PHP gross total excluding 13th month",
        "column_mapping": {column: column for column in KEYS},
        "parser_options": {
            "paid_days_per_month": "26",
            "months_per_year": "12",
            **({"posted_target": "midpoint"} if postings else {}),
        },
        "table_id": None,
        "sheet_name": None,
    }
    (config / "salary.json").write_text(json.dumps(data))
    return tmp_path


def test_pdf_preview_retains_real_table_and_page_lineage() -> None:
    report, records = inspect(ROOT, PDF_MANIFEST, "psa_ows_2022_public_tables", preview_limit=1)
    assert report.scanned_count == report.accepted_count == 560
    assert report.rejected_count == 0
    assert report.eligible_to_stage is True
    assert {row["record"]["source_table"] for row in records} == {"Table 1", "Table 3"}
    assert {row["record"]["source_page"] for row in records} >= {1, 10, 17}
    assert all(row["lineage"]["raw_sha256"] == report.raw_sha256 for row in records)

    cases = [
        ("Table 1", "Software Developers", "J62", None, "45520"),
        ("Table 1", "Farm/Plantation/Forest Supervisors", "A01/A02", None, "18174"),
        ("Table 3", "Accounting and Bookkeeping Clerks", None, "National Capital Region", "24530"),
    ]
    for table, occupation, industry, region, amount in cases:
        matches = [
            item["record"] for item in records
            if item["record"]["source_table"] == table
            and item["record"]["occupation_label"] == occupation
            and item["record"]["industry_code"] == industry
            and item["record"]["region_label"] == region
        ]
        assert len(matches) == 1
        assert matches[0]["amount_php_monthly"] == amount

    assert all(
        item["record"]["region_label"] is None for item in records
        if item["record"]["source_table"] == "Table 1"
    )
    assert all(
        item["record"]["industry_code"] is None for item in records
        if item["record"]["source_table"] == "Table 3"
    )
    assert any(
        x["lineage"].get("source_occupation_code") == "9000b" for x in records
    )


def test_preview_is_repeatable_and_has_no_write_side_effects() -> None:
    report, _ = inspect(ROOT, PDF_MANIFEST, "psa_ows_2022_public_tables", preview_limit=0)
    second, _ = inspect(ROOT, PDF_MANIFEST, "psa_ows_2022_public_tables", preview_limit=0)
    assert report.record_fingerprint == second.record_fingerprint
    assert report.to_dict() == second.to_dict()


def test_esco_csv_and_psoc_xlsx_use_distinct_taxonomy_adapters() -> None:
    esco, occupations = inspect(ROOT, OCC_MANIFEST, "esco_english_bundle", preview_limit=1)
    assert (esco.scanned_count, esco.accepted_count, esco.rejected_count) == (3043, 3039, 0)
    assert occupations[0]["record"]["kind"] == "occupation"
    assert esco.eligible_to_stage is False

    psoc, groups = inspect(ROOT, PSOC_MANIFEST, "psa_psoc_2022_update", preview_limit=1)
    assert psoc.scanned_count == psoc.accepted_count == 639
    assert psoc.rejected_count == 0
    assert all("parent_code" in row["record"] for row in groups)
    assert psoc.eligible_to_stage is False


def test_unknown_or_changed_source_is_blocked() -> None:
    with pytest.raises(ValueError, match="does not match manifest"):
        inspect(ROOT, PDF_MANIFEST, "esco_english_bundle")
    with pytest.raises(ValueError, match="reviewed manifest path"):
        inspect(
            ROOT, PDF_MANIFEST, "psa_ows_2022_public_tables",
            input_path="data/raw/psa/other.pdf",
        )


def test_unapproved_source_cannot_stage_records(tmp_path: Path) -> None:
    report, records = inspect(ROOT, OCC_MANIFEST, "esco_english_bundle", preview_limit=0)
    with pytest.raises(PermissionError, match="disabled"):
        stage(tmp_path, report, records)


def test_synthetic_worker_csv_counts_rejects_and_normalizes_known_periods(
    tmp_path: Path,
) -> None:
    root = salary_workspace(tmp_path)
    report, records = inspect(
        root, Path("configs/ingest/salary.json"), "fixture_salary_cohort", preview_limit=3
    )
    assert report.eligible_to_stage is True
    assert report.scanned_count == 9
    assert report.accepted_count == 3
    assert report.rejected_count == 6
    assert all("original_amount_min" not in example for example in report.preview)
    assert {row["record"]["normalized"]["monthly_php"] for row in records} == {
        "48000.00", "31200.00", "50000.00",
    }
    assert all(row["record"]["normalized"]["cohort"] == "individual_earnings"
               for row in records)
    assert any("Repeated salary observation ID" in row["reason"] for row in report.rejected)
    assert not (root / "data/staged").exists()

    first = stage(root, report, records)
    second = stage(root, report, records)
    assert first == second
    output = root / first["records"]
    assert len(output.read_text().splitlines()) == 3
    assert output.parent == root / "data/staged"
    assert json.loads((root / first["lineage"]).read_text())["release_status"] == (
        "not_approved_for_runtime"
    )


def test_job_ad_midpoint_remains_separate_from_worker_salary(tmp_path: Path) -> None:
    root = salary_workspace(tmp_path, postings=True)
    report, records = inspect(root, Path("configs/ingest/salary.json"), "fixture_salary_cohort")
    assert (report.scanned_count, report.accepted_count, report.rejected_count) == (1, 1, 0)
    observation = records[0]["record"]["salary_observation"]
    result = records[0]["record"]["normalized"]
    assert observation["original_amount_min"] == "30000"
    assert observation["original_amount_max"] == "50000"
    assert result["monthly_php"] == "40000.00"
    assert result["calculation"] == "advertised_midpoint_not_employee_earnings"


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("currency", "USD", "Only PHP"),
        ("tax_basis", "net", "Only documented gross"),
        ("original_period", "unknown", "Unknown or unsupported"),
        ("thirteenth_month_included", "true", "13th-month"),
    ],
)
def test_known_salary_units_are_required(
    tmp_path: Path, field: str, value: str, error: str
) -> None:
    root = salary_workspace(tmp_path)
    path = root / "data/raw/salaries/synthetic_salary_rows.csv"
    text = path.read_text().replace("48000.00", "48000.00", 1)
    lines = text.splitlines()
    header = lines[0].split(",")
    row = lines[1].split(",")
    row[header.index(field)] = value
    lines[1] = ",".join(row)
    path.write_text("\n".join(lines) + "\n")
    config = root / "configs/ingest/salary.json"
    manifest = json.loads(config.read_text())
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest["file_sha256"] = checksum
    config.write_text(json.dumps(manifest))
    registry = root / "configs/data_sources.json"
    s = json.loads(registry.read_text())
    s["sources"][0]["sha256"] = checksum
    registry.write_text(json.dumps(s))
    report, _ = inspect(root, Path("configs/ingest/salary.json"), "fixture_salary_cohort")
    assert report.rejected_count >= 6
    assert any(error in item["reason"] for item in report.rejected)


def test_missing_conversion_factor_blocks_daily_pay(tmp_path: Path) -> None:
    root = salary_workspace(tmp_path)
    manifest_path = root / "configs/ingest/salary.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["parser_options"].pop("paid_days_per_month")
    manifest_path.write_text(json.dumps(manifest))
    report, _ = inspect(root, Path("configs/ingest/salary.json"), "fixture_salary_cohort")
    assert report.rejected_count == 7
    assert any("paid_days_per_month" in e["reason"] for e in report.rejected)


def test_manifest_cannot_point_outside_workspace(tmp_path: Path) -> None:
    root = salary_workspace(tmp_path)
    manifest_path = root / "configs/ingest/salary.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["local_path"] = "../private_salary.csv"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="data/raw|Source path"):
        inspect(root, Path("configs/ingest/salary.json"), "fixture_salary_cohort")



def test_benchmark_staging_creates_deterministic_csv_with_lineage(tmp_path: Path) -> None:
    report, records = inspect(ROOT, PDF_MANIFEST, "psa_ows_2022_public_tables", preview_limit=0)
    first = stage(tmp_path, report, records)
    again = stage(tmp_path, report, records)
    assert first == again
    path = tmp_path / first["benchmark_csv"]
    with path.open(encoding="utf-8", newline="") as file:
        lines = list(csv.DictReader(file))
    assert len(lines) == 560
    assert all(row["raw_sha256"] == report.raw_sha256 for row in lines)
    assert {row["source_table"] for row in lines} == {"Table 1", "Table 3"}
    assert any(row["source_occupation_code"] == "9000b" for row in lines)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == first["benchmark_csv_sha256"]


def test_large_multiline_and_short_trailing_esco_fields_are_readable() -> None:
    csv.field_size_limit(sys.maxsize)
    with (ROOT / "data/raw/esco/conceptSchemes_en.csv").open(
        encoding="utf-8-sig", newline=""
    ) as file:
        reader = csv.DictReader(file)
        items = list(reader)
    assert len(items) == 20
    assert max(len(str(value)) for row in items for value in row.values()) > 131072

    with (ROOT / "data/raw/esco/skillsHierarchy_en.csv").open(
        encoding="utf-8-sig", newline=""
    ) as file:
        reader = csv.reader(file)
        header = next(reader)
        widths = {len(row) for row in reader}
    assert len(header) == 14
    assert widths == {11, 12, 13, 14}


def test_duplicate_salary_headers_are_rejected(tmp_path: Path) -> None:
    root = salary_workspace(tmp_path)
    path = root / "data/raw/salaries/synthetic_salary_rows.csv"
    lines = path.read_text().splitlines()
    columns = lines[0].split(",")
    columns[1] = columns[0]
    lines[0] = ",".join(columns)
    path.write_text("\n".join(lines) + "\n")
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    config = root / "configs/ingest/salary.json"
    manifest = json.loads(config.read_text())
    manifest["file_sha256"] = checksum
    config.write_text(json.dumps(manifest))
    registry = root / "configs/data_sources.json"
    data = json.loads(registry.read_text())
    data["sources"][0]["sha256"] = checksum
    registry.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Repeated source CSV"):
        inspect(root, Path("configs/ingest/salary.json"), "fixture_salary_cohort")



def test_pdf_exclusion_industry_headers_do_not_inherit_previous_code() -> None:
    report, records = inspect(ROOT, PDF_MANIFEST, "psa_ows_2022_public_tables", preview_limit=0)
    assert (report.accepted_count, report.rejected_count) == (560, 0)
    def lookup(code: str, occupation: str) -> list[dict]:
        return [
            row["record"] for row in records
            if row["record"]["industry_code"] == code
            and row["record"]["occupation_label"] == occupation
        ]

    transportation = lookup("H49 excl. H492", "Accounting and Bookkeeping Clerks")
    finance = lookup("K64 excl. K6411", "Accounting and Bookkeeping Clerks")
    retail = lookup("G47", "Accounting and Bookkeeping Clerks")
    software = lookup("J62", "Software Developers")
    assert len(transportation) == len(finance) == len(retail) == len(software) == 1
    assert transportation[0]["amount_php_monthly"] == "15998"
    assert retail[0]["amount_php_monthly"] == "15491"
    assert software[0]["amount_php_monthly"] == "45520"
