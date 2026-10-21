from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

from salary_predictor.source_inventory import verify_raw_files
from salary_predictor.source_registry import load_source_register, validate_source_register

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def make_manifest(tmp_path: Path) -> dict:
    directory = tmp_path / "data" / "raw" / "esco"
    directory.mkdir(parents=True)
    path = directory / "sample.csv"
    path.write_text('conceptUri,preferredLabel,notes\n123,role,"two\nlines"\n', encoding="utf-8")
    pdf = tmp_path / "data" / "raw" / "psa" / "sample.pdf"
    pdf.parent.mkdir()
    pdf.write_bytes(b"%PDF-1.7\nthis is a test only")

    return {
        "schema_version": 1,
        "files": [
            {
                "path": "data/raw/esco/sample.csv",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "rows": 1,
                "columns": 3,
            },
            {
                "path": "data/raw/psa/sample.pdf",
                "sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
                "bytes": pdf.stat().st_size,
            },
        ],
    }


def test_local_inventory_checks_csv_and_pdf_without_import(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path)
    assert verify_raw_files(tmp_path, manifest) == []


def test_local_inventory_detects_changed_content(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path)
    path = tmp_path / manifest["files"][0]["path"]
    path.write_text("a,b\n1,2\n", encoding="utf-8")

    errors = verify_raw_files(tmp_path, manifest)
    assert any("Checksum changed" in error for error in errors)
    assert any("Column count changed" in error for error in errors)


def test_local_inventory_detects_missing_file(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path)
    (tmp_path / manifest["files"][1]["path"]).unlink()
    assert any("Missing file" in e for e in verify_raw_files(tmp_path, manifest))


def test_local_inventory_rejects_path_escape(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path)
    manifest["files"][0]["path"] = "../private.csv"
    assert any("outside data/raw" in e for e in verify_raw_files(tmp_path, manifest))


def test_local_inventory_rejects_unlisted_files(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path)
    (tmp_path / "data" / "raw" / "esco" / "new.csv").write_text("id\n1\n")
    assert any("Unexpected raw file" in e for e in verify_raw_files(tmp_path, manifest))


def test_real_manifest_and_registry_agree_without_loading_raw_data() -> None:
    manifest = json.loads(
        (PROJECT_ROOT / "configs/local_raw_file_manifest.json").read_text(encoding="utf-8")
    )
    register = load_source_register(PROJECT_ROOT / "configs/data_sources.json")
    assert not validate_source_register(register)
    files = {item["path"] for item in manifest["files"]}
    esco = next(source for source in register["sources"] if source["id"] == "esco_english_bundle")
    psa = next(
        source for source in register["sources"] if source["id"] == "psa_ows_2022_public_tables"
    )

    assert len([p for p in files if p.startswith("data/raw/esco/")]) == 19
    assert psa["local_path"] in files
    assert esco["local_path"] == "data/raw/esco"
    assert [s["id"] for s in register["sources"] if s["enabled_for_import"]] == [
        "psa_ows_2022_public_tables"
    ]



def test_local_xlsx_structure_and_checksum(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path)
    path = tmp_path / "data" / "raw" / "psoc" / "example.xlsx"
    path.parent.mkdir(parents=True)
    with ZipFile(path, "w") as file:
        file.writestr("[Content_Types].xml", "<Types/>")
        file.writestr("xl/workbook.xml", "<workbook/>")
        file.writestr("xl/worksheets/sheet1.xml", "<worksheet/>")
    manifest["files"].append({
        "path": "data/raw/psoc/example.xlsx",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    })
    assert verify_raw_files(tmp_path, manifest) == []

    path.write_bytes(b"not a workbook")
    errors = verify_raw_files(tmp_path, manifest)
    assert any("Checksum changed" in e for e in errors)
    assert any("Cannot read XLSX" in e for e in errors)
