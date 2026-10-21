from __future__ import annotations

import hashlib
from pathlib import Path
from zipfile import ZipFile

import pytest

from salary_predictor.taxonomy.psoc_workbook import load_psoc_groups

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data" / "raw" / "psoc" / "2022-psoc.xlsx"
SHA256 = "666e276e061efcf6939a4839420ddb0a9a30d3786ab41c5892a81a915858941c"


def test_psoc_source_matches_reviewed_file_hash() -> None:
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SHA256


def test_psoc_workbook_groups_have_complete_hierarchy() -> None:
    groups = load_psoc_groups(SOURCE, SHA256)
    codes = {group.code for group in groups}
    assert len(groups) == 639
    assert len(codes) == 639
    assert {group.major_group for group in groups} == set("0123456789")
    for group in groups:
        if group.level != "sub_major":
            assert group.parent_code in codes
        assert group.source_sheet == f"Group {group.major_group}"
        assert group.source_row > 0


def test_psoc_code_and_source_relation_values_remain_distinct() -> None:
    groups = {group.code: group for group in load_psoc_groups(SOURCE, SHA256)}
    software = groups["2512"]
    assert software.label == "SOFTWARE DEVELOPERS"
    assert software.level == "unit"
    assert software.parent_code == "251"
    assert software.isco_2008_relation == "2512"
    assert software.psoc_1992_relation == "2131 p"
    assert groups["0110"].major_group == "0"


def test_changed_file_hash_is_rejected_before_reading() -> None:
    with pytest.raises(ValueError, match="changed"):
        load_psoc_groups(SOURCE, "0" * 64)


def test_invalid_workbook_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "not-an-excel-file.xlsx"
    path.write_bytes(b"not an XLSX file")
    file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="Invalid PSOC workbook"):
        load_psoc_groups(path, file_hash)

    with ZipFile(path, "w") as file:
        file.writestr("xl/workbook.xml", "<workbook/>")
    file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="Invalid PSOC workbook"):
        load_psoc_groups(path, file_hash)
