from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import BadZipFile, ZipFile

MAIN = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
PACKAGE_REL = "{http://schemas.openxmlformats.org/package/2006/relationships}"


@dataclass(frozen=True)
class PsocGroup:
    code: str
    label: str
    level: str
    parent_code: str | None
    major_group: str
    major_label: str
    source_sheet: str
    source_row: int
    psoc_1992_relation: str | None
    isco_2008_relation: str | None


def _strings(file: ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in file.namelist():
        return []
    root = ET.fromstring(file.read("xl/sharedStrings.xml"))
    return [
        "".join(node.text or "" for node in item.iter(MAIN + "t"))
        for item in root.findall(MAIN + "si")
    ]


def _cell_value(cell: ET.Element, shared: list[str]) -> str:
    value = cell.findtext(MAIN + "v", default="")
    if cell.get("t") == "s":
        value = shared[int(value)]
    elif cell.get("t") == "inlineStr":
        value = "".join(node.text or "" for node in cell.iter(MAIN + "t"))
    return " ".join(value.split())


def _sheet_cells(file: ZipFile, target: str, shared: list[str]) -> dict[int, dict[str, str]]:
    if target.startswith("/"):
        name = target.lstrip("/")
    else:
        name = "xl/" + target
    if not re.fullmatch(r"xl/worksheets/sheet\d+\.xml", name):
        raise ValueError("Unexpected worksheet path")
    root = ET.fromstring(file.read(name))
    rows: dict[int, dict[str, str]] = {}
    for row in root.findall(".//" + MAIN + "sheetData/" + MAIN + "row"):
        values: dict[str, str] = {}
        for cell in row.findall(MAIN + "c"):
            address = cell.get("r", "")
            col = re.match(r"[A-Z]+", address)
            if col is None or col.group() not in {"A", "B", "C", "D", "E", "F"}:
                continue
            value = _cell_value(cell, shared)
            if value:
                values[col.group()] = value
        if values:
            rows[int(row.get("r", "0"))] = values
    return rows


def load_psoc_groups(path: Path, expected_sha256: str) -> tuple[PsocGroup, ...]:
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected_sha256:
        raise ValueError("PSOC workbook changed; review it before use")

    results: list[PsocGroup] = []
    sheets_seen: set[str] = set()
    try:
        with ZipFile(path) as file:
            shared = _strings(file)
            book = ET.fromstring(file.read("xl/workbook.xml"))
            relations = ET.fromstring(file.read("xl/_rels/workbook.xml.rels"))
            targets = {
                node.get("Id"): node.get("Target")
                for node in relations.findall(PACKAGE_REL + "Relationship")
            }
            for sheet in book.find(MAIN + "sheets"):
                sheet_name = sheet.get("name", "")
                major = sheet_name.removeprefix("Group ")
                if sheet_name != f"Group {major}" or major not in "0123456789":
                    raise ValueError(f"Unexpected PSOC sheet: {sheet_name}")
                if major in sheets_seen:
                    raise ValueError("Repeated PSOC major-group sheet")
                sheets_seen.add(major)
                rows = _sheet_cells(file, targets[sheet.get(REL + "id")], shared)
                if not all(rows.get(1, {}).get(key) for key in ("A", "B", "C", "D")):
                    raise ValueError(f"Missing PSOC table columns in {sheet_name}")
                major_label = rows.get(4, {}).get("D")
                if not major_label:
                    raise ValueError(f"Missing major group title in {sheet_name}")
                for row_number, values in rows.items():
                    for col, length, level in (
                        ("A", 2, "sub_major"),
                        ("B", 3, "minor"),
                        ("C", 4, "unit"),
                    ):
                        code = values.get(col)
                        if code is None or not re.fullmatch(rf"\d{{{length}}}", code):
                            continue
                        if not code.startswith(major):
                            raise ValueError(f"Bad major group for code {code}")
                        label = values.get("D")
                        if not label:
                            raise ValueError(f"PSOC group {code} has no title")
                        results.append(PsocGroup(
                            code=code,
                            label=label,
                            level=level,
                            parent_code=code[:-1] if length > 2 else major,
                            major_group=major,
                            major_label=major_label,
                            source_sheet=sheet_name,
                            source_row=row_number,
                            psoc_1992_relation=values.get("E") if level == "unit" else None,
                            isco_2008_relation=values.get("F") if level == "unit" else None,
                        ))
    except (BadZipFile, KeyError, ET.ParseError, TypeError, AttributeError) as exc:
        raise ValueError("Invalid PSOC workbook structure") from exc

    if sheets_seen != set("0123456789"):
        raise ValueError("PSOC major groups are incomplete")
    by_code: dict[str, PsocGroup] = {}
    for entry in results:
        if entry.code in by_code:
            raise ValueError(f"Repeated PSOC code: {entry.code}")
        by_code[entry.code] = entry
    for entry in results:
        if entry.level != "sub_major" and entry.parent_code not in by_code:
            raise ValueError(f"PSOC parent missing: {entry.code}")
    return tuple(sorted(results, key=lambda entry: (entry.code, entry.level)))
