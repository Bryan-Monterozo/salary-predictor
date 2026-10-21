from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from zipfile import BadZipFile, ZipFile


def verify_raw_files(root: Path, manifest: dict) -> list[str]:
    errors: list[str] = []
    source_dir = (root / "data" / "raw").resolve()
    records = manifest.get("files")
    if manifest.get("schema_version") != 1 or not isinstance(records, list) or not records:
        return ["Invalid raw file manifest"]

    seen: set[str] = set()
    for record in records:
        relative_path = record.get("path", "")
        if not isinstance(relative_path, str) or relative_path in seen:
            errors.append(f"Invalid or repeated path: {relative_path}")
            continue
        seen.add(relative_path)
        path = (root / relative_path).resolve()

        if not path.is_relative_to(source_dir) or path == source_dir:
            errors.append(f"Path is outside data/raw: {relative_path}")
            continue
        if not path.is_file():
            errors.append(f"Missing file: {relative_path}")
            continue

        with path.open("rb") as file:
            actual_hash = hashlib.file_digest(file, "sha256").hexdigest()
        if actual_hash != record.get("sha256"):
            errors.append(f"Checksum changed: {relative_path}")

        if "bytes" in record and path.stat().st_size != record["bytes"]:
            errors.append(f"File size changed: {relative_path}")

        if path.suffix.lower() == ".csv":
            try:
                csv.field_size_limit(sys.maxsize)
                with path.open(encoding="utf-8-sig", newline="") as file:
                    reader = csv.reader(file)
                    header = next(reader)
                    count = 0
                    for row in reader:
                        count += 1
                        if len(row) > len(header):
                            errors.append(f"Row wider than header: {relative_path}")
                            break
                if count != record.get("rows"):
                    errors.append(f"Row count changed: {relative_path}")
                if len(header) != record.get("columns"):
                    errors.append(f"Column count changed: {relative_path}")
            except (UnicodeError, csv.Error, StopIteration, OverflowError) as exc:
                errors.append(f"Cannot read CSV {relative_path}: {type(exc).__name__}")
        elif path.suffix.lower() == ".xlsx":
            try:
                with ZipFile(path) as workbook:
                    files = set(workbook.namelist())
                    if not (
                        "[Content_Types].xml" in files
                        and "xl/workbook.xml" in files
                        and any(name.startswith("xl/worksheets/sheet") for name in files)
                    ):
                        errors.append(f"Invalid workbook: {relative_path}")
            except BadZipFile:
                errors.append(f"Cannot read XLSX: {relative_path}")
        elif path.suffix.lower() == ".pdf":
            with path.open("rb") as file:
                if not file.read(5).startswith(b"%PDF-"):
                    errors.append(f"Not a PDF: {relative_path}")
        else:
            errors.append(f"Unsupported source format: {relative_path}")

    if source_dir.is_dir():
        discovered = {
            p.relative_to(root).as_posix()
            for p in source_dir.rglob("*")
            if p.is_file()
        }
        for extra in sorted(discovered - seen):
            errors.append(f"Unexpected raw file: {extra}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Check local raw data without importing it.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--manifest", type=Path, default=Path("configs/local_raw_file_manifest.json")
    )
    args = parser.parse_args()

    manifest_path = args.root / args.manifest
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"Cannot read manifest: {type(exc).__name__}")
        return 1

    errors = verify_raw_files(args.root.resolve(), manifest)
    if errors:
        for error in errors:
            print(error)
        return 1

    print(f"Verified {len(manifest['files'])} raw files. No import or training done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
