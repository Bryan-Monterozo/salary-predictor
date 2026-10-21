from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from salary_predictor.contracts import SourceManifest
from salary_predictor.source_registry import load_source_register, validate_source_register

from .adapters import (
    Batch,
    parse_esco,
    parse_esco_relations,
    parse_ows_2022,
    parse_psoc_xlsx,
    parse_salary_csv,
)


@dataclass
class IngestReport:
    source_id: str
    source_version: str
    raw_sha256: str
    adapter: str
    accepted_count: int
    rejected_count: int
    scanned_count: int
    warnings: list[str]
    rejected: list[dict]
    preview: list[dict]
    record_fingerprint: str
    eligible_to_stage: bool
    stage_reason: str

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "source_version": self.source_version,
            "raw_sha256": self.raw_sha256,
            "adapter": self.adapter,
            "scanned_count": self.scanned_count,
            "accepted_count": self.accepted_count,
            "rejected_count": self.rejected_count,
            "warnings": self.warnings,
            "rejected": self.rejected,
            "preview": self.preview,
            "record_fingerprint": self.record_fingerprint,
            "eligible_to_stage": self.eligible_to_stage,
            "stage_reason": self.stage_reason,
        }


def _json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def _fingerprint(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _path(root: Path, local: str, *, within: str) -> Path:
    candidate = (root / local).resolve(strict=False)
    allowed = (root / within).resolve(strict=False)
    if not candidate.is_relative_to(allowed) or candidate == allowed:
        raise ValueError(f"Path must remain under {within}")
    return candidate


def _gate(source: dict, manifest: SourceManifest) -> tuple[bool, str]:
    if source["dataset_role"] != manifest.dataset_role:
        raise ValueError("Source role mismatch")
    if source.get("local_path") not in {manifest.local_path, str(Path(manifest.local_path).parent)}:
        raise ValueError("Source path not in reviewed register")
    if source.get("sha256") and source["sha256"] != manifest.file_sha256:
        raise ValueError("Source registry checksum mismatch")
    if source.get("rights_status") != manifest.rights_status:
        raise ValueError("Source rights status mismatch")
    if not set(manifest.approved_uses).issubset(set(source.get("approved_use", []))):
        raise ValueError("Manifest approved uses exceed source register")
    if not source["enabled_for_import"]:
        return False, "Registered source is disabled; preview only"
    if (
        not manifest.rights_status.startswith("verified_")
        or not manifest.approved_uses
    ):
        return False, "Dataset reuse approval is incomplete"
    if manifest.dataset_role in {"individual_earnings", "advertised_salary"} and (
        "train_on_labeled_salary" not in manifest.approved_uses
    ):
        return False, "No explicit permission to stage supervised salary labels"
    return True, "Approved for local staging only; no public release"


def _parse(path: Path, manifest: SourceManifest, root: Path) -> tuple[str, Batch]:
    name = path.name
    if manifest.source_id == "psa_ows_2022_public_tables" and name.lower().endswith(".pdf"):
        return "psa_ows_2022_pdf", parse_ows_2022(path, manifest)
    if manifest.source_id == "esco_english_bundle" and name in {
        "occupations_en.csv", "skills_en.csv",
    }:
        return "esco_csv_concepts", parse_esco(path, manifest)
    if manifest.source_id == "esco_english_bundle" and name == "occupationSkillRelations_en.csv":
        return "esco_csv_relationships", parse_esco_relations(path, manifest, root)
    if manifest.source_id == "psa_psoc_2022_update" and name == "2022-psoc.xlsx":
        return "psoc_hierarchy_xlsx", parse_psoc_xlsx(path, manifest)
    if manifest.dataset_role in {"individual_earnings", "advertised_salary"}:
        return "approved_salary_rows", parse_salary_csv(path, manifest)
    raise ValueError("No adapter for this source ID, file and cohort combination")


def inspect(
    root: Path, manifest_path: Path, source_id: str, *, input_path: str | None = None,
    preview_limit: int = 3,
) -> tuple[IngestReport, list[dict]]:
    root = root.resolve(strict=True)
    manifest_location = _path(root, str(manifest_path), within="configs")
    manifest = SourceManifest.from_dict(
        json.loads(manifest_location.read_text(encoding="utf-8"))
    )
    if source_id != manifest.source_id:
        raise ValueError("Requested source does not match manifest")
    source_register = load_source_register(root / "configs/data_sources.json")
    issues = validate_source_register(source_register)
    if issues:
        raise ValueError(f"Source registry invalid: {issues[0]}")
    sources = [row for row in source_register["sources"] if row["id"] == source_id]
    if len(sources) != 1:
        raise ValueError("Source ID not uniquely registered")
    source = sources[0]
    allowed, reason = _gate(source, manifest)

    local = input_path or manifest.local_path
    if local != manifest.local_path:
        raise ValueError("Input override must match the reviewed manifest path")
    path = _path(root, local, within="data/raw")
    if not path.is_file():
        raise ValueError("Source file unavailable")
    if path.suffix.lower() != "." + manifest.file_format:
        raise ValueError("Source format does not match manifest")
    with path.open("rb") as file:
        fingerprint = hashlib.file_digest(file, "sha256").hexdigest()
    if fingerprint != manifest.file_sha256:
        raise ValueError("Source checksum mismatch; review the new version")
    if not 0 <= preview_limit <= 20:
        raise ValueError("Preview count must be 0–20")

    adapter, batch = _parse(path, manifest, root)
    if batch.accepted and batch.scanned != len(batch.accepted) + len(batch.rejected):
        # Revisions are counted in scanned lines, but collapse into one concept.
        if adapter != "esco_csv_concepts":
            raise ValueError("Adapter lost a source row")
    records = [
        {
            "record": item["record"],
            "lineage": {
                "source_id": source_id,
                "source_version": manifest.source_version,
                "raw_sha256": fingerprint,
                "source_locator": item["locator"],
                "rights_status": manifest.rights_status,
                "adapter": adapter,
                **({"source_occupation_code": item["original_code"]}
                   if "original_code" in item else {}),
            },
        }
        for item in batch.accepted
    ]
    records.sort(key=lambda item: (
        item["lineage"]["source_locator"],
        json.dumps(item["record"], sort_keys=True),
    ))
    payload = b"".join(_json(row) for row in records)
    # Preview leaves the original values on disk. Only show canonical public/taxonomy samples.
    examples = [item["record"] for item in records[:preview_limit]]
    if manifest.dataset_role in {"individual_earnings", "advertised_salary"}:
        examples = [{"record_type": item["record"]["salary_observation"]["record_type"],
                     "normalized": item["record"]["normalized"]["calculation"]}
                    for item in records[:preview_limit]]
    report = IngestReport(
        source_id=source_id,
        source_version=manifest.source_version,
        raw_sha256=fingerprint,
        adapter=adapter,
        accepted_count=len(records),
        rejected_count=len(batch.rejected),
        scanned_count=batch.scanned,
        warnings=batch.warnings,
        rejected=batch.rejected,
        preview=examples,
        record_fingerprint=_fingerprint(payload),
        eligible_to_stage=allowed,
        stage_reason=reason,
    )
    return report, records


def stage(
    root: Path, report: IngestReport, records: list[dict], *, output: str = "data/staged",
) -> dict:
    if not report.eligible_to_stage:
        raise PermissionError(report.stage_reason)
    root = root.resolve(strict=True)
    folder = _path(root, output, within="data/staged") if output != "data/staged" else (
        root / "data/staged"
    ).resolve()
    if not folder.is_relative_to((root / "data/staged").resolve()):
        raise ValueError("Invalid staging folder")
    if folder.exists() and folder.is_symlink():
        raise ValueError("Refusing a symlink staging folder")
    if not records:
        raise ValueError("Nothing to stage")
    folder.mkdir(parents=True, exist_ok=True)
    name = f"{report.source_id}-{report.record_fingerprint[:16]}"
    records_path = folder / f"{name}.jsonl"
    payload = b"".join(_json(item) for item in records)
    if _fingerprint(payload) != report.record_fingerprint:
        raise ValueError("Records changed after inspection")
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    metadata = {
        "source_id": report.source_id,
        "source_version": report.source_version,
        "source_sha256": report.raw_sha256,
        "staged_at_utc": now,
        "record_fingerprint": report.record_fingerprint,
        "accepted_count": report.accepted_count,
        "rejected_count": report.rejected_count,
        "release_status": "not_approved_for_runtime",
        "records_path": records_path.relative_to(root).as_posix(),
    }
    report_path = folder / f"{name}.report.json"
    rejects_path = folder / f"{name}.rejects.json"
    manifest_path = folder / f"{name}.lineage.json"
    extra: dict[str, str] = {}
    if report.adapter == "psa_ows_2022_pdf":
        csv_data = _benchmark_csv(records)
        csv_path = folder / f"{name}.benchmarks.csv"
        _write_once(csv_path, csv_data)
        extra = {
            "benchmark_csv": csv_path.relative_to(root).as_posix(),
            "benchmark_csv_sha256": _fingerprint(csv_data),
        }
    _write_once(records_path, payload)
    _write_once(rejects_path, _json(report.rejected))
    _write_once(report_path, _json(report.to_dict()))
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        for item in (
            "source_id", "source_version", "source_sha256", "record_fingerprint",
            "accepted_count", "rejected_count", "records_path", "release_status",
        ):
            if existing.get(item) != metadata[item]:
                raise ValueError("Existing staged lineage conflicts with this run")
    else:
        _write_once(manifest_path, _json(metadata))
    return {
        "records": records_path.relative_to(root).as_posix(),
        "rejects": rejects_path.relative_to(root).as_posix(),
        "report": report_path.relative_to(root).as_posix(),
        "lineage": manifest_path.relative_to(root).as_posix(),
        "fingerprint": report.record_fingerprint,
        **extra,
    }


def _benchmark_csv(records: list[dict]) -> bytes:
    fields = [
        "observation_id", "source_id", "source_version", "reference_date",
        "source_table", "source_page", "population", "published_dimensions",
        "occupation_psoc_code", "occupation_label", "region_psgc_code",
        "region_label", "industry_code", "industry_label", "metric",
        "pay_component", "tax_basis", "currency", "period", "amount_php_monthly",
        "source_occupation_code", "source_locator", "raw_sha256",
    ]
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for item in records:
        record = item["record"]
        lineage = item["lineage"]
        row = {
            **record,
            "published_dimensions": "+".join(record["published_dimensions"]),
            "source_occupation_code": lineage.get("source_occupation_code"),
            "source_locator": lineage["source_locator"],
            "raw_sha256": lineage["raw_sha256"],
        }
        writer.writerow({field: row.get(field) for field in fields})
    return output.getvalue().encode("utf-8")


def _write_once(path: Path, payload: bytes) -> None:
    try:
        with path.open("xb") as file:
            file.write(payload)
    except FileExistsError:
        if path.read_bytes() != payload:
            raise FileExistsError(f"Staged immutable artifact differs: {path.name}") from None
