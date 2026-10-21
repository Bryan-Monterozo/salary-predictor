from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from .quality import (
    audit_benchmarks,
    blocked_split_manifest,
    leakage_checklist,
    load_policy,
    load_staged_benchmarks,
)


def _encode(obj: dict) -> bytes:
    return (json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def run_audit(
    root: Path, staged: str, *, as_of: date,
) -> dict[str, dict]:
    policy = load_policy(root)
    rows, provenance = load_staged_benchmarks(root, staged)
    products = audit_benchmarks(rows, provenance, policy, as_of)
    audit = products["audit"]
    if audit["quality_status"] != "PASS_BENCHMARK_ONLY":
        raise ValueError("Source has duplicate cells or mixed vintages; audit blocked")
    products["split"] = blocked_split_manifest(provenance["fingerprint"], policy)
    products["leakage"] = leakage_checklist(provenance["fingerprint"], audit, policy)
    return products


def _write_immutable(path: Path, payload: bytes) -> None:
    try:
        with path.open("xb") as file:
            file.write(payload)
    except FileExistsError:
        if path.read_bytes() != payload:
            raise ValueError(f"Existing audit file differs: {path.name}") from None


def write_audit(root: Path, products: dict[str, dict]) -> dict[str, str]:
    root = root.resolve()
    folder = root / "data/audit"
    if folder.is_symlink():
        raise ValueError("Audit directory cannot be a symlink")
    folder.mkdir(parents=True, exist_ok=True)
    info = products["audit"]
    stem = f"{info['source_id']}-{info['snapshot_sha256'][:16]}-{info['as_of']}"
    created = {}
    for label, data in sorted(products.items()):
        path = folder / f"{stem}.{label}.json"
        _write_immutable(path, _encode(data))
        created[label] = path.relative_to(root).as_posix()
    return created


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit staged salary evidence without training")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--staged", required=True, help="Existing reviewed staged benchmark JSONL")
    parser.add_argument("--as-of", default=date.today().isoformat(), help="YYYY-MM-DD")
    parser.add_argument("--write", action="store_true", help="Write immutable local audit outputs")
    args = parser.parse_args(argv)
    try:
        as_of = date.fromisoformat(args.as_of)
        results = run_audit(args.root, args.staged, as_of=as_of)
        audit = results["audit"]
        summary = {
            "audit_status": audit["quality_status"],
            "snapshot_sha256": audit["snapshot_sha256"],
            "as_of": audit["as_of"],
            "published_cells": audit["accepted_cells"],
            "tables": audit["tables"],
            "outlier_review_flags": audit["outliers_flagged_count"],
            "staleness_warning": audit["staleness_warning"],
            "model_splits": results["split"]["status"],
            "training_or_model_evaluation": False,
        }
        if args.write:
            summary["outputs"] = write_audit(args.root, results)
        print(json.dumps(summary, indent=2))
        return 0
    except (ValueError, OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"Audit blocked: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
