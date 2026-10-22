from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

from .baseline import BenchmarkOnlyBaseline, encode_report


def write_report(root: Path, report: dict) -> dict:
    root = root.resolve(strict=True)
    directory = root / "data/audit"
    if directory.is_symlink():
        raise ValueError("Audit directory cannot be a symlink")
    directory.mkdir(parents=True, exist_ok=True)
    body = encode_report(report)
    digest = hashlib.sha256(body).hexdigest()
    target = directory / f"part07-benchmark-only-{digest[:16]}.json"
    try:
        with target.open("xb") as file:
            file.write(body)
    except FileExistsError:
        if target.read_bytes() != body:
            raise ValueError("Existing baseline report has different bytes")
    return {
        "path": target.relative_to(root).as_posix(),
        "sha256": digest,
        "report_status": report["supervised_status"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Part 07: offline source-backed benchmark preview; no ML training"
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--as-of", type=date.fromisoformat, default=date.today())
    parser.add_argument("--occupation", help="Exact original 2022 survey occupation label")
    parser.add_argument("--industry", help="Exact 2022 Table 1 industry scope")
    parser.add_argument("--region", help="Exact 2022 Table 3 region label")
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args(argv)
    try:
        baseline = BenchmarkOnlyBaseline(args.root, as_of=args.as_of)
        report = baseline.reproducible_report()
        output = {
            "baseline_kind": report["baseline_kind"],
            "supervised_status": report["supervised_status"],
            "runtime_release": report["runtime_release"],
            "dataset_snapshot_sha256": report["dataset_snapshot_sha256"],
            "published_cells": report["published_cells"],
            "model_trained": False,
            "mae_php": None,
        }
        if args.occupation:
            output["published_cell_preview"] = baseline.lookup(
                args.occupation, industry_code=args.industry, region_label=args.region
            ).to_dict()
            output["runtime_result"] = baseline.runtime_result(
                args.occupation, industry_code=args.industry, region_label=args.region
            ).to_dict()
        elif args.industry is not None or args.region is not None:
            raise ValueError("Occupation must accompany source industry or region")
        if args.write_report:
            output["report"] = write_report(args.root, report)
        print(json.dumps(output, indent=2, ensure_ascii=False))
        return 0
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as exc:
        print(f"Part 07 benchmark preview blocked: {exc}")
        return 2


def dispatch(argv: list[str] | None = None) -> int:
    import sys

    from .operator_cli import main as operator_main

    args = sys.argv[1:] if argv is None else argv
    if args and args[0] == "experimental":
        from .experimental_cli import main as experiment_main

        return experiment_main(args[1:])
    if args and args[0] == "ingest":
        from salary_predictor.ingest.__main__ import main as ingest_main

        return ingest_main(args[1:])
    if args and args[0] == "audit":
        from salary_predictor.audit.__main__ import main as audit_main

        return audit_main(args[1:])
    if args and args[0] in {
        "sources", "train", "evaluate", "prepare", "verify", "promote",
        "rollback", "active",
    }:
        return operator_main(args)
    if args and args[0] in {
        "--root", "--as-of", "--occupation", "--industry", "--region", "--write-report",
    }:
        return main(args)
    return operator_main(args)


if __name__ == "__main__":
    raise SystemExit(dispatch())
