from __future__ import annotations

import argparse
import json
from pathlib import Path

from .jobicy import (
    collect_available,
    export_unique_salary_ranges,
    fetch_archived,
    inspect_archive,
)
from .step import inspect_step, inspect_step_questionnaire, write_aggregate_report


def run(root: Path) -> dict:
    root = root.resolve(strict=True)
    manifest = json.loads((root / "configs/local_raw_file_manifest.json").read_text())
    return {
        "status": "READ_ONLY_FEASIBILITY_NO_MODEL_TRAINING",
        "sources": {
            "step": inspect_step(root, manifest),
            "step_questionnaire": inspect_step_questionnaire(root, manifest),
            "jobicy": inspect_archive(root),
            "psa_ows": {
                "cohort": "published_aggregate_benchmark",
                "current_staged_snapshot": "25a47e4e13d22b33",
                "published_cells": 560,
                "not_individual_salary_records": True,
            },
        },
        "decisions": {
            "step_individual_hourly_research": "BLOCKED_PENDING_WEIGHTING_AND_USAGE_RIGHTS",
            "jobicy_advertised_salary_baseline": "BLOCKED_SPARSE_APAC_DISCLOSED_PAY_AND_MODEL_RIGHTS",
            "psa_published_benchmark_only": "ELIGIBLE_FOR_FUTURE_EXACT_CELL_RELEASE_REVIEW",
            "part07_model_training_started": False,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="STEP and archived Jobicy research preview; no model training"
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--fetch-jobicy", choices=["api", "rss"])
    parser.add_argument("--collect-jobicy", action="store_true", help="Collect unique jobs using all cursor pages")
    parser.add_argument("--target", type=int, default=5000)
    parser.add_argument("--max-pages", type=int, default=60)
    parser.add_argument("--geo", default=None, help="Optional eligibility slug; omit for all listings")
    parser.add_argument("--salary-preview", type=int, default=20)
    parser.add_argument("--export-salaries", action="store_true", help="Write deduplicated local valid-salary JSON under data/audit")
    parser.add_argument("--cursor", default=None)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--write-aggregate", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.fetch_jobicy and args.collect_jobicy:
            raise ValueError("Choose fetch one page OR collect all pages")
        if args.collect_jobicy:
            result = collect_available(
                args.root, target=args.target, count=args.count,
                geo=args.geo, max_pages=args.max_pages,
            )
            print(json.dumps({"collection": result}, indent=2))
        if args.fetch_jobicy:
            archive = fetch_archived(
                args.root, kind=args.fetch_jobicy, count=args.count,
                cursor=args.cursor, geo=args.geo,
            )
            archive.pop("next_cursor", None)
            print(json.dumps({"archived": archive}, indent=2))
        if not 0 <= args.salary_preview <= 50:
            raise ValueError("Salary preview must be between 0 and 50")
        summary = run(args.root)
        if args.export_salaries:
            exported = export_unique_salary_ranges(args.root)
            print(json.dumps({"salary_export": exported}, indent=2))
        if args.write_aggregate:
            path = write_aggregate_report(args.root, summary)
            print(f"Aggregate report saved at {path.relative_to(args.root)}")
        print(json.dumps({
            "status": summary["status"],
            "step_respondents": summary["sources"]["step"]["respondents"],
            "step_hourly_wage_worker_candidates": summary["sources"]["step"][
                "salary_label_candidate_wage_workers"
            ],
            "jobicy_jobs": summary["sources"]["jobicy"]["distinct_job_ids"],
            "jobicy_valid_salary_ranges": summary["sources"]["jobicy"][
                "salary_disclosed_valid_range"
            ],
            "salary_range_preview": summary["sources"]["jobicy"]["salary_range_preview"][
                :args.salary_preview
            ],
            "decisions": summary["decisions"],
        }, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"Feasibility audit blocked: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
