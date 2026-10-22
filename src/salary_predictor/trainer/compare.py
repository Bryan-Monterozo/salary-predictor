from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from .experiment import immutable_report, readiness_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Part 08 candidate comparison readiness; never trains a model"
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--as-of", type=date.fromisoformat, default=date.today())
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = readiness_report(args.root, as_of=args.as_of)
        summary = {
            "part": report["part"],
            "status": report["status"],
            "source_integrity": report["source_integrity"],
            "approved_training_rows": report["eligible_supervised_training_rows"],
            "STEP_hourly_candidates_not_approved": report["cohort_inspection"]["STEP"][
                "observed_hourly_wage_worker_candidates"
            ],
            "Jobicy_valid_remote_salary_ranges_not_approved": (
                report["cohort_inspection"]["Jobicy"]["valid_advertised_salary_ranges"]
            ),
            "candidates": report["candidates"],
            "model_trained": False,
            "mae_php": None,
            "benchmark_path": report["benchmark_path"],
        }
        if args.write_report:
            summary["report"] = immutable_report(args.root, report)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"Part 08 readiness blocked by source validation: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
