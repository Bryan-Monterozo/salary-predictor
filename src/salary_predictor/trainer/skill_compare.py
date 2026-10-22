from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from .skill_ablation import immutable_report, local_skills_preview, readiness


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Part 09 skill feature readiness and explicit skill lookup; no model training"
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--as-of", type=date.fromisoformat, default=date.today())
    parser.add_argument("--write-report", action="store_true")
    parser.add_argument("--skill", action="append", default=None, help="Explicit user skill")
    parser.add_argument(
        "--origin",
        choices=["user_explicit", "resume_explicit_reviewed", "job_ad_explicit_reviewed"],
        default="user_explicit",
    )
    args = parser.parse_args(argv)
    try:
        status = readiness(args.root, as_of=args.as_of)
        output = {
            "part": "09",
            "status": status["status"],
            "paired_training_rows": status["approved_salary_plus_observed_skill_training_rows"],
            "with_skills_model": status["with_skills_model"],
            "without_skills_model": status["without_skills_model"],
            "mae_delta_php": status["delta_mae_php"],
            "skill_salary_feature_enabled": False,
            "skill_search_and_extraction_enabled": True,
            "runtime_recommendation": status["runtime_recommendation"],
        }
        if args.skill is not None:
            output["explicit_skill_preview"] = local_skills_preview(
                args.root, args.skill, origin=args.origin
            )
        if args.write_report:
            output["report"] = immutable_report(args.root, status)
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"Part 09 readiness blocked by validation: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
