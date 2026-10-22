from __future__ import annotations

import argparse
import json
from datetime import date
from decimal import Decimal
from pathlib import Path

from .consolidated import (
    _policy,
    consolidate,
    export_consolidated,
    preview_career_match,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Consolidate local Jobicy evidence without fabricating training labels"
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--as-of", type=date.fromisoformat, default=date.today())
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--job-title")
    parser.add_argument("--skill", action="append", default=[])
    parser.add_argument("--years-experience", type=Decimal)
    parser.add_argument("--current-salary-php", type=Decimal)
    args = parser.parse_args(argv)
    try:
        report, rows, queue = consolidate(args.root, as_of=args.as_of)
        answer = {"consolidation": report}
        if args.job_title:
            policy = _policy(args.root)
            answer["career_research_preview"] = preview_career_match(
                rows, job_title=args.job_title,
                skills=args.skill,
                years_experience=args.years_experience,
                current_salary_php=args.current_salary_php,
                min_support=policy["minimum_role_support_for_path_suggestion"],
            )
        elif args.skill or args.years_experience is not None or args.current_salary_php is not None:
            raise ValueError("--job-title is required with user information")
        if args.write:
            answer["local_export"] = export_consolidated(args.root, report, rows, queue)
        print(json.dumps(answer, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"Consolidation blocked: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
