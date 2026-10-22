from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from salary_predictor.contracts import UserProfile

from .decision import LocalResultResolver, write_policy_audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Part 10: local fail-closed salary routing; does not promote artifacts"
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--as-of", type=date.fromisoformat, default=date.today())
    parser.add_argument("--write-report", action="store_true")
    parser.add_argument("--psoc", default=None)
    parser.add_argument("--psgc", default=None)
    parser.add_argument("--experience", default=None)
    parser.add_argument("--education", default=None)
    parser.add_argument("--industry", default=None)
    parser.add_argument("--employment-context", default=None)
    parser.add_argument("--published-occupation-label", default=None)
    parser.add_argument("--published-industry-code", default=None)
    parser.add_argument("--published-region-label", default=None)
    args = parser.parse_args(argv)
    try:
        profile = UserProfile.from_dict({
            "feature_schema_version": 1,
            "occupation_psoc_code": args.psoc,
            "region_psgc_code": args.psgc,
            "years_experience": args.experience,
            "education_level": args.education,
            "normalized_skills": [],
            "industry_code": args.industry,
            "employment_context": args.employment_context,
        })
        resolver = LocalResultResolver(args.root, as_of=args.as_of)
        decision = resolver.local_diagnostic(
            profile,
            published_occupation_label=args.published_occupation_label,
            published_industry_code=args.published_industry_code,
            published_region_label=args.published_region_label,
        )
        if args.write_report:
            decision["policy_audit_report"] = write_policy_audit(
                args.root, resolver.policy_audit()
            )
        print(json.dumps(decision, indent=2, ensure_ascii=False))
        return 0
    except (ValueError, OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"Part 10 local result resolver blocked: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
