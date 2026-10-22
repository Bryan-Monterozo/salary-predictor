from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from .experimental_multisource import train_experiment, verify_experiment
from .experimental_predict import predict_experimental


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Personal local research prototype: NOT the production salary release"
    )
    sub = parser.add_subparsers(dest="task", required=True)
    for command in ("train", "verify", "predict"):
        p = sub.add_parser(command)
        p.add_argument("--root", type=Path, default=Path.cwd())
        p.add_argument("--run-id")
        p.add_argument("--as-of", type=date.fromisoformat, default=date.today())
        if command == "predict":
            p.add_argument("--job-title", required=True)
            p.add_argument("--skill", action="append", default=[])
            p.add_argument("--years-experience", type=float)
            p.add_argument("--current-salary-php", type=float)
            p.add_argument("--usd-to-php", type=float)
            p.add_argument("--seniority", default="Any")
            p.add_argument("--isco-major")
            p.add_argument("--step-region-code")
            p.add_argument("--education-years", type=float)
            p.add_argument("--psa-occupation")
            p.add_argument("--psa-industry")
            p.add_argument("--psa-region")
            p.add_argument("--scenario-annual-growth", type=float)
    args = parser.parse_args(argv)
    try:
        if args.task == "train":
            if args.run_id:
                raise ValueError("Train creates a new content-addressed ID")
            out = train_experiment(args.root, as_of=args.as_of)
        elif args.task == "verify":
            metadata, location = verify_experiment(args.root, run_id=args.run_id)
            out = {
                "status": "VERIFIED_LOCAL_ONLY",
                "run_id": location.name,
                "selected": metadata["selected"],
                "heldout": metadata["test_selected"],
                "production_release_approved": False,
            }
        else:
            out = predict_experimental(
                args.root, job_title=args.job_title, skills=args.skill,
                years_experience=args.years_experience,
                current_salary_php=args.current_salary_php,
                usd_to_php=args.usd_to_php, seniority=args.seniority,
                isco_major=args.isco_major, step_region_code=args.step_region_code,
                education_years=args.education_years,
                psa_occupation=args.psa_occupation,
                psa_industry=args.psa_industry, psa_region=args.psa_region,
                scenario_annual_growth=args.scenario_annual_growth,
                run_id=args.run_id, as_of=args.as_of,
            )
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, TypeError, OSError, KeyError, json.JSONDecodeError) as exc:
        print(f"Experimental local model blocked: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
