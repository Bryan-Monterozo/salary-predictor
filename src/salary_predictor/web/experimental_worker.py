from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    try:
        payload = json.loads(sys.stdin.buffer.read(6001))
        if not isinstance(payload, dict) or set(payload) != {
            "job_title", "skills", "years_experience", "current_salary_php",
            "usd_to_php", "seniority", "isco_major", "step_region_code",
            "education_years", "psa_occupation", "psa_industry", "psa_region",
            "scenario_annual_growth",
        }:
            return 2
        from salary_predictor.trainer.experimental_predict import predict_experimental

        prediction = predict_experimental(Path.cwd(), **payload)
        sys.stdout.write(json.dumps(prediction, ensure_ascii=True, allow_nan=False))
        return 0
    except (ValueError, TypeError, OSError, KeyError, ImportError, RuntimeError):
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
