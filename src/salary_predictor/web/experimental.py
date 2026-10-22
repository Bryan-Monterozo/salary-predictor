from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

SENIORITIES = ("Any", "Senior", "Director", "Midweight", "Entry-Level, Junior")
EXTRA_FIELDS = (
    "usd_to_php", "current_salary_php", "seniority", "isco_major",
    "step_region_code", "education_years", "psa_occupation",
    "psa_industry", "psa_region", "scenario_annual_growth",
)
MAX_WORKER_BYTES = 256 * 1024
WORKER_TIMEOUT = 60


def parameters(form) -> tuple[dict, dict[str, str], dict[str, str]]:
    raw = {field: str(form.get(field, "")).strip() for field in EXTRA_FIELDS}
    errors: dict[str, str] = {}

    def optional_float(field: str, minimum: float, maximum: float) -> float | None:
        value = raw[field]
        if not value:
            return None
        try:
            parsed = float(value)
            if not math.isfinite(parsed) or not minimum <= parsed <= maximum:
                raise ValueError
        except ValueError:
            errors[field] = f"Enter a value from {minimum:g} to {maximum:g}."
            return None
        return parsed

    values = {
        "usd_to_php": optional_float("usd_to_php", 1, 150),
        "current_salary_php": optional_float("current_salary_php", 0.01, 100_000_000),
        "education_years": optional_float("education_years", 0, 30),
        "scenario_annual_growth": optional_float("scenario_annual_growth", -0.5, 0.5),
        "seniority": raw["seniority"] or "Any",
        "isco_major": raw["isco_major"] or None,
        "step_region_code": raw["step_region_code"] or None,
        "psa_occupation": raw["psa_occupation"] or None,
        "psa_industry": raw["psa_industry"] or None,
        "psa_region": raw["psa_region"] or None,
    }
    if values["seniority"] not in SENIORITIES:
        errors["seniority"] = "Choose a listed advertised seniority proxy."
    if values["isco_major"] is not None and values["isco_major"] not in set("123456789"):
        errors["isco_major"] = "Choose an ISCO major group (1–9)."
    if values["step_region_code"] is not None and (
        not values["step_region_code"].isdigit()
        or not 1 <= int(values["step_region_code"]) <= 16
    ):
        errors["step_region_code"] = "Choose a historical STEP region code (1–16)."
    for field, maximum in (("psa_occupation", 120), ("psa_industry", 50), ("psa_region", 100)):
        if values[field] is not None and len(values[field]) > maximum:
            errors[field] = "Text is too long."
    if values["psa_industry"] and values["psa_region"]:
        errors["psa_region"] = "Use either source industry or source region, not both."
    if values["psa_occupation"] and not (values["psa_industry"] or values["psa_region"]):
        errors["psa_occupation"] = "Choose a published source industry or region too."
    if (values["psa_industry"] or values["psa_region"]) and not values["psa_occupation"]:
        errors["psa_occupation"] = "Enter the exact published PSA occupation label."
    if any(
        values[k] is not None for k in ("isco_major", "step_region_code", "education_years")
    ) and not (
        values["isco_major"] and values["step_region_code"]
        and values["education_years"] is not None
    ):
        errors.setdefault("step_region_code", "STEP requires ISCO major, historical region and education years.")
    return values, raw, errors


def worker_predict(root: Path, payload: dict) -> dict:
    body = json.dumps(payload, ensure_ascii=True, allow_nan=False).encode("utf-8")
    if len(body) > 6_000:
        raise ValueError("Too much experimental input")
    try:
        process = subprocess.run(
            [sys.executable, "-m", "salary_predictor.web.experimental_worker"],
            input=body,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            cwd=root,
            timeout=WORKER_TIMEOUT,
            check=False,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError("The local experimental calculation could not finish.") from exc
    if process.returncode != 0 or len(process.stdout) > MAX_WORKER_BYTES:
        raise ValueError("The local experimental calculation is unavailable.")
    try:
        result = json.loads(process.stdout)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError("The local experimental calculation returned invalid data.") from exc
    if not isinstance(result, dict) or result.get("mode") != "EXPERIMENTAL_LOCAL_ONLY":
        raise ValueError("No verified experimental model is currently available.")
    if result.get("model_release_approved") is not False:
        raise ValueError("Unexpected experimental artifact status.")
    return result
