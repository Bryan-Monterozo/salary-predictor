from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path


def load_draft_weighting(root: Path) -> dict:
    config = json.loads(
        (root / "configs/feasibility/jobicy_weighting.json").read_text(encoding="utf-8")
    )
    if config.get("schema_version") != 1 or config.get("training_enabled") is not False:
        raise ValueError("Jobicy weighting must remain an unapproved draft")
    if config.get("cohort") != "jobicy_advertised_remote_salaries_only":
        raise ValueError("The draft weighting applies only to Jobicy advertised salaries")
    weights = config.get("rules", {})
    for key in ("apac_eligible_weight", "non_apac_or_unknown_weight", "anywhere_weight"):
        number = Decimal(weights[key])
        if not number.is_finite() or number <= 0 or number > 5:
            raise ValueError("Invalid experimental weighting")
    return config


def geo_priority(job_geo: str, policy: dict) -> dict:
    if not isinstance(job_geo, str):
        raise ValueError("Jobicy jobGeo must be text")
    segments = {token.strip().casefold() for token in job_geo.split(",") if token.strip()}
    known_apac = {token.casefold() for token in policy["apac_geographic_tokens"]}
    if segments & known_apac:
        return {
            "bucket": "APAC_ELIGIBLE",
            "proposed_sample_weight": policy["rules"]["apac_eligible_weight"],
            "training_enabled": False,
        }
    if segments == {"anywhere"}:
        bucket, key = "GLOBAL_ANYWHERE", "anywhere_weight"
    else:
        bucket, key = "OTHER_OR_UNKNOWN", "non_apac_or_unknown_weight"
    return {
        "bucket": bucket,
        "proposed_sample_weight": policy["rules"][key],
        "training_enabled": False,
    }
