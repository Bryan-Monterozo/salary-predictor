from __future__ import annotations

import json
from pathlib import Path

SOURCE_ROLES = {
    "aggregate_benchmark",
    "taxonomy",
    "statutory_minimum",
    "restricted_microdata_candidate",
    "individual_earnings",
    "advertised_salary",
}

REQUIRED_FIELDS = {
    "id",
    "name",
    "owner",
    "source_url",
    "rights_status",
    "access_status",
    "dataset_role",
    "format",
    "reference_date",
    "retrieved_date",
    "refresh_frequency",
    "population",
    "observation_unit",
    "target_definition",
    "currency",
    "wage_period",
    "available_features",
    "unsupported_features",
    "approved_use",
    "public_benchmark_eligible",
    "enabled_for_import",
}


def load_source_register(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_source_register(register: dict) -> list[str]:
    errors: list[str] = []
    if register.get("schema_version") != 1:
        errors.append("Unknown source register version")

    sources = register.get("sources")
    if not isinstance(sources, list) or not sources:
        return errors + ["No sources in register"]

    seen: set[str] = set()
    for position, source in enumerate(sources):
        if not isinstance(source, dict):
            errors.append(f"Source {position}: expected an object")
            continue

        name = source.get("id") or f"source_{position}"
        missing = REQUIRED_FIELDS - source.keys()
        if missing:
            errors.append(f"{name}: missing {', '.join(sorted(missing))}")
            continue

        if not isinstance(name, str) or not name.strip() or name in seen:
            errors.append(f"{name}: empty or repeated source ID")
        seen.add(str(name))

        if source["dataset_role"] not in SOURCE_ROLES:
            errors.append(f"{name}: unknown dataset role")
        for field in ("available_features", "unsupported_features", "approved_use"):
            if not isinstance(source[field], list) or any(
                not isinstance(value, str) for value in source[field]
            ):
                errors.append(f"{name}: {field} must be a list of strings")
        for field in ("public_benchmark_eligible", "enabled_for_import"):
            if not isinstance(source[field], bool):
                errors.append(f"{name}: {field} must be true or false")
        if source["enabled_for_import"] and (
            not source["source_url"]
            or not source["rights_status"].startswith("verified_")
            or not source["approved_use"]
            or not source["reference_date"]
            or not source["available_features"]
        ):
            errors.append(f"{name}: enabled source needs verified rights and metadata")

        if source["dataset_role"] == "aggregate_benchmark":
            if (source["currency"], source["wage_period"]) != ("PHP", "monthly"):
                errors.append(f"{name}: published salary basis is unclear")
            if not source["target_definition"] or not source["observation_unit"]:
                errors.append(f"{name}: missing wage basis or observation unit")
        if source["public_benchmark_eligible"] and (
            source["dataset_role"] != "aggregate_benchmark"
            or "public_benchmark_with_attribution" not in source["approved_use"]
            or not source["rights_status"].startswith("verified_")
        ):
            errors.append(f"{name}: benchmark has no approved public source")

        if source["dataset_role"] in {"individual_earnings", "advertised_salary"}:
            if "salary" not in source["available_features"]:
                errors.append(f"{name}: cannot train without observed salary")

    return errors
