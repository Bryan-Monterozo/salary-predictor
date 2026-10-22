from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path

from salary_predictor.source_registry import load_source_register, validate_source_register

COHORTS = {
    "advertised_salary": "advertised_salary",
    "individual_earnings": "individual_earnings",
}
TARGET_BASIS = {
    "advertised_salary": "advertised_range_midpoint",
    "individual_earnings": "individual_gross_total",
}
REQUIRED = {
    "observation_id", "source_id", "cohort", "job_title", "occupation_code",
    "region_code", "market_country", "market_country_verified", "employer_group",
    "duplicate_cluster_id", "observation_date", "salary_php_monthly",
    "currency", "period", "salary_basis", "normalization_evidence",
    "years_experience", "education_level", "employment_context",
}
FORBIDDEN_INPUTS = {"predicted_salary", "imputed_salary", "model_estimate", "ai_estimate"}


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _inside_curated(root: Path, relative: str, *, must_exist: bool = True) -> Path:
    if not isinstance(relative, str) or not relative.startswith("data/curated/"):
        raise ValueError("Reviewed training inputs must live under data/curated/")
    safe_root = (root / "data/curated").resolve(strict=False)
    result = (root / relative).resolve(strict=must_exist)
    if not result.is_relative_to(safe_root) or result == safe_root:
        raise ValueError("Unsafe reviewed dataset path")
    return result


def _positive(value: object) -> Decimal:
    if isinstance(value, (bool, dict, list)) or value is None:
        raise ValueError("Salary label must be an observed finite positive amount")
    try:
        amount = Decimal(str(value))
    except (ValueError, InvalidOperation) as exc:
        raise ValueError("Invalid salary label") from exc
    if not amount.is_finite() or amount <= 0 or amount > 100_000_000:
        raise ValueError("Salary label out of reviewed range")
    return amount


def reviewed_rows(root: Path, manifest_path: str) -> tuple[dict, list[dict]]:
    root = root.resolve(strict=True)
    approved_path = _inside_curated(root, manifest_path)
    if approved_path.suffix != ".json":
        raise ValueError("Reviewed manifest must be a JSON file")
    manifest = json.loads(approved_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != 1
        or manifest.get("status") != "REVIEWED_FOR_SUPERVISED_RESEARCH"
        or manifest.get("independent_rights_review") is not True
        or not isinstance(manifest.get("reviewed_by"), str)
        or len(manifest["reviewed_by"].strip()) < 3
        or not isinstance(manifest.get("rights_evidence_reference"), str)
        or not manifest["rights_evidence_reference"].strip()
        or not isinstance(manifest.get("source_ids"), list)
        or not manifest["source_ids"]
        or len(set(manifest["source_ids"])) != len(manifest["source_ids"])
        or manifest.get("cohort") not in COHORTS
        or manifest.get("target_currency") != "PHP"
        or manifest.get("target_period") != "monthly"
        or manifest.get("target_salary_basis") != TARGET_BASIS[manifest["cohort"]]
        or not isinstance(manifest.get("reviewed_normalization_method"), str)
        or not manifest["reviewed_normalization_method"].strip()
        or manifest.get("holdout_reviewed") is not True
    ):
        raise ValueError("Reviewed training manifest/permission/target approval is incomplete")
    try:
        val_from = date.fromisoformat(manifest["validation_from"])
        test_from = date.fromisoformat(manifest["test_from"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Reviewed split cutoff dates are missing") from exc
    if val_from >= test_from:
        raise ValueError("Validation cutoff must precede test cutoff")
    data_path = _inside_curated(root, manifest["dataset_file"])
    if data_path.suffix != ".jsonl" or _sha(data_path.read_bytes()) != manifest.get("dataset_sha256"):
        raise ValueError("Reviewed dataset SHA mismatch or unexpected extension")

    register = load_source_register(root / "configs/data_sources.json")
    errors = validate_source_register(register)
    if errors:
        raise ValueError("Source registry invalid: " + "; ".join(errors[:3]))
    sources = {source["id"]: source for source in register["sources"]}
    for sid in manifest["source_ids"]:
        source = sources.get(sid)
        if (
            source is None or source["dataset_role"] != COHORTS[manifest["cohort"]]
            or not source["enabled_for_import"]
            or "train_on_labeled_salary" not in source["approved_use"]
            or not source["rights_status"].startswith("verified_")
        ):
            raise ValueError(f"Source {sid} not approved for supervised salary training")
    policy = json.loads((root / "configs/release_policy.yaml").read_text())
    if (
        policy.get("evidence", {}).get("supervised_training_enabled") is not True
        or policy["evidence"].get("allow_published_aggregate_as_person_level_label") is not False
    ):
        raise ValueError("Independent supervised-training policy approval missing")

    rows = []
    seen: set[str] = set()
    for lineno, line in enumerate(data_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or REQUIRED - row.keys():
            raise ValueError(f"Row {lineno}: missing observed salary and predictor fields")
        if FORBIDDEN_INPUTS & row.keys():
            raise ValueError(f"Row {lineno}: synthetic prediction must not be a salary label")
        if (
            row["source_id"] not in manifest["source_ids"]
            or row["cohort"] != manifest["cohort"]
            or row["currency"] != "PHP"
            or row["period"] != "monthly"
            or row["salary_basis"] != manifest["target_salary_basis"]
            or not row["normalization_evidence"]
            or not isinstance(row["market_country_verified"], bool)
            or not row["market_country_verified"]
        ):
            raise ValueError(f"Row {lineno}: incompatible target, origin or market")
        if any(
            not isinstance(row[field], str) or not row[field].strip()
            for field in (
                "observation_id", "job_title", "occupation_code", "employer_group",
                "duplicate_cluster_id", "market_country"
            )
        ):
            raise ValueError(f"Row {lineno}: missing verified identity or occupation")
        if not isinstance(row["region_code"], (str, type(None))):
            raise ValueError(f"Row {lineno}: invalid region")
        if row["observation_id"] in seen:
            raise ValueError("Repeated observation ID")
        seen.add(row["observation_id"])
        try:
            date.fromisoformat(row["observation_date"])
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Row {lineno}: invalid observation date") from exc
        row["salary_php_monthly"] = float(_positive(row["salary_php_monthly"]))
        for field in ("years_experience",):
            value = row[field]
            if value is not None:
                try:
                    value = float(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Row {lineno}: invalid experience") from exc
                if not 0 <= value <= 70:
                    raise ValueError(f"Row {lineno}: experience outside valid range")
                row[field] = value
        for field in ("education_level", "employment_context"):
            if row[field] is not None and not isinstance(row[field], str):
                raise ValueError(f"Row {lineno}: invalid {field}")
        rows.append(row)
    if not rows:
        raise ValueError("No permissioned salary observations available")
    return manifest, rows


def frozen_splits(rows: list[dict], *, validation_from: str, test_from: str) -> dict:
    v, t = date.fromisoformat(validation_from), date.fromisoformat(test_from)
    if v >= t:
        raise ValueError("Split cutoff ordering invalid")
    parent = list(range(len(rows)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        parent[find(a)] = find(b)

    owners: dict[str, int] = {}
    for i, row in enumerate(rows):
        for field in ("employer_group", "duplicate_cluster_id"):
            label = field + ":" + row[field]
            if label in owners:
                union(i, owners[label])
            else:
                owners[label] = i
    components: dict[int, list[int]] = {}
    for i in range(len(rows)):
        components.setdefault(find(i), []).append(i)
    buckets = {"train": [], "validation": [], "test": [], "quarantine": []}
    for indices in components.values():
        periods = set()
        for idx in indices:
            when = date.fromisoformat(rows[idx]["observation_date"])
            periods.add("train" if when < v else "validation" if when < t else "test")
        bucket = next(iter(periods)) if len(periods) == 1 else "quarantine"
        buckets[bucket].extend(indices)
    for key in buckets:
        buckets[key].sort()
    return buckets


def split_checks(rows: list[dict], splits: dict, policy: dict) -> dict:
    minimum = policy["supervised_draft_thresholds"]
    train, val, test = (splits[p] for p in ("train", "validation", "test"))
    if min(len(train), len(val), len(test)) == 0:
        raise ValueError("Frozen split lacks train, validation or test observations")
    if len({rows[i]["employer_group"] for i in train}) < minimum["min_train_distinct_groups"]:
        raise ValueError("Insufficient independent train groups")
    if len({rows[i]["employer_group"] for i in test}) < minimum["min_test_distinct_groups"]:
        raise ValueError("Insufficient independent test groups")
    if len({rows[i]["observation_date"][:7] for i in train}) < minimum["min_training_time_periods"]:
        raise ValueError("Insufficient historical training periods")
    if len({rows[i]["observation_date"][:7] for i in test}) < minimum["min_test_time_periods"]:
        raise ValueError("Insufficient holdout periods")
    occupations = Counter(rows[i]["occupation_code"] for i in train)
    if max(occupations.values(), default=0) < minimum["min_occupation_labeled_records"]:
        raise ValueError("Occupation has inadequate training support")
    ids = [rows[i]["observation_id"] for group in splits.values() for i in group]
    if len(ids) != len(set(ids)) or len(ids) != len(rows):
        raise ValueError("Split did not partition every unique observation exactly once")
    return {
        "counts": {name: len(values) for name, values in splits.items()},
        "groups": {
            name: len({rows[i]["employer_group"] for i in values})
            for name, values in splits.items()
        },
        "split_ids_sha256": {
            name: _sha(json.dumps([rows[i]["observation_id"] for i in values]).encode())
            for name, values in splits.items()
        },
        "quarantine_reason": "group_or_duplicate_cluster_spans_multiple_time_splits",
    }
