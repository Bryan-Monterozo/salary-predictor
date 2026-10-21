from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import date

ALLOWED_RECORD_TYPES = frozenset({"individual_earnings", "advertised_salary"})


def _month(value: str) -> str:
    if not isinstance(value, str) or len(value) not in (7, 10):
        raise ValueError("Observation date needs YYYY-MM or YYYY-MM-DD")
    try:
        date.fromisoformat(value + "-01" if len(value) == 7 else value)
    except ValueError as exc:
        raise ValueError("Invalid observation date") from exc
    return value[:7]


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def propose_group_time_holdout(
    rows: list[dict], *, cutoff_month: str, policy: dict,
) -> dict:
    """Plan disjoint test groups, not fit a model or approve data."""
    cutoff = _month(cutoff_month)
    if len(cutoff_month) != 7:
        raise ValueError("Cutoff must be YYYY-MM")
    if not rows:
        return {
            "status": "BLOCKED_NO_ELIGIBLE_LABELS",
            "train_ids": [], "test_ids": [], "quarantined_ids": [],
            "reason": "No labeled observations",
        }
    if not policy["evidence"]["supervised_training_enabled"]:
        # Only synthetic fixture tests may use the proposal; nothing gets released.
        approval = "NOT_APPROVED_FOR_TRAINING"
    else:
        approval = "REQUIRE_SEPARATE_SOURCE_AND_MODEL_APPROVAL"

    allowed_predictors = set(policy["holdout"]["eligible_predictor_fields"])
    forbidden = set(policy["holdout"]["never_use_as_feature"])
    root: dict[str, str] = {}

    def find(x: str) -> str:
        if x not in root:
            root[x] = x
        if root[x] != x:
            root[x] = find(root[x])
        return root[x]

    def union(x: str, y: str) -> None:
        a, b = find(x), find(y)
        if a != b:
            root[max(a, b)] = min(a, b)

    seen_rows: set[tuple[str, str]] = set()
    labels: dict[str, list[dict]] = defaultdict(list)
    cohorts: set[str] = set()
    for item in rows:
        mandatory = (
            "source_id", "observation_id", "record_type", "observation_date",
            "group_id", "duplicate_cluster_id", "predictor_fields",
        )
        if any(field not in item for field in mandatory):
            raise ValueError("Labeled split row is missing required audit metadata")
        if any(not isinstance(item[name], str) or not item[name].strip()
               for name in mandatory[:-1]):
            raise ValueError("Missing source, record or deduplication identity")
        cohort = item["record_type"]
        if cohort not in ALLOWED_RECORD_TYPES:
            raise ValueError("Aggregate benchmark cannot enter a supervised split")
        cohorts.add(cohort)
        if len(cohorts) != 1:
            raise ValueError("Cannot merge employee earnings and advertised salaries")
        predictors = item["predictor_fields"]
        if not isinstance(predictors, list) or any(
            not isinstance(name, str) for name in predictors
        ):
            raise ValueError("Observed predictor field list is required")
        if set(predictors) & forbidden or not set(predictors).issubset(allowed_predictors):
            raise ValueError("Outcome-derived or unsupported predictors are forbidden")
        month = _month(item["observation_date"])
        identity = (item["source_id"], item["observation_id"])
        if identity in seen_rows:
            raise ValueError("Repeated labeled observation ID")
        seen_rows.add(identity)
        node = "row:" + "\x1f".join(identity)
        union(node, "group:" + item["group_id"])
        union(node, "duplicate:" + item["duplicate_cluster_id"])
        labels[node].append({"identity": identity, "month": month, "group": item["group_id"],
                             "duplicate": item["duplicate_cluster_id"]})

    components: dict[str, list[dict]] = defaultdict(list)
    for node, members in labels.items():
        components[find(node)].extend(members)

    train, test, quarantined = [], [], []
    train_groups, test_groups = set(), set()
    for component_id, members in sorted(components.items()):
        is_test_component = any(m["month"] >= cutoff for m in members)
        for member in members:
            uid = ":".join(member["identity"])
            if is_test_component:
                if member["month"] >= cutoff:
                    test.append(uid)
                    test_groups.add(component_id)
                else:
                    quarantined.append(uid)
            elif member["month"] < cutoff:
                train.append(uid)
                train_groups.add(component_id)
            else:
                quarantined.append(uid)
    if train_groups & test_groups:
        raise ValueError("Leakage detected across train and test groups")

    train.sort()
    test.sort()
    quarantined.sort()
    draft = policy["supervised_draft_thresholds"]
    unique_train_periods = {
        m["month"] for c in components.values() if all(x["month"] < cutoff for x in c)
        for m in c
    }
    unique_test_periods = {
        m["month"] for c in components.values() if any(x["month"] >= cutoff for x in c)
        for m in c if m["month"] >= cutoff
    }
    sufficient = (
        len(train_groups) >= draft["min_train_distinct_groups"]
        and len(test_groups) >= draft["min_test_distinct_groups"]
        and len(unique_train_periods) >= draft["min_training_time_periods"]
        and len(unique_test_periods) >= draft["min_test_time_periods"]
    )
    return {
        "status": "PROPOSAL_ONLY" if sufficient else "BLOCKED_INSUFFICIENT_SUPPORT",
        "approval": approval,
        "cohort": next(iter(cohorts)),
        "cutoff_month": cutoff,
        "train_ids": train,
        "test_ids": test,
        "quarantined_ids": quarantined,
        "distinct_train_groups": len(train_groups),
        "distinct_test_groups": len(test_groups),
        "train_periods": sorted(unique_train_periods),
        "test_periods": sorted(unique_test_periods),
        "partition_fingerprint": _hash({
            "cutoff": cutoff, "train": train, "test": test,
            "quarantined": quarantined, "cohort": next(iter(cohorts)),
        }),
        "frozen": False,
        "training_and_evaluation_performed": False,
        "notes": "Freeze an approved manifest before fitting; fit preprocessors on training only",
    }
