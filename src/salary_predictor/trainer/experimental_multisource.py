from __future__ import annotations

import hashlib
import io
import json
import math
from datetime import date
from pathlib import Path
from statistics import median

from salary_predictor.feasibility.consolidated import consolidate

SEED = 20261022
PERIOD_MONTHS = {"yearly": 1 / 12, "monthly": 1.0, "hourly": 40 * 52 / 12}
FEATURES = ("title", "skill_mentions", "seniority", "eligibility_bucket")
SOURCE = "jobicy_public_remote_jobs"


def _json(obj: dict) -> bytes:
    return (json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _folder(root: Path) -> Path:
    root = root.resolve(strict=True)
    folder = root / "data/artifacts/experimental"
    if folder.is_symlink() or (root / "data").is_symlink():
        raise ValueError("Unsafe experimental artifact directory")
    return folder


def _once(path: Path, data: bytes) -> None:
    if path.is_symlink():
        raise ValueError("Artifact symlinks are not allowed")
    try:
        with path.open("xb") as file:
            file.write(data)
    except FileExistsError:
        if path.read_bytes() != data:
            raise ValueError("Immutable experiment artifact differs")


def _offered_monthly_usd(item: dict) -> float | None:
    salary = item["salary"]
    if not salary or salary["currency"] != "USD":
        return None
    if salary["period"] not in PERIOD_MONTHS:
        return None
    try:
        low, high = float(salary["amount_min"]), float(salary["amount_max"])
    except (ValueError, TypeError):
        return None
    value = (low + high) / 2 * PERIOD_MONTHS[salary["period"]]
    return value if all(map(math.isfinite, (low, high, value))) and 0 < low <= high and 50 <= value <= 150_000 else None


def _training_records(items: list[dict]) -> list[dict]:
    records = []
    for item in items:
        target = _offered_monthly_usd(item)
        if target is None:
            continue
        records.append({
            "record_id": item["record_id"],
            "title": item["title"].strip(),
            "title_group": item["title_search_key"],
            "skill_mentions": " ".join(
                sorted({s["phrase"].casefold() for s in item["description_skill_mentions"]})
            ) or "unspecified",
            "seniority": item["seniority_label"] or "Any",
            "eligibility_bucket": item["relevance"]["bucket"],
            "weight": float(item["relevance"]["ranking_weight"]),
            "target_usd_monthly": target,
            "source_url": item["source_url"],
        })
    return sorted(records, key=lambda r: r["record_id"])


def _partition(items: list[dict]) -> dict[str, list[int]]:
    groups = {}
    for i, item in enumerate(items):
        key = item["title_group"]
        score = int(hashlib.sha256(f"{SEED}|{key}".encode()).hexdigest()[:12], 16) % 100
        groups[i] = "train" if score < 70 else "validation" if score < 85 else "test"
    result = {split: [i for i in range(len(items)) if groups[i] == split]
              for split in ("train", "validation", "test")}
    if min(len(result[x]) for x in result) < 20:
        raise ValueError("Insufficient independent title-group holdout")
    if len({items[i]["title_group"] for i in result["test"]}) < 10:
        raise ValueError("Too few independent test titles")
    return result


def _frame(records: list[dict], indices: list[int]):
    import pandas as pd

    return pd.DataFrame([
        {name: records[i][name] for name in FEATURES} for i in indices
    ], columns=list(FEATURES))


def _estimator(family: str):
    import numpy as np
    from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    pre = ColumnTransformer([
        ("title", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                                  min_df=2, max_features=1800), "title"),
        ("skills", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                                   max_features=350), "skill_mentions"),
        ("scope", OneHotEncoder(handle_unknown="ignore"),
         ["seniority", "eligibility_bucket"]),
    ])
    regressor = Ridge(alpha=10) if family == "ridge" else RandomForestRegressor(
        n_estimators=90, max_depth=11, min_samples_leaf=3, random_state=SEED, n_jobs=1
    )
    return TransformedTargetRegressor(
        regressor=Pipeline([("features", pre), ("estimator", regressor)]),
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False,
    )


def _score(items: list[dict], indices: list[int], pred) -> dict:
    import numpy as np

    y = np.asarray([items[i]["target_usd_monthly"] for i in indices])
    estimate = np.clip(np.asarray(pred, dtype=float), 0, None)
    error = np.abs(y - estimate)
    philippine_indices = [
        index for index, i in enumerate(indices)
        if items[i]["eligibility_bucket"] == "PH_ELIGIBLE"
    ]
    return {
        "count": len(indices),
        "independent_titles": len({items[i]["title_group"] for i in indices}),
        "mae_usd_monthly": round(float(np.mean(error)), 2),
        "median_absolute_error_usd_monthly": round(float(np.median(error)), 2),
        "p90_absolute_error_usd_monthly": round(float(np.quantile(error, .90)), 2),
        "philippines_eligible_count": len(philippine_indices),
        "philippines_eligible_mae_usd_monthly": (
            round(float(np.mean(error[philippine_indices])), 2)
            if len(philippine_indices) >= 5 else None
        ),
    }


def _fit(items: list[dict], split: dict, family: str, *, weighted: bool):
    import numpy as np

    train = split["train"]
    model = _estimator(family)
    opts = {"estimator__sample_weight": np.asarray(
        [items[i]["weight"] for i in train])} if weighted else {}
    model.fit(
        _frame(items, train),
        np.asarray([items[i]["target_usd_monthly"] for i in train]),
        **opts,
    )
    return model


def train_experiment(root: Path, *, as_of: date) -> dict:
    import joblib

    root = root.resolve(strict=True)
    summary, listings, _missing = consolidate(root, as_of=as_of)
    records = _training_records(listings)
    partitions = _partition(records)
    tr, val, test = (partitions[x] for x in ("train", "validation", "test"))
    baseline = median(records[i]["target_usd_monthly"] for i in tr)
    validations = {"train_median": _score(records, val, [baseline] * len(val))}
    fitted = {}
    for family in ("ridge", "forest"):
        for weighted in (False, True):
            name = family + ("_ph_relevance_weighted" if weighted else "_unweighted")
            model = _fit(records, partitions, family, weighted=weighted)
            fitted[name] = model
            validations[name] = _score(
                records, val, model.predict(_frame(records, val))
            )
    winner = min(fitted, key=lambda key: (
        validations[key]["mae_usd_monthly"], key
    ))
    winning = fitted[winner]
    heldout = _score(records, test, winning.predict(_frame(records, test)))
    baseline_test = _score(records, test, [baseline] * len(test))
    stream = io.BytesIO()
    joblib.dump(winning, stream)
    model_data = stream.getvalue()
    original_source = root / "configs/local_raw_file_manifest.json"
    policy_file = root / "configs/feasibility/consolidation_policy.json"
    report = {
        "schema_version": 1,
        "status": "LOCAL_EXPERIMENT_NOT_APPROVED_FOR_DEPLOYMENT",
        "as_of": as_of.isoformat(),
        "source_id": SOURCE,
        "source_snapshot_sha256": _sha(original_source.read_bytes()),
        "source_policy_sha256": _sha(policy_file.read_bytes()),
        "evidence_record_count": summary["unique_job_postings"],
        "disclosed_offer_count": summary["valid_disclosed_salary_ranges"],
        "usd_offer_training_rows": len(records),
        "original_salary_target": "ad_min_max_midpoint_normalized_to_USD_per_month",
        "assumptions": {"monthly_per_year": 12, "hours_per_week": 40, "weeks_per_year": 52},
        "holdout": "title_group_SHA256_bucket_not_chronological_all_ads_october_2026",
        "split_counts": {k: len(v) for k, v in partitions.items()},
        "split_title_groups": {
            k: len({records[i]["title_group"] for i in v}) for k, v in partitions.items()
        },
        "validation_candidates": validations,
        "selected": winner,
        "test_selected": heldout,
        "test_training_median": baseline_test,
        "psa_benchmarks_joined_to_ad_training": False,
        "step_hourly_earnings_joined_to_ad_training": False,
        "skill_description_mentions_are_not_worker_skills": True,
        "geographic_weights_are_remote_eligibility_not_pay_market": True,
        "salary_fx_conversion_applied_during_training": False,
        "production_release_approved": False,
        "rights_review_required_before_distribution": True,
    }
    report_bytes = _json(report)
    marker = _sha(report_bytes + model_data)[:20]
    run_id = f"exp-{marker}"
    directory = _folder(root) / run_id
    if directory.is_symlink():
        raise ValueError("Unsafe experiment run directory")
    directory.mkdir(parents=True, exist_ok=True)
    files = {"model.joblib": model_data, "evaluation.json": report_bytes}
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "LOCAL_ONLY",
        "files_sha256": {name: _sha(data) for name, data in files.items()},
        "source_snapshot_sha256": report["source_snapshot_sha256"],
        "source_policy_sha256": report["source_policy_sha256"],
        "feature_names": list(FEATURES),
        "estimator": winner,
        "training_rows": len(records),
        "production_release_approved": False,
        "loading_policy": "trusted_local_build_only_no_web_uploads",
    }
    for name, data in files.items():
        _once(directory / name, data)
    _once(directory / "manifest.json", _json(manifest))
    pointer = _folder(root) / "local_current.json"
    pointer_bytes = _json({
        "run_id": run_id,
        "manifest_sha256": _sha(_json(manifest)),
        "mode": "LOCAL_EXPERIMENT_ONLY",
    })
    if pointer.is_symlink():
        raise ValueError("Unsafe local current pointer")
    pointer.write_bytes(pointer_bytes)
    return {
        "run_id": run_id,
        "artifact": directory.relative_to(root).as_posix(),
        "training_rows": len(records),
        "selected": winner,
        "heldout": heldout,
        "baseline_heldout": baseline_test,
        "status": report["status"],
        "production_release_approved": False,
    }


def verify_experiment(root: Path, run_id: str | None = None) -> tuple[dict, Path]:
    import re

    root = root.resolve(strict=True)
    folder = _folder(root)
    pointer = folder / "local_current.json"
    if run_id is None:
        if pointer.is_symlink() or not pointer.is_file():
            raise ValueError("No local experiment trained")
        selected = json.loads(pointer.read_text())
        run_id = selected["run_id"]
    if not re.fullmatch(r"exp-[0-9a-f]{20}", run_id):
        raise ValueError("Invalid experimental run ID")
    directory = folder / run_id
    path = directory / "manifest.json"
    if directory.is_symlink() or path.is_symlink() or not path.is_file():
        raise ValueError("Experiment missing or unsafe")
    manifest = json.loads(path.read_text())
    if (
        manifest.get("run_id") != run_id or manifest.get("schema_version") != 1
        or manifest.get("status") != "LOCAL_ONLY"
        or manifest.get("production_release_approved") is not False
        or set(manifest.get("files_sha256", {})) != {"model.joblib", "evaluation.json"}
        or manifest.get("feature_names") != list(FEATURES)
    ):
        raise ValueError("Experimental model manifest is invalid")
    for name, digest in manifest["files_sha256"].items():
        target = directory / name
        if target.is_symlink() or not target.is_file() or _sha(target.read_bytes()) != digest:
            raise ValueError("Experimental model bytes do not match manifest")
    for name, raw in (
        ("source_snapshot_sha256", "configs/local_raw_file_manifest.json"),
        ("source_policy_sha256", "configs/feasibility/consolidation_policy.json"),
    ):
        if manifest[name] != _sha((root / raw).read_bytes()):
            raise ValueError("Source inventory or experiment policy changed")
    evaluation = json.loads((directory / "evaluation.json").read_text())
    if evaluation["source_snapshot_sha256"] != manifest["source_snapshot_sha256"]:
        raise ValueError("Evaluation source lineage mismatch")
    if pointer.is_file():
        if pointer.is_symlink():
            raise ValueError("Unsafe local current pointer")
        current = json.loads(pointer.read_text())
        if current.get("run_id") == run_id:
            if current.get("manifest_sha256") != _sha(path.read_bytes()):
                raise ValueError("Local pointer changed")
    return evaluation, directory
