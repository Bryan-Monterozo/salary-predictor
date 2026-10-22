from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from statistics import median

DATASET = "data/raw/step/step_ph_weights_merged_public.csv"
SOURCE_ID = "world_bank_step_ph_2015"
MISSING = {"", "-99", "-88", "-77", "-66", "-44"}
FEATURES = ("occupation", "region_code", "years_educ")
SEED = 20261021


def _bytes(obj: dict) -> bytes:
    return (json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode()


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _positive(text: str) -> float | None:
    if text.strip() in MISSING:
        return None
    try:
        value = float(text)
    except (ValueError, TypeError):
        return None
    return value if 0 < value < float("inf") else None


def load_research_sample(root: Path) -> tuple[list[dict], dict]:
    root = root.resolve(strict=True)
    manifest = json.loads(
        (root / "configs/local_raw_file_manifest.json").read_text(encoding="utf-8")
    )
    original = next(
        (file for file in manifest["files"] if file["path"] == DATASET), None
    )
    if original is None:
        raise ValueError("STEP survey is not in the original source inventory")
    path = root / DATASET
    with path.open("rb") as f:
        actual = hashlib.file_digest(f, "sha256").hexdigest()
    if actual != original["sha256"]:
        raise ValueError("Original STEP microdata fingerprint mismatch")
    source_reg = json.loads((root / "configs/data_sources.json").read_text())
    source = next((s for s in source_reg["sources"] if s["id"] == SOURCE_ID), None)
    if source is None or source["dataset_role"] != "individual_earnings":
        raise ValueError("STEP source role not valid")
    # This is a research-only route. Do not change supervised release permissions.
    if source["enabled_for_import"] or "train_on_labeled_salary" in source["approved_use"]:
        raise ValueError("STEP production permission changed; research route needs review")
    dataset: list[dict] = []
    counts: Counter = Counter()
    with path.open(encoding="cp1252", errors="strict", newline="") as f:
        reader = csv.DictReader(f)
        if len(reader.fieldnames or []) != 1276:
            raise ValueError("Unexpected STEP survey schema")
        for row in reader:
            counts["respondents"] += 1
            if row["wage_worker"] != "1":
                continue
            wage = _positive(row["earnings_h"])
            weight = _positive(row["w3_finspwt"])
            if wage is None or weight is None:
                counts["no_usable_hourly_or_weight"] += 1
                continue
            if any(row[name].strip() in MISSING for name in (*FEATURES, "psu")):
                counts["missing_predictor_or_cluster"] += 1
                continue
            try:
                years = float(row["years_educ"])
            except ValueError:
                counts["invalid_education"] += 1
                continue
            if not 0 <= years <= 30:
                counts["invalid_education"] += 1
                continue
            dataset.append({
                "occupation": row["occupation"].strip(),
                "region_code": row["region_code"].strip(),
                "years_educ": years,
                "hourly_php": wage,
                "person_weight": weight,
                "psu": row["psu"].strip(),
            })
    if counts["respondents"] != original["rows"]:
        raise ValueError("STEP row count changed")
    if len(dataset) < 150 or len({r["psu"] for r in dataset}) < 30:
        raise ValueError("Insufficient survey-linked clusters for research")
    info = {
        "source_id": SOURCE_ID,
        "raw_sha256": actual,
        "response_count": counts["respondents"],
        "eligible_hourly_research_rows": len(dataset),
        "distinct_psus": len({r["psu"] for r in dataset}),
        "excluded_counts": dict(sorted(counts.items())),
        "measurement": "derived_hourly_php_earnings_NOT_gross_monthly_salary",
        "population": "surveyed_urban_philippines_2015_2016_wage_workers",
    }
    return dataset, info


def split_clusters(rows: list[dict]) -> dict[str, list[int]]:
    import numpy as np

    psus = sorted({r["psu"] for r in rows})
    rng = np.random.default_rng(SEED)
    rng.shuffle(psus)
    n = len(psus)
    boundaries = (round(n * .65), round(n * .80))
    membership = {psu: ("train" if k < boundaries[0] else
                        "validation" if k < boundaries[1] else "test")
                  for k, psu in enumerate(psus)}
    groups = {part: [] for part in ("train", "validation", "test")}
    for i, row in enumerate(rows):
        groups[membership[row["psu"]]].append(i)
    if min(len(x) for x in groups.values()) < 40:
        raise ValueError("STEP cluster split has insufficient heldout support")
    return groups


def _frame(rows: list[dict], indices: list[int]):
    import pandas as pd

    return pd.DataFrame([
        {key: rows[i][key] for key in FEATURES} for i in indices
    ])


def _estimator(kind: str):
    import numpy as np
    from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    pre = ColumnTransformer([
        ("categories", OneHotEncoder(handle_unknown="ignore"),
         ["occupation", "region_code"]),
        ("education", Pipeline([("imputer", SimpleImputer(strategy="median")),
                                ("scale", StandardScaler())]), ["years_educ"]),
    ])
    if kind == "ridge":
        return TransformedTargetRegressor(
            regressor=Pipeline([("pre", pre), ("ridge", Ridge(alpha=12))]),
            func=np.log1p, inverse_func=np.expm1,
            check_inverse=False,
        )
    if kind == "forest":
        return Pipeline([
            ("pre", pre),
            ("forest", RandomForestRegressor(
                n_estimators=100, min_samples_leaf=5,
                max_depth=9, random_state=SEED, n_jobs=1,
            )),
        ])
    raise ValueError("Unsupported research model")


def _median_predict(rows: list[dict], train: list[int], target: list[int]) -> list[float]:
    values = defaultdict(list)
    for i in train:
        values[rows[i]["occupation"]].append(rows[i]["hourly_php"])
    fallback = median(rows[i]["hourly_php"] for i in train)
    return [
        median(values[rows[i]["occupation"]]) if rows[i]["occupation"] in values
        else fallback for i in target
    ]


def _score(rows: list[dict], subset: list[int], pred: list[float]) -> dict:
    import numpy as np

    y = np.asarray([rows[i]["hourly_php"] for i in subset], dtype=float)
    weights = np.asarray([rows[i]["person_weight"] for i in subset], dtype=float)
    predicted = np.maximum(0, np.asarray(pred, dtype=float))
    error = np.abs(y - predicted)
    by_occupation = Counter(rows[i]["occupation"] for i in subset)
    return {
        "count": len(subset),
        "psu_groups": len({rows[i]["psu"] for i in subset}),
        "mae_php_per_hour": round(float(np.mean(error)), 4),
        "person_weighted_mae_php_per_hour": round(float(np.average(error, weights=weights)), 4),
        "median_absolute_error_php_per_hour": round(float(np.median(error)), 4),
        "error_q90_php_per_hour": round(float(np.quantile(error, .9)), 4),
        "occupation_support": dict(sorted(by_occupation.items())),
    }


def train_step_research(root: Path, *, as_of: date, write: bool = True) -> dict:
    import numpy as np

    root = root.resolve(strict=True)
    rows, source = load_research_sample(root)
    parts = split_clusters(rows)
    train, val, test = (parts[k] for k in ("train", "validation", "test"))
    x_train = _frame(rows, train)
    x_val = _frame(rows, val)
    y_train = np.asarray([rows[i]["hourly_php"] for i in train], dtype=float)
    w_train = np.asarray([rows[i]["person_weight"] for i in train], dtype=float)
    w_train = w_train / np.mean(w_train)
    validation = {
        "occupation_median": _score(
            rows, val, _median_predict(rows, train, val)
        )
    }
    fitted = {}
    for name in ("ridge", "forest"):
        model = _estimator(name)
        if name == "ridge":
            model.fit(x_train, y_train, ridge__sample_weight=w_train)
        else:
            model.fit(x_train, y_train, forest__sample_weight=w_train)
        fitted[name] = model
        validation[name] = _score(rows, val, list(model.predict(x_val)))
    selected = min(validation, key=lambda x: (
        validation[x]["person_weighted_mae_php_per_hour"],
        validation[x]["mae_php_per_hour"], x,
    ))
    predictions = (
        _median_predict(rows, train, test)
        if selected == "occupation_median" else
        list(fitted[selected].predict(_frame(rows, test)))
    )
    report = {
        "schema_version": 1,
        "status": "LOCAL_STEP_STATISTICAL_RESEARCH_ONLY",
        "as_of": as_of.isoformat(),
        "source": source,
        "sampling_group": "PSU",
        "split_seed": SEED,
        "split_counts": {name: len(indices) for name, indices in parts.items()},
        "split_psu_counts": {
            name: len({rows[i]["psu"] for i in indices}) for name, indices in parts.items()
        },
        "inputs": list(FEATURES),
        "omitted_inputs": [
            "gender", "survey_ids", "respondent_locations", "work_hours_target_denominator",
            "job_skills_not_validated", "experience_not_observed"
        ],
        "models": ["occupation_median", "ridge_log_hourly", "random_forest_hourly"],
        "validation_metrics": validation,
        "winner": selected,
        "test_metrics_selected": _score(rows, test, predictions),
        "test_metrics_occupation_baseline": _score(
            rows, test, _median_predict(rows, train, test)
        ),
        "training_was_actual": True,
        "heldout_measurement_is_hourly_not_monthly": True,
        "person_sampling_weights_used_for_training_and_evaluation": True,
        "cluster_disjointness_verified": all(
            not {rows[i]["psu"] for i in parts[a]} &
            {rows[i]["psu"] for i in parts[b]}
            for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))
        ),
        "not_a_nationally_representative_variance_estimate": True,
        "no_future_forecast": True,
        "research_data_not_approved_for_model_deployment": True,
        "personal_monthly_php_model_available": False,
        "user_salary_inference_allowed": False,
        "source_rights_basis": "World Bank Microdata Library Public Use File statistical research; no redistribution of microdata",
    }
    report["weighted_test_gain_over_baseline"] = (
        report["test_metrics_selected"]["person_weighted_mae_php_per_hour"]
        < report["test_metrics_occupation_baseline"]["person_weighted_mae_php_per_hour"]
    )
    if not write:
        return report
    data = _bytes(report)
    fingerprint = _digest(data)
    folder = root / "data/artifacts/research-step"
    if folder.is_symlink():
        raise ValueError("Unsafe research artifact path")
    folder.mkdir(parents=True, exist_ok=True)
    out = folder / f"step-hourly-research-{fingerprint[:16]}.json"
    try:
        with out.open("xb") as f:
            f.write(data)
    except FileExistsError:
        if out.read_bytes() != data:
            raise ValueError("Research-only immutable evaluation mismatch")
    report["report_path"] = out.relative_to(root).as_posix()
    report["report_sha256"] = fingerprint
    # No serialized model. Survey terms limit outputs to aggregate statistics.
    return report
