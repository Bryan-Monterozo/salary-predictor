from __future__ import annotations

import hashlib
import io
import json
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

from .reviewed_data import _inside_curated, frozen_splits, reviewed_rows, split_checks

CATEGORICAL = (
    "job_title", "occupation_code", "region_code", "market_country",
    "education_level", "employment_context",
)
NUMERIC = ("years_experience",)
PEERS = frozenset({"ID", "MY", "TH", "VN", "IN", "PK", "BD", "KH", "LK", "NP", "MM", "LA"})
APAC = frozenset({"AU", "NZ", "SG", "JP", "KR", "CN", "HK", "TW", "FJ"})


def _json(data: dict) -> bytes:
    return (json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _weight(country: str) -> float:
    country = country.strip().upper()
    if country == "PH":
        return 3.0
    if country in PEERS:
        return 2.0
    if country in APAC:
        return 1.0
    return 0.5


def _matrix(rows: list[dict], ids: list[int]):
    import pandas as pd

    frame = pd.DataFrame([{key: rows[i].get(key) for key in (*CATEGORICAL, *NUMERIC)}
                          for i in ids])
    return frame.replace({None: float("nan")})


def _model(family: str):
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestRegressor, VotingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    def one(f: str):
        transformer = ColumnTransformer([
            ("categories", Pipeline([
                ("missing", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), list(CATEGORICAL)),
            ("numeric", Pipeline([
                ("missing", SimpleImputer(strategy="constant", fill_value=0)),
                ("scale", StandardScaler()),
            ]), list(NUMERIC)),
        ])
        estimator = (
            Ridge(alpha=10.0)
            if f == "ridge" else
            RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=2,
                                  random_state=42, n_jobs=1)
        )
        return Pipeline([("transform", transformer), ("model", estimator)])

    if family == "ensemble":
        return VotingRegressor([("ridge", one("ridge")), ("forest", one("forest"))])
    return one(family)


def _occupation_median(rows: list[dict], train: list[int], ids: list[int]) -> list[float]:
    from statistics import median

    values: dict[tuple, list[float]] = defaultdict(list)
    for i in train:
        row = rows[i]
        value = row["salary_php_monthly"]
        values[(row["occupation_code"], row["region_code"])].append(value)
        values[(row["occupation_code"],)].append(value)
    global_value = median(rows[i]["salary_php_monthly"] for i in train)
    return [
        median(values[key])
        if (key := (rows[i]["occupation_code"], rows[i]["region_code"])) in values
        else median(values[(rows[i]["occupation_code"],)])
        if (rows[i]["occupation_code"],) in values
        else global_value
        for i in ids
    ]


def _summary(rows: list[dict], ids: list[int], predictions) -> dict:
    from .metrics import absolute_error_metrics, subgroup_error_metrics

    observed = [str(rows[i]["salary_php_monthly"]) for i in ids]
    pred = [str(float(value)) for value in predictions]
    result = absolute_error_metrics(observed, pred)
    result["by_occupation"] = subgroup_error_metrics(
        observed, pred, [rows[i]["occupation_code"] for i in ids], min_support=5
    )
    result["by_region"] = subgroup_error_metrics(
        observed, pred, [rows[i]["region_code"] for i in ids], min_support=5
    )
    ph_ids = [j for j, i in enumerate(ids) if rows[i]["market_country"].upper() == "PH"]
    result["ph_count"] = len(ph_ids)
    result["ph_mae_php"] = (
        absolute_error_metrics([observed[j] for j in ph_ids],
                               [pred[j] for j in ph_ids])["mae_php"]
        if len(ph_ids) >= 5 else None
    )
    return result


def run_supervised(root: Path, reviewed_manifest: str, *, as_of: date) -> dict:
    import joblib

    root = root.resolve(strict=True)
    reviewed, rows = reviewed_rows(root, reviewed_manifest)
    splits = frozen_splits(
        rows,
        validation_from=reviewed["validation_from"],
        test_from=reviewed["test_from"],
    )
    policy = json.loads((root / "configs/release_policy.yaml").read_text())
    split_info = split_checks(rows, splits, policy)

    train = splits["train"]
    validation = splits["validation"]
    test = splits["test"]
    x_train, x_val, x_test = (
        _matrix(rows, indices) for indices in (train, validation, test)
    )
    y_train = [rows[i]["salary_php_monthly"] for i in train]
    validations = {}
    baseline = _occupation_median(rows, train, validation)
    validations["occupation_region_median"] = _summary(rows, validation, baseline)
    from statistics import median

    naive = [median(y_train)] * len(validation)
    validations["global_median"] = _summary(rows, validation, naive)
    fitted = {}
    ph_support = validations["global_median"]["ph_count"] >= 5
    for family in ("ridge", "forest", "ensemble"):
        for weighting in (False, True) if family != "ensemble" else (False,):
            model = _model(family)
            label = family + ("_geo_weighted" if weighting else "_unweighted")
            options = {}
            if weighting:
                options["model__sample_weight"] = [
                    _weight(rows[i]["market_country"]) for i in train
                ]
            model.fit(x_train, y_train, **options)
            predictions = model.predict(x_val)
            validations[label] = _summary(rows, validation, predictions)
            fitted[label] = model

    def score(item: str) -> tuple[float, float, str]:
        report = validations[item]
        overall = float(report["mae_php"])
        ph = report["ph_mae_php"]
        primary = 0.7 * float(ph) + 0.3 * overall if ph_support and ph else overall
        return (primary, overall, item)

    winner = min(validations, key=score)
    occupation_test = _summary(
        rows, test, _occupation_median(rows, train, test)
    )
    global_test = _summary(rows, test, [median(y_train)] * len(test))
    if winner in fitted:
        test_prediction = fitted[winner].predict(x_test)
        model_bytes = io.BytesIO()
        joblib.dump(fitted[winner], model_bytes)
        serialized = model_bytes.getvalue()
    else:
        test_prediction = (
            _occupation_median(rows, train, test)
            if winner == "occupation_region_median"
            else [median(y_train)] * len(test)
        )
        serialized = b""
    winning_test = _summary(rows, test, test_prediction)
    source_path = (root / reviewed_manifest).resolve(strict=True)
    data_path = (root / reviewed["dataset_file"]).resolve(strict=True)
    evaluation = {
        "status": "TRAINED_REVIEWED_LOCAL_EXPERIMENT_NOT_RELEASED",
        "as_of": as_of.isoformat(),
        "cohort": reviewed["cohort"],
        "sources": reviewed["source_ids"],
        "salary_definition": reviewed["target_salary_basis"],
        "source_rights_evidence": reviewed["rights_evidence_reference"],
        "reviewed_manifest_path": reviewed_manifest,
        "reviewed_data_file": reviewed["dataset_file"],
        "source_manifest_sha256": _sha(source_path.read_bytes()),
        "dataset_sha256": _sha(data_path.read_bytes()),
        "source_registry_sha256": _sha(
            (root / "configs/data_sources.json").read_bytes()
        ),
        "training_policy_sha256": _sha(
            (root / "configs/release_policy.yaml").read_bytes()
        ),
        "splits": split_info,
        "validation_candidates": validations,
        "selection_rule": (
            "validation_0.7_ph_mae_plus_0.3_all_mae"
            if ph_support else "validation_overall_mae"
        ),
        "selected": winner,
        "test_metrics_selected": winning_test,
        "test_metrics_global_baseline": global_test,
        "test_metrics_occupation_baseline": occupation_test,
        "model_trained": winner in fitted,
        "runtime_release_approved": False,
        "no_causal_skill_salary_claims": True,
        "worker_and_advertised_cohorts_separate": True,
    }
    digest = _sha(_json(evaluation) + serialized)
    run_id = "ml-" + digest[:20]
    folder = root / "data/artifacts/ml-candidates" / run_id
    if folder.is_symlink():
        raise ValueError("Unsafe model candidate directory")
    folder.mkdir(parents=True, exist_ok=True)
    files = {"evaluation.json": _json(evaluation), "split.json": _json({
        name: [rows[i]["observation_id"] for i in indices]
        for name, indices in splits.items()
    })}
    if serialized:
        files["model.joblib"] = serialized
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "LOCAL_RESEARCH_CANDIDATE_NOT_APPROVED",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files_sha256": {name: _sha(body) for name, body in files.items()},
        "feature_schema_version": 1,
        "features": list((*CATEGORICAL, *NUMERIC)),
        "source_manifest_sha256": evaluation["source_manifest_sha256"],
        "source_registry_sha256": evaluation["source_registry_sha256"],
        "training_policy_sha256": evaluation["training_policy_sha256"],
        "dataset_sha256": evaluation["dataset_sha256"],
        "cohort": reviewed["cohort"],
        "selected": winner,
        "runtime_release_approved": False,
        "model_loader_allowed": False,
        "model_format": "trusted_operator_only_joblib" if serialized else None,
        "test_metrics_sha256": _sha(_json(winning_test)),
    }
    for filename, body in files.items():
        dest = folder / filename
        if dest.is_symlink():
            raise ValueError("Unsafe candidate output symlink")
        try:
            with dest.open("xb") as handle:
                handle.write(body)
        except FileExistsError:
            if dest.read_bytes() != body:
                raise ValueError("Immutable candidate bytes changed")
    mp = folder / "manifest.json"
    if mp.exists():
        existing = json.loads(mp.read_text())
        manifest["created_at_utc"] = existing["created_at_utc"]
    manifest_body = _json(manifest)
    try:
        with mp.open("xb") as handle:
            handle.write(manifest_body)
    except FileExistsError:
        if mp.read_bytes() != manifest_body:
            raise ValueError("Immutable candidate manifest changed")
    return {
        "run_id": run_id, "status": manifest["status"], "selected": winner,
        "cohort": reviewed["cohort"], "split_counts": split_info["counts"],
        "heldout_mae_php": winning_test["mae_php"],
        "heldout_ph_mae_php": winning_test["ph_mae_php"],
        "artifact_path": folder.relative_to(root).as_posix(),
        "runtime_release_approved": False,
    }


def verify_model_candidate(root: Path, run_id: str) -> dict:
    import re

    if not re.fullmatch(r"ml-[a-f0-9]{20}", run_id):
        raise ValueError("Invalid model candidate ID")
    root = root.resolve(strict=True)
    folder = root / "data/artifacts/ml-candidates" / run_id
    mp = folder / "manifest.json"
    if folder.is_symlink() or mp.is_symlink() or not mp.is_file():
        raise ValueError("Missing or unsafe model candidate manifest")
    manifest = json.loads(mp.read_text(encoding="utf-8"))
    if (
        manifest.get("run_id") != run_id
        or manifest.get("schema_version") != 1
        or manifest.get("status") != "LOCAL_RESEARCH_CANDIDATE_NOT_APPROVED"
        or manifest.get("runtime_release_approved") is not False
        or manifest.get("model_loader_allowed") is not False
        or manifest.get("feature_schema_version") != 1
        or set(manifest.get("files_sha256", {}))
        not in ({"evaluation.json", "split.json"},
                {"evaluation.json", "split.json", "model.joblib"})
    ):
        raise ValueError("Candidate schema or trust boundary invalid")
    for filename, expected in manifest["files_sha256"].items():
        path = folder / filename
        if path.is_symlink() or not path.is_file() or _sha(path.read_bytes()) != expected:
            raise ValueError("Model candidate file integrity failed")
    evaluation = json.loads((folder / "evaluation.json").read_text())
    if evaluation.get("runtime_release_approved") is not False:
        raise ValueError("Evaluation claims unapproved runtime permissions")
    for name, path in (
        ("source_manifest_sha256", None),
        ("source_registry_sha256", root / "configs/data_sources.json"),
        ("training_policy_sha256", root / "configs/release_policy.yaml"),
    ):
        if evaluation.get(name) != manifest.get(name):
            raise ValueError("Candidate lineage fingerprint mismatch")
        if path is not None and _sha(path.read_bytes()) != evaluation[name]:
            raise ValueError("Candidate source policy fingerprint changed")
    if (
        _sha(_inside_curated(root, evaluation["reviewed_manifest_path"]).read_bytes())
        != manifest["source_manifest_sha256"]
        or _sha(_inside_curated(root, evaluation["reviewed_data_file"]).read_bytes())
        != manifest["dataset_sha256"]
    ):
        raise ValueError("Reviewed salary data or permission manifest changed")
    if manifest["test_metrics_sha256"] != _sha(_json(evaluation["test_metrics_selected"])):
        raise ValueError("Heldout test evidence mismatch")
    return {
        "run_id": run_id,
        "status": manifest["status"],
        "model_bytes_validated_but_not_loaded": "model.joblib" in manifest["files_sha256"],
        "runtime_release_approved": False,
    }
