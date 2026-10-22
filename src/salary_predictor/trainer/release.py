from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import date, datetime, timezone
from pathlib import Path

from salary_predictor.audit.quality import load_staged_benchmarks, months_older
from salary_predictor.source_registry import load_source_register, validate_source_register

from .baseline import (
    BENCHMARK_ID,
    CORRECTED_PSA_STAGED,
    CORRECTED_SHA256,
    BenchmarkOnlyBaseline,
)

ARTIFACT_SCHEMA = 1
RELATIVE = Path("data/artifacts")
ID_PATTERN = re.compile(r"^bench-[0-9a-f]{20}$")
TAXONOMY_FILES = (
    "configs/taxonomy/psoc_reviewed_v1.json",
    "configs/taxonomy/psgc_regions_v1.json",
    "configs/taxonomy/esco_psoc_crosswalk_v1.json",
)
MODEL_STATUS = "NO_ELIGIBLE_SUPERVISED_MODEL"


def _json_bytes(value: dict) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def _sha(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _safe_id(value: str) -> str:
    if not isinstance(value, str) or not ID_PATTERN.fullmatch(value):
        raise ValueError("Invalid immutable release ID")
    return value


def _artifact_root(root: Path) -> Path:
    root = root.resolve(strict=True)
    data = root / "data"
    if data.is_symlink():
        raise ValueError("Artifact data root cannot be a symlink")
    output = root / RELATIVE
    if output.is_symlink():
        raise ValueError("Artifact root cannot be a symlink")
    return output


def _write_once(path: Path, payload: bytes) -> None:
    if path.is_symlink():
        raise ValueError("Cannot write an artifact through a symlink")
    try:
        with path.open("xb") as stream:
            stream.write(payload)
    except FileExistsError:
        if path.read_bytes() != payload:
            raise ValueError("An immutable artifact differs from its recorded content")


def build_benchmark_candidate(root: Path, *, as_of: date) -> dict:
    root = root.resolve(strict=True)
    baseline = BenchmarkOnlyBaseline(root, as_of=as_of)
    audit = baseline.reproducible_report()
    if audit["runtime_release"] != "BLOCKED_NOT_APPROVED":
        raise ValueError("Benchmark release policy changed: re-review before packaging")
    rows, provenance = load_staged_benchmarks(root, CORRECTED_PSA_STAGED)
    if provenance["fingerprint"] != CORRECTED_SHA256 or len(rows) != 560:
        raise ValueError("Staged source changed; refusing benchmark bundle")
    register = load_source_register(root / "configs/data_sources.json")
    if validate_source_register(register):
        raise ValueError("Source permissions registry failed")
    source = next(
        (entry for entry in register["sources"] if entry["id"] == BENCHMARK_ID), None
    )
    if (
        source is None
        or source["dataset_role"] != "aggregate_benchmark"
        or not source["public_benchmark_eligible"]
        or "public_benchmark_with_attribution" not in source["approved_use"]
    ):
        raise ValueError("Source lacks approved research benchmark provenance")
    records = [row["record"] for row in rows]
    bundle = {
        "schema_version": ARTIFACT_SCHEMA,
        "kind": "published_benchmark_only",
        "source_id": BENCHMARK_ID,
        "source_version": records[0]["source_version"],
        "source_fingerprint": CORRECTED_SHA256,
        "reference_date": "2022-08",
        "source_table_count": len(records),
        "records": records,
        "runtime_release_approved": False,
        "personal_model": None,
    }
    body = _json_bytes(bundle)
    taxonomy = {
        path: _sha((root / path).read_bytes()) for path in TAXONOMY_FILES
    }
    dependencies = {
        "source_registry": _sha((root / "configs/data_sources.json").read_bytes()),
        "release_policy": _sha((root / "configs/release_policy.yaml").read_bytes()),
        "result_policy": _sha((root / "configs/result_decision_v1.json").read_bytes()),
        "raw_inventory": _sha(
            (root / "configs/local_raw_file_manifest.json").read_bytes()
        ),
    }
    fingerprint = _sha(_json_bytes({
        "benchmark_sha256": _sha(body),
        "taxonomy": taxonomy,
        "dependencies": dependencies,
        "as_of": as_of.isoformat(),
    }))
    run_id = "bench-" + fingerprint[:20]
    folder = _artifact_root(root) / "candidates" / run_id
    if folder.is_symlink():
        raise ValueError("Candidate directory cannot be a symlink")
    folder.mkdir(parents=True, exist_ok=True)
    _write_once(folder / "benchmark.json", body)
    manifest_path = folder / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {
            "artifact_schema_version": ARTIFACT_SCHEMA,
            "run_id": run_id,
            "kind": "published_benchmark_only",
            "status": "CANDIDATE_NOT_APPROVED",
            "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "as_of": as_of.isoformat(),
            "benchmark_file": "benchmark.json",
            "benchmark_sha256": _sha(body),
            "source_id": BENCHMARK_ID,
            "source_fingerprint": CORRECTED_SHA256,
            "permissions": {
                "rights_status": source["rights_status"],
                "approved_use": source["approved_use"],
                "operator_runtime_release": False,
            },
            "taxonomy_sha256": taxonomy,
            "dependencies_sha256": dependencies,
            "feature_schema_version": 1,
            "contract_version": 1,
            "preprocessing_version": "not_applicable_published_cells",
            "model_file": None,
            "model_format": None,
            "training_status": MODEL_STATUS,
            "training_records": 0,
            "heldout_mae_php": None,
            "prediction_interval": None,
            "model_trained_at_utc": None,
            "published_cell_count": len(records),
        }
        _write_once(manifest_path, _json_bytes(manifest))
    info = verify_candidate(root, run_id)
    return {
        "run_id": run_id,
        "status": info["status"],
        "bundle_path": str((folder / "benchmark.json").relative_to(root)),
        "manifest_path": str(manifest_path.relative_to(root)),
        "manifest_sha256": _sha(manifest_path.read_bytes()),
        "published_cells": info["published_cell_count"],
        "model_trained": False,
        "runtime_approved": False,
    }


def verify_candidate(root: Path, run_id: str) -> dict:
    root = root.resolve(strict=True)
    _safe_id(run_id)
    folder = _artifact_root(root) / "candidates" / run_id
    manifest_path = folder / "manifest.json"
    bundle_path = folder / "benchmark.json"
    if (
        folder.is_symlink() or manifest_path.is_symlink() or bundle_path.is_symlink()
        or not manifest_path.is_file() or not bundle_path.is_file()
    ):
        raise ValueError("Missing or unsafe candidate release files")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload = bundle_path.read_bytes()
    bundle = json.loads(payload)
    if (
        manifest.get("run_id") != run_id
        or manifest.get("artifact_schema_version") != ARTIFACT_SCHEMA
        or manifest.get("contract_version") != 1
        or manifest.get("feature_schema_version") != 1
        or manifest.get("status") != "CANDIDATE_NOT_APPROVED"
        or manifest.get("model_file") is not None
        or manifest.get("kind") != "published_benchmark_only"
        or manifest.get("benchmark_file") != "benchmark.json"
        or manifest.get("benchmark_sha256") != _sha(payload)
        or manifest.get("source_fingerprint") != CORRECTED_SHA256
        or manifest.get("permissions", {}).get("operator_runtime_release") is not False
        or bundle.get("source_fingerprint") != CORRECTED_SHA256
        or bundle.get("source_id") != BENCHMARK_ID
        or bundle.get("runtime_release_approved") is not False
        or bundle.get("personal_model") is not None
        or bundle.get("reference_date") != "2022-08"
        or bundle.get("source_table_count") != 560
        or bundle.get("records") is None
        or len(bundle["records"]) != manifest.get("published_cell_count")
    ):
        raise ValueError("Artifact schema, release flags or hashes are invalid")
    if set(manifest["taxonomy_sha256"]) != set(TAXONOMY_FILES):
        raise ValueError("Incomplete taxonomy compatibility handshake")
    for path, digest in manifest["taxonomy_sha256"].items():
        if path not in TAXONOMY_FILES or _sha((root / path).read_bytes()) != digest:
            raise ValueError("Taxonomy mismatch; refuse artifact")
    if set(manifest["dependencies_sha256"]) != {
        "source_registry", "release_policy", "result_policy", "raw_inventory"
    }:
        raise ValueError("Incomplete manifest permissions fingerprint")
    for key, path in (
        ("source_registry", "configs/data_sources.json"),
        ("release_policy", "configs/release_policy.yaml"),
        ("result_policy", "configs/result_decision_v1.json"),
        ("raw_inventory", "configs/local_raw_file_manifest.json"),
    ):
        if _sha((root / path).read_bytes()) != manifest["dependencies_sha256"][key]:
            raise ValueError("Artifact permission/configuration fingerprint mismatch")
    baseline = BenchmarkOnlyBaseline(
        root, as_of=date.fromisoformat(manifest["as_of"])
    )
    if baseline.reproducible_report()["published_cells"] != manifest["published_cell_count"]:
        raise ValueError("Underlying audited source changed")
    original_rows, _ = load_staged_benchmarks(root, CORRECTED_PSA_STAGED)
    if bundle["records"] != [row["record"] for row in original_rows]:
        raise ValueError("Bundle does not match the original source records")
    return manifest


def training_gate(root: Path, *, as_of: date) -> dict:
    from .experiment import readiness_report

    report = readiness_report(root, as_of=as_of)
    return {
        "status": "BLOCKED_NO_APPROVED_COMPARABLE_LABELED_HOLDOUT",
        "eligible_training_rows": report["eligible_supervised_training_rows"],
        "STEP_hourly_candidates": report["cohort_inspection"]["STEP"][
            "observed_hourly_wage_worker_candidates"
        ],
        "Jobicy_advertised_salary_ranges": report["cohort_inspection"]["Jobicy"][
            "valid_advertised_salary_ranges"
        ],
        "reason": (
            "STEP needs rights/survey normalization; Jobicy has sparse local PHP pay and "
            "unreviewed training rights; PSA OWS provides aggregates, not worker labels. "
            "No fitted model or valid held-out MAE can be produced."
        ),
    }


def promotion_blockers(root: Path, run_id: str, *, today: date) -> list[str]:
    manifest = verify_candidate(root, run_id)
    reasons = []
    policy = json.loads((root / "configs/release_policy.yaml").read_text())
    decision = json.loads((root / "configs/result_decision_v1.json").read_text())
    if policy.get("status") != "APPROVED_FOR_RUNTIME":
        reasons.append("Release policy has not been approved for runtime")
    if not decision.get("runtime_release_enabled") or not decision.get(
        "benchmark_release_enabled"
    ):
        reasons.append("Result policy does not permit active benchmark release")
    if months_older("2022-08", today) > decision["benchmark"]["max_reference_age_months"]:
        reasons.append("2022 OWS cells exceed the approved benchmark age limit")
    if manifest["model_file"] is not None:
        reasons.append("Unexpected executable model content")
    try:
        _approval(root, run_id)
    except (OSError, ValueError, KeyError):
        reasons.append("No valid operator-reviewed release approval is registered")
    return reasons



def _approval(root: Path, run_id: str) -> dict:
    artifact_dir = _artifact_root(root)
    path = artifact_dir / "approvals" / (_safe_id(run_id) + ".json")
    if path.is_symlink() or not path.is_file():
        raise ValueError("Missing local operator release approval")
    record = json.loads(path.read_text(encoding="utf-8"))
    candidate = artifact_dir / "candidates" / run_id / "manifest.json"
    if (
        record.get("schema_version") != 1
        or record.get("release_id") != run_id
        or record.get("approved_for") != "published_benchmark_only"
        or record.get("operator_reviewed") is not True
        or record.get("source_rights_reviewed") is not True
        or record.get("tests_passed_reviewed") is not True
        or record.get("release_policy_reviewed") is not True
        or record.get("manifest_sha256") != _sha(candidate.read_bytes())
        or not isinstance(record.get("operator_id"), str)
        or not re.fullmatch(r"[a-zA-Z0-9_.-]{3,64}", record["operator_id"])
        or not isinstance(record.get("approved_at_utc"), str)
    ):
        raise ValueError("Operator release approval invalid or bound to another artifact")
    return record


def _write_pointer_atomic(path: Path, value: dict) -> None:
    if path.is_symlink():
        raise ValueError("Refuse pointer symlink")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _json_bytes(value)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        with tmp.open("xb") as file:
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _read_active(root: Path, *, today: date | None = None) -> dict | None:
    directory = _artifact_root(root)
    path = directory / "active.json"
    if path.is_symlink():
        raise ValueError("Active pointer cannot be a symlink")
    if not path.exists():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    id_ = _safe_id(record["release_id"])
    manifest = verify_candidate(root, id_)
    if record.get("manifest_sha256") != _sha(
        (directory / "candidates" / id_ / "manifest.json").read_bytes()
    ):
        raise ValueError("Active pointer manifest fingerprint mismatch")
    _approval(root, id_)
    blockers = promotion_blockers(root, id_, today=today or date.today())
    if blockers:
        raise ValueError("Previously active release is no longer eligible: " + "; ".join(blockers))
    return {"release_id": id_, "manifest": manifest, "activated_at_utc": record.get("activated_at_utc")}


def active_release(root: Path) -> dict:
    try:
        entry = _read_active(root)
    except (ValueError, OSError, KeyError) as exc:
        return {"status": "BLOCKED_INVALID_ACTIVE_RELEASE", "reason": str(exc)}
    if entry is None:
        return {"status": "NO_APPROVED_ACTIVE_RELEASE"}
    return {
        "status": "ACTIVE_BENCHMARK_ONLY",
        "release_id": entry["release_id"],
        "model_trained": False,
        "runtime_prediction_enabled": False,
        "note": "Part 10/Flask does not load or serve this release automatically",
    }


def _approved_activation(root: Path, run_id: str, *, today: date) -> dict:
    errors = promotion_blockers(root, run_id, today=today)
    if errors:
        raise ValueError("Promotion blocked: " + "; ".join(errors))
    folder = _artifact_root(root)
    previous = _read_active(root, today=today)
    history = folder / "promotion_history.jsonl"
    if history.is_symlink():
        raise ValueError("Refuse history symlink")
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    marker = {
        "release_id": run_id,
        "manifest_sha256": _sha(
            (folder / "candidates" / run_id / "manifest.json").read_bytes()
        ),
        "scope": "published_benchmark_only",
        "activated_at_utc": stamp,
    }
    with history.open("a", encoding="utf-8") as log:
        log.write(json.dumps({
            "action": "activate",
            "from_id": previous["release_id"] if previous else None,
            "to_id": run_id,
            "at": stamp,
        }, sort_keys=True) + "\n")
        log.flush()
        os.fsync(log.fileno())
    _write_pointer_atomic(folder / "active.json", marker)
    return {
        "status": "ACTIVE_BENCHMARK_ONLY",
        "release_id": run_id,
        "previous_release_id": previous["release_id"] if previous else None,
        "model_trained": False,
        "runtime_prediction_enabled": False,
    }


def promote(root: Path, run_id: str, *, confirm: bool) -> dict:
    if not confirm:
        raise ValueError("Explicit operator --confirm required to promote")
    return _approved_activation(root, _safe_id(run_id), today=date.today())


def rollback(root: Path, run_id: str, *, confirm: bool) -> dict:
    if not confirm:
        raise ValueError("Explicit operator --confirm required for rollback")
    _safe_id(run_id)
    folder = _artifact_root(root)
    history = folder / "promotion_history.jsonl"
    if history.is_symlink() or not history.is_file():
        raise ValueError("No approved release history to roll back")
    seen = set()
    for line in history.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        if item.get("action") == "activate":
            seen.add(item.get("to_id"))
    current = _read_active(root)
    if run_id not in seen or current is None or current["release_id"] == run_id:
        raise ValueError("Target is not a prior, different approved active release")
    return _approved_activation(root, run_id, today=date.today())
