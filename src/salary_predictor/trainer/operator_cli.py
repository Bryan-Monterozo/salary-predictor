from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from salary_predictor.source_inventory import verify_raw_files
from salary_predictor.source_registry import load_source_register, validate_source_register

from .release import (
    active_release,
    build_benchmark_candidate,
    promote,
    promotion_blockers,
    rollback,
    training_gate,
    verify_candidate,
)


def _sources(root: Path) -> dict:
    root = root.resolve(strict=True)
    register = load_source_register(root / "configs/data_sources.json")
    issues = validate_source_register(register)
    inventory = json.loads(
        (root / "configs/local_raw_file_manifest.json").read_text(encoding="utf-8")
    )
    issues.extend(verify_raw_files(root, inventory))
    if issues:
        raise ValueError("Source validation failed: " + "; ".join(issues[:5]))
    return {
        "status": "SOURCE_INVENTORY_VERIFIED",
        "raw_files_verified": len(inventory["files"]),
        "source_roles": [{
            "id": item["id"],
            "role": item["dataset_role"],
            "enabled_for_import": item["enabled_for_import"],
            "approved_use": item["approved_use"],
        } for item in register["sources"]],
        "permission_to_train_salary_model": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m salary_predictor.trainer",
        description="Offline salary evidence and trusted candidate artifact operations",
    )
    options = parser.add_subparsers(dest="command", required=True)
    for name in (
        "sources", "ingest", "audit", "train", "evaluate", "prepare", "verify", "promote",
        "rollback", "active",
    ):
        entry = options.add_parser(name)
        entry.add_argument("--root", type=Path, default=Path.cwd())
        if name in ("train", "evaluate", "prepare"):
            entry.add_argument("--as-of", type=date.fromisoformat, default=date.today())
        if name in ("verify", "promote", "rollback"):
            entry.add_argument("--run-id", required=True)
        if name in ("train", "evaluate"):
            entry.add_argument("--reviewed-manifest", default=None)
        if name == "train":
            entry.add_argument("--step-research", action="store_true")
        if name == "evaluate":
            entry.add_argument("--run-id", default=None)
        if name in ("promote", "rollback"):
            entry.add_argument("--confirm", action="store_true")
    args = parser.parse_args(argv)
    try:
        root = args.root
        if args.command in ("ingest", "audit"):
            raise ValueError("Run ingest/audit through the top-level trainer dispatcher")
        if args.command == "sources":
            value = _sources(root)
        elif args.command in ("train", "evaluate"):
            if args.command == "train" and args.step_research:
                if args.reviewed_manifest:
                    raise ValueError("STEP research cannot combine with reviewed monthly dataset")
                from .step_research import train_step_research

                value = train_step_research(root, as_of=args.as_of)
            elif args.command == "train" and args.reviewed_manifest:
                from .supervised import run_supervised

                value = run_supervised(root, args.reviewed_manifest, as_of=args.as_of)
            elif args.command == "evaluate" and args.run_id:
                from .supervised import verify_model_candidate

                value = verify_model_candidate(root, args.run_id)
            else:
                value = training_gate(root, as_of=args.as_of)
                value["requested_operation"] = args.command
                print(json.dumps(value, indent=2))
                return 2
        elif args.command == "prepare":
            value = build_benchmark_candidate(root, as_of=args.as_of)
        elif args.command == "verify" and args.run_id.startswith("ml-"):
            from .supervised import verify_model_candidate

            value = verify_model_candidate(root, args.run_id)
        elif args.command == "verify":
            item = verify_candidate(root, args.run_id)
            value = {
                "status": "VERIFIED_UNAPPROVED_CANDIDATE",
                "run_id": item["run_id"],
                "benchmark_cells": item["published_cell_count"],
                "model_trained": False,
                "runtime_approved": False,
                "promotion_blockers": promotion_blockers(
                    root, args.run_id, today=date.today()
                ),
            }
        elif args.command == "promote":
            value = promote(root, args.run_id, confirm=args.confirm)
        elif args.command == "rollback":
            value = rollback(root, args.run_id, confirm=args.confirm)
        else:
            value = active_release(root)
        print(json.dumps(value, indent=2, ensure_ascii=False))
        return 0
    except (OSError, ValueError, PermissionError, KeyError, TypeError) as exc:
        print(f"Artifact operation blocked: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
