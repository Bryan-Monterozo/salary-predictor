from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import inspect, stage


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect a reviewed source locally; staging requires explicit approval."
    )
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--input", help="Must match the input in the reviewed manifest")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", default="data/staged")
    parser.add_argument("--preview", type=int, default=2)
    parser.add_argument("--commit", action="store_true", help="Write immutable local stage files")
    parser.add_argument(
        "--strict", action="store_true", help="Block staging when any records are rejected"
    )
    args = parser.parse_args(argv)

    try:
        report, records = inspect(
            args.root, args.manifest, args.source_id,
            input_path=args.input, preview_limit=args.preview,
        )
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
        if args.commit:
            if args.strict and report.rejected_count:
                raise ValueError("Strict mode blocks staging with rejected records")
            files = stage(args.root, report, records, output=args.output)
            print(json.dumps({"staged": files}, indent=2, ensure_ascii=False))
        return 0
    except (OSError, ValueError, PermissionError, KeyError) as exc:
        print(f"Ingestion blocked: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
