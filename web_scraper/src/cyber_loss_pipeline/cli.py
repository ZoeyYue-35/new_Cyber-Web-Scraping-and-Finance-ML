from __future__ import annotations

import argparse
import json
from pathlib import Path

from .export import export_all
from .pipeline import run_jsonl
from .storage import Store


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cyber loss research data pipeline")
    parser.add_argument("--mode", required=True, choices=("init", "backfill", "update", "export"))
    parser.add_argument("--input", type=Path, help="Authorized normalized JSONL input")
    parser.add_argument("--db", type=Path, default=Path("data/cyber_loss.sqlite"))
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=Path("config/taxonomy_mapping.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "init":
        with Store(args.db):
            pass
        print(json.dumps({"database": str(args.db), "status": "initialized"}, indent=2))
        return 0

    if args.mode in {"backfill", "update"}:
        if args.input is None:
            raise SystemExit("--input is required for backfill and update modes")
        result = run_jsonl(args.input, args.db, args.taxonomy)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    paths = export_all(args.db, args.output_dir, args.taxonomy)
    print(json.dumps({"exported": [str(path) for path in paths]}, indent=2))
    return 0
