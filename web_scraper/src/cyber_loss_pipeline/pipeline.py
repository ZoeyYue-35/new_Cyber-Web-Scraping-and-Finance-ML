from __future__ import annotations

import os
import tomllib
from collections import Counter
from pathlib import Path

from .collectors import CaliforniaAGCollector, JsonlCollector, SECCollector
from .export import export_all
from .storage import Store
from .taxonomy import Taxonomy


def run_jsonl(
    input_path: str | Path,
    database_path: str | Path,
    taxonomy_path: str | Path,
) -> dict[str, int]:
    taxonomy = Taxonomy.from_csv(taxonomy_path)
    counts: Counter[str] = Counter()
    with Store(database_path) as store:
        for record in JsonlCollector(input_path).collect():
            counts[store.upsert(record, taxonomy)] += 1
    return {status: counts.get(status, 0) for status in ("new", "updated", "unchanged")}


def run_live(
    sources_path: str | Path,
    database_path: str | Path,
    taxonomy_path: str | Path,
) -> dict[str, object]:
    sources_config = tomllib.loads(Path(sources_path).read_text(encoding="utf-8"))
    taxonomy = Taxonomy.from_csv(taxonomy_path)
    source_counts: dict[str, dict[str, int]] = {}
    skipped: dict[str, str] = {}

    with Store(database_path) as store:
        for source in sources_config.get("sources", []):
            if not source.get("enabled", False):
                continue
            collector_name = source.get("collector")
            if collector_name == "california_ag_csv":
                collector = CaliforniaAGCollector(
                    source["data_url"],
                    source["landing_url"],
                )
            elif collector_name == "sec_edgar":
                ciks = source.get("ciks", [])
                if not ciks:
                    skipped[source["key"]] = (
                        "Set one or more issuer CIKs in config/sources.toml."
                    )
                    continue
                user_agent = os.environ.get("SEC_USER_AGENT", "")
                collector = SECCollector(
                    ciks,
                    user_agent,
                    lookback_days=int(source.get("lookback_days", 30)),
                )
            else:
                raise ValueError(
                    f"Enabled source {source.get('key')!r} has no supported collector"
                )

            counts: Counter[str] = Counter()
            for record in collector.collect():
                counts[store.upsert(record, taxonomy)] += 1
            source_result = {
                status: counts.get(status, 0)
                for status in ("new", "updated", "unchanged")
            }
            if hasattr(collector, "skipped_rows"):
                source_result["skipped_rows"] = collector.skipped_rows
            source_counts[source["key"]] = source_result

    return {"sources": source_counts, "skipped": skipped}


def run_weekly(
    sources_path: str | Path,
    database_path: str | Path,
    taxonomy_path: str | Path,
    output_dir: str | Path,
) -> dict[str, object]:
    collection = run_live(sources_path, database_path, taxonomy_path)
    exported = export_all(database_path, output_dir, taxonomy_path)
    return {
        **collection,
        "exported": [str(path) for path in exported],
    }
