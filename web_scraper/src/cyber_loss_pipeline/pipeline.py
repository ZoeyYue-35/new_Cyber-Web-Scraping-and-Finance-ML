from __future__ import annotations

from collections import Counter
from pathlib import Path

from .collectors import JsonlCollector
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
