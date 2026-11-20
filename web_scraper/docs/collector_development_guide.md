# Collector Development Guide

This guide explains how to add a source collector to the Cyber Loss pipeline. It is intended as a practical handoff for a teammate implementing another source.

## 1. Understand the data flow

```text
web_scraper/config/sources.toml
        │  source URLs, enabled flag, collector key
        ▼
web_scraper/src/cyber_loss_pipeline/collectors/<source>.py
        │  fetch source → parse → yield SourceRecord objects
        ▼
web_scraper/src/cyber_loss_pipeline/pipeline.py
        │  selects collector → calls Store.upsert(record, taxonomy)
        ▼
web_scraper/src/cyber_loss_pipeline/storage.py
        │  writes normalized records to SQLite
        ▼
web_scraper/src/cyber_loss_pipeline/export.py
        │  exports database tables
        ▼
web_scraper/data/processed/
   incidents.csv, incident_sources.csv, loss_observations.csv,
   review_queue.csv, model_input_wide.csv, loss_observations.xlsx
```

**Important:** A collector does not choose the output folder and normally should not write files directly. The shared pipeline handles persistence and export. The default database and export paths are relative to the directory from which the CLI is run, so run commands from `web_scraper/` to use `web_scraper/data/processed/`.

## 2. Inspect the source before coding

For the source you are assigned:

1. Identify the official landing page and the actual data endpoint (HTML page, CSV, JSON, API, or linked documents).
2. Confirm which fields are available and inspect representative records, including missing or unusual values.
3. Decide what the assignment requires: parsing a published structured download is different from parsing the HTML page or the linked notice documents. If the requirement specifically says to scrape page content, confirm that a CSV/API endpoint is acceptable before implementing only that endpoint.
4. Record the source URL and any access, rate-limit, or terms considerations. Do not collect or store unnecessary personal information.
5. Do not infer monetary losses, incident dates, or company attributes that the source does not report. Keep unknown fields blank/`None`.

## 3. Implement the collector

Create a module under:

```text
web_scraper/src/cyber_loss_pipeline/collectors/<source_name>.py
```

The collector should:

- Subclass `Collector` and implement `collect()`.
- Fetch source data using the shared HTTP helper where appropriate (`collectors/http.py` provides `fetch_bytes` with timeout, retry, and response-size handling).
- Parse the source into one `SourceRecord` per source record/incident candidate.
- Yield records rather than writing them to `data/` or opening the database itself.
- Preserve source-backed values and useful raw fields for auditability.
- Generate stable source-record and incident identifiers so repeated runs can recognize the same records.
- Handle missing values and schema changes deliberately; fail clearly if required columns or structures disappear.
- Set `losses=[]` when the source provides no supported monetary loss observations. Do not invent amounts to populate the loss table.

`SourceRecord.from_mapping(...)` requires `source_name`, `source_record_id`, `source_url`, and `company_name`. Common optional fields include `incident_key`, `incident_date`, `published_date`, `incident_type`, `event`, `individuals_affected`, `company_features`, `losses`, and `raw`. See `cyber_loss_pipeline/models.py` for the data contract.

A minimal shape is:

```python
from collections.abc import Iterator

from ..models import SourceRecord
from .base import Collector


class ExampleCollector(Collector):
    def __init__(self, data_url: str, landing_url: str) -> None:
        self.data_url = data_url
        self.landing_url = landing_url

    def collect(self) -> Iterator[SourceRecord]:
        # Fetch and parse the source here.
        # Build one mapping per source record and yield:
        yield SourceRecord.from_mapping(
            {
                "source_name": "example_source",
                "source_record_id": "stable-source-record-id",
                "incident_key": "stable-incident-key",
                "source_url": self.data_url,
                "company_name": "Company as reported by the source",
                "event": "Incident description as reported",
                "incident_date": None,
                "published_date": None,
                "losses": [],
                "raw": {"landing_url": self.landing_url},
            }
        )
```

Adapt the parsing and mapping to the assigned source; do not copy placeholder values into real records.

## 4. Add the collector to the shared pipeline

A complete collector requires integration outside its own module:

1. **Export the class** from `collectors/__init__.py` and add it to `__all__` if the pipeline imports collectors from that package.
2. **Add a source entry** to `config/sources.toml` with a unique `key`, `name`, `landing_url`, `enabled`, `collector`, and any source-specific settings needed by the collector.
3. **Add routing** to `pipeline.py` so that the configured collector key creates the new collector with its settings.
4. Keep database and file-output logic in the existing shared `Store` and export code unless the team has explicitly agreed to change the pipeline contract.

For example, the routing pattern in `pipeline.py` matches a configured collector name and constructs the appropriate collector. An enabled source with no supported routing is an error, not a successful empty collection.

## 5. Write tests

Add tests in `web_scraper/tests/` (or extend the relevant existing test file). At minimum, test:

- A representative valid source response produces the expected `SourceRecord` values.
- Missing optional fields remain missing rather than becoming fabricated values.
- Empty or malformed rows are skipped or rejected according to a documented rule.
- A changed/missing required schema raises a clear error.
- Stable inputs produce stable IDs.
- The collector works without live network access in unit tests: mock the HTTP fetch/helper and provide fixture content.

If source-level integration has been explicitly approved, test it separately from deterministic unit tests; do not make ordinary tests depend on an external website being available.

## 6. Run and verify the full pipeline

From the `web_scraper/` directory:

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python -m cyber_loss_pipeline --mode weekly
```

Check the command output for the source's record counts (`new`, `updated`, `unchanged`, and, if supported, `skipped_rows`) and exported paths. Then verify that:

- `data/processed/cyber_loss.sqlite` exists and contains the collected source records;
- `data/processed/incidents.csv` contains the incident candidates;
- `data/processed/incident_sources.csv` contains source names, record IDs, and source URLs;
- loss-related exports contain only observations actually supported by the source.

The CLI also supports `init`, `backfill`, `update`, and `export`, but those modes do not replace `weekly` for live collection. `export` regenerates exports from the database; it does not fetch sources.

## 7. Definition of done

A source collector is ready to hand off when:

- Its source access method matches the assignment (HTML vs. published CSV/API is explicitly understood).
- It emits valid, source-backed `SourceRecord` objects with stable IDs.
- The collector is exported, configured, and routed through the pipeline.
- Unit tests cover parsing and failure cases without live network access.
- A `weekly` run stores records in SQLite and produces the expected exports.
- The teammate can explain which source fields are captured, which are missing, and why.
