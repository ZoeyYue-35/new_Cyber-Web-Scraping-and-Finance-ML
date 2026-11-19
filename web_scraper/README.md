# Cyber Loss Intelligence Pipeline

This directory contains the starter infrastructure for the cyber loss web-scraping project. It is intentionally small and testable. It does not claim that any live website collector is working yet.

## What works now

- Imports normalized source records from a local JSONL file.
- Stores incidents, original source records, source versions, loss observations, update history, and review items in SQLite.
- Detects `new`, `updated`, and `unchanged` source records using stable identifiers and SHA-256 content hashes.
- Preserves older loss observations rather than silently overwriting revised amounts.
- Maps reported loss terms through a configurable taxonomy table.
- Sends uncertain mappings to the review queue.
- Exports research CSV files, including a wide company-by-loss-category table.
- Runs entirely with the Python standard library.

## What is not implemented yet

- Live HHS OCR, SEC EDGAR, VCDB, California AG, or other collectors.
- Cross-source entity resolution for incidents reported by several sources.
- PDF loss extraction or LLM-assisted classification.
- Automatic conversion into the repository's legacy aggregate CSV files.

The live collectors remain disabled until their access method, rate limits, terms, and schemas have been verified.

## Setup

Run these commands from the `web_scraper` directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
```

## Commands

Initialize an empty database:

```bash
python -m cyber_loss_pipeline --mode init
```

Import a JSONL backfill:

```bash
python -m cyber_loss_pipeline \
  --mode backfill \
  --input path/to/authorized_source_records.jsonl
```

Run an incremental import of the same source format:

```bash
python -m cyber_loss_pipeline \
  --mode update \
  --input path/to/authorized_source_records.jsonl
```

Export research CSV files:

```bash
python -m cyber_loss_pipeline \
  --mode export \
  --output-dir data/processed
```

Default database: `data/cyber_loss.sqlite`

Default taxonomy: `config/taxonomy_mapping.csv`

## JSONL input contract

Each line is one source record. A source record can contain zero or more loss observations.

```json
{
  "source_name": "source_key",
  "source_record_id": "stable-record-id",
  "source_url": "https://example.org/record/stable-record-id",
  "company_name": "Company name as reported",
  "incident_date": "2025-01-15",
  "published_date": "2025-02-01",
  "incident_type": "Hacking/IT Incident",
  "individuals_affected": 1000,
  "losses": [
    {
      "amount": "125000.00",
      "currency": "USD",
      "reported_term": "Forensic investigation costs",
      "evidence_text": "Short source-backed evidence excerpt",
      "evidence_locator": "page 4",
      "review_status": "needs_review"
    }
  ]
}
```

Missing monetary amounts must stay missing. Do not encode an unknown amount as zero.

## Export files

- `incidents.csv`: one internal incident record per imported source record at this stage.
- `incident_sources.csv`: source URL, source identifier, hashes, and observation timestamps.
- `loss_observations.csv`: one row per observed loss amount with evidence and review status.
- `review_queue.csv`: ambiguous mappings and other items needing human review.
- `model_input_wide.csv`: one row per incident, with the seven target loss categories as columns.

`model_input_wide.csv` excludes `needs_review` observations. This prevents ambiguous classifications from entering modeling data automatically.

## Relationship to the existing model

The existing prediction code currently reads legacy aggregate CSV files such as `economic_loss.csv`, `response_costs.csv`, `litigated_cases.csv`, and `fines_penalties.csv`. Those files use a different classification system from Professor Arora's target taxonomy.

This starter pipeline therefore exports a clean research layer first. It does not overwrite the legacy files. See `docs/model_integration.md` for the proposed integration boundary.
