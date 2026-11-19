from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .models import LossCandidate, SourceRecord
from .normalization import (
    content_hash,
    decimal_to_text,
    normalize_company_name,
    normalize_iso_date,
    stable_identifier,
)
from .taxonomy import Taxonomy, TaxonomyMatch


SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS incidents (
    incident_id TEXT PRIMARY KEY,
    company_name TEXT NOT NULL,
    incident_date TEXT,
    incident_type TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS incident_sources (
    source_id TEXT PRIMARY KEY,
    incident_id TEXT NOT NULL REFERENCES incidents(incident_id),
    source_name TEXT NOT NULL,
    source_record_id TEXT NOT NULL,
    source_url TEXT NOT NULL,
    published_date TEXT,
    individuals_affected INTEGER,
    content_hash TEXT NOT NULL,
    raw_json TEXT NOT NULL,
    first_observed_at TEXT NOT NULL,
    last_observed_at TEXT NOT NULL,
    UNIQUE(source_name, source_record_id)
);

CREATE TABLE IF NOT EXISTS source_versions (
    version_id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id TEXT NOT NULL REFERENCES incident_sources(source_id),
    content_hash TEXT NOT NULL,
    raw_json TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    UNIQUE(source_id, content_hash)
);

CREATE TABLE IF NOT EXISTS loss_observations (
    observation_id TEXT PRIMARY KEY,
    incident_id TEXT NOT NULL REFERENCES incidents(incident_id),
    source_id TEXT NOT NULL REFERENCES incident_sources(source_id),
    amount TEXT,
    currency TEXT,
    reported_term TEXT,
    category TEXT,
    subcategory TEXT,
    evidence_text TEXT,
    evidence_locator TEXT,
    review_status TEXT NOT NULL,
    observed_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS update_history (
    update_id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_name TEXT NOT NULL,
    source_record_id TEXT NOT NULL,
    sync_status TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    observed_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS review_queue (
    review_id INTEGER PRIMARY KEY AUTOINCREMENT,
    observation_id TEXT NOT NULL REFERENCES loss_observations(observation_id),
    reason TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',
    created_at TEXT NOT NULL,
    UNIQUE(observation_id, reason)
);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class Store:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript(SCHEMA)
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def upsert(self, record: SourceRecord, taxonomy: Taxonomy) -> str:
        observed_at = utc_now()
        incident_id = stable_identifier(record.source_name, record.source_record_id)
        source_id = stable_identifier("source", record.source_name, record.source_record_id)
        raw_json = json.dumps(record.raw, sort_keys=True, ensure_ascii=False)
        record_hash = content_hash(record.raw)

        existing = self.connection.execute(
            "SELECT content_hash FROM incident_sources WHERE source_id = ?",
            (source_id,),
        ).fetchone()
        sync_status = "new" if existing is None else (
            "unchanged" if existing["content_hash"] == record_hash else "updated"
        )

        company_name = normalize_company_name(record.company_name)
        incident_date = normalize_iso_date(record.incident_date)
        published_date = normalize_iso_date(record.published_date)

        with self.connection:
            self.connection.execute(
                """
                INSERT INTO incidents (
                    incident_id, company_name, incident_date, incident_type, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(incident_id) DO UPDATE SET
                    company_name = excluded.company_name,
                    incident_date = excluded.incident_date,
                    incident_type = excluded.incident_type,
                    updated_at = excluded.updated_at
                """,
                (
                    incident_id,
                    company_name,
                    incident_date,
                    record.incident_type,
                    observed_at,
                    observed_at,
                ),
            )
            self.connection.execute(
                """
                INSERT INTO incident_sources (
                    source_id, incident_id, source_name, source_record_id, source_url,
                    published_date, individuals_affected, content_hash, raw_json,
                    first_observed_at, last_observed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_id) DO UPDATE SET
                    source_url = excluded.source_url,
                    published_date = excluded.published_date,
                    individuals_affected = excluded.individuals_affected,
                    content_hash = excluded.content_hash,
                    raw_json = excluded.raw_json,
                    last_observed_at = excluded.last_observed_at
                """,
                (
                    source_id,
                    incident_id,
                    record.source_name,
                    record.source_record_id,
                    record.source_url,
                    published_date,
                    record.individuals_affected,
                    record_hash,
                    raw_json,
                    observed_at,
                    observed_at,
                ),
            )
            self.connection.execute(
                """
                INSERT OR IGNORE INTO source_versions (
                    source_id, content_hash, raw_json, observed_at
                ) VALUES (?, ?, ?, ?)
                """,
                (source_id, record_hash, raw_json, observed_at),
            )
            for candidate in record.losses:
                self._insert_loss(
                    incident_id,
                    source_id,
                    candidate,
                    taxonomy,
                    observed_at,
                )
            self.connection.execute(
                """
                INSERT INTO update_history (
                    source_name, source_record_id, sync_status, content_hash, observed_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    record.source_name,
                    record.source_record_id,
                    sync_status,
                    record_hash,
                    observed_at,
                ),
            )
        return sync_status

    def _insert_loss(
        self,
        incident_id: str,
        source_id: str,
        candidate: LossCandidate,
        taxonomy: Taxonomy,
        observed_at: str,
    ) -> None:
        match = self._resolve_taxonomy(candidate, taxonomy)
        review_status = candidate.review_status
        if match.status == "approved_alias" and review_status == "needs_review":
            review_status = "auto_classified"
        elif match.status == "canonical" and review_status == "needs_review":
            review_status = "auto_classified"
        elif match.status == "needs_review":
            review_status = "needs_review"

        amount = decimal_to_text(candidate.amount)
        observation_id = stable_identifier(
            source_id,
            amount,
            candidate.currency,
            candidate.reported_term,
            candidate.evidence_text,
            candidate.evidence_locator,
            length=32,
        )
        self.connection.execute(
            """
            INSERT OR IGNORE INTO loss_observations (
                observation_id, incident_id, source_id, amount, currency,
                reported_term, category, subcategory, evidence_text,
                evidence_locator, review_status, observed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                observation_id,
                incident_id,
                source_id,
                amount,
                candidate.currency,
                candidate.reported_term,
                match.category,
                match.subcategory,
                candidate.evidence_text,
                candidate.evidence_locator,
                review_status,
                observed_at,
            ),
        )
        if review_status == "needs_review":
            reason = match.notes or "Human review required"
            self.connection.execute(
                """
                INSERT OR IGNORE INTO review_queue (
                    observation_id, reason, status, created_at
                ) VALUES (?, ?, 'open', ?)
                """,
                (observation_id, reason, observed_at),
            )

    @staticmethod
    def _resolve_taxonomy(candidate: LossCandidate, taxonomy: Taxonomy) -> TaxonomyMatch:
        if candidate.category and taxonomy.validate(candidate.category, candidate.subcategory):
            return TaxonomyMatch(candidate.category, candidate.subcategory, "canonical")
        return taxonomy.match(candidate.reported_term)
