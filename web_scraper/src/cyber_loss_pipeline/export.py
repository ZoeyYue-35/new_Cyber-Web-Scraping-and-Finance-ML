from __future__ import annotations

import csv
import sqlite3
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

from .taxonomy import Taxonomy


def export_all(
    database_path: str | Path,
    output_dir: str | Path,
    taxonomy_path: str | Path,
) -> list[Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row

    exported = [
        _export_query(connection, output / "incidents.csv", "SELECT * FROM incidents ORDER BY incident_id"),
        _export_query(
            connection,
            output / "incident_sources.csv",
            """
            SELECT source_id, incident_id, source_name, source_record_id, source_url,
                   published_date, individuals_affected, content_hash,
                   first_observed_at, last_observed_at
            FROM incident_sources
            ORDER BY source_name, source_record_id
            """,
        ),
        _export_query(
            connection,
            output / "loss_observations.csv",
            "SELECT * FROM loss_observations ORDER BY incident_id, observation_id",
        ),
        _export_query(
            connection,
            output / "review_queue.csv",
            """
            SELECT q.review_id, q.observation_id, q.reason, q.status, q.created_at,
                   l.incident_id, l.reported_term, l.amount, l.currency,
                   l.category, l.subcategory, l.evidence_text, l.evidence_locator
            FROM review_queue q
            JOIN loss_observations l USING (observation_id)
            ORDER BY q.review_id
            """,
        ),
    ]
    taxonomy = Taxonomy.from_csv(taxonomy_path)
    exported.append(_export_model_wide(connection, output / "model_input_wide.csv", taxonomy))
    connection.close()
    return exported


def _export_query(connection: sqlite3.Connection, path: Path, query: str) -> Path:
    rows = connection.execute(query).fetchall()
    headers = [item[0] for item in connection.execute(query).description]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows([list(row) for row in rows])
    return path


def _export_model_wide(
    connection: sqlite3.Connection,
    path: Path,
    taxonomy: Taxonomy,
) -> Path:
    categories = taxonomy.categories
    incident_rows = connection.execute(
        """
        SELECT i.incident_id, i.company_name, i.incident_date, s.source_name,
               s.source_record_id
        FROM incidents i
        JOIN incident_sources s USING (incident_id)
        ORDER BY i.incident_id
        """
    ).fetchall()
    totals: dict[str, dict[str, Decimal]] = defaultdict(lambda: defaultdict(Decimal))
    for row in connection.execute(
        """
        SELECT incident_id, category, amount
        FROM loss_observations
        WHERE review_status IN ('confirmed', 'auto_classified')
          AND amount IS NOT NULL
          AND category IS NOT NULL
        """
    ):
        totals[row["incident_id"]][row["category"]] += Decimal(row["amount"])

    headers = [
        "incident_id",
        "company_name",
        "incident_date",
        "source_name",
        "source_record_id",
        *categories,
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for incident in incident_rows:
            output_row = dict(incident)
            for category in categories:
                amount = totals[incident["incident_id"]].get(category)
                output_row[category] = "" if amount is None else format(amount, "f")
            writer.writerow(output_row)
    return path
