import csv
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from cyber_loss_pipeline.export import export_all
from cyber_loss_pipeline.pipeline import run_jsonl


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.db = self.root / "test.sqlite"
        self.input = self.root / "records.jsonl"
        self.taxonomy = Path(__file__).parents[1] / "config" / "taxonomy_mapping.csv"

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_record(self, amount="125000", term="Forensic investigation costs"):
        record = {
            "source_name": "test_fixture",
            "source_record_id": "record-001",
            "source_url": "https://example.invalid/record-001",
            "company_name": "Example Company",
            "incident_date": "2025-01-15",
            "published_date": "2025-02-01",
            "incident_type": "Test incident",
            "individuals_affected": 1000,
            "losses": [
                {
                    "amount": amount,
                    "currency": "USD",
                    "reported_term": term,
                    "evidence_text": "Synthetic unit-test evidence only.",
                    "review_status": "needs_review",
                }
            ],
        }
        self.input.write_text(json.dumps(record) + "\n", encoding="utf-8")

    def test_incremental_import_preserves_revisions(self):
        self._write_record()
        self.assertEqual(run_jsonl(self.input, self.db, self.taxonomy)["new"], 1)
        self.assertEqual(run_jsonl(self.input, self.db, self.taxonomy)["unchanged"], 1)

        self._write_record(amount="150000")
        self.assertEqual(run_jsonl(self.input, self.db, self.taxonomy)["updated"], 1)

        connection = sqlite3.connect(self.db)
        self.assertEqual(connection.execute("SELECT COUNT(*) FROM source_versions").fetchone()[0], 2)
        self.assertEqual(connection.execute("SELECT COUNT(*) FROM loss_observations").fetchone()[0], 2)
        connection.close()

    def test_ambiguous_term_enters_review_queue(self):
        self._write_record(term="Malicious Attack")
        run_jsonl(self.input, self.db, self.taxonomy)
        connection = sqlite3.connect(self.db)
        self.assertEqual(connection.execute("SELECT COUNT(*) FROM review_queue").fetchone()[0], 1)
        connection.close()

    def test_export_creates_wide_model_file(self):
        self._write_record()
        run_jsonl(self.input, self.db, self.taxonomy)
        output = self.root / "exports"
        export_all(self.db, output, self.taxonomy)
        with (output / "model_input_wide.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["Incident Response"], "125000")


if __name__ == "__main__":
    unittest.main()
