"""
test_data_pipeline.py
Unit tests for TMDB Box Office dataset loading, JSON parsing, and release date parsing.
"""

import unittest
import pandas as pd
import numpy as np

from src.data_loader import (
    load_tmdb_train_data,
    load_tmdb_test_data,
    safe_parse_json,
    parse_release_date,
    resolve_file_path,
    get_dataset_availability,
)


class TestDataPipeline(unittest.TestCase):
    """Verifies dataset loading, resolution, and safe metadata parsing."""

    def test_train_data_loading(self):
        """Verifies train.csv loads correctly with expected columns and non-empty rows."""
        df, msg = load_tmdb_train_data()
        self.assertIsNotNone(df, f"Failed to load train.csv: {msg}")
        self.assertGreaterEqual(len(df), 3000)
        self.assertIn("revenue", df.columns)
        self.assertIn("budget", df.columns)
        self.assertIn("genres", df.columns)
        self.assertIn("release_date", df.columns)

    def test_safe_parse_json(self):
        """Verifies safe JSON parsing across valid, null, empty, and malformed inputs."""
        # Valid list of dicts
        valid_str = "[{'id': 35, 'name': 'Comedy'}, {'id': 18, 'name': 'Drama'}]"
        res = safe_parse_json(valid_str)
        self.assertEqual(len(res), 2)
        self.assertEqual(res[0]["name"], "Comedy")
        self.assertEqual(res[1]["name"], "Drama")

        # Null / NaN / empty
        self.assertEqual(safe_parse_json(None), [])
        self.assertEqual(safe_parse_json(np.nan), [])
        self.assertEqual(safe_parse_json(""), [])
        self.assertEqual(safe_parse_json("[]"), [])

        # Malformed strings should not crash
        self.assertEqual(safe_parse_json("invalid json string {}"), [])

    def test_release_date_century_correction(self):
        """Verifies 2-digit year conversion maps 1900s correctly (e.g. 65 -> 1965, not 2065)."""
        dates = pd.Series(["4/7/65", "8/6/87", "2/20/15", "11/12/20", "invalid", None])
        parsed = parse_release_date(dates)

        self.assertEqual(parsed.iloc[0].year, 1965)
        self.assertEqual(parsed.iloc[1].year, 1987)
        self.assertEqual(parsed.iloc[2].year, 2015)
        self.assertEqual(parsed.iloc[3].year, 2020)
        self.assertTrue(pd.isna(parsed.iloc[4]))
        self.assertTrue(pd.isna(parsed.iloc[5]))

    def test_dataset_availability_reporting(self):
        """Verifies dataset availability dictionary returns valid status."""
        avail = get_dataset_availability()
        self.assertIn("train.csv", avail)
        self.assertIn("test.csv", avail)
        self.assertIn("sample_submission.csv", avail)
        self.assertIsNotNone(avail["train.csv"])


if __name__ == "__main__":
    unittest.main()
