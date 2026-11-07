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

    def test_safe_parse_json_valid_and_collections(self):
        """Verifies safe JSON parsing across Python collections, valid JSON, and Python literal strings."""
        # Python list of dicts directly
        list_input = [{"id": 35, "name": "Comedy"}, {"id": 18, "name": "Drama"}]
        self.assertEqual(safe_parse_json(list_input), list_input)

        # Python single dict
        dict_input = {"id": 28, "name": "Action"}
        self.assertEqual(safe_parse_json(dict_input), [dict_input])

        # Python tuple
        tuple_input = ({"id": 12, "name": "Adventure"},)
        self.assertEqual(safe_parse_json(tuple_input), [{"id": 12, "name": "Adventure"}])

        # Valid string with single quotes (Python literal)
        valid_single = "[{'id': 35, 'name': 'Comedy'}, {'id': 18, 'name': 'Drama'}]"
        res = safe_parse_json(valid_single)
        self.assertEqual(len(res), 2)
        self.assertEqual(res[0]["name"], "Comedy")
        self.assertEqual(res[1]["name"], "Drama")

        # Valid double-quoted JSON string
        valid_json = '[{"id": 80, "name": "Crime"}]'
        res_json = safe_parse_json(valid_json)
        self.assertEqual(len(res_json), 1)
        self.assertEqual(res_json[0]["name"], "Crime")

    def test_safe_parse_json_missing_and_malformed(self):
        """Verifies safe JSON parsing across null, empty, unexpected scalars, and malformed inputs."""
        self.assertEqual(safe_parse_json(None), [])
        self.assertEqual(safe_parse_json(np.nan), [])
        self.assertEqual(safe_parse_json(pd.NA), [])
        self.assertEqual(safe_parse_json(""), [])
        self.assertEqual(safe_parse_json("   "), [])
        self.assertEqual(safe_parse_json("[]"), [])
        self.assertEqual(safe_parse_json("{}"), [])
        self.assertEqual(safe_parse_json("null"), [])
        self.assertEqual(safe_parse_json("None"), [])
        self.assertEqual(safe_parse_json("nan"), [])

        # Non-string scalars
        self.assertEqual(safe_parse_json(12345), [])
        self.assertEqual(safe_parse_json(True), [])

        # Malformed strings should not crash
        self.assertEqual(safe_parse_json("invalid json string {}"), [])
        self.assertEqual(safe_parse_json("[{'unterminated': "), [])

        # List containing non-dict items should filter safely
        self.assertEqual(safe_parse_json(["not_a_dict", 123]), [])

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

    def test_release_date_explicit_four_digit_years(self):
        """Verifies explicit four-digit years and future dates remain untouched."""
        dates = pd.Series(["2025-04-10", "2030-01-01", "1999-12-31", "05/23/1983"])
        parsed = parse_release_date(dates)

        self.assertEqual(parsed.iloc[0].year, 2025)
        self.assertEqual(parsed.iloc[0].month, 4)
        self.assertEqual(parsed.iloc[0].day, 10)

        self.assertEqual(parsed.iloc[1].year, 2030)
        self.assertEqual(parsed.iloc[2].year, 1999)
        self.assertEqual(parsed.iloc[3].year, 1983)

    def test_dataset_availability_reporting(self):
        """Verifies dataset availability dictionary returns valid status for all 3 official files."""
        avail = get_dataset_availability()
        self.assertIn("train.csv", avail)
        self.assertIn("test.csv", avail)
        self.assertIn("sample_submission.csv", avail)
        self.assertIsNotNone(avail["train.csv"], "train.csv not resolved")
        self.assertIsNotNone(avail["test.csv"], "test.csv not resolved")
        self.assertIsNotNone(avail["sample_submission.csv"], "sample_submission.csv not resolved")


if __name__ == "__main__":
    unittest.main()
