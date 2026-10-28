"""
test_data_explorer.py
Automated test suite for Phase 1 DataLab: Interactive Dataset Explorer.

Tests core profiling, schema evaluation, missingness auditing, filtering logic,
pagination, exports, encoding fallbacks, and edge-case resilience.
Uses synthetic test fixtures so tests run quickly and reliably without external dependencies.
"""

import io
import math
import os
import unittest
import numpy as np
import pandas as pd

from src.data_explorer import (
    resolve_builtin_netflix_path,
    compute_dataset_identity,
    load_csv_from_file_or_bytes,
    get_missing_mask,
    classify_column_type,
    compute_overview_metrics,
    compute_column_schema,
    compute_missing_rankings,
    get_missingness_matrix,
    compute_numeric_stats,
    compute_categorical_stats,
    compute_datetime_stats,
    preview_datetime_parsing,
    apply_explorer_filters,
    paginate_dataframe,
    export_filtered_csv,
    export_profile_csv,
)
from src.data_generator import ensure_data_exists


class TestDataLabCoreExplorer(unittest.TestCase):
    """Test suite for DataLab Phase 1 core exploration and data engine."""

    def setUp(self):
        """Create a synthetic test fixture with diverse types, missingness, and anomalies."""
        self.synthetic_df = pd.DataFrame({
            "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "title": [
                "Inception", "Stranger Things", "Breaking Bad", "The Crown",
                "Dark", "Interstellar", "Mindhunter", "Narcos", "Ozark", "Peaky Blinders"
            ],
            "type": ["Movie", "TV Show", "TV Show", "TV Show", "TV Show", "Movie", "TV Show", "TV Show", "TV Show", "TV Show"],
            "score": [8.8, 8.7, 9.5, 8.6, 8.7, 8.6, 8.6, 8.8, 8.5, 8.8],
            "budget_mil": [160.0, np.nan, 3.5, 13.0, np.nan, 165.0, np.nan, 25.0, np.nan, 10.0],
            "runtime_min": [148, np.nan, np.nan, np.nan, np.nan, 169, np.nan, np.nan, np.nan, np.nan],
            "country": ["United States", "United States", "United States", "United Kingdom", "Germany", "United States", "United States", "Colombia", "United States", "United Kingdom"],
            "notes": ["Good", "   ", "", None, "Classic", "Sci-fi", "Thriller", "Crime", "Drama", "Period"],
            "date_released": pd.to_datetime([
                "2010-07-16", "2016-07-15", "2008-01-20", "2016-11-04", "2017-12-01",
                "2014-11-07", "2017-10-13", "2015-08-28", "2017-07-21", "2013-09-12"
            ]),
            "is_active": [True, True, False, True, False, False, False, False, False, False],
            "constant_col": [42, 42, 42, 42, 42, 42, 42, 42, 42, 42],
            "all_null_col": [np.nan] * 10,
            "col with spaces & #punctuation": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
        })

    def test_builtin_netflix_source_resolution(self):
        """Verifies that the built-in Netflix catalog path is resolved robustly."""
        path = resolve_builtin_netflix_path()
        self.assertTrue(os.path.exists(path), f"Resolved path '{path}' does not exist.")
        self.assertTrue(path.lower().endswith("netflix_titles.csv"))

        # Also test ensure_data_exists
        gen_path = ensure_data_exists()
        self.assertTrue(os.path.exists(gen_path))

    def test_load_csv_from_bytes_utf8_and_bom(self):
        """Verifies loading CSV from bytes with UTF-8 and UTF-8 with BOM."""
        csv_text = "id,name,value\n1,Alpha,100\n2,Beta,200\n"
        # Standard UTF-8
        df1, meta1 = load_csv_from_file_or_bytes(csv_text.encode("utf-8"), filename="test_utf8.csv")
        self.assertEqual(len(df1), 2)
        self.assertEqual(list(df1.columns), ["id", "name", "value"])
        self.assertEqual(meta1["row_count"], 2)

        # UTF-8 with BOM
        df2, meta2 = load_csv_from_file_or_bytes(csv_text.encode("utf-8-sig"), filename="test_bom.csv")
        self.assertEqual(len(df2), 2)
        self.assertIn("id", df2.columns)

    def test_load_csv_from_bytes_latin1_encoding(self):
        """Verifies loading CSV encoded in Latin-1 with accented characters."""
        csv_latin1 = "id,name,city\n1,François,Montréal\n2,José,São Paulo\n".encode("latin-1")
        df, meta = load_csv_from_file_or_bytes(csv_latin1, filename="latin1.csv")
        self.assertEqual(len(df), 2)
        self.assertEqual(df.loc[0, "name"], "François")
        self.assertEqual(df.loc[1, "city"], "São Paulo")

    def test_load_csv_empty_file_handling(self):
        """Verifies that empty (0-byte) or unusable CSVs raise descriptive ValueErrors."""
        with self.assertRaises(ValueError) as ctx:
            load_csv_from_file_or_bytes(b"", filename="empty.csv")
        self.assertIn("empty", str(ctx.exception).lower())

        # Header only without data is valid structure (0 rows, 2 columns)
        df, meta = load_csv_from_file_or_bytes(b"col1,col2\n", filename="header_only.csv")
        self.assertEqual(len(df), 0)
        self.assertEqual(df.shape[1], 2)

    def test_load_csv_duplicate_columns(self):
        """Verifies that duplicate column names are parsed and noted in warnings."""
        csv_dups = b"colA,colB,colA\n1,2,3\n4,5,6\n"
        df, meta = load_csv_from_file_or_bytes(csv_dups, filename="dups.csv")
        self.assertEqual(len(df), 2)
        self.assertEqual(df.shape[1], 3)
        self.assertTrue(len(meta["duplicate_columns"]) > 0 or len(meta["warnings"]) > 0)

    def test_missing_mask_treat_empty_as_missing(self):
        """Verifies consistent missing-value detection for NaN, empty string, and whitespace."""
        df = self.synthetic_df.copy()

        # With treat_empty_as_missing=True (default)
        mask_true = get_missing_mask(df, treat_empty_as_missing=True)
        # In notes column: row index 1 is "   " (whitespace), row index 2 is "" (empty), row index 3 is None
        self.assertTrue(mask_true.loc[1, "notes"], "Whitespace-only string should be marked missing.")
        self.assertTrue(mask_true.loc[2, "notes"], "Empty string should be marked missing.")
        self.assertTrue(mask_true.loc[3, "notes"], "None should be marked missing.")
        self.assertFalse(mask_true.loc[0, "notes"], "Valid string 'Good' should NOT be missing.")

        # With treat_empty_as_missing=False
        mask_false = get_missing_mask(df, treat_empty_as_missing=False)
        self.assertFalse(mask_false.loc[1, "notes"], "Whitespace-only should not be missing if disabled.")
        self.assertFalse(mask_false.loc[2, "notes"], "Empty string should not be missing if disabled.")
        self.assertTrue(mask_false.loc[3, "notes"], "None is always missing.")

        # Immutability check: original df must not be altered
        self.assertEqual(df.loc[1, "notes"], "   ")

    def test_column_type_classification(self):
        """Verifies that columns are classified into correct logical types."""
        df = self.synthetic_df
        self.assertEqual(classify_column_type(df["id"]), "Numeric")
        self.assertEqual(classify_column_type(df["score"]), "Numeric")
        self.assertEqual(classify_column_type(df["title"]), "Categorical/Text")
        self.assertEqual(classify_column_type(df["date_released"]), "Datetime")
        self.assertEqual(classify_column_type(df["is_active"]), "Boolean")

    def test_compute_overview_metrics(self):
        """Verifies overview metrics calculations from actual active dataset."""
        df = self.synthetic_df
        metrics = compute_overview_metrics(df, treat_empty_as_missing=True)

        self.assertEqual(metrics["total_rows"], 10)
        self.assertEqual(metrics["total_columns"], 13)
        self.assertEqual(metrics["duplicate_rows"], 0)
        self.assertEqual(metrics["duplicate_rows_pct"], 0.0)
        self.assertTrue(metrics["total_cells"] == 130)
        self.assertTrue(metrics["total_missing_cells"] > 0)
        self.assertTrue(metrics["missing_cells_pct"] > 0)
        self.assertTrue(metrics["numeric_cols_count"] >= 4)
        self.assertTrue(metrics["datetime_cols_count"] >= 1)
        self.assertTrue(metrics["boolean_cols_count"] >= 1)
        self.assertTrue("B" in metrics["memory_usage_str"] or "KB" in metrics["memory_usage_str"])

        # Test with duplicate rows added
        df_with_dups = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        dup_metrics = compute_overview_metrics(df_with_dups)
        self.assertEqual(dup_metrics["duplicate_rows"], 1)
        self.assertEqual(dup_metrics["total_rows"], 11)

    def test_compute_column_schema(self):
        """Verifies interactive column schema table generation."""
        df = self.synthetic_df
        schema_df = compute_column_schema(df, treat_empty_as_missing=True)

        self.assertEqual(len(schema_df), df.shape[1])
        expected_cols = [
            "Column", "Logical Type", "Data Type", "Non-Null Count",
            "Missing Count", "Missing %", "Distinct Values", "Sample Values"
        ]
        self.assertEqual(list(schema_df.columns), expected_cols)

        # For every column: Non-Null Count + Missing Count must equal total_rows
        for _, row in schema_df.iterrows():
            self.assertEqual(row["Non-Null Count"] + row["Missing Count"], 10)

        # Punctuation column should be preserved
        self.assertIn("col with spaces & #punctuation", schema_df["Column"].values)

    def test_missing_rankings_and_matrix(self):
        """Verifies ranking of missing columns and sampled matrix heatmap."""
        df = self.synthetic_df
        rankings = compute_missing_rankings(df, treat_empty_as_missing=True)

        # all_null_col must have 100% missing rate and be at top
        top_row = rankings.iloc[0]
        self.assertTrue(top_row["Missing %"] >= rankings.iloc[1]["Missing %"])
        self.assertIn("all_null_col", rankings["Column"].tolist())

        # Test matrix sampling
        matrix_df, was_sampled, total_rows = get_missingness_matrix(df, max_rows=5, treat_empty_as_missing=True)
        self.assertTrue(was_sampled)
        self.assertEqual(total_rows, 10)
        self.assertLessEqual(len(matrix_df), 5)
        self.assertEqual(matrix_df.shape[1], df.shape[1])

    def test_numeric_stats_and_outliers(self):
        """Verifies numeric statistics, quartile calculations, and outlier detection."""
        scores = self.synthetic_df["score"]
        stats = compute_numeric_stats(scores)

        self.assertTrue(stats["has_data"])
        self.assertEqual(stats["valid_count"], 10)
        self.assertAlmostEqual(stats["min"], 8.5, places=1)
        self.assertAlmostEqual(stats["max"], 9.5, places=1)
        self.assertTrue(stats["q25"] <= stats["median"] <= stats["q75"])
        self.assertEqual(stats["iqr"], round(stats["q75"] - stats["q25"], 4))
        self.assertFalse(stats["is_constant"])

        # Test constant column
        const_stats = compute_numeric_stats(self.synthetic_df["constant_col"])
        self.assertTrue(const_stats["is_constant"])
        self.assertEqual(const_stats["min"], 42.0)
        self.assertEqual(const_stats["max"], 42.0)
        self.assertEqual(const_stats["std"], 0.0)

        # Test all null column
        null_stats = compute_numeric_stats(self.synthetic_df["all_null_col"])
        self.assertFalse(null_stats["has_data"])
        self.assertIn("reason", null_stats)

    def test_categorical_stats_and_frequency(self):
        """Verifies categorical top category and frequency table computation."""
        countries = self.synthetic_df["country"]
        cat_stats = compute_categorical_stats(countries, top_n=5)

        self.assertTrue(cat_stats["has_data"])
        self.assertEqual(cat_stats["top_category"], "United States")
        self.assertEqual(cat_stats["top_category_count"], 6)
        self.assertEqual(cat_stats["top_category_pct"], 60.0)
        self.assertIn("freq_table", cat_stats)
        self.assertEqual(len(cat_stats["freq_table"]), 4)  # US, UK, Germany, Colombia

    def test_datetime_stats_and_parsing_preview(self):
        """Verifies datetime statistics and non-destructive parsing preview."""
        dates = self.synthetic_df["date_released"]
        dt_stats = compute_datetime_stats(dates)

        self.assertTrue(dt_stats["has_data"])
        self.assertEqual(dt_stats["min_date"], "2008-01-20")
        self.assertEqual(dt_stats["max_date"], "2017-12-01")
        self.assertTrue(dt_stats["span_days"] > 3000)

        # Test string datetime preview on a string series
        str_dates = pd.Series(["January 1, 2020", "2021-05-15", "invalid date", "July 4, 2018"])
        preview = preview_datetime_parsing(str_dates)
        self.assertEqual(preview["parsed_count"], 3)
        self.assertEqual(preview["failed_count"], 1)
        self.assertEqual(preview["success_rate_pct"], 75.0)

    def test_explorer_filters_isolated_and_combined(self):
        """Verifies global search, numeric, categorical, text, and missingness filters."""
        df = self.synthetic_df

        # 1. Global search
        f_search = {"global_search": "nolan or inception"}
        # Match 'inception'
        res_search = apply_explorer_filters(df, {"global_search": "inception"})
        self.assertEqual(len(res_search), 1)
        self.assertEqual(res_search.iloc[0]["title"], "Inception")

        # 2. Numeric range filter
        res_num = apply_explorer_filters(df, {"numeric_ranges": {"score": (8.8, 9.6)}})
        # Should include scores: 8.8, 9.5, 8.8, 8.8 (4 titles: Inception, Breaking Bad, Narcos, Peaky Blinders)
        self.assertEqual(len(res_num), 4)

        # 3. Categorical selection
        res_cat = apply_explorer_filters(df, {"categorical_selections": {"type": ["Movie"]}})
        self.assertEqual(len(res_cat), 2)  # Inception, Interstellar

        # 4. Text substring filter
        res_txt = apply_explorer_filters(df, {"text_searches": {"title": "the"}})
        # "The Crown"
        self.assertEqual(len(res_txt), 1)
        self.assertEqual(res_txt.iloc[0]["title"], "The Crown")

        # 5. Missingness filter (only missing)
        res_missing = apply_explorer_filters(df, {"missing_filters": {"budget_mil": "missing_only"}})
        self.assertEqual(len(res_missing), 4)

        # 6. Combined filters with AND logic
        combined_filters = {
            "categorical_selections": {"type": ["Movie"]},
            "numeric_ranges": {"score": (8.7, 9.0)},
        }
        res_comb = apply_explorer_filters(df, combined_filters)
        self.assertEqual(len(res_comb), 1)
        self.assertEqual(res_comb.iloc[0]["title"], "Inception")

        # 7. Empty result filter
        res_empty = apply_explorer_filters(df, {"global_search": "nonexistent_query_xyz"})
        self.assertEqual(len(res_empty), 0)

        # 8. Reset / Empty filters returns complete DataFrame
        res_reset = apply_explorer_filters(df, {})
        self.assertEqual(len(res_reset), len(df))

        # Immutability check: original df unchanged
        self.assertEqual(len(df), 10)

    def test_paginate_dataframe_and_sorting(self):
        """Verifies pagination slices, total pages, and sorting."""
        df = self.synthetic_df

        # Page 1 of page size 4
        page1, total_pages, total_recs = paginate_dataframe(df, page_number=1, page_size=4)
        self.assertEqual(len(page1), 4)
        self.assertEqual(total_pages, 3)  # 10 records / 4 per page = 3 pages
        self.assertEqual(total_recs, 10)
        self.assertEqual(page1.iloc[0]["title"], "Inception")

        # Page 3 (last page, should have 2 records)
        page3, _, _ = paginate_dataframe(df, page_number=3, page_size=4)
        self.assertEqual(len(page3), 2)

        # Sorting: sort by score descending
        sorted_page, _, _ = paginate_dataframe(df, page_number=1, page_size=5, sort_by="score", ascending=False)
        self.assertEqual(sorted_page.iloc[0]["title"], "Breaking Bad")
        self.assertEqual(sorted_page.iloc[0]["score"], 9.5)

    def test_exports_clean_and_unmodified(self):
        """Verifies CSV export for filtered records and schema profile."""
        df = self.synthetic_df.copy()
        # Add an internal helper column starting with '__'
        df["__helper_internal_id"] = range(len(df))

        csv_str = export_filtered_csv(df)
        self.assertNotIn("__helper_internal_id", csv_str)
        self.assertIn("Inception", csv_str)

        # Reload from exported CSV to verify integrity
        reloaded = pd.read_csv(io.StringIO(csv_str))
        self.assertEqual(len(reloaded), 10)
        self.assertNotIn("__helper_internal_id", reloaded.columns)

        # Export schema profile
        schema = compute_column_schema(self.synthetic_df)
        prof_csv = export_profile_csv(schema)
        self.assertIn("Logical Type", prof_csv)
        self.assertIn("Non-Null Count", prof_csv)

    def test_unusual_column_names_and_unicode(self):
        """Verifies safe handling of columns with spaces, slashes, unicode, and emojis."""
        odd_df = pd.DataFrame({
            "col with spaces": [1, 2, 3],
            "col/with/slash": ["x", "y", "z"],
            "col (parens) & %": [10.5, 20.2, 30.1],
            "🎬 movie_title_unicode_café": ["A", "B", "C"],
        })
        schema = compute_column_schema(odd_df)
        self.assertEqual(len(schema), 4)
        self.assertIn("🎬 movie_title_unicode_café", schema["Column"].values)

        # Filtering with odd names
        filtered = apply_explorer_filters(odd_df, {
            "text_searches": {"col/with/slash": "y"},
            "numeric_ranges": {"col (parens) & %": (15.0, 25.0)}
        })
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered.iloc[0]["col with spaces"], 2)

        # Export odd names
        csv_out = export_filtered_csv(filtered)
        self.assertIn("col/with/slash", csv_out)
        self.assertIn("🎬 movie_title_unicode_café", csv_out)

    def test_empty_dataset_and_all_null_handling(self):
        """Verifies graceful handling of empty DataFrames without exceptions."""
        empty_df = pd.DataFrame()
        metrics = compute_overview_metrics(empty_df)
        self.assertEqual(metrics["total_rows"], 0)
        self.assertEqual(metrics["total_columns"], 0)

        schema = compute_column_schema(empty_df)
        self.assertEqual(len(schema), 0)

        filtered = apply_explorer_filters(empty_df, {"global_search": "test"})
        self.assertEqual(len(filtered), 0)

        page, pages, total = paginate_dataframe(empty_df, page_number=1, page_size=25)
        self.assertEqual(len(page), 0)
        self.assertEqual(total, 0)

    def test_row_level_missingness_filters(self):
        """Verifies complete_only and has_missing row filters."""
        df = self.synthetic_df
        # Complete rows only (no missing in any column)
        # Note: in synthetic_df, rows have budget_mil null, notes whitespace, etc., plus all_null_col is all null!
        # Because all_null_col is all null, 0 rows are 100% complete across ALL columns!
        res_comp = apply_explorer_filters(df, {"has_any_missing": "complete_only"})
        self.assertEqual(len(res_comp), 0)

        res_miss = apply_explorer_filters(df, {"has_any_missing": "has_missing"})
        self.assertEqual(len(res_miss), 10)

        # Without all_null_col
        sub_df = df[["id", "title", "score"]].copy()
        res_sub_comp = apply_explorer_filters(sub_df, {"has_any_missing": "complete_only"})
        self.assertEqual(len(res_sub_comp), 10)

    def test_data_isolation_source_immutability(self):
        """Verifies that exploring a custom CSV does not mutate or affect the Netflix catalog."""
        netflix_path = resolve_builtin_netflix_path()
        original_mtime = os.path.getmtime(netflix_path)

        # Create custom uploaded dataset
        custom_csv = b"custom_id,custom_metric\n101,99.9\n102,88.8\n"
        custom_df, meta = load_csv_from_file_or_bytes(custom_csv, filename="custom.csv")

        # Profile and filter custom data
        metrics = compute_overview_metrics(custom_df)
        filtered = apply_explorer_filters(custom_df, {"numeric_ranges": {"custom_metric": (90.0, 100.0)}})

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered.iloc[0]["custom_id"], 101)

        # Verify Netflix file on disk was NOT touched or overwritten
        new_mtime = os.path.getmtime(netflix_path)
        self.assertEqual(original_mtime, new_mtime)

    def test_dataset_identity_collision_prevention(self):
        """Regression test A2: Two CSVs with identical filename and dimensions but different values produce distinct identities."""
        csv1 = b"id,val\n1,Alpha\n2,Beta\n"
        csv2 = b"id,val\n1,Gamma\n2,Delta\n"
        df1, _ = load_csv_from_file_or_bytes(csv1, filename="dataset.csv")
        df2, _ = load_csv_from_file_or_bytes(csv2, filename="dataset.csv")

        self.assertEqual(len(df1), len(df2))
        self.assertEqual(df1.shape, df2.shape)

        id1 = compute_dataset_identity(df1, source_type="uploaded", file_bytes=csv1, filename="dataset.csv")
        id2 = compute_dataset_identity(df2, source_type="uploaded", file_bytes=csv2, filename="dataset.csv")

        self.assertNotEqual(id1, id2, "Cryptographic dataset identity failed to distinguish different datasets with identical dimensions.")

    def test_missingness_matrix_linspace_sampling_across_all_ranges(self):
        """Regression test A8: Matrix sampling at 501, 750, 999, 1,000, and 10,000 rows spans entire row range."""
        test_sizes = [501, 750, 999, 1000, 10000]
        max_rows = 500

        for n_rows in test_sizes:
            df = pd.DataFrame({
                "a": np.arange(n_rows),
                "b": [np.nan if i % 2 == 0 else 1.0 for i in range(n_rows)],
            })
            sampled_mat, was_sampled, total_recs = get_missingness_matrix(df, max_rows=max_rows)

            self.assertTrue(was_sampled)
            self.assertEqual(total_recs, n_rows)
            self.assertEqual(len(sampled_mat), max_rows)
            # Verify sampled indices span from row 0 to the very last row (n_rows - 1)
            self.assertEqual(sampled_mat.index[0], 0, f"Sampling for {n_rows} rows did not start at index 0.")
            self.assertEqual(sampled_mat.index[-1], n_rows - 1, f"Sampling for {n_rows} rows did not reach final index {n_rows - 1}.")

    def test_missingness_matrix_below_limit(self):
        """Matrix sampling for datasets <= max_rows returns exact rows without sampling."""
        df = pd.DataFrame({"a": np.arange(250), "b": np.arange(250)})
        sampled_mat, was_sampled, total_recs = get_missingness_matrix(df, max_rows=500)
        self.assertFalse(was_sampled)
        self.assertEqual(len(sampled_mat), 250)
        self.assertEqual(total_recs, 250)

    def test_unified_missing_value_semantics_literal_strings(self):
        """Regression test A4: Literal strings 'nan', 'none', 'NULL' are NOT treated as missing."""
        df = pd.DataFrame({
            "code": ["nan", "none", "NULL", "N/A", "Valid1", "Valid2", None, np.nan, "   "]
        })

        # treat_empty_as_missing=False: only None, np.nan are missing (2 of 9)
        mask_false = get_missing_mask(df, treat_empty_as_missing=False)
        self.assertEqual(mask_false["code"].sum(), 2)

        cat_stats_false = compute_categorical_stats(df["code"], treat_empty_as_missing=False)
        self.assertEqual(cat_stats_false["valid_count"], 7)
        self.assertIn("nan", cat_stats_false["freq_table"]["Category"].values)
        self.assertIn("none", cat_stats_false["freq_table"]["Category"].values)

        # treat_empty_as_missing=True: None, np.nan, and "   " are missing (3 of 9)
        mask_true = get_missing_mask(df, treat_empty_as_missing=True)
        self.assertEqual(mask_true["code"].sum(), 3)

        cat_stats_true = compute_categorical_stats(df["code"], treat_empty_as_missing=True)
        self.assertEqual(cat_stats_true["valid_count"], 6)

        # compute_column_schema distinct count must agree
        schema = compute_column_schema(df, treat_empty_as_missing=True)
        code_schema = schema[schema["Column"] == "code"].iloc[0]
        self.assertEqual(code_schema["Missing Count"], 3)
        self.assertEqual(code_schema["Non-Null Count"], 6)

    def test_high_cardinality_categorical_filtering(self):
        """Regression test A6: Categorical column with more than 50 distinct values filters correctly."""
        num_categories = 75
        df = pd.DataFrame({
            "category": [f"Cat_{i:03d}" for i in range(num_categories)],
            "val": np.arange(num_categories),
        })

        # Filter by choosing categories beyond the 50th item
        target_cats = ["Cat_055", "Cat_060", "Cat_070"]
        filtered = apply_explorer_filters(df, {"categorical_selections": {"category": target_cats}})
        self.assertEqual(len(filtered), 3)
        self.assertEqual(set(filtered["category"].tolist()), set(target_cats))

    def test_date_range_filtering_inclusive_bounds(self):
        """Regression test A5: Date filtering respects inclusive boundaries on start and end dates."""
        df = pd.DataFrame({
            "title": ["Movie 1", "Movie 2", "Movie 3", "Movie 4"],
            "release_date": ["2020-01-01", "2020-06-15", "2020-12-31", "2021-01-01"]
        })

        # Inclusive filter from 2020-01-01 to 2020-12-31 should capture Movies 1, 2, and 3
        filtered = apply_explorer_filters(
            df,
            {"date_ranges": {"release_date": ("2020-01-01", "2020-12-31")}}
        )
        self.assertEqual(len(filtered), 3)
        self.assertNotIn("Movie 4", filtered["title"].values)


if __name__ == "__main__":
    unittest.main()
