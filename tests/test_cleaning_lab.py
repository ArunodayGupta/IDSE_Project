"""
test_cleaning_lab.py
Automated test suite for Phase 2 DataLab: Clean & Transform Lab.

Tests data quality auditing, missing-value imputation, duplicate deduplication,
type transformations, feature engineering recipes, allowlisted operation registry,
What Changed? engine, working copy state management, undo/reset deterministic replay,
source immutability, and export generation.
"""

import unittest
import numpy as np
import pandas as pd
import json

from src.cleaning_lab import (
    WorkingDatasetManager,
    generate_data_quality_audit,
    compute_what_changed,
    apply_cleaning_operation,
    export_working_csv,
    export_cleaning_report,
    NETFLIX_RECIPES,
    op_convert_type,
    op_feature_binning,
    op_feature_category_map,
    op_feature_binary_indicator,
)


class TestDataLabCleaningLab(unittest.TestCase):
    """Unit test suite for Phase 2 DataLab Clean & Transform Lab transformation engine."""

    def setUp(self):
        """Build synthetic test fixtures for data cleaning and feature engineering."""
        self.raw_df = pd.DataFrame({
            "id": [1, 2, 3, 4, 5, 6, 7, 8],
            "title": ["Inception", "Stranger Things", "Breaking Bad", "The Crown", "Dark", "Interstellar", "Dark", "Mindhunter"],
            "type": ["Movie", "TV Show", "TV Show", "TV Show", "TV Show", "Movie", "TV Show", "TV Show"],
            "country": ["United States", "United States", "United States", "United Kingdom", "Germany", "United States", "Germany", None],
            "score": [8.8, 8.7, 9.5, np.nan, 8.7, 8.6, 8.7, np.nan],
            "budget_mil": [160.0, np.nan, 3.5, 13.0, np.nan, 165.0, np.nan, 25.0],
            "revenue_mil": [836.8, np.nan, 10.0, 50.0, np.nan, 701.7, np.nan, 60.0],
            "release_date": ["2010-07-16", "2016-07-15", "2008-01-20", "2016-11-04", "2017-12-01", "2014-11-07", "2017-12-01", "2017-10-13"],
            "text_notes": ["  Masterpiece sci-fi film  ", "Sci-fi thriller series", "Crime drama", "Royal historical", "Mind-bending puzzle", "Space adventure", "Mind-bending puzzle", "   "],
            "category_code": ["A", "B", "B", "C", "A", "A", "A", None],
            "constant_col": [99, 99, 99, 99, 99, 99, 99, 99],
            "all_null_col": [np.nan] * 8,
        })
        self.manager = WorkingDatasetManager(self.raw_df, source_name="Synthetic Catalog", source_identity="synthetic_hash_123")

    def test_source_immutability(self):
        """Verifies that operations on the working copy never mutate the original source DataFrame."""
        orig_copy = self.raw_df.copy()
        # Apply an operation that drops rows and adds columns
        self.manager.apply_step({
            "op_type": "drop_missing_rows",
            "columns": ["score"],
            "description": "Drop missing scores"
        })
        self.manager.apply_step({
            "op_type": "math_transform",
            "column": "budget_mil",
            "transform": "log",
            "output_col": "log_budget",
            "description": "Log budget"
        })

        # Working copy should have fewer rows and more columns
        self.assertLess(len(self.manager.get_working_df()), len(self.raw_df))
        self.assertIn("log_budget", self.manager.get_working_df().columns)

        # Source DataFrame inside manager and orig_copy must be completely unchanged
        pd.testing.assert_frame_equal(self.manager.source_df, orig_copy)
        pd.testing.assert_frame_equal(self.raw_df, orig_copy)

    def test_data_quality_audit(self):
        """Verifies initial data quality audit computation."""
        audit = generate_data_quality_audit(self.raw_df)
        self.assertEqual(audit["row_count"], 8)
        self.assertEqual(audit["column_count"], 12)
        self.assertGreater(audit["total_missing_cells"], 0)
        self.assertIn("constant_col", audit["constant_cols"])
        self.assertIn("all_null_col", audit["all_null_cols"])
        self.assertIn("release_date", audit["date_candidate_cols"])
        self.assertIn("text_notes", audit["whitespace_only_cols"])

    def test_impute_missing_numeric_mean_and_median(self):
        """Verifies mean and median imputation on numeric columns."""
        # Mean imputation
        res_mean = apply_cleaning_operation(self.raw_df, {
            "op_type": "impute_numeric",
            "column": "score",
            "strategy": "mean",
        })
        self.assertTrue(res_mean.success)
        expected_mean = round(float(self.raw_df["score"].dropna().mean()), 2)
        self.assertEqual(res_mean.df["score"].isna().sum(), 0)
        self.assertAlmostEqual(res_mean.df.loc[3, "score"], expected_mean, places=1)

        # Median imputation
        res_median = apply_cleaning_operation(self.raw_df, {
            "op_type": "impute_numeric",
            "column": "score",
            "strategy": "median",
        })
        self.assertTrue(res_median.success)
        expected_median = float(self.raw_df["score"].dropna().median())
        self.assertEqual(res_median.df["score"].isna().sum(), 0)
        self.assertEqual(res_median.df.loc[3, "score"], expected_median)

        # Attempting mean imputation on all-null column returns failure
        res_fail = apply_cleaning_operation(self.raw_df, {
            "op_type": "impute_numeric",
            "column": "all_null_col",
            "strategy": "mean",
        })
        self.assertFalse(res_fail.success)

    def test_impute_categorical_mode_and_constant(self):
        """Verifies mode imputation and constant value filling."""
        # Mode imputation for country
        res_mode = apply_cleaning_operation(self.raw_df, {
            "op_type": "impute_categorical_mode",
            "column": "country",
        })
        self.assertTrue(res_mode.success)
        self.assertEqual(res_mode.df["country"].isna().sum(), 0)
        self.assertEqual(res_mode.df.loc[7, "country"], "United States")

        # Constant filling
        res_const = apply_cleaning_operation(self.raw_df, {
            "op_type": "fill_constant",
            "column": "country",
            "fill_value": "International",
        })
        self.assertTrue(res_const.success)
        self.assertEqual(res_mode.df["country"].isna().sum(), 0)
        self.assertEqual(res_const.df.loc[7, "country"], "International")

    def test_whitespace_to_nan(self):
        """Verifies explicit conversion of blank/whitespace strings to np.nan."""
        res = apply_cleaning_operation(self.raw_df, {
            "op_type": "whitespace_to_nan",
            "columns": ["text_notes"],
        })
        self.assertTrue(res.success)
        # Row 7 text_notes was "   ", now it should be NaN
        self.assertTrue(pd.isna(res.df.loc[7, "text_notes"]))

    def test_drop_missing_rows_and_columns(self):
        """Verifies dropping rows with missing values and dropping selected columns."""
        # Drop rows missing score (2 rows are missing score)
        res_drop_rows = apply_cleaning_operation(self.raw_df, {
            "op_type": "drop_missing_rows",
            "columns": ["score"],
        })
        self.assertTrue(res_drop_rows.success)
        self.assertEqual(len(res_drop_rows.df), 6)

        # Drop columns
        res_drop_cols = apply_cleaning_operation(self.raw_df, {
            "op_type": "drop_columns",
            "columns": ["all_null_col", "constant_col"],
        })
        self.assertTrue(res_drop_cols.success)
        self.assertNotIn("all_null_col", res_drop_cols.df.columns)
        self.assertNotIn("constant_col", res_drop_cols.df.columns)

    def test_duplicate_removal_strategies(self):
        """Verifies deduplication policies (keep first, keep last, drop all duplicates)."""
        # In raw_df, row 4 and row 6 are identical on 'title', 'type', 'country': ("Dark", "TV Show", "Germany")
        # Keep first
        res_first = apply_cleaning_operation(self.raw_df, {
            "op_type": "remove_duplicates",
            "columns": ["title", "type", "country"],
            "keep": "first",
        })
        self.assertTrue(res_first.success)
        self.assertEqual(len(res_first.df), 7)
        self.assertIn(4, res_first.df.index)
        self.assertNotIn(6, res_first.df.index)

        # Keep last
        res_last = apply_cleaning_operation(self.raw_df, {
            "op_type": "remove_duplicates",
            "columns": ["title", "type", "country"],
            "keep": "last",
        })
        self.assertTrue(res_last.success)
        self.assertEqual(len(res_last.df), 7)
        self.assertNotIn(4, res_last.df.index)
        self.assertIn(6, res_last.df.index)

        # Drop all (keep=False)
        res_none = apply_cleaning_operation(self.raw_df, {
            "op_type": "remove_duplicates",
            "columns": ["title", "type", "country"],
            "keep": "none",
        })
        self.assertTrue(res_none.success)
        self.assertEqual(len(res_none.df), 6)
        self.assertNotIn(4, res_none.df.index)
        self.assertNotIn(6, res_none.df.index)

    def test_type_conversion_and_normalization(self):
        """Verifies numeric/datetime parsing and text trimming/casing."""
        # Text normalization
        res_norm = apply_cleaning_operation(self.raw_df, {
            "op_type": "normalize_text",
            "column": "text_notes",
            "trim": True,
            "case": "lower",
        })
        self.assertTrue(res_norm.success)
        self.assertEqual(res_norm.df.loc[0, "text_notes"], "masterpiece sci-fi film")

        # Datetime parse
        res_dt = apply_cleaning_operation(self.raw_df, {
            "op_type": "convert_type",
            "column": "release_date",
            "target_type": "datetime",
        })
        self.assertTrue(res_dt.success)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(res_dt.df["release_date"]))

        # Numeric parse with coercible values
        test_df = pd.DataFrame({"num_str": ["10", "20.5", "invalid", "40"]})
        res_num = apply_cleaning_operation(test_df, {
            "op_type": "convert_type",
            "column": "num_str",
            "target_type": "numeric",
        })
        self.assertTrue(res_num.success)
        self.assertEqual(res_num.df["num_str"].isna().sum(), 1)  # 'invalid' coerced to NaN

    def test_feature_engineering_numeric_arithmetic_and_div_zero(self):
        """Verifies numeric arithmetic combinations and division by zero protection."""
        # Multi-column arithmetic: revenue - budget = profit
        res_sub = apply_cleaning_operation(self.raw_df, {
            "op_type": "numeric_arithmetic",
            "col_a": "revenue_mil",
            "col_b": "budget_mil",
            "operator": "subtract",
            "output_col": "profit_mil",
        })
        self.assertTrue(res_sub.success)
        self.assertAlmostEqual(res_sub.df.loc[0, "profit_mil"], 836.8 - 160.0, places=2)

        # Division with zero protection
        df_div = pd.DataFrame({"rev": [100.0, 50.0, 30.0], "cost": [20.0, 0.0, np.nan]})
        res_div = apply_cleaning_operation(df_div, {
            "op_type": "numeric_arithmetic",
            "col_a": "rev",
            "col_b": "cost",
            "operator": "divide",
            "output_col": "roi",
        })
        self.assertTrue(res_div.success)
        self.assertEqual(res_div.df.loc[0, "roi"], 5.0)
        # Division by zero must yield NaN (not infinity or error)
        self.assertTrue(pd.isna(res_div.df.loc[1, "roi"]))
        self.assertTrue(pd.isna(res_div.df.loc[2, "roi"]))

    def test_feature_engineering_math_transforms_and_bins(self):
        """Verifies log, abs, and binning features."""
        # Math transform: log
        res_log = apply_cleaning_operation(self.raw_df, {
            "op_type": "math_transform",
            "column": "budget_mil",
            "transform": "log",
            "output_col": "log_budget",
        })
        self.assertTrue(res_log.success)
        self.assertAlmostEqual(res_log.df.loc[0, "log_budget"], np.log(160.0), places=2)

        # Numeric binning (quantile)
        res_bins = apply_cleaning_operation(self.raw_df, {
            "op_type": "numeric_binning",
            "column": "score",
            "method": "quantile",
            "bins": 3,
            "output_col": "score_tier",
        })
        self.assertTrue(res_bins.success)
        self.assertIn("score_tier", res_bins.df.columns)

    def test_feature_engineering_datetime_extraction_and_diff(self):
        """Verifies datetime part extraction and date delta computation."""
        # Convert release_date first
        df_with_dt = self.raw_df.copy()
        df_with_dt["dt"] = pd.to_datetime(df_with_dt["release_date"])

        # Extract year
        res_dt_yr = apply_cleaning_operation(df_with_dt, {
            "op_type": "datetime_extract",
            "column": "dt",
            "part": "year",
            "output_col": "release_year",
        })
        self.assertTrue(res_dt_yr.success)
        self.assertEqual(res_dt_yr.df.loc[0, "release_year"], 2010)

        # Date diff in days
        df_with_dt["dt_acquired"] = pd.to_datetime(["2011-01-01"] * 8)
        res_diff = apply_cleaning_operation(df_with_dt, {
            "op_type": "date_difference",
            "date_col_a": "dt_acquired",
            "date_col_b": "dt",
            "unit": "days",
            "output_col": "days_to_acquire",
        })
        self.assertTrue(res_diff.success)
        self.assertIn("days_to_acquire", res_diff.df.columns)

    def test_feature_engineering_categorical_and_text(self):
        """Verifies categorical mapping, one-hot encoding, and text features."""
        # Category mapping
        res_map = apply_cleaning_operation(self.raw_df, {
            "op_type": "categorical_map",
            "column": "type",
            "mapping": {"Movie": "Film", "TV Show": "Series"},
            "default_value": "Other",
            "output_col": "format_clean",
        })
        self.assertTrue(res_map.success)
        self.assertEqual(res_map.df.loc[0, "format_clean"], "Film")
        self.assertEqual(res_map.df.loc[1, "format_clean"], "Series")

        # One-hot encoding
        res_ohe = apply_cleaning_operation(self.raw_df, {
            "op_type": "one_hot_encode",
            "column": "type",
            "prefix": "type",
            "drop_first": False,
        })
        self.assertTrue(res_ohe.success)
        self.assertIn("type_Movie", res_ohe.df.columns)
        self.assertIn("type_TV_Show", res_ohe.df.columns)

        # Text length and word count
        res_txt = apply_cleaning_operation(self.raw_df, {
            "op_type": "text_transform",
            "column": "text_notes",
            "transform": "word_count",
            "output_col": "notes_word_count",
        })
        self.assertTrue(res_txt.success)
        # "Masterpiece sci-fi film" -> 3 words
        self.assertEqual(res_txt.df.loc[0, "notes_word_count"], 3)

    def test_column_name_collision_and_safety(self):
        """Verifies that creating a feature with an existing column name fails without overwrite flag."""
        res_collision = apply_cleaning_operation(self.raw_df, {
            "op_type": "math_transform",
            "column": "score",
            "transform": "log",
            "output_col": "score",  # Collision!
            "overwrite": False,
        })
        self.assertFalse(res_collision.success)
        self.assertIn("already exists", res_collision.error_message)

    def test_no_arbitrary_code_execution_security(self):
        """Verifies that unknown or unallowlisted operations are strictly rejected."""
        res_unsafe = apply_cleaning_operation(self.raw_df, {
            "op_type": "eval_arbitrary_python",
            "code": "__import__('os').system('echo hack')",
        })
        self.assertFalse(res_unsafe.success)
        self.assertIn("Unsupported or unallowlisted", res_unsafe.error_message)

    def test_netflix_specific_recipes_and_schema_guard(self):
        """Verifies Netflix-specific recipe execution and schema guard for missing preconditions."""
        # Test on compatible Netflix fixture
        netflix_mock = pd.DataFrame({
            "title": ["Inception", "Stranger Things", "Family Movie"],
            "release_year": [2010, 2016, 2020],
            "year_added": [2010, 2018, 2020],
            "rating": ["PG-13", "TV-MA", "TV-Y"],
            "listed_in": ["Dramas, International Movies", "Action & Adventure", "Children & Family Movies"],
            "country": ["United States, France", "Japan", None],
            "director": ["Christopher Nolan", None, "Director C"],
            "cast": ["Leonardo DiCaprio", "Actor B", "Actor C"],
            "description": ["A mind-bending heist thriller", "Anime adventure", "Family friendly fun"],
        })

        for recipe_key in NETFLIX_RECIPES.keys():
            res_rec = apply_cleaning_operation(netflix_mock, {
                "op_type": "netflix_recipe",
                "recipe_key": recipe_key,
            })
            self.assertTrue(res_rec.success, f"Failed recipe: {recipe_key}: {res_rec.error_message}")

        # Check acquisition_lag and content_origin_proxy values
        res_lag = apply_cleaning_operation(netflix_mock, {"op_type": "netflix_recipe", "recipe_key": "acquisition_lag"})
        self.assertEqual(res_lag.df.loc[0, "acquisition_lag"], 0)
        self.assertEqual(res_lag.df.loc[1, "acquisition_lag"], 2)

        res_proxy = apply_cleaning_operation(netflix_mock, {"op_type": "netflix_recipe", "recipe_key": "content_origin_proxy"})
        self.assertEqual(res_proxy.df.loc[0, "content_origin_proxy"], "Likely Original / Day-and-Date")
        self.assertEqual(res_proxy.df.loc[1, "content_origin_proxy"], "Recent Release")

        # Test on incompatible generic dataset: must reject safely without crashing
        generic_df = pd.DataFrame({"random_col": [1, 2, 3]})
        res_guard = apply_cleaning_operation(generic_df, {
            "op_type": "netflix_recipe",
            "recipe_key": "acquisition_lag",
        })
        self.assertFalse(res_guard.success)
        self.assertIn("Missing required columns", res_guard.error_message)

    def test_undo_and_reset_deterministic_state(self):
        """Verifies deterministic undo of the last operation and reset to original source."""
        # 1. Apply step 1: Impute missing score
        s1 = self.manager.apply_step({
            "op_type": "impute_numeric",
            "column": "score",
            "strategy": "mean",
            "description": "Impute score with mean"
        })
        self.assertTrue(s1["success"])
        self.assertEqual(len(self.manager.applied_steps), 1)
        self.assertEqual(self.manager.get_working_df()["score"].isna().sum(), 0)

        # 2. Apply step 2: Normalize text
        s2 = self.manager.apply_step({
            "op_type": "normalize_text",
            "column": "text_notes",
            "trim": True,
            "case": "lower",
            "description": "Normalize text"
        })
        self.assertTrue(s2["success"])
        self.assertEqual(len(self.manager.applied_steps), 2)

        # 3. Undo step 2
        undo_ok = self.manager.undo_last_step()
        self.assertTrue(undo_ok)
        self.assertEqual(len(self.manager.applied_steps), 1)
        # text_notes should be back to its untrimmed state
        self.assertEqual(self.manager.get_working_df().loc[0, "text_notes"], "  Masterpiece sci-fi film  ")
        # Step 1 imputation must still be active
        self.assertEqual(self.manager.get_working_df()["score"].isna().sum(), 0)

        # 4. Reset all
        self.manager.reset_to_source()
        self.assertEqual(len(self.manager.applied_steps), 0)
        pd.testing.assert_frame_equal(self.manager.get_working_df(), self.raw_df)

    def test_failed_operation_leaves_working_df_intact(self):
        """Verifies that a failed operation does not corrupt or alter the valid working copy."""
        # Apply valid step
        self.manager.apply_step({
            "op_type": "normalize_text",
            "column": "text_notes",
            "trim": True,
            "description": "Trim notes"
        })
        df_valid_snapshot = self.manager.get_working_df().copy()

        # Attempt invalid operation (dividing on string column)
        fail_res = self.manager.apply_step({
            "op_type": "numeric_arithmetic",
            "col_a": "text_notes",
            "col_b": "score",
            "operator": "divide",
            "output_col": "invalid_res",
            "description": "Invalid arithmetic"
        })
        self.assertFalse(fail_res["success"])

        # Applied steps count should remain 1
        self.assertEqual(len(self.manager.applied_steps), 1)
        # Working DataFrame must be identical to snapshot before failure
        pd.testing.assert_frame_equal(self.manager.get_working_df(), df_valid_snapshot)

    def test_what_changed_engine(self):
        """Verifies that What Changed? correctly calculates row, column, and cell deltas."""
        before_df = self.raw_df.copy()
        after_df = before_df.dropna(subset=["score"]).drop(columns=["all_null_col"])

        delta = compute_what_changed(before_df, after_df, affected_columns=["score", "all_null_col"])
        self.assertEqual(delta["rows_before"], 8)
        self.assertEqual(delta["rows_after"], 6)
        self.assertEqual(delta["row_delta"], -2)
        self.assertEqual(delta["cols_before"], 12)
        self.assertEqual(delta["cols_after"], 11)
        self.assertEqual(delta["col_delta"], -1)
        self.assertIn("all_null_col", delta["cols_removed"])

    def test_export_working_csv_and_cleaning_report(self):
        """Verifies export generation of working DataFrame and markdown/csv cleaning report."""
        self.manager.apply_step({
            "op_type": "impute_numeric",
            "column": "score",
            "strategy": "mean",
            "description": "Mean imputation"
        })
        csv_str = export_working_csv(self.manager.get_working_df())
        self.assertIn("Inception", csv_str)
        self.assertIn("score", csv_str)
        # Must not contain unnamed index column
        self.assertNotIn("Unnamed: 0", csv_str)

        report_md = export_cleaning_report(self.manager)
        self.assertIn("DataLab: Clean & Transform Audit Report", report_md)
        self.assertIn("Mean imputation", report_md)
        self.assertIn("Session Disclosure", report_md)

        # JSON pipeline config serialization
        pipeline_json = self.manager.export_pipeline_json()
        loaded = json.loads(pipeline_json)
        self.assertEqual(len(loaded["steps"]), 1)
        self.assertEqual(loaded["steps"][0]["op_type"], "impute_numeric")

    def test_boolean_conversion_explicit_mappings_and_invalids(self):
        """Regression test A1: Boolean conversion explicitly maps true/false strings, preserves nulls, and rejects invalids."""
        df = pd.DataFrame({
            "bool_raw": ["True", "False", "1", "0", "yes", "no", "y", "n", None, "invalid_str", True, False, 1, 0]
        })
        res_op, details = op_convert_type(df, column="bool_raw", target_type="boolean", errors="coerce")
        s = res_op["bool_raw"]

        # Index 0: "True" -> True
        self.assertEqual(s.iloc[0], True)
        # Index 1: "False" -> False (NOT True from naive astype(bool)!)
        self.assertEqual(s.iloc[1], False)
        # Index 2: "1" -> True
        self.assertEqual(s.iloc[2], True)
        # Index 3: "0" -> False
        self.assertEqual(s.iloc[3], False)
        # Index 4: "yes" -> True
        self.assertEqual(s.iloc[4], True)
        # Index 5: "no" -> False
        self.assertEqual(s.iloc[5], False)
        # Index 6: "y" -> True
        self.assertEqual(s.iloc[6], True)
        # Index 7: "n" -> False
        self.assertEqual(s.iloc[7], False)
        # Index 8: None -> pd.NA (NOT False!)
        self.assertTrue(pd.isna(s.iloc[8]))
        # Index 9: "invalid_str" -> pd.NA (rejected)
        self.assertTrue(pd.isna(s.iloc[9]))
        self.assertGreater(details["invalid_rejected"], 0)
        self.assertEqual(details["missing_preserved"], 2)

    def test_string_conversion_preserves_missing_values(self):
        """Regression test A1: String conversion does not turn actual missing values into 'nan' or 'None'."""
        df = pd.DataFrame({
            "col_mixed": ["Hello", None, np.nan, "World"]
        })
        res_op, details = op_convert_type(df, column="col_mixed", target_type="string")
        s = res_op["col_mixed"]

        self.assertEqual(s.iloc[0], "Hello")
        self.assertTrue(pd.isna(s.iloc[1]))
        self.assertTrue(pd.isna(s.iloc[2]))
        self.assertEqual(str(s.iloc[1]), "<NA>")
        self.assertEqual(str(s.iloc[2]), "<NA>")
        self.assertEqual(s.iloc[3], "World")

    def test_constant_fill_with_falsy_values_and_categorical(self):
        """Regression test A2: Constant-fill supports falsy values (0, False, '') and updates categorical categories."""
        # 1. Fill with 0
        df_zero = pd.DataFrame({"score": [10.0, None, 20.0]})
        mgr_zero = WorkingDatasetManager(df_zero)
        mgr_zero.apply_step({
            "op_type": "fill_constant",
            "column": "score",
            "fill_value": 0,
        })
        self.assertEqual(mgr_zero.get_working_df().loc[1, "score"], 0)

        # 2. Fill with False
        df_bool = pd.DataFrame({"flag": [True, None, True]})
        mgr_bool = WorkingDatasetManager(df_bool)
        mgr_bool.apply_step({
            "op_type": "fill_constant",
            "column": "flag",
            "fill_value": False,
        })
        self.assertEqual(mgr_bool.get_working_df().loc[1, "flag"], False)

        # 3. Fill with empty string ""
        df_str = pd.DataFrame({"txt": ["A", None, "B"]})
        mgr_str = WorkingDatasetManager(df_str)
        mgr_str.apply_step({
            "op_type": "fill_constant",
            "column": "txt",
            "fill_value": "",
        })
        self.assertEqual(mgr_str.get_working_df().loc[1, "txt"], "")

        # 4. Fill categorical column with new category
        df_cat = pd.DataFrame({"genre": pd.Categorical(["Action", None, "Drama"])})
        mgr_cat = WorkingDatasetManager(df_cat)
        mgr_cat.apply_step({
            "op_type": "fill_constant",
            "column": "genre",
            "fill_value": "Documentary",
        })
        self.assertEqual(mgr_cat.get_working_df().loc[1, "genre"], "Documentary")
        self.assertIn("Documentary", mgr_cat.get_working_df()["genre"].cat.categories)

    def test_feature_engineering_preserves_missing_values(self):
        """Regression test A3: Binning, category mapping, and binary indicators preserve missing inputs."""
        df = pd.DataFrame({
            "num_val": [10.0, 20.0, np.nan, 40.0],
            "cat_val": ["apple", None, "banana", "orange"],
        })

        # Binning
        res_bin, det_bin = op_feature_binning(df, column="num_val", new_col_name="num_bin")
        self.assertTrue(pd.isna(res_bin.loc[2, "num_bin"]))
        self.assertNotEqual(res_bin.loc[2, "num_bin"], "nan")

        # Category mapping
        mapping = {"apple": "Fruit_A", "banana": "Fruit_B"}
        res_map, det_map = op_feature_category_map(df, column="cat_val", mapping=mapping, default_val="Other", new_col_name="cat_map")
        self.assertEqual(res_map.loc[0, "cat_map"], "Fruit_A")
        # Missing input (row 1) must remain NaN, NOT become 'Other' or 'nan'
        self.assertTrue(pd.isna(res_map.loc[1, "cat_map"]))
        # Unmapped known input (row 3: 'orange') gets 'Other'
        self.assertEqual(res_map.loc[3, "cat_map"], "Other")

        # Binary indicator
        res_bin_ind, _ = op_feature_binary_indicator(df, column="cat_val", target_value="apple", new_col_name="is_apple", preserve_missing=True)
        self.assertEqual(res_bin_ind.loc[0, "is_apple"], 1.0)
        # Missing input must remain NaN, NOT silently become 0.0
        self.assertTrue(pd.isna(res_bin_ind.loc[1, "is_apple"]))
        self.assertEqual(res_bin_ind.loc[2, "is_apple"], 0.0)

    def test_json_pipeline_import_validation_and_transactional_replay(self):
        """Regression test A6: JSON pipeline import validates schema, replays deterministically, and rolls back on error."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "score": [10.0, np.nan, 30.0],
            "note": ["  hello ", " world  ", "test"]
        })
        mgr = WorkingDatasetManager(df, source_name="test_source")

        # 1. Valid plan JSON
        valid_plan = json.dumps({
            "pipeline_steps": [
                {"op_type": "impute_numeric", "params": {"column": "score", "strategy": "mean"}},
                {"op_type": "normalize_text", "params": {"column": "note", "strip_whitespace": True, "case_style": "upper"}}
            ]
        })

        # Dry run validation
        val_res = mgr.import_pipeline_json(valid_plan, validate_only=True)
        self.assertTrue(val_res["success"])
        self.assertEqual(val_res["step_count"], 2)

        # Full transactional execution
        apply_res = mgr.import_pipeline_json(valid_plan, validate_only=False)
        self.assertTrue(apply_res["success"])
        self.assertEqual(len(mgr.applied_steps), 2)
        self.assertEqual(mgr.get_working_df().loc[1, "score"], 20.0)
        self.assertEqual(mgr.get_working_df().loc[0, "note"], "HELLO")

        # 2. Invalid plan JSON (unsupported operation or failing column)
        invalid_plan = json.dumps({
            "pipeline_steps": [
                {"op_type": "impute_numeric", "params": {"column": "score", "strategy": "mean"}},
                {"op_type": "unallowlisted_evil_op", "params": {}}
            ]
        })
        snap_df = mgr.get_working_df().copy()
        fail_res = mgr.import_pipeline_json(invalid_plan, validate_only=False)
        self.assertFalse(fail_res["success"])
        # Working copy must remain unchanged
        pd.testing.assert_frame_equal(mgr.get_working_df(), snap_df)

    def test_what_changed_retained_cells_and_removed_rows(self):
        """Regression test A5: What Changed identifies cell modifications and representative removed records."""
        df_before = pd.DataFrame({
            "id": [1, 2, 3, 4],
            "title": ["A", "B", "C", "D"],
            "score": [10.0, np.nan, 30.0, 40.0]
        })

        # Drop row 3 and fill NaN in row 1
        df_after = df_before.iloc[:3].copy()
        df_after.loc[1, "score"] = 20.0

        diff = compute_what_changed(df_before, df_after, affected_columns=["score"])
        self.assertEqual(diff["rows_removed_count"], 1)
        self.assertEqual(len(diff["representative_removed_records"]), 1)
        self.assertEqual(diff["representative_removed_records"][0]["id"], 4)
        self.assertEqual(diff["changed_cells_count"], 1)
        self.assertEqual(len(diff["representative_changed_cells"]), 1)
        self.assertEqual(diff["representative_changed_cells"][0]["column"], "score")


if __name__ == "__main__":
    unittest.main()
