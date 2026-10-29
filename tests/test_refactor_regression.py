"""
test_refactor_regression.py
Regression test suite for DataLab Codebase Cleanup & UI Redesign.

Verifies:
- Canonical dataset resolution to project root netflix_titles.csv.
- ui_theme design tokens and component renderers.
- All 6 workspace modules and lazy-loaded UI exports in src.__init__.
- data_context working dataset isolation and state consistency.
- Plotly figure theme application.
"""

import os
import unittest
import pandas as pd
import plotly.graph_objects as go

import src
from src.data_generator import ensure_data_exists
from src.data_explorer import resolve_builtin_netflix_path
from src.cleaning_lab import WorkingDatasetManager
from src.data_context import (
    get_source_dataset,
    set_source_dataset,
    get_working_dataset,
    set_working_dataset_manager,
    compute_modeling_dataset_identity,
    is_result_stale,
)
from src.ui_theme import (
    BG_APP,
    BG_CARD,
    COLOR_CYAN,
    COLOR_SUCCESS,
    apply_figure_theme,
    render_page_header,
    render_metric_card,
    render_badge,
    escape_html,
)


class TestRefactorRegression(unittest.TestCase):
    """Test suite covering cleanup and refactored module boundaries."""

    def setUp(self):
        try:
            import streamlit as st
            st.session_state.clear()
        except Exception:
            pass

    def tearDown(self):
        try:
            import streamlit as st
            st.session_state.clear()
        except Exception:
            pass

    def test_canonical_dataset_resolution(self):
        """Verify resolve_builtin_netflix_path points to root netflix_titles.csv."""
        resolved_path = resolve_builtin_netflix_path()
        self.assertIsNotNone(resolved_path)
        self.assertTrue(os.path.exists(resolved_path), f"Resolved path does not exist: {resolved_path}")
        self.assertTrue(
            resolved_path.endswith("netflix_titles.csv"),
            f"Expected netflix_titles.csv, got: {resolved_path}"
        )
        
        # Verify ensure_data_exists returns an existing path
        path = ensure_data_exists()
        self.assertTrue(os.path.exists(path))

    def test_ui_theme_tokens_and_renderers(self):
        """Verify ui_theme tokens and HTML helper functions."""
        self.assertEqual(BG_APP, "#0B0F19")
        self.assertEqual(COLOR_CYAN, "#00D2FF")
        self.assertEqual(COLOR_SUCCESS, "#10B981")

        # Test escape_html
        raw_text = '<script>alert("xss")</script> & "quotes"'
        escaped = escape_html(raw_text)
        self.assertNotIn("<script>", escaped)
        self.assertIn("&lt;script&gt;", escaped)
        self.assertIn("&amp;", escaped)

        # Test render_badge
        badge_html = render_badge("Active", "success")
        self.assertIn("Active", badge_html)
        self.assertIn("badge-pill", badge_html)
        self.assertIn("success", badge_html)

        # Test render_metric_card
        render_metric_card("Total Rows", "8,807", "+12%", accent_color="#10B981")

    def test_plotly_figure_theme_application(self):
        """Verify apply_figure_theme modifies Plotly layout consistently."""
        fig = go.Figure(data=[go.Bar(x=["A", "B", "C"], y=[10, 20, 30])])
        themed_fig = apply_figure_theme(fig, height=350)
        self.assertEqual(themed_fig.layout.paper_bgcolor, "rgba(0,0,0,0)")
        self.assertEqual(themed_fig.layout.plot_bgcolor, "rgba(0,0,0,0)")
        self.assertEqual(themed_fig.layout.height, 350)

    def test_all_workspace_ui_exports(self):
        """Verify all 6 workspace tab renderers are registered in _UI_EXPORTS and __all__."""
        expected_exports = [
            "render_explorer_tab",
            "render_cleaning_lab_tab",
            "render_workflow_studio_tab",
            "render_model_lab_tab",
            "render_missions_tab",
            "render_netflix_case_study_tab",
        ]
        for exp in expected_exports:
            self.assertIn(exp, src.__all__, f"{exp} missing from src.__all__")
            self.assertTrue(hasattr(src, exp), f"{exp} attribute not accessible on src")
            fn = getattr(src, exp)
            self.assertTrue(callable(fn), f"{exp} is not callable")

    def test_data_context_working_dataset_isolation(self):
        """Verify data_context maintains clean working dataset isolation."""
        df_src = pd.DataFrame({"col_a": [1, 2, 3], "col_b": ["x", "y", "z"]})
        set_source_dataset(df_src, "test_ident", "test_source.csv")

        src_res, src_id, src_name = get_source_dataset()
        self.assertEqual(len(src_res), 3)
        self.assertEqual(src_id, "test_ident")
        self.assertEqual(src_name, "test_source.csv")

        # Initially no transformed working dataset
        wk_df, wk_id, wk_name, is_trans, count = get_working_dataset()
        self.assertFalse(is_trans)

        # Register working dataset manager
        mgr = WorkingDatasetManager(df_src, "test_ident", "test_source.csv")
        mgr.working_df = pd.DataFrame({"col_a": [1, 2], "col_b": ["x", "y"]})
        mgr.history = [{"op": "drop_rows", "description": "Dropped row"}]
        set_working_dataset_manager(mgr)

        wk_df2, wk_id2, wk_name2, is_trans2, state_id2 = get_working_dataset()
        self.assertTrue(is_trans2)
        self.assertEqual(len(wk_df2), 2)

        # Source remains untouched
        src_res2, _, _ = get_source_dataset()
        self.assertEqual(len(src_res2), 3)

        # Modeling dataset identity differentiates raw vs transformed
        id_raw = compute_modeling_dataset_identity("test_ident", False, mgr, df_src)
        id_wk = compute_modeling_dataset_identity("test_ident", True, mgr, mgr.working_df)
        self.assertNotEqual(id_raw, id_wk)
        self.assertTrue(is_result_stale(id_raw, id_wk))
        self.assertFalse(is_result_stale(id_wk, id_wk))


if __name__ == "__main__":
    unittest.main()
