"""
cleaning_lab_ui.py
Streamlit UI rendering module for Phase 2 DataLab: Clean & Transform Lab.

Interactive workspace for:
- Initial Data Quality Audit
- Configurable Missing-Value Imputation & Removal
- Duplicate Management & Deduplication Policies
- Type Conversion & Text Normalization
- Feature Engineering Playground (Arithmetic, Math, Binning, Dates, Categorical, Text, Netflix Recipes)
- The 'What Changed?' Before-and-After Comparison Engine
- Transformation Audit Ledger, Step-by-Step History, and Undo / Reset
- Transformed Data and Cleaning Report Export Hub
"""

from typing import Dict, Any, List, Optional
import streamlit as st
import pandas as pd
import numpy as np

from src.data_explorer import (
    classify_column_type,
    compute_overview_metrics,
    format_bytes,
    compute_dataset_identity,
)
from src.cleaning_lab import (
    generate_data_quality_audit,
    WorkingDatasetManager,
)

PRIMARY_COLOR = "#2C7BE5"
ACCENT_COLOR = "#6EA8FE"
SUCCESS_COLOR = "#00D97E"
DANGER_COLOR = "#E63757"
TEXT_COLOR = "#2B3035"


def _render_what_changed_card(comparison: Dict[str, Any]) -> None:
    """Renders an elevated visual summary of the What Changed comparison."""
    step_info = comparison["step_info"]
    op_name = step_info.get("operation", "Transformation")
    status = comparison.get("status", "completed").upper()
    is_preview = status == "PREVIEW"

    badge_bg = "#FFF8E6" if is_preview else "#E6FBF2"
    badge_color = "#D97706" if is_preview else "#00D97E"
    badge_border = "#FCD34D" if is_preview else "#A6F4D0"
    border_left = "#D97706" if is_preview else "#00D97E"

    st.markdown(f"""
        <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; border-left: 5px solid {border_left}; border-radius: 8px; padding: 18px 20px; margin: 16px 0;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                <div style="font-size: 1.15rem; font-weight: 700; color: #2B3035;">
                    🔍 What Changed? — {op_name}
                </div>
                <span style="font-size: 0.8rem; font-weight: 600; background: {badge_bg}; color: {badge_color}; padding: 4px 10px; border-radius: 4px; border: 1px solid {badge_border};">
                    {status}
                </span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric(
            "Rows Delta",
            f"{comparison['rows_after']:,}",
            delta=f"{comparison['rows_delta']:,} rows" if comparison['rows_delta'] != 0 else "0 change"
        )
    with c2:
        st.metric(
            "Columns Delta",
            f"{comparison['columns_after_count']:,}",
            delta=f"{len(comparison['columns_added']) - len(comparison['columns_removed']):,} cols" if (len(comparison['columns_added']) != len(comparison['columns_removed'])) else "0 change"
        )
    with c3:
        st.metric(
            "Missing Cells Delta",
            f"{comparison['missing_cells_after']:,}",
            delta=f"{comparison['missing_cells_delta']:,} cells",
            delta_color="inverse"
        )
    with c4:
        st.metric(
            "Duplicates Delta",
            f"{comparison['duplicates_after']:,}",
            delta=f"{comparison['duplicates_delta']:,} rows",
            delta_color="inverse"
        )
    with c5:
        st.metric(
            "Changed Cells",
            f"{comparison.get('changed_cells_count', 0):,}",
            delta=f"{comparison.get('changed_cells_count', 0):,} cells modified" if comparison.get('changed_cells_count', 0) > 0 else "0 modified",
            delta_color="off"
        )

    # Detailed differences
    if comparison["columns_added"]:
        st.info(f"✨ **New Columns Created:** `{', '.join(comparison['columns_added'])}`")
    if comparison["columns_removed"]:
        st.warning(f"🗑️ **Columns Dropped:** `{', '.join(comparison['columns_removed'])}`")
    if comparison["type_changes"]:
        type_str = ", ".join(f"`{col}` ({info['before']} ➔ {info['after']})" for col, info in comparison["type_changes"].items())
        st.info(f"🔄 **Data Types Converted:** {type_str}")

    # Representative Changed Cells in Retained Rows
    rep_changes = comparison.get("representative_changed_cells", [])
    if rep_changes:
        with st.expander(f"🔬 Inspect Modified Cells ({comparison.get('changed_cells_count', 0):,} cells modified across retained rows)", expanded=False):
            st.dataframe(pd.DataFrame(rep_changes), use_container_width=True, hide_index=True)

    # Representative Removed Records
    rep_removed = comparison.get("representative_removed_records", [])
    if rep_removed:
        with st.expander(f"🗑️ Inspect Representative Removed Records ({comparison.get('rows_removed_count', 0):,} rows dropped)", expanded=False):
            st.dataframe(pd.DataFrame(rep_removed), use_container_width=True)

    # Column Stats Comparison if available
    if comparison.get("stats_comparison"):
        st.markdown("**Descriptive Statistics Delta:**")
        stats_rows = []
        for col, s in comparison["stats_comparison"].items():
            bef = s.get("before", {})
            aft = s.get("after", {})
            stats_rows.append({
                "Feature": col,
                "Mean (Before)": bef.get("mean"),
                "Mean (After)": aft.get("mean"),
                "Median (Before)": bef.get("median"),
                "Median (After)": aft.get("median"),
                "Missing (Before)": bef.get("missing"),
                "Missing (After)": aft.get("missing"),
            })
        st.dataframe(pd.DataFrame(stats_rows), use_container_width=True, hide_index=True)


def render_cleaning_lab_tab(netflix_raw_df: Optional[pd.DataFrame] = None) -> None:
    """
    Main entry point for the Clean & Transform Lab Streamlit tab.
    """
    st.markdown("""
        <div style="margin-bottom: 8px;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                <div>
                    <h2 style="font-weight: 700; font-size: 1.85rem; color: #2B3035; margin-bottom: 2px;">
                        🧹 DataLab — Clean & Transform Lab
                    </h2>
                    <p style="color: #6C757D; font-size: 0.95rem; margin-top: 0px;">
                        Interactive data cleaning, feature engineering, and transformation workbench.
                        Preview operations, inspect real consequences with the <b>What Changed?</b> engine, and export reproducible workflows.
                    </p>
                </div>
                <span style="display: inline-block; padding: 4px 10px; border-radius: 4px; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; background-color: #E7F1FF; color: #2C7BE5; border: 1px solid #B6D4FE;">
                    PHASE 2: CLEAN & TRANSFORM
                </span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # Retrieve or Initialize Shared Dataset State
    # ------------------------------------------------------------------
    shared_source_df = st.session_state.get("datalab_source_data")
    source_identity = st.session_state.get("datalab_source_identity", "")
    source_name = st.session_state.get("datalab_source_name", "Built-in Netflix Catalog")

    if shared_source_df is None or shared_source_df.empty:
        # Fallback to netflix_raw_df if available
        if netflix_raw_df is not None and not netflix_raw_df.empty:
            shared_source_df = netflix_raw_df.copy()
            source_name = "Built-in Netflix Catalog"
            source_identity = compute_dataset_identity(shared_source_df, source_name=source_name)
            st.session_state["datalab_source_data"] = shared_source_df
            st.session_state["datalab_source_identity"] = source_identity
            st.session_state["datalab_source_name"] = source_name
        else:
            st.info("👆 Please select or upload a dataset in the **🔬 DataLab: Explore Dataset** tab to begin cleaning and transformation.")
            return

    # Initialize or reset WorkingDatasetManager if source changed
    current_manager: Optional[WorkingDatasetManager] = st.session_state.get("datalab_working_manager")
    if current_manager is None or current_manager.source_identity != source_identity:
        current_manager = WorkingDatasetManager(
            source_df=shared_source_df,
            source_identity=source_identity,
            source_name=source_name
        )
        st.session_state["datalab_working_manager"] = current_manager

    # Workspace Header Banner
    step_count = len(current_manager.history)
    b_col1, b_col2, b_col3 = st.columns([2, 1, 1])
    with b_col1:
        st.markdown(f"📁 **Active Source Dataset:** `{source_name}` &nbsp;|&nbsp; `{len(current_manager.source_df):,} rows × {current_manager.source_df.shape[1]} cols`")
    with b_col2:
        st.markdown(f"⚙️ **Working Copy:** `{len(current_manager.working_df):,} rows × {current_manager.working_df.shape[1]} cols`")
    with b_col3:
        status_color = "#00D97E" if step_count > 0 else "#6C757D"
        st.markdown(f"📜 **Transformations Applied:** <b style='color: {status_color};'>{step_count} step{'s' if step_count != 1 else ''}</b>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # Clean & Transform Subtabs
    # ------------------------------------------------------------------
    cl_tab1, cl_tab2, cl_tab3, cl_tab4, cl_tab5, cl_tab6 = st.tabs([
        "📊 1. Data Quality Audit",
        "🧹 2. Cleaning Workspace",
        "⚡ 3. Feature Engineering Playground",
        "🔍 4. What Changed? Comparison",
        "📜 5. Audit Ledger & Undo",
        "💾 6. Export Hub"
    ])

    # ==================================================================
    # SUBTAB 1: DATA QUALITY AUDIT
    # ==================================================================
    with cl_tab1:
        st.markdown("#### Automated Data Quality Audit")
        st.caption("Inspect missingness, duplicates, constant columns, and format anomalies across the active raw dataset before deciding on transformations.")

        audit_data = generate_data_quality_audit(current_manager.source_df)

        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Total Rows", f"{audit_data['total_rows']:,}")
        a2.metric("Total Columns", f"{audit_data['total_cols']:,}")
        a3.metric("Missing Cells", f"{audit_data['total_missing_cells']:,} ({audit_data['missing_cells_pct']}%)")
        a4.metric("Duplicate Rows", f"{audit_data['duplicate_rows']:,} ({audit_data['duplicate_rows_pct']}%)")

        if audit_data["warnings"]:
            st.markdown("##### ⚠️ Audit Findings & Structural Warnings")
            for w in audit_data["warnings"]:
                st.warning(w)

        col_aud1, col_aud2 = st.columns(2)
        with col_aud1:
            st.markdown("##### Missing Values by Column")
            if audit_data["missing_by_col"]:
                m_table = pd.DataFrame([
                    {"Column": c, "Missing Count": v["count"], "Missing %": f"{v['pct']}%"}
                    for c, v in audit_data["missing_by_col"].items()
                ])
                st.dataframe(m_table, use_container_width=True, hide_index=True)
            else:
                st.success("✅ No missing values detected in any column.")

        with col_aud2:
            st.markdown("##### Special Column Detections")
            st.write(f"**Constant Columns:** {', '.join(audit_data['constant_columns']) if audit_data['constant_columns'] else 'None'}")
            st.write(f"**Whitespace-Only Columns:** {', '.join(audit_data['whitespace_columns'].keys()) if audit_data['whitespace_columns'] else 'None'}")
            st.write(f"**Potential Date Candidates:** {', '.join(audit_data['date_candidate_columns']) if audit_data['date_candidate_columns'] else 'None'}")

    # ==================================================================
    # SUBTAB 2: CLEANING WORKSPACE
    # ==================================================================
    with cl_tab2:
        st.markdown("#### Interactive Cleaning Operations Workspace")
        st.caption("Select and configure data cleaning transformations. Preview outcomes before committing changes to the working copy.")

        clean_category = st.radio(
            "Select Cleaning Category:",
            options=[
                "Missing-Value Handling",
                "Duplicate Row Management",
                "Type Conversion & Text Normalization",
                "Drop Columns"
            ],
            horizontal=True
        )

        st.markdown("---")

        working_df = current_manager.working_df

        # ---------------- 1. Missing-Value Handling ----------------
        if clean_category == "Missing-Value Handling":
            st.markdown("##### Missing-Value Strategies")
            strategy_choice = st.selectbox(
                "Choose Missing-Value Strategy:",
                options=[
                    "Drop rows missing values in selected columns",
                    "Impute numeric column with Mean",
                    "Impute numeric column with Median",
                    "Impute categorical/text column with Mode (Most Frequent)",
                    "Fill missing values with a Constant value",
                    "Replace empty / whitespace-only strings with NaN"
                ]
            )

            if strategy_choice == "Drop rows missing values in selected columns":
                cols_to_check = st.multiselect(
                    "Select columns to evaluate for missingness (or leave empty to check all columns):",
                    options=list(working_df.columns),
                    default=[]
                )
                drop_how = st.radio("Drop condition:", options=["any (drop if ANY selected column is missing)", "all (drop only if ALL selected columns are missing)"], horizontal=True)
                how_val = "all" if "all" in drop_how else "any"

                if st.button("Preview Drop Rows", key="btn_prev_drop_rows"):
                    prev = current_manager.preview_operation("drop_missing_rows", {"columns": cols_to_check, "how": how_val})
                    _render_what_changed_card(prev)

                if st.button("Apply Drop Rows", key="btn_apply_drop_rows", type="primary"):
                    res = current_manager.apply_operation("drop_missing_rows", {"columns": cols_to_check, "how": how_val})
                    st.success(f"Applied successfully! Removed {abs(res['rows_delta']):,} rows.")
                    st.rerun()

            elif strategy_choice in ("Impute numeric column with Mean", "Impute numeric column with Median"):
                strat = "mean" if "Mean" in strategy_choice else "median"
                num_cols = [c for c in working_df.columns if classify_column_type(working_df[c]) == "Numeric"]
                if not num_cols:
                    st.warning("No numeric columns available for mean/median imputation.")
                else:
                    selected_col = st.selectbox("Select numeric column:", options=num_cols)
                    if st.button("Preview Imputation", key=f"btn_prev_imp_{strat}"):
                        prev = current_manager.preview_operation("impute_missing", {"column": selected_col, "strategy": strat})
                        _render_what_changed_card(prev)
                    if st.button("Apply Imputation", key=f"btn_apply_imp_{strat}", type="primary"):
                        res = current_manager.apply_operation("impute_missing", {"column": selected_col, "strategy": strat})
                        st.success(f"Imputed missing values in '{selected_col}' using {strat} ({res['step_info'].get('fill_value')}).")
                        st.rerun()

            elif strategy_choice == "Impute categorical/text column with Mode (Most Frequent)":
                cat_cols = [c for c in working_df.columns if classify_column_type(working_df[c]) in ("Categorical/Text", "Boolean")]
                if not cat_cols:
                    st.warning("No categorical or text columns available.")
                else:
                    selected_col = st.selectbox("Select categorical column:", options=cat_cols)
                    if st.button("Preview Mode Imputation", key="btn_prev_mode"):
                        prev = current_manager.preview_operation("impute_missing", {"column": selected_col, "strategy": "mode"})
                        _render_what_changed_card(prev)
                    if st.button("Apply Mode Imputation", key="btn_apply_mode", type="primary"):
                        res = current_manager.apply_operation("impute_missing", {"column": selected_col, "strategy": "mode"})
                        st.success(f"Imputed missing values in '{selected_col}' using mode ('{res['step_info'].get('fill_value')}').")
                        st.rerun()

            elif strategy_choice == "Fill missing values with a Constant value":
                col_to_fill = st.selectbox("Select column to fill:", options=list(working_df.columns))
                constant_val = st.text_input("Enter constant fill value:", value="Unknown")
                if st.button("Preview Constant Fill", key="btn_prev_const"):
                    prev = current_manager.preview_operation("impute_missing", {"column": col_to_fill, "strategy": "constant", "constant_value": constant_val})
                    _render_what_changed_card(prev)
                if st.button("Apply Constant Fill", key="btn_apply_const", type="primary"):
                    res = current_manager.apply_operation("impute_missing", {"column": col_to_fill, "strategy": "constant", "constant_value": constant_val})
                    st.success(f"Filled missing values in '{col_to_fill}' with constant '{constant_val}'.")
                    st.rerun()

            elif strategy_choice == "Replace empty / whitespace-only strings with NaN":
                text_cols = [c for c in working_df.columns if classify_column_type(working_df[c]) == "Categorical/Text"]
                target_ws_cols = st.multiselect("Select text columns to clean (leave empty for all text columns):", options=text_cols, default=[])
                if st.button("Preview Whitespace Cleaning", key="btn_prev_ws"):
                    prev = current_manager.preview_operation("replace_whitespace_with_nan", {"columns": target_ws_cols})
                    _render_what_changed_card(prev)
                if st.button("Apply Whitespace Cleaning", key="btn_apply_ws", type="primary"):
                    res = current_manager.apply_operation("replace_whitespace_with_nan", {"columns": target_ws_cols})
                    st.success(f"Replaced {res['step_info'].get('cells_replaced', 0):,} whitespace cells with proper NaN.")
                    st.rerun()

        # ---------------- 2. Duplicate Management ----------------
        elif clean_category == "Duplicate Row Management":
            st.markdown("##### Duplicate Row Management")
            subset_dup_cols = st.multiselect(
                "Evaluate duplicates across specific columns (leave empty for all columns):",
                options=list(working_df.columns),
                default=[]
            )
            keep_rule = st.selectbox(
                "Keep policy:",
                options=[
                    "Keep first occurrence (drop subsequent duplicates)",
                    "Keep last occurrence (drop preceding duplicates)",
                    "Drop all duplicate occurrences (keep neither)"
                ]
            )
            keep_val = "first" if "first" in keep_rule else ("last" if "last" in keep_rule else False)

            if st.button("Preview Deduplication", key="btn_prev_dedup"):
                prev = current_manager.preview_operation("remove_duplicates", {"subset_columns": subset_dup_cols, "keep": keep_val})
                _render_what_changed_card(prev)
            if st.button("Apply Deduplication", key="btn_apply_dedup", type="primary"):
                res = current_manager.apply_operation("remove_duplicates", {"subset_columns": subset_dup_cols, "keep": keep_val})
                st.success(f"Deduplicated! Removed {abs(res['rows_delta']):,} duplicate rows.")
                st.rerun()

        # ---------------- 3. Type Conversion & Text Normalization ----------------
        elif clean_category == "Type Conversion & Text Normalization":
            action_sub = st.radio("Choose action:", options=["Type Conversion", "Text Normalization"], horizontal=True)

            if action_sub == "Type Conversion":
                col_conv = st.selectbox("Select column to convert:", options=list(working_df.columns))
                tgt_type = st.selectbox("Target data type:", options=["numeric", "datetime", "categorical", "string", "boolean"])
                dt_fmt = None
                if tgt_type == "datetime":
                    dt_fmt = st.text_input("Optional datetime format string (e.g. %Y-%m-%d or leave blank for automatic mixed parsing):", value="")
                    if not dt_fmt.strip():
                        dt_fmt = None

                if st.button("Preview Conversion", key="btn_prev_conv"):
                    prev = current_manager.preview_operation("convert_type", {"column": col_conv, "target_type": tgt_type, "datetime_format": dt_fmt})
                    _render_what_changed_card(prev)
                if st.button("Apply Conversion", key="btn_apply_conv", type="primary"):
                    res = current_manager.apply_operation("convert_type", {"column": col_conv, "target_type": tgt_type, "datetime_format": dt_fmt})
                    st.success(f"Converted '{col_conv}' to {tgt_type}.")
                    st.rerun()

            else:  # Text Normalization
                text_cols = [c for c in working_df.columns if classify_column_type(working_df[c]) in ("Categorical/Text", "Other")]
                if not text_cols:
                    st.warning("No text columns found.")
                else:
                    col_norm = st.selectbox("Select text column to normalize:", options=text_cols)
                    do_strip = st.checkbox("Strip leading and trailing whitespace", value=True)
                    case_opt = st.selectbox("Case normalization style:", options=["lower", "upper", "title", "None (preserve existing case)"])
                    case_val = None if "None" in case_opt else case_opt

                    if st.button("Preview Normalization", key="btn_prev_norm"):
                        prev = current_manager.preview_operation("normalize_text", {"column": col_norm, "strip_whitespace": do_strip, "case_style": case_val})
                        _render_what_changed_card(prev)
                    if st.button("Apply Normalization", key="btn_apply_norm", type="primary"):
                        current_manager.apply_operation("normalize_text", {"column": col_norm, "strip_whitespace": do_strip, "case_style": case_val})
                        st.success(f"Normalized text for '{col_norm}'.")
                        st.rerun()

        # ---------------- 4. Drop Columns ----------------
        elif clean_category == "Drop Columns":
            st.markdown("##### Drop Columns")
            cols_to_drop = st.multiselect("Select columns to permanently drop from working copy:", options=list(working_df.columns))
            if cols_to_drop:
                if st.button("Preview Column Drop", key="btn_prev_drop_col"):
                    prev = current_manager.preview_operation("drop_columns", {"columns": cols_to_drop})
                    _render_what_changed_card(prev)
                if st.button("Apply Column Drop", key="btn_apply_drop_col", type="primary"):
                    current_manager.apply_operation("drop_columns", {"columns": cols_to_drop})
                    st.success(f"Dropped columns: {', '.join(cols_to_drop)}")
                    st.rerun()

    # ==================================================================
    # SUBTAB 3: FEATURE ENGINEERING PLAYGROUND
    # ==================================================================
    with cl_tab3:
        st.markdown("#### Feature Engineering Playground")
        st.caption("Construct new features using guided templates or existing Netflix strategic recipes. All transformations use safe allowlisted implementations.")

        fe_category = st.radio(
            "Select Feature Engineering Category:",
            options=[
                "Numeric Transformations & Math",
                "Datetime Feature Extraction",
                "Categorical Encoding & Mapping",
                "Text Feature Extraction",
                "Netflix Strategy Recipes"
            ],
            horizontal=True
        )

        st.markdown("---")
        working_df = current_manager.working_df

        # ---------------- Numeric Transformations ----------------
        if fe_category == "Numeric Transformations & Math":
            fe_num_mode = st.radio("Numeric mode:", options=["Column Arithmetic (+, -, *, /)", "Mathematical Function (abs, log, log1p)", "Numeric Binning (Quantiles / Thresholds)"], horizontal=True)

            num_cols = [c for c in working_df.columns if classify_column_type(working_df[c]) == "Numeric"]
            if not num_cols:
                st.warning("No numeric columns available in working copy.")
            else:
                if "Arithmetic" in fe_num_mode:
                    c_a1, c_a2, c_a3, c_a4 = st.columns(4)
                    with c_a1:
                        col_a = st.selectbox("Column A:", options=num_cols, key="fe_num_col_a")
                    with c_a2:
                        arith_op = st.selectbox("Operation:", options=["add (+)", "subtract (-)", "multiply (*)", "divide (/)"], key="fe_arith_op")
                        op_symbol = "add" if "+" in arith_op else ("subtract" if "-" in arith_op else ("multiply" if "*" in arith_op else "divide"))
                    with c_a3:
                        operand_choice = st.selectbox("Column B or Scalar:", options=["Column"] + ["Custom Scalar"], key="fe_operand_choice")
                        if operand_choice == "Column":
                            col_b = st.selectbox("Column B:", options=num_cols, key="fe_num_col_b")
                            b_val = col_b
                        else:
                            b_val = st.number_input("Scalar Value:", value=1.0, key="fe_scalar_val")
                    with c_a4:
                        new_name = st.text_input("New Feature Name:", value=f"{col_a}_{op_symbol}_{b_val}", key="fe_arith_name")

                    if st.button("Preview Arithmetic Feature", key="btn_prev_arith"):
                        prev = current_manager.preview_operation("feature_arithmetic", {"col_a": col_a, "col_b_or_scalar": b_val, "operation": op_symbol, "new_col_name": new_name})
                        _render_what_changed_card(prev)
                    if st.button("Apply Arithmetic Feature", key="btn_apply_arith", type="primary"):
                        current_manager.apply_operation("feature_arithmetic", {"col_a": col_a, "col_b_or_scalar": b_val, "operation": op_symbol, "new_col_name": new_name})
                        st.success(f"Created feature '{new_name}'.")
                        st.rerun()

                elif "Mathematical" in fe_num_mode:
                    c_m1, c_m2, c_m3 = st.columns(3)
                    with c_m1:
                        target_col = st.selectbox("Target Numeric Column:", options=num_cols, key="fe_math_col")
                    with c_m2:
                        func_choice = st.selectbox("Math Function:", options=["log (Natural Logarithm)", "log1p (log(1 + x))", "abs (Absolute Value)"], key="fe_math_func")
                        func_code = "log" if "log (" in func_choice else ("log1p" if "log1p" in func_choice else "abs")
                    with c_m3:
                        new_name = st.text_input("New Feature Name:", value=f"{target_col}_{func_code}", key="fe_math_name")

                    if st.button("Preview Math Feature", key="btn_prev_math"):
                        prev = current_manager.preview_operation("feature_math", {"column": target_col, "math_func": func_code, "new_col_name": new_name})
                        _render_what_changed_card(prev)
                    if st.button("Apply Math Feature", key="btn_apply_math", type="primary"):
                        current_manager.apply_operation("feature_math", {"column": target_col, "math_func": func_code, "new_col_name": new_name})
                        st.success(f"Created feature '{new_name}'.")
                        st.rerun()

                else:  # Binning
                    c_b1, c_b2, c_b3 = st.columns(3)
                    with c_b1:
                        target_col = st.selectbox("Column to Bin:", options=num_cols, key="fe_bin_col")
                    with c_b2:
                        bin_type = st.selectbox("Binning Strategy:", options=["quantile (Equal Count / Quartiles)", "thresholds (Equal Width Intervals)"])
                        b_type_val = "quantile" if "quantile" in bin_type else "thresholds"
                        num_bins = st.slider("Number of Bins:", min_value=2, max_value=10, value=4)
                    with c_b3:
                        new_name = st.text_input("New Feature Name:", value=f"{target_col}_binned_{num_bins}", key="fe_bin_name")

                    if st.button("Preview Binning Feature", key="btn_prev_bin"):
                        prev = current_manager.preview_operation("feature_binning", {"column": target_col, "bin_type": b_type_val, "bins": num_bins, "new_col_name": new_name})
                        _render_what_changed_card(prev)
                    if st.button("Apply Binning Feature", key="btn_apply_bin", type="primary"):
                        current_manager.apply_operation("feature_binning", {"column": target_col, "bin_type": b_type_val, "bins": num_bins, "new_col_name": new_name})
                        st.success(f"Created binned feature '{new_name}'.")
                        st.rerun()

        # ---------------- Datetime Features ----------------
        elif fe_category == "Datetime Feature Extraction":
            dt_cols = [c for c in working_df.columns if classify_column_type(working_df[c]) == "Datetime" or "date" in str(c).lower()]
            if not dt_cols:
                st.warning("No datetime columns detected. Please convert a date column in the Cleaning Workspace first.")
            else:
                date_fe_mode = st.radio("Datetime Mode:", options=["Extract Date Components (Year, Month, etc.)", "Date Difference (Days Between Two Dates)"], horizontal=True)

                if "Extract" in date_fe_mode:
                    sel_dt = st.selectbox("Select Date Column:", options=dt_cols, key="fe_dt_sel")
                    comps = st.multiselect("Select components to extract:", options=["year", "month", "day", "dayofweek", "quarter"], default=["year", "month"])
                    dt_prefix = st.text_input("Feature Prefix:", value=sel_dt, key="fe_dt_prefix")

                    if st.button("Preview Date Extraction", key="btn_prev_dt"):
                        prev = current_manager.preview_operation("feature_datetime_extract", {"column": sel_dt, "components": comps, "prefix": dt_prefix})
                        _render_what_changed_card(prev)
                    if st.button("Apply Date Extraction", key="btn_apply_dt", type="primary"):
                        current_manager.apply_operation("feature_datetime_extract", {"column": sel_dt, "components": comps, "prefix": dt_prefix})
                        st.success(f"Extracted components for '{sel_dt}'.")
                        st.rerun()

                else:  # Date diff
                    if len(dt_cols) < 2:
                        st.warning("Date difference requires at least two date columns.")
                    else:
                        d_c1, d_c2, d_c3 = st.columns(3)
                        with d_c1:
                            start_col = st.selectbox("Start Date Column:", options=dt_cols, key="fe_dt_start")
                        with d_c2:
                            end_col = st.selectbox("End Date Column:", options=dt_cols, key="fe_dt_end")
                        with d_c3:
                            diff_name = st.text_input("New Feature Name:", value=f"{end_col}_minus_{start_col}_days", key="fe_dt_diff_name")

                        if st.button("Preview Date Diff", key="btn_prev_diff"):
                            prev = current_manager.preview_operation("feature_date_diff", {"col_start": start_col, "col_end": end_col, "new_col_name": diff_name})
                            _render_what_changed_card(prev)
                        if st.button("Apply Date Diff", key="btn_apply_diff", type="primary"):
                            current_manager.apply_operation("feature_date_diff", {"col_start": start_col, "col_end": end_col, "new_col_name": diff_name})
                            st.success(f"Created date difference feature '{diff_name}'.")
                            st.rerun()

        # ---------------- Categorical Encoding & Mapping ----------------
        elif fe_category == "Categorical Encoding & Mapping":
            cat_cols = [c for c in working_df.columns if classify_column_type(working_df[c]) in ("Categorical/Text", "Boolean")]
            if not cat_cols:
                st.warning("No categorical columns available.")
            else:
                cat_fe_mode = st.radio("Encoding Mode:", options=["Binary Indicator (1/0)", "One-Hot Encoding", "Category Mapping"], horizontal=True)

                if "Binary Indicator" in cat_fe_mode:
                    c_b1, c_b2, c_b3 = st.columns(3)
                    with c_b1:
                        target_col = st.selectbox("Select Categorical Column:", options=cat_cols, key="fe_bin_ind_col")
                    with c_b2:
                        top_vals = working_df[target_col].dropna().astype(str).str.strip().unique()[:20].tolist()
                        target_val = st.selectbox("Target Value (matches = 1):", options=top_vals, key="fe_bin_ind_val")
                    with c_b3:
                        new_name = st.text_input("New Feature Name:", value=f"is_{target_val.lower().replace(' ', '_')}", key="fe_bin_ind_name")

                    if st.button("Preview Binary Indicator", key="btn_prev_bin_ind"):
                        prev = current_manager.preview_operation("feature_binary_indicator", {"column": target_col, "target_value": target_val, "new_col_name": new_name})
                        _render_what_changed_card(prev)
                    if st.button("Apply Binary Indicator", key="btn_apply_bin_ind", type="primary"):
                        current_manager.apply_operation("feature_binary_indicator", {"column": target_col, "target_value": target_val, "new_col_name": new_name})
                        st.success(f"Created binary indicator '{new_name}'.")
                        st.rerun()

                elif "One-Hot" in cat_fe_mode:
                    c_o1, c_o2, c_o3 = st.columns(3)
                    with c_o1:
                        target_col = st.selectbox("Select Column to Encode:", options=cat_cols, key="fe_ohe_col")
                    with c_o2:
                        drop_f = st.checkbox("Drop First Category (Avoid Collinearity)", value=False)
                        max_c = st.slider("Max Categories to Encode:", min_value=3, max_value=25, value=10)
                    with c_o3:
                        prefix_val = st.text_input("Prefix for encoded columns:", value=target_col, key="fe_ohe_prefix")

                    if st.button("Preview One-Hot Encoding", key="btn_prev_ohe"):
                        prev = current_manager.preview_operation("feature_one_hot", {"column": target_col, "drop_first": drop_f, "prefix": prefix_val, "max_categories": max_c})
                        _render_what_changed_card(prev)
                    if st.button("Apply One-Hot Encoding", key="btn_apply_ohe", type="primary"):
                        current_manager.apply_operation("feature_one_hot", {"column": target_col, "drop_first": drop_f, "prefix": prefix_val, "max_categories": max_c})
                        st.success(f"Applied One-Hot Encoding on '{target_col}'.")
                        st.rerun()

                else:  # Category mapping
                    target_col = st.selectbox("Select Column to Map:", options=cat_cols, key="fe_cmap_col")
                    unique_vals = working_df[target_col].dropna().astype(str).str.strip().unique()[:10].tolist()
                    st.caption("Define mapping for top categories (other categories will be set to default):")

                    mapping_dict = {}
                    m_cols = st.columns(min(len(unique_vals), 4) if unique_vals else 1)
                    for i, val in enumerate(unique_vals[:8]):
                        with m_cols[i % len(m_cols)]:
                            new_val = st.text_input(f"Map '{val}' to:", value=val, key=f"fe_map_val_{i}")
                            mapping_dict[val] = new_val

                    default_category = st.text_input("Default category for unmapped values:", value="Other", key="fe_map_def")
                    new_name = st.text_input("New Feature Name:", value=f"{target_col}_mapped", key="fe_map_name")

                    if st.button("Preview Category Mapping", key="btn_prev_cmap"):
                        prev = current_manager.preview_operation("feature_category_map", {"column": target_col, "mapping": mapping_dict, "default_val": default_category, "new_col_name": new_name})
                        _render_what_changed_card(prev)
                    if st.button("Apply Category Mapping", key="btn_apply_cmap", type="primary"):
                        current_manager.apply_operation("feature_category_map", {"column": target_col, "mapping": mapping_dict, "default_val": default_category, "new_col_name": new_name})
                        st.success(f"Created mapped feature '{new_name}'.")
                        st.rerun()

        # ---------------- Text Features ----------------
        elif fe_category == "Text Feature Extraction":
            text_cols = [c for c in working_df.columns if classify_column_type(working_df[c]) in ("Categorical/Text", "Other")]
            if not text_cols:
                st.warning("No text columns available.")
            else:
                sel_text_col = st.selectbox("Select Text Column:", options=text_cols, key="fe_text_col")
                text_op = st.selectbox("Text Feature Type:", options=["Text Character Length", "Word Count"], key="fe_text_op")

                new_name = st.text_input(
                    "New Feature Name:",
                    value=f"{sel_text_col}_len" if "Length" in text_op else f"{sel_text_col}_word_count",
                    key="fe_text_name"
                )

                op_key = "feature_text_length" if "Length" in text_op else "feature_word_count"

                if st.button("Preview Text Feature", key="btn_prev_txt"):
                    prev = current_manager.preview_operation(op_key, {"column": sel_text_col, "new_col_name": new_name})
                    _render_what_changed_card(prev)
                if st.button("Apply Text Feature", key="btn_apply_txt", type="primary"):
                    current_manager.apply_operation(op_key, {"column": sel_text_col, "new_col_name": new_name})
                    st.success(f"Created text feature '{new_name}'.")
                    st.rerun()

        # ---------------- Netflix Strategic Recipes ----------------
        elif fe_category == "Netflix Strategy Recipes":
            st.markdown("##### Pre-Configured Strategic Netflix Catalog Recipes")
            st.caption("Apply proven executive business feature formulas. Each recipe validates required schema preconditions before execution.")

            recipe_choices = [
                ("acquisition_lag", "Licensing / Acquisition Lag (year_added - release_year)", ["year_added", "release_year"]),
                ("content_origin_proxy", "Content Origin Proxy (Original vs Recent vs Back-Catalog)", ["acquisition_lag"]),
                ("audience_tier", "Audience Maturity Tier (Kids / Family / Mature)", ["rating"]),
                ("primary_genre", "Primary Genre Extraction (Lead Category)", ["listed_in"]),
                ("primary_country", "Primary Producing Country", ["country"]),
                ("metadata_soup", "NLP Metadata Soup (Title + Director + Cast + Genres + Synopsis)", ["title", "description"]),
            ]

            for r_key, r_title, req_cols in recipe_choices:
                has_reqs = all(c in working_df.columns for c in req_cols) or (r_key == "acquisition_lag" and ("release_year" in working_df.columns and "date_added" in working_df.columns))
                st.markdown(f"**{r_title}**")
                st.caption(f"Requires: `{', '.join(req_cols)}` — Status: {'✅ Eligible' if has_reqs else '❌ Missing required columns'}")

                r_col1, r_col2 = st.columns(2)
                with r_col1:
                    if st.button(f"Preview '{r_key}'", key=f"btn_prev_recipe_{r_key}", disabled=not has_reqs):
                        prev = current_manager.preview_operation("netflix_recipe", {"recipe_name": r_key})
                        _render_what_changed_card(prev)
                with r_col2:
                    if st.button(f"Apply '{r_key}'", key=f"btn_apply_recipe_{r_key}", disabled=not has_reqs, type="primary"):
                        current_manager.apply_operation("netflix_recipe", {"recipe_name": r_key})
                        st.success(f"Applied Netflix strategic recipe '{r_key}'.")
                        st.rerun()

                st.markdown("---")

    # ==================================================================
    # SUBTAB 4: WHAT CHANGED? COMPARISON ENGINE
    # ==================================================================
    with cl_tab4:
        st.markdown("#### The 'What Changed?' Engine")
        st.caption("Detailed before-and-after state analysis for the most recently applied operation.")

        if current_manager.last_comparison is None:
            st.info("ℹ️ No transformations have been applied yet in this session. Apply an operation in Tab 2 or 3 to inspect its detailed consequences.")
        else:
            _render_what_changed_card(current_manager.last_comparison)

            st.markdown("##### Data Preview (Top 3 Records Before vs. After)")
            c_prev_b, c_prev_a = st.columns(2)
            with c_prev_b:
                st.markdown("**State Before Operation:**")
                st.dataframe(pd.DataFrame(current_manager.last_comparison["sample_preview_before"]), use_container_width=True)
            with c_prev_a:
                st.markdown("**State After Operation:**")
                st.dataframe(pd.DataFrame(current_manager.last_comparison["sample_preview_after"]), use_container_width=True)

    # ==================================================================
    # SUBTAB 5: AUDIT LEDGER & UNDO
    # ==================================================================
    with cl_tab5:
        st.markdown("#### Applied Transformation Audit Ledger & Reversible State")
        st.caption("Review all ordered operations applied to the working copy. Undo recent steps or reset completely to raw source.")

        u_col1, u_col2 = st.columns(2)
        with u_col1:
            if st.button("↩️ Undo Most Recent Operation", key="btn_undo_step", disabled=(len(current_manager.history) == 0)):
                undone = current_manager.undo_last_step()
                if undone:
                    st.success(f"Undid step #{undone['step_number']} ({undone['operation_type']}).")
                    st.rerun()

        with u_col2:
            if st.button("🔄 Reset Working Copy to Raw Source", key="btn_reset_source", type="secondary"):
                current_manager.reset_to_source()
                st.success("Working copy reset to immutable raw source!")
                st.rerun()

        st.markdown("---")
        st.markdown("##### Applied Steps History")

        if not current_manager.history:
            st.info("No transformations applied yet. Working copy is identical to raw source.")
        else:
            ledger_rows = []
            for item in current_manager.history:
                step_num = item["step_number"]
                details = item["details"]
                ledger_rows.append({
                    "Step": step_num,
                    "Operation": details.get("operation", item["operation_type"]),
                    "Parameters": str(item["params"]),
                    "Key Outcome": str({k: v for k, v in details.items() if k not in ("operation",)}),
                })
            st.dataframe(pd.DataFrame(ledger_rows), use_container_width=True, hide_index=True)

    # ==================================================================
    # SUBTAB 6: EXPORT HUB
    # ==================================================================
    with cl_tab6:
        st.markdown("#### DataLab Export Hub")
        st.caption("Download your transformed dataset and reproducible transformation audit reports.")

        e_col1, e_col2, e_col3 = st.columns(3)

        with e_col1:
            transformed_csv = current_manager.export_transformed_csv()
            st.download_button(
                label=f"📥 Download Transformed Dataset ({len(current_manager.working_df):,} rows CSV)",
                data=transformed_csv,
                file_name=f"{source_name.replace(' ', '_').lower()}_transformed.csv",
                mime="text/csv",
                help="Exports the full working DataFrame without index columns."
            )

        with e_col2:
            audit_report_csv = current_manager.export_cleaning_report_csv()
            st.download_button(
                label="📥 Download Audit Ledger Report (CSV)",
                data=audit_report_csv,
                file_name=f"{source_name.replace(' ', '_').lower()}_cleaning_report.csv",
                mime="text/csv",
                help="Exports step-by-step before-and-after audit report as a CSV table."
            )

        with e_col3:
            pipeline_json = current_manager.export_pipeline_json()
            st.download_button(
                label="📥 Download Transformation Plan (JSON)",
                data=pipeline_json,
                file_name=f"{source_name.replace(' ', '_').lower()}_pipeline_plan.json",
                mime="application/json",
                help="Exports the serializable ordered transformation configuration plan for replay."
            )

        st.markdown("---")
        st.markdown("##### 🔄 Import & Replay Transformation Pipeline (JSON)")
        st.caption("Replay an exported JSON transformation pipeline plan transactionally on the current active raw source dataset.")

        uploaded_json_file = st.file_uploader(
            "Upload Transformation Plan (.json):",
            type=["json"],
            key="uploader_pipeline_json",
            help="Select a valid JSON pipeline plan exported from DataLab Clean & Transform."
        )

        p_json_text = ""
        if uploaded_json_file is not None:
            try:
                p_json_text = uploaded_json_file.getvalue().decode("utf-8")
            except Exception as e:
                st.error(f"Failed to read JSON file: {e}")
        else:
            p_json_text = st.text_area(
                "Or paste JSON Pipeline Plan text directly:",
                placeholder="{\"pipeline_steps\": [...]}",
                key="textarea_pipeline_json"
            )

        if p_json_text.strip():
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                if st.button("🔍 Validate Pipeline Plan", key="btn_validate_pipeline"):
                    v_res = current_manager.import_pipeline_json(p_json_text.strip(), validate_only=True)
                    if v_res.get("success"):
                        st.success(f"✅ Pipeline is valid! Found {v_res.get('step_count')} compatible operations ready for replay.")
                        if v_res.get("audit_preview"):
                            preview_rows = [{"Step": p["step"], "Operation": p["operation"], "Details": str(p["details"])} for p in v_res["audit_preview"]]
                            st.dataframe(pd.DataFrame(preview_rows), use_container_width=True, hide_index=True)
                    else:
                        st.error(f"❌ Validation failed: {v_res.get('error')}")

            with btn_col2:
                if st.button("🚀 Apply Pipeline Replay", key="btn_apply_pipeline_replay", type="primary"):
                    r_res = current_manager.import_pipeline_json(p_json_text.strip(), validate_only=False)
                    if r_res.get("success"):
                        st.success(f"🎉 Replay succeeded! Applied {r_res.get('applied_steps')} operations. Working dataset now has {r_res.get('working_rows'):,} rows × {r_res.get('working_cols')} columns.")
                        st.rerun()
                    else:
                        st.error(f"❌ Replay failed: {r_res.get('error')}")
