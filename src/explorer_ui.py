"""
explorer_ui.py
Streamlit UI rendering module for Phase 1 DataLab: Interactive Dataset Explorer.

Provides modular UI components for:
- Dataset source selection (Built-in Netflix vs Upload Custom CSV)
- Dataset overview KPI cards (total/filtered rows, duplicate rows, missingness, memory)
- Column schema explorer / data dictionary
- Missing-value analysis (rankings, interactive bar chart, sampled matrix heatmap)
- Interactive column-level distributions (numeric histograms/boxplots, categorical bar charts, datetime)
- Search, filtering, and sorting workspace
- Record browser with pagination and detailed individual row inspection
- CSV and schema profile report exports
"""

from typing import Dict, Any, Optional
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.data_explorer import (
    resolve_builtin_netflix_path,
    load_csv_from_file_or_bytes,
    compute_dataset_identity,
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
    format_bytes,
)
from src.ui_theme import (
    render_page_header,
    render_kpi_row,
    render_callout,
    apply_figure_theme,
    COLOR_CYAN,
    COLOR_VIOLET,
    COLOR_SUCCESS,
    COLOR_WARNING,
    COLOR_DANGER
)

# Plotly theme styling constants (modern dark analytics palette)
PRIMARY_COLOR = COLOR_CYAN
ACCENT_COLOR = COLOR_VIOLET
SLATE_COLOR = "#64748B"
DARK_TEXT = "#F8FAFC"
LIGHT_BG = "rgba(0,0,0,0)"
CARD_BORDER = "#1E2B44"
GRID_COLOR = "#1E2B44"
MISSING_COLOR = COLOR_DANGER
PRESENT_COLOR = COLOR_CYAN


def _apply_plotly_style(fig: go.Figure, title: str = "", height: int = 400) -> go.Figure:
    """Applies unified modern dark analytics styling to Plotly figures."""
    return apply_figure_theme(fig, title=title, height=height)


def reset_explorer_filters_callback() -> None:
    """
    Reliable callback to clear all active filters and associated widget states.
    Deletes widget keys from session_state so they reinitialize to clean defaults.
    """
    st.session_state["explorer_filters"] = {}
    st.session_state["explorer_page_num"] = 1
    keys_to_clear = [
        k for k in list(st.session_state.keys())
        if k.startswith("filter_")
        or k.startswith("rng_slider_")
        or k.startswith("radio_missing_")
        or k.startswith("ms_cat_")
        or k.startswith("txt_match_")
        or k.startswith("cat_search_")
        or k.startswith("dt_picker_")
        or k.startswith("cb_use_temp_dt_")
        or k.startswith("dt_col_")
        or k == "input_global_search"
        or k == "schema_col_search"
    ]
    for k in keys_to_clear:
        if k in st.session_state:
            del st.session_state[k]


def render_explorer_tab(netflix_raw_df: Optional[pd.DataFrame] = None) -> None:
    """
    Main entry point for the DataLab Interactive Dataset Explorer tab.
    Completely isolated from legacy Netflix analytical pipelines.
    """
    # ------------------------------------------------------------------
    # Explorer Header
    # ------------------------------------------------------------------
    render_page_header(
        title="Interactive Dataset Explorer",
        subtitle="Inspect schema quality, discover distributions, execute multi-condition filters, and investigate records.",
        badge="Dataset Profile",
        badge_type="info"
    )

    # ------------------------------------------------------------------
    # Dataset Source Selection & Loading
    # ------------------------------------------------------------------
    st.markdown("### 1. Dataset Source Selection")
    source_choice = st.radio(
        "Choose a dataset to explore:",
        options=[
            "Option A: Built-in Netflix Catalog (8,807 titles)",
            "Option B: Upload your own CSV file"
        ],
        index=0,
        horizontal=True,
        help="Select the built-in Kaggle Netflix catalog or upload any CSV dataset."
    )

    active_df: Optional[pd.DataFrame] = None
    dataset_name: str = ""
    source_metadata: Dict[str, Any] = {}

    if "Option A" in source_choice:
        try:
            resolved_path = resolve_builtin_netflix_path()
            if netflix_raw_df is not None and not netflix_raw_df.empty:
                active_df = netflix_raw_df.copy()
            else:
                active_df, _ = load_csv_from_file_or_bytes(resolved_path, filename="netflix_titles.csv")
            dataset_name = "Built-in Netflix Catalog"
            current_sig = compute_dataset_identity(active_df, source_type="builtin", resolved_path=resolved_path)
            source_metadata = {
                "Source Type": "Local Repository Dataset",
                "Resolved Path": resolved_path,
                "Original Rows": len(active_df),
                "Original Columns": active_df.shape[1],
                "content_hash": current_sig,
            }
        except Exception as e:
            st.error(f"Failed to load built-in Netflix dataset: {str(e)}")
            return

    else:  # Option B: Upload custom CSV
        uploaded_file = st.file_uploader(
            "Upload a CSV dataset from your local machine:",
            type=["csv"],
            help="Accepts CSV files with standard delimiters and text encodings (UTF-8, Latin-1, CP1252)."
        )

        if uploaded_file is None:
            st.info(" Please upload a CSV file above to begin exploratory data profiling, or switch to Option A to explore the built-in Netflix catalog.")
            return

        try:
            file_bytes = uploaded_file.getvalue()
            parsed_df, meta = load_csv_from_file_or_bytes(file_bytes, filename=uploaded_file.name)
            active_df = parsed_df
            dataset_name = uploaded_file.name
            current_sig = compute_dataset_identity(active_df, source_type="uploaded", file_bytes=file_bytes, filename=uploaded_file.name)
            source_metadata = {
                "Source Type": "Uploaded CSV File",
                "Filename": meta["filename"],
                "Encoding Used": meta["encoding_used"],
                "File Size": format_bytes(meta["file_size_bytes"]),
                "Original Rows": meta["row_count"],
                "Original Columns": meta["column_count"],
                "content_hash": current_sig,
            }
            if meta["warnings"]:
                for warn in meta["warnings"]:
                    st.warning(warn)
        except Exception as e:
            st.error(f"Could not parse uploaded CSV: {str(e)}")
            st.caption("Please verify that your file is a valid CSV and is not empty or corrupted.")
            return

    if active_df is None or active_df.empty:
        st.warning("The selected dataset is empty (0 records).")
        return

    # Check and initialize session state for dataset switching using cryptographic identity
    if st.session_state.get("explorer_active_sig") != current_sig:
        st.session_state["explorer_active_sig"] = current_sig
        reset_explorer_filters_callback()
        st.session_state["explorer_page_size"] = 25
        st.session_state["explorer_sort_col"] = None
        st.session_state["explorer_sort_asc"] = True
        if "datalab_working_manager" in st.session_state:
            del st.session_state["datalab_working_manager"]

    # Shared DataLab state for Phase 2 (Clean & Transform)
    st.session_state["datalab_source_data"] = active_df
    st.session_state["datalab_source_identity"] = current_sig
    st.session_state["datalab_source_name"] = dataset_name
    st.session_state["datalab_source_meta"] = source_metadata

    # ------------------------------------------------------------------
    # Missing-Value Rule Toggle
    # ------------------------------------------------------------------
    m_col1, m_col2 = st.columns([3, 1])
    with m_col1:
        treat_empty_as_missing = st.checkbox(
            "Treat empty or whitespace-only strings as missing values",
            value=True,
            help="When checked, blank strings or whitespace-only cells in text columns are treated as missing for profiling and analysis without mutating the source DataFrame."
        )
    with m_col2:
        st.caption(f"**Active Dataset:** `{dataset_name}`")

    # Compute overview metrics from active dataset
    overview = compute_overview_metrics(active_df, treat_empty_as_missing=treat_empty_as_missing)

    # ------------------------------------------------------------------
    # Filtering Workspace State Application
    # ------------------------------------------------------------------
    current_filters = st.session_state.get("explorer_filters", {})
    filtered_df = apply_explorer_filters(
        active_df,
        filters=current_filters,
        treat_empty_as_missing=treat_empty_as_missing
    )

    total_rows = overview["total_rows"]
    filtered_rows = len(filtered_df)
    is_filtered = filtered_rows != total_rows

    # ------------------------------------------------------------------
    # Overview KPI Cards
    # ------------------------------------------------------------------
    st.markdown("### 2. Dataset Overview Dashboard")
    overview_cards = [
        {
            "title": "Row Count",
            "value": f"{filtered_rows:,}",
            "subtext": f"Filtered from {total_rows:,}" if is_filtered else f"{total_rows:,} total rows",
            "subtext_neutral": not is_filtered,
            "accent_color": COLOR_CYAN if is_filtered else COLOR_SUCCESS
        },
        {
            "title": "Columns",
            "value": f"{overview['total_columns']:,}",
            "subtext": f"{overview['numeric_cols_count']} num  {overview['categorical_cols_count']} cat  {overview['datetime_cols_count']} dt",
            "subtext_neutral": True,
            "accent_color": COLOR_VIOLET
        },
        {
            "title": "Duplicate Rows",
            "value": f"{overview['duplicate_rows']:,}",
            "subtext": f"{overview['duplicate_rows_pct']}% of full dataset",
            "subtext_neutral": True,
            "accent_color": COLOR_CYAN
        },
        {
            "title": "Missing Cells",
            "value": f"{overview['total_missing_cells']:,}",
            "subtext": f"{overview['missing_cells_pct']}% of full dataset",
            "subtext_neutral": False,
            "accent_color": COLOR_WARNING if overview['total_missing_cells'] > 0 else COLOR_SUCCESS
        },
        {
            "title": "Complete Rows",
            "value": f"{overview['complete_rows']:,}",
            "subtext": f"{overview['complete_rows_pct']}% zero-null records",
            "subtext_neutral": True,
            "accent_color": COLOR_SUCCESS
        },
        {
            "title": "Memory Size",
            "value": overview['memory_usage_str'],
            "subtext": "Active catalog memory",
            "subtext_neutral": True,
            "accent_color": COLOR_VIOLET
        }
    ]
    render_kpi_row(overview_cards)

    if is_filtered:
        st.info(
            f"**Filtered View Active:** Row Count reflects the active filtered view (**{filtered_rows:,}** of {total_rows:,} records), "
            f"while baseline quality indicators (duplicates, missing cells, memory) represent the complete source dataset (**{total_rows:,}** records)."
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # Detailed Explorer Workspace Subtabs
    # ------------------------------------------------------------------
    sub1, sub2, sub3, sub4, sub5 = st.tabs([
        "Column Schema & Data Dictionary",
        "Missing-Value Audit & Visualizer",
        "Column Distributions & Deep Dive",
        "Filter & Search Workspace",
        "Record Browser & Inspector"
    ])

    # ==================================================================
    # SUBTAB 1: COLUMN SCHEMA & DATA DICTIONARY
    # ==================================================================
    with sub1:
        st.markdown("#### Interactive Column Schema & Data Dictionary")
        st.markdown(
            "Inspect the schema, pandas data types, non-null rates, distinct values, and sample observations for every column."
        )

        schema_df = compute_column_schema(active_df, treat_empty_as_missing=treat_empty_as_missing)

        col_filter_term = st.text_input(
            "Filter schema table by column name:",
            placeholder="Type column name to filter schema...",
            key="schema_col_search"
        )
        displayed_schema = schema_df
        if col_filter_term.strip():
            displayed_schema = schema_df[schema_df["Column"].str.contains(col_filter_term.strip(), case=False, na=False)]

        st.dataframe(
            displayed_schema,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Missing %": st.column_config.ProgressColumn(
                    "Missing %",
                    format="%.2f%%",
                    min_value=0.0,
                    max_value=100.0,
                ),
                "Sample Values": st.column_config.TextColumn("Sample Values", width="large"),
            }
        )

        # Profile CSV Download Button
        col_prof_btn, _ = st.columns([2, 2])
        with col_prof_btn:
            profile_csv_str = export_profile_csv(schema_df)
            st.download_button(
                label="Download Data Profile Report (CSV)",
                data=profile_csv_str,
                file_name=f"{dataset_name.replace(' ', '_').lower()}_data_profile.csv",
                mime="text/csv",
                help="Download the complete data dictionary schema report as a CSV file."
            )

    # ==================================================================
    # SUBTAB 2: MISSING-VALUE AUDIT & VISUALIZER
    # ==================================================================
    with sub2:
        st.markdown("#### Missing-Value Analysis & Visualizer")

        render_callout(
            title="Understanding Missingness Patterns",
            body=(
                "A missing value indicates that an individual cell lacks data, distinct from empty rows. "
                "High missingness (>20% or >50%) often indicates optional fields, format conventions "
                "(such as episodic television rotating directors), or localized variations."
            ),
            callout_type="info"
        )

        rankings_df = compute_missing_rankings(active_df, treat_empty_as_missing=treat_empty_as_missing)

        c_rank, c_bar = st.columns([1, 1])
        with c_rank:
            st.markdown("##### Missing Value Rankings")
            st.dataframe(
                rankings_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Missing %": st.column_config.ProgressColumn(
                        "Missing %",
                        format="%.2f%%",
                        min_value=0.0,
                        max_value=100.0,
                    ),
                }
            )

        with c_bar:
            st.markdown("##### Missingness Distribution by Column")
            missing_cols_df = rankings_df[rankings_df["Missing Count"] > 0]
            if missing_cols_df.empty:
                st.success(" No missing values detected in any column! The dataset is 100% complete.")
            else:
                bar_fig = px.bar(
                    missing_cols_df,
                    x="Column",
                    y="Missing %",
                    text="Missing %",
                    color="Missing %",
                    color_continuous_scale="Reds",
                    labels={"Missing %": "Missing (%)"},
                )
                bar_fig.update_traces(texttemplate="%{y:.1f}%", textposition="outside")
                _apply_plotly_style(bar_fig, title="", height=360)
                st.plotly_chart(bar_fig, use_container_width=True)

        # Missingness Matrix Heatmap
        st.markdown("##### Missingness Sparsity Matrix (Heatmap)")
        matrix_df, was_sampled, total_recs = get_missingness_matrix(
            active_df, max_rows=500, treat_empty_as_missing=treat_empty_as_missing
        )
        if matrix_df.empty:
            st.info("No data available to construct missingness matrix.")
        else:
            if was_sampled:
                st.caption(
                    f" Displaying an evenly spaced systematic sample of 500 rows across all {total_recs:,} total records to ensure smooth rendering."
                )
            else:
                st.caption(f"Displaying all {total_recs:,} rows in the matrix.")

            # Heatmap Plotly
            # 1 = Missing (red/orange), 0 = Present (electric cyan)
            matrix_fig = go.Figure(
                data=go.Heatmap(
                    z=matrix_df.values,
                    x=list(matrix_df.columns),
                    y=[f"Row {i}" for i in matrix_df.index],
                    colorscale=[[0.0, COLOR_CYAN], [1.0, COLOR_DANGER]],
                    showscale=False,
                    hovertemplate="Column: %{x}<br>Row: %{y}<br>Status: %{z}<extra></extra>",
                )
            )
            matrix_fig.update_layout(
                height=320,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin={"l": 40, "r": 20, "t": 20, "b": 40},
                xaxis={"tickangle": -45, "tickfont": {"size": 11, "color": DARK_TEXT}},
                yaxis={"showticklabels": False},
            )
            st.plotly_chart(matrix_fig, use_container_width=True)
            st.caption(" Cyan = Present / Non-Null &nbsp;|&nbsp;  Red = Missing / Null")

        # Inspect Records with Missing Values
        st.markdown("##### Inspect Records with Missing Values")
        with st.expander("Preview records with missing values"):
            cols_with_missing = rankings_df[rankings_df["Missing Count"] > 0]["Column"].tolist()
            if not cols_with_missing:
                st.write("All records are 100% complete.")
            else:
                col_to_inspect = st.selectbox(
                    "Select column to isolate missing records (or choose 'Any Column'):",
                    options=["Any Column"] + cols_with_missing,
                    key="missing_inspect_col"
                )
                m_mask = get_missing_mask(active_df, treat_empty_as_missing=treat_empty_as_missing)
                if col_to_inspect == "Any Column":
                    missing_rows = active_df[m_mask.any(axis=1)]
                else:
                    missing_rows = active_df[m_mask[col_to_inspect]]

                st.write(f"Found **{len(missing_rows):,}** records with missing values in `{col_to_inspect}`.")
                st.dataframe(missing_rows.head(50), use_container_width=True)

    # ==================================================================
    # SUBTAB 3: COLUMN DISTRIBUTIONS & DEEP DIVE
    # ==================================================================
    with sub3:
        st.markdown("#### Interactive Column Analysis & Distributions")
        st.markdown(
            "Select any column to investigate descriptive statistics, distribution shapes, and outlier metrics."
        )

        all_cols = list(active_df.columns)
        col_choice = st.selectbox(
            "Select column to inspect:",
            options=all_cols,
            format_func=lambda c: f"[{classify_column_type(active_df[c])}] {c}",
            key="col_distribution_selector"
        )

        if col_choice:
            selected_series = active_df[col_choice]
            col_type = classify_column_type(selected_series)

            # ---------------- Numeric Column Analysis ----------------
            if col_type == "Numeric":
                stats = compute_numeric_stats(selected_series, treat_empty_as_missing=treat_empty_as_missing)
                if not stats["has_data"]:
                    st.warning(f"Column '{col_choice}' cannot be analyzed: {stats.get('reason')}")
                else:
                    n1, n2, n3, n4, n5, n6 = st.columns(6)
                    n1.metric("Non-Missing", f"{stats['valid_count']:,}")
                    n2.metric("Mean", f"{stats['mean']:,}")
                    n3.metric("Median", f"{stats['median']:,}")
                    n4.metric("Std Dev", f"{stats['std']:,}")
                    n5.metric("Min", f"{stats['min']:,}")
                    n6.metric("Max", f"{stats['max']:,}")

                    n7, n8, n9, n10, n11, n12 = st.columns(6)
                    n7.metric("Q1 (25%)", f"{stats['q25']:,}")
                    n8.metric("Q3 (75%)", f"{stats['q75']:,}")
                    n9.metric("IQR", f"{stats['iqr']:,}")
                    n10.metric("Skewness", f"{stats['skewness']}")
                    n11.metric("Zeros", f"{stats['zeros_count']:,} ({stats['zeros_pct']}%)")
                    n12.metric("Outliers (1.5 IQR)", f"{stats['outlier_count']:,} ({stats['outlier_pct']}%)")

                    # Configurable bins slider
                    bin_count = st.slider(
                        "Histogram Bins:",
                        min_value=5,
                        max_value=100,
                        value=30,
                        step=5,
                        key=f"bins_slider_{col_choice}"
                    )

                    c_hist, c_box = st.columns([1, 1])
                    with c_hist:
                        hist_fig = px.histogram(
                            selected_series.dropna(),
                            x=col_choice,
                            nbins=bin_count,
                            marginal="rug",
                            color_discrete_sequence=[PRIMARY_COLOR],
                            labels={col_choice: str(col_choice)},
                        )
                        _apply_plotly_style(hist_fig, title=f"Distribution of {col_choice}", height=380)
                        st.plotly_chart(hist_fig, use_container_width=True)

                    with c_box:
                        box_fig = px.box(
                            selected_series.dropna(),
                            y=col_choice,
                            color_discrete_sequence=[ACCENT_COLOR],
                            labels={col_choice: str(col_choice)},
                            points="outliers",
                        )
                        _apply_plotly_style(box_fig, title=f"Box Plot & Outlier Spread: {col_choice}", height=380)
                        st.plotly_chart(box_fig, use_container_width=True)

            # ---------------- Categorical / Text Column Analysis ----------------
            elif col_type in ("Categorical/Text", "Other"):
                top_n_slider = st.slider(
                    "Number of Top Categories to Display:",
                    min_value=5,
                    max_value=50,
                    value=10,
                    step=5,
                    key=f"cat_slider_{col_choice}"
                )
                cat_stats = compute_categorical_stats(
                    selected_series, top_n=top_n_slider, treat_empty_as_missing=treat_empty_as_missing
                )

                if not cat_stats["has_data"]:
                    st.warning(f"Column '{col_choice}' cannot be analyzed: {cat_stats.get('reason')}")
                else:
                    k_c1, k_c2, k_c3, k_c4 = st.columns(4)
                    k_c1.metric("Non-Missing Records", f"{cat_stats['valid_count']:,}")
                    k_c2.metric("Unique Categories", f"{cat_stats['unique_count']:,}")
                    k_c3.metric("Top Category", str(cat_stats['top_category']))
                    k_c4.metric("Top Category Share", f"{cat_stats['top_category_pct']}%")

                    if cat_stats["is_high_cardinality"]:
                        st.info(
                            f" High cardinality detected ({cat_stats['unique_count']:,} distinct values). Showing the top {top_n_slider} most frequent categories."
                        )

                    c_chart, c_tbl = st.columns([3, 2])
                    with c_chart:
                        freq_df = cat_stats["freq_table"]
                        bar_cat_fig = px.bar(
                            freq_df,
                            x="Count",
                            y="Category",
                            orientation="h",
                            text="Count",
                            color="Count",
                            color_continuous_scale="Blues",
                        )
                        bar_cat_fig.update_layout(yaxis={"autorange": "reversed"})
                        bar_cat_fig.update_traces(textposition="outside")
                        _apply_plotly_style(bar_cat_fig, title=f"Top {top_n_slider} Categories: {col_choice}", height=400)
                        st.plotly_chart(bar_cat_fig, use_container_width=True)

                    with c_tbl:
                        st.markdown(f"**Frequency Breakdown (Top {top_n_slider})**")
                        st.dataframe(
                            cat_stats["freq_table"],
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "Percentage": st.column_config.ProgressColumn(
                                    "Percentage",
                                    format="%.2f%%",
                                    min_value=0.0,
                                    max_value=100.0,
                                ),
                            }
                        )

                    # Inspect Category Records Drill-down
                    st.markdown("##### Inspect Records for a Specific Category")
                    cat_options = cat_stats["freq_table"]["Category"].tolist()
                    if cat_options:
                        selected_cat = st.selectbox(
                            f"Filter records where '{col_choice}' equals:",
                            options=cat_options,
                            key=f"inspect_cat_{col_choice}"
                        )
                        cat_matches = active_df[active_df[col_choice].astype(str).str.strip() == str(selected_cat)]
                        st.write(f"Displaying {len(cat_matches):,} records matching `{selected_cat}`:")
                        st.dataframe(cat_matches.head(25), use_container_width=True)

                # Optional Datetime Preview for String columns
                with st.expander(f"Preview Datetime Parsing for '{col_choice}' (Non-destructive)"):
                    st.caption("Test if this string column can be parsed into calendar dates. Does not mutate the source dataset.")
                    dt_preview = preview_datetime_parsing(selected_series)
                    st.write(f"**Parse Success Rate:** `{dt_preview['success_rate_pct']}%` ({dt_preview['parsed_count']:,} valid dates, {dt_preview['failed_count']:,} unparseable)")
                    if dt_preview["samples"]:
                        st.dataframe(pd.DataFrame(dt_preview["samples"]), use_container_width=True, hide_index=True)

            # ---------------- Datetime Column Analysis ----------------
            elif col_type == "Datetime":
                dt_stats = compute_datetime_stats(selected_series)
                if not dt_stats["has_data"]:
                    st.warning(f"Column '{col_choice}' has no valid datetime values.")
                else:
                    d1, d2, d3, d4 = st.columns(4)
                    d1.metric("Non-Missing Dates", f"{dt_stats['valid_count']:,}")
                    d2.metric("Earliest Date", dt_stats["min_date"])
                    d3.metric("Latest Date", dt_stats["max_date"])
                    d4.metric("Time Span", f"{dt_stats['span_days']:,} days")

                    if not dt_stats["year_freq_df"].empty:
                        dt_fig = px.bar(
                            dt_stats["year_freq_df"],
                            x="Year",
                            y="Count",
                            text="Count",
                            color_discrete_sequence=[PRIMARY_COLOR],
                        )
                        dt_fig.update_traces(textposition="outside")
                        _apply_plotly_style(dt_fig, title=f"Timeline Frequency by Year: {col_choice}", height=380)
                        st.plotly_chart(dt_fig, use_container_width=True)

            # ---------------- Boolean Column Analysis ----------------
            elif col_type == "Boolean":
                bool_counts = selected_series.value_counts(dropna=False)
                st.write(f"**Boolean Counts for '{col_choice}':**")
                st.dataframe(bool_counts)

    # ==================================================================
    # SUBTAB 4: FILTER & SEARCH WORKSPACE
    # ==================================================================
    with sub4:
        st.markdown("#### Search, Filtering & Sorting Workspace")
        st.markdown(
            "Execute multi-condition filters across numeric ranges, categorical values, text substrings, and missingness states. "
            "All active filters are combined with **AND logic** and update downstream records in real time."
        )

        filter_status_col, reset_col = st.columns([3, 1])
        with filter_status_col:
            if is_filtered:
                st.warning(f"**Filters Active:** Showing **{filtered_rows:,}** of **{total_rows:,}** records ({round(filtered_rows / total_rows * 100, 1)}%)")
            else:
                st.success(f"**No Filters Active:** Showing all **{total_rows:,}** records.")

        with reset_col:
            st.button(
                "Reset All Filters",
                key="btn_reset_filters",
                on_click=reset_explorer_filters_callback,
                use_container_width=True,
                help="Clears all active filters, restores full original dataset view, and resets pagination to page 1."
            )

        st.markdown("---")

        # Global Search
        search_term = st.text_input(
            " Global Search (across searchable text columns):",
            value=current_filters.get("global_search", ""),
            placeholder="Type any search query...",
            key="input_global_search"
        )
        if search_term != current_filters.get("global_search", ""):
            current_filters["global_search"] = search_term
            st.session_state["explorer_filters"] = current_filters
            st.session_state["explorer_page_num"] = 1

        # Granular Filters in Expandable Columns (Numeric, Categorical, Datetime, Missingness)
        with st.expander("Advanced Column Filters", expanded=True):
            f_col1, f_col2, f_col3 = st.columns(3)

            # Left Column: Numeric Range & Missingness
            with f_col1:
                st.markdown("**Numeric Range Filters**")
                numeric_cols = [c for c in active_df.columns if classify_column_type(active_df[c]) == "Numeric"]
                if numeric_cols:
                    selected_num_col = st.selectbox(
                        "Select numeric column to filter:",
                        options=["None"] + numeric_cols,
                        key="filter_num_col_select"
                    )
                    if selected_num_col != "None":
                        s_vals = pd.to_numeric(active_df[selected_num_col], errors="coerce").dropna()
                        if not s_vals.empty:
                            min_data = float(s_vals.min())
                            max_data = float(s_vals.max())
                            if min_data < max_data:
                                current_rng = current_filters.get("numeric_ranges", {}).get(selected_num_col, (min_data, max_data))
                                new_rng = st.slider(
                                    f"Filter range for '{selected_num_col}':",
                                    min_value=min_data,
                                    max_value=max_data,
                                    value=current_rng,
                                    key=f"rng_slider_{selected_num_col}"
                                )
                                if "numeric_ranges" not in current_filters:
                                    current_filters["numeric_ranges"] = {}
                                current_filters["numeric_ranges"][selected_num_col] = new_rng
                else:
                    st.caption("No numeric columns available in active dataset.")

                st.markdown("**Missingness Filters**")
                all_col_options = ["None"] + list(active_df.columns)
                missing_col = st.selectbox(
                    "Filter by missingness in column:",
                    options=all_col_options,
                    key="filter_missing_col_select"
                )
                if missing_col != "None":
                    missing_mode = st.radio(
                        f"Missingness criteria for '{missing_col}':",
                        options=["All records", "Only Missing rows", "Only Non-Missing rows"],
                        horizontal=True,
                        key=f"radio_missing_{missing_col}"
                    )
                    if "missing_filters" not in current_filters:
                        current_filters["missing_filters"] = {}
                    if missing_mode == "Only Missing rows":
                        current_filters["missing_filters"][missing_col] = "missing_only"
                    elif missing_mode == "Only Non-Missing rows":
                        current_filters["missing_filters"][missing_col] = "not_missing_only"
                    else:
                        current_filters["missing_filters"].pop(missing_col, None)

            # Middle Column: Categorical & Text Substring
            with f_col2:
                st.markdown("**Categorical Filter**")
                cat_cols = [c for c in active_df.columns if classify_column_type(active_df[c]) in ("Categorical/Text", "Boolean")]
                if cat_cols:
                    selected_cat_col = st.selectbox(
                        "Select categorical column:",
                        options=["None"] + cat_cols,
                        key="filter_cat_col_select"
                    )
                    if selected_cat_col != "None":
                        all_cat_vals = sorted(active_df[selected_cat_col].dropna().astype(str).str.strip().unique().tolist())
                        total_cats = len(all_cat_vals)
                        if total_cats > 50:
                            st.caption(f" High cardinality: **{total_cats:,}** distinct values.")
                            cat_search = st.text_input(
                                f"Search values in '{selected_cat_col}':",
                                key=f"cat_search_{selected_cat_col}",
                                placeholder="Type to filter categories..."
                            )
                            if cat_search.strip():
                                visible_cats = [v for v in all_cat_vals if cat_search.strip().lower() in v.lower()]
                                st.caption(f"Showing {len(visible_cats):,} of {total_cats:,} matching values.")
                            else:
                                visible_cats = all_cat_vals[:50]
                                st.caption(f"Showing first 50 values. Use the search input above to access any category.")
                        else:
                            visible_cats = all_cat_vals

                        current_cat_sel = current_filters.get("categorical_selections", {}).get(selected_cat_col, [])
                        options_union = sorted(list(set(visible_cats + current_cat_sel)))
                        new_cat_sel = st.multiselect(
                            f"Select values for '{selected_cat_col}':",
                            options=options_union,
                            default=[c for c in current_cat_sel if c in options_union],
                            key=f"ms_cat_{selected_cat_col}",
                            help="Filter rows matching any selected category (exact case-sensitive match)."
                        )
                        if "categorical_selections" not in current_filters:
                            current_filters["categorical_selections"] = {}
                        if new_cat_sel:
                            current_filters["categorical_selections"][selected_cat_col] = new_cat_sel
                        else:
                            current_filters["categorical_selections"].pop(selected_cat_col, None)

                st.markdown("**Text Substring Filter**")
                if cat_cols:
                    selected_text_col = st.selectbox(
                        "Select text column for substring match:",
                        options=["None"] + cat_cols,
                        key="filter_text_col_select"
                    )
                    if selected_text_col != "None":
                        txt_val = st.text_input(
                            f"Substring to search in '{selected_text_col}':",
                            value=current_filters.get("text_searches", {}).get(selected_text_col, ""),
                            key=f"txt_match_{selected_text_col}"
                        )
                        if "text_searches" not in current_filters:
                            current_filters["text_searches"] = {}
                        if txt_val.strip():
                            current_filters["text_searches"][selected_text_col] = txt_val.strip()
                        else:
                            current_filters["text_searches"].pop(selected_text_col, None)

            # Right Column: Datetime Range Filter
            with f_col3:
                st.markdown("**Date Range Filter**")
                date_candidates = ["None"]
                dt_meta = {}
                for c in active_df.columns:
                    ctype = classify_column_type(active_df[c])
                    if ctype == "Datetime":
                        date_candidates.append(c)
                        dt_meta[c] = {"is_native": True}
                    elif ctype in ("Categorical/Text", "Other"):
                        pr = preview_datetime_parsing(active_df[c])
                        if pr["success_rate_pct"] >= 50.0 and pr["parsed_count"] > 0:
                            date_candidates.append(c)
                            dt_meta[c] = {"is_native": False, "preview": pr}

                if len(date_candidates) > 1:
                    selected_dt_col = st.selectbox(
                        "Select column for date filter:",
                        options=date_candidates,
                        key="filter_date_col_select"
                    )
                    if selected_dt_col != "None":
                        info = dt_meta[selected_dt_col]
                        can_filter = False
                        parsed_s = None

                        if not info["is_native"]:
                            pr = info["preview"]
                            st.caption(
                                f" String dates: **{pr['success_rate_pct']}%** parseable "
                                f"({pr['parsed_count']:,} valid, {pr['failed_count']:,} failed)."
                            )
                            use_temp = st.checkbox(
                                f"Use temporary parsed datetime for '{selected_dt_col}'",
                                value=selected_dt_col in current_filters.get("date_ranges", {}),
                                key=f"cb_use_temp_dt_{selected_dt_col}",
                                help="Applies date filtering using an in-memory datetime representation without changing source data types."
                            )
                            if use_temp:
                                can_filter = True
                                try:
                                    parsed_s = pd.to_datetime(active_df[selected_dt_col], errors="coerce", format="mixed")
                                except (ValueError, TypeError):
                                    parsed_s = pd.to_datetime(active_df[selected_dt_col], errors="coerce")
                        else:
                            can_filter = True
                            parsed_s = pd.to_datetime(active_df[selected_dt_col], errors="coerce")

                        if can_filter and parsed_s is not None:
                            valid_dates = parsed_s.dropna()
                            if valid_dates.empty:
                                st.warning(f"No valid dates found in '{selected_dt_col}'.")
                            else:
                                min_dt = valid_dates.min().date()
                                max_dt = valid_dates.max().date()
                                if min_dt == max_dt:
                                    st.caption(f"All records have date: {min_dt}")
                                else:
                                    curr_d_range = current_filters.get("date_ranges", {}).get(selected_dt_col, (min_dt, max_dt))
                                    try:
                                        d_val = (
                                            curr_d_range[0] if hasattr(curr_d_range[0], "year") else pd.to_datetime(curr_d_range[0]).date(),
                                            curr_d_range[1] if hasattr(curr_d_range[1], "year") else pd.to_datetime(curr_d_range[1]).date(),
                                        )
                                    except Exception:
                                        d_val = (min_dt, max_dt)

                                    date_choice = st.date_input(
                                        f"Inclusive dates for '{selected_dt_col}':",
                                        value=d_val,
                                        min_value=min_dt,
                                        max_value=max_dt,
                                        key=f"dt_picker_{selected_dt_col}",
                                        help="Rows where the date falls within [Start Date 00:00:00, End Date 23:59:59] (inclusive)."
                                    )
                                    if isinstance(date_choice, (tuple, list)) and len(date_choice) == 2:
                                        if "date_ranges" not in current_filters:
                                            current_filters["date_ranges"] = {}
                                        current_filters["date_ranges"][selected_dt_col] = (str(date_choice[0]), str(date_choice[1]))
                                    elif selected_dt_col in current_filters.get("date_ranges", {}):
                                        del current_filters["date_ranges"][selected_dt_col]
                    else:
                        current_filters.pop("date_ranges", None)
                else:
                    st.caption("No datetime or parseable date columns found in active dataset.")

        st.session_state["explorer_filters"] = current_filters

    # ==================================================================
    # SUBTAB 5: RECORD BROWSER & INSPECTOR
    # ==================================================================
    with sub5:
        st.markdown("#### Record Browser & Individual Row Inspector")

        if filtered_df.empty:
            st.warning("No records match your active filters. Click 'Reset All Filters' in the Filter Workspace.")
        else:
            # Table Controls: Columns to display, Sorting, Page Size
            c_ctrl1, c_ctrl2, c_ctrl3 = st.columns([3, 2, 1])

            with c_ctrl1:
                display_cols = st.multiselect(
                    "Select columns to display in table view:",
                    options=list(active_df.columns),
                    default=list(active_df.columns)[:min(8, len(active_df.columns))],
                    key="table_display_cols"
                )
                if not display_cols:
                    display_cols = list(active_df.columns)

            with c_ctrl2:
                sort_col = st.selectbox(
                    "Sort by column:",
                    options=["None"] + list(active_df.columns),
                    key="table_sort_col"
                )
                sort_asc = st.radio("Order:", options=["Ascending", "Descending"], horizontal=True, key="table_sort_order") == "Ascending"

            with c_ctrl3:
                page_size = st.selectbox("Page Size:", options=[25, 50, 100], index=0, key="table_page_size")

            # Pagination computation with safe clamping
            current_page = st.session_state.get("explorer_page_num", 1)
            actual_sort_col = sort_col if sort_col != "None" else None

            calc_total_pages = max(1, (len(filtered_df) + page_size - 1) // page_size) if not filtered_df.empty else 1
            if current_page > calc_total_pages:
                current_page = calc_total_pages
                st.session_state["explorer_page_num"] = current_page
            elif current_page < 1:
                current_page = 1
                st.session_state["explorer_page_num"] = current_page

            page_slice, total_pages, total_recs = paginate_dataframe(
                filtered_df,
                page_number=current_page,
                page_size=page_size,
                sort_by=actual_sort_col,
                ascending=sort_asc
            )

            # Pagination Navigation Buttons
            p_prev, p_info, p_next = st.columns([1, 2, 1])
            with p_prev:
                if st.button("⬅ Previous Page", disabled=(current_page <= 1), use_container_width=True):
                    st.session_state["explorer_page_num"] = max(1, current_page - 1)
                    st.rerun()

            with p_info:
                st.markdown(
                    f"<div style='text-align: center; padding-top: 6px; font-weight: 600; color: #F8FAFC;'>"
                    f"Page {current_page} of {total_pages} &nbsp;|&nbsp; Showing records {((current_page - 1) * page_size) + 1:,}{min(current_page * page_size, total_recs):,} of {total_recs:,}"
                    f"</div>",
                    unsafe_allow_html=True
                )

            with p_next:
                if st.button("Next Page ", disabled=(current_page >= total_pages), use_container_width=True):
                    st.session_state["explorer_page_num"] = min(total_pages, current_page + 1)
                    st.rerun()

            # Render Table View
            st.dataframe(
                page_slice[display_cols],
                use_container_width=True,
                hide_index=False
            )

            # Record Detail Inspector
            st.markdown("---")
            st.markdown("##### Individual Record Inspector")
            st.caption("Inspect all fields for any specific record on the displayed page.")

            record_indices = list(page_slice.index)
            # Create readable labels
            label_col = "title" if "title" in page_slice.columns else page_slice.columns[0]
            format_record_choice = lambda idx: f"Row Index {idx}: {str(page_slice.loc[idx, label_col])[:40]}"

            selected_record_idx = st.selectbox(
                "Select a record from the current page to inspect:",
                options=record_indices,
                format_func=format_record_choice,
                key="record_detail_selector"
            )

            if selected_record_idx is not None and selected_record_idx in page_slice.index:
                record_series = page_slice.loc[selected_record_idx]
                st.markdown(f"""
                    <div class="insight-box" style="margin-top: 10px;">
                        <div class="insight-title">
                            <span></span> Record Details (Index: {selected_record_idx})
                        </div>
                    </div>
                """, unsafe_allow_html=True)

                # Format fields cleanly into 2 columns
                r_c1, r_c2 = st.columns(2)
                fields = list(active_df.columns)
                midpoint = (len(fields) + 1) // 2

                with r_c1:
                    for f in fields[:midpoint]:
                        val = record_series.get(f)
                        val_str = "[Missing / Null]" if pd.isna(val) or (treat_empty_as_missing and str(val).strip() == "") else str(val)
                        st.markdown(f"**{f}:** {val_str}")

                with r_c2:
                    for f in fields[midpoint:]:
                        val = record_series.get(f)
                        val_str = "[Missing / Null]" if pd.isna(val) or (treat_empty_as_missing and str(val).strip() == "") else str(val)
                        st.markdown(f"**{f}:** {val_str}")

            # ----------------------------------------------------------
            # Export Hub
            # ----------------------------------------------------------
            st.markdown("---")
            st.markdown("##### Export Hub")
            st.caption("Download the complete filtered dataset or profiling data dictionary.")

            d_col1, d_col2 = st.columns(2)
            with d_col1:
                filtered_csv_str = export_filtered_csv(filtered_df)
                st.download_button(
                    label=f" Download Filtered Data ({len(filtered_df):,} rows CSV)",
                    data=filtered_csv_str,
                    file_name=f"{dataset_name.replace(' ', '_').lower()}_filtered_data.csv",
                    mime="text/csv",
                    help="Exports all filtered records (not just the visible page) with original columns."
                )

            with d_col2:
                profile_csv_str = export_profile_csv(schema_df)
                st.download_button(
                    label=" Download Dataset Schema Profile (CSV)",
                    data=profile_csv_str,
                    file_name=f"{dataset_name.replace(' ', '_').lower()}_schema_profile.csv",
                    mime="text/csv",
                    help="Exports the column-level data dictionary report."
                )
