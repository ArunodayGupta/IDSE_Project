"""
data_explorer_tab.py
Section B: Data Explorer for TMDB Movie Box Office Prediction dataset.
Provides dataset loading, interactive preview and filtering, data quality summary,
and reproducible cleaning and transformation without emojis or excessive UI elements.
"""

from typing import Optional, List, Dict, Any, Tuple
import io
import streamlit as st
import pandas as pd
import numpy as np

from src.data_loader import (
    load_tmdb_train_data,
    load_tmdb_test_data,
    get_dataset_availability,
    safe_parse_json,
    parse_release_date,
)
from src.visualizations import plot_missing_values_summary
from src.ui_theme import (
    render_page_header,
    render_section_header,
    render_metric_card,
    render_kpi_row,
    render_callout,
    COLOR_CYAN,
    COLOR_BLUE,
    COLOR_SUCCESS,
    COLOR_WARNING,
)


def get_dataset_quality_metrics(df: pd.DataFrame) -> Tuple[pd.DataFrame, int, pd.DataFrame]:
    """Computes missing value summary, duplicate count, and descriptive statistics."""
    # Missing values
    null_counts = df.isna().sum()
    null_cols = null_counts[null_counts > 0]
    if not null_cols.empty:
        missing_df = pd.DataFrame({
            "column": null_cols.index,
            "missing_count": null_cols.values,
            "missing_pct": (null_cols.values / len(df) * 100).round(2)
        }).sort_values(by="missing_count", ascending=False).reset_index(drop=True)
    else:
        missing_df = pd.DataFrame(columns=["column", "missing_count", "missing_pct"])

    # Duplicates
    dup_count = int(df.duplicated().sum())

    # Numeric descriptive stats
    num_cols = df.select_dtypes(include=[np.number]).columns
    stats_df = df[num_cols].describe().T[["count", "mean", "std", "min", "50%", "max"]].rename(
        columns={"50%": "median"}
    ).round(2) if len(num_cols) > 0 else pd.DataFrame()

    return missing_df, dup_count, stats_df


def execute_cleaning_pipeline(
    df: pd.DataFrame,
    remove_dups: bool = True,
    impute_numerics: bool = True,
    parse_json_cols: bool = True,
    fix_dates: bool = True,
    flag_suspicious: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Applies reproducible cleaning transformations to a copy of the dataset.
    Returns (cleaned_df, audit_report).
    """
    clean_df = df.copy()
    initial_rows = len(clean_df)
    initial_nulls = int(clean_df.isna().sum().sum())
    actions_taken: List[str] = []

    # 1. Remove Duplicate Records
    if remove_dups:
        dups = clean_df.duplicated().sum()
        if dups > 0:
            clean_df = clean_df.drop_duplicates().reset_index(drop=True)
            actions_taken.append(f"Removed {dups:,} duplicate records.")

    # 2. Identify Suspicious Zero Budget and Runtime
    if flag_suspicious:
        if "budget" in clean_df.columns:
            clean_df["budget_is_zero"] = (pd.to_numeric(clean_df["budget"], errors="coerce").fillna(0) <= 0).astype(int)
        if "runtime" in clean_df.columns:
            clean_df["runtime_is_zero"] = (pd.to_numeric(clean_df["runtime"], errors="coerce").fillna(0) <= 0).astype(int)
        actions_taken.append("Flagged unrecorded zero budgets and runtimes with binary indicator columns.")

    # 3. Impute Missing Numerical Values
    if impute_numerics:
        if "budget" in clean_df.columns:
            b_num = pd.to_numeric(clean_df["budget"], errors="coerce")
            valid_b = b_num[b_num > 0]
            med_b = float(valid_b.median()) if not valid_b.empty else 16000000.0
            clean_df["budget_imputed"] = np.where(b_num > 0, b_num, med_b)
            actions_taken.append(f"Imputed unrecorded budgets with median non-zero budget (${med_b:,.0f}).")

        if "runtime" in clean_df.columns:
            r_num = pd.to_numeric(clean_df["runtime"], errors="coerce")
            valid_r = r_num[r_num > 0]
            med_r = float(valid_r.median()) if not valid_r.empty else 104.0
            clean_df["runtime_imputed"] = np.where(r_num > 0, r_num, med_r)
            actions_taken.append(f"Imputed missing or zero runtimes with median runtime ({med_r:.0f} mins).")

    # 4. Parse Release Date with Century Adjustment
    if fix_dates and "release_date" in clean_df.columns:
        parsed_dates = parse_release_date(clean_df["release_date"])
        clean_df["release_year"] = parsed_dates.dt.year.fillna(2000).astype(int)
        clean_df["release_month"] = parsed_dates.dt.month.fillna(6).astype(int)
        clean_df["release_quarter"] = parsed_dates.dt.quarter.fillna(2).astype(int)
        clean_df["release_dayofweek"] = parsed_dates.dt.dayofweek.fillna(4).astype(int)
        actions_taken.append("Converted release dates and corrected 2-digit century rollover (years > 2020 mapped to 1900s).")

    # 5. Parse Nested JSON Columns
    if parse_json_cols:
        if "genres" in clean_df.columns:
            clean_df["genre_count"] = clean_df["genres"].apply(safe_parse_json).apply(len)
        if "production_companies" in clean_df.columns:
            clean_df["company_count"] = clean_df["production_companies"].apply(safe_parse_json).apply(len)
        if "production_countries" in clean_df.columns:
            clean_df["country_count"] = clean_df["production_countries"].apply(safe_parse_json).apply(len)
        if "spoken_languages" in clean_df.columns:
            clean_df["language_count"] = clean_df["spoken_languages"].apply(safe_parse_json).apply(len)
        if "cast" in clean_df.columns:
            clean_df["cast_size"] = clean_df["cast"].apply(safe_parse_json).apply(len)
        if "crew" in clean_df.columns:
            clean_df["crew_size"] = clean_df["crew"].apply(safe_parse_json).apply(len)
        actions_taken.append("Parsed nested JSON metadata columns into tabular integer counts.")

    # 6. Belongs to Collection Flag
    if "belongs_to_collection" in clean_df.columns:
        clean_df["has_collection"] = (
            clean_df["belongs_to_collection"].notna()
            & (clean_df["belongs_to_collection"].astype(str).str.strip() != "")
            & (clean_df["belongs_to_collection"].astype(str).str.strip() != "[]")
        ).astype(int)
        actions_taken.append("Created binary collection / franchise indicator.")

    final_nulls = int(clean_df.isna().sum().sum())

    audit = {
        "initial_rows": initial_rows,
        "final_rows": len(clean_df),
        "initial_nulls": initial_nulls,
        "final_nulls": final_nulls,
        "actions": actions_taken,
    }

    return clean_df, audit


def render_data_explorer_section(raw_df: pd.DataFrame) -> None:
    """Renders Section B: Data Explorer tab."""
    render_page_header(
        title="Data Explorer",
        subtitle="Inspect, filter, audit quality, and prepare movie dataset records",
        badge="Dataset Tools",
        badge_type="info"
    )

    # -------------------------------------------------------------------------
    # 1. Dataset Loading Controls
    # -------------------------------------------------------------------------
    render_section_header("Dataset Loading", "Select active dataset source or upload custom CSV")

    avail = get_dataset_availability()
    src_choice = st.radio(
        "Dataset Source",
        options=[
            "Default Training Dataset (train.csv)",
            "Upload Custom CSV File",
            "Competition Test Dataset (test.csv)",
        ],
        index=0,
        horizontal=True,
        label_visibility="collapsed"
    )

    active_df = raw_df
    active_label = "train.csv"

    if src_choice == "Default Training Dataset (train.csv)":
        if avail.get("train.csv") is None:
            render_callout(
                title="Default Dataset Missing",
                body=(
                    "train.csv was not detected in project directories. "
                    "Please place train.csv into the 'data/' folder or use the file uploader below."
                ),
                callout_type="warning"
            )
        else:
            active_df = raw_df
            active_label = "train.csv"

    elif src_choice == "Upload Custom CSV File":
        uploaded_file = st.file_uploader("Upload CSV File", type=["csv"], key="explorer_csv_uploader")
        if uploaded_file is not None:
            try:
                active_df = pd.read_csv(uploaded_file)
                active_label = uploaded_file.name
                st.success(f"Loaded {len(active_df):,} rows and {active_df.shape[1]} columns from {uploaded_file.name}")
            except Exception as e:
                st.error(f"Error parsing uploaded file: {str(e)}")
        else:
            st.info("Upload a CSV file to inspect custom movie metadata.")

    elif src_choice == "Competition Test Dataset (test.csv)":
        test_df, test_msg = load_tmdb_test_data()
        if test_df is not None:
            active_df = test_df
            active_label = "test.csv"
            st.info(f"{test_msg} (Revenue target column is excluded as expected for test inference).")
        else:
            render_callout(
                title="Competition Test File Not Found",
                body=(
                    "test.csv was not found in 'data/' or 'tmdb-box-office-prediction/'. "
                    "Place test.csv in the 'data/' directory to inspect test records."
                ),
                callout_type="warning"
            )

    # KPI summary of active dataset
    kpi_items = [
        {"title": "Active Dataset", "value": active_label, "subtext": "Current source", "accent_color": COLOR_CYAN},
        {"title": "Total Rows", "value": f"{len(active_df):,}", "subtext": "Records", "accent_color": COLOR_BLUE},
        {"title": "Total Columns", "value": str(active_df.shape[1]), "subtext": "Attributes", "accent_color": COLOR_SUCCESS},
    ]
    render_kpi_row(kpi_items)

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 2. Data Quality Summary
    # -------------------------------------------------------------------------
    render_section_header("Data Quality Summary", "Audit missing values, duplicate records, and numeric distributions")

    missing_df, dup_count, num_stats = get_dataset_quality_metrics(active_df)

    q_col1, q_col2 = st.columns([1, 1], gap="medium")

    with q_col1:
        st.markdown(f"**Duplicate Records:** `{dup_count:,}`")
        if not missing_df.empty:
            fig_miss = plot_missing_values_summary(missing_df.head(12))
            st.plotly_chart(fig_miss, use_container_width=True)
        else:
            st.success("No missing values detected in the active dataset.")

    with q_col2:
        st.markdown("**Numeric Descriptive Statistics:**")
        if not num_stats.empty:
            st.dataframe(num_stats, use_container_width=True, height=280)
        else:
            st.write("No numeric columns present in active dataset.")

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 3. Interactive Dataset Preview & Filtering
    # -------------------------------------------------------------------------
    render_section_header("Dataset Preview & Filtering", "Inspect raw data with search, column selection, and filtering")

    filter_col1, filter_col2, filter_col3 = st.columns([1.5, 1, 1])

    with filter_col1:
        search_query = st.text_input("Search Title or Synopsis", placeholder="Filter by keyword...", key="exp_search")

    with filter_col2:
        # Resolve release years for filtering
        if "release_date" in active_df.columns:
            dts = parse_release_date(active_df["release_date"])
            valid_years = dts.dt.year.dropna()
            min_yr = int(valid_years.min()) if not valid_years.empty else 1920
            max_yr = int(valid_years.max()) if not valid_years.empty else 2020
            selected_year_range = st.slider("Release Year Range", min_yr, max_yr, (1980, max_yr), key="exp_year_slider")
        else:
            selected_year_range = None

    with filter_col3:
        if "original_language" in active_df.columns:
            top_langs = list(active_df["original_language"].dropna().value_counts().head(8).index)
            selected_langs = st.multiselect("Language Filter", options=top_langs, default=[], key="exp_lang_filter")
        else:
            selected_langs = []

    # Apply filters
    filtered_df = active_df.copy()

    if search_query.strip():
        term = search_query.strip().lower()
        title_mask = filtered_df["title"].astype(str).str.lower().str.contains(term, na=False) if "title" in filtered_df.columns else False
        orig_mask = filtered_df["original_title"].astype(str).str.lower().str.contains(term, na=False) if "original_title" in filtered_df.columns else False
        over_mask = filtered_df["overview"].astype(str).str.lower().str.contains(term, na=False) if "overview" in filtered_df.columns else False
        filtered_df = filtered_df[title_mask | orig_mask | over_mask]

    if selected_year_range and "release_date" in filtered_df.columns:
        dts_filtered = parse_release_date(filtered_df["release_date"]).dt.year
        filtered_df = filtered_df[dts_filtered.between(selected_year_range[0], selected_year_range[1], inclusive="both")]

    if selected_langs and "original_language" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["original_language"].isin(selected_langs)]

    # Column selector
    all_cols = list(active_df.columns)
    default_cols = [c for c in ["id", "title", "budget", "revenue", "release_date", "runtime", "original_language", "genres"] if c in all_cols]
    if not default_cols:
        default_cols = all_cols[:8]

    selected_columns = st.multiselect("Visible Columns", options=all_cols, default=default_cols, key="exp_col_select")

    display_columns = selected_columns if selected_columns else all_cols
    st.dataframe(filtered_df[display_columns], use_container_width=True, height=350)
    st.caption(f"Displaying {len(filtered_df):,} of {len(active_df):,} records.")

    # Download filtered dataset button
    csv_buffer = io.StringIO()
    filtered_df[display_columns].to_csv(csv_buffer, index=False)
    st.download_button(
        label="Download Filtered Records as CSV",
        data=csv_buffer.getvalue(),
        file_name="tmdb_filtered_records.csv",
        mime="text/csv",
        key="btn_download_filtered_csv"
    )

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 4. Cleaning & Transformation Pipeline
    # -------------------------------------------------------------------------
    render_section_header("Cleaning & Transformation", "Apply reproducible data hygiene without modifying raw source")

    st.markdown("Select transformations to apply to a working copy:")

    c_col1, c_col2, c_col3 = st.columns([1, 1, 1])
    with c_col1:
        opt_dups = st.checkbox("Remove Duplicate Records", value=True, key="clean_opt_dups")
        opt_numerics = st.checkbox("Impute Missing Budget & Runtime", value=True, key="clean_opt_numerics")
    with c_col2:
        opt_dates = st.checkbox("Convert Dates & Fix Century Rollover", value=True, key="clean_opt_dates")
        opt_suspicious = st.checkbox("Flag Suspicious 0 Budget / Runtime", value=True, key="clean_opt_suspicious")
    with c_col3:
        opt_json = st.checkbox("Parse JSON Metadata (Cast, Crew, Genres)", value=True, key="clean_opt_json")

    if st.button("Apply Cleaning Transformations", type="primary", key="btn_apply_cleaning"):
        cleaned_df, audit = execute_cleaning_pipeline(
            active_df,
            remove_dups=opt_dups,
            impute_numerics=opt_numerics,
            parse_json_cols=opt_json,
            fix_dates=opt_dates,
            flag_suspicious=opt_suspicious
        )
        st.session_state["tmdb_cleaned_data"] = cleaned_df
        st.session_state["tmdb_cleaning_audit"] = audit
        st.success(f"Transformations applied successfully. Working dataset now has {len(cleaned_df):,} rows and {cleaned_df.shape[1]} columns.")

    # If cleaned dataset exists in session state, display audit and download button
    if "tmdb_cleaned_data" in st.session_state:
        cleaned_copy = st.session_state["tmdb_cleaned_data"]
        audit = st.session_state.get("tmdb_cleaning_audit", {})

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        st.markdown("**Cleaning Audit Report:**")
        for action in audit.get("actions", []):
            st.markdown(f"- {action}")

        clean_csv_buf = io.StringIO()
        cleaned_copy.to_csv(clean_csv_buf, index=False)
        st.download_button(
            label="Download Cleaned Dataset as CSV",
            data=clean_csv_buf.getvalue(),
            file_name="tmdb_cleaned_dataset.csv",
            mime="text/csv",
            key="btn_download_cleaned_csv"
        )
