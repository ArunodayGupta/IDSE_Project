"""
data_explorer_tab.py
Section B: Data Explorer for TMDB Movie Box Office Prediction dataset.
Provides dataset loading, interactive preview and filtering, data quality summary,
reproducible in-app cleaning & transformation, before/after statistical comparison,
and transformation impact analysis without downloadable files or emojis.
"""

from typing import List, Dict, Any, Tuple
import streamlit as st
import pandas as pd
import numpy as np

from src.data_loader import (
    load_tmdb_test_data,
    get_dataset_availability,
    safe_parse_json,
    parse_release_date,
)
from src.visualizations import plot_missing_values_summary
from src.ui_theme import (
    render_page_header,
    render_section_header,
    render_kpi_row,
    render_callout,
    COLOR_CYAN,
    COLOR_BLUE,
    COLOR_VIOLET,
    COLOR_SUCCESS,
    COLOR_WARNING,
)


def get_dataset_quality_metrics(df: pd.DataFrame) -> Tuple[pd.DataFrame, int, pd.DataFrame]:
    """Computes missing value summary, duplicate count, and descriptive statistics."""
    if df.empty:
        return pd.DataFrame(columns=["column", "missing_count", "missing_pct"]), 0, pd.DataFrame()

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

    dup_count = int(df.duplicated().sum())

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
    Returns (cleaned_df, rich_audit_report).
    Accurately computes before-and-after statistics and transformation impact.
    """
    clean_df = df.copy()
    initial_rows = len(clean_df)
    initial_cols = clean_df.shape[1]
    initial_col_names = set(clean_df.columns)
    initial_nulls = int(clean_df.isna().sum().sum())
    initial_dups = int(clean_df.duplicated().sum())

    actions_taken: List[str] = []
    applied_transforms: List[str] = []
    impact_details: Dict[str, Any] = {}
    dups_removed = 0

    # 1. Remove Duplicate Records
    if remove_dups:
        applied_transforms.append("Remove Duplicate Records")
        if initial_dups > 0:
            clean_df = clean_df.drop_duplicates().reset_index(drop=True)
            dups_removed = initial_dups
            actions_taken.append(f"Removed {dups_removed:,} duplicate records.")
        else:
            actions_taken.append("Duplicate audit complete: zero duplicate records detected.")
        impact_details["duplicates"] = {
            "initial_duplicates": initial_dups,
            "duplicates_removed": dups_removed,
            "remaining_duplicates": 0
        }

    # 2. Identify Suspicious Zero Budget and Runtime
    if flag_suspicious:
        applied_transforms.append("Flag Suspicious 0 Budget / Runtime")
        flagged_items = []
        zero_b_count = 0
        zero_r_count = 0
        if "budget" in clean_df.columns:
            clean_df["budget_is_zero"] = (pd.to_numeric(clean_df["budget"], errors="coerce").fillna(0) <= 0).astype(int)
            zero_b_count = int(clean_df["budget_is_zero"].sum())
            flagged_items.append(f"{zero_b_count:,} zero budgets ({zero_b_count / len(clean_df) * 100:.1f}%)")
        if "runtime" in clean_df.columns:
            clean_df["runtime_is_zero"] = (pd.to_numeric(clean_df["runtime"], errors="coerce").fillna(0) <= 0).astype(int)
            zero_r_count = int(clean_df["runtime_is_zero"].sum())
            flagged_items.append(f"{zero_r_count:,} zero runtimes ({zero_r_count / len(clean_df) * 100:.1f}%)")
        if flagged_items:
            actions_taken.append(f"Flagged unrecorded values ({', '.join(flagged_items)}) with binary indicators.")
        impact_details["suspicious"] = {
            "zero_budget_count": zero_b_count,
            "zero_budget_pct": round(zero_b_count / len(clean_df) * 100, 2) if len(clean_df) else 0.0,
            "zero_runtime_count": zero_r_count,
            "zero_runtime_pct": round(zero_r_count / len(clean_df) * 100, 2) if len(clean_df) else 0.0,
        }

    # 3. Impute Missing Numerical Values
    if impute_numerics:
        applied_transforms.append("Impute Missing Budget & Runtime")
        impute_stats: Dict[str, Any] = {}
        if "budget" in clean_df.columns:
            b_num = pd.to_numeric(clean_df["budget"], errors="coerce")
            valid_b = b_num[b_num > 0]
            med_b = float(valid_b.median()) if not valid_b.empty else 16000000.0
            imputed_b_mask = b_num.isna() | (b_num <= 0)
            b_count = int(imputed_b_mask.sum())
            clean_df["budget_imputed"] = np.where(~imputed_b_mask, b_num, med_b)
            actions_taken.append(f"Imputed {b_count:,} missing/zero budgets with median (${med_b:,.0f}).")
            impute_stats["budget"] = {
                "imputed_count": b_count,
                "median_used": med_b,
                "original_mean": float(valid_b.mean()) if not valid_b.empty else 0.0,
                "new_mean": float(clean_df["budget_imputed"].mean()),
            }

        if "runtime" in clean_df.columns:
            r_num = pd.to_numeric(clean_df["runtime"], errors="coerce")
            valid_r = r_num[r_num > 0]
            med_r = float(valid_r.median()) if not valid_r.empty else 104.0
            imputed_r_mask = r_num.isna() | (r_num <= 0)
            r_count = int(imputed_r_mask.sum())
            clean_df["runtime_imputed"] = np.where(~imputed_r_mask, r_num, med_r)
            actions_taken.append(f"Imputed {r_count:,} missing/zero runtimes with median ({med_r:.0f} mins).")
            impute_stats["runtime"] = {
                "imputed_count": r_count,
                "median_used": med_r,
                "original_mean": float(valid_r.mean()) if not valid_r.empty else 0.0,
                "new_mean": float(clean_df["runtime_imputed"].mean()),
            }

        impact_details["imputation"] = impute_stats

    # 4. Parse Release Date with Century Adjustment
    if fix_dates and "release_date" in clean_df.columns:
        applied_transforms.append("Convert Dates & Standardize Rollover")
        parsed_dates = parse_release_date(clean_df["release_date"])
        clean_df["release_year"] = parsed_dates.dt.year
        clean_df["release_month"] = parsed_dates.dt.month
        clean_df["release_quarter"] = parsed_dates.dt.quarter
        clean_df["release_dayofweek"] = parsed_dates.dt.dayofweek
        actions_taken.append("Parsed release dates; normalized 2-digit century rollover while preserving explicit 4-digit dates.")
        valid_yrs = clean_df["release_year"].dropna()
        impact_details["dates"] = {
            "parsed_count": int(parsed_dates.notna().sum()),
            "year_min": int(valid_yrs.min()) if not valid_yrs.empty else None,
            "year_max": int(valid_yrs.max()) if not valid_yrs.empty else None,
            "features_created": ["release_year", "release_month", "release_quarter", "release_dayofweek"]
        }

    # 5. Parse Nested JSON Columns
    if parse_json_cols:
        applied_transforms.append("Parse JSON Metadata (Cast, Crew, Genres)")
        json_parsed_cols = []
        json_feature_names = []
        col_mapping = {
            "genres": "genre_count",
            "production_companies": "company_count",
            "production_countries": "country_count",
            "spoken_languages": "language_count",
            "cast": "cast_size",
            "crew": "crew_size",
        }
        for col in ["genres", "production_companies", "production_countries", "spoken_languages", "cast", "crew"]:
            if col in clean_df.columns:
                target_col = col_mapping.get(col, f"{col}_count")
                clean_df[target_col] = clean_df[col].apply(safe_parse_json).apply(len)
                json_parsed_cols.append(col)
                json_feature_names.append(target_col)
        if json_parsed_cols:
            actions_taken.append(f"Extracted entity counts from JSON metadata columns ({', '.join(json_parsed_cols)}).")
        impact_details["json_parsing"] = {
            "source_columns": json_parsed_cols,
            "features_created": json_feature_names
        }

    # 6. Belongs to Collection Flag
    if "belongs_to_collection" in clean_df.columns:
        clean_df["has_collection"] = (
            clean_df["belongs_to_collection"].notna()
            & (clean_df["belongs_to_collection"].astype(str).str.strip() != "")
            & (clean_df["belongs_to_collection"].astype(str).str.strip() != "[]")
            & (clean_df["belongs_to_collection"].astype(str).str.strip() != "{}")
        ).astype(int)
        coll_count = int(clean_df["has_collection"].sum())
        actions_taken.append(f"Created binary franchise indicator ({coll_count:,} franchise films).")
        impact_details["collection"] = {"franchise_count": coll_count}

    final_nulls = int(clean_df.isna().sum().sum())
    final_cols = clean_df.shape[1]
    new_cols = [c for c in clean_df.columns if c not in initial_col_names]

    audit = {
        "initial_rows": initial_rows,
        "final_rows": len(clean_df),
        "initial_cols": initial_cols,
        "final_cols": final_cols,
        "new_columns": new_cols,
        "initial_nulls": initial_nulls,
        "final_nulls": final_nulls,
        "initial_dups": initial_dups,
        "dups_removed": dups_removed,
        "applied_transforms": applied_transforms,
        "actions": actions_taken,
        "impact_details": impact_details,
    }

    return clean_df, audit


def render_data_explorer_section(raw_df: pd.DataFrame) -> None:
    """Renders Section B: Data Explorer tab."""
    render_page_header(
        title="Data Explorer",
        subtitle="Inspect, filter, audit quality, and prepare movie dataset records"
    )

    # -------------------------------------------------------------------------
    # 1. Dataset Loading Controls
    # -------------------------------------------------------------------------
    render_section_header("Dataset Loading", "Select active dataset partition")

    avail = get_dataset_availability()
    src_choice = st.radio(
        "Dataset Source",
        options=[
            "Default Training Dataset (train.csv)",
            "Competition Test Dataset (test.csv)",
        ],
        index=0,
        horizontal=True,
        label_visibility="collapsed",
        key="explorer_dataset_source_radio"
    )

    # Invalidate stale cleaning state if dataset source changed
    current_source_key = f"src_{src_choice}"

    if st.session_state.get("active_clean_source") != current_source_key:
        st.session_state.pop("tmdb_cleaned_data", None)
        st.session_state.pop("tmdb_cleaning_audit", None)
        st.session_state["active_clean_source"] = current_source_key

    active_df = raw_df
    active_label = "train.csv"

    if src_choice == "Default Training Dataset (train.csv)":
        if avail.get("train.csv") is None:
            render_callout(
                title="Default Dataset Missing",
                body=(
                    "train.csv was not detected in project directories. "
                    "Please place train.csv into 'data/' or 'tmdb-box-office-prediction/'."
                ),
                callout_type="warning"
            )
        else:
            active_df = raw_df
            active_label = "train.csv"

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
                    "Place test.csv in project directories to inspect test records."
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
        # Resolve release years safely
        if "release_date" in active_df.columns:
            dts = parse_release_date(active_df["release_date"])
            valid_years = dts.dt.year.dropna()
            if not valid_years.empty:
                min_yr = int(valid_years.min())
                max_yr = int(valid_years.max())
                if min_yr < max_yr:
                    default_start = max(min_yr, 1980) if min_yr <= 1980 <= max_yr else min_yr
                    selected_year_range = st.slider(
                        "Release Year Range",
                        min_yr,
                        max_yr,
                        (default_start, max_yr),
                        key="exp_year_slider"
                    )
                else:
                    st.caption(f"All releases from year {min_yr}")
                    selected_year_range = None
            else:
                selected_year_range = None
        else:
            selected_year_range = None

    with filter_col3:
        if "original_language" in active_df.columns:
            top_langs = list(active_df["original_language"].dropna().value_counts().head(8).index)
            selected_langs = st.multiselect("Language Filter", options=top_langs, default=[], key="exp_lang_filter")
        else:
            selected_langs = []

    # Apply filters safely
    filtered_df = active_df.copy()

    if search_query.strip():
        term = search_query.strip().lower()
        search_mask = pd.Series(False, index=filtered_df.index)
        for col in ["title", "original_title", "overview"]:
            if col in filtered_df.columns:
                search_mask = search_mask | filtered_df[col].astype(str).str.lower().str.contains(term, regex=False, na=False)
        filtered_df = filtered_df[search_mask]

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
        opt_dates = st.checkbox("Convert Dates & Standardize Rollover", value=True, key="clean_opt_dates")
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
        st.success(f"Transformations applied successfully: {len(cleaned_df):,} rows and {cleaned_df.shape[1]} columns.")

    # Display in-app inspection, statistical comparison, transformation summary, and preview
    if "tmdb_cleaned_data" in st.session_state:
        cleaned_df = st.session_state["tmdb_cleaned_data"]
        audit = st.session_state.get("tmdb_cleaning_audit", {})

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        # ---------------------------------------------------------------------
        # 4.1 Transformation Summary
        # ---------------------------------------------------------------------
        render_section_header(
            "Transformation Summary",
            "Summary of transformations performed and their observed impact"
        )

        # Calculate exact differences dynamically directly from active_df and cleaned_df
        before_rows = len(active_df)
        after_rows = len(cleaned_df)
        rows_removed = before_rows - after_rows

        before_cols = active_df.shape[1]
        after_cols = cleaned_df.shape[1]
        new_cols = [c for c in cleaned_df.columns if c not in active_df.columns]

        before_nulls = int(active_df.isna().sum().sum())
        after_nulls = int(cleaned_df.isna().sum().sum())

        before_dups = int(active_df.duplicated().sum())
        after_dups = int(cleaned_df.duplicated().sum())
        dups_removed = before_dups - after_dups if before_dups >= after_dups else 0

        # Construct clear, factual summary items
        summary_items: List[str] = []

        if opt_dups:
            if dups_removed > 0:
                summary_items.append(f"**Duplicate Records:** Removed {dups_removed:,} duplicate rows.")
            else:
                summary_items.append("**Duplicate Records:** Audited dataset; 0 duplicate records detected (0 removed).")

        if opt_numerics:
            impact_imp = audit.get("impact_details", {}).get("imputation", {})
            b_info = impact_imp.get("budget")
            if b_info:
                summary_items.append(
                    f"**Budget Imputation:** Handled {b_info['imputed_count']:,} missing/zero budget values "
                    f"(imputed into separate column `budget_imputed` using median ${b_info['median_used']:,.0f})."
                )
            r_info = impact_imp.get("runtime")
            if r_info:
                summary_items.append(
                    f"**Runtime Imputation:** Handled {r_info['imputed_count']:,} missing/zero runtime values "
                    f"(imputed into separate column `runtime_imputed` using median {r_info['median_used']:.0f} mins)."
                )

        if opt_suspicious:
            susp = audit.get("impact_details", {}).get("suspicious", {})
            zb = susp.get("zero_budget_count", 0)
            zr = susp.get("zero_runtime_count", 0)
            summary_items.append(
                f"**Suspicious Values:** Flagged {zb:,} zero-budget records (`budget_is_zero`) "
                f"and {zr:,} zero-runtime records (`runtime_is_zero`)."
            )

        if opt_dates and "release_year" in cleaned_df.columns:
            date_features = [c for c in ["release_year", "release_month", "release_quarter", "release_dayofweek"] if c in cleaned_df.columns]
            summary_items.append(
                f"**Date Conversion:** Created {len(date_features)} date-related features ({', '.join(date_features)}) "
                "with standardized century rollover."
            )

        if opt_json:
            json_features = audit.get("impact_details", {}).get("json_parsing", {}).get("features_created", [])
            if json_features:
                summary_items.append(
                    f"**JSON Metadata:** Created {len(json_features)} entity count/size features ({', '.join(json_features)})."
                )

        if "has_collection" in cleaned_df.columns and "has_collection" not in active_df.columns:
            fc = int(cleaned_df["has_collection"].sum())
            summary_items.append(f"**Collection Flag:** Created `has_collection` feature ({fc:,} franchise films flagged).")

        if summary_items:
            for item in summary_items:
                st.markdown(f"- {item}")
        else:
            st.info("No transformations were selected or applied.")

        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

        # ---------------------------------------------------------------------
        # 4.2 Basic Comparison Table
        # ---------------------------------------------------------------------
        render_section_header(
            "Before vs After Statistical Comparison",
            "Direct comparison of dataset dimensions and data quality metrics"
        )

        comp_df = pd.DataFrame([
            {"Metric": "Total Rows", "Before": f"{before_rows:,}", "After": f"{after_rows:,}"},
            {"Metric": "Total Columns", "Before": f"{before_cols:,}", "After": f"{after_cols:,}"},
            {"Metric": "Missing Values", "Before": f"{before_nulls:,}", "After": f"{after_nulls:,}"},
            {"Metric": "Duplicate Records", "Before": f"{before_dups:,}", "After": f"{after_dups:,}"},
        ])
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

        # ---------------------------------------------------------------------
        # 4.3 Numerical Statistics
        # ---------------------------------------------------------------------
        st.markdown("#### Numerical Column Statistics")
        st.caption("Descriptive statistics (Count, Mean, Std, Min, Median, Max) computed via Pandas describe().")

        col_stat1, col_stat2 = st.columns(2, gap="medium")

        with col_stat1:
            st.markdown("**Original Dataset Numerical Statistics:**")
            orig_num_cols = active_df.select_dtypes(include=[np.number]).columns
            if len(orig_num_cols) > 0:
                orig_stats = active_df[orig_num_cols].describe().T[["count", "mean", "std", "min", "50%", "max"]].rename(
                    columns={"50%": "median"}
                ).round(2)
                st.dataframe(orig_stats, use_container_width=True, height=280)
            else:
                st.write("No numerical columns in original dataset.")

        with col_stat2:
            st.markdown("**Transformed Dataset Numerical Statistics:**")
            trans_num_cols = cleaned_df.select_dtypes(include=[np.number]).columns
            if len(trans_num_cols) > 0:
                trans_stats = cleaned_df[trans_num_cols].describe().T[["count", "mean", "std", "min", "50%", "max"]].rename(
                    columns={"50%": "median"}
                ).round(2)
                st.dataframe(trans_stats, use_container_width=True, height=280)
            else:
                st.write("No numerical columns in transformed dataset.")

        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

        # ---------------------------------------------------------------------
        # 4.4 Transformed Dataset Preview
        # ---------------------------------------------------------------------
        render_section_header(
            "Transformed Dataset Preview",
            "Inspect transformed records and newly generated features directly"
        )

        all_clean_cols = list(cleaned_df.columns)
        default_preview = [
            c for c in ["id", "title", "budget", "budget_imputed", "budget_is_zero", "runtime_imputed", "release_year", "genre_count", "cast_size", "has_collection"]
            if c in all_clean_cols
        ]
        if not default_preview:
            default_preview = all_clean_cols[:10]

        if new_cols:
            st.caption(f"Newly Generated Columns ({len(new_cols)}): {', '.join(new_cols)}")

        selected_clean_cols = st.multiselect(
            "Select Columns to Inspect in Transformed Preview",
            options=all_clean_cols,
            default=default_preview,
            key="clean_preview_col_select"
        )

        preview_cols_to_show = selected_clean_cols if selected_clean_cols else all_clean_cols

        st.dataframe(
            cleaned_df[preview_cols_to_show],
            use_container_width=True,
            height=350
        )
        st.caption(f"Displaying {len(cleaned_df):,} transformed records across {len(preview_cols_to_show)} selected attributes (Horizontal scrolling enabled).")
