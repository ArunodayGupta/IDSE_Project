"""
netflix_case_study_ui.py
Dedicated presentation module for the Netflix Content Strategy & Catalog Case Study.

Preserves the domain-specific executive analytics, strategic licensing insights,
global audience localization, format mechanics, and the data hygiene pipeline audit
(including the Missing Director Hypothesis).
"""

from typing import Dict, Any
import streamlit as st
import pandas as pd

from src.feature_engineer import explode_column
from src.visualizations import (
    plot_catalog_growth,
    plot_acquisition_lag_trend,
    plot_top_countries_audience,
    plot_genre_cooccurrence,
    plot_duration_distributions,
    plot_missing_directors_comparison,
)
from src.ui_theme import (
    render_page_header,
    render_kpi_row,
    render_callout,
    apply_figure_theme,
    COLOR_CYAN,
    COLOR_VIOLET,
    COLOR_SUCCESS,
)


def render_netflix_case_study_tab(
    raw_df: pd.DataFrame,
    processed_df: pd.DataFrame,
    hygiene_report: Dict[str, Any]
) -> None:
    """
    Renders the executive Netflix Content Strategy & Catalog Intelligence workspace.
    Includes strategic slicers, high-impact KPI scorecards, interactive Plotly visualizations,
    and the structural Missing Director data quality study.
    """
    render_page_header(
        title="Netflix Content Strategy & Catalog Intelligence",
        subtitle="Executive case study: catalog scaling, licensing transitions, international distribution, and format mechanics.",
        badge="Executive Briefing",
        badge_type="info"
    )

    # -------------------------------------------------------------------------
    # 1. Netflix Strategic Slicers (Contained locally within the Case Study)
    # -------------------------------------------------------------------------
    with st.expander("Filter & Slice Catalog Data", expanded=False):
        c1, c2, c3 = st.columns([1.2, 1, 1])

        with c1:
            min_year = int(processed_df["year_added"].min()) if "year_added" in processed_df.columns and not processed_df.empty else 2010
            max_year = int(processed_df["year_added"].max()) if "year_added" in processed_df.columns and not processed_df.empty else 2024
            selected_year_range = st.slider(
                "Year Added to Netflix",
                min_value=min_year,
                max_value=max_year,
                value=(max(min_year, 2014), max_year),
                step=1
            )

        with c2:
            available_types = sorted(processed_df["type"].dropna().unique().tolist()) if "type" in processed_df.columns else ["Movie", "TV Show"]
            selected_types = st.multiselect(
                "Content Format",
                options=available_types,
                default=available_types
            )

        with c3:
            available_audiences = ["Kids", "Family / Teens", "Mature / Adults"]
            selected_audiences = st.multiselect(
                "Audience Maturity Tier",
                options=available_audiences,
                default=available_audiences
            )

        # Country Filter
        all_countries_exploded = explode_column(processed_df, "country")
        top_country_options = all_countries_exploded["country"].value_counts().head(20).index.tolist() if not all_countries_exploded.empty else []
        selected_countries = st.multiselect(
            "Filter by Country (Multi-Country)",
            options=top_country_options,
            default=[]
        )

    # Apply Slicers to active analysis slice
    filtered_df = processed_df.copy()

    if "year_added" in filtered_df.columns:
        filtered_df = filtered_df[
            (filtered_df["year_added"] >= selected_year_range[0]) &
            (filtered_df["year_added"] <= selected_year_range[1])
        ]

    if selected_types:
        filtered_df = filtered_df[filtered_df["type"].isin(selected_types)]

    if selected_audiences:
        filtered_df = filtered_df[filtered_df["audience_tier"].isin(selected_audiences)]

    if selected_countries:
        country_pattern = "|".join(selected_countries)
        filtered_df = filtered_df[filtered_df["country"].astype(str).str.contains(country_pattern, case=False, na=False)]

    # -------------------------------------------------------------------------
    # 2. Executive KPI Scorecard
    # -------------------------------------------------------------------------
    total_titles = len(filtered_df)
    base_total = len(processed_df)
    pct_active = (total_titles / base_total * 100) if base_total > 0 else 100.0

    movie_count = (filtered_df["type"] == "Movie").sum() if "type" in filtered_df.columns else 0
    tv_count = (filtered_df["type"] == "TV Show").sum() if "type" in filtered_df.columns else 0
    tv_share_pct = (tv_count / total_titles * 100) if total_titles > 0 else 0.0
    ratio_str = f"{round(movie_count / max(1, tv_count), 1)} : 1"

    median_runtime = (
        int(filtered_df[filtered_df["type"] == "Movie"]["duration_min"].median())
        if "type" in filtered_df.columns and (filtered_df["type"] == "Movie").any() and "duration_min" in filtered_df.columns else 0
    )

    same_year_count = (filtered_df["acquisition_lag"] == 0).sum() if "acquisition_lag" in filtered_df.columns else 0
    same_year_pct = (same_year_count / total_titles * 100) if total_titles > 0 else 0.0

    kpi_cards = [
        {
            "title": "Active Catalog Volume",
            "value": f"{total_titles:,}",
            "subtext": f"{round(pct_active, 1)}% of total catalog ({base_total:,})",
            "accent_color": COLOR_CYAN
        },
        {
            "title": "TV Catalog Share & Ratio",
            "value": f"{round(tv_share_pct, 1)}%",
            "subtext": f"Movie-to-TV Ratio: {ratio_str}",
            "subtext_neutral": True,
            "accent_color": COLOR_VIOLET
        },
        {
            "title": "Median Movie Runtime",
            "value": f"{median_runtime} min",
            "subtext": "Sweet spot: 90110 minutes",
            "subtext_neutral": True,
            "accent_color": COLOR_CYAN
        },
        {
            "title": "Originals / Day-and-Date",
            "value": f"{round(same_year_pct, 1)}%",
            "subtext": "Same-Year Additions (Lag = 0)",
            "accent_color": COLOR_SUCCESS
        }
    ]
    render_kpi_row(kpi_cards)

    st.markdown("<br>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 3. Four Strategic Subtabs
    # -------------------------------------------------------------------------
    cs_tab1, cs_tab2, cs_tab3, cs_tab4 = st.tabs([
        "Executive Strategy & Growth",
        "Global Distribution & Ratings",
        "Genre & Format Analysis",
        "Pipeline Quality & Missing Director Study"
    ])

    # Tab 1: Strategy & Growth
    with cs_tab1:
        render_callout(
            title="Strategic Overview: Studio Licensing Transition",
            body=(
                "Between 2012 and 2016, distribution strategy relied heavily on "
                "<b>deep back-catalogs (licensing lag > 3 years)</b> from legacy studios. "
                "In response to shifting studio licensing models and competitor platforms, "
                "capital allocation transitioned toward <b>in-house Originals and same-year exclusive "
                "acquisitions (Lag = 0)</b>, paired with increased episodic TV volume for subscriber retention."
            ),
            callout_type="info"
        )

        st.markdown("### Catalog Growth Trends")
        growth_fig = apply_figure_theme(plot_catalog_growth(filtered_df), title="Cumulative & Annual Additions by Type")
        st.plotly_chart(growth_fig, use_container_width=True)

        st.markdown("### Licensing Strategy Evolution (Originals vs. Licensed)")
        lag_fig = apply_figure_theme(plot_acquisition_lag_trend(filtered_df), title="Acquisition Lag Distribution Over Time")
        st.plotly_chart(lag_fig, use_container_width=True)

        st.markdown("#### Annual Ingestion Breakdown")
        if "year_added" in filtered_df.columns:
            summary_table = (
                filtered_df.groupby("year_added")
                .agg(
                    Total_Added=("show_id", "count"),
                    Movies=("type", lambda x: (x == "Movie").sum()),
                    TV_Shows=("type", lambda x: (x == "TV Show").sum()),
                    Same_Year_Releases=("acquisition_lag", lambda x: (x == 0).sum()),
                    Back_Catalog=("acquisition_lag", lambda x: (x > 3).sum())
                )
                .reset_index()
                .sort_values("year_added", ascending=False)
            )
            summary_table["TV_Share_%"] = (summary_table["TV_Shows"] / summary_table["Total_Added"] * 100).round(1)
            summary_table["Originals_Proxy_%"] = (summary_table["Same_Year_Releases"] / summary_table["Total_Added"] * 100).round(1)

            formatted_summary = summary_table.style.format({
                "Total_Added": "{:,}",
                "Movies": "{:,}",
                "TV_Shows": "{:,}",
                "Same_Year_Releases": "{:,}",
                "Back_Catalog": "{:,}",
                "TV_Share_%": "{:.1f}%",
                "Originals_Proxy_%": "{:.1f}%"
            })
            st.dataframe(formatted_summary, use_container_width=True, hide_index=True)

    # Tab 2: Global Distribution & Ratings
    with cs_tab2:
        render_callout(
            title="Strategic Overview: International Content Distribution",
            body=(
                "To sustain subscriber growth beyond mature domestic markets, content development "
                "expanded into key international production hubs including <b>India, the United Kingdom, "
                "and South Korea</b>. Western markets emphasize mature programming (TV-MA / R), while "
                "regions like India prioritize <b>Family and Teen programming (TV-14 / TV-PG)</b> "
                "tailored to multi-generational co-viewing patterns."
            ),
            callout_type="info"
        )

        st.markdown("### Top Producing Countries by Target Audience Tier")
        country_fig = apply_figure_theme(plot_top_countries_audience(filtered_df, top_n=14), title="Content Distribution Across Audience Tiers")
        st.plotly_chart(country_fig, use_container_width=True)

        st.markdown("#### Top Content Origin Matrix")
        exp_c = explode_column(filtered_df, "country")
        if not exp_c.empty:
            country_summary = (
                exp_c.groupby("country")
                .agg(
                    Total_Titles=("show_id", "count"),
                    Mature_Adults=("audience_tier", lambda x: (x == "Mature / Adults").sum()),
                    Family_Teens=("audience_tier", lambda x: (x == "Family / Teens").sum()),
                    Kids=("audience_tier", lambda x: (x == "Kids").sum()),
                    Median_Lag=("acquisition_lag", "median")
                )
                .sort_values("Total_Titles", ascending=False)
                .head(15)
                .reset_index()
            )
            country_summary["Mature_%"] = (country_summary["Mature_Adults"] / country_summary["Total_Titles"] * 100).round(1)
            country_summary["Family_%"] = (country_summary["Family_Teens"] / country_summary["Total_Titles"] * 100).round(1)

            formatted_country_summary = country_summary.style.format({
                "Total_Titles": "{:,}",
                "Mature_Adults": "{:,}",
                "Family_Teens": "{:,}",
                "Kids": "{:,}",
                "Mature_%": "{:.1f}%",
                "Family_%": "{:.1f}%"
            })
            st.dataframe(formatted_country_summary, use_container_width=True, hide_index=True)

    # Tab 3: Genre & Format Mechanics
    with cs_tab3:
        render_callout(
            title="Format Analysis: Runtime Optimization & Series Retention",
            body=(
                "<b>1. Feature Film Duration:</b> Movie runtimes cluster consistently between <b>90 and 110 minutes</b>, "
                "optimizing completion rates and viewer satisfaction.<br>"
                "<b>2. Episodic Season Attrition:</b> Television series face steep renewal hurdles beyond Season 1; "
                "over 80% do not survive past Season 2, reflecting marginal subscriber acquisition economics."
            ),
            callout_type="info"
        )

        st.markdown("### Runtime & Season Distribution Analysis")
        dist_fig = apply_figure_theme(plot_duration_distributions(filtered_df), title="Movie Duration vs TV Season Distribution")
        st.plotly_chart(dist_fig, use_container_width=True)

        st.markdown("### Genre Co-Occurrence Matrix")
        st.markdown("Analysis of paired genres across catalog titles.")
        heatmap_fig = apply_figure_theme(plot_genre_cooccurrence(filtered_df, top_n=10), title="Top 10 Genre Pairwise Co-Occurrence")
        st.plotly_chart(heatmap_fig, use_container_width=True)

    # Tab 4: Pipeline Quality & Missing Director Study
    with cs_tab4:
        render_callout(
            title="Data Pipeline Quality Audit",
            body="Comprehensive data profiling ensures all missing fields, anomalies, and structural nulls are methodically audited, verified, and documented prior to downstream analytics.",
            callout_type="success"
        )

        q_cards = [
            {"title": "Raw Dataset Rows", "value": f"{hygiene_report['initial_rows']:,}", "accent_color": COLOR_CYAN},
            {"title": "Cleaned Pipeline Rows", "value": f"{hygiene_report['final_rows']:,}", "accent_color": COLOR_SUCCESS},
            {"title": "Anomalous Rows Dropped", "value": f"{hygiene_report['dropped_rows']} ({hygiene_report['dropped_pct']}%)", "accent_color": COLOR_VIOLET},
            {"title": "Engineered Features Added", "value": "5 Dimensions", "subtext": "Lag, Origin, Tier, Exploded Country/Genre", "subtext_neutral": True}
        ]
        render_kpi_row(q_cards)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("### Missing Director Field Analysis")
        st.markdown(f"""
            Analysis demonstrates that missing director entries are largely structural to the medium of episodic television rather than data collection discrepancies:

            * **Movies Missing Director:** `{hygiene_report['director_hypothesis']['movie_missing_dir_pct']}%`
              ({hygiene_report['director_hypothesis']['movie_missing_dir']:,} of {hygiene_report['director_hypothesis']['movie_total']:,})
            * **TV Shows Missing Director:** `{hygiene_report['director_hypothesis']['tv_missing_dir_pct']}%`
              ({hygiene_report['director_hypothesis']['tv_missing_dir']:,} of {hygiene_report['director_hypothesis']['tv_total']:,})

            {hygiene_report['director_hypothesis']['explanation']}
        """)

        dir_fig = apply_figure_theme(plot_missing_directors_comparison(hygiene_report), title="Director Missingness Rate: Movies vs TV Shows")
        st.plotly_chart(dir_fig, use_container_width=True)

        st.markdown("#### Missing Value Audit (Pre- vs. Post-Processing)")
        b_df = hygiene_report["missing_before_df"].rename(columns={
            "Missing_Count": "Raw_Missing_Count",
            "Missing_Pct": "Raw_Missing_%"
        })
        a_df = hygiene_report["missing_after_df"].rename(columns={
            "Missing_Count": "Cleaned_Missing_Count",
            "Missing_Pct": "Cleaned_Missing_%"
        })
        audit_table = pd.merge(b_df, a_df, on="Column", how="outer").fillna(0)
        audit_table["Raw_Missing_Count"] = audit_table["Raw_Missing_Count"].astype(int)
        audit_table["Cleaned_Missing_Count"] = audit_table["Cleaned_Missing_Count"].astype(int)

        formatted_audit = audit_table.style.format({
            "Raw_Missing_Count": "{:,}",
            "Cleaned_Missing_Count": "{:,}",
            "Raw_Missing_%": "{:.2f}%",
            "Cleaned_Missing_%": "{:.2f}%"
        })
        st.dataframe(formatted_audit, use_container_width=True, hide_index=True)
