"""
overview_tab.py
Section A: Overview of the TMDB Movie Box Office Revenue dataset.
Presents core KPIs and essential interactive Plotly visualizations.
Strictly adheres to professional styling with zero emojis.
"""

import streamlit as st
import pandas as pd
import numpy as np

from src.data_loader import safe_parse_json, parse_release_date
from src.visualizations import (
    plot_revenue_distribution,
    plot_budget_vs_revenue,
    plot_revenue_by_genre,
    plot_revenue_trend_over_time,
    plot_budget_distribution,
    format_currency_compact,
)
from src.ui_theme import (
    render_page_header,
    render_kpi_row,
    render_section_header,
    COLOR_CYAN,
    COLOR_BLUE,
    COLOR_VIOLET,
    COLOR_SUCCESS,
)


@st.cache_data(show_spinner=False)
def compute_genre_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Computes median and average revenue and movie counts per major genre."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["genre", "median_revenue", "mean_revenue", "movie_count"])

    records = []
    for _, row in df.iterrows():
        rev = row.get("revenue")
        bud = row.get("budget", 0)
        if pd.notna(rev) and rev > 0:
            parsed = safe_parse_json(row.get("genres"))
            for g in parsed:
                if isinstance(g, dict) and "name" in g:
                    records.append({"genre": g["name"], "revenue": float(rev), "budget": float(bud)})

    if not records:
        return pd.DataFrame(columns=["genre", "median_revenue", "mean_revenue", "movie_count"])

    gdf = pd.DataFrame(records)
    stats = gdf.groupby("genre").agg(
        median_revenue=("revenue", "median"),
        mean_revenue=("revenue", "mean"),
        movie_count=("revenue", "count")
    ).reset_index()

    # Retain genres with meaningful sample sizes (>= 10 movies)
    return stats[stats["movie_count"] >= 10].sort_values(by="median_revenue", ascending=False)


@st.cache_data(show_spinner=False)
def compute_yearly_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Computes median and average worldwide revenue across release years."""
    if df is None or df.empty or "release_date" not in df.columns or "revenue" not in df.columns:
        return pd.DataFrame(columns=["release_year", "median_revenue", "mean_revenue", "movie_count"])

    data = df.copy()
    parsed_dates = parse_release_date(data["release_date"])
    data["release_year"] = parsed_dates.dt.year

    valid = data[data["release_year"].notna() & (data["release_year"] >= 1970) & (data["revenue"] > 0)]
    if valid.empty:
        return pd.DataFrame(columns=["release_year", "median_revenue", "mean_revenue", "movie_count"])

    stats = valid.groupby("release_year").agg(
        median_revenue=("revenue", "median"),
        mean_revenue=("revenue", "mean"),
        movie_count=("revenue", "count")
    ).reset_index()

    return stats.sort_values(by="release_year")


def render_overview_section(df: pd.DataFrame) -> None:
    """Renders Section A: Overview of the Movie Box Office Revenue dataset."""
    render_page_header(
        title="Overview",
        subtitle="Worldwide movie box-office revenue trends and production budget dynamics",
        badge="TMDB Dataset",
        badge_type="info"
    )

    # 1. Compute High-Level Platform KPIs
    total_movies = len(df)
    valid_rev = df[df["revenue"].notna() & (df["revenue"] > 0)]["revenue"] if "revenue" in df.columns else pd.Series(dtype=float)
    avg_revenue = float(valid_rev.mean()) if not valid_rev.empty else 0.0
    med_revenue = float(valid_rev.median()) if not valid_rev.empty else 0.0

    valid_bud = df[df["budget"].notna() & (df["budget"] > 0)]["budget"] if "budget" in df.columns else pd.Series(dtype=float)
    med_budget = float(valid_bud.median()) if not valid_bud.empty else 0.0

    # Robust Median gross revenue-to-budget multiple (strictly NOT labeled as profit or ROI)
    if "budget" in df.columns and "revenue" in df.columns:
        movies_with_budget = df[(df["budget"] > 0) & (df["revenue"] > 0)]
        if not movies_with_budget.empty:
            multiples = (movies_with_budget["revenue"] / movies_with_budget["budget"]).values
            med_multiple = float(np.median(multiples))
        else:
            med_multiple = 0.0
    else:
        med_multiple = 0.0

    kpi_cards = [
        {
            "title": "Total Movies",
            "value": f"{total_movies:,}",
            "subtext": "Training dataset sample",
            "subtext_neutral": True,
            "accent_color": COLOR_CYAN
        },
        {
            "title": "Average Worldwide Revenue",
            "value": format_currency_compact(avg_revenue),
            "subtext": f"Exact: ${avg_revenue:,.0f}" if avg_revenue > 0 else "No data",
            "subtext_neutral": True,
            "accent_color": COLOR_BLUE
        },
        {
            "title": "Median Worldwide Revenue",
            "value": format_currency_compact(med_revenue),
            "subtext": f"Exact: ${med_revenue:,.0f}" if med_revenue > 0 else "No data",
            "subtext_neutral": True,
            "accent_color": COLOR_VIOLET
        },
        {
            "title": "Median Movie Budget",
            "value": format_currency_compact(med_budget),
            "subtext": f"Reported non-zero budgets ({len(valid_bud):,} films)",
            "subtext_neutral": True,
            "accent_color": COLOR_SUCCESS
        },
        {
            "title": "Median Gross Multiple",
            "value": f"{med_multiple:.2f}x" if med_multiple > 0 else "N/A",
            "subtext": "Gross Multiple (Not Profit or ROI)",
            "subtext_neutral": True,
            "accent_color": COLOR_CYAN
        }
    ]

    render_kpi_row(kpi_cards)

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    # 2. Essential Visualizations Grid
    # Row 1: Revenue Distribution & Budget vs Revenue
    col1, col2 = st.columns([1, 1], gap="medium")

    with col1:
        render_section_header("Revenue Distribution", "Distribution of worldwide box office receipts")
        log_toggle = st.toggle("Logarithmic Scale", value=True, key="overview_rev_dist_log")
        fig_rev = plot_revenue_distribution(df, log_scale=log_toggle)
        st.plotly_chart(fig_rev, use_container_width=True)
        st.caption(f"Worldwide theatrical distribution exhibits extreme positive skewness (median: {format_currency_compact(med_revenue)}).")

    with col2:
        render_section_header("Budget vs Revenue", "Scatter relationship on logarithmic scales with 1.0x parity line")
        st.markdown("<div style='height: 38px;'></div>", unsafe_allow_html=True)
        fig_scatter = plot_budget_vs_revenue(df)
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.caption("Points above the dashed amber reference line achieved gross theatrical revenue exceeding production budget.")

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # Row 2: Revenue by Genre & Movie Budget Distribution
    col3, col4 = st.columns([1, 1], gap="medium")

    with col3:
        render_section_header("Revenue by Genre", "Median box-office across major film genres")
        genre_stats = compute_genre_aggregates(df)
        fig_genre = plot_revenue_by_genre(genre_stats)
        st.plotly_chart(fig_genre, use_container_width=True)
        st.caption("Median worldwide box-office receipts aggregated across major genres with at least 10 titles.")

    with col4:
        render_section_header("Movie Budget Distribution", "Reported production budgets excluding unrecorded entries")
        fig_budget = plot_budget_distribution(df)
        st.plotly_chart(fig_budget, use_container_width=True)
        st.caption(f"Reported production budgets distribution (median: {format_currency_compact(med_budget)}).")

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # Row 3: Revenue Trends Over Time
    render_section_header("Revenue Trends Over Time", "Longitudinal evolution of median and mean box office since 1970")
    yearly_stats = compute_yearly_aggregates(df)
    fig_time = plot_revenue_trend_over_time(yearly_stats)
    st.plotly_chart(fig_time, use_container_width=True)
    st.caption("Historical evolution of median and mean worldwide box office across release years.")
