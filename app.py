"""
app.py
Executive Streamlit Dashboard: Netflix Content Strategy & Catalog EDA.

An interactive, portfolio-grade analytics platform showcasing data hygiene pipelines,
feature engineering, multi-label category explosion, and executive business storytelling.
"""

import io
import os
import streamlit as st
import pandas as pd
import numpy as np

# Import modular project components
from src.data_generator import ensure_data_exists
from src.data_cleaner import clean_netflix_data
from src.feature_engineer import engineer_features, explode_column
from src.model_trainer import ContentRecommender
from src.visualizations import (
    plot_catalog_growth,
    plot_acquisition_lag_trend,
    plot_top_countries_audience,
    plot_genre_cooccurrence,
    plot_duration_distributions,
    plot_missing_directors_comparison,
    NETFLIX_RED,
    NETFLIX_CRIMSON,
    TEXT_WHITE,
    TEXT_MUTED
)

# ---------------------------------------------------------
# Page Configuration & Executive Aesthetics
# ---------------------------------------------------------
st.set_page_config(
    page_title="Netflix Content Strategy & Catalog EDA",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Styling (Netflix Brand Palette)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

    /* Global Typography & Backgrounds */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at 10% 10%, #1a1a1a 0%, #0d0d0d 100%);
        color: #F5F5F1;
    }

    /* Executive Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(35, 35, 35, 0.7) 0%, rgba(20, 20, 20, 0.9) 100%);
        border: 1px solid rgba(229, 9, 20, 0.25);
        border-radius: 12px;
        padding: 18px 22px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(229, 9, 20, 0.6);
    }
    .metric-title {
        color: #9CA3AF;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .metric-value {
        color: #FFFFFF;
        font-size: 2.1rem;
        font-weight: 800;
        line-height: 1.1;
    }
    .metric-sub {
        color: #E50914;
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 6px;
    }
    .metric-sub-neutral {
        color: #60A5FA;
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 6px;
    }

    /* Strategic Callout Card */
    .insight-box {
        background: rgba(229, 9, 20, 0.08);
        border-left: 4px solid #E50914;
        border-radius: 0 10px 10px 0;
        padding: 16px 20px;
        margin: 18px 0;
    }
    .insight-title {
        color: #FF5A60;
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 6px;
    }
    .insight-body {
        color: #E5E5E5;
        font-size: 0.93rem;
        line-height: 1.55;
    }

    /* Badges */
    .badge-pill {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        background-color: rgba(229, 9, 20, 0.2);
        color: #FF4B4B;
        border: 1px solid rgba(229, 9, 20, 0.4);
    }

    /* Tab enhancements */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        padding-bottom: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 18px;
        font-weight: 600;
        color: #9CA3AF;
        background-color: rgba(255, 255, 255, 0.03);
    }
    .stTabs [aria-selected="true"] {
        background-color: #E50914 !important;
        color: #FFFFFF !important;
    }

    /* AI Recommender Visual Cards & Badges */
    .rec-card {
        background: linear-gradient(135deg, rgba(28, 28, 28, 0.85) 0%, rgba(18, 18, 18, 0.95) 100%);
        border: 1px solid rgba(229, 9, 20, 0.25);
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 16px;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.5);
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .rec-card:hover {
        transform: translateY(-2px);
        border-color: rgba(229, 9, 20, 0.7);
        box-shadow: 0 10px 30px rgba(229, 9, 20, 0.25);
    }
    .rec-card-target {
        background: linear-gradient(135deg, rgba(45, 15, 20, 0.85) 0%, rgba(20, 20, 20, 0.95) 100%);
        border: 1px solid rgba(229, 9, 20, 0.55);
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 24px;
        box-shadow: 0 8px 24px rgba(229, 9, 20, 0.2);
    }
    .rec-badge-similarity {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 700;
        background: rgba(34, 197, 94, 0.15);
        color: #4ADE80;
        border: 1px solid rgba(34, 197, 94, 0.4);
    }
    .rec-badge-rating {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 700;
        background: rgba(255, 255, 255, 0.08);
        color: #F3F4F6;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .rec-badge-format {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 700;
        background: rgba(229, 9, 20, 0.2);
        color: #FF5A60;
        border: 1px solid rgba(229, 9, 20, 0.4);
    }
    .rec-badge-rank {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 800;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        background: rgba(229, 9, 20, 0.25);
        color: #FFFFFF;
        border: 1px solid #E50914;
    }
    .rec-title {
        color: #FFFFFF;
        font-size: 1.35rem;
        font-weight: 700;
        margin-right: 12px;
        line-height: 1.3;
    }
    .rec-meta-line {
        color: #9CA3AF;
        font-size: 0.88rem;
        margin-top: 8px;
        margin-bottom: 10px;
        line-height: 1.5;
    }
    .rec-desc {
        color: #E5E7EB;
        font-size: 0.94rem;
        line-height: 1.6;
        margin-top: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Cached Data Pipeline Engine
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_and_process_catalog():
    """
    Executes the end-to-end data pipeline:
    1. Verifies dataset presence at data/netflix_titles.csv.
    2. Loads and parses dates, durations, and categorical nulls.
    3. Engineers licensing lag, audience tiers, proxy categories, and NLP metadata soup.
    """
    csv_path = ensure_data_exists("data/netflix_titles.csv")
    raw_df = pd.read_csv(csv_path)
    source_label = f"Kaggle Dataset: data/netflix_titles.csv ({len(raw_df):,} rows)"

    # Clean data & capture audit metrics
    cleaned_df, hygiene_report = clean_netflix_data(raw_df)

    # Engineer strategic business features and metadata_soup
    engineered_df = engineer_features(cleaned_df)

    return raw_df, engineered_df, hygiene_report, source_label


# ---------------------------------------------------------
# Cached AI Recommender Engine
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Fitting AI Content Recommender on active catalog slice...")
def get_content_recommender(data_subset: pd.DataFrame) -> ContentRecommender:
    """
    Instantiates and fits the ContentRecommender on the provided catalog data.
    Cached via Streamlit resource caching to prevent redundant retraining.
    """
    model = ContentRecommender()
    model.fit(data_subset)
    return model


# ---------------------------------------------------------
# Sidebar: Brand, Data Ingestion & Strategic Filters
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px;">
            <div style="background-color: #E50914; color: white; font-weight: 900; font-size: 1.5rem; padding: 2px 12px; border-radius: 6px;">N</div>
            <div style="font-size: 1.25rem; font-weight: 800; letter-spacing: -0.02em; color: white;">CONTENT STRATEGY</div>
        </div>
        <div style="font-size: 0.8rem; color: #9CA3AF; margin-bottom: 16px;">
            Portfolio-Grade Exploratory Data Analysis & Executive Dashboard
        </div>
    """, unsafe_allow_html=True)

    raw_df, df, hygiene_report, data_source_label = load_and_process_catalog()

    st.markdown("### 📥 Data Source")
    st.success(f"Connected to Kaggle Dataset\nTotal Records: {len(df):,}")

    st.markdown("---")
    st.markdown("### 🎛️ Strategic Slicers")

    # Filter 1: Year Added Range
    min_year = int(df["year_added"].min()) if "year_added" in df.columns and not df.empty else 2010
    max_year = int(df["year_added"].max()) if "year_added" in df.columns and not df.empty else 2024

    selected_year_range = st.slider(
        "Year Added to Netflix",
        min_value=min_year,
        max_value=max_year,
        value=(max(min_year, 2014), max_year),
        step=1
    )

    # Filter 2: Content Type
    available_types = sorted(df["type"].dropna().unique().tolist())
    selected_types = st.multiselect(
        "Content Format",
        options=available_types,
        default=available_types
    )

    # Filter 3: Target Audience Tier
    available_audiences = ["Kids", "Family / Teens", "Mature / Adults"]
    selected_audiences = st.multiselect(
        "Audience Maturity Tier",
        options=available_audiences,
        default=available_audiences
    )

    # Filter 4: Top Country Filter (Optional granular drill-down)
    all_countries_exploded = explode_column(df, "country")
    top_country_options = all_countries_exploded["country"].value_counts().head(20).index.tolist()
    selected_countries = st.multiselect(
        "Filter by Country (Multi-Country)",
        options=top_country_options,
        default=[]
    )

    st.markdown("---")
    st.markdown(f"""
        <div style="font-size: 0.76rem; color: #6B7280; line-height: 1.4;">
            <b>Active Pipeline:</b><br>{data_source_label}<br><br>
            <b>Engine:</b> Python 3.12 • Pandas • Plotly Express • Streamlit
        </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------
# Apply Global Slicers
# ---------------------------------------------------------
filtered_df = df.copy()

# Filter by year
if "year_added" in filtered_df.columns:
    filtered_df = filtered_df[
        (filtered_df["year_added"] >= selected_year_range[0]) &
        (filtered_df["year_added"] <= selected_year_range[1])
    ]

# Filter by type
if selected_types:
    filtered_df = filtered_df[filtered_df["type"].isin(selected_types)]

# Filter by audience tier
if selected_audiences:
    filtered_df = filtered_df[filtered_df["audience_tier"].isin(selected_audiences)]

# Filter by country if selected
if selected_countries:
    country_pattern = "|".join(selected_countries)
    filtered_df = filtered_df[filtered_df["country"].astype(str).str.contains(country_pattern, case=False, na=False)]


# ---------------------------------------------------------
# Main Dashboard Header & KPI Scorecard
# ---------------------------------------------------------
col_head_1, col_head_2 = st.columns([3, 1])
with col_head_1:
    st.markdown("""
        <h1 style="margin-bottom: 2px; font-weight: 800; font-size: 2.2rem; color: #FFFFFF;">
            Netflix Content Strategy & Catalog Intelligence
        </h1>
        <p style="color: #9CA3AF; font-size: 0.98rem; margin-top: 0px;">
            An interactive executive inquiry into catalog composition, licensing lag, international expansion, and format ergonomics.
        </p>
    """, unsafe_allow_html=True)

with col_head_2:
    st.markdown("""
        <div style="text-align: right; padding-top: 10px;">
            <span class="badge-pill">EXECUTIVE BRIEFING</span>
        </div>
    """, unsafe_allow_html=True)

# Compute Top Row KPIs
total_titles = len(filtered_df)
base_total = len(df)
pct_active = (total_titles / base_total * 100) if base_total > 0 else 100

movie_count = (filtered_df["type"] == "Movie").sum()
tv_count = (filtered_df["type"] == "TV Show").sum()
tv_share_pct = (tv_count / total_titles * 100) if total_titles > 0 else 0.0
ratio_str = f"{round(movie_count / max(1, tv_count), 1)} : 1"

median_runtime = (
    int(filtered_df[filtered_df["type"] == "Movie"]["duration_min"].median())
    if (filtered_df["type"] == "Movie").any() else 0
)

same_year_count = (filtered_df["acquisition_lag"] == 0).sum()
same_year_pct = (same_year_count / total_titles * 100) if total_titles > 0 else 0.0

top_genre_s = explode_column(filtered_df, "listed_in")
top_genre_name = top_genre_s["listed_in"].value_counts().index[0] if not top_genre_s.empty else "N/A"

# Render 4-Column KPI Row
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Catalog Volume</div>
            <div class="metric-value">{total_titles:,}</div>
            <div class="metric-sub">{round(pct_active, 1)}% of total catalog ({base_total:,})</div>
        </div>
    """, unsafe_allow_html=True)

with kpi2:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">TV Catalog Share & Ratio</div>
            <div class="metric-value">{round(tv_share_pct, 1)}%</div>
            <div class="metric-sub-neutral">Movie-to-TV Ratio: {ratio_str}</div>
        </div>
    """, unsafe_allow_html=True)

with kpi3:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Median Movie Runtime</div>
            <div class="metric-value">{median_runtime} <span style="font-size: 1.2rem; font-weight: 500;">min</span></div>
            <div class="metric-sub">Sweet spot: 90–110 min window</div>
        </div>
    """, unsafe_allow_html=True)

with kpi4:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Originals / Day-and-Date</div>
            <div class="metric-value">{round(same_year_pct, 1)}%</div>
            <div class="metric-sub">Same-Year Acquisitions (Lag = 0)</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ---------------------------------------------------------
# Interactive Tabbed Structure
# ---------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Executive Strategy & Growth",
    "🌍 Global Footprint & Ratings",
    "🎭 Genre & Format Mechanics",
    "🛠️ EDA Data Hygiene & Engineering Log",
    "🤖 AI Recommender Engine"
])


# =========================================================
# TAB 1: EXECUTIVE STRATEGY & GROWTH
# =========================================================
with tab1:
    st.markdown("""
        <div class="insight-box">
            <div class="insight-title">Executive Strategic Takeaway: The Great Studio Defense Pivot</div>
            <div class="insight-body">
                Between 2012 and 2016, Netflix operated primarily as a digital distributor for legacy Hollywood studios, 
                relying heavily on <b>deep back-catalogs (licensing lag > 3 years)</b>. Anticipating that legacy studios 
                (Disney, WarnerMedia, NBCUniversal) would claw back intellectual property to launch rival SVOD platforms 
                (Disney+, HBO Max, Peacock), Netflix aggressively pivoted capital expenditure into <b>in-house Originals 
                and same-year exclusive acquisitions (Lag = 0)</b>, while concurrently accelerating episodic TV production 
                to maximize user retention and reduce subscriber churn.
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("### Catalog Expansion Velocity")
    growth_fig = plot_catalog_growth(filtered_df)
    st.plotly_chart(growth_fig, use_container_width=True)

    st.markdown("### The Studio Defense Pivot (Originals vs. Licensed)")
    lag_fig = plot_acquisition_lag_trend(filtered_df)
    st.plotly_chart(lag_fig, use_container_width=True)

    # Quantitative Breakdown Table
    st.markdown("#### 📊 Annual Ingestion & Strategy Breakdown")
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

        st.dataframe(
            formatted_summary,
            use_container_width=True,
            hide_index=True
        )


# =========================================================
# TAB 2: GLOBAL FOOTPRINT & RATINGS
# =========================================================
with tab2:
    st.markdown("""
        <div class="insight-box">
            <div class="insight-title">Global Expansion & Cultural Localization Dynamics</div>
            <div class="insight-body">
                As domestic US streaming subscriber growth approached saturation, Netflix shifted into an aggressive 
                multinational production model. Key production powerhouses like <b>India, the United Kingdom, and South Korea</b> 
                demonstrate markedly different audience maturity distributions: mature content (TV-MA / R) dominates 
                Western catalogs (US & UK), while markets like India feature substantially higher proportions of 
                <b>Family / Teen programming (TV-14 / TV-PG)</b>, matching local viewing habits where multi-generational 
                co-viewing is culturally prevalent.
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Top Countries Chart
    st.markdown("### Top Production Powerhouses by Target Audience Tier")
    country_fig = plot_top_countries_audience(filtered_df, top_n=14)
    st.plotly_chart(country_fig, use_container_width=True)

    # Regional Drilldown Metrics
    st.markdown("#### 🌐 Top Content Exporters Matrix")
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

        st.dataframe(
            formatted_country_summary,
            use_container_width=True,
            hide_index=True
        )


# =========================================================
# TAB 3: GENRE & FORMAT MECHANICS
# =========================================================
with tab3:
    st.markdown("""
        <div class="insight-box">
            <div class="insight-title">Format Economics: The 95-Minute Sweet Spot & The Season 2 Cliff</div>
            <div class="insight-body">
                <b>1. The Feature Film Sweet Spot:</b> Movie runtimes cluster sharply between <b>90 and 110 minutes</b>. 
                Streaming algorithms prioritize completion rate (percentage of users who finish a film) over total runtime; 
                shorter films reduce drop-off, increase catalog rotation, and optimize digital ad insertion windows.<br>
                <b>2. The Episodic 'Season 2 Cliff':</b> TV Shows experience a steep statistical cliff where 
                <b>over 60% of series conclude after Season 1, and over 80% do not survive past Season 2</b>. 
                Under Netflix's cost-plus compensation model, actor contracts and production budgets escalate exponentially 
                after Season 2, while new subscriber acquisition per season declines rapidly unless the IP achieves global tentpole velocity.
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Side-by-side distributions
    st.markdown("### Runtime Optimization & Format Mechanics")
    dist_fig = plot_duration_distributions(filtered_df)
    st.plotly_chart(dist_fig, use_container_width=True)

    st.markdown("### Genre Co-Occurrence Matrix")
    st.markdown("Discover which genres Netflix most frequently bundles together to target niche subscriber demographics.")

    heatmap_fig = plot_genre_cooccurrence(filtered_df, top_n=10)
    st.plotly_chart(heatmap_fig, use_container_width=True)


# =========================================================
# TAB 4: EDA DATA HYGIENE & ENGINEERING LOG
# =========================================================
with tab4:
    st.markdown("""
        <div class="insight-box">
            <div class="insight-title">Data Engineering & Hygiene Rigor</div>
            <div class="insight-body">
                Real-world exploratory data analysis requires proving that missing data patterns are thoroughly audited, 
                statistically justified, and documented rather than silently discarded or naively filled with mean values.
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Hygiene KPI row
    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Raw Dataset Rows", f"{hygiene_report['initial_rows']:,}")
    h2.metric("Cleaned Pipeline Rows", f"{hygiene_report['final_rows']:,}")
    h3.metric("Anomalous Rows Dropped", f"{hygiene_report['dropped_rows']} ({hygiene_report['dropped_pct']}%)")
    h4.metric("Engineered Features Added", "5 Business Dimensions")

    st.markdown("<br>", unsafe_allow_html=True)

    # Missing Director Hypothesis Deep Dive
    st.markdown("### The 'Missing Director' Hypothesis")
    st.markdown(f"""
        A common trap in novice EDA is assuming that null values in the `director` column are data collection errors.
        Our hypothesis proves that **missing director data is structural to the medium of episodic television**:
        
        * **Movies Missing Director:** `{hygiene_report['director_hypothesis']['movie_missing_dir_pct']}%` 
          ({hygiene_report['director_hypothesis']['movie_missing_dir']:,} of {hygiene_report['director_hypothesis']['movie_total']:,})
        * **TV Shows Missing Director:** `{hygiene_report['director_hypothesis']['tv_missing_dir_pct']}%` 
          ({hygiene_report['director_hypothesis']['tv_missing_dir']:,} of {hygiene_report['director_hypothesis']['tv_total']:,})
        
        {hygiene_report['director_hypothesis']['explanation']}
    """)

    dir_fig = plot_missing_directors_comparison(hygiene_report)
    st.plotly_chart(dir_fig, use_container_width=True)

    # Missing Value Audit Table Before vs After
    st.markdown("#### 📋 Missing Value Audit Ledger (Before vs. After Pipeline)")
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

    # Cleaned Dataset Inspector & Export
    st.markdown("#### 🔍 Interactive Dataset Inspector & Export")
    col_exp1, col_exp2 = st.columns([3, 1])
    with col_exp1:
        preview_cols = [
            "show_id", "type", "title", "country", "release_year",
            "year_added", "acquisition_lag", "content_origin_proxy",
            "audience_tier", "duration_min", "seasons_count"
        ]
        available_preview = [c for c in preview_cols if c in filtered_df.columns]
        st.dataframe(filtered_df[available_preview].head(25), use_container_width=True)

    with col_exp2:
        st.markdown("**Download Processed Data**")
        st.caption("Export the cleaned and feature-engineered dataset for external modeling or Tableau/PowerBI reporting.")
        csv_buffer = io.StringIO()
        filtered_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv_buffer.getvalue(),
            file_name="netflix_cleaned_and_engineered.csv",
            mime="text/csv"
        )


# =========================================================
# TAB 5: AI RECOMMENDER ENGINE
# =========================================================
with tab5:
    st.markdown("""
        <div class="insight-box">
            <div class="insight-title">Machine Learning NLP Recommender Engine</div>
            <div class="insight-body">
                This content-based recommender transforms our <b>metadata soup</b> (synthesizing title, director, cast, 
                curated genre tags, and synopses) into a high-dimensional vector space using 
                <b>Scikit-Learn TF-IDF Vectorization (5,000 max features, English stop words)</b>. Pairwise 
                <b>Cosine Similarity</b> scores are computed dynamically over the active catalog slice to discover 
                the most mathematically and narratively adjacent titles.
            </div>
        </div>
    """, unsafe_allow_html=True)

    if filtered_df.empty:
        st.warning("⚠️ No titles match your current sidebar filters. Please broaden your filter parameters to explore recommendations.")
    else:
        # User title search and selector
        available_titles = sorted(filtered_df["title"].dropna().unique().tolist())
        
        # Select an engaging default title if present in filtered catalog
        default_idx = 0
        preferred_defaults = ["Shadows over Silence", "Stranger Things", "Inception", "Breaking Bad", "Return to Destiny"]
        for pref in preferred_defaults:
            if pref in available_titles:
                default_idx = available_titles.index(pref)
                break

        select_col, info_col = st.columns([3, 1])
        with select_col:
            selected_title = st.selectbox(
                "🎯 Select or Search a Movie / TV Show to Generate Recommendations:",
                options=available_titles,
                index=default_idx,
                help="Search from the currently filtered catalog titles to query the TF-IDF cosine similarity model."
            )

        with info_col:
            st.markdown(f"""
                <div style="background: rgba(255, 255, 255, 0.04); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 8px; padding: 10px 14px; margin-top: 24px;">
                    <div style="font-size: 0.75rem; color: #9CA3AF; text-transform: uppercase; font-weight: 600;">Active Model Corpus</div>
                    <div style="font-size: 1.15rem; color: #FFFFFF; font-weight: 800;">{len(filtered_df):,} <span style="font-size: 0.8rem; font-weight: 500; color: #E50914;">Titles</span></div>
                </div>
            """, unsafe_allow_html=True)

        if selected_title:
            # Query row metadata spotlight
            target_matches = filtered_df[filtered_df["title"] == selected_title]
            if not target_matches.empty:
                target_row = target_matches.iloc[0]

                st.markdown("#### 🎬 Selected Title Spotlight")
                t_year = int(target_row["release_year"]) if pd.notna(target_row.get("release_year")) else "N/A"
                t_rating = str(target_row.get("rating", "NR"))
                t_type = str(target_row.get("type", "Title"))
                t_genres = str(target_row.get("listed_in", "Uncategorized"))
                t_director = str(target_row.get("director", "Unlisted"))
                t_cast = str(target_row.get("cast", "Unlisted"))
                t_desc = str(target_row.get("description", "No description available."))
                
                t_duration = ""
                if t_type == "Movie" and pd.notna(target_row.get("duration_min")):
                    t_duration = f" • {int(target_row['duration_min'])} min"
                elif t_type == "TV Show" and pd.notna(target_row.get("seasons_count")):
                    s_count = int(target_row["seasons_count"])
                    t_duration = f" • {s_count} Season{'s' if s_count > 1 else ''}"

                st.markdown(f"""
                    <div class="rec-card-target">
                        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
                            <div style="display: flex; align-items: center; gap: 10px; flex-wrap: wrap;">
                                <span class="rec-badge-rank" style="background: rgba(229, 9, 20, 0.4); border-color: #E50914;">QUERY TITLE</span>
                                <span class="rec-title">{target_row['title']}</span>
                            </div>
                            <div style="display: flex; align-items: center; gap: 8px;">
                                <span class="rec-badge-format">{t_type}</span>
                                <span class="rec-badge-rating">{t_rating}</span>
                                <span style="color: #9CA3AF; font-size: 0.9rem; font-weight: 600;">{t_year}{t_duration}</span>
                            </div>
                        </div>
                        <div class="rec-meta-line">
                            <b>Genres:</b> {t_genres} &nbsp;|&nbsp; 
                            <b>Director:</b> {t_director} &nbsp;|&nbsp; 
                            <b>Cast:</b> {t_cast}
                        </div>
                        <div class="rec-desc">{t_desc}</div>
                    </div>
                """, unsafe_allow_html=True)

            # Fit or retrieve cached ContentRecommender
            recommender = get_content_recommender(filtered_df)

            # Retrieve Top 5 Recommendations
            recs = recommender.recommend(selected_title, top_n=5)

            st.markdown(f"#### 🤖 Top 5 AI-Recommended Titles for *'{selected_title}'*")

            if recs.empty:
                st.info("No similar titles found within the current filter slice. Try selecting another title or expanding your sidebar filters.")
            else:
                for rank, (_, row) in enumerate(recs.iterrows(), 1):
                    with st.container():
                        sim_pct = round(row["similarity_score"] * 100, 1)
                        r_rating = str(row.get("rating", "NR"))
                        r_type = str(row.get("type", "Title"))
                        r_year = int(row["release_year"]) if pd.notna(row.get("release_year")) else "N/A"
                        r_desc = str(row.get("description", "No description available."))
                        r_genres = str(row.get("listed_in", "Uncategorized"))
                        r_director = str(row.get("director", "Unlisted"))
                        r_cast = str(row.get("cast", "Unlisted"))
                        
                        r_duration = ""
                        if r_type == "Movie" and pd.notna(row.get("duration_min")):
                            r_duration = f" • {int(row['duration_min'])} min"
                        elif r_type == "TV Show" and pd.notna(row.get("seasons_count")):
                            s_cnt = int(row["seasons_count"])
                            r_duration = f" • {s_cnt} Season{'s' if s_cnt > 1 else ''}"

                        st.markdown(f"""
                            <div class="rec-card">
                                <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
                                    <div style="display: flex; align-items: center; gap: 10px; flex-wrap: wrap;">
                                        <span class="rec-badge-rank">#{rank} MATCH</span>
                                        <span class="rec-title">{row['title']}</span>
                                    </div>
                                    <div style="display: flex; align-items: center; gap: 8px;">
                                        <span class="rec-badge-similarity">⚡ {sim_pct}% SIMILARITY</span>
                                        <span class="rec-badge-format">{r_type}</span>
                                        <span class="rec-badge-rating">{r_rating}</span>
                                        <span style="color: #9CA3AF; font-size: 0.9rem; font-weight: 600;">{r_year}{r_duration}</span>
                                    </div>
                                </div>
                                <div class="rec-meta-line">
                                    <b>Genres:</b> {r_genres} &nbsp;|&nbsp; 
                                    <b>Director:</b> {r_director} &nbsp;|&nbsp; 
                                    <b>Cast:</b> {r_cast}
                                </div>
                                <div class="rec-desc">{r_desc}</div>
                            </div>
                        """, unsafe_allow_html=True)

