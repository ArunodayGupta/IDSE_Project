"""
app.py
Executive Streamlit Dashboard: Netflix Content Strategy & Catalog Intelligence.

An interactive analytics platform showcasing data hygiene pipelines,
feature engineering, multi-label category analysis, and corporate business reporting.
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
from src.explorer_ui import render_explorer_tab
from src.cleaning_lab_ui import render_cleaning_lab_tab
from src.model_lab_ui import render_model_lab_tab
from src.visualizations import (
    plot_catalog_growth,
    plot_acquisition_lag_trend,
    plot_top_countries_audience,
    plot_genre_cooccurrence,
    plot_duration_distributions,
    plot_missing_directors_comparison,
    PRIMARY_BLUE,
    LIGHT_BLUE,
    SLATE_GREY,
    DARK_SLATE
)

# ---------------------------------------------------------
# Page Configuration & Corporate Aesthetics
# ---------------------------------------------------------
st.set_page_config(
    page_title="Netflix Content Strategy & Catalog Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Corporate Styling (Light Blue, White, and Grey Palette)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global Typography & Light Theme Backgrounds */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .stApp {
        background-color: #F8F9FA;
        color: #2B3035;
    }

    /* Sidebar Styling & Text Clarity */
    [data-testid="stSidebar"] {
        background-color: #E9ECEF;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span {
        color: #2B3035 !important;
    }
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stMultiSelect label {
        color: #2B3035 !important;
        font-weight: 600;
    }

    /* Executive Metric Cards (Elevated Left-Accent Design) */
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 5px solid #2C7BE5;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        transition: box-shadow 0.2s ease;
    }
    .metric-card:hover {
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.1);
    }
    .metric-title {
        color: #6C757D;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .metric-value {
        color: #2B3035;
        font-size: 2.2rem;
        font-weight: 700;
        line-height: 1.1;
    }
    .metric-sub {
        color: #2C7BE5;
        font-size: 0.85rem;
        font-weight: 500;
        margin-top: 6px;
    }
    .metric-sub-neutral {
        color: #6C757D;
        font-size: 0.85rem;
        font-weight: 500;
        margin-top: 6px;
    }

    /* Strategic Callout Card (Elevated Left-Accent Design) */
    .insight-box {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 5px solid #6EA8FE;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.04);
        margin: 20px 0;
    }
    .insight-title {
        color: #2C7BE5;
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .insight-body {
        color: #495057;
        font-size: 0.95rem;
        line-height: 1.6;
    }

    /* Badges */
    .badge-pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        background-color: #E7F1FF;
        color: #2C7BE5;
        border: 1px solid #B6D4FE;
    }

    /* Tab enhancements & Visibility Fix */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #DEE2E6;
        padding-bottom: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #495057 !important;
        background-color: transparent;
        border-radius: 6px;
        padding: 8px 18px;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab"] p,
    .stTabs [data-baseweb="tab"] div,
    .stTabs [data-baseweb="tab"] span {
        color: inherit !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2C7BE5 !important;
        color: #FFFFFF !important;
        border-radius: 6px;
    }
    .stTabs [aria-selected="true"] p,
    .stTabs [aria-selected="true"] div,
    .stTabs [aria-selected="true"] span {
        color: #FFFFFF !important;
    }

    /* Corporate Recommender Visual Cards & Badges */
    .rec-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 5px solid #6EA8FE;
        border-radius: 8px;
        padding: 20px 24px;
        margin-bottom: 16px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.04);
        transition: box-shadow 0.2s ease, border-left-color 0.2s ease;
    }
    .rec-card:hover {
        border-left-color: #2C7BE5;
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.08);
    }
    .rec-card-target {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 5px solid #2C7BE5;
        border-radius: 8px;
        padding: 20px 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.04);
    }
    .rec-badge-similarity {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 4px;
        font-size: 0.82rem;
        font-weight: 600;
        background-color: #EBF5FF;
        color: #2C7BE5;
        border: 1px solid #B6D4FE;
    }
    .rec-badge-rating {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 0.78rem;
        font-weight: 600;
        background-color: #F8F9FA;
        color: #495057;
        border: 1px solid #DEE2E6;
    }
    .rec-badge-format {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 0.78rem;
        font-weight: 600;
        background-color: #E7F1FF;
        color: #2C7BE5;
        border: 1px solid #B6D4FE;
    }
    .rec-badge-rank {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        background-color: #2C7BE5;
        color: #FFFFFF;
        border: 1px solid #2C7BE5;
    }
    .rec-title {
        color: #2B3035;
        font-size: 1.25rem;
        font-weight: 700;
        margin-right: 12px;
        line-height: 1.3;
    }
    .rec-meta-line {
        color: #6C757D;
        font-size: 0.88rem;
        margin-top: 8px;
        margin-bottom: 10px;
        line-height: 1.5;
    }
    .rec-desc {
        color: #495057;
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
    source_label = f"Dataset: data/netflix_titles.csv ({len(raw_df):,} rows)"

    # Clean data & capture audit metrics
    cleaned_df, hygiene_report = clean_netflix_data(raw_df)

    # Engineer strategic business features and metadata_soup
    engineered_df = engineer_features(cleaned_df)

    return raw_df, engineered_df, hygiene_report, source_label


# ---------------------------------------------------------
# Cached Content Recommender Engine
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Fitting Content Recommender on active catalog slice...")
def get_content_recommender(data_subset: pd.DataFrame) -> ContentRecommender:
    """
    Instantiates and fits the ContentRecommender on the provided catalog data.
    Cached via Streamlit resource caching to prevent redundant retraining.
    """
    model = ContentRecommender()
    model.fit(data_subset)
    return model


# ---------------------------------------------------------
# Sidebar: Navigation, Data Ingestion & Strategic Filters
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px;">
            <div style="background-color: #2C7BE5; color: white; font-weight: 700; font-size: 1.25rem; padding: 4px 10px; border-radius: 4px;">BI</div>
            <div style="font-size: 1.15rem; font-weight: 700; letter-spacing: -0.01em; color: #2B3035;">CATALOG ANALYTICS</div>
        </div>
        <div style="font-size: 0.82rem; color: #6C757D; margin-bottom: 16px;">
            Exploratory Data Analysis & Executive Dashboard
        </div>
    """, unsafe_allow_html=True)

    raw_df, df, hygiene_report, data_source_label = load_and_process_catalog()

    st.markdown("### Data Source")
    st.success(f"Connected to Dataset\nTotal Records: {len(df):,}")

    st.markdown("---")
    st.markdown("### Netflix Strategic Slicers")
    st.caption("Filters below slice the Netflix Executive, Global, Format, and Recommender tabs. Use the DataLab tab for universal dataset exploration.")


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
        <div style="font-size: 0.76rem; color: #6C757D; line-height: 1.4;">
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
        <h1 style="margin-bottom: 4px; font-weight: 700; font-size: 2.1rem; color: #2B3035;">
            Content Strategy & Catalog Intelligence
        </h1>
        <p style="color: #6C757D; font-size: 0.98rem; margin-top: 0px;">
            Executive analysis of catalog composition, licensing timelines, international distribution, and format specifications.
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
            <div class="metric-sub">Typical range: 90–110 minutes</div>
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
tab_explore, tab_clean, tab_model, tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔬 DataLab: Explore Dataset",
    "🧹 DataLab: Clean & Transform",
    "🤖 DataLab: Model Lab",
    "📈 Executive Strategy & Growth",
    "🌍 Global Distribution & Ratings",
    "🎭 Genre & Format Analysis",
    "🛠️ Data Quality & Pipeline Log",
    "🎯 Content Recommendation Engine"
])


# =========================================================
# TAB: DATALAB INTERACTIVE DATASET EXPLORER (PHASE 1)
# =========================================================
with tab_explore:
    render_explorer_tab(netflix_raw_df=raw_df)


# =========================================================
# TAB: DATALAB CLEAN & TRANSFORM LAB (PHASE 2)
# =========================================================
with tab_clean:
    render_cleaning_lab_tab(netflix_raw_df=raw_df)


# =========================================================
# TAB: DATALAB MODEL LAB (PHASE 3)
# =========================================================
with tab_model:
    render_model_lab_tab(netflix_raw_df=raw_df)


# =========================================================
# TAB 1: EXECUTIVE STRATEGY & GROWTH
# =========================================================
with tab1:
    st.markdown("""
        <div class="insight-box">
            <div class="insight-title">Strategic Overview: Studio Licensing Transition</div>
            <div class="insight-body">
                Between 2012 and 2016, distribution strategy relied heavily on
                <b>deep back-catalogs (licensing lag > 3 years)</b> from legacy studios.
                In response to shifting studio licensing models and the rise of competitor platforms,
                capital allocation shifted toward <b>in-house Originals and same-year exclusive
                acquisitions (Lag = 0)</b>, paired with an increased volume of episodic television
                to support subscriber engagement and retention.
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("### Catalog Growth Trends")
    growth_fig = plot_catalog_growth(filtered_df)
    st.plotly_chart(growth_fig, use_container_width=True)

    st.markdown("### Licensing Strategy Evolution (Originals vs. Licensed)")
    lag_fig = plot_acquisition_lag_trend(filtered_df)
    st.plotly_chart(lag_fig, use_container_width=True)

    # Quantitative Breakdown Table
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

        st.dataframe(
            formatted_summary,
            use_container_width=True,
            hide_index=True
        )


# =========================================================
# TAB 2: GLOBAL DISTRIBUTION & RATINGS
# =========================================================
with tab2:
    st.markdown("""
        <div class="insight-box">
            <div class="insight-title">Strategic Overview: International Content Distribution</div>
            <div class="insight-body">
                To sustain subscriber growth beyond mature domestic markets, content development
                expanded into key international production regions including <b>India, the United Kingdom,
                and South Korea</b>. These markets exhibit distinct audience maturity profiles: mature
                programming (TV-MA / R) forms a substantial portion of Western catalogs, while regions
                such as India emphasize <b>Family and Teen programming (TV-14 / TV-PG)</b> aligned with
                multi-generational co-viewing patterns.
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Top Countries Chart
    st.markdown("### Top Producing Countries by Target Audience Tier")
    country_fig = plot_top_countries_audience(filtered_df, top_n=14)
    st.plotly_chart(country_fig, use_container_width=True)

    # Regional Drilldown Metrics
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

        st.dataframe(
            formatted_country_summary,
            use_container_width=True,
            hide_index=True
        )


# =========================================================
# TAB 3: GENRE & FORMAT ANALYSIS
# =========================================================
with tab3:
    st.markdown("""
        <div class="insight-box">
            <div class="insight-title">Format Analysis: Runtime Distribution & Season Retention</div>
            <div class="insight-body">
                <b>1. Feature Film Duration:</b> Movie runtimes cluster consistently between <b>90 and 110 minutes</b>,
                balancing narrative completeness with viewer completion rates and programmatic scheduling efficiency.<br>
                <b>2. Episodic Season Attrition:</b> Television series exhibit a pronounced drop-off rate, where
                the majority of titles conclude within one to two seasons. Multi-season renewal decisions are heavily
                driven by cumulative viewership metrics and cost-per-subscriber acquisition thresholds.
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Side-by-side distributions
    st.markdown("### Runtime & Season Distribution Analysis")
    dist_fig = plot_duration_distributions(filtered_df)
    st.plotly_chart(dist_fig, use_container_width=True)

    st.markdown("### Genre Co-Occurrence Matrix")
    st.markdown("Analysis of paired genres across catalog titles.")

    heatmap_fig = plot_genre_cooccurrence(filtered_df, top_n=10)
    st.plotly_chart(heatmap_fig, use_container_width=True)


# =========================================================
# TAB 4: DATA QUALITY & PIPELINE LOG
# =========================================================
with tab4:
    st.markdown("""
        <div class="insight-box">
            <div class="insight-title">Data Pipeline Audit & Quality Assurance</div>
            <div class="insight-body">
                Comprehensive data profiling ensures all missing fields, anomalies, and structural nulls
                are methodically audited, verified, and documented prior to downstream analytics and reporting.
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

    # Missing Director Field Analysis
    st.markdown("### Missing Director Field Analysis")
    st.markdown(f"""
        Analysis of missing director entries reveals that missing values are largely structural to the format of episodic television rather than data collection discrepancies:

        * **Movies Missing Director:** `{hygiene_report['director_hypothesis']['movie_missing_dir_pct']}%`
          ({hygiene_report['director_hypothesis']['movie_missing_dir']:,} of {hygiene_report['director_hypothesis']['movie_total']:,})
        * **TV Shows Missing Director:** `{hygiene_report['director_hypothesis']['tv_missing_dir_pct']}%`
          ({hygiene_report['director_hypothesis']['tv_missing_dir']:,} of {hygiene_report['director_hypothesis']['tv_total']:,})

        {hygiene_report['director_hypothesis']['explanation']}
    """)

    dir_fig = plot_missing_directors_comparison(hygiene_report)
    st.plotly_chart(dir_fig, use_container_width=True)

    # Missing Value Audit Table Before vs After
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

    # Cleaned Dataset Inspector & Export
    st.markdown("#### Dataset Preview and Export")
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
            label="Download CSV",
            data=csv_buffer.getvalue(),
            file_name="netflix_cleaned_and_engineered.csv",
            mime="text/csv"
        )


# =========================================================
# TAB 5: CONTENT RECOMMENDATION ENGINE
# =========================================================
with tab5:
    st.markdown("""
        <div class="insight-box">
            <div class="insight-title">Content Similarity Recommendation Model</div>
            <div class="insight-body">
                This content-based recommender transforms title metadata (synthesizing title, director, cast,
                curated genre tags, and synopses) into a high-dimensional vector space using
                <b>Scikit-Learn TF-IDF Vectorization (5,000 max features, English stop words)</b>. Pairwise
                <b>Cosine Similarity</b> scores are computed dynamically over the active catalog slice to discover
                the most mathematically and thematically adjacent titles.
            </div>
        </div>
    """, unsafe_allow_html=True)

    if filtered_df.empty:
        st.warning("No titles match your current sidebar filters. Please expand your filter selections.")
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
                "Select or search a title to generate recommendations:",
                options=available_titles,
                index=default_idx,
                help="Search from the currently filtered catalog titles to query the TF-IDF cosine similarity model."
            )

        with info_col:
            st.markdown(f"""
                <div style="background: #FFFFFF; border: 1px solid #DEE2E6; border-radius: 8px; padding: 10px 14px; margin-top: 24px;">
                    <div style="font-size: 0.75rem; color: #6C757D; text-transform: uppercase; font-weight: 600;">Active Model Corpus</div>
                    <div style="font-size: 1.15rem; color: #2B3035; font-weight: 700;">{len(filtered_df):,} <span style="font-size: 0.8rem; font-weight: 500; color: #2C7BE5;">Titles</span></div>
                </div>
            """, unsafe_allow_html=True)

        if selected_title:
            # Query row metadata spotlight
            target_matches = filtered_df[filtered_df["title"] == selected_title]
            if not target_matches.empty:
                target_row = target_matches.iloc[0]

                st.markdown("#### Selected Title Profile")
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
                                <span class="rec-badge-rank" style="background: #E7F1FF; color: #2C7BE5; border: 1px solid #B6D4FE;">QUERY TITLE</span>
                                <span class="rec-title">{target_row['title']}</span>
                            </div>
                            <div style="display: flex; align-items: center; gap: 8px;">
                                <span class="rec-badge-format">{t_type}</span>
                                <span class="rec-badge-rating">{t_rating}</span>
                                <span style="color: #6C757D; font-size: 0.9rem; font-weight: 600;">{t_year}{t_duration}</span>
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

            st.markdown(f"#### Recommended Titles for *'{selected_title}'*")

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
                                        <span class="rec-badge-rank">MATCH #{rank}</span>
                                        <span class="rec-title">{row['title']}</span>
                                    </div>
                                    <div style="display: flex; align-items: center; gap: 8px;">
                                        <span class="rec-badge-similarity">{sim_pct}% MATCH</span>
                                        <span class="rec-badge-format">{r_type}</span>
                                        <span class="rec-badge-rating">{r_rating}</span>
                                        <span style="color: #6C757D; font-size: 0.9rem; font-weight: 600;">{r_year}{r_duration}</span>
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
