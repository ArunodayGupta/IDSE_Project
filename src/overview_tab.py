"""
overview_tab.py
Section A: Overview of the TMDB Movie Box Office Revenue dataset.

Presents:
1. Executive KPIs and core interactive Plotly visualizations.
2. Understanding the Dataset: Dimensions, partition structure, and prediction scope.
3. Raw Dataset Column Dictionary: Explanations of all 23 original attributes.
4. How We Transform Movie Data into 37 Predictive Features: 8 domain dimensions.
5. Feature Transformation Mapping: End-to-end table from raw columns to features.
6. Understanding Our Prediction Target: Worldwide Box Office Revenue and RMSLE.
7. Why Are Some Dataset Columns Not Used?: Pre-release scope and leakage prevention.
8. Complete Machine Learning Workflow: End-to-end pipeline architecture.

Strictly adheres to professional styling with zero emojis.
"""

from typing import Dict, Any, List, Optional
import streamlit as st
import pandas as pd
import numpy as np

from src.data_loader import safe_parse_json, parse_release_date, load_tmdb_test_data
from src.feature_engineering import ALL_FEATURE_COLUMNS, CORE_GENRES, MAJOR_STUDIOS
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
    render_callout,
    COLOR_CYAN,
    COLOR_BLUE,
    COLOR_VIOLET,
    COLOR_SUCCESS,
    COLOR_WARNING,
    COLOR_INFO,
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
        subtitle="Worldwide movie box-office revenue trends, dataset architecture, and predictive feature engineering"
    )

    # -------------------------------------------------------------------------
    # 1. Executive KPIs
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # 2. Essential Visualizations Grid
    # -------------------------------------------------------------------------
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

    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 3. Understanding the Dataset
    # -------------------------------------------------------------------------
    render_section_header(
        "Understanding the Dataset",
        "Introduction to the TMDB Movie Box Office dataset, dimensions, and partition structure"
    )

    st.markdown(
        "Our project uses the **TMDB Movie Box Office Prediction dataset** to predict the worldwide theatrical "
        "revenue of movies using historical movie metadata. The dataset contains rich information about movie budgets, "
        "genres, production companies, cast, crew, release dates, runtime, and other pre-release characteristics."
    )

    # Dynamically resolve dataset dimensions
    test_df, _ = load_tmdb_test_data()
    n_train_movies = len(df)
    n_test_movies = len(test_df) if test_df is not None else 4398
    n_train_cols = df.shape[1]
    n_engineered = len(ALL_FEATURE_COLUMNS)

    dataset_summary_cards = [
        {
            "title": "Training Set",
            "value": f"{n_train_movies:,}",
            "subtext": "Historical labeled movies",
            "subtext_neutral": True,
            "accent_color": COLOR_CYAN
        },
        {
            "title": "Testing Set",
            "value": f"{n_test_movies:,}",
            "subtext": "Unlabeled competition movies",
            "subtext_neutral": True,
            "accent_color": COLOR_BLUE
        },
        {
            "title": "Raw Attributes",
            "value": f"{n_train_cols}",
            "subtext": "Original training columns (22 in test)",
            "subtext_neutral": True,
            "accent_color": COLOR_VIOLET
        },
        {
            "title": "Engineered Features",
            "value": f"{n_engineered}",
            "subtext": "Extracted pre-release predictors",
            "subtext_neutral": True,
            "accent_color": COLOR_SUCCESS
        },
        {
            "title": "Prediction Target",
            "value": "Revenue (USD)",
            "subtext": "Worldwide box office receipts",
            "subtext_neutral": True,
            "accent_color": COLOR_CYAN
        }
    ]
    render_kpi_row(dataset_summary_cards)

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    st.markdown("""
        <div style="background: rgba(20, 28, 46, 0.7); border: 1px solid #1E2B44; border-radius: 8px; padding: 14px 18px; line-height: 1.55; font-size: 0.88rem; color: #CBD5E1;">
            <b>Dataset Partition Roles:</b><br>
            • <b>Training Dataset (train.csv):</b> Contains complete movie metadata alongside the true target variable (<code>revenue</code>). Used to perform exploratory data analysis, establish imputation baselines, cross-validate models, and tune hyperparameters.<br>
            • <b>Testing Dataset (test.csv):</b> Contains identical pre-release movie metadata but excludes the <code>revenue</code> target column. Used to evaluate generalization on unseen future movies and generate competition submission forecasts.
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 4. Raw Dataset Column Dictionary
    # -------------------------------------------------------------------------
    render_section_header(
        "Understanding the Original Dataset Columns",
        "Comprehensive dictionary of all 23 original attributes across five functional categories"
    )

    st.markdown(
        "The original TMDB dataset contains 23 raw columns (22 in the test set). To prevent cognitive overload, "
        "these columns are categorized below with clear indicators of whether each field is "
        "<span style='color: #10B981; font-weight: 600;'>Transformed into Predictive Features</span>, "
        "<span style='color: #00D2FF; font-weight: 600;'>Target Variable</span>, or "
        "<span style='color: #94A3B8; font-weight: 600;'>Excluded from Model</span>.",
        unsafe_allow_html=True
    )

    # Category 1: Financial & Target Attributes
    with st.expander("1. Financial & Prediction Target Attributes (2 Columns)", expanded=True):
        st.markdown("""
| Column | Description | Real Example from Dataset | Modeling Status |
| :--- | :--- | :--- | :--- |
| `revenue` | Actual worldwide theatrical box-office revenue in USD. | `12314651` ($12.3M USD) | **Target Variable** (Log-transformed during training) |
| `budget` | Production budget of the movie in USD. Contains zeroes for unrecorded entries. | `14000000` ($14.0M USD) | **Transformed into Features** (`budget`, `log_budget`, `budget_is_zero`) |
        """)

    # Category 2: Creative Content & Textual Metadata
    with st.expander("2. Creative Content & Marketing Metadata (6 Columns)", expanded=False):
        st.markdown("""
| Column | Description | Real Example from Dataset | Modeling Status |
| :--- | :--- | :--- | :--- |
| `genres` | Nested JSON list of movie genres (e.g. Comedy, Action, Drama). | `[{'id': 35, 'name': 'Comedy'}]` | **Transformed into Features** (`genre_count`, `primary_genre`, 14 genre binary flags) |
| `homepage` | Official marketing or promotional website URL. | `'http://sonyclassics.com/whiplash/'` | **Transformed into Features** (`has_homepage` binary flag) |
| `tagline` | Promotional marketing tagline or catchphrase. | `'The Laws of Space and Time are About to be Violated.'` | **Transformed into Features** (`has_tagline` binary flag) |
| `overview` | Unstructured plot synopsis or storyline narrative. | `'When Lou, who has become the "father of the Internet,"...'` | **Excluded from Model** (Unstructured text; NLP out of tabular scope) |
| `Keywords` | Nested JSON list of thematic plot keyword tags. | `[{'id': 4379, 'name': 'time travel'}]` | **Excluded from Model** (High-cardinality sparse tags; risk of overfitting) |
| `poster_path` | Relative file path / hash of the movie poster image on TMDB CDN. | `'/tQtWuwvMf0hCc2QR2tkolwl7c3c.jpg'` | **Excluded from Model** (Image reference; computer vision out of scope) |
        """)

    # Category 3: Temporal & Linguistic Attributes
    with st.expander("3. Temporal & Linguistic Attributes (4 Columns)", expanded=False):
        st.markdown("""
| Column | Description | Real Example from Dataset | Modeling Status |
| :--- | :--- | :--- | :--- |
| `release_date` | Theatrical release date (M/D/YY or YYYY-MM-DD). | `'2/20/15'` (February 20, 2015) | **Transformed into Features** (5 temporal features; century rollover adjusted) |
| `runtime` | Theatrical runtime duration in minutes. | `93.0` minutes | **Transformed into Features** (`runtime`, imputed for missing/zero values) |
| `original_language` | Primary two-letter ISO 639-1 language code. | `'en'` | **Transformed into Features** (`original_language`, `is_english`) |
| `spoken_languages` | Nested JSON list of all spoken dialogue languages. | `[{'iso_639_1': 'en', 'name': 'English'}]` | **Transformed into Features** (`language_count`) |
        """)

    # Category 4: Cast, Crew & Production Entities
    with st.expander("4. Cast, Crew & Production Entities (5 Columns)", expanded=False):
        st.markdown("""
| Column | Description | Real Example from Dataset | Modeling Status |
| :--- | :--- | :--- | :--- |
| `cast` | Nested JSON list of credited actors, characters, and billing order. | `[{'cast_id': 4, 'character': 'Lou', 'name': 'Rob Corddry'}]` | **Transformed into Features** (`cast_size` total cast count) |
| `crew` | Nested JSON list of directors, writers, producers, and crew members. | `[{'job': 'Director', 'name': 'Steve Pink'}]` | **Transformed into Features** (`crew_size` total crew count) |
| `production_companies` | Nested JSON list of studios and production companies. | `[{'name': 'Paramount Pictures', 'id': 4}]` | **Transformed into Features** (`company_count`, `is_major_studio`) |
| `production_countries` | Nested JSON list of sovereign countries involved in production. | `[{'iso_3166_1': 'US', 'name': 'United States'}]` | **Transformed into Features** (`country_count`, `is_us_country`) |
| `belongs_to_collection` | Nested JSON dictionary with movie franchise details. | `[{'id': 313576, 'name': 'Hot Tub Time Machine Collection'}]` | **Transformed into Features** (`has_collection` binary franchise indicator) |
        """)

    # Category 5: Identifiers & Administrative Attributes
    with st.expander("5. Identifiers & Administrative Attributes (6 Columns)", expanded=False):
        st.markdown("""
| Column | Description | Real Example from Dataset | Modeling Status |
| :--- | :--- | :--- | :--- |
| `id` | Arbitrary database row index. | `1` | **Excluded from Model** (Identifier with zero generalizable predictive signal) |
| `imdb_id` | External IMDb web identifier string (e.g. tt...). | `'tt2637294'` | **Excluded from Model** (External database key; no market signal) |
| `title` | Localized commercial movie title. | `'Hot Tub Time Machine 2'` | **Excluded from Model** (High cardinality unique strings; high risk of memorization) |
| `original_title` | Native-language original release title. | `'Hot Tub Time Machine 2'` | **Excluded from Model** (High cardinality text) |
| `status` | Release status (e.g. Released, Post Production, Rumored). | `'Released'` | **Excluded from Model** (Overwhelmingly homogeneous; risk of post-release leakage) |
| `popularity` | Dynamic TMDB user community popularity metric. | `6.575393` | **Excluded from Model** (Post-release target leakage; computed from modern queries) |
        """)

    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 5. How We Transform Movie Data into 37 Predictive Features
    # -------------------------------------------------------------------------
    render_section_header(
        "How We Transform Movie Data into 37 Predictive Features",
        "Pre-release feature extraction across 8 domain dimensions (37 total engineered features)"
    )

    st.markdown(
        "Raw movie information cannot be directly consumed by machine learning models in its unstructured form: "
        "dates require temporal decomposition, nested JSON strings must be parsed into integer counts, and skewed financial "
        "metrics demand variance stabilization. Our pipeline engineers **exactly 37 pre-release features** before categorical encoding:"
    )

    feat_tab1, feat_tab2 = st.columns([1, 1], gap="medium")

    with feat_tab1:
        # Group A
        with st.expander("A. Financial Features (3 Features)", expanded=True):
            st.markdown("""
* `budget`: Reported production budget in USD. Missing values or unrecorded zeroes are imputed using the positive-budget training partition median ($16,000,000).
* `log_budget`: Natural logarithmic transformation calculated via `log1p(budget)` to stabilize exponential variance across micro-indie and tentpole films.
* `budget_is_zero`: A dedicated binary indicator (1 if original budget was 0 or missing, 0 otherwise), enabling tree models to distinguish unrecorded budgets from true low-cost productions.

**Domain Insight:** While production budget indicates the financial scale and visual spectacle of a project, budget alone does not guarantee box-office returns; many expensive productions fail to recoup production outlays.
            """)

        # Group B
        with st.expander("B. Movie Characteristics (2 Features)", expanded=True):
            st.markdown("""
* `runtime`: Theatrical duration in minutes. Non-positive or missing runtimes are imputed using the training median (104 minutes).
* `has_collection`: Binary flag indicating whether the film belongs to an established movie franchise or cinematic universe (e.g. Marvel, Star Wars, James Bond).

**Domain Insight:** Franchise films benefit from established intellectual property, built-in audience awareness, and lower customer acquisition friction, resulting in substantially higher median box office receipts.
            """)

        # Group C
        with st.expander("C. Release Timing Features (5 Features)", expanded=True):
            st.markdown("""
* `release_year`: Calendar year of theatrical release (e.g. 1985, 2018). Normalized for 2-digit century rollover (years > 2020 mapped to 1900s).
* `release_month`: Calendar month of release (1 to 12).
* `release_quarter`: Financial quarter of release (1 to 4).
* `release_dayofweek`: Day of the week of theatrical release (0 = Monday, 6 = Sunday).
* `release_is_weekend`: Binary flag identifying Friday, Saturday, or Sunday releases (peak theatrical opening window).

**Domain Insight:** Theatrical revenue exhibits strong seasonality. Studios intentionally target high-traffic release corridors such as Summer (May–July) and the Winter Holiday season (November–December).
            """)

        # Group D
        with st.expander("D. Production Scale Features (4 Features)", expanded=True):
            st.markdown("""
* `company_count`: Total number of production companies credited.
* `is_major_studio`: Binary indicator checking whether any producing studio matches major Hollywood distributors (*Warner Bros, Universal, Paramount, 20th Century Fox, Columbia, Walt Disney, Marvel, Sony, MGM, New Line, Lionsgate, DreamWorks*).
* `cast_size`: Total count of credited actors appearing in the parsed cast list.
* `crew_size`: Total count of technical, creative, and administrative crew members.

**Domain Insight:** Major studio backing secures wide theatrical footprint (3,000+ screens) and extensive marketing muscle, providing a higher revenue floor than independent distributors.
            """)

    with feat_tab2:
        # Group E
        with st.expander("E. Genre Features (16 Features)", expanded=True):
            st.markdown("""
* `genre_count`: Total number of genres assigned to the title.
* `primary_genre`: The first listed genre in the movie metadata (used as a categorical baseline).
* **14 Explicit Binary Genre Indicators (0 or 1):**
  - `genre_drama`, `genre_comedy`, `genre_thriller`, `genre_action`, `genre_romance`
  - `genre_crime`, `genre_adventure`, `genre_horror`, `genre_scifi`, `genre_family`
  - `genre_fantasy`, `genre_animation`, `genre_mystery`, `genre_documentary`

**Domain Insight:** Binary genre indicators allow multi-genre films (such as Action + Sci-Fi + Adventure) to activate multiple signals simultaneously, reflecting the cross-genre appeal typical of commercial blockbusters.
            """)

        # Group F
        with st.expander("F. Geographic Features (2 Features)", expanded=False):
            st.markdown("""
* `country_count`: Number of countries contributing to film production.
* `is_us_country`: Binary flag indicating whether the primary production country is the United States (the primary domestic box-office engine).
            """)

        # Group G
        with st.expander("G. Language Features (3 Features)", expanded=False):
            st.markdown("""
* `language_count`: Number of distinct spoken languages represented in dialogue.
* `is_english`: Binary flag indicating whether original production language is English.
* `original_language`: Cleaned categorical language code retaining high-frequency languages (`en`, `fr`, `es`, `zh`, `de`, `hi`, `ja`, `it`, `ko`, `ru`), grouping low-frequency languages into `other`.

**Domain Insight:** English-language films benefit from immediate access to the global distribution market, whereas foreign-language titles typically rely on domestic territory receipts and specialty circuits.
            """)

        # Group H
        with st.expander("H. Marketing Presence Features (2 Features)", expanded=False):
            st.markdown("""
* `has_homepage`: Binary flag indicating presence of an official promotional website.
* `has_tagline`: Binary flag indicating presence of a promotional marketing slogan.

**Domain Insight:** While these binary flags do not measure dollar marketing expenditures, they serve as proxies for commercial intent and distributor commitment.
            """)

    st.markdown("""
        <div style="background: rgba(20, 28, 46, 0.7); border: 1px solid #1E2B44; border-left: 3px solid #10B981; border-radius: 8px; padding: 12px 16px; margin-top: 14px; font-size: 0.84rem; color: #CBD5E1;">
            <b>Feature Count Verification:</b> 3 (Financial) + 2 (Characteristics) + 5 (Timing) + 16 (Genres) + 4 (Scale) + 2 (Geography) + 3 (Language) + 2 (Marketing) = <b>37 Total Engineered Features</b>.<br>
            <i>Note on Preprocessing Expansion:</i> The two categorical attributes (<code>primary_genre</code> and <code>original_language</code>) are subsequently one-hot encoded by the Scikit-Learn ColumnTransformer pipeline, expanding the final model feature matrix to 69 encoded columns.
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 6. Feature Transformation Mapping Table
    # -------------------------------------------------------------------------
    render_section_header(
        "Feature Transformation Mapping",
        "End-to-end mapping from raw TMDB attributes to engineered model features"
    )

    mapping_df = pd.DataFrame([
        {
            "Original Dataset Column": "budget",
            "Engineered Feature(s)": "budget, log_budget, budget_is_zero",
            "Transformation Applied": "Median imputation for zeroes/missing values, natural log1p transformation, and binary zero-indicator flag."
        },
        {
            "Original Dataset Column": "release_date",
            "Engineered Feature(s)": "release_year, release_month, release_quarter, release_dayofweek, release_is_weekend",
            "Transformation Applied": "Calendar parsing, 2-digit century rollover correction (years > 2020 to 1900s), and seasonal/weekend decomposition."
        },
        {
            "Original Dataset Column": "genres",
            "Engineered Feature(s)": "genre_count, primary_genre, 14 genre binary flags",
            "Transformation Applied": "Structured JSON decoding, genre count calculation, first genre selection, and 14 one-hot binary indicator flags."
        },
        {
            "Original Dataset Column": "belongs_to_collection",
            "Engineered Feature(s)": "has_collection",
            "Transformation Applied": "JSON/dictionary validation; converted into a binary franchise indicator flag (1 if movie belongs to collection, 0 otherwise)."
        },
        {
            "Original Dataset Column": "production_companies",
            "Engineered Feature(s)": "company_count, is_major_studio",
            "Transformation Applied": "JSON decoding, company count extraction, and case-insensitive string matching against major studio registry."
        },
        {
            "Original Dataset Column": "production_countries",
            "Engineered Feature(s)": "country_count, is_us_country",
            "Transformation Applied": "JSON decoding, country count extraction, and ISO 3166-1 country code inspection for domestic US origin."
        },
        {
            "Original Dataset Column": "spoken_languages",
            "Engineered Feature(s)": "language_count",
            "Transformation Applied": "JSON decoding and distinct spoken language count extraction."
        },
        {
            "Original Dataset Column": "original_language",
            "Engineered Feature(s)": "original_language, is_english",
            "Transformation Applied": "Top 12 language retention with rare languages mapped to 'other', and binary English indicator flag."
        },
        {
            "Original Dataset Column": "cast",
            "Engineered Feature(s)": "cast_size",
            "Transformation Applied": "JSON list decoding and calculation of total credited actors."
        },
        {
            "Original Dataset Column": "crew",
            "Engineered Feature(s)": "crew_size",
            "Transformation Applied": "JSON list decoding and calculation of total credited production staff."
        },
        {
            "Original Dataset Column": "runtime",
            "Engineered Feature(s)": "runtime",
            "Transformation Applied": "Numeric conversion, non-positive validation, and training partition median imputation (104 mins)."
        },
        {
            "Original Dataset Column": "homepage",
            "Engineered Feature(s)": "has_homepage",
            "Transformation Applied": "Non-empty string presence detection mapped to binary indicator (1 if official URL present, 0 otherwise)."
        },
        {
            "Original Dataset Column": "tagline",
            "Engineered Feature(s)": "has_tagline",
            "Transformation Applied": "Non-empty string presence detection mapped to binary indicator (1 if marketing tagline present, 0 otherwise)."
        },
    ])

    st.dataframe(mapping_df, use_container_width=True, hide_index=True)

    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 7. What Are We Predicting?
    # -------------------------------------------------------------------------
    render_section_header(
        "Understanding Our Prediction Target: Worldwide Box Office Revenue",
        "Target variable formulation, logarithmic scaling, and evaluation metrics"
    )

    t_col1, t_col2 = st.columns([1, 1], gap="medium")

    with t_col1:
        st.markdown("""
### Prediction Target: Worldwide Revenue ($ USD)
The target variable is **`revenue`**, representing the total worldwide theatrical box-office revenue of a movie in US Dollars.

#### Why Logarithmic Transformation is Essential:
Raw box-office receipts span six orders of magnitude—ranging from under $10,000 for independent films to over $2,000,000,000 for global blockbusters. This produces an extremely positive, heavy-tailed skewness.

If trained directly on raw USD:
1. Multi-billion dollar tentpoles would dominate Mean Squared Error loss gradients.
2. The model would ignore relative percentage errors on smaller mid-budget titles.
3. Errors on small films would be penalized insignificantly compared to blockbusters.

We apply the natural logarithmic transformation during training:
$$\\text{log\\_revenue} = \\ln(\\text{revenue} + 1) = \\text{log1p}(\\text{revenue})$$

And recover real dollar predictions via the inverse transformation clipped at zero:
$$\\widehat{\\text{revenue}} = \\max\\left(0, e^{\\widehat{\\text{log\\_revenue}}} - 1\\right) = \\max\\left(0, \\text{expm1}(\\widehat{\\text{log\\_revenue}})\\right)$$
        """)

    with t_col2:
        st.markdown("""
### Model Evaluation Metrics
Because theatrical box office operates multiplicatively, models are evaluated with metrics tailored to heavy-tailed distributions:

* **RMSLE (Root Mean Squared Logarithmic Error) — Primary Metric:**
  $$\\text{RMSLE} = \\sqrt{\\frac{1}{N} \\sum_{i=1}^N \\left(\\ln(\\hat{y}_i + 1) - \\ln(y_i + 1)\\right)^2}$$
  RMSLE measures relative percentage error rather than absolute dollar error, treating a 50% forecasting error on a $5M movie identically to a 50% error on a $200M blockbuster.

* **MAE (Mean Absolute Error in $ USD) — Secondary Diagnostic:**
  $$\\text{MAE} = \\frac{1}{N} \\sum_{i=1}^N |y_i - \\hat{y}_i|$$
  Quantifies the average dollar discrepancy between actual and predicted revenues.

* **$R^2$ Score (Coefficient of Determination) — Variance Explained:**
  Measures the proportion of variance in actual revenue explained by the model.

> **Important Industry Distinction:** Worldwide theatrical gross revenue represents total ticket sales at the box office. It is **strictly NOT equivalent to studio profit or ROI**, as theatrical exhibitors typically retain 40–50% of box office gross, and figures do not deduct P&A marketing budgets, distribution fees, or production expenses.
        """)

    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 8. Features Excluded from Prediction
    # -------------------------------------------------------------------------
    render_section_header(
        "Why Are Some Dataset Columns Not Used?",
        "Target leakage prevention, high-cardinality filtering, and pre-release modeling assumptions"
    )

    st.markdown("""
To guarantee rigorous statistical validity and simulate a realistic **pre-release commercial environment**, several raw columns are deliberately excluded from predictive modeling:

* **`id` & `imdb_id` (Arbitrary Identifiers):** Database primary keys and external reference IDs contain zero generalizable economic signal. Retaining identifiers would allow tree-based models to memorize training indices.
* **`popularity` (Severe Post-Release Target Leakage):** The TMDB `popularity` score is dynamically recalculated on TMDB servers based on daily user views, votes, watchlist additions, and search traffic. For historical films, a high popularity score reflects current streaming interest and cult status acquired *after* theatrical release, introducing catastrophic target leakage.
* **`poster_path` (Image References):** Image URLs point to JPG files on the TMDB CDN. Computer vision processing is outside the tabular regression scope of this project.
* **`title` & `original_title` (High-Cardinality Text):** Raw movie titles contain thousands of unique string values. Incorporating raw titles without massive pre-trained language models would induce severe overfitting.
* **`overview` & `Keywords` (Unstructured Text & Sparse Tags):** Freeform synopsis text and high-cardinality keyword tags are excluded to keep the pipeline lightweight, interpretable, and reproducible.
* **`status` (Post-Release Bias):** Over 99% of training records are marked as `'Released'`. Titles labeled `'Rumored'` or `'Post Production'` reflect status updates recorded after release.
* **`revenue` (Prediction Target):** The actual revenue is the dependent variable ($y$) to be predicted and is never passed to feature matrices.

> **Pre-Release Modeling Assumption:** Our pipeline operates under the contract that all retained features (budget, genres, studio, cast/crew, release date) are knowable before a film opens in theaters. While historical snapshots in public databases can occasionally receive retrospective updates, excluding dynamic metrics (`popularity`) ensures the highest degree of real-world temporal validity.
    """)

    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 9. Complete Machine Learning Workflow
    # -------------------------------------------------------------------------
    render_section_header(
        "Complete Machine Learning Workflow",
        "End-to-end data science and engineering architecture from raw records to production deployment"
    )

    st.markdown("""
The diagram below illustrates the end-to-end data pipeline implemented across our Jupyter notebooks, Scikit-Learn pipelines, and Streamlit platform:
    """)

    st.markdown("""
```
┌────────────────────────┐      ┌────────────────────────┐      ┌────────────────────────┐
│    Raw TMDB Dataset    │ ───► │     Data Profiling     │ ───► │     Data Hygiene       │
│  (train.csv, test.csv) │      │  Missingness & Schemas │      │ Date Parsing, JSON Fix │
└────────────────────────┘      └────────────────────────┘      └────────────────────────┘
                                                                             │
                                                                             ▼
┌────────────────────────┐      ┌────────────────────────┐      ┌────────────────────────┐
│   Leakage Prevention   │ ◄─── │  Feature Engineering   │ ◄─── │      In-Depth EDA      │
│ Imputation on Train Only│      │ 37 Pre-Release Signals │      │ Skewness & Multiples   │
└────────────────────────┘      └────────────────────────┘      └────────────────────────┘
            │
            ▼
┌────────────────────────┐      ┌────────────────────────┐      ┌────────────────────────┐
│   Train/Holdout Split  │ ───► │  Pipeline Transform    │ ───► │  5-Fold Cross-Val      │
│ 80% Train / 20% Holdout│      │ ColumnTransformer OHE  │      │ 7 Model Architectures  │
└────────────────────────┘      └────────────────────────┘      └────────────────────────┘
                                                                             │
                                                                             ▼
┌────────────────────────┐      ┌────────────────────────┐      ┌────────────────────────┐
│  Final Model Retrain   │ ◄─── │    Holdout Validation  │ ◄─── │ Hyperparameter Tuning  │
│ 100% Data Production   │      │ Locked 20% Evaluation  │      │ 30-Iter RandomizedCV   │
└────────────────────────┘      └────────────────────────┘      └────────────────────────┘
            │
            ▼
┌────────────────────────┐
│ Production Forecasting │
│ Web App & Submission   │
└────────────────────────┘
```
    """)

    wf_steps = [
        ("1. Raw TMDB Dataset", "Ingesting 3,000 labeled training movies and 4,398 unlabelled testing movies from the TMDB Box Office Prediction competition."),
        ("2. Data Profiling", "Auditing variable types, null proportions, cardinality, and data formatting across all 23 raw attributes."),
        ("3. Data Hygiene & Cleaning", "Standardizing dates with 2-digit century rollover, decoding nested JSON entities, and flagging zero values."),
        ("4. Exploratory Data Analysis (EDA)", "Quantifying heavy-tailed revenue skewness, analyzing budget-to-revenue gross multiples, and exploring seasonal trends."),
        ("5. Feature Engineering (37 Features)", "Extracting 37 domain predictors spanning financial, temporal, genre, studio, geographic, and marketing dimensions."),
        ("6. Leakage Prevention", "Computing all imputation medians, scale bounds, and reference statistics strictly from the training partition."),
        ("7. Train / Holdout Split", "Partitioning data into an 80% training development set ($N = 2,400$) and a locked 20% holdout set ($N = 600$)."),
        ("8. Preprocessing Pipeline", "Constructing a Scikit-Learn ColumnTransformer applying median imputation, standard scaling, and one-hot encoding."),
        ("9. 5-Fold Cross-Validation", "Evaluating 7 distinct model families (Dummy, Random Forest, Extra Trees, HistGradientBoosting, XGBoost, LightGBM, CatBoost)."),
        ("10. Systematic Hyperparameter Tuning", "Running 30-iteration RandomizedSearchCV across all candidate algorithms (900 fits total) optimizing custom RMSLE scorer."),
        ("11. Model Selection & Ensembling", "Benchmarking tuned models and out-of-fold stacking ensembles, selecting tuned LightGBM as production champion (RMSLE: 2.1576)."),
        ("12. Locked Holdout Validation", "Evaluating final champion and baseline models on the untouched 20% holdout set ($N = 600$) to verify generalization."),
        ("13. Production Retraining", "Fitting the champion LightGBM architecture on 100% of labeled training data ($N = 3,000$) to maximize learning capacity."),
        ("14. Worldwide Revenue Forecasting", "Generating verified competition test predictions (submission.csv) and serving live interactive pre-release forecasts.")
    ]

    for title, desc in wf_steps:
        st.markdown(f"• **{title}:** {desc}")

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
