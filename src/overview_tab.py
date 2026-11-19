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

from typing import Dict, Any, Optional
import streamlit as st
import pandas as pd
import numpy as np
import joblib

from src.data_loader import safe_parse_json, parse_release_date, load_tmdb_test_data, load_tmdb_train_data
from src.feature_engineering import ALL_FEATURE_COLUMNS, engineer_movie_features
from src.ml_studio_tab import resolve_model_path
from src.visualizations import (
    plot_revenue_distribution,
    plot_budget_vs_revenue,
    plot_revenue_by_genre,
    plot_revenue_trend_over_time,
    plot_budget_distribution,
    plot_budget_vs_revenue_franchise,
    plot_genre_budget_heatmap,
    plot_shap_summary_bar,
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


@st.cache_data(show_spinner=False)
def compute_shap_explanation_data(sample_size: int = 350, random_state: int = 42) -> Optional[Dict[str, Any]]:
    """
    Computes and caches SHAP values from the saved production model artifact.
    Uses fitted TreeExplainer on a representative sample of labeled records.
    """
    resolved_path = resolve_model_path()
    if not resolved_path:
        return None

    try:
        import shap
        bundle = joblib.load(resolved_path)
        pipeline = bundle["pipeline"]
        stats = bundle.get("training_stats")
        model_name = bundle.get("model_name", "lightgbm")

        raw_df, _ = load_tmdb_train_data()
        if raw_df is None or raw_df.empty:
            return None

        valid_raw = raw_df[pd.to_numeric(raw_df["revenue"], errors="coerce") > 0].copy()
        X_mat, _, _ = engineer_movie_features(valid_raw, is_train=True, stats=stats)

        preproc = pipeline.named_steps.get("preprocessor")
        regressor = pipeline.named_steps.get("regressor")
        if preproc is None or regressor is None:
            return None

        X_trans = preproc.transform(X_mat)
        raw_feature_names = preproc.get_feature_names_out()
        feature_names = [f.replace("num__", "").replace("cat__", "").replace("remainder__", "") for f in raw_feature_names]

        rng = np.random.RandomState(random_state)
        sample_indices = rng.choice(len(X_mat), size=min(sample_size, len(X_mat)), replace=False)
        X_sample = X_trans[sample_indices]

        # TreeExplainer for tree estimators
        explainer = shap.TreeExplainer(regressor)
        shap_values = explainer.shap_values(X_sample)

        mean_abs = np.mean(np.abs(shap_values), axis=0)
        shap_ranking_df = pd.DataFrame({
            "feature": feature_names,
            "mean_abs_shap": mean_abs
        }).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)

        return {
            "model_name": model_name,
            "regressor_class": regressor.__class__.__name__,
            "shap_values": shap_values,
            "X_sample": X_sample,
            "feature_names": feature_names,
            "shap_ranking_df": shap_ranking_df,
            "sample_size": len(sample_indices)
        }
    except Exception as e:
        return {"error": str(e)}


def render_shap_beeswarm_figure(shap_data: Dict[str, Any]):
    """Renders the SHAP beeswarm figure styled to match DataLab's dark theme."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    fig = plt.figure(figsize=(10, 6), facecolor="#0B132B")
    shap.summary_plot(
        shap_data["shap_values"],
        shap_data["X_sample"],
        feature_names=shap_data["feature_names"],
        max_display=16,
        show=False,
        color_bar=True
    )
    plt.gcf().patch.set_facecolor("#0B132B")
    ax = plt.gca()
    ax.set_facecolor("#0B132B")
    ax.tick_params(colors="#F8FAFC", labelsize=10)
    ax.xaxis.label.set_color("#F8FAFC")
    ax.yaxis.label.set_color("#F8FAFC")
    for spine in ax.spines.values():
        spine.set_color("#1E2B44")
    plt.title("SHAP Feature Impact Beeswarm (Log-Revenue Prediction Space)", color="#F8FAFC", fontsize=13, pad=15)
    plt.tight_layout()
    return fig


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
    # 2B. Revenue Drivers & Feature Interactions
    # -------------------------------------------------------------------------
    render_section_header(
        "Revenue Drivers & Feature Interactions",
        "Investigating franchise revenue multiples, genre-budget interaction heatmaps, and SHAP model explainability"
    )

    # 1. Budget vs Revenue: Franchise vs Standalone
    st.markdown("#### 1. Budget vs Revenue — Franchise vs Standalone Movies")
    st.markdown(
        "Examining how production investment translates to worldwide box-office receipts for franchise properties "
        "(sequels, prequels, shared cinematic universes) versus standalone original films."
    )

    # Calculate summary statistics for franchise vs standalone
    valid_bf = df[(pd.to_numeric(df["budget"], errors="coerce") > 0) & (pd.to_numeric(df["revenue"], errors="coerce") > 0)].copy()
    if not valid_bf.empty and "belongs_to_collection" in valid_bf.columns:
        col = valid_bf["belongs_to_collection"]
        is_franchise = (
            col.notna()
            & (col.astype(str).str.strip() != "")
            & (col.astype(str).str.strip() != "[]")
            & (col.astype(str).str.strip() != "{}")
            & (col.astype(str).str.strip().str.lower() != "nan")
        )
        fran_df = valid_bf[is_franchise]
        stand_df = valid_bf[~is_franchise]

        n_fran = len(fran_df)
        n_stand = len(stand_df)
        med_b_fran = float(fran_df["budget"].median()) if n_fran > 0 else 0.0
        med_r_fran = float(fran_df["revenue"].median()) if n_fran > 0 else 0.0
        med_m_fran = float((fran_df["revenue"] / fran_df["budget"]).median()) if n_fran > 0 else 0.0

        med_b_stand = float(stand_df["budget"].median()) if n_stand > 0 else 0.0
        med_r_stand = float(stand_df["revenue"].median()) if n_stand > 0 else 0.0
        med_m_stand = float((stand_df["revenue"] / stand_df["budget"]).median()) if n_stand > 0 else 0.0

        franchise_kpis = [
            {
                "title": "Franchise Movies",
                "value": f"{n_fran:,}",
                "subtext": f"Median Budget: {format_currency_compact(med_b_fran)}",
                "subtext_neutral": True,
                "accent_color": COLOR_CYAN
            },
            {
                "title": "Franchise Median Revenue",
                "value": format_currency_compact(med_r_fran),
                "subtext": f"Median Multiple: {med_m_fran:.2f}x gross",
                "subtext_neutral": True,
                "accent_color": COLOR_CYAN
            },
            {
                "title": "Standalone Movies",
                "value": f"{n_stand:,}",
                "subtext": f"Median Budget: {format_currency_compact(med_b_stand)}",
                "subtext_neutral": True,
                "accent_color": COLOR_VIOLET
            },
            {
                "title": "Standalone Median Revenue",
                "value": format_currency_compact(med_r_stand),
                "subtext": f"Median Multiple: {med_m_stand:.2f}x gross",
                "subtext_neutral": True,
                "accent_color": COLOR_VIOLET
            }
        ]
        render_kpi_row(franchise_kpis)
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    fig_franchise = plot_budget_vs_revenue_franchise(df)
    st.plotly_chart(fig_franchise, use_container_width=True)

    render_callout(
        title="Key Takeaways: Franchise Multiples & Gross Revenue Definition",
        body=(
            "• <b>Association vs Causation:</b> Franchise movies demonstrate a 3.63x median gross multiple (vs 1.56x standalone). Franchise status reflects studio selection of proven IP and wider exhibitor distribution, rather than pure causal lift.<br>"
            "• <b>Gross Multiple vs Profit:</b> The revenue-to-budget ratio measures box-office ticket gross against production negative cost. Theaters typically retain ~50% of ticket receipts, and figures exclude marketing (P&A) and distribution expenses."
        ),
        callout_type="info"
    )

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # 2. Genre vs Budget Category: Revenue Heatmap
    st.markdown("#### 2. Genre vs Budget Category — Revenue Heatmap")
    st.markdown(
        "Median box-office receipts across film genres segmented by production budget quartiles "
        "(Q1: &le; $5.2M, Q2: $5.2M–$16.0M, Q3: $16.0M–$40.0M, Q4: &gt; $40.0M). Multi-genre titles contribute to each declared genre."
    )

    fig_heatmap = plot_genre_budget_heatmap(df, min_sample=10)
    st.plotly_chart(fig_heatmap, use_container_width=True)

    render_callout(
        title="Key Takeaways: Genre Capital Efficiency & Scale",
        body=(
            "• <b>Low-Budget Efficiency:</b> In Q1 (&le; $5.2M), <b>Horror</b> achieves the highest median box office ($10.4M), outperforming Action ($5.9M) and Drama ($3.2M) at comparable spend.<br>"
            "• <b>High-Budget Scaling:</b> In Q4 (&gt; $40.0M), spectacle-driven genres lead: <b>Animation</b> ($347.4M), <b>Adventure</b> ($297.1M), and <b>Fantasy</b> ($223.5M).<br>"
            "• <b>Sample Reliability:</b> Cells with under 10 films are masked to prevent small-sample distortion."
        ),
        callout_type="info"
    )

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # 3. SHAP Summary Plot
    st.markdown("#### 3. SHAP Summary Plot — Understanding Model Feature Impact")
    st.markdown(
        "Shapley Additive exPlanations (SHAP) decompose predictions from the serialized production pipeline (`best_movie_revenue_model.joblib`), "
        "revealing which engineered pre-release features exert the strongest directional pull on predicted worldwide box-office revenue."
    )

    shap_data = compute_shap_explanation_data(sample_size=350, random_state=42)

    if shap_data is None:
        st.warning("Production model artifact ('best_movie_revenue_model.joblib') is not present. SHAP summary cannot be generated.")
    elif "error" in shap_data:
        st.warning(f"Unable to compute SHAP values: {shap_data['error']}")
    else:
        # Display SHAP Beeswarm Figure
        fig_shap = render_shap_beeswarm_figure(shap_data)
        st.pyplot(fig_shap, clear_figure=True)
        st.caption(f"SHAP beeswarm summary plot calculated across {shap_data['sample_size']} representative films using {shap_data['regressor_class']} TreeExplainer.")

        # Interactive Bar Chart of Top 15 Mean |SHAP| Values
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        fig_shap_bar = plot_shap_summary_bar(shap_data["shap_ranking_df"], top_n=15)
        st.plotly_chart(fig_shap_bar, use_container_width=True)

        # Top 15 Ranking Table
        with st.expander("View Top 15 Features Ranked by Mean Absolute SHAP Impact", expanded=False):
            display_table = shap_data["shap_ranking_df"].head(15).copy()
            display_table.columns = ["Feature Name", "Mean |SHAP| (Log Scale)"]
            display_table["Mean |SHAP| (Log Scale)"] = display_table["Mean |SHAP| (Log Scale)"].apply(lambda x: f"{x:.4f}")
            display_table.index = range(1, len(display_table) + 1)
            st.table(display_table)

        render_callout(
            title="Key Takeaways: SHAP in Log-Revenue Target Space",
            body=(
                "• <b>Log Multiplier Scale:</b> Models predict ln(revenue + 1); a +1.0 SHAP shift corresponds to an &asymp; e<sup>1</sup> &asymp; 2.72x gross revenue multiplier.<br>"
                "• <b>Top Revenue Drivers:</b> Budget (<code>budget</code>, <code>log_budget</code>), major studio backing (<code>is_major_studio</code>), runtime, franchise affiliation (<code>has_collection</code>), and cast size exert the strongest directional pull.<br>"
                "• <b>Directional Impact:</b> Red points indicate high feature values; rightward displacement increases predicted box-office gross."
            ),
            callout_type="info"
        )

    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 3. Understanding the Dataset
    # -------------------------------------------------------------------------
    render_section_header(
        "Understanding the Dataset",
        "TMDB Movie Box Office dataset dimensions, scope, and partition roles"
    )

    test_df, _ = load_tmdb_test_data()
    n_train_movies = len(df)
    n_test_movies = len(test_df) if test_df is not None else 4398
    n_train_cols = df.shape[1]
    n_engineered = len(ALL_FEATURE_COLUMNS)

    dataset_summary_cards = [
        {"title": "Training Set", "value": f"{n_train_movies:,}", "subtext": "Historical labeled movies", "subtext_neutral": True, "accent_color": COLOR_CYAN},
        {"title": "Testing Set", "value": f"{n_test_movies:,}", "subtext": "Unlabeled competition movies", "subtext_neutral": True, "accent_color": COLOR_BLUE},
        {"title": "Raw Attributes", "value": f"{n_train_cols}", "subtext": "23 original attributes (22 in test)", "subtext_neutral": True, "accent_color": COLOR_VIOLET},
        {"title": "Engineered Features", "value": f"{n_engineered}", "subtext": "Pre-release predictors", "subtext_neutral": True, "accent_color": COLOR_SUCCESS},
        {"title": "Prediction Target", "value": "Revenue (USD)", "subtext": "Worldwide box office receipts", "subtext_neutral": True, "accent_color": COLOR_CYAN}
    ]
    render_kpi_row(dataset_summary_cards)

    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 4. Raw Dataset Column Dictionary
    # -------------------------------------------------------------------------
    render_section_header(
        "Original Dataset Column Dictionary",
        "Reference dictionary of all 23 original attributes categorized by functional domain"
    )

    tab_fin, tab_creat, tab_temp, tab_prod, tab_id = st.tabs([
        "Financial & Target (2)",
        "Creative & Marketing (6)",
        "Temporal & Language (4)",
        "Cast & Production (5)",
        "Identifiers & Admin (6)",
    ])

    with tab_fin:
        st.markdown("""
| Column | Description | Real Example from Dataset | Modeling Status |
| :--- | :--- | :--- | :--- |
| `revenue` | Actual worldwide theatrical box-office revenue in USD. | `12314651` ($12.3M USD) | **Target Variable** (Log-transformed during training) |
| `budget` | Production budget of the movie in USD. Contains zeroes for unrecorded entries. | `14000000` ($14.0M USD) | **Transformed into Features** (`budget`, `log_budget`, `budget_is_zero`) |
        """)

    with tab_creat:
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

    with tab_temp:
        st.markdown("""
| Column | Description | Real Example from Dataset | Modeling Status |
| :--- | :--- | :--- | :--- |
| `release_date` | Theatrical release date (M/D/YY or YYYY-MM-DD). | `'2/20/15'` (February 20, 2015) | **Transformed into Features** (5 temporal features; century rollover adjusted) |
| `runtime` | Theatrical runtime duration in minutes. | `93.0` minutes | **Transformed into Features** (`runtime`, imputed for missing/zero values) |
| `original_language` | Primary two-letter ISO 639-1 language code. | `'en'` | **Transformed into Features** (`original_language`, `is_english`) |
| `spoken_languages` | Nested JSON list of all spoken dialogue languages. | `[{'iso_639_1': 'en', 'name': 'English'}]` | **Transformed into Features** (`language_count`) |
        """)

    with tab_prod:
        st.markdown("""
| Column | Description | Real Example from Dataset | Modeling Status |
| :--- | :--- | :--- | :--- |
| `cast` | Nested JSON list of credited actors, characters, and billing order. | `[{'cast_id': 4, 'character': 'Lou', 'name': 'Rob Corddry'}]` | **Transformed into Features** (`cast_size` total cast count) |
| `crew` | Nested JSON list of directors, writers, producers, and crew members. | `[{'job': 'Director', 'name': 'Steve Pink'}]` | **Transformed into Features** (`crew_size` total crew count) |
| `production_companies` | Nested JSON list of studios and production companies. | `[{'name': 'Paramount Pictures', 'id': 4}]` | **Transformed into Features** (`company_count`, `is_major_studio`) |
| `production_countries` | Nested JSON list of sovereign countries involved in production. | `[{'iso_3166_1': 'US', 'name': 'United States'}]` | **Transformed into Features** (`country_count`, `is_us_country`) |
| `belongs_to_collection` | Nested JSON dictionary with movie franchise details. | `[{'id': 313576, 'name': 'Hot Tub Time Machine Collection'}]` | **Transformed into Features** (`has_collection` binary franchise indicator) |
        """)

    with tab_id:
        st.markdown("""
| Column | Description | Real Example from Dataset | Modeling Status |
| :--- | :--- | :--- | :--- |
| `id` | Arbitrary database row index. | `1` | **Excluded from Model** (Identifier with zero generalizable predictive signal) |
| `imdb_id` | External IMDb web identifier string (e.g. tt...). | `'tt2637294'` | **Excluded from Model** (External database key; no market signal) |
| `title` | Localized commercial movie title. | `'Hot Tub Time Machine 2'` | **Excluded from Model** (High cardinality unique strings; memorization risk) |
| `original_title` | Native-language original release title. | `'Hot Tub Time Machine 2'` | **Excluded from Model** (High cardinality text) |
| `status` | Release status (e.g. Released, Post Production, Rumored). | `'Released'` | **Excluded from Model** (99%+ homogeneous; post-release leakage risk) |
| `popularity` | Dynamic TMDB user community popularity metric. | `6.575393` | **Excluded from Model** (Post-release target leakage; computed from modern queries) |
        """)

    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 5. Engineered Predictive Features (37 Total)
    # -------------------------------------------------------------------------
    render_section_header(
        "Engineered Predictive Features (37 Total)",
        "Pre-release feature extraction across 8 domain dimensions"
    )

    f_tab1, f_tab2, f_tab3, f_tab4 = st.tabs([
        "Financial & Characteristics (5)",
        "Release Timing & Geography (7)",
        "Production Scale & Marketing (6)",
        "Genre Flags (16) & Language (3)",
    ])

    with f_tab1:
        st.markdown("""
* **`budget`**: Reported production budget in USD (median imputation of $16M for unrecorded zeroes).
* **`log_budget`**: Natural log transformation `log1p(budget)` to stabilize exponential variance across micro-indie and tentpole films.
* **`budget_is_zero`**: Binary indicator (1 if original budget was 0 or missing, 0 otherwise) enabling tree models to distinguish unrecorded budgets from true low-cost productions.
* **`runtime`**: Theatrical runtime in minutes (median imputation of 104 mins for missing values).
* **`has_collection`**: Binary flag indicating franchise/cinematic universe affiliation (e.g. Marvel, Star Wars).
        """)

    with f_tab2:
        st.markdown("""
* **`release_year`**: Calendar year of release, normalized for 2-digit century rollover (years > 2020 mapped to 1900s).
* **`release_month`**: Calendar month of release (1 to 12) capturing holiday and summer corridors.
* **`release_quarter`**: Financial quarter of release (1 to 4).
* **`release_dayofweek`**: Day of the week of release (0 = Monday, 6 = Sunday).
* **`release_is_weekend`**: Binary flag identifying Friday, Saturday, or Sunday releases (peak theatrical opening window).
* **`country_count`**: Total count of sovereign nations contributing to production.
* **`is_us_country`**: Binary flag indicating whether the primary production country is the United States.
        """)

    with f_tab3:
        st.markdown("""
* **`company_count`**: Total number of production companies credited.
* **`is_major_studio`**: Binary flag indicating backing by major Hollywood distributors (*Warner Bros, Universal, Paramount, 20th Century Fox, Columbia, Walt Disney, Marvel, Sony, MGM, New Line, Lionsgate, DreamWorks*).
* **`cast_size`**: Total count of credited actors appearing in the parsed cast list.
* **`crew_size`**: Total count of technical, creative, and administrative crew members.
* **`has_homepage`**: Binary flag indicating presence of an official promotional website.
* **`has_tagline`**: Binary flag indicating presence of a promotional marketing slogan.
        """)

    with f_tab4:
        st.markdown("""
* **`genre_count`**: Total number of genres assigned to the movie.
* **`primary_genre`**: First listed genre in movie metadata (categorical baseline).
* **14 Explicit Binary Genre Indicators (0 or 1):**
  `genre_drama`, `genre_comedy`, `genre_thriller`, `genre_action`, `genre_romance`, `genre_crime`, `genre_adventure`, `genre_horror`, `genre_scifi`, `genre_family`, `genre_fantasy`, `genre_animation`, `genre_mystery`, `genre_documentary`.
* **`language_count`**: Number of distinct spoken languages represented in dialogue.
* **`is_english`**: Binary flag indicating whether the original production language is English.
* **`original_language`**: Top 10 frequency-retained language code with rare languages mapped to `other`.
        """)

    render_callout(
        title="Pipeline Dimensionality",
        body="<b>37 Engineered Pre-Release Features:</b> 3 Financial + 2 Characteristics + 5 Timing + 16 Genre Flags + 4 Scale + 2 Geography + 3 Language + 2 Marketing. The ColumnTransformer one-hot encodes categorical attributes, expanding the matrix to 69 encoded modeling features.",
        callout_type="info"
    )

    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 6. Feature Transformation Mapping
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

    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 7. What Are We Predicting?
    # -------------------------------------------------------------------------
    render_section_header(
        "Prediction Target & Evaluation Metrics",
        "Formulation of worldwide box office revenue, logarithmic scaling, and evaluation criteria"
    )

    t_col1, t_col2 = st.columns([1, 1], gap="medium")

    with t_col1:
        st.markdown("""
**Target Variable:** Total worldwide theatrical box-office revenue in US Dollars (`revenue`).

**Logarithmic Transformation Formulation:**
Because raw box-office receipts span six orders of magnitude ($10,000 to $2,000,000,000+), models are trained on log-transformed targets:
$$\\text{log\\_revenue} = \\ln(\\text{revenue} + 1) = \\text{log1p}(\\text{revenue})$$

Real-dollar predictions are recovered via the inverse exponential clipped at zero:
$$\\widehat{\\text{revenue}} = \\max\\left(0, e^{\\widehat{\\text{log\\_revenue}}} - 1\\right) = \\max\\left(0, \\text{expm1}(\\widehat{\\text{log\\_revenue}})\\right)$$
        """)

    with t_col2:
        st.markdown("""
**Evaluation Metrics:**

* **RMSLE (Root Mean Squared Logarithmic Error) — Primary:**
  $$\\text{RMSLE} = \\sqrt{\\frac{1}{N} \\sum_{i=1}^N \\left(\\ln(\\hat{y}_i + 1) - \\ln(y_i + 1)\\right)^2}$$
  Penalizes relative percentage errors equally across mid-tier titles and tentpole blockbusters.
* **MAE (Mean Absolute Error) — Secondary:** Quantifies average absolute dollar discrepancy.
* **$R^2$ Score (Coefficient of Determination):** Proportion of target variance explained.

*Note:* Theatrical gross revenue measures total ticket sales; exhibitors retain ~50%, and marketing/distribution costs are excluded.
        """)

    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 8. Features Excluded from Prediction
    # -------------------------------------------------------------------------
    render_section_header(
        "Excluded Columns & Leakage Prevention",
        "Raw attributes omitted to ensure pre-release validity and prevent target leakage"
    )

    excluded_df = pd.DataFrame([
        {"Column(s)": "popularity", "Reason Excluded": "Dynamic metric computed from daily site traffic and queries", "Risk Prevented": "Post-release target leakage"},
        {"Column(s)": "revenue", "Reason Excluded": "Dependent target variable to be predicted (y)", "Risk Prevented": "Direct label leakage"},
        {"Column(s)": "id, imdb_id", "Reason Excluded": "Arbitrary database keys and external reference IDs", "Risk Prevented": "Memorization of non-generalizable indices"},
        {"Column(s)": "status", "Reason Excluded": "99%+ homogeneous ('Released'); reflects retrospective label", "Risk Prevented": "Post-release status bias"},
        {"Column(s)": "title, original_title", "Reason Excluded": "High-cardinality unique text strings", "Risk Prevented": "Severe overfitting without large NLP models"},
        {"Column(s)": "overview, Keywords", "Reason Excluded": "Unstructured synopsis text and sparse tag lists", "Risk Prevented": "Noise injection outside tabular modeling scope"},
        {"Column(s)": "poster_path", "Reason Excluded": "Image CDN URLs", "Risk Prevented": "Computer vision scope outside tabular regression"},
    ])
    st.dataframe(excluded_df, use_container_width=True, hide_index=True)
    st.caption("Pre-Release Modeling Principle: Retained features must be knowable before a movie opens theatrically.")

    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 9. Complete Machine Learning Workflow
    # -------------------------------------------------------------------------
    render_section_header(
        "Complete Machine Learning Workflow",
        "End-to-end architecture from raw TMDB records to production deployment"
    )

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

    wf_col1, wf_col2 = st.columns(2, gap="medium")
    with wf_col1:
        st.markdown("""
* **Phase 1: Ingestion & Hygiene:** Ingest labeled train (3,000) and unlabeled test (4,398) records; standardize 2-digit century dates; parse nested JSON schemas (cast, crew, genres, studios).
* **Phase 2: Leak-Free Preprocessing:** Derive 37 pre-release domain predictors; compute imputation medians strictly on training partitions; apply ColumnTransformer one-hot encoding.
        """)
    with wf_col2:
        st.markdown("""
* **Phase 3: Model Benchmark & Tuning:** Evaluate 7 model families via 5-fold CV; execute 30-iteration RandomizedSearchCV optimizing RMSLE; construct out-of-fold stacking ensemble.
* **Phase 4: Validation & Deployment:** Verify generalization on locked 20% holdout (N = 600); retrain champion LightGBM on 100% labeled data for live production inference.
        """)

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
