"""
visualizations.py
Modular Plotly chart engine using an enterprise corporate design system
(blue, white, and grey palette).

Visualizations:
1. plot_catalog_growth: Area / line chart of Movies vs. TV Shows over time.
2. plot_acquisition_lag_trend: Evolution of Same-Year Originals vs. Back-Catalog licensing.
3. plot_top_countries_audience: Top producing countries segmented by audience maturity tier.
4. plot_genre_cooccurrence: 2D co-occurrence heatmap of top genres.
5. plot_duration_distributions: Side-by-side subplots of Movie runtime & TV season distribution.
6. plot_missing_directors_comparison: Rate comparison of TV vs. Movie missing director values.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .feature_engineer import explode_column


# Corporate Design System Constants (Blue, White, and Grey Palette)
PRIMARY_BLUE = "#2C7BE5"
LIGHT_BLUE = "#6EA8FE"
NAVY_BLUE = "#0D6EFD"
SLATE_GREY = "#6C757D"
DARK_SLATE = "#2B3035"
LIGHT_GREY = "#E9ECEF"
BORDER_GREY = "#DEE2E6"
CARD_BG = "#FFFFFF"
TEXT_MAIN = "#2B3035"
TEXT_MUTED = "#6C757D"
GRID_COLOR = "#E9ECEF"

# Backward-compatibility aliases if referenced elsewhere
NETFLIX_RED = PRIMARY_BLUE
NETFLIX_CRIMSON = NAVY_BLUE
TEXT_WHITE = DARK_SLATE

AUDIENCE_COLOR_MAP = {
    "Mature / Adults": PRIMARY_BLUE,
    "Family / Teens": LIGHT_BLUE,
    "Kids": SLATE_GREY
}

ORIGIN_COLOR_MAP = {
    "Same-Year (Original/Exclusive Proxy)": PRIMARY_BLUE,
    "Recent Release (1-3 Yrs)": LIGHT_BLUE,
    "Back-Catalog (>3 Yrs)": SLATE_GREY
}


def _apply_corporate_theme(fig: go.Figure, title: str = "", height: int = 450) -> go.Figure:
    """Helper to apply consistent modern corporate light BI styling to any Plotly figure."""
    fig.update_layout(
        title={
            "text": f"<b>{title}</b>" if title else "",
            "y": 0.95,
            "x": 0.03,
            "xanchor": "left",
            "yanchor": "top",
            "font": {"size": 14, "color": DARK_SLATE, "family": "Inter, Segoe UI, sans-serif"}
        },
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin={"l": 40, "r": 20, "t": 50, "b": 80},
        font={"family": "Inter, Segoe UI, sans-serif", "color": DARK_SLATE, "size": 12},
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.2,
            "xanchor": "center",
            "x": 0.5,
            "bgcolor": "rgba(255, 255, 255, 0.9)",
            "bordercolor": BORDER_GREY,
            "borderwidth": 1,
            "font": {"color": DARK_SLATE, "size": 11}
        },
        hoverlabel={
            "bgcolor": CARD_BG,
            "font_size": 12,
            "font_family": "Inter, Segoe UI, sans-serif",
            "font_color": DARK_SLATE,
            "bordercolor": BORDER_GREY
        }
    )
    fig.update_xaxes(
        gridcolor=LIGHT_GREY,
        zerolinecolor=LIGHT_GREY,
        tickfont={"color": SLATE_GREY}
    )
    fig.update_yaxes(
        gridcolor=LIGHT_GREY,
        zerolinecolor=LIGHT_GREY,
        tickfont={"color": SLATE_GREY}
    )
    return fig


def plot_catalog_growth(df: pd.DataFrame) -> go.Figure:
    """
    1. Catalog Growth: Stacked area / line chart showing additions of Movies vs. TV Shows over time.
    Highlights the strategic shift in content mix.
    """
    if "year_added" not in df.columns or "type" not in df.columns:
        return go.Figure()

    # Filter out anomalous years (e.g. prior to 2010 when streaming was nascent)
    valid_df = df[df["year_added"] >= 2010].copy()
    if valid_df.empty:
        valid_df = df.copy()

    growth = (
        valid_df.groupby(["year_added", "type"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    if "Movie" not in growth.columns:
        growth["Movie"] = 0
    if "TV Show" not in growth.columns:
        growth["TV Show"] = 0

    growth["Total"] = growth["Movie"] + growth["TV Show"]
    growth["TV_Pct"] = (growth["TV Show"] / growth["Total"] * 100).round(1)

    fig = go.Figure()

    # Add Movie Area trace
    fig.add_trace(go.Scatter(
        x=growth["year_added"],
        y=growth["Movie"],
        mode="lines",
        name="Movies",
        line={"width": 2.5, "color": SLATE_GREY},
        stackgroup="one",
        fillcolor="rgba(108, 117, 125, 0.25)",
        hovertemplate="<b>Year %{x}</b><br>Movies Added: %{y:,}<extra></extra>"
    ))

    # Add TV Show Area trace
    fig.add_trace(go.Scatter(
        x=growth["year_added"],
        y=growth["TV Show"],
        mode="lines",
        name="TV Shows",
        line={"width": 2.5, "color": PRIMARY_BLUE},
        stackgroup="one",
        fillcolor="rgba(44, 123, 229, 0.25)",
        hovertemplate="<b>Year %{x}</b><br>TV Shows Added: %{y:,}<extra></extra>"
    ))

    # Add Line for Total additions
    fig.add_trace(go.Scatter(
        x=growth["year_added"],
        y=growth["Total"],
        mode="lines+markers",
        name="Total Catalog Added",
        line={"width": 2, "color": DARK_SLATE, "dash": "dot"},
        marker={"size": 6, "color": PRIMARY_BLUE},
        hovertemplate="<b>Year %{x}</b><br>Total Additions: %{y:,}<extra></extra>"
    ))

    # Annotate peak additions
    if not growth.empty:
        peak_row = growth.loc[growth["Total"].idxmax()]
        fig.add_annotation(
            x=peak_row["year_added"],
            y=peak_row["Total"],
            text=f"Peak Additions ({int(peak_row['Total']):,} titles)",
            showarrow=True,
            arrowhead=2,
            arrowcolor=PRIMARY_BLUE,
            ax=0,
            ay=-40,
            font={"color": DARK_SLATE, "size": 11},
            bgcolor=CARD_BG,
            bordercolor=BORDER_GREY,
            borderwidth=1
        )

    _apply_corporate_theme(
        fig,
        title="",
        height=430
    )
    fig.update_xaxes(title="Year Added to Catalog")
    fig.update_yaxes(title="Titles Added per Year")
    return fig


def plot_acquisition_lag_trend(df: pd.DataFrame) -> go.Figure:
    """
    2. Original vs. Licensed Strategy:
    100% Normalized stacked bar chart showing the composition of acquisitions
    (Same-Year Originals proxy vs. Recent releases vs. Back-Catalog studio licensing).
    """
    if "year_added" not in df.columns or "content_origin_proxy" not in df.columns:
        return go.Figure()

    valid_df = df[df["year_added"] >= 2014].copy()
    if valid_df.empty:
        valid_df = df.copy()

    # Aggregate counts
    agg = (
        valid_df.groupby(["year_added", "content_origin_proxy"])
        .size()
        .unstack(fill_value=0)
    )

    # Compute 100% normalized shares
    row_sums = agg.sum(axis=1)
    agg_pct = (agg.div(row_sums, axis=0) * 100).round(1).reset_index()

    fig = go.Figure()

    # Standard order: Back-Catalog, Recent, Same-Year
    categories = [
        "Back-Catalog (>3 Yrs)",
        "Recent Release (1-3 Yrs)",
        "Same-Year (Original/Exclusive Proxy)"
    ]

    for cat in categories:
        if cat in agg_pct.columns:
            fig.add_trace(go.Bar(
                x=agg_pct["year_added"],
                y=agg_pct[cat],
                name=cat,
                marker_color=ORIGIN_COLOR_MAP.get(cat, SLATE_GREY),
                hovertemplate=f"<b>%{{x}}</b><br>{cat}: <b>%{{y}}%</b><extra></extra>"
            ))

    fig.update_layout(barmode="stack")
    _apply_corporate_theme(
        fig,
        title="",
        height=430
    )
    fig.update_xaxes(title="Year Added")
    fig.update_yaxes(title="Share of Catalog Additions (%)", range=[0, 100])
    return fig


def plot_top_countries_audience(df: pd.DataFrame, top_n: int = 12) -> go.Figure:
    """
    3. Global Production:
    Horizontal stacked bar chart of Top N producing countries, segmented by audience tier.
    Uses multi-label exploded country data.
    """
    if "country" not in df.columns or "audience_tier" not in df.columns:
        return go.Figure()

    # Explode multi-country titles
    exploded_c = explode_column(df, "country", drop_values=["Unknown", "Unlisted", ""])
    if exploded_c.empty:
        return go.Figure()

    # Top N countries
    top_countries = exploded_c["country"].value_counts().head(top_n).index.tolist()
    filtered = exploded_c[exploded_c["country"].isin(top_countries)]

    # Cross-tabulate
    ct = (
        filtered.groupby(["country", "audience_tier"])
        .size()
        .unstack(fill_value=0)
    )

    # Order countries by total titles
    ct["Total"] = ct.sum(axis=1)
    ct = ct.sort_values("Total", ascending=True)

    fig = go.Figure()
    tiers = ["Kids", "Family / Teens", "Mature / Adults"]

    for tier in tiers:
        if tier in ct.columns:
            fig.add_trace(go.Bar(
                y=ct.index,
                x=ct[tier],
                name=tier,
                orientation="h",
                marker_color=AUDIENCE_COLOR_MAP.get(tier, SLATE_GREY),
                hovertemplate=f"<b>%{{y}}</b><br>{tier}: %{{x:,}} titles<extra></extra>"
            ))

    fig.update_layout(barmode="stack")
    _apply_corporate_theme(
        fig,
        title="",
        height=460
    )
    fig.update_xaxes(title="Number of Titles Produced")
    fig.update_yaxes(title="", dtick=1)
    return fig


def plot_genre_cooccurrence(df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """
    4. Genre Co-Occurrence Matrix Heatmap:
    Calculates pairwise co-occurrence frequencies between top genres.
    """
    if "listed_in" not in df.columns:
        return go.Figure()

    # First find top genres
    exploded = explode_column(df, "listed_in", drop_values=["", "Unknown"])
    if exploded.empty:
        return go.Figure()

    top_genres = exploded["listed_in"].value_counts().head(top_n).index.tolist()

    # Build co-occurrence matrix
    matrix = pd.DataFrame(0, index=top_genres, columns=top_genres)

    # Iterate over rows with listed_in
    for genres_str in df["listed_in"].dropna():
        genres = [g.strip() for g in genres_str.split(",") if g.strip() in top_genres]
        for g1 in genres:
            for g2 in genres:
                if g1 != g2:
                    matrix.loc[g1, g2] += 1

    custom_colorscale = [
        [0.0, "#F8F9FA"],
        [0.2, "#E3EBF6"],
        [0.4, "#ADC8F8"],
        [0.7, "#6EA8FE"],
        [1.0, "#2C7BE5"]
    ]

    fig = go.Figure(data=go.Heatmap(
        z=matrix.values,
        x=matrix.columns,
        y=matrix.index,
        colorscale=custom_colorscale,
        text=matrix.values,
        texttemplate="%{text}",
        textfont={"size": 11, "color": DARK_SLATE},
        hoverongaps=False,
        hovertemplate="<b>%{y}</b> + <b>%{x}</b><br>Co-occurrences: %{z:,}<extra></extra>"
    ))

    _apply_corporate_theme(
        fig,
        title="",
        height=480
    )
    fig.update_xaxes(tickangle=-35)
    return fig


def plot_duration_distributions(df: pd.DataFrame) -> go.Figure:
    """
    5. Runtime & Season Distribution:
    Side-by-side subplots:
    - Left: Movie runtime distribution (duration_min) highlighting the 90-110 min typical window.
    - Right: TV Show season count distribution (seasons_count) detailing season drop-off rates.
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "Movie Runtimes (90–110 Min Window)",
            "TV Show Seasons (Season Count Distribution)"
        ),
        horizontal_spacing=0.12
    )

    # Subplot 1: Movie Runtimes
    movies_df = df[df["type"] == "Movie"].dropna(subset=["duration_min"])
    if not movies_df.empty:
        runtimes = movies_df["duration_min"]
        median_runtime = runtimes.median()

        fig.add_trace(
            go.Histogram(
                x=runtimes,
                nbinsx=35,
                marker_color=PRIMARY_BLUE,
                opacity=0.85,
                name="Movies",
                hovertemplate="Runtime: <b>%{x} min</b><br>Count: %{y}<extra></extra>"
            ),
            row=1, col=1
        )

        # Vertical line for median
        fig.add_vline(
            x=median_runtime,
            line_dash="dash",
            line_color=DARK_SLATE,
            line_width=2,
            annotation_text=f"Median: {int(median_runtime)}m",
            annotation_position="top right",
            annotation_font={"color": DARK_SLATE, "size": 11},
            row=1, col=1
        )

        # Highlight sweet spot band (90-110m)
        fig.add_vrect(
            x0=90, x1=110,
            fillcolor="rgba(44, 123, 229, 0.12)",
            layer="below",
            line_width=1,
            line_color="rgba(44, 123, 229, 0.3)",
            row=1, col=1
        )

    # Subplot 2: TV Show Season Drop-Off
    tv_df = df[df["type"] == "TV Show"].dropna(subset=["seasons_count"])
    if not tv_df.empty:
        season_counts = tv_df["seasons_count"].value_counts().sort_index()
        total_shows = len(tv_df)
        season_labels = [f"S{int(s)}" for s in season_counts.index]
        season_pcts = (season_counts / total_shows * 100).round(1)

        # Bar colors: gradient from primary blue to slate grey
        colors = ["#2C7BE5", "#4B8FE8", "#6EA8FE", "#9EC5FE", "#6C757D", "#ADB5BD"]
        bar_colors = [colors[min(i, len(colors)-1)] for i in range(len(season_counts))]

        fig.add_trace(
            go.Bar(
                x=season_labels,
                y=season_counts.values,
                marker_color=bar_colors,
                name="TV Shows",
                customdata=season_pcts.values,
                hovertemplate="Season %{x}<br>Count: <b>%{y}</b> (%{customdata}% of TV catalog)<extra></extra>"
            ),
            row=1, col=2
        )

    _apply_corporate_theme(fig, title="", height=430)
    fig.update_xaxes(title_text="Duration (Minutes)", row=1, col=1)
    fig.update_yaxes(title_text="Number of Movies", row=1, col=1)
    fig.update_xaxes(title_text="Season Count", row=1, col=2)
    fig.update_yaxes(title_text="Number of TV Shows", row=1, col=2)
    fig.update_layout(showlegend=False)

    return fig


def plot_missing_directors_comparison(hygiene_report: Dict[str, Any]) -> go.Figure:
    """
    6. Missing Director Field Analysis:
    Compares the missing director rate in Movies vs. TV Shows.
    Demonstrates that missing metadata is an inherent property of episodic TV.
    """
    h = hygiene_report.get("director_hypothesis", {})
    categories = ["Feature Films (Movies)", "Episodic TV Shows"]
    missing_rates = [
        h.get("movie_missing_dir_pct", 12.0),
        h.get("tv_missing_dir_pct", 74.0)
    ]
    totals = [
        h.get("movie_total", 0),
        h.get("tv_total", 0)
    ]
    missing_counts = [
        h.get("movie_missing_dir", 0),
        h.get("tv_missing_dir", 0)
    ]

    fig = go.Figure(go.Bar(
        x=categories,
        y=missing_rates,
        marker_color=[LIGHT_BLUE, PRIMARY_BLUE],
        text=[f"<b>{r}%</b> Missing<br>({c:,} / {t:,})" for r, c, t in zip(missing_rates, missing_counts, totals)],
        textposition="inside",
        textfont={"color": "#FFFFFF", "size": 13},
        hovertemplate="<b>%{x}</b><br>Missing Director: %{y}%<extra></extra>"
    ))

    _apply_corporate_theme(fig, title="", height=350)
    fig.update_yaxes(title="Missing Director Rate (%)", range=[0, 100])
    fig.update_layout(showlegend=False)
    return fig
