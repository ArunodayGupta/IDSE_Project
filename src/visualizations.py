"""
visualizations.py
Interactive Plotly visualizations for TMDB Movie Box Office Revenue platform.

Charts:
1. plot_revenue_distribution: Worldwide revenue distribution with log/linear scale support.
2. plot_budget_vs_revenue: Production budget vs worldwide revenue interactive scatter.
3. plot_revenue_by_genre: Median revenue and movie counts across film genres.
4. plot_revenue_trend_over_time: Evolution of box office revenue across release years.
5. plot_budget_distribution: Distribution of production budgets.
6. plot_actual_vs_predicted: Regression evaluation scatter with y=x parity reference.
7. plot_residuals_distribution: Distribution of prediction residuals with explicit trimming note.
8. plot_feature_importance: Top model feature importance rankings.
9. plot_missing_values_summary: Visual summary of missing metadata across columns.
Strictly adheres to professional styling with zero emojis.
"""

from typing import List, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from src.data_loader import safe_parse_json
from src.ui_theme import (
    apply_figure_theme,
    COLOR_CYAN,
    COLOR_BLUE,
    COLOR_VIOLET,
    COLOR_SUCCESS,
    COLOR_WARNING,
    COLOR_DANGER,
    TEXT_PRIMARY,
    TEXT_MUTED,
    BORDER_SUBTLE,
    BG_CARD,
)


def format_currency_compact(val: float) -> str:
    """Formats numeric amounts into readable compact currency strings (e.g. $12.5M)."""
    if pd.isna(val) or val is None:
        return "N/A"
    abs_v = abs(val)
    if abs_v >= 1e9:
        return f"${val / 1e9:.2f}B"
    elif abs_v >= 1e6:
        return f"${val / 1e6:.1f}M"
    elif abs_v >= 1e3:
        return f"${val / 1e3:.0f}K"
    else:
        return f"${val:,.0f}"


def create_empty_chart(title: str = "Data Not Available", height: int = 350) -> go.Figure:
    """Creates a standardized empty-state Plotly figure when records are absent."""
    fig = go.Figure()
    fig.add_annotation(
        text="No data available for display",
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=13, color=TEXT_MUTED)
    )
    return apply_figure_theme(fig, title=title, height=height)


# -----------------------------------------------------------------------------
# Section A: Overview Visualizations
# -----------------------------------------------------------------------------

def plot_revenue_distribution(df: pd.DataFrame, log_scale: bool = True) -> go.Figure:
    """
    Renders the worldwide movie revenue distribution.
    Supports logarithmic x-axis to accommodate extreme skewness.
    Handles empty datasets gracefully.
    """
    if df is None or df.empty or "revenue" not in df.columns:
        return create_empty_chart("Worldwide Box Office Revenue Distribution", height=380)

    valid = df[df["revenue"].notna() & (df["revenue"] > 0)].copy()
    if valid.empty:
        return create_empty_chart("Worldwide Box Office Revenue Distribution (No Positive Revenue)", height=380)

    rev = valid["revenue"].values

    fig = go.Figure()

    if log_scale:
        log_rev = np.log10(rev)
        fig.add_trace(go.Histogram(
            x=log_rev,
            nbinsx=40,
            marker_color=COLOR_CYAN,
            marker_line_color=BORDER_SUBTLE,
            marker_line_width=1,
            opacity=0.88,
            hovertemplate="Log10 Revenue: %{x:.2f}<br>Count: %{y:,}<extra></extra>"
        ))
        fig.update_xaxes(
            title_text="Worldwide Revenue (Log10 USD)",
            tickvals=[3, 4, 5, 6, 7, 8, 9],
            ticktext=["$1K", "$10K", "$100K", "$1M", "$10M", "$100M", "$1B"]
        )
    else:
        fig.add_trace(go.Histogram(
            x=rev / 1e6,
            nbinsx=50,
            marker_color=COLOR_CYAN,
            marker_line_color=BORDER_SUBTLE,
            marker_line_width=1,
            opacity=0.88,
            hovertemplate="Revenue: $%{x:.1f}M<br>Count: %{y:,}<extra></extra>"
        ))
        fig.update_xaxes(title_text="Worldwide Revenue (Million USD)")

    fig.update_yaxes(title_text="Movie Count")
    return apply_figure_theme(fig, title="Worldwide Box Office Revenue Distribution", height=380)


def plot_budget_vs_revenue(df: pd.DataFrame) -> go.Figure:
    """
    Interactive scatter plot showing the relationship between production budget
    and worldwide revenue with logarithmic axes.
    """
    if df is None or df.empty or "budget" not in df.columns or "revenue" not in df.columns:
        return create_empty_chart("Production Budget vs Worldwide Revenue", height=420)

    valid = df[(df["budget"] > 0) & (df["revenue"] > 0)].copy()
    if valid.empty:
        return create_empty_chart("Production Budget vs Worldwide Revenue (No Positive Budget/Revenue)", height=420)

    # Pre-format hover labels
    valid["budget_fmt"] = valid["budget"].apply(format_currency_compact)
    valid["revenue_fmt"] = valid["revenue"].apply(format_currency_compact)
    valid["multiple"] = (valid["revenue"] / valid["budget"]).round(2)
    valid["title_disp"] = valid["title"].fillna(valid.get("original_title", "Untitled")) if "title" in valid.columns else "Movie"

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=valid["budget"],
        y=valid["revenue"],
        mode="markers",
        marker=dict(
            size=6,
            color=valid["revenue"],
            colorscale="Viridis",
            opacity=0.75,
            showscale=False,
            line=dict(width=0.5, color=BORDER_SUBTLE)
        ),
        text=valid["title_disp"],
        customdata=np.stack((valid["budget_fmt"], valid["revenue_fmt"], valid["multiple"]), axis=-1),
        hovertemplate="<b>%{text}</b><br>Budget: %{customdata[0]}<br>Revenue: %{customdata[1]}<br>Gross Multiple: %{customdata[2]}x<extra></extra>"
    ))

    # Add 1:1 reference line
    min_b = float(valid["budget"].min())
    min_r = float(valid["revenue"].min())
    max_b = float(valid["budget"].max())
    max_r = float(valid["revenue"].max())

    min_val = max(1e4, min(min_b, min_r))
    max_val = max(max_b, max_r)

    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode="lines",
        name="1.0x Multiple Reference",
        line=dict(color=COLOR_WARNING, dash="dash", width=1.5),
        hoverinfo="skip"
    ))

    fig.update_xaxes(type="log", title_text="Production Budget (USD, Log Scale)")
    fig.update_yaxes(type="log", title_text="Worldwide Revenue (USD, Log Scale)")

    return apply_figure_theme(fig, title="Production Budget vs Worldwide Revenue", height=420)


def plot_revenue_by_genre(genre_stats_df: pd.DataFrame) -> go.Figure:
    """
    Compares median worldwide box office revenue across film genres
    with movie count indicators.
    """
    if genre_stats_df is None or genre_stats_df.empty or "genre" not in genre_stats_df.columns:
        return create_empty_chart("Median Worldwide Revenue by Genre", height=420)

    df_sorted = genre_stats_df.sort_values(by="median_revenue", ascending=True)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=df_sorted["genre"],
        x=df_sorted["median_revenue"] / 1e6,
        orientation="h",
        marker=dict(
            color=COLOR_CYAN,
            line=dict(color=BORDER_SUBTLE, width=1)
        ),
        text=[f" ${m/1e6:.1f}M ({c} films)" for m, c in zip(df_sorted["median_revenue"], df_sorted["movie_count"])],
        textposition="outside",
        textfont=dict(size=10, color=TEXT_MUTED),
        hovertemplate="<b>%{y}</b><br>Median Revenue: $%{x:.2f}M<extra></extra>"
    ))

    fig.update_xaxes(title_text="Median Worldwide Revenue (Million USD)")
    fig.update_yaxes(title_text="", tickfont=dict(color=TEXT_PRIMARY, size=11))

    return apply_figure_theme(fig, title="Median Worldwide Revenue by Genre (>= 10 Films)", height=420)


def plot_revenue_trend_over_time(yearly_stats_df: pd.DataFrame) -> go.Figure:
    """
    Visualizes median and mean worldwide box office revenue evolution over release years.
    """
    if yearly_stats_df is None or yearly_stats_df.empty or "release_year" not in yearly_stats_df.columns:
        return create_empty_chart("Box Office Revenue Trends Over Time", height=380)

    fig = go.Figure()

    # Mean Revenue Line
    fig.add_trace(go.Scatter(
        x=yearly_stats_df["release_year"],
        y=yearly_stats_df["mean_revenue"] / 1e6,
        mode="lines+markers",
        name="Mean Worldwide Revenue",
        line=dict(color=COLOR_CYAN, width=2.5),
        marker=dict(size=5),
        hovertemplate="<b>Year %{x}</b><br>Mean: $%{y:.1f}M<extra></extra>"
    ))

    # Median Revenue Line
    fig.add_trace(go.Scatter(
        x=yearly_stats_df["release_year"],
        y=yearly_stats_df["median_revenue"] / 1e6,
        mode="lines+markers",
        name="Median Worldwide Revenue",
        line=dict(color=COLOR_VIOLET, width=2, dash="dot"),
        marker=dict(size=4),
        hovertemplate="<b>Year %{x}</b><br>Median: $%{y:.1f}M<extra></extra>"
    ))

    fig.update_xaxes(title_text="Release Year")
    fig.update_yaxes(title_text="Worldwide Revenue (Million USD)")

    return apply_figure_theme(fig, title="Evolution of Worldwide Box Office Revenue (1970 - Present)", height=380)


def plot_budget_distribution(df: pd.DataFrame) -> go.Figure:
    """
    Plots the histogram of reported production budgets (excluding 0/unrecorded budgets).
    """
    if df is None or df.empty or "budget" not in df.columns:
        return create_empty_chart("Movie Production Budget Distribution", height=380)

    valid = df[df["budget"] > 0].copy()
    if valid.empty:
        return create_empty_chart("Movie Production Budget Distribution (No Positive Budgets)", height=380)

    budgets_m = valid["budget"] / 1e6

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=budgets_m,
        nbinsx=40,
        marker_color=COLOR_VIOLET,
        marker_line_color=BORDER_SUBTLE,
        marker_line_width=1,
        opacity=0.88,
        hovertemplate="Budget: $%{x:.1f}M<br>Movies: %{y:,}<extra></extra>"
    ))

    median_b = float(valid["budget"].median() / 1e6)
    fig.add_vline(
        x=median_b,
        line_dash="dash",
        line_color=COLOR_WARNING,
        line_width=1.5,
        annotation_text=f"Median: ${median_b:.1f}M",
        annotation_position="top right",
        annotation_font_color=COLOR_WARNING
    )

    fig.update_xaxes(title_text="Reported Production Budget (Million USD)")
    fig.update_yaxes(title_text="Movie Count")

    return apply_figure_theme(fig, title="Movie Production Budget Distribution (Reported Budgets)", height=380)


# -----------------------------------------------------------------------------
# Section B: Data Explorer Visualizations
# -----------------------------------------------------------------------------

def plot_missing_values_summary(missing_df: pd.DataFrame) -> go.Figure:
    """
    Renders a horizontal bar chart displaying missing value percentages by column.
    """
    if missing_df is None or missing_df.empty:
        fig = go.Figure()
        return apply_figure_theme(fig, title="Zero Missing Values Detected", height=240)

    df_sorted = missing_df.sort_values(by="missing_pct", ascending=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df_sorted["column"],
        x=df_sorted["missing_pct"],
        orientation="h",
        marker=dict(
            color=COLOR_WARNING,
            line=dict(color=BORDER_SUBTLE, width=1)
        ),
        text=[f" {pct:.1f}% ({cnt:,})" for pct, cnt in zip(df_sorted["missing_pct"], df_sorted["missing_count"])],
        textposition="outside",
        textfont=dict(size=10, color=TEXT_MUTED),
        hovertemplate="<b>%{y}</b><br>Missing: %{x:.2f}%<extra></extra>"
    ))

    fig.update_xaxes(title_text="Missing Percentage (%)", range=[0, max(100, float(df_sorted["missing_pct"].max()) + 15)])
    fig.update_yaxes(title_text="", tickfont=dict(color=TEXT_PRIMARY, size=11))

    return apply_figure_theme(fig, title="Missing Metadata Percentage by Column", height=320)


# -----------------------------------------------------------------------------
# Section C: ML Studio Visualizations
# -----------------------------------------------------------------------------

def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model"
) -> go.Figure:
    """
    Scatter plot comparing actual vs. predicted worldwide box office revenue on log scales.
    Includes y=x reference line indicating exact prediction parity.
    """
    if len(y_true) == 0 or len(y_pred) == 0:
        return create_empty_chart(f"Actual vs Predicted Revenue ({model_name})", height=420)

    y_true_c = np.clip(np.asarray(y_true, dtype=float), 1.0, None)
    y_pred_c = np.clip(np.asarray(y_pred, dtype=float), 1.0, None)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=y_true_c,
        y=y_pred_c,
        mode="markers",
        marker=dict(
            size=6,
            color=COLOR_CYAN,
            opacity=0.60,
            line=dict(width=0.5, color=BORDER_SUBTLE)
        ),
        name="Validation Movies",
        customdata=np.stack((
            [format_currency_compact(v) for v in y_true_c],
            [format_currency_compact(v) for v in y_pred_c]
        ), axis=-1),
        hovertemplate="Actual: %{customdata[0]}<br>Predicted: %{customdata[1]}<extra></extra>"
    ))

    min_val = min(float(y_true_c.min()), float(y_pred_c.min()))
    max_val = max(float(y_true_c.max()), float(y_pred_c.max()))

    # Parity identity line
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode="lines",
        name="Parity (y = x)",
        line=dict(color=COLOR_WARNING, dash="dash", width=1.5),
        hoverinfo="skip"
    ))

    fig.update_xaxes(type="log", title_text="Actual Worldwide Revenue (USD, Log Scale)")
    fig.update_yaxes(type="log", title_text="Predicted Worldwide Revenue (USD, Log Scale)")

    return apply_figure_theme(fig, title=f"Actual vs Predicted Revenue ({model_name})", height=420)


def plot_residuals_distribution(residuals: np.ndarray) -> go.Figure:
    """
    Visualizes the prediction residuals distribution in millions of USD.
    Explicitly documents outlier tail trimming for display stability.
    """
    if len(residuals) == 0:
        return create_empty_chart("Residual Error Distribution (Empty Sample)", height=380)

    res_m = np.asarray(residuals, dtype=float) / 1e6

    # Trim extreme 1% tails for visualization stability without distorting central variance
    q01, q99 = np.percentile(res_m, [1, 99])
    trimmed = res_m[(res_m >= q01) & (res_m <= q99)]

    if len(trimmed) == 0:
        trimmed = res_m

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=trimmed,
        nbinsx=40,
        marker_color=COLOR_BLUE,
        marker_line_color=BORDER_SUBTLE,
        marker_line_width=1,
        opacity=0.85,
        hovertemplate="Residual: $%{x:.1f}M<br>Count: %{y:,}<extra></extra>"
    ))

    fig.add_vline(
        x=0,
        line_color=COLOR_SUCCESS,
        line_width=1.5,
        annotation_text="Zero Error",
        annotation_position="top",
        annotation_font_color=COLOR_SUCCESS
    )

    fig.update_xaxes(title_text="Residual: (Actual - Predicted) Million USD [1st-99th percentile displayed]")
    fig.update_yaxes(title_text="Validation Count")

    return apply_figure_theme(fig, title="Residual Error Distribution", height=380)


def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """
    Renders top feature importances as a horizontal bar chart.
    """
    if importance_df is None or importance_df.empty or "importance" not in importance_df.columns:
        return create_empty_chart("Feature Importances Not Available", height=280)

    top_df = importance_df.head(top_n).sort_values(by="importance", ascending=True)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=top_df["feature"],
        x=top_df["importance"],
        orientation="h",
        marker=dict(
            color=COLOR_CYAN,
            line=dict(color=BORDER_SUBTLE, width=1)
        ),
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>"
    ))

    fig.update_xaxes(title_text="Relative Feature Importance")
    fig.update_yaxes(title_text="", tickfont=dict(color=TEXT_PRIMARY, size=11))

    return apply_figure_theme(fig, title=f"Top {top_n} Pre-Release Predictive Features", height=420)


# -----------------------------------------------------------------------------
# Section D: Revenue Insights & Feature Interactions
# -----------------------------------------------------------------------------

def plot_budget_vs_revenue_franchise(df: pd.DataFrame) -> go.Figure:
    """
    Renders Production Budget vs Worldwide Revenue with Franchise vs Standalone comparison.
    Both axes use logarithmic scale. Includes separate OLS trendlines for both categories.
    Excludes zero/missing budgets and non-positive revenue.
    """
    if df is None or df.empty or "budget" not in df.columns or "revenue" not in df.columns:
        return create_empty_chart("Budget vs Revenue: Franchise vs Standalone", height=480)

    # Filter movies with valid positive budget and revenue
    b_num = pd.to_numeric(df["budget"], errors="coerce")
    r_num = pd.to_numeric(df["revenue"], errors="coerce")
    valid = df[(b_num > 0) & (r_num > 0)].copy()
    if valid.empty:
        return create_empty_chart("Budget vs Revenue (No Positive Records)", height=480)

    valid["budget"] = pd.to_numeric(valid["budget"], errors="coerce")
    valid["revenue"] = pd.to_numeric(valid["revenue"], errors="coerce")
    valid["multiple"] = valid["revenue"] / valid["budget"]

    # Parse collection field consistently with feature engineering
    if "belongs_to_collection" in valid.columns:
        col = valid["belongs_to_collection"]
        is_franchise = (
            col.notna()
            & (col.astype(str).str.strip() != "")
            & (col.astype(str).str.strip() != "[]")
            & (col.astype(str).str.strip() != "{}")
            & (col.astype(str).str.strip().str.lower() != "nan")
        )
    else:
        is_franchise = pd.Series(False, index=valid.index)

    valid["category"] = np.where(is_franchise, "Franchise Movie", "Standalone Film")
    if "title" not in valid.columns:
        valid["title"] = "Movie #" + valid.index.astype(str)

    fig = go.Figure()

    category_config = [
        {"cat": "Standalone Film", "color": COLOR_VIOLET, "line_dash": "dash", "name": "Standalone Film"},
        {"cat": "Franchise Movie", "color": COLOR_CYAN, "line_dash": "dot", "name": "Franchise Movie"},
    ]

    for cfg in category_config:
        sub = valid[valid["category"] == cfg["cat"]].copy()
        if sub.empty:
            continue

        customdata = np.stack((
            sub["title"].fillna("Unknown Title"),
            sub["budget"].apply(format_currency_compact),
            sub["revenue"].apply(format_currency_compact),
            sub["category"],
            sub["multiple"]
        ), axis=-1)

        # Scatter points
        fig.add_trace(go.Scatter(
            x=sub["budget"],
            y=sub["revenue"],
            mode="markers",
            name=f"{cfg['name']} (N={len(sub):,})",
            marker=dict(
                size=6,
                color=cfg["color"],
                opacity=0.68,
                line=dict(width=0.5, color=BORDER_SUBTLE)
            ),
            customdata=customdata,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Classification: %{customdata[3]}<br>"
                "Budget: %{customdata[1]} ($%{x:,.0f})<br>"
                "Revenue: %{customdata[2]} ($%{y:,.0f})<br>"
                "Gross Multiple: %{customdata[4]:.2f}x<extra></extra>"
            )
        ))

        # OLS Trendline in log-log space
        if len(sub) > 5:
            log_x = np.log10(sub["budget"].values)
            log_y = np.log10(sub["revenue"].values)
            poly = np.polyfit(log_x, log_y, deg=1)
            x_line = np.linspace(log_x.min(), log_x.max(), 50)
            y_line = poly[0] * x_line + poly[1]
            fig.add_trace(go.Scatter(
                x=10 ** x_line,
                y=10 ** y_line,
                mode="lines",
                name=f"{cfg['name']} Trendline",
                line=dict(color=cfg["color"], width=2.2, dash=cfg["line_dash"]),
                hoverinfo="skip"
            ))

    # Reference 1.0x parity line: Revenue == Budget
    min_val = max(1e4, min(valid["budget"].min(), valid["revenue"].min()))
    max_val = max(valid["budget"].max(), valid["revenue"].max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode="lines",
        name="1.0x Parity (Rev = Budget)",
        line=dict(color=COLOR_WARNING, width=1.5, dash="dashdot"),
        hoverinfo="skip"
    ))

    fig.update_xaxes(type="log", title_text="Production Budget (USD, Log Scale)")
    fig.update_yaxes(type="log", title_text="Worldwide Revenue (USD, Log Scale)")

    return apply_figure_theme(
        fig,
        title="Production Budget vs Worldwide Revenue: Franchise vs Standalone",
        height=480
    )


def plot_genre_budget_heatmap(df: pd.DataFrame, min_sample: int = 10) -> go.Figure:
    """
    Renders interactive Heatmap of Genre vs Budget Quartiles.
    Cell color reflects median worldwide revenue on a log color scale.
    Cell annotations display the sample count (number of movies).
    Cells with fewer than min_sample movies are masked as insufficient sample size.
    A movie belonging to multiple genres contributes to each applicable genre.
    """
    if df is None or df.empty or "budget" not in df.columns or "revenue" not in df.columns or "genres" not in df.columns:
        return create_empty_chart("Genre vs Budget Category Revenue Heatmap", height=540)

    b_num = pd.to_numeric(df["budget"], errors="coerce")
    r_num = pd.to_numeric(df["revenue"], errors="coerce")
    valid = df[(b_num > 0) & (r_num > 0)].copy()
    if valid.empty:
        return create_empty_chart("Genre vs Budget Category (No Positive Records)", height=540)

    valid["budget"] = pd.to_numeric(valid["budget"], errors="coerce")
    valid["revenue"] = pd.to_numeric(valid["revenue"], errors="coerce")

    # Quantiles for budget categories (4 quartiles)
    try:
        q_bins = valid["budget"].quantile([0.0, 0.25, 0.50, 0.75, 1.0]).values
        q_bins[0] = 0.0
        for i in range(1, len(q_bins)):
            if q_bins[i] <= q_bins[i-1]:
                q_bins[i] = q_bins[i-1] + 1.0

        q_labels = [
            f"Q1: Lower\n(≤ {format_currency_compact(q_bins[1])})",
            f"Q2: Lower-Mid\n({format_currency_compact(q_bins[1])} - {format_currency_compact(q_bins[2])})",
            f"Q3: Upper-Mid\n({format_currency_compact(q_bins[2])} - {format_currency_compact(q_bins[3])})",
            f"Q4: Higher\n(> {format_currency_compact(q_bins[3])})"
        ]
        valid["budget_cat"] = pd.cut(valid["budget"], bins=q_bins, labels=q_labels, include_lowest=True)
    except Exception:
        valid["budget_cat"] = pd.qcut(valid["budget"], q=4, duplicates="drop")
        q_labels = [str(c) for c in valid["budget_cat"].cat.categories]

    # Expand multi-genre memberships
    rows = []
    for _, row in valid.iterrows():
        b_cat = row["budget_cat"]
        rev = row["revenue"]
        bud = row["budget"]
        parsed_g = safe_parse_json(row.get("genres"))
        for g in parsed_g:
            if isinstance(g, dict) and "name" in g and str(g["name"]).strip():
                rows.append({
                    "genre": str(g["name"]).strip(),
                    "budget_cat": str(b_cat),
                    "revenue": rev,
                    "budget": bud
                })

    if not rows:
        return create_empty_chart("Genre vs Budget Category (No Parsable Genres)", height=540)

    gdf = pd.DataFrame(rows)
    genre_totals = gdf.groupby("genre")["revenue"].count()
    top_genres = genre_totals[genre_totals >= 15].index.tolist()
    gdf = gdf[gdf["genre"].isin(top_genres)]

    piv_count = gdf.pivot_table(index="genre", columns="budget_cat", values="revenue", aggfunc="count", fill_value=0)
    piv_med = gdf.pivot_table(index="genre", columns="budget_cat", values="revenue", aggfunc="median")

    piv_count = piv_count.reindex(columns=q_labels, fill_value=0)
    piv_med = piv_med.reindex(columns=q_labels)

    genre_sort_order = gdf.groupby("genre")["revenue"].median().sort_values(ascending=True).index
    piv_count = piv_count.reindex(index=genre_sort_order)
    piv_med = piv_med.reindex(index=genre_sort_order)

    z_vals = []
    text_annotations = []
    hover_texts = []

    for g in piv_med.index:
        z_row = []
        t_row = []
        h_row = []
        for c in q_labels:
            cnt = int(piv_count.loc[g, c]) if g in piv_count.index and c in piv_count.columns else 0
            med = piv_med.loc[g, c] if g in piv_med.index and c in piv_med.columns else np.nan

            if cnt >= min_sample and pd.notna(med) and med > 0:
                z_row.append(np.log10(med))
                t_row.append(f"{cnt}<br><span style='font-size:9px;'>{format_currency_compact(med)}</span>")
                h_row.append(
                    f"<b>Genre:</b> {g}<br>"
                    f"<b>Budget Tier:</b> {c.replace(chr(10), ' ')}<br>"
                    f"<b>Median Revenue:</b> ${med:,.0f} ({format_currency_compact(med)})<br>"
                    f"<b>Movie Count:</b> {cnt} films"
                )
            else:
                z_row.append(None)
                if cnt > 0:
                    t_row.append(f"N={cnt}<br><span style='font-size:8px;color:#94A3B8;'>(&lt;{min_sample})</span>")
                    h_row.append(
                        f"<b>Genre:</b> {g}<br>"
                        f"<b>Budget Tier:</b> {c.replace(chr(10), ' ')}<br>"
                        f"<b>Median Revenue:</b> Insufficient sample (&lt;{min_sample} films)<br>"
                        f"<b>Movie Count:</b> {cnt} films"
                    )
                else:
                    t_row.append("-")
                    h_row.append(f"<b>Genre:</b> {g}<br><b>Tier:</b> {c.replace(chr(10), ' ')}<br>No movies recorded")
        z_vals.append(z_row)
        text_annotations.append(t_row)
        hover_texts.append(h_row)

    fig = go.Figure(data=go.Heatmap(
        z=z_vals,
        x=q_labels,
        y=list(piv_med.index),
        text=text_annotations,
        texttemplate="%{text}",
        textfont=dict(size=10, color="#FFFFFF"),
        hovertext=hover_texts,
        hoverinfo="text",
        colorscale=[
            [0.0, "#0B1528"],
            [0.2, "#1E2B44"],
            [0.4, "#1E3A8A"],
            [0.65, "#0284C7"],
            [0.85, "#00D2FF"],
            [1.0, "#38BDF8"]
        ],
        colorbar=dict(
            title=dict(text="Log10 Median Revenue", font=dict(size=11, color=TEXT_MUTED)),
            tickvals=[6, 7, 8, 8.5],
            ticktext=["$1M", "$10M", "$100M", "$316M"],
            tickfont=dict(size=10, color=TEXT_MUTED),
            len=0.85
        )
    ))

    fig.update_xaxes(title_text="Production Budget Quartile", tickfont=dict(size=11, color=TEXT_PRIMARY))
    fig.update_yaxes(title_text="Film Genre", tickfont=dict(size=11, color=TEXT_PRIMARY))

    return apply_figure_theme(
        fig,
        title=f"Genre vs Budget Category Revenue Heatmap (Min {min_sample} Films)",
        height=540
    )


def plot_shap_summary_bar(shap_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """
    Renders top features ranked by mean absolute SHAP value (impact on log1p revenue prediction).
    """
    if shap_df is None or shap_df.empty or "mean_abs_shap" not in shap_df.columns:
        return create_empty_chart("SHAP Summary Not Available", height=420)

    top_df = shap_df.head(top_n).sort_values(by="mean_abs_shap", ascending=True)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=top_df["feature"],
        x=top_df["mean_abs_shap"],
        orientation="h",
        marker=dict(
            color=COLOR_CYAN,
            line=dict(color=BORDER_SUBTLE, width=1)
        ),
        hovertemplate="<b>%{y}</b><br>Mean |SHAP| (Log Space): %{x:.4f}<extra></extra>"
    ))

    fig.update_xaxes(title_text="Mean Absolute SHAP Value (Impact on Log-Revenue Space)")
    fig.update_yaxes(title_text="", tickfont=dict(color=TEXT_PRIMARY, size=11))

    return apply_figure_theme(
        fig,
        title=f"Top {top_n} Features by Model Impact (Mean |SHAP| in Log-Revenue Space)",
        height=440
    )


# -----------------------------------------------------------------------------
# Section E: Model Comparison Visualizations
# -----------------------------------------------------------------------------

def plot_model_comparison_rmsle(comp_df: pd.DataFrame) -> go.Figure:
    """
    Graph A: RMSLE Comparison Across Models.
    Displays 5-fold CV mean RMSLE with standard deviation error bars alongside Holdout RMSLE.
    Lower is better.
    """
    if comp_df is None or comp_df.empty or "Model" not in comp_df.columns:
        return create_empty_chart("RMSLE Comparison Across Models", height=440)

    df_sorted = comp_df.sort_values(by="CV_RMSLE_Mean", ascending=True).copy()

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="5-Fold CV Mean RMSLE",
        x=df_sorted["Model"],
        y=df_sorted["CV_RMSLE_Mean"],
        error_y=dict(
            type="data",
            array=df_sorted["CV_RMSLE_Std"],
            visible=True,
            color=TEXT_PRIMARY,
            thickness=1.2,
            width=4
        ),
        marker=dict(color=COLOR_CYAN, line=dict(color=BORDER_SUBTLE, width=1)),
        hovertemplate="<b>%{x}</b><br>CV Mean RMSLE: %{y:.4f}<extra></extra>"
    ))

    fig.add_trace(go.Bar(
        name="Holdout RMSLE (20% Split)",
        x=df_sorted["Model"],
        y=df_sorted["Holdout_RMSLE"],
        marker=dict(color=COLOR_VIOLET, line=dict(color=BORDER_SUBTLE, width=1)),
        hovertemplate="<b>%{x}</b><br>Holdout RMSLE: %{y:.4f}<extra></extra>"
    ))

    fig.update_layout(
        barmode="group",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color=TEXT_MUTED)
        ),
        xaxis=dict(tickangle=-30, tickfont=dict(size=10, color=TEXT_PRIMARY)),
        yaxis=dict(title="RMSLE (Lower is Better)")
    )

    return apply_figure_theme(fig, title="Graph A: Cross-Validation vs Holdout RMSLE Across Models", height=440)


def plot_model_comparison_mae(comp_df: pd.DataFrame) -> go.Figure:
    """
    Graph B: Holdout MAE Comparison Across Models in USD.
    Lower is better.
    """
    if comp_df is None or comp_df.empty or "Holdout_MAE" not in comp_df.columns:
        return create_empty_chart("MAE Comparison Across Models", height=420)

    df_sorted = comp_df.sort_values(by="Holdout_MAE", ascending=True).copy()
    text_labels = [format_currency_compact(v) for v in df_sorted["Holdout_MAE"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_sorted["Model"],
        y=df_sorted["Holdout_MAE"],
        text=text_labels,
        textposition="outside",
        textfont=dict(size=10, color=TEXT_PRIMARY),
        marker=dict(color=COLOR_BLUE, line=dict(color=BORDER_SUBTLE, width=1)),
        hovertemplate="<b>%{x}</b><br>Holdout MAE: $%{y:,.0f}<extra></extra>"
    ))

    fig.update_layout(
        xaxis=dict(tickangle=-30, tickfont=dict(size=10, color=TEXT_PRIMARY)),
        yaxis=dict(title="Holdout MAE in USD (Lower is Better)", tickprefix="$")
    )

    return apply_figure_theme(fig, title="Graph B: Holdout Mean Absolute Error (MAE in USD)", height=420)


def plot_model_comparison_r2(comp_df: pd.DataFrame) -> go.Figure:
    """
    Graph C: Holdout R2 Comparison Across Models.
    Higher is better.
    """
    if comp_df is None or comp_df.empty or "Holdout_R2" not in comp_df.columns:
        return create_empty_chart("R2 Comparison Across Models", height=420)

    df_sorted = comp_df.sort_values(by="Holdout_R2", ascending=False).copy()
    text_labels = [f"{v:.4f}" for v in df_sorted["Holdout_R2"]]
    colors = [COLOR_CYAN if i == 0 else COLOR_SUCCESS for i in range(len(df_sorted))]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_sorted["Model"],
        y=df_sorted["Holdout_R2"],
        text=text_labels,
        textposition="outside",
        textfont=dict(size=10, color=TEXT_PRIMARY),
        marker=dict(color=colors, line=dict(color=BORDER_SUBTLE, width=1)),
        hovertemplate="<b>%{x}</b><br>Holdout R²: %{y:.4f}<extra></extra>"
    ))

    max_r2 = max(0.8, float(df_sorted["Holdout_R2"].max()) * 1.15) if not df_sorted.empty else 1.0

    fig.update_layout(
        xaxis=dict(tickangle=-30, tickfont=dict(size=10, color=TEXT_PRIMARY)),
        yaxis=dict(title="Holdout R² Variance Explained (Higher is Better)", range=[-0.25, max_r2])
    )

    return apply_figure_theme(fig, title="Graph C: Holdout Explained Variance (R²)", height=420)


def plot_training_time_vs_rmsle(comp_df: pd.DataFrame) -> go.Figure:
    """
    Graph D: Training Time vs Prediction Performance.
    Scatter plot of training time (seconds) vs Holdout RMSLE.
    Annotated with model names.
    """
    if comp_df is None or comp_df.empty or "Training_Time_Sec" not in comp_df.columns:
        return create_empty_chart("Training Time vs RMSLE", height=440)

    sub = comp_df[comp_df["Model_Key"] != "dummy"].copy()
    if sub.empty:
        sub = comp_df.copy()

    short_names = [m.replace(" Regressor", "").replace(" Baseline", "") for m in sub["Model"]]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sub["Training_Time_Sec"],
        y=sub["Holdout_RMSLE"],
        mode="markers+text",
        text=short_names,
        textposition="top right",
        textfont=dict(size=10, color=TEXT_PRIMARY),
        marker=dict(
            size=10,
            color=np.where(sub["Is_Tuned"], COLOR_CYAN, COLOR_VIOLET),
            line=dict(color=BORDER_SUBTLE, width=1.5),
            symbol=np.where(sub["Is_Tuned"], "diamond", "circle")
        ),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Training Time: %{x:.2f}s<br>"
            "Holdout RMSLE: %{y:.4f}<extra></extra>"
        )
    ))

    fig.update_layout(
        xaxis=dict(title="Model Training Time on 80% Split (Seconds)", rangemode="tozero"),
        yaxis=dict(title="Holdout RMSLE (Lower is Better)")
    )

    return apply_figure_theme(fig, title="Graph D: Training Computational Cost vs Holdout Performance", height=440)


def plot_tuning_improvement(comp_df: pd.DataFrame) -> go.Figure:
    """
    Graph E: Hyperparameter Tuning Improvement (Default vs Tuned CV RMSLE).
    Compares baseline vs tuned 5-fold CV RMSLE for applicable architectures.
    """
    if comp_df is None or comp_df.empty:
        return create_empty_chart("Hyperparameter Tuning Improvement", height=440)

    keys = ["lightgbm", "random_forest", "xgboost", "hist_gradient_boosting", "extra_trees", "catboost"]
    pairs = []
    for k in keys:
        base_rows = comp_df[(comp_df["Model_Key"] == k) & (~comp_df["Is_Tuned"])]
        tuned_rows = comp_df[(comp_df["Model_Key"] == k) & (comp_df["Is_Tuned"])]
        if not base_rows.empty and not tuned_rows.empty:
            b_rmsle = float(base_rows.iloc[0]["CV_RMSLE_Mean"])
            t_rmsle = float(tuned_rows.iloc[0]["CV_RMSLE_Mean"])
            delta = b_rmsle - t_rmsle
            pct = (delta / b_rmsle) * 100
            name = str(base_rows.iloc[0]["Model"]).replace(" (Baseline)", "")
            pairs.append({
                "model": name,
                "base_rmsle": b_rmsle,
                "tuned_rmsle": t_rmsle,
                "delta": delta,
                "pct": pct
            })

    if not pairs:
        return create_empty_chart("No Baseline vs Tuned Model Pairs Available", height=440)

    df_pairs = pd.DataFrame(pairs).sort_values(by="delta", ascending=False)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Default Baseline CV RMSLE",
        x=df_pairs["model"],
        y=df_pairs["base_rmsle"],
        marker=dict(color=COLOR_VIOLET, line=dict(color=BORDER_SUBTLE, width=1)),
        hovertemplate="<b>%{x}</b><br>Default CV RMSLE: %{y:.4f}<extra></extra>"
    ))

    fig.add_trace(go.Bar(
        name="Tuned Model CV RMSLE",
        x=df_pairs["model"],
        y=df_pairs["tuned_rmsle"],
        marker=dict(color=COLOR_CYAN, line=dict(color=BORDER_SUBTLE, width=1)),
        hovertemplate="<b>%{x}</b><br>Tuned CV RMSLE: %{y:.4f}<extra></extra>"
    ))

    fig.update_layout(
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11, color=TEXT_MUTED)),
        xaxis=dict(tickangle=-25, tickfont=dict(size=10, color=TEXT_PRIMARY)),
        yaxis=dict(title="5-Fold CV RMSLE (Lower is Better)")
    )

    return apply_figure_theme(fig, title="Graph E: Hyperparameter Tuning Impact (Default vs Tuned CV RMSLE)", height=440)
