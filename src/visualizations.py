"""
visualizations.py
Interactive Plotly visualizations for TMDB Movie Box Office Revenue platform.

Charts:
1. plot_revenue_distribution: Skewed revenue distribution with log/linear scale support.
2. plot_budget_vs_revenue: Production budget vs worldwide revenue interactive scatter.
3. plot_revenue_by_genre: Median revenue and movie counts across film genres.
4. plot_revenue_trend_over_time: Evolution of box office revenue across release years.
5. plot_budget_distribution: Distribution of production budgets.
6. plot_actual_vs_predicted: Regression evaluation scatter with y=x parity reference.
7. plot_residuals_distribution: Distribution of prediction residuals.
8. plot_feature_importance: Top model feature importance rankings.
9. plot_missing_values_summary: Visual summary of missing metadata across columns.
"""

from typing import List, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

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


# -----------------------------------------------------------------------------
# Section A: Overview Visualizations
# -----------------------------------------------------------------------------

def plot_revenue_distribution(df: pd.DataFrame, log_scale: bool = True) -> go.Figure:
    """
    Renders the worldwide movie revenue distribution.
    Supports logarithmic x-axis to accommodate extreme skewness ($1 to $1.5B+).
    """
    valid = df[df["revenue"].notna() & (df["revenue"] > 0)].copy()
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
    valid = df[(df["budget"] > 0) & (df["revenue"] > 0)].copy()

    # Pre-format hover labels
    valid["budget_fmt"] = valid["budget"].apply(format_currency_compact)
    valid["revenue_fmt"] = valid["revenue"].apply(format_currency_compact)
    valid["multiple"] = (valid["revenue"] / valid["budget"]).round(2)
    valid["title_disp"] = valid["title"].fillna(valid.get("original_title", "Untitled"))

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
        hovertemplate="<b>%{text}</b><br>Budget: %{customdata[0]}<br>Revenue: %{customdata[1]}<br>Multiple: %{customdata[2]}x<extra></extra>"
    ))

    # Add 1:1 reference line
    min_val = max(1e4, min(valid["budget"].min(), valid["revenue"].min()))
    max_val = max(valid["budget"].max(), valid["revenue"].max())
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
    df_sorted = genre_stats_df.sort_values(by="median_revenue", ascending=True)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=df_sorted["genre"],
        x=df_sorted["median_revenue"] / 1e6,
        orientation="h",
        marker=dict(
            color=COLOR_BLUE,
            line=dict(color=BORDER_SUBTLE, width=1)
        ),
        customdata=np.stack((
            df_sorted["median_revenue"].apply(format_currency_compact),
            df_sorted["movie_count"],
            df_sorted["mean_revenue"].apply(format_currency_compact)
        ), axis=-1),
        hovertemplate="<b>%{y}</b><br>Median Revenue: %{customdata[0]}<br>Average Revenue: %{customdata[2]}<br>Movies: %{customdata[1]:,}<extra></extra>"
    ))

    fig.update_xaxes(title_text="Median Worldwide Revenue (Million USD)")
    fig.update_yaxes(title_text="", tickfont=dict(color=TEXT_PRIMARY, size=11))

    return apply_figure_theme(fig, title="Median Worldwide Revenue by Major Genre", height=440)


def plot_revenue_trend_over_time(yearly_stats_df: pd.DataFrame) -> go.Figure:
    """
    Visualizes box office revenue trends across release years.
    Displays median revenue and average revenue lines with markers.
    """
    df = yearly_stats_df.sort_values(by="release_year").reset_index(drop=True)

    fig = go.Figure()

    # Median revenue line
    fig.add_trace(go.Scatter(
        x=df["release_year"],
        y=df["median_revenue"] / 1e6,
        mode="lines+markers",
        name="Median Revenue",
        line=dict(color=COLOR_CYAN, width=2.5),
        marker=dict(size=5, color=COLOR_CYAN),
        customdata=df["movie_count"],
        hovertemplate="Year %{x}<br>Median Revenue: $%{y:.1f}M<br>Sample: %{customdata:,} films<extra></extra>"
    ))

    # Mean revenue line
    fig.add_trace(go.Scatter(
        x=df["release_year"],
        y=df["mean_revenue"] / 1e6,
        mode="lines",
        name="Average Revenue",
        line=dict(color=COLOR_VIOLET, width=1.8, dash="dot"),
        hovertemplate="Year %{x}<br>Average Revenue: $%{y:.1f}M<extra></extra>"
    ))

    fig.update_xaxes(title_text="Release Year")
    fig.update_yaxes(title_text="Worldwide Revenue (Million USD)")

    return apply_figure_theme(fig, title="Worldwide Box Office Trends Over Time", height=380)


def plot_budget_distribution(df: pd.DataFrame) -> go.Figure:
    """
    Shows how production budgets are distributed, focusing on non-zero reported budgets.
    """
    valid = df[df["budget"] > 0].copy()
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

    return apply_figure_theme(fig, title="Movie Production Budget Distribution (Non-Zero Budgets)", height=380)


# -----------------------------------------------------------------------------
# Section B: Data Explorer Visualizations
# -----------------------------------------------------------------------------

def plot_missing_values_summary(missing_df: pd.DataFrame) -> go.Figure:
    """
    Renders a clean horizontal bar chart displaying missing value percentages by column.
    """
    if missing_df.empty:
        fig = go.Figure()
        return apply_figure_theme(fig, title="Zero Missing Values Detected", height=240)

    df_sorted = missing_df.sort_values(by="missing_pct", ascending=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df_sorted["column"],
        x=df_sorted["missing_pct"],
        orientation="h",
        marker=dict(
            color=COLOR_DANGER,
            opacity=0.85,
            line=dict(color=BORDER_SUBTLE, width=1)
        ),
        customdata=df_sorted["missing_count"],
        hovertemplate="<b>%{y}</b><br>Missing: %{x:.1f}% (%{customdata:,} rows)<extra></extra>"
    ))

    fig.update_xaxes(title_text="Missing Percentage (%)", range=[0, 100])
    fig.update_yaxes(title_text="", tickfont=dict(color=TEXT_PRIMARY, size=11))

    return apply_figure_theme(fig, title="Missing Metadata by Column", height=max(260, len(df_sorted) * 22))


# -----------------------------------------------------------------------------
# Section C: ML Studio Visualizations
# -----------------------------------------------------------------------------

def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Selected Model"
) -> go.Figure:
    """
    Renders an actual vs predicted revenue scatter plot on a logarithmic scale
    with a parity identity reference line.
    """
    y_true_c = np.clip(y_true, 1, None)
    y_pred_c = np.clip(y_pred, 1, None)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=y_true_c,
        y=y_pred_c,
        mode="markers",
        marker=dict(
            size=6,
            color=COLOR_CYAN,
            opacity=0.65,
            line=dict(color=BORDER_SUBTLE, width=0.5)
        ),
        name="Validation Sample",
        customdata=np.stack((
            [format_currency_compact(v) for v in y_true_c],
            [format_currency_compact(v) for v in y_pred_c]
        ), axis=-1),
        hovertemplate="Actual: %{customdata[0]}<br>Predicted: %{customdata[1]}<extra></extra>"
    ))

    min_val = min(y_true_c.min(), y_pred_c.min())
    max_val = max(y_true_c.max(), y_pred_c.max())

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
    """
    res_m = residuals / 1e6

    # Trim extreme 1% tails for visualization stability
    q01, q99 = np.percentile(res_m, [1, 99])
    trimmed = res_m[(res_m >= q01) & (res_m <= q99)]

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

    fig.update_xaxes(title_text="Prediction Residual: (Actual - Predicted) Million USD")
    fig.update_yaxes(title_text="Validation Count")

    return apply_figure_theme(fig, title="Residual Error Distribution", height=380)


def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """
    Renders top feature importances as a clean horizontal bar chart.
    """
    if importance_df.empty or "importance" not in importance_df.columns:
        fig = go.Figure()
        return apply_figure_theme(fig, title="Feature Importances Not Available", height=280)

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
