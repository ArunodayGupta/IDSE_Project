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
