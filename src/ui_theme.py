"""
ui_theme.py
Single canonical design system and UI component library for DataLab.

Centralizes:
1. Theme tokens: modern analytics palette (deep midnight-navy surfaces, electric cyan/blue accents,
   restrained violet secondary, emerald green success, high-contrast typography).
2. Global Streamlit stylesheet injection (Inter typography, refined cards, custom buttons, inputs).
3. Consistent Plotly figure formatting matching the dark analytics workspace theme.
4. Component renderers: page headers, KPI metric cards, callouts, status badges, empty states.
5. HTML escaping utility for safe rendering of user/dataset values.
"""

from typing import Dict, Any, List, Optional
import html
import streamlit as st
import plotly.graph_objects as go


# =============================================================================
# 1. DESIGN SYSTEM TOKENS (Modern Analytics Palette)
# =============================================================================

# Surface Colors
BG_APP = "#0B0F19"             # Deepest canvas background
BG_SIDEBAR = "#0E1424"         # Refined sidebar navy
BG_CARD = "#141C2E"            # Elevated card surface
BG_CARD_HOVER = "#1A253C"      # Interactive hover card surface
BG_INPUT = "#0D1424"           # Input / form control background
BG_MUTED = "#1E293B"           # Secondary container / badge background

# Borders & Separators
BORDER_SUBTLE = "#1E2B44"      # Default container border
BORDER_MEDIUM = "#2B3B59"      # Elevated container border
BORDER_FOCUS = "#00D2FF"       # Primary focus / active border

# Primary & Secondary Accents
COLOR_CYAN = "#00D2FF"         # Primary electric cyan accent
COLOR_BLUE = "#38BDF8"         # Sky blue accent
COLOR_VIOLET = "#818CF8"       # Restrained secondary violet accent
COLOR_PURPLE = "#A78BFA"       # Deep violet accent

# Status Colors
COLOR_SUCCESS = "#10B981"      # Emerald green (successful operations / positive status)
COLOR_SUCCESS_BG = "rgba(16, 185, 129, 0.12)"
COLOR_WARNING = "#F59E0B"      # Amber warning (stale data / cautions)
COLOR_WARNING_BG = "rgba(245, 158, 11, 0.12)"
COLOR_DANGER = "#EF4444"       # Rose red (errors / dropped rows)
COLOR_DANGER_BG = "rgba(239, 68, 68, 0.12)"
COLOR_INFO = "#38BDF8"         # Informational blue
COLOR_INFO_BG = "rgba(56, 189, 248, 0.12)"

# Typography Colors
TEXT_PRIMARY = "#F8FAFC"       # Crisp high-contrast headings & primary values
TEXT_SECONDARY = "#CBD5E1"     # Readable body text & labels
TEXT_MUTED = "#94A3B8"         # Subtitles, captions, metadata
TEXT_FAINT = "#64748B"         # Micro-labels, disabled text

# Plotly Palette Series
CHART_COLORWAY = [
    COLOR_CYAN,
    COLOR_VIOLET,
    COLOR_BLUE,
    COLOR_SUCCESS,
    COLOR_WARNING,
    "#EC4899",                 # Pink accent
    "#F97316",                 # Orange accent
    "#14B8A6"                  # Teal accent
]


# =============================================================================
# 2. HTML ESCAPING UTILITY
# =============================================================================

def escape_html(val: Any) -> str:
    """Safely escapes text for direct HTML injection."""
    if val is None:
        return ""
    return html.escape(str(val))


# =============================================================================
# 3. GLOBAL CUSTOM THEME INJECTION
# =============================================================================

def inject_custom_theme() -> None:
    """
    Injects the complete modern analytics workspace stylesheet into the active Streamlit app.
    Enforces deep graphite / midnight-navy surfaces, electric cyan accents, refined cards,
    and crisp high-contrast typography.
    """
    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

        /* Global Typography & Base Surfaces */
        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            color: {TEXT_PRIMARY};
        }}

        .stApp {{
            background-color: {BG_APP};
            color: {TEXT_PRIMARY};
        }}

        /* Clean Sidebar */
        [data-testid="stSidebar"] {{
            background-color: {BG_SIDEBAR};
            border-right: 1px solid {BORDER_SUBTLE};
        }}
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] label {{
            color: {TEXT_SECONDARY} !important;
        }}
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {{
            color: {TEXT_PRIMARY} !important;
            font-weight: 700;
        }}

        /* Main Content Container Max-Width & Spacing */
        .block-container {{
            padding-top: 1.8rem;
            padding-bottom: 3.5rem;
            padding-left: 2.2rem;
            padding-right: 2.2rem;
            max-width: 1440px;
        }}

        /* Sleek Cards */
        .datalab-card, .metric-card {{
            background: linear-gradient(180deg, rgba(20, 28, 46, 0.95) 0%, rgba(14, 20, 34, 0.95) 100%);
            border: 1px solid {BORDER_SUBTLE};
            border-top: 2px solid {COLOR_CYAN};
            border-radius: 10px;
            padding: 18px 20px;
            box-shadow: 0 4px 18px rgba(0, 0, 0, 0.3);
            transition: transform 0.15s ease, border-color 0.15s ease, box-shadow 0.15s ease;
        }}
        .datalab-card:hover, .metric-card:hover {{
            border-color: {COLOR_BLUE};
            box-shadow: 0 8px 24px rgba(0, 210, 255, 0.1);
        }}

        .metric-title {{
            color: {TEXT_MUTED};
            font-size: 0.78rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 6px;
        }}
        .metric-value {{
            color: {TEXT_PRIMARY};
            font-size: 2.0rem;
            font-weight: 800;
            line-height: 1.15;
            letter-spacing: -0.02em;
        }}
        .metric-sub {{
            color: {COLOR_CYAN};
            font-size: 0.82rem;
            font-weight: 500;
            margin-top: 6px;
        }}
        .metric-sub-neutral {{
            color: {TEXT_MUTED};
            font-size: 0.82rem;
            font-weight: 500;
            margin-top: 6px;
        }}

        /* Page Headers */
        .page-header-box {{
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            margin-bottom: 22px;
            padding-bottom: 14px;
            border-bottom: 1px solid {BORDER_SUBTLE};
        }}
        .page-header-title {{
            font-size: 1.95rem;
            font-weight: 800;
            letter-spacing: -0.025em;
            color: {TEXT_PRIMARY};
            margin: 0;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .page-header-sub {{
            font-size: 0.95rem;
            color: {TEXT_MUTED};
            margin-top: 4px;
            margin-bottom: 0;
            line-height: 1.5;
        }}

        /* Section Headings */
        .section-header-box {{
            margin-top: 24px;
            margin-bottom: 14px;
            display: flex;
            align-items: baseline;
            gap: 12px;
            border-bottom: 1px solid rgba(34, 49, 71, 0.4);
            padding-bottom: 8px;
        }}
        .section-header-title {{
            font-size: 1.22rem;
            font-weight: 700;
            color: {TEXT_PRIMARY};
            letter-spacing: -0.01em;
            margin: 0;
        }}
        .section-header-sub {{
            font-size: 0.85rem;
            color: {TEXT_MUTED};
            margin: 0;
        }}

        /* Callouts / Insight Boxes */
        .insight-box, .callout-box {{
            background: linear-gradient(180deg, rgba(20, 28, 46, 0.85) 0%, rgba(14, 20, 34, 0.85) 100%);
            border: 1px solid {BORDER_SUBTLE};
            border-left: 4px solid {COLOR_CYAN};
            border-radius: 8px;
            padding: 16px 20px;
            margin: 16px 0;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
        }}
        .insight-box.warning, .callout-box.warning {{
            border-left-color: {COLOR_WARNING};
        }}
        .insight-box.success, .callout-box.success {{
            border-left-color: {COLOR_SUCCESS};
        }}
        .insight-box.danger, .callout-box.danger {{
            border-left-color: {COLOR_DANGER};
        }}
        .insight-title {{
            color: {COLOR_CYAN};
            font-size: 1.02rem;
            font-weight: 700;
            margin-bottom: 6px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .insight-box.warning .insight-title {{ color: {COLOR_WARNING}; }}
        .insight-box.success .insight-title {{ color: {COLOR_SUCCESS}; }}
        .insight-box.danger .insight-title {{ color: {COLOR_DANGER}; }}
        .insight-body {{
            color: {TEXT_SECONDARY};
            font-size: 0.92rem;
            line-height: 1.6;
        }}

        /* Status Pills & Badges */
        .badge-pill, .status-pill {{
            display: inline-flex;
            align-items: center;
            gap: 5px;
            padding: 3px 10px;
            border-radius: 9999px;
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            background-color: rgba(0, 210, 255, 0.12);
            color: {COLOR_CYAN};
            border: 1px solid rgba(0, 210, 255, 0.25);
        }}
        .badge-pill.success, .status-pill.success {{
            background-color: {COLOR_SUCCESS_BG};
            color: {COLOR_SUCCESS};
            border-color: rgba(16, 185, 129, 0.3);
        }}
        .badge-pill.warning, .status-pill.warning {{
            background-color: {COLOR_WARNING_BG};
            color: {COLOR_WARNING};
            border-color: rgba(245, 158, 11, 0.3);
        }}
        .badge-pill.danger, .status-pill.danger {{
            background-color: {COLOR_DANGER_BG};
            color: {COLOR_DANGER};
            border-color: rgba(239, 68, 68, 0.3);
        }}
        .badge-pill.neutral, .status-pill.neutral {{
            background-color: rgba(148, 163, 184, 0.12);
            color: {TEXT_MUTED};
            border-color: rgba(148, 163, 184, 0.25);
        }}

        /* Empty State Box */
        .empty-state-box {{
            text-align: center;
            padding: 42px 24px;
            background: rgba(20, 28, 46, 0.5);
            border: 1px dashed {BORDER_MEDIUM};
            border-radius: 12px;
            margin: 20px 0;
        }}
        .empty-state-icon {{
            font-size: 2.4rem;
            margin-bottom: 12px;
            opacity: 0.85;
        }}
        .empty-state-title {{
            font-size: 1.15rem;
            font-weight: 700;
            color: {TEXT_PRIMARY};
            margin-bottom: 6px;
        }}
        .empty-state-desc {{
            font-size: 0.90rem;
            color: {TEXT_MUTED};
            max-width: 480px;
            margin: 0 auto;
            line-height: 1.5;
        }}

        /* Polished Form Inputs & Buttons */
        div[data-baseweb="select"] > div {{
            background-color: {BG_INPUT} !important;
            border-color: {BORDER_SUBTLE} !important;
            color: {TEXT_PRIMARY} !important;
        }}
        input, textarea {{
            background-color: {BG_INPUT} !important;
            color: {TEXT_PRIMARY} !important;
            border-color: {BORDER_SUBTLE} !important;
        }}
        .stButton > button {{
            background: linear-gradient(180deg, #1A253C 0%, #121A2C 100%);
            color: {TEXT_PRIMARY};
            border: 1px solid {BORDER_MEDIUM};
            border-radius: 6px;
            font-weight: 600;
            font-size: 0.88rem;
            padding: 6px 16px;
            transition: all 0.15s ease;
        }}
        .stButton > button:hover {{
            background: linear-gradient(180deg, #243352 0%, #18233B 100%);
            border-color: {COLOR_CYAN};
            color: {COLOR_CYAN};
            box-shadow: 0 0 12px rgba(0, 210, 255, 0.2);
        }}
        .stButton > button[kind="primary"] {{
            background: linear-gradient(180deg, #00D2FF 0%, #0088CC 100%) !important;
            color: #050B14 !important;
            border: none !important;
            font-weight: 700 !important;
        }}
        .stButton > button[kind="primary"]:hover {{
            background: linear-gradient(180deg, #38BDF8 0%, #0099E6 100%) !important;
            box-shadow: 0 0 16px rgba(0, 210, 255, 0.45) !important;
        }}

        /* Clean Tabs */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
            border-bottom: 1px solid {BORDER_SUBTLE};
            padding-bottom: 2px;
        }}
        .stTabs [data-baseweb="tab"] {{
            background-color: transparent !important;
            color: {TEXT_MUTED} !important;
            border-radius: 6px 6px 0 0 !important;
            padding: 8px 16px !important;
            font-weight: 600 !important;
            font-size: 0.90rem !important;
            border: none !important;
            transition: color 0.15s ease;
        }}
        .stTabs [aria-selected="true"] {{
            color: {COLOR_CYAN} !important;
            border-bottom: 2px solid {COLOR_CYAN} !important;
        }}

        /* DataTables */
        [data-testid="stDataFrame"] {{
            border: 1px solid {BORDER_SUBTLE};
            border-radius: 8px;
            overflow: hidden;
        }}
    </style>
    """, unsafe_allow_html=True)


# =============================================================================
# 4. CONSISTENT PLOTLY FIGURE THEME
# =============================================================================

def apply_figure_theme(fig: go.Figure, title: str = "", height: int = 420) -> go.Figure:
    """
    Applies the unified modern dark analytics workspace theme to any Plotly figure.
    Sets transparent canvas, high-contrast typography, clean gridlines, and dark tooltips.
    """
    fig.update_layout(
        title={
            "text": f"<b>{title}</b>" if title else "",
            "y": 0.96,
            "x": 0.02,
            "xanchor": "left",
            "yanchor": "top",
            "font": {
                "size": 14,
                "color": TEXT_PRIMARY,
                "family": "Inter, Segoe UI, sans-serif"
            }
        },
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=CHART_COLORWAY,
        height=height,
        margin={"l": 45, "r": 25, "t": 48 if title else 25, "b": 60},
        font={
            "family": "Inter, Segoe UI, sans-serif",
            "color": TEXT_SECONDARY,
            "size": 12
        },
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.18,
            "xanchor": "center",
            "x": 0.5,
            "bgcolor": "rgba(20, 28, 46, 0.75)",
            "bordercolor": BORDER_SUBTLE,
            "borderwidth": 1,
            "font": {"color": TEXT_SECONDARY, "size": 11}
        },
        hoverlabel={
            "bgcolor": BG_CARD,
            "font_size": 12,
            "font_family": "Inter, Segoe UI, sans-serif",
            "font_color": TEXT_PRIMARY,
            "bordercolor": BORDER_MEDIUM
        }
    )
    fig.update_xaxes(
        gridcolor=BORDER_SUBTLE,
        zerolinecolor=BORDER_SUBTLE,
        tickfont={"color": TEXT_MUTED, "size": 11},
        linecolor=BORDER_SUBTLE
    )
    fig.update_yaxes(
        gridcolor=BORDER_SUBTLE,
        zerolinecolor=BORDER_SUBTLE,
        tickfont={"color": TEXT_MUTED, "size": 11},
        linecolor=BORDER_SUBTLE
    )
    return fig


# =============================================================================
# 5. REUSABLE UI COMPONENT RENDERERS
# =============================================================================

def render_page_header(
    title: str,
    subtitle: str = "",
    badge: str = "",
    badge_type: str = "info"
) -> None:
    """Renders a standard, high-impact page or tab header."""
    badge_html = f'<span class="badge-pill {escape_html(badge_type)}">{escape_html(badge)}</span>' if badge else ""
    st.markdown(f"""
        <div class="page-header-box">
            <div>
                <h1 class="page-header-title">{escape_html(title)}</h1>
                {f'<p class="page-header-sub">{escape_html(subtitle)}</p>' if subtitle else ''}
            </div>
            <div>{badge_html}</div>
        </div>
    """, unsafe_allow_html=True)


def render_section_header(title: str, subtitle: str = "") -> None:
    """Renders a standardized section heading with optional subtitle."""
    st.markdown(f"""
        <div class="section-header-box">
            <h2 class="section-header-title">{escape_html(title)}</h2>
            {f'<span class="section-header-sub">{escape_html(subtitle)}</span>' if subtitle else ''}
        </div>
    """, unsafe_allow_html=True)


def render_metric_card(
    title: str,
    value: Any,
    subtext: str = "",
    subtext_neutral: bool = False,
    accent_color: Optional[str] = None
) -> None:
    """Renders an elevated metric card."""
    accent_style = f"border-top-color: {accent_color};" if accent_color else ""
    sub_class = "metric-sub-neutral" if subtext_neutral else "metric-sub"
    st.markdown(f"""
        <div class="metric-card" style="{accent_style}">
            <div class="metric-title">{escape_html(title)}</div>
            <div class="metric-value">{escape_html(value)}</div>
            {f'<div class="{sub_class}">{escape_html(subtext)}</div>' if subtext else ''}
        </div>
    """, unsafe_allow_html=True)


def render_kpi_row(cards: List[Dict[str, Any]]) -> None:
    """
    Renders an array of KPI cards across Streamlit columns.
    Each item in cards should have keys: title, value, subtext (optional), subtext_neutral (optional), accent_color (optional).
    """
    if not cards:
        return
    cols = st.columns(len(cards))
    for col, card in zip(cols, cards):
        with col:
            render_metric_card(
                title=card.get("title", ""),
                value=card.get("value", ""),
                subtext=card.get("subtext", ""),
                subtext_neutral=card.get("subtext_neutral", False),
                accent_color=card.get("accent_color")
            )


def render_callout(
    title: str,
    body: str,
    callout_type: str = "info"
) -> None:
    """
    Renders a modern insight / callout box.
    callout_type can be 'info', 'warning', 'success', or 'danger'.
    """
    c_type = callout_type.lower()
    icon = {
        "info": "",
        "warning": "",
        "success": "",
        "danger": ""
    }.get(c_type, "")

    st.markdown(f"""
        <div class="callout-box {escape_html(c_type)}">
            <div class="insight-title">
                <span>{icon}</span> {escape_html(title)}
            </div>
            <div class="insight-body">
                {body}
            </div>
        </div>
    """, unsafe_allow_html=True)


def render_badge(text: str, badge_type: str = "info") -> str:
    """Returns an inline HTML badge pill string."""
    return f'<span class="badge-pill {escape_html(badge_type)}">{escape_html(text)}</span>'


def render_empty_state(
    title: str,
    message: str,
    icon: str = ""
) -> None:
    """Renders a visually intentional empty state container."""
    st.markdown(f"""
        <div class="empty-state-box">
            <div class="empty-state-icon">{escape_html(icon)}</div>
            <div class="empty-state-title">{escape_html(title)}</div>
            <div class="empty-state-desc">{escape_html(message)}</div>
        </div>
    """, unsafe_allow_html=True)
