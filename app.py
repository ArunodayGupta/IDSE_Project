"""
app.py
DataLab: Movie Box Office Worldwide Revenue Prediction Platform.
Supervised regression platform based on the Kaggle TMDB Box Office Prediction dataset.

Structure:
- Section A: Overview (KPIs, box office distribution, budget vs revenue, genres, longitudinal trends)
- Section B: Data Explorer (dataset loading, preview, quality audit, reproducible cleaning, filtering)
- Section C: ML Studio (model training, algorithm comparison, evaluation, interactive prediction, Kaggle submission)
"""

import streamlit as st
import pandas as pd

from src.data_loader import load_tmdb_train_data
from src.ui_theme import inject_custom_theme, COLOR_CYAN, COLOR_SUCCESS
from src.overview_tab import render_overview_section
from src.data_explorer_tab import render_data_explorer_section
from src.ml_studio_tab import render_ml_studio_section


# -----------------------------------------------------------------------------
# 1. Page Configuration & Theme
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="DataLab - Movie Box Office Revenue Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

inject_custom_theme()


# -----------------------------------------------------------------------------
# 2. Cached TMDB Training Dataset Loader
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_canonical_train_data():
    """Loads and caches the TMDB movie box office training dataset."""
    df, msg = load_tmdb_train_data()
    return df, msg


raw_df, load_msg = get_canonical_train_data()


# -----------------------------------------------------------------------------
# 3. Compact Sidebar Navigation
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 6px;">
            <div style="background: linear-gradient(135deg, #00D2FF 0%, #0088CC 100%); color: #050B14; font-weight: 800; font-size: 1.15rem; padding: 4px 9px; border-radius: 6px; letter-spacing: -0.02em;">DL</div>
            <div>
                <div style="font-size: 1.12rem; font-weight: 800; letter-spacing: -0.02em; color: #F8FAFC; line-height: 1.1;">DataLab</div>
                <div style="font-size: 0.70rem; color: #64748B; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">Box Office Prediction</div>
            </div>
        </div>
        <div style="font-size: 0.78rem; color: #94A3B8; margin-bottom: 16px; line-height: 1.35;">
            End-to-end data science platform predicting worldwide theatrical revenues from pre-release metadata.
        </div>
    """, unsafe_allow_html=True)

    st.markdown("### Navigation")
    sections = [
        "Overview",
        "Data Explorer",
        "ML Studio"
    ]

    active_section = st.radio(
        "Select Section",
        options=sections,
        index=0,
        label_visibility="collapsed",
        key="datalab_main_navigation"
    )

    st.markdown("---")

    # Compact Active Dataset Context Pill
    if raw_df is not None:
        st.markdown("### Dataset Context")
        st.markdown(f"""
            <div style="background: rgba(20, 28, 46, 0.85); border: 1px solid #1E2B44; border-left: 3px solid {COLOR_CYAN}; border-radius: 8px; padding: 10px 12px; margin-bottom: 12px;">
                <div style="font-size: 0.82rem; font-weight: 700; color: #F8FAFC;">
                    TMDB Box Office Dataset
                </div>
                <div style="font-size: 0.72rem; color: {COLOR_SUCCESS}; font-weight: 600; margin-top: 2px;">
                    Training Partition
                </div>
                <div style="font-size: 0.72rem; color: #94A3B8; margin-top: 4px;">
                    <b>{len(raw_df):,}</b> movies  <b>{raw_df.shape[1]}</b> features
                </div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("""
        <div style="font-size: 0.70rem; color: #64748B; line-height: 1.45;">
            <b>Problem:</b> Supervised Regression<br>
            <b>Target:</b> Worldwide Revenue (USD)<br>
            <b>Stack:</b> Python  Pandas  Scikit-Learn  Plotly  Streamlit
        </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 4. Dispatch Execution to Active Section
# -----------------------------------------------------------------------------
if raw_df is None:
    st.error(load_msg)
else:
    if active_section == "Overview":
        render_overview_section(raw_df)

    elif active_section == "Data Explorer":
        render_data_explorer_section(raw_df)

    elif active_section == "ML Studio":
        render_ml_studio_section(raw_df)
