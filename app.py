"""
app.py
DataLab: Modern Interactive Data Science Platform & Netflix Strategy Case Study.

Unified entry point:
- Configures Streamlit page and modern analytics workspace layout.
- Injects canonical design system tokens, styles, and typography.
- Loads shared catalog pipeline (cached).
- Renders active-workspace navigation without evaluating inactive tabs.
- Dispatches execution to the active workspace.
"""

import streamlit as st
import pandas as pd

from src.data_generator import ensure_data_exists
from src.data_cleaner import clean_netflix_data
from src.feature_engineer import engineer_features
from src.data_context import get_source_dataset, set_source_dataset, get_working_dataset
from src.ui_theme import inject_custom_theme
from src.explorer_ui import render_explorer_tab
from src.cleaning_lab_ui import render_cleaning_lab_tab
from src.workflow_ui import render_workflow_studio_tab
from src.model_lab_ui import render_model_lab_tab
from src.missions_ui import render_missions_tab
from src.netflix_case_study_ui import render_netflix_case_study_tab


# -----------------------------------------------------------------------------
# 1. Page Configuration & Theme Initialization
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="DataLab  Interactive Data Science Platform",
    page_icon="D",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject centralized modern analytics workspace stylesheet
inject_custom_theme()


# -----------------------------------------------------------------------------
# 2. Cached Data Pipeline Engine
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_and_process_catalog():
    """
    Executes the canonical data preparation pipeline:
    1. Resolves canonical dataset at netflix_titles.csv (or data/ fallback).
    2. Loads and parses dates, durations, and structural nulls.
    3. Engineers licensing lag, audience tiers, proxy categories, and metadata soup.
    """
    csv_path = ensure_data_exists("netflix_titles.csv")
    raw_df = pd.read_csv(csv_path)
    source_label = f"netflix_titles.csv ({len(raw_df):,} rows)"

    cleaned_df, hygiene_report = clean_netflix_data(raw_df)
    engineered_df = engineer_features(cleaned_df)

    return raw_df, engineered_df, hygiene_report, source_label


raw_df, df, hygiene_report, data_source_label = load_and_process_catalog()

# Ensure default source is initialized in canonical data_context if not yet set
existing_src, _, _ = get_source_dataset()
if existing_src is None:
    set_source_dataset(raw_df, "builtin_netflix", "netflix_titles.csv")


# -----------------------------------------------------------------------------
# 3. Sidebar: Navigation & Active Dataset Status
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
            <div style="background: linear-gradient(135deg, #00D2FF 0%, #0088CC 100%); color: #050B14; font-weight: 800; font-size: 1.25rem; padding: 4px 10px; border-radius: 6px; letter-spacing: -0.02em;">DL</div>
            <div>
                <div style="font-size: 1.18rem; font-weight: 800; letter-spacing: -0.02em; color: #F8FAFC; line-height: 1.1;">DataLab</div>
                <div style="font-size: 0.72rem; color: #64748B; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em;">Analytics Workspace</div>
            </div>
        </div>
        <div style="font-size: 0.80rem; color: #94A3B8; margin-bottom: 18px; line-height: 1.4;">
            End-to-end data exploration, interactive cleaning, visual DAG pipelines, and predictive modeling.
        </div>
    """, unsafe_allow_html=True)

    st.markdown("### Navigation")
    workspace_options = [
        "Explore Dataset",
        "Clean & Transform",
        "Workflow Studio",
        "Model Lab",
        "Missions",
        "Netflix Case Study"
    ]

    active_workspace = st.radio(
        "Select Workspace",
        options=workspace_options,
        index=0,
        label_visibility="collapsed",
        key="datalab_primary_workspace"
    )

    st.markdown("---")

    # Dynamic Active Dataset Context Pill
    w_df, s_id, s_name, is_trans, state_id = get_working_dataset()
    display_df = w_df if w_df is not None else raw_df
    dataset_name = s_name if s_name else "netflix_titles.csv"
    mode_label = "Transformed Working Copy" if is_trans else "Raw Source Snapshot"
    mode_color = COLOR_CYAN if is_trans else COLOR_SUCCESS

    st.markdown("### Active Dataset")
    st.markdown(f"""
        <div style="background: rgba(20, 28, 46, 0.85); border: 1px solid #1E2B44; border-left: 3px solid {mode_color}; border-radius: 8px; padding: 12px 14px; margin-bottom: 14px;">
            <div style="font-size: 0.85rem; font-weight: 700; color: #F8FAFC; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                {dataset_name}
            </div>
            <div style="font-size: 0.75rem; color: {mode_color}; font-weight: 600; margin-top: 3px;">
                {mode_label}
            </div>
            <div style="font-size: 0.75rem; color: #94A3B8; margin-top: 6px;">
                <b>{len(display_df):,}</b> rows  <b>{display_df.shape[1]}</b> columns
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
        <div style="font-size: 0.74rem; color: #64748B; line-height: 1.5;">
            <b>Architecture:</b> 4-Phase DataLab Platform<br>
            <b>Stack:</b> Python 3.12  Pandas  Scikit-Learn  Plotly  Streamlit
        </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 4. Active Workspace Execution Dispatcher
# -----------------------------------------------------------------------------
# Only the selected workspace is executed, eliminating performance bottlenecks
# and preventing unintended cross-workspace reruns.
if active_workspace == "Explore Dataset":
    render_explorer_tab(netflix_raw_df=raw_df)

elif active_workspace == "Clean & Transform":
    render_cleaning_lab_tab(netflix_raw_df=raw_df)

elif active_workspace == "Workflow Studio":
    render_workflow_studio_tab(netflix_raw_df=raw_df)

elif active_workspace == "Model Lab":
    render_model_lab_tab(netflix_raw_df=raw_df)

elif active_workspace == "Missions":
    render_missions_tab(netflix_raw_df=raw_df)

elif active_workspace == "Netflix Case Study":
    render_netflix_case_study_tab(
        raw_df=raw_df,
        processed_df=df,
        hygiene_report=hygiene_report
    )
