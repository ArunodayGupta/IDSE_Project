"""
data_context.py
Canonical state management and dataset identity contract for DataLab.

Maintains a consistent contract across all workspaces:
1. Source dataset retrieval and identity.
2. Working copy retrieval and reliable revision-based identity.
3. Transformed vs Raw selection contract.
4. Stale-state detection for modeling and recommender runs.
5. Session experiment records.
"""

from typing import Tuple, Optional, Any
import pandas as pd


def get_source_dataset() -> Tuple[Optional[pd.DataFrame], str, str]:
    """
    Retrieves the active raw source DataFrame, source identity, and source name
    from Streamlit session state, or (None, '', '') if outside Streamlit.
    """
    try:
        import streamlit as st
        df = st.session_state.get("datalab_source_data")
        ident = st.session_state.get("datalab_source_identity", "")
        name = st.session_state.get("datalab_source_name", "")
        return df, ident, name
    except Exception:
        return None, "", ""


def set_source_dataset(df: pd.DataFrame, source_identity: str, source_name: str) -> None:
    """
    Sets the active raw source DataFrame in Streamlit session state.
    """
    try:
        import streamlit as st
        st.session_state["datalab_source_data"] = df
        st.session_state["datalab_source_identity"] = source_identity
        st.session_state["datalab_source_name"] = source_name
    except Exception:
        pass


def get_working_dataset_manager() -> Optional[Any]:
    """
    Retrieves the active WorkingDatasetManager from Streamlit session state.
    """
    try:
        import streamlit as st
        return st.session_state.get("datalab_working_manager") or st.session_state.get("cleaning_manager")
    except Exception:
        return None


def set_working_dataset_manager(manager: Any) -> None:
    """
    Registers the active WorkingDatasetManager in Streamlit session state.
    """
    try:
        import streamlit as st
        st.session_state["datalab_working_manager"] = manager
        st.session_state["cleaning_manager"] = manager
    except Exception:
        pass


def get_working_dataset() -> Tuple[Optional[pd.DataFrame], str, str, bool, str]:
    """
    Safely retrieves the active working copy DataFrame, source identity, source name,
    a boolean indicating whether transformations have been applied, and the deterministic
    working state identity.

    CRITICAL FIX (A1):
    Distinguishes between an absent manager and an existing manager whose working
    DataFrame contains zero rows. If a manager exists, its working_df is ALWAYS returned,
    even when empty. Never silently falls back to the original source.

    Returns:
        (working_df, source_identity, source_name, is_transformed, working_state_id)
    """
    try:
        import streamlit as st
        manager = get_working_dataset_manager()
        if manager is not None:
            # Manager exists - working copy is active regardless of row count!
            w_df = manager.working_df
            is_transformed = len(getattr(manager, "history", [])) > 0
            state_id = getattr(manager, "working_state_id", f"{manager.source_identity}__working")
            return w_df, manager.source_identity, manager.source_name, is_transformed, state_id

        # Absent manager: fall back to raw source snapshot
        src_df = st.session_state.get("datalab_source_data")
        ident = st.session_state.get("datalab_source_identity", "")
        name = st.session_state.get("datalab_source_name", "")
        return src_df, ident, name, False, f"{ident}__source"
    except Exception:
        return None, "", "", False, ""


def compute_modeling_dataset_identity(
    source_identity: str,
    is_working_copy: bool,
    working_manager: Optional[Any] = None,
    df: Optional[pd.DataFrame] = None
) -> str:
    """
    Generates a deterministic identity for the dataset used in modeling or workflows.
    Accounts for source identity, raw vs transformed choice, and working state revision.
    Avoids expensive rehashing of large DataFrames on every rerun.
    """
    if not is_working_copy or working_manager is None:
        return f"{source_identity}::source_raw"

    if hasattr(working_manager, "working_state_id"):
        return f"{source_identity}::working::{working_manager.working_state_id}"

    # Fallback lightweight fingerprint based on history and shape
    hist_len = len(getattr(working_manager, "history", []))
    shape_str = f"{len(df)}x{df.shape[1]}" if df is not None else "0x0"
    cols_hash = hash(tuple(df.columns)) if df is not None else 0
    return f"{source_identity}::working::hist{hist_len}::{shape_str}::{cols_hash}"


def is_result_stale(
    saved_dataset_id: Optional[str],
    current_dataset_id: str
) -> bool:
    """
    Determines whether a previously trained model or analysis is stale
    relative to the currently active dataset state.
    """
    if not saved_dataset_id:
        return False
    return saved_dataset_id != current_dataset_id


# Aliases for compatibility
get_active_dataset_context = get_working_dataset
register_working_dataset_manager = set_working_dataset_manager
resolve_modeling_dataset = compute_modeling_dataset_identity
