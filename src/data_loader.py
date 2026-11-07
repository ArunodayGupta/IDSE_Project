"""
data_loader.py
Dataset loading, resolution, and safe JSON/date parsing for TMDB Box Office Prediction.
"""

import os
import ast
import json
import re
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
import pandas as pd
import numpy as np


# Anchored project root path to ensure directory-independent resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Candidate search directories relative to project root
SEARCH_CANDIDATES = [
    PROJECT_ROOT / "data",
    PROJECT_ROOT / "tmdb-box-office-prediction",
    PROJECT_ROOT,
]


def resolve_file_path(filename: str) -> Optional[str]:
    """
    Searches known project directories for the specified dataset file.
    Paths are anchored to the project root directory, guaranteeing correct resolution
    regardless of whether the process was launched from the repository root,
    notebooks directory, or a subfolder. Also supports explicitly provided absolute
    or relative paths.

    Returns the resolved absolute path as a string, or None if not found.
    """
    if not filename:
        return None

    # 1. Check if provided directly as an existing path (absolute or relative to cwd)
    if os.path.isfile(filename):
        return os.path.abspath(filename)

    # 2. Search anchored project directories
    base_name = os.path.basename(filename)
    for candidate_dir in SEARCH_CANDIDATES:
        candidate_path = candidate_dir / base_name
        if candidate_path.is_file():
            return str(candidate_path.resolve())

    # 3. Direct check relative to project root with original relative path
    candidate_direct = PROJECT_ROOT / filename
    if candidate_direct.is_file():
        return str(candidate_direct.resolve())

    return None


def get_dataset_availability() -> Dict[str, Optional[str]]:
    """
    Returns resolution status for standard TMDB dataset files:
    train.csv, test.csv, sample_submission.csv.
    """
    return {
        "train.csv": resolve_file_path("train.csv"),
        "test.csv": resolve_file_path("test.csv"),
        "sample_submission.csv": resolve_file_path("sample_submission.csv"),
    }


def safe_parse_json(val: Any) -> List[Dict[str, Any]]:
    """
    Safely parses JSON-like string representations of dictionaries or lists.

    Checks collection types before missingness tests to prevent ambiguous truth-value
    exceptions when evaluates against actual lists or dictionaries.

    Handles:
    - None, NaN, pd.NA
    - Empty strings, empty lists, empty dictionaries
    - Python lists of dictionaries
    - Python dictionaries (wrapped into a single-element list)
    - Valid JSON strings
    - Python literal representations using single quotes
    - Malformed strings (returns empty list without crashing)
    - Unexpected scalar values (returns empty list)
    """
    # 1. Check if already a Python collection (prevents pd.isna array truth value error)
    if isinstance(val, list):
        return [item for item in val if isinstance(item, dict)]
    if isinstance(val, tuple):
        return [item for item in val if isinstance(item, dict)]
    if isinstance(val, dict):
        return [val]

    # 2. Check for None and pandas/numpy NA / NaN
    if val is None or val is pd.NA:
        return []
    if isinstance(val, (float, np.floating)) and np.isnan(val):
        return []

    # 3. Validate string input
    if not isinstance(val, str):
        return []

    val_str = val.strip()
    if not val_str or val_str in ("[]", "{}", "null", "None", "nan", "NaN"):
        return []

    # 4. Attempt Python literal evaluation (handles single-quoted TMDB strings)
    try:
        parsed = ast.literal_eval(val_str)
        if isinstance(parsed, list):
            return [x for x in parsed if isinstance(x, dict)]
        elif isinstance(parsed, dict):
            return [parsed]
        return []
    except (ValueError, SyntaxError, MemoryError):
        pass

    # 5. Attempt standard JSON parsing
    try:
        parsed = json.loads(val_str)
        if isinstance(parsed, list):
            return [x for x in parsed if isinstance(x, dict)]
        elif isinstance(parsed, dict):
            return [parsed]
        return []
    except (json.JSONDecodeError, ValueError):
        pass

    return []


def parse_release_date(series: pd.Series) -> pd.Series:
    """
    Parses movie release dates in various formats (MM/DD/YY, MM/DD/YYYY, YYYY-MM-DD).

    Documented Pivot Rule for Two-Digit Years (MM/DD/YY):
    - When dates use a 2-digit year format (e.g., '4/7/65' or '2/20/15'), standard
      datetime parsing defaults to century 2000 for years 00-68 (e.g., '65' -> 2065).
    - For TMDB records, years in the future (> 2020) originating from two-digit
      representations are adjusted by subtracting 100 years ('4/7/65' -> 1965).
    - Explicit four-digit years (e.g., '2025-04-10', '1983-05-23') and ISO dates
      are strictly preserved and never adjusted.
    - Missing or malformed dates produce pd.NaT without fabricating default dates.
    """
    if series is None or series.empty:
        return pd.Series(dtype="datetime64[ns]")

    def _parse_single_date(val: Any) -> Any:
        if val is None or val is pd.NA:
            return pd.NaT
        if isinstance(val, (float, np.floating)) and np.isnan(val):
            return pd.NaT
        if isinstance(val, (pd.Timestamp, np.datetime64)):
            return pd.to_datetime(val)
        if hasattr(val, "year") and hasattr(val, "month") and hasattr(val, "day"):
            return pd.to_datetime(val)

        val_str = str(val).strip()
        if not val_str or val_str.lower() in ("nan", "none", "nat", ""):
            return pd.NaT

        # Explicit four-digit year check (e.g. 2025-04-10, 05/23/1983, 1999)
        # If an explicit 4-digit year is present, parse directly without century adjustment
        if re.search(r"\b(18|19|20)\d{2}\b", val_str):
            return pd.to_datetime(val_str, errors="coerce")

        # Check for 2-digit year pattern: M/D/YY or MM/DD/YY
        parts = val_str.split("/")
        if len(parts) == 3 and len(parts[2]) == 2:
            dt = pd.to_datetime(val_str, format="%m/%d/%y", errors="coerce")
            if pd.notna(dt) and dt.year > 2020:
                try:
                    return dt.replace(year=dt.year - 100)
                except Exception:
                    return dt
            return dt

        # Fallback parsing
        return pd.to_datetime(val_str, errors="coerce")

    parsed_series = series.apply(_parse_single_date)
    return pd.to_datetime(parsed_series, errors="coerce")


def load_tmdb_train_data(filepath: Optional[str] = None) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Loads and returns the TMDB training dataset.
    If filepath is not provided, searches anchored project directories.
    Returns (DataFrame, status_or_error_message).
    """
    if filepath is None:
        filepath = resolve_file_path("train.csv")

    if not filepath or not os.path.isfile(filepath):
        msg = (
            "Default training dataset (train.csv) was not found. "
            "Please ensure train.csv is present in 'data/' or 'tmdb-box-office-prediction/'."
        )
        return None, msg

    try:
        df = pd.read_csv(filepath)
        return df, f"Successfully loaded {len(df):,} movies from {os.path.basename(filepath)}"
    except Exception as e:
        return None, f"Failed to read dataset from {filepath}: {str(e)}"


def load_tmdb_test_data(filepath: Optional[str] = None) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Loads and returns the TMDB test dataset (test.csv).
    Returns (DataFrame, status_or_error_message).
    """
    if filepath is None:
        filepath = resolve_file_path("test.csv")

    if not filepath or not os.path.isfile(filepath):
        return None, "test.csv not found in 'data/' or 'tmdb-box-office-prediction/'."

    try:
        df = pd.read_csv(filepath)
        return df, f"Loaded {len(df):,} test movies from {os.path.basename(filepath)}"
    except Exception as e:
        return None, f"Failed to read test data from {filepath}: {str(e)}"
