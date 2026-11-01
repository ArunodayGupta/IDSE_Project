"""
data_loader.py
Dataset loading, resolution, and safe JSON/date parsing for TMDB Box Office Prediction.
"""

import os
import ast
import json
from typing import Optional, Tuple, Dict, Any, List
import pandas as pd
import numpy as np


# Candidate search directories for TMDB dataset files
SEARCH_DIRS = [
    "data",
    "tmdb-box-office-prediction",
    "."
]


def resolve_file_path(filename: str) -> Optional[str]:
    """
    Searches known project directories for the specified dataset file.
    Returns the resolved absolute path or None if not found.
    """
    for directory in SEARCH_DIRS:
        candidate = os.path.join(directory, filename)
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
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
    Handles NaN, empty strings, and malformed strings gracefully without crashing.
    """
    if pd.isna(val) or val is None:
        return []
    if isinstance(val, (list, tuple)):
        return list(val)
    if isinstance(val, dict):
        return [val]
    
    val_str = str(val).strip()
    if not val_str or val_str in ("[]", "{}"):
        return []

    try:
        parsed = ast.literal_eval(val_str)
        if isinstance(parsed, list):
            return parsed
        elif isinstance(parsed, dict):
            return [parsed]
        return []
    except Exception:
        try:
            parsed = json.loads(val_str)
            if isinstance(parsed, list):
                return parsed
            elif isinstance(parsed, dict):
                return [parsed]
            return []
        except Exception:
            return []


def parse_release_date(series: pd.Series) -> pd.Series:
    """
    Parses release dates in format MM/DD/YY or YYYY-MM-DD.
    Corrects 2-digit century rollover: TMDB dates like '8/6/87' or '4/7/65'
    should resolve to 1987 or 1965, not 2087 or 2065.
    Years > 2020 are adjusted by subtracting 100 years.
    """
    dates = pd.to_datetime(series, format="%m/%d/%y", errors="coerce")
    fallback_mask = dates.isna() & series.notna()
    if fallback_mask.any():
        dates.loc[fallback_mask] = pd.to_datetime(series.loc[fallback_mask], errors="coerce")

    # Century correction
    valid_years = dates.dt.year
    excessive_mask = valid_years > 2020
    if excessive_mask.any():
        corrected = dates.copy()
        for idx in dates[excessive_mask].index:
            dt = dates.loc[idx]
            if pd.notna(dt):
                try:
                    corrected.loc[idx] = dt.replace(year=dt.year - 100)
                except Exception:
                    pass
        return corrected

    return dates


def load_tmdb_train_data(filepath: Optional[str] = None) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Loads and returns the TMDB training dataset.
    If filepath is not provided, searches known directories.
    Returns (DataFrame, status_or_error_message).
    """
    if filepath is None:
        filepath = resolve_file_path("train.csv")

    if not filepath or not os.path.isfile(filepath):
        msg = (
            "Default training dataset (train.csv) was not found. "
            "Please place train.csv, test.csv, and sample_submission.csv "
            "into the 'data/' folder or upload a CSV file in the Data Explorer."
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
        return None, "test.csv not found in data/ or tmdb-box-office-prediction/."

    try:
        df = pd.read_csv(filepath)
        return df, f"Loaded {len(df):,} test movies from {os.path.basename(filepath)}"
    except Exception as e:
        return None, f"Failed to read test data from {filepath}: {str(e)}"
