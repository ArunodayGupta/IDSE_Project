"""
data_explorer.py
Core data exploration, profiling, schema analysis, missing-value auditing,
and filtering engine for Phase 1 DataLab.

Designed with clean separation of concerns: pure functional data manipulation
and profiling algorithms that can be tested independently of Streamlit.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import io
import math
import os
from pathlib import Path
import numpy as np
import pandas as pd


def get_project_root() -> Path:
    """Returns the absolute Path to the project root directory."""
    return Path(__file__).resolve().parent.parent


def resolve_builtin_netflix_path() -> str:
    """
    Resolves the absolute path to the repository's built-in Netflix catalog CSV.
    Checks data/netflix_titles.csv and project root fallbacks.
    """
    root = get_project_root()
    candidates = [
        root / "data" / "netflix_titles.csv",
        root / "netflix_titles.csv",
        Path("data/netflix_titles.csv"),
        Path("netflix_titles.csv"),
    ]
    for c in candidates:
        if c.exists():
            return str(c.resolve())
    raise FileNotFoundError("Built-in netflix_titles.csv not found in data/ or project root.")


def load_csv_from_file_or_bytes(
    file_or_bytes: Union[bytes, io.BytesIO, str, Path],
    filename: str = "dataset.csv"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Robustly loads a CSV from raw bytes, a file buffer, or a file path.
    Handles multiple text encodings (UTF-8 with BOM, UTF-8, Latin-1, CP1252),
    validates headers, duplicate columns, and empty files.

    Args:
        file_or_bytes: Bytes, file-like object, or path string.
        filename: Original filename for logging/metadata.

    Returns:
        Tuple of (DataFrame, metadata_dict)

    Raises:
        ValueError: If file is empty, non-parseable, or contains 0 rows/columns.
    """
    if isinstance(file_or_bytes, (str, Path)):
        path = Path(file_or_bytes)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_or_bytes}")
        file_bytes = path.read_bytes()
        filename = path.name
    elif isinstance(file_or_bytes, io.BytesIO):
        file_bytes = file_or_bytes.getvalue()
    elif isinstance(file_or_bytes, bytes):
        file_bytes = file_or_bytes
    elif hasattr(file_or_bytes, "read"):
        file_bytes = file_or_bytes.read()
    else:
        raise ValueError("Unsupported input type for CSV loader.")

    if len(file_bytes) == 0:
        raise ValueError("The uploaded file is empty (0 bytes). Please upload a valid CSV file.")

    encodings_to_try = ["utf-8-sig", "utf-8", "latin-1", "cp1252"]
    parsed_df: Optional[pd.DataFrame] = None
    successful_encoding = ""
    parse_errors: List[str] = []

    for enc in encodings_to_try:
        try:
            buf = io.BytesIO(file_bytes)
            # Use low_memory=False to avoid mixed type warnings on large datasets
            df = pd.read_csv(buf, encoding=enc, low_memory=False)
            parsed_df = df
            successful_encoding = enc
            break
        except Exception as e:
            parse_errors.append(f"{enc}: {str(e)}")

    if parsed_df is None:
        raise ValueError(
            f"Failed to parse CSV file '{filename}' with supported encodings (UTF-8, Latin-1, CP1252). "
            f"Parsing details: {'; '.join(parse_errors[:2])}"
        )

    # Check for empty dataframe
    if parsed_df.shape[1] == 0:
        raise ValueError("The CSV file contains no recognizable columns or headers.")

    # Detect duplicate columns from raw header line
    duplicate_cols: List[str] = []
    try:
        import csv
        first_line = file_bytes.splitlines()[0] if file_bytes else b""
        header_reader = csv.reader([first_line.decode(successful_encoding, errors="replace")])
        raw_headers = next(header_reader, [])
        header_counts = pd.Series(raw_headers).value_counts()
        duplicate_cols = header_counts[header_counts > 1].index.tolist()
    except Exception:
        duplicate_cols = []

    # Also check if pandas disambiguated columns with suffixes
    if not duplicate_cols:
        for c in parsed_df.columns:
            if "." in str(c) and str(c).rsplit(".", 1)[-1].isdigit():
                base_c = str(c).rsplit(".", 1)[0]
                if base_c in parsed_df.columns and base_c not in duplicate_cols:
                    duplicate_cols.append(base_c)

    warnings: List[str] = []
    if duplicate_cols:
        warnings.append(
            f"Duplicate column names detected: {', '.join(duplicate_cols)}. "
            "Pandas automatically disambiguated them with numeric suffixes."
        )

    if parsed_df.shape[1] > 100:
        warnings.append(
            f"Dataset is wide ({parsed_df.shape[1]} columns). "
            "Consider selecting a subset of columns for focused analysis."
        )

    metadata: Dict[str, Any] = {
        "filename": filename,
        "encoding_used": successful_encoding,
        "file_size_bytes": len(file_bytes),
        "row_count": len(parsed_df),
        "column_count": parsed_df.shape[1],
        "duplicate_columns": duplicate_cols,
        "warnings": warnings,
    }

    return parsed_df, metadata


def get_missing_mask(df: pd.DataFrame, treat_empty_as_missing: bool = True) -> pd.DataFrame:
    """
    Computes a boolean DataFrame of identical shape and index to `df`, where True
    indicates that the corresponding cell is considered missing.

    When `treat_empty_as_missing` is True (default), whitespace-only strings
    in object/string columns are treated as missing without mutating the source DataFrame.

    Args:
        df: Input DataFrame.
        treat_empty_as_missing: If True, strings consisting solely of whitespace
                                are classified as missing.

    Returns:
        pd.DataFrame (boolean mask)
    """
    if df.empty or df.shape[1] == 0:
        return pd.DataFrame(False, index=df.index, columns=df.columns, dtype=bool)

    # Base pandas missingness (NaN, None, NaT, etc.)
    mask = df.isna().copy()

    if treat_empty_as_missing:
        for col in df.columns:
            series = df[col]
            if (
                pd.api.types.is_object_dtype(series)
                or pd.api.types.is_string_dtype(series)
                or isinstance(series.dtype, pd.CategoricalDtype)
            ):
                # Only check non-null string entries
                non_null_idx = (~mask[col])
                if non_null_idx.any():
                    str_vals = series[non_null_idx].astype(str)
                    whitespace_empty = str_vals.str.strip() == ""
                    if whitespace_empty.any():
                        empty_locs = whitespace_empty[whitespace_empty].index
                        mask.loc[empty_locs, col] = True

    return mask


def classify_column_type(series: pd.Series) -> str:
    """
    Classifies a pandas Series into one of five logical types:
    'Numeric', 'Datetime', 'Boolean', 'Categorical/Text', or 'Other'.

    Args:
        series: Pandas Series to evaluate.

    Returns:
        str: Logical classification name.
    """
    if pd.api.types.is_bool_dtype(series):
        return "Boolean"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "Datetime"
    if pd.api.types.is_numeric_dtype(series):
        return "Numeric"
    if (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
    ):
        return "Categorical/Text"
    return "Other"


def format_bytes(size_bytes: int) -> str:
    """Formats a byte count into a human-readable string (B, KB, MB, GB)."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def compute_overview_metrics(
    df: pd.DataFrame,
    treat_empty_as_missing: bool = True
) -> Dict[str, Any]:
    """
    Computes high-level overview metrics for the active dataset.
    All calculations are derived dynamically from the input DataFrame.

    Args:
        df: Input DataFrame.
        treat_empty_as_missing: Whether whitespace-only strings count as missing.

    Returns:
        Dict containing counts, percentages, type distributions, and memory statistics.
    """
    total_rows = len(df)
    total_columns = df.shape[1]

    if total_rows == 0 or total_columns == 0:
        return {
            "total_rows": total_rows,
            "total_columns": total_columns,
            "duplicate_rows": 0,
            "duplicate_rows_pct": 0.0,
            "total_cells": 0,
            "total_missing_cells": 0,
            "missing_cells_pct": 0.0,
            "complete_rows": 0,
            "complete_rows_pct": 0.0,
            "all_missing_rows": 0,
            "numeric_cols_count": 0,
            "categorical_cols_count": 0,
            "datetime_cols_count": 0,
            "boolean_cols_count": 0,
            "other_cols_count": 0,
            "memory_usage_bytes": 0,
            "memory_usage_str": "0 B",
            "treat_empty_as_missing": treat_empty_as_missing,
        }

    duplicate_rows = int(df.duplicated().sum())
    duplicate_rows_pct = round((duplicate_rows / total_rows * 100), 2)

    missing_mask = get_missing_mask(df, treat_empty_as_missing=treat_empty_as_missing)
    total_cells = total_rows * total_columns
    total_missing_cells = int(missing_mask.to_numpy().sum())
    missing_cells_pct = round((total_missing_cells / total_cells * 100), 2) if total_cells > 0 else 0.0

    complete_rows = int((~missing_mask.any(axis=1)).sum())
    complete_rows_pct = round((complete_rows / total_rows * 100), 2) if total_rows > 0 else 0.0

    all_missing_rows = int((missing_mask.all(axis=1)).sum())

    # Column type categorization
    type_counts = {"Numeric": 0, "Categorical/Text": 0, "Datetime": 0, "Boolean": 0, "Other": 0}
    for col in df.columns:
        ctype = classify_column_type(df[col])
        type_counts[ctype] = type_counts.get(ctype, 0) + 1

    memory_bytes = int(df.memory_usage(deep=True).sum())

    return {
        "total_rows": total_rows,
        "total_columns": total_columns,
        "duplicate_rows": duplicate_rows,
        "duplicate_rows_pct": duplicate_rows_pct,
        "total_cells": total_cells,
        "total_missing_cells": total_missing_cells,
        "missing_cells_pct": missing_cells_pct,
        "complete_rows": complete_rows,
        "complete_rows_pct": complete_rows_pct,
        "all_missing_rows": all_missing_rows,
        "numeric_cols_count": type_counts["Numeric"],
        "categorical_cols_count": type_counts["Categorical/Text"],
        "datetime_cols_count": type_counts["Datetime"],
        "boolean_cols_count": type_counts["Boolean"],
        "other_cols_count": type_counts["Other"],
        "memory_usage_bytes": memory_bytes,
        "memory_usage_str": format_bytes(memory_bytes),
        "treat_empty_as_missing": treat_empty_as_missing,
    }


def compute_column_schema(
    df: pd.DataFrame,
    treat_empty_as_missing: bool = True
) -> pd.DataFrame:
    """
    Constructs an interactive data dictionary / schema table for all columns.
    Preserves exact column names without internal mutations.

    Args:
        df: Input DataFrame.
        treat_empty_as_missing: Whether whitespace-only strings count as missing.

    Returns:
        pd.DataFrame with schema metrics per column.
    """
    if df.empty or df.shape[1] == 0:
        return pd.DataFrame(
            columns=[
                "Column", "Logical Type", "Data Type", "Non-Null Count",
                "Missing Count", "Missing %", "Distinct Values", "Sample Values"
            ]
        )

    total_rows = len(df)
    missing_mask = get_missing_mask(df, treat_empty_as_missing=treat_empty_as_missing)

    records = []
    for col in df.columns:
        series = df[col]
        col_missing = int(missing_mask[col].sum())
        col_non_null = total_rows - col_missing
        missing_pct = round((col_missing / total_rows * 100), 2) if total_rows > 0 else 0.0
        distinct_count = int(series.nunique(dropna=True))
        ltype = classify_column_type(series)

        # Retrieve a few distinct, non-missing sample values
        non_null_series = series[~missing_mask[col]]
        samples_list = non_null_series.dropna().astype(str).unique()[:3].tolist()
        sample_str = ", ".join(f"'{s}'" if len(s) < 25 else f"'{s[:22]}...'" for s in samples_list)
        if not sample_str:
            sample_str = "[None / All Missing]"

        records.append({
            "Column": str(col),
            "Logical Type": ltype,
            "Data Type": str(series.dtype),
            "Non-Null Count": col_non_null,
            "Missing Count": col_missing,
            "Missing %": missing_pct,
            "Distinct Values": distinct_count,
            "Sample Values": sample_str,
        })

    return pd.DataFrame(records)


def compute_missing_rankings(
    df: pd.DataFrame,
    treat_empty_as_missing: bool = True
) -> pd.DataFrame:
    """
    Ranks columns by missing count and missing percentage in descending order.
    Flags columns with high or critical missingness.

    Args:
        df: Input DataFrame.
        treat_empty_as_missing: Whether whitespace-only strings count as missing.

    Returns:
        pd.DataFrame sorted descending by missing percentage.
    """
    if df.empty or df.shape[1] == 0:
        return pd.DataFrame(
            columns=["Column", "Missing Count", "Missing %", "Valid Count", "Missingness Level"]
        )

    total_rows = len(df)
    missing_mask = get_missing_mask(df, treat_empty_as_missing=treat_empty_as_missing)
    missing_counts = missing_mask.sum()

    records = []
    for col in df.columns:
        m_count = int(missing_counts[col])
        m_pct = round((m_count / total_rows * 100), 2) if total_rows > 0 else 0.0
        v_count = total_rows - m_count

        if m_pct >= 50.0:
            level = "Critical (>50%)"
        elif m_pct >= 20.0:
            level = "High (>20%)"
        elif m_pct > 0.0:
            level = "Moderate (>0%)"
        else:
            level = "Zero Missing (100% Complete)"

        records.append({
            "Column": str(col),
            "Missing Count": m_count,
            "Missing %": m_pct,
            "Valid Count": v_count,
            "Missingness Level": level,
        })

    ranking_df = pd.DataFrame(records).sort_values(by=["Missing Count", "Missing %"], ascending=False).reset_index(drop=True)
    return ranking_df


def get_missingness_matrix(
    df: pd.DataFrame,
    max_rows: int = 500,
    treat_empty_as_missing: bool = True
) -> Tuple[pd.DataFrame, bool, int]:
    """
    Generates a missingness indicator matrix (1 = Missing, 0 = Present) for heatmap rendering.
    If the dataset exceeds max_rows, systematically samples up to max_rows evenly across
    the entire dataset to prevent freezing the browser while faithfully visualizing sparsity.

    Args:
        df: Input DataFrame.
        max_rows: Maximum rows to sample for the visualization.
        treat_empty_as_missing: Whether whitespace-only strings count as missing.

    Returns:
        Tuple of (matrix_df, was_sampled, total_rows)
    """
    total_rows = len(df)
    if total_rows == 0 or df.shape[1] == 0:
        return pd.DataFrame(), False, 0

    missing_mask = get_missing_mask(df, treat_empty_as_missing=treat_empty_as_missing)

    was_sampled = total_rows > max_rows
    if was_sampled:
        step = max(1, total_rows // max_rows)
        sampled_indices = df.index[::step][:max_rows]
        matrix_df = missing_mask.loc[sampled_indices].astype(int)
    else:
        matrix_df = missing_mask.astype(int)

    return matrix_df, was_sampled, total_rows


def compute_numeric_stats(
    series: pd.Series,
    treat_empty_as_missing: bool = True
) -> Dict[str, Any]:
    """
    Computes comprehensive descriptive statistics and distribution metrics
    for a numeric column.

    Args:
        series: Numeric series to analyze.
        treat_empty_as_missing: Handled consistently with overall profiling.

    Returns:
        Dict of statistical metrics.
    """
    # Coerce to numeric safely, treating errors as NaN
    numeric_s = pd.to_numeric(series, errors="coerce")
    total_count = len(series)
    valid_s = numeric_s.dropna()
    valid_count = len(valid_s)
    missing_count = total_count - valid_count
    missing_pct = round((missing_count / total_count * 100), 2) if total_count > 0 else 0.0

    if valid_count == 0:
        return {
            "has_data": False,
            "total_count": total_count,
            "valid_count": 0,
            "missing_count": missing_count,
            "missing_pct": missing_pct,
            "reason": "Column contains only missing or non-numeric values."
        }

    mean_val = float(valid_s.mean())
    std_val = float(valid_s.std()) if valid_count > 1 else 0.0
    median_val = float(valid_s.median())
    min_val = float(valid_s.min())
    max_val = float(valid_s.max())

    q25 = float(valid_s.quantile(0.25))
    q75 = float(valid_s.quantile(0.75))
    iqr = q75 - q25

    # Outliers detection via standard 1.5 * IQR rule
    lower_bound = q25 - 1.5 * iqr
    upper_bound = q75 + 1.5 * iqr
    outliers = valid_s[(valid_s < lower_bound) | (valid_s > upper_bound)]
    outlier_count = len(outliers)
    outlier_pct = round((outlier_count / valid_count * 100), 2) if valid_count > 0 else 0.0

    zeros_count = int((valid_s == 0).sum())
    zeros_pct = round((zeros_count / valid_count * 100), 2) if valid_count > 0 else 0.0

    # Skewness
    skew_val = float(valid_s.skew()) if valid_count > 2 and std_val > 0 else 0.0

    return {
        "has_data": True,
        "total_count": total_count,
        "valid_count": valid_count,
        "missing_count": missing_count,
        "missing_pct": missing_pct,
        "mean": round(mean_val, 4),
        "median": round(median_val, 4),
        "std": round(std_val, 4),
        "min": round(min_val, 4),
        "max": round(max_val, 4),
        "q25": round(q25, 4),
        "q75": round(q75, 4),
        "iqr": round(iqr, 4),
        "lower_bound": round(lower_bound, 4),
        "upper_bound": round(upper_bound, 4),
        "outlier_count": outlier_count,
        "outlier_pct": outlier_pct,
        "zeros_count": zeros_count,
        "zeros_pct": zeros_pct,
        "skewness": round(skew_val, 4),
        "is_constant": bool(min_val == max_val),
    }


def compute_categorical_stats(
    series: pd.Series,
    top_n: int = 10,
    treat_empty_as_missing: bool = True
) -> Dict[str, Any]:
    """
    Computes frequency distribution, top categories, and cardinality metrics
    for a categorical or text column.

    Args:
        series: Categorical/text series to analyze.
        top_n: Number of top categories to return in summary table.
        treat_empty_as_missing: Whether whitespace-only strings count as missing.

    Returns:
        Dict of categorical metrics and frequency table.
    """
    total_count = len(series)

    # Determine missingness
    if treat_empty_as_missing:
        cleaned_series = series.astype(str).str.strip()
        is_missing = series.isna() | (cleaned_series == "") | (cleaned_series.str.lower().isin(["nan", "none"]))
    else:
        is_missing = series.isna()

    missing_count = int(is_missing.sum())
    valid_count = total_count - missing_count
    missing_pct = round((missing_count / total_count * 100), 2) if total_count > 0 else 0.0

    if valid_count == 0:
        return {
            "has_data": False,
            "total_count": total_count,
            "valid_count": 0,
            "missing_count": missing_count,
            "missing_pct": missing_pct,
            "unique_count": 0,
            "reason": "Column contains only missing values."
        }

    valid_series = series[~is_missing].astype(str).str.strip()
    value_counts = valid_series.value_counts()
    unique_count = len(value_counts)

    # Frequency table for top_n
    top_counts = value_counts.head(top_n)
    freq_table = pd.DataFrame({
        "Category": top_counts.index,
        "Count": top_counts.values,
        "Percentage": (top_counts.values / valid_count * 100).round(2),
    })

    top_category = value_counts.index[0] if not value_counts.empty else "N/A"
    top_category_count = int(value_counts.iloc[0]) if not value_counts.empty else 0
    top_category_pct = round((top_category_count / valid_count * 100), 2) if valid_count > 0 else 0.0

    is_high_cardinality = bool(unique_count > 100 or (valid_count > 10 and unique_count / valid_count > 0.5))

    return {
        "has_data": True,
        "total_count": total_count,
        "valid_count": valid_count,
        "missing_count": missing_count,
        "missing_pct": missing_pct,
        "unique_count": unique_count,
        "top_category": top_category,
        "top_category_count": top_category_count,
        "top_category_pct": top_category_pct,
        "freq_table": freq_table,
        "is_high_cardinality": is_high_cardinality,
        "is_constant": bool(unique_count == 1),
    }


def compute_datetime_stats(series: pd.Series) -> Dict[str, Any]:
    """
    Computes timeline range and temporal spread for datetime columns.

    Args:
        series: Datetime series to analyze.

    Returns:
        Dict of datetime metrics.
    """
    try:
        dt_series = pd.to_datetime(series, errors="coerce", format="mixed")
    except (ValueError, TypeError):
        dt_series = pd.to_datetime(series, errors="coerce")
    total_count = len(series)
    valid_dt = dt_series.dropna()
    valid_count = len(valid_dt)
    missing_count = total_count - valid_count
    missing_pct = round((missing_count / total_count * 100), 2) if total_count > 0 else 0.0

    if valid_count == 0:
        return {
            "has_data": False,
            "total_count": total_count,
            "valid_count": 0,
            "missing_count": missing_count,
            "missing_pct": missing_pct,
            "reason": "Column contains no valid datetime entries."
        }

    min_date = valid_dt.min()
    max_date = valid_dt.max()
    span_days = (max_date - min_date).days

    # Yearly aggregation for temporal distribution
    years_series = valid_dt.dt.year.value_counts().sort_index()
    year_freq_df = pd.DataFrame({
        "Year": years_series.index.astype(str),
        "Count": years_series.values
    })

    return {
        "has_data": True,
        "total_count": total_count,
        "valid_count": valid_count,
        "missing_count": missing_count,
        "missing_pct": missing_pct,
        "min_date": str(min_date.date()) if hasattr(min_date, "date") else str(min_date),
        "max_date": str(max_date.date()) if hasattr(max_date, "date") else str(max_date),
        "span_days": span_days,
        "year_freq_df": year_freq_df,
    }


def preview_datetime_parsing(series: pd.Series, max_samples: int = 5) -> Dict[str, Any]:
    """
    Simulates parsing an arbitrary string column as datetime without modifying
    the original column or DataFrame.

    Args:
        series: String series to preview parsing on.
        max_samples: Number of sample conversions to return.

    Returns:
        Dict with success rate and sample original-to-parsed pairs.
    """
    total_count = len(series)
    non_null_s = series.dropna()
    if non_null_s.empty:
        return {
            "success_rate_pct": 0.0,
            "parsed_count": 0,
            "failed_count": 0,
            "samples": []
        }

    try:
        parsed = pd.to_datetime(non_null_s, errors="coerce", format="mixed")
    except (ValueError, TypeError):
        parsed = pd.to_datetime(non_null_s, errors="coerce")
    success_mask = ~parsed.isna()
    success_count = int(success_mask.sum())
    failed_count = len(non_null_s) - success_count
    rate = round((success_count / len(non_null_s) * 100), 2)

    samples = []
    for orig, p in zip(non_null_s[success_mask][:max_samples], parsed[success_mask][:max_samples]):
        samples.append({
            "Original": str(orig),
            "Parsed Datetime": str(p)
        })

    return {
        "success_rate_pct": rate,
        "parsed_count": success_count,
        "failed_count": failed_count,
        "samples": samples
    }


def apply_explorer_filters(
    df: pd.DataFrame,
    filters: Dict[str, Any],
    treat_empty_as_missing: bool = True
) -> pd.DataFrame:
    """
    Applies multi-condition search, numeric, categorical, text, and missingness
    filters to the input DataFrame with deterministic AND logic.
    Does NOT mutate the original DataFrame.

    Filter structure expected in `filters`:
        - global_search: Optional[str]
        - search_columns: Optional[List[str]] (defaults to all string/object cols)
        - numeric_ranges: Optional[Dict[str, Tuple[float, float]]]
        - categorical_selections: Optional[Dict[str, List[Any]]]
        - text_searches: Optional[Dict[str, str]]
        - missing_filters: Optional[Dict[str, str]] ('all', 'missing_only', 'not_missing_only')
        - date_ranges: Optional[Dict[str, Tuple[str, str]]]
        - has_any_missing: Optional[str] ('all', 'has_missing', 'complete_only')

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    if df.empty or not filters:
        return df.copy()

    mask = pd.Series(True, index=df.index)
    missing_mask = get_missing_mask(df, treat_empty_as_missing=treat_empty_as_missing)

    # 1. Global Search (case-insensitive substring across selected or string columns)
    global_search = filters.get("global_search")
    if global_search and str(global_search).strip():
        term = str(global_search).strip().lower()
        search_cols = filters.get("search_columns")
        if not search_cols:
            # Default to all string/object/categorical columns
            search_cols = [
                c for c in df.columns
                if pd.api.types.is_object_dtype(df[c])
                or pd.api.types.is_string_dtype(df[c])
                or isinstance(df[c].dtype, pd.CategoricalDtype)
            ]
            if not search_cols:
                search_cols = list(df.columns)

        search_mask = pd.Series(False, index=df.index)
        for c in search_cols:
            if c in df.columns:
                # Fill na with empty string before search
                c_str = df[c].astype(str).str.lower()
                search_mask = search_mask | c_str.str.contains(term, regex=False, na=False)

        mask = mask & search_mask

    # 2. Numeric Range Filters
    num_ranges = filters.get("numeric_ranges", {})
    for col, (min_v, max_v) in num_ranges.items():
        if col in df.columns:
            num_series = pd.to_numeric(df[col], errors="coerce")
            mask = mask & (num_series >= min_v) & (num_series <= max_v)

    # 3. Categorical Multiselect Filters
    cat_filters = filters.get("categorical_selections", {})
    for col, selected_vals in cat_filters.items():
        if col in df.columns and selected_vals:
            cat_series = df[col].astype(str).str.strip()
            sel_str = [str(v).strip() for v in selected_vals]
            mask = mask & cat_series.isin(sel_str)

    # 4. Text Substring Column Filters
    text_searches = filters.get("text_searches", {})
    for col, substr in text_searches.items():
        if col in df.columns and substr and str(substr).strip():
            target_str = str(substr).strip().lower()
            mask = mask & df[col].astype(str).str.lower().str.contains(target_str, regex=False, na=False)

    # 5. Missingness Filters per Column ('missing_only', 'not_missing_only')
    missing_filters = filters.get("missing_filters", {})
    for col, mode in missing_filters.items():
        if col in df.columns:
            if mode == "missing_only":
                mask = mask & missing_mask[col]
            elif mode == "not_missing_only":
                mask = mask & (~missing_mask[col])

    # 6. Row-level Missingness Filter ('has_missing' or 'complete_only')
    has_any_missing_mode = filters.get("has_any_missing", "all")
    if has_any_missing_mode == "has_missing":
        mask = mask & missing_mask.any(axis=1)
    elif has_any_missing_mode == "complete_only":
        mask = mask & (~missing_mask.any(axis=1))

    # 7. Date Range Filters
    date_ranges = filters.get("date_ranges", {})
    for col, (start_dt, end_dt) in date_ranges.items():
        if col in df.columns and (start_dt is not None or end_dt is not None):
            dt_series = pd.to_datetime(df[col], errors="coerce")
            if start_dt:
                mask = mask & (dt_series >= pd.to_datetime(start_dt))
            if end_dt:
                mask = mask & (dt_series <= pd.to_datetime(end_dt))

    return df[mask].copy()


def paginate_dataframe(
    df: pd.DataFrame,
    page_number: int,
    page_size: int = 25,
    sort_by: Optional[str] = None,
    ascending: bool = True
) -> Tuple[pd.DataFrame, int, int]:
    """
    Sorts and paginates a DataFrame for tabular browsing.

    Args:
        df: DataFrame to paginate.
        page_number: 1-indexed page number.
        page_size: Records per page (e.g. 25, 50, 100).
        sort_by: Optional column name to sort by.
        ascending: Sort order (True = asc, False = desc).

    Returns:
        Tuple of (page_slice_df, total_pages, total_records)
    """
    total_records = len(df)
    page_size = max(1, page_size)
    total_pages = max(1, math.ceil(total_records / page_size))
    current_page = max(1, min(page_number, total_pages))

    if total_records == 0:
        return df.copy(), total_pages, 0

    working_df = df
    if sort_by and sort_by in df.columns:
        working_df = df.sort_values(by=sort_by, ascending=ascending, na_position="last")

    start_idx = (current_page - 1) * page_size
    end_idx = min(start_idx + page_size, total_records)
    page_slice = working_df.iloc[start_idx:end_idx].copy()

    return page_slice, total_pages, total_records


def export_filtered_csv(df: pd.DataFrame) -> str:
    """
    Exports the filtered DataFrame to a CSV string.
    Preserves all original column names and order,
    excludes any internal helper columns (prefixed with '__'),
    and omits the DataFrame index.

    Args:
        df: DataFrame to export.

    Returns:
        str: CSV formatted string.
    """
    export_cols = [c for c in df.columns if not str(c).startswith("__")]
    export_df = df[export_cols]
    buf = io.StringIO()
    export_df.to_csv(buf, index=False)
    return buf.getvalue()


def export_profile_csv(schema_df: pd.DataFrame) -> str:
    """
    Exports the column profiling data dictionary to a CSV string.

    Args:
        schema_df: Schema DataFrame computed by compute_column_schema.

    Returns:
        str: CSV formatted string.
    """
    buf = io.StringIO()
    schema_df.to_csv(buf, index=False)
    return buf.getvalue()
