"""
cleaning_lab.py
Core data cleaning, transformation, and feature engineering engine for Phase 2 DataLab.

Provides pure functional transformation operations, allowlisted feature templates,
the What Changed? comparison engine, step-by-step audit logging, and reversible pipeline state.
Independent of Streamlit for 100% automated test coverage.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import json
import io
import math
import numpy as np
import pandas as pd

from src.data_explorer import (
    get_missing_mask,
    classify_column_type,
    compute_overview_metrics,
    format_bytes,
)


def generate_data_quality_audit(
    df: pd.DataFrame,
    treat_empty_as_missing: bool = True
) -> Dict[str, Any]:
    """
    Performs an automated data quality audit on a DataFrame.
    Identifies missing fields, duplicate records, constant columns,
    whitespace-only string values, and potential date columns.

    Args:
        df: Input DataFrame to audit.
        treat_empty_as_missing: Whether whitespace-only strings count as missing.

    Returns:
        Dict of quality findings and warnings.
    """
    total_rows = len(df)
    total_cols = df.shape[1]

    if total_rows == 0 or total_cols == 0:
        return {
            "total_rows": total_rows,
            "total_cols": total_cols,
            "total_cells": 0,
            "total_missing_cells": 0,
            "missing_cells_pct": 0.0,
            "duplicate_rows": 0,
            "duplicate_rows_pct": 0.0,
            "missing_by_col": {},
            "constant_columns": [],
            "whitespace_columns": {},
            "date_candidate_columns": [],
            "warnings": ["Dataset is empty."],
        }

    missing_mask = get_missing_mask(df, treat_empty_as_missing=treat_empty_as_missing)
    total_cells = total_rows * total_cols
    total_missing_cells = int(missing_mask.to_numpy().sum())
    missing_cells_pct = round((total_missing_cells / total_cells * 100), 2) if total_cells > 0 else 0.0

    duplicate_rows = int(df.duplicated().sum())
    duplicate_rows_pct = round((duplicate_rows / total_rows * 100), 2) if total_rows > 0 else 0.0

    missing_by_col = {}
    constant_columns = []
    whitespace_columns = {}
    date_candidate_columns = []
    warnings: List[str] = []

    for col in df.columns:
        s = df[col]
        m_cnt = int(missing_mask[col].sum())
        if m_cnt > 0:
            m_pct = round((m_cnt / total_rows * 100), 2)
            missing_by_col[col] = {"count": m_cnt, "pct": m_pct}
            if m_pct >= 50.0:
                warnings.append(f"Column '{col}' has critical missingness ({m_pct}% missing).")

        # Check for constant column (only 1 distinct valid value)
        valid_s = s[~missing_mask[col]]
        if valid_s.nunique(dropna=True) == 1 and total_rows > 1:
            constant_columns.append(col)
            warnings.append(f"Column '{col}' is constant (all valid values are identical).")

        # Check for whitespace-only strings in text columns
        if (
            pd.api.types.is_object_dtype(s)
            or pd.api.types.is_string_dtype(s)
            or isinstance(s.dtype, pd.CategoricalDtype)
        ):
            non_null = s.dropna()
            if not non_null.empty:
                ws_cnt = int((non_null.astype(str).str.strip() == "").sum())
                if ws_cnt > 0:
                    whitespace_columns[col] = ws_cnt

            # Test if candidate for datetime parsing
            if not pd.api.types.is_datetime64_any_dtype(s) and not pd.api.types.is_numeric_dtype(s):
                sample = non_null.head(50)
                if len(sample) >= 5:
                    try:
                        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
                    except (ValueError, TypeError):
                        parsed = pd.to_datetime(sample, errors="coerce")
                    success_rate = parsed.notna().mean()
                    if success_rate >= 0.70:
                        date_candidate_columns.append(col)

    all_null_cols = [col for col in df.columns if missing_mask[col].sum() == total_rows]
    for col in all_null_cols:
        warnings.append(f"Column '{col}' is 100% missing (all null).")

    if duplicate_rows > 0:
        warnings.append(f"Found {duplicate_rows:,} duplicate rows ({duplicate_rows_pct}% of dataset).")

    return {
        "total_rows": total_rows,
        "row_count": total_rows,
        "total_cols": total_cols,
        "column_count": total_cols,
        "total_cells": total_cells,
        "total_missing_cells": total_missing_cells,
        "missing_cells_pct": missing_cells_pct,
        "duplicate_rows": duplicate_rows,
        "duplicate_rows_pct": duplicate_rows_pct,
        "missing_by_col": missing_by_col,
        "constant_columns": constant_columns,
        "constant_cols": constant_columns,
        "all_null_cols": all_null_cols,
        "whitespace_columns": whitespace_columns,
        "whitespace_only_cols": list(whitespace_columns.keys()),
        "date_candidate_columns": date_candidate_columns,
        "date_candidate_cols": date_candidate_columns,
        "warnings": warnings,
    }


# =====================================================================
# ALLOWLISTED TRANSFORMATION FUNCTIONS (Pure Pandas, No eval/exec)
# =====================================================================

def op_drop_missing_rows(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    how: str = "any",
    treat_empty_as_missing: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Drops rows containing missing values in specified columns or all columns.
    """
    target_cols = [c for c in columns if c in df.columns] if columns else list(df.columns)
    missing_mask = get_missing_mask(df[target_cols], treat_empty_as_missing=treat_empty_as_missing)

    if how == "all":
        rows_to_drop = missing_mask.all(axis=1)
    else:  # "any"
        rows_to_drop = missing_mask.any(axis=1)

    dropped_count = int(rows_to_drop.sum())
    result_df = df[~rows_to_drop].copy()

    details = {
        "operation": "Drop Missing Rows",
        "columns_evaluated": target_cols,
        "strategy": f"Drop if {how} missing",
        "rows_removed": dropped_count,
        "rows_remaining": len(result_df),
    }
    return result_df, details


def op_drop_columns(
    df: pd.DataFrame,
    columns: List[str]
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Drops specified columns from the DataFrame.
    """
    cols_to_drop = [c for c in columns if c in df.columns]
    if not cols_to_drop:
        return df.copy(), {"operation": "Drop Columns", "columns_dropped": [], "status": "No matching columns found."}

    result_df = df.drop(columns=cols_to_drop).copy()
    details = {
        "operation": "Drop Columns",
        "columns_dropped": cols_to_drop,
        "columns_remaining": list(result_df.columns),
    }
    return result_df, details


def op_impute_missing(
    df: pd.DataFrame,
    column: str,
    strategy: str = "mean",
    constant_value: Optional[Any] = None,
    treat_empty_as_missing: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Imputes missing values in a column using mean, median, mode, or constant value.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataset.")

    result_df = df.copy()
    series = result_df[column]
    missing_mask = get_missing_mask(result_df[[column]], treat_empty_as_missing=treat_empty_as_missing)[column]
    missing_count = int(missing_mask.sum())

    if missing_count == 0:
        return result_df, {
            "operation": "Impute Missing",
            "column": column,
            "strategy": strategy,
            "cells_imputed": 0,
            "note": "Column contains no missing values."
        }

    fill_val = None
    if strategy == "mean":
        num_s = pd.to_numeric(series[~missing_mask], errors="coerce").dropna()
        if num_s.empty:
            raise ValueError(f"Cannot compute mean for column '{column}': contains no usable numeric values.")
        fill_val = float(num_s.mean())
        result_df.loc[missing_mask, column] = fill_val

    elif strategy == "median":
        num_s = pd.to_numeric(series[~missing_mask], errors="coerce").dropna()
        if num_s.empty:
            raise ValueError(f"Cannot compute median for column '{column}': contains no usable numeric values.")
        fill_val = float(num_s.median())
        result_df.loc[missing_mask, column] = fill_val

    elif strategy == "mode":
        valid_s = series[~missing_mask]
        if valid_s.empty:
            raise ValueError(f"Cannot compute mode for column '{column}': all values are missing.")
        modes = valid_s.mode()
        fill_val = modes.iloc[0] if not modes.empty else "Unknown"
        result_df.loc[missing_mask, column] = fill_val

    elif strategy == "constant":
        if constant_value is None:
            raise ValueError("Constant value must be provided for constant imputation.")
        fill_val = constant_value
        result_df.loc[missing_mask, column] = fill_val

    else:
        raise ValueError(f"Unsupported imputation strategy: '{strategy}'.")

    details = {
        "operation": "Impute Missing",
        "column": column,
        "strategy": strategy,
        "fill_value": fill_val,
        "cells_imputed": missing_count,
    }
    return result_df, details


def op_replace_whitespace_with_nan(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Replaces empty or whitespace-only strings with np.nan in target text columns.
    """
    target_cols = [c for c in columns if c in df.columns] if columns else [
        c for c in df.columns
        if pd.api.types.is_object_dtype(df[c])
        or pd.api.types.is_string_dtype(df[c])
        or isinstance(df[c].dtype, pd.CategoricalDtype)
    ]

    result_df = df.copy()
    cells_replaced = 0

    for col in target_cols:
        s = result_df[col]
        non_null_idx = s.dropna().index
        if not non_null_idx.empty:
            ws_mask = s.loc[non_null_idx].astype(str).str.strip() == ""
            if ws_mask.any():
                match_indices = ws_mask[ws_mask].index
                cells_replaced += len(match_indices)
                result_df.loc[match_indices, col] = np.nan

    details = {
        "operation": "Replace Whitespace with NaN",
        "columns_affected": target_cols,
        "cells_replaced": cells_replaced,
    }
    return result_df, details


def op_remove_duplicates(
    df: pd.DataFrame,
    subset_columns: Optional[List[str]] = None,
    keep: Union[str, bool] = "first"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Removes duplicate rows based on all or a subset of columns.
    keep: 'first', 'last', or False (drop all occurrences of duplicates).
    """
    target_cols = [c for c in subset_columns if c in df.columns] if subset_columns else None
    initial_rows = len(df)
    result_df = df.drop_duplicates(subset=target_cols, keep=keep).copy()
    dropped_rows = initial_rows - len(result_df)

    details = {
        "operation": "Remove Duplicates",
        "subset_columns": target_cols if target_cols else "All Columns",
        "keep_policy": str(keep),
        "rows_removed": dropped_rows,
        "rows_remaining": len(result_df),
    }
    return result_df, details


def op_convert_type(
    df: pd.DataFrame,
    column: str,
    target_type: str,
    errors: str = "coerce",
    datetime_format: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Converts a column to a target data type: 'numeric', 'datetime', 'categorical', 'string', 'boolean'.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found.")

    result_df = df.copy()
    orig_dtype = str(result_df[column].dtype)
    orig_nulls = int(result_df[column].isna().sum())

    if target_type == "numeric":
        result_df[column] = pd.to_numeric(result_df[column], errors=errors)
    elif target_type == "datetime":
        try:
            if datetime_format:
                result_df[column] = pd.to_datetime(result_df[column], errors=errors, format=datetime_format)
            else:
                result_df[column] = pd.to_datetime(result_df[column], errors=errors, format="mixed")
        except (ValueError, TypeError):
            result_df[column] = pd.to_datetime(result_df[column], errors=errors)
    elif target_type == "categorical":
        result_df[column] = result_df[column].astype("category")
    elif target_type == "string":
        result_df[column] = result_df[column].astype(str)
    elif target_type == "boolean":
        result_df[column] = result_df[column].astype(bool)
    else:
        raise ValueError(f"Unsupported target type: '{target_type}'.")

    new_nulls = int(result_df[column].isna().sum())
    failed_conversions = max(0, new_nulls - orig_nulls)

    details = {
        "operation": "Type Conversion",
        "column": column,
        "original_type": orig_dtype,
        "target_type": target_type,
        "new_type": str(result_df[column].dtype),
        "failed_conversions": failed_conversions,
    }
    return result_df, details


def op_normalize_text(
    df: pd.DataFrame,
    column: str,
    strip_whitespace: bool = True,
    case_style: Optional[str] = "lower"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Normalizes text values by stripping whitespace and applying case style (lower, upper, title).
    Preserves missing values.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found.")

    result_df = df.copy()
    s = result_df[column]
    non_null_mask = s.notna()

    if non_null_mask.any():
        text_s = s[non_null_mask].astype(str)
        if strip_whitespace:
            text_s = text_s.str.strip()
        if case_style == "lower":
            text_s = text_s.str.lower()
        elif case_style == "upper":
            text_s = text_s.str.upper()
        elif case_style == "title":
            text_s = text_s.str.title()

        result_df.loc[non_null_mask, column] = text_s

    details = {
        "operation": "Normalize Text",
        "column": column,
        "strip_whitespace": strip_whitespace,
        "case_style": case_style,
        "records_processed": int(non_null_mask.sum()),
    }
    return result_df, details


# =====================================================================
# FEATURE ENGINEERING TEMPLATES
# =====================================================================

def op_feature_arithmetic(
    df: pd.DataFrame,
    col_a: str,
    col_b_or_scalar: Union[str, float, int],
    operation: str,
    new_col_name: str
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Creates a new numeric feature by combining two columns or a column with a scalar:
    add, subtract, multiply, divide.
    Handles divide-by-zero safely by inserting np.nan.
    """
    if col_a not in df.columns:
        raise ValueError(f"Column '{col_a}' not found.")
    if not new_col_name or str(new_col_name).strip() == "":
        raise ValueError("A valid new feature name must be provided.")

    result_df = df.copy()
    val_a = pd.to_numeric(result_df[col_a], errors="coerce")
    if val_a.dropna().empty:
        raise ValueError(f"Column '{col_a}' contains no numeric values for arithmetic.")

    if isinstance(col_b_or_scalar, str) and col_b_or_scalar in df.columns:
        val_b = pd.to_numeric(result_df[col_b_or_scalar], errors="coerce")
        if val_b.dropna().empty:
            raise ValueError(f"Column '{col_b_or_scalar}' contains no numeric values for arithmetic.")
        operand_label = col_b_or_scalar
    else:
        try:
            val_b = float(col_b_or_scalar)
            operand_label = str(col_b_or_scalar)
        except (ValueError, TypeError):
            raise ValueError(f"Operand '{col_b_or_scalar}' is not a valid column or numeric scalar.")

    div_by_zero_count = 0
    if operation == "add":
        res = val_a + val_b
        formula = f"{col_a} + {operand_label}"
    elif operation == "subtract":
        res = val_a - val_b
        formula = f"{col_a} - {operand_label}"
    elif operation == "multiply":
        res = val_a * val_b
        formula = f"{col_a} * {operand_label}"
    elif operation == "divide":
        # Handle divide by zero safely
        if isinstance(val_b, (int, float)):
            if val_b == 0:
                res = pd.Series(np.nan, index=df.index)
                div_by_zero_count = len(df)
            else:
                res = val_a / val_b
        else:
            zero_mask = val_b == 0
            div_by_zero_count = int(zero_mask.sum())
            res = val_a / val_b.replace(0, np.nan)
        formula = f"{col_a} / {operand_label}"
    else:
        raise ValueError(f"Unsupported arithmetic operation: '{operation}'.")

    result_df[new_col_name] = res

    details = {
        "operation": "Feature: Arithmetic",
        "formula": formula,
        "new_column": new_col_name,
        "divide_by_zero_count": div_by_zero_count,
        "valid_count": int(res.notna().sum()),
    }
    return result_df, details


def op_feature_math(
    df: pd.DataFrame,
    column: str,
    math_func: str,
    new_col_name: str
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Applies mathematical transformations: 'abs', 'log', 'log1p'.
    For log and log1p, invalid non-positive inputs are set to np.nan safely.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found.")
    if not new_col_name or str(new_col_name).strip() == "":
        raise ValueError("A valid new feature name must be provided.")

    result_df = df.copy()
    num_s = pd.to_numeric(result_df[column], errors="coerce")
    invalid_inputs = 0

    if math_func == "abs":
        res = num_s.abs()
        formula = f"abs({column})"
    elif math_func == "log":
        invalid_mask = num_s <= 0
        invalid_inputs = int(invalid_mask.sum())
        clean_num = num_s.mask(invalid_mask, np.nan)
        res = np.log(clean_num)
        formula = f"log({column})"
    elif math_func == "log1p":
        invalid_mask = num_s < -1
        invalid_inputs = int(invalid_mask.sum())
        clean_num = num_s.mask(invalid_mask, np.nan)
        res = np.log1p(clean_num)
        formula = f"log1p({column})"
    else:
        raise ValueError(f"Unsupported math transformation: '{math_func}'.")

    result_df[new_col_name] = res

    details = {
        "operation": "Feature: Math Transformation",
        "formula": formula,
        "new_column": new_col_name,
        "invalid_inputs": invalid_inputs,
        "valid_count": int(res.notna().sum()),
    }
    return result_df, details


def op_feature_binning(
    df: pd.DataFrame,
    column: str,
    bin_type: str = "quantile",
    bins: int = 4,
    labels: Optional[List[str]] = None,
    new_col_name: str = "binned_feature"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Bins numeric features into quantile bins or uniform cut bins.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found.")

    result_df = df.copy()
    num_s = pd.to_numeric(result_df[column], errors="coerce")

    if bin_type == "quantile":
        # pd.qcut
        try:
            binned = pd.qcut(num_s, q=bins, labels=labels, duplicates="drop")
        except ValueError as e:
            raise ValueError(f"Cannot perform quantile binning: {str(e)}")
    else:  # uniform cut
        binned = pd.cut(num_s, bins=bins, labels=labels)

    result_df[new_col_name] = binned.astype(str)

    details = {
        "operation": "Feature: Binning",
        "column": column,
        "bin_type": bin_type,
        "bins_requested": bins,
        "new_column": new_col_name,
    }
    return result_df, details


def op_feature_datetime_extract(
    df: pd.DataFrame,
    column: str,
    components: List[str],
    prefix: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Extracts datetime parts: 'year', 'month', 'day', 'dayofweek', 'quarter'.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found.")

    result_df = df.copy()
    try:
        dt_s = pd.to_datetime(result_df[column], errors="coerce", format="mixed")
    except (ValueError, TypeError):
        dt_s = pd.to_datetime(result_df[column], errors="coerce")

    prefix_str = f"{prefix}_" if prefix else f"{column}_"
    created_cols = []

    for comp in components:
        col_name = f"{prefix_str}{comp}"
        if comp == "year":
            result_df[col_name] = dt_s.dt.year
        elif comp == "month":
            result_df[col_name] = dt_s.dt.month
        elif comp == "day":
            result_df[col_name] = dt_s.dt.day
        elif comp == "dayofweek":
            result_df[col_name] = dt_s.dt.dayofweek
        elif comp == "quarter":
            result_df[col_name] = dt_s.dt.quarter
        created_cols.append(col_name)

    details = {
        "operation": "Feature: Datetime Extraction",
        "source_column": column,
        "components_extracted": components,
        "created_columns": created_cols,
    }
    return result_df, details


def op_feature_date_diff(
    df: pd.DataFrame,
    col_start: str,
    col_end: str,
    new_col_name: str
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Calculates time difference in days between two date columns: (col_end - col_start).
    """
    if col_start not in df.columns or col_end not in df.columns:
        raise ValueError(f"Both '{col_start}' and '{col_end}' must exist in DataFrame.")

    result_df = df.copy()
    dt_start = pd.to_datetime(result_df[col_start], errors="coerce")
    dt_end = pd.to_datetime(result_df[col_end], errors="coerce")

    diff_days = (dt_end - dt_start).dt.total_seconds() / (24 * 3600)
    result_df[new_col_name] = diff_days

    details = {
        "operation": "Feature: Date Difference",
        "col_start": col_start,
        "col_end": col_end,
        "new_column": new_col_name,
        "formula": f"({col_end} - {col_start}) in days",
    }
    return result_df, details


def op_feature_category_map(
    df: pd.DataFrame,
    column: str,
    mapping: Dict[str, Any],
    default_val: Optional[Any] = "Other",
    new_col_name: str = "mapped_feature"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Maps categorical values using an explicit mapping dictionary.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found.")

    result_df = df.copy()
    s_clean = result_df[column].astype(str).str.strip()
    mapped = s_clean.map(mapping)
    if default_val is not None:
        mapped = mapped.fillna(default_val)

    result_df[new_col_name] = mapped

    details = {
        "operation": "Feature: Category Mapping",
        "column": column,
        "new_column": new_col_name,
        "mapped_categories_count": len(mapping),
    }
    return result_df, details


def op_feature_binary_indicator(
    df: pd.DataFrame,
    column: str,
    target_value: str,
    new_col_name: str
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Creates a binary indicator column (1 if column equals target_value, else 0).
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found.")

    result_df = df.copy()
    target_clean = str(target_value).strip().lower()
    match_mask = result_df[column].astype(str).str.strip().str.lower() == target_clean

    result_df[new_col_name] = match_mask.astype(int)

    details = {
        "operation": "Feature: Binary Indicator",
        "column": column,
        "target_value": target_value,
        "new_column": new_col_name,
        "positive_count": int(match_mask.sum()),
    }
    return result_df, details


def op_feature_one_hot(
    df: pd.DataFrame,
    column: str,
    drop_first: bool = False,
    prefix: Optional[str] = None,
    max_categories: int = 20
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    One-hot encodes a categorical column up to max_categories.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found.")

    result_df = df.copy()
    series = result_df[column].dropna().astype(str).str.strip()
    top_cats = series.value_counts().head(max_categories).index.tolist()

    prefix_str = prefix if prefix else str(column)
    created_cols = []

    # Filter to top categories or other
    cats_to_encode = top_cats[1:] if drop_first else top_cats
    for cat in cats_to_encode:
        col_name = f"{prefix_str}_{cat}".replace(" ", "_")
        result_df[col_name] = (result_df[column].astype(str).str.strip() == cat).astype(int)
        created_cols.append(col_name)

    details = {
        "operation": "Feature: One-Hot Encoding",
        "column": column,
        "drop_first": drop_first,
        "categories_encoded": len(cats_to_encode),
        "created_columns": created_cols,
    }
    return result_df, details


def op_feature_text_length(
    df: pd.DataFrame,
    column: str,
    new_col_name: str
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Creates a numeric feature measuring string character length.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found.")

    result_df = df.copy()
    s = result_df[column]
    res = s.astype(str).str.len()
    # If original was NaN, preserve NaN
    res = res.mask(s.isna(), np.nan)
    result_df[new_col_name] = res

    details = {
        "operation": "Feature: Text Length",
        "column": column,
        "new_column": new_col_name,
    }
    return result_df, details


def op_feature_word_count(
    df: pd.DataFrame,
    column: str,
    new_col_name: str
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Creates a numeric feature counting the number of words in a text column.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found.")

    result_df = df.copy()
    s = result_df[column]
    word_counts = s.dropna().astype(str).str.strip().str.split().str.len()
    res = pd.Series(np.nan, index=df.index)
    res.loc[word_counts.index] = word_counts
    result_df[new_col_name] = res

    details = {
        "operation": "Feature: Word Count",
        "column": column,
        "new_column": new_col_name,
    }
    return result_df, details


# =====================================================================
# NETFLIX FEATURE RECIPES (Integrated & Preconditioned)
# =====================================================================

NETFLIX_RECIPES = {
    "acquisition_lag": {
        "title": "Licensing / Acquisition Lag (year_added - release_year)",
        "required_columns": ["year_added", "release_year"],
        "formula": "np.clip(year_added - release_year, 0, None)",
        "description": "Difference in years between release and platform addition. Same-year acquisition is a proxy for Originals.",
    },
    "content_origin_proxy": {
        "title": "Content Origin Proxy (Original vs Recent vs Back-Catalog)",
        "required_columns": ["acquisition_lag"],
        "formula": "Tiers: Same-Year (0 yrs), Recent Release (1-3 yrs), Back-Catalog (>3 yrs)",
        "description": "Categorizes licensing lag into strategic release timing tiers.",
    },
    "audience_tier": {
        "title": "Audience Maturity Tier (Kids / Family / Mature)",
        "required_columns": ["rating"],
        "formula": "Consolidates MPAA and TV ratings into Kids, Family / Teens, Mature / Adults",
        "description": "Groups ratings into business-oriented audience tiers.",
    },
    "primary_genre": {
        "title": "Primary Genre Extraction (Lead Category)",
        "required_columns": ["listed_in"],
        "formula": "Extracts first item from comma-separated listed_in",
        "description": "Identifies the lead genre category for each title.",
    },
    "primary_country": {
        "title": "Primary Producing Country",
        "required_columns": ["country"],
        "formula": "Extracts first item from comma-separated country",
        "description": "Identifies the primary production territory.",
    },
    "metadata_soup": {
        "title": "NLP Metadata Soup",
        "required_columns": ["title", "description"],
        "formula": "Concatenates cleaned text fields for content embeddings",
        "description": "Combines metadata tokens for recommendation TF-IDF vectorization.",
    },
}


def op_recipe_netflix_features(
    df: pd.DataFrame,
    recipe_name: str
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Applies existing Netflix-specific feature engineering recipes safely
    after verifying schema preconditions.
    """
    result_df = df.copy()

    if recipe_name == "acquisition_lag":
        if "year_added" not in df.columns or "release_year" not in df.columns:
            if "date_added" in df.columns:
                dt = pd.to_datetime(df["date_added"], errors="coerce")
                result_df["year_added"] = dt.dt.year

        if "year_added" not in result_df.columns or "release_year" not in result_df.columns:
            raise ValueError("Recipe 'acquisition_lag' requires both 'year_added' and 'release_year' columns. Missing required columns.")

        y_add = pd.to_numeric(result_df["year_added"], errors="coerce")
        r_year = pd.to_numeric(result_df["release_year"], errors="coerce")
        lag = np.clip(y_add - r_year, 0, None)
        result_df["acquisition_lag"] = lag

        details = {
            "operation": "Netflix Recipe: Acquisition Lag",
            "formula": "np.clip(year_added - release_year, 0, None)",
            "new_column": "acquisition_lag",
            "explanation": "Difference in years between release and platform addition. Same-year (lag=0) is a proxy for Originals.",
        }
        return result_df, details

    elif recipe_name == "content_origin_proxy":
        if "acquisition_lag" not in result_df.columns:
            if "year_added" not in df.columns and "date_added" not in df.columns:
                raise ValueError("Recipe 'content_origin_proxy' requires 'acquisition_lag' or ('year_added' and 'release_year') columns. Missing required columns.")
            result_df, _ = op_recipe_netflix_features(result_df, "acquisition_lag")

        lag = result_df["acquisition_lag"]
        conditions = [
            lag == 0,
            (lag >= 1) & (lag <= 3),
            lag > 3
        ]
        choices = [
            "Likely Original / Day-and-Date",
            "Recent Release",
            "Catalog Acquisition"
        ]
        result_df["content_origin_proxy"] = np.select(conditions, choices, default="Catalog Acquisition")

        details = {
            "operation": "Netflix Recipe: Content Origin Proxy",
            "new_column": "content_origin_proxy",
            "categories": choices,
            "explanation": "Classifies licensing lag into strategic release timing tiers.",
        }
        return result_df, details

    elif recipe_name == "audience_tier":
        if "rating" not in df.columns:
            raise ValueError("Recipe 'audience_tier' requires 'rating' column. Missing required columns.")

        from src.feature_engineer import AUDIENCE_TIER_MAP
        tiers = df["rating"].astype(str).str.strip().map(AUDIENCE_TIER_MAP).fillna("Mature / Adults")
        result_df["audience_tier"] = tiers

        details = {
            "operation": "Netflix Recipe: Audience Maturity Tier",
            "new_column": "audience_tier",
            "mapped_tiers": ["Kids", "Family / Teens", "Mature / Adults"],
            "explanation": "Consolidates 10+ MPAA and TV ratings into 3 standard business tiers.",
        }
        return result_df, details

    elif recipe_name == "primary_genre":
        if "listed_in" not in df.columns:
            raise ValueError("Recipe 'primary_genre' requires 'listed_in' column. Missing required columns.")

        result_df["primary_genre"] = (
            df["listed_in"].dropna().astype(str).apply(lambda s: s.split(",")[0].strip())
        )
        details = {
            "operation": "Netflix Recipe: Primary Genre",
            "new_column": "primary_genre",
            "explanation": "Extracts the dominant leading category from multi-label comma-separated strings.",
        }
        return result_df, details

    elif recipe_name == "primary_country":
        if "country" not in df.columns:
            raise ValueError("Recipe 'primary_country' requires 'country' column. Missing required columns.")

        result_df["primary_country"] = (
            df["country"].dropna().astype(str).apply(lambda s: s.split(",")[0].strip())
        )
        details = {
            "operation": "Netflix Recipe: Primary Country",
            "new_column": "primary_country",
            "explanation": "Extracts the lead producing nation from multi-country co-productions.",
        }
        return result_df, details

    elif recipe_name == "metadata_soup":
        if "title" not in df.columns:
            raise ValueError("Recipe 'metadata_soup' requires 'title' column. Missing required columns.")
        from src.feature_engineer import create_metadata_soup
        result_df["metadata_soup"] = create_metadata_soup(result_df)
        details = {
            "operation": "Netflix Recipe: Metadata Soup",
            "new_column": "metadata_soup",
            "explanation": "Concatenates cleaned titles, director tokens, cast tokens, genres, and synopses for NLP embeddings.",
        }
        return result_df, details

    else:
        raise ValueError(f"Unknown Netflix recipe: '{recipe_name}'.")


# =====================================================================
# WHAT CHANGED? COMPARISON ENGINE
# =====================================================================

def compute_what_changed(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    step_info: Optional[Dict[str, Any]] = None,
    affected_columns: Optional[List[str]] = None,
    treat_empty_as_missing: bool = True
) -> Dict[str, Any]:
    """
    Computes a comprehensive quantitative and qualitative comparison
    between DataFrame states immediately before and after a transformation.
    """
    if step_info is None:
        step_info = {}

    rows_before = len(df_before)
    rows_after = len(df_after)
    rows_delta = rows_after - rows_before

    cols_before = list(df_before.columns)
    cols_after = list(df_after.columns)
    cols_added = [c for c in cols_after if c not in cols_before]
    cols_removed = [c for c in cols_before if c not in cols_after]
    cols_retained = [c for c in cols_after if c in cols_before]

    # Data type modifications
    type_changes = {}
    for c in cols_retained:
        t_bef = str(df_before[c].dtype)
        t_aft = str(df_after[c].dtype)
        if t_bef != t_aft:
            type_changes[c] = {"before": t_bef, "after": t_aft}

    # Missing cell audits
    m_mask_before = get_missing_mask(df_before, treat_empty_as_missing=treat_empty_as_missing)
    m_mask_after = get_missing_mask(df_after, treat_empty_as_missing=treat_empty_as_missing)
    missing_cells_before = int(m_mask_before.to_numpy().sum()) if not df_before.empty else 0
    missing_cells_after = int(m_mask_after.to_numpy().sum()) if not df_after.empty else 0
    missing_cells_delta = missing_cells_after - missing_cells_before

    # Duplicate rows
    dup_before = int(df_before.duplicated().sum()) if not df_before.empty else 0
    dup_after = int(df_after.duplicated().sum()) if not df_after.empty else 0
    dup_delta = dup_after - dup_before

    # Statistics comparison for affected or newly created columns
    stats_comparison = {}
    candidate_cols = cols_added + list(type_changes.keys())
    if affected_columns:
        candidate_cols.extend(affected_columns)
    if "column" in step_info and step_info["column"] in cols_retained:
        candidate_cols.append(step_info["column"])

    for col in list(dict.fromkeys(candidate_cols)):
        col_stats = {}
        if col in df_before.columns and pd.api.types.is_numeric_dtype(df_before[col]):
            s_b = pd.to_numeric(df_before[col], errors="coerce").dropna()
            col_stats["before"] = {
                "mean": round(float(s_b.mean()), 2) if not s_b.empty else None,
                "median": round(float(s_b.median()), 2) if not s_b.empty else None,
                "min": round(float(s_b.min()), 2) if not s_b.empty else None,
                "max": round(float(s_b.max()), 2) if not s_b.empty else None,
                "missing": int(m_mask_before[col].sum()),
            }
        if col in df_after.columns and pd.api.types.is_numeric_dtype(df_after[col]):
            s_a = pd.to_numeric(df_after[col], errors="coerce").dropna()
            col_stats["after"] = {
                "mean": round(float(s_a.mean()), 2) if not s_a.empty else None,
                "median": round(float(s_a.median()), 2) if not s_a.empty else None,
                "min": round(float(s_a.min()), 2) if not s_a.empty else None,
                "max": round(float(s_a.max()), 2) if not s_a.empty else None,
                "missing": int(m_mask_after[col].sum()),
            }
        if col_stats:
            stats_comparison[col] = col_stats

    # Sample rows preview before & after
    sample_preview_before = df_before.head(3).to_dict(orient="records") if not df_before.empty else []
    sample_preview_after = df_after.head(3).to_dict(orient="records") if not df_after.empty else []

    return {
        "step_info": step_info,
        "rows_before": rows_before,
        "rows_after": rows_after,
        "row_delta": rows_delta,
        "rows_delta": rows_delta,
        "cols_before": len(cols_before),
        "cols_after": len(cols_after),
        "col_delta": len(cols_after) - len(cols_before),
        "columns_before_count": len(cols_before),
        "columns_after_count": len(cols_after),
        "columns_added": cols_added,
        "columns_removed": cols_removed,
        "cols_removed": cols_removed,
        "type_changes": type_changes,
        "missing_cells_before": missing_cells_before,
        "missing_cells_after": missing_cells_after,
        "missing_cells_delta": missing_cells_delta,
        "duplicates_before": dup_before,
        "duplicates_after": dup_after,
        "duplicates_delta": dup_delta,
        "stats_comparison": stats_comparison,
        "sample_preview_before": sample_preview_before,
        "sample_preview_after": sample_preview_after,
    }


# =====================================================================
# WORKING DATASET & REVERSIBLE STATE MANAGER
# =====================================================================

class WorkingDatasetManager:
    """
    Manages interactive data cleaning and feature engineering state.
    Maintains an immutable raw source, a mutable working copy, a list of
    serializable operation configurations, and an audit ledger for deterministic undo/replay.
    """

    def __init__(self, source_df: pd.DataFrame, source_identity: str = "custom_source", source_name: str = "dataset"):
        self.source_df = source_df.copy()
        self.source_identity = source_identity
        self.source_name = source_name
        self.working_df = source_df.copy()
        self.history: List[Dict[str, Any]] = []
        self.audit_records: List[Dict[str, Any]] = []
        self.last_comparison: Optional[Dict[str, Any]] = None

    def get_working_df(self) -> pd.DataFrame:
        """Returns the current working copy DataFrame."""
        return self.working_df

    @property
    def applied_steps(self) -> List[Dict[str, Any]]:
        """Returns the list of applied step configurations."""
        return self.history

    def apply_step(self, step_config: Dict[str, Any]) -> Dict[str, Any]:
        """Unified step execution method used by programmatic pipelines and tests."""
        op_type = step_config.get("op_type") or step_config.get("operation_type")
        params = {k: v for k, v in step_config.items() if k not in ("op_type", "operation_type")}
        try:
            comparison = self.apply_operation(op_type, params)
            return {"success": True, "comparison": comparison}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def preview_operation(
        self,
        op_type: str,
        params: Dict[str, Any],
        treat_empty_as_missing: bool = True
    ) -> Dict[str, Any]:
        """
        Simulates an operation on the current working copy without persisting changes.
        Returns a 'What Changed?' preview dictionary.
        """
        df_copy = self.working_df.copy()
        transformed_df, details = self._dispatch_operation(df_copy, op_type, params)
        comparison = compute_what_changed(
            df_before=self.working_df,
            df_after=transformed_df,
            step_info=details,
            treat_empty_as_missing=treat_empty_as_missing
        )
        return comparison

    def apply_operation(
        self,
        op_type: str,
        params: Dict[str, Any],
        treat_empty_as_missing: bool = True
    ) -> Dict[str, Any]:
        """
        Applies an operation to the working copy, logs it to history, and records comparison.
        """
        df_before = self.working_df.copy()
        transformed_df, details = self._dispatch_operation(df_before, op_type, params)

        step_number = len(self.history) + 1
        step_config = {
            "step_number": step_number,
            "operation_type": op_type,
            "op_type": op_type,
            "params": params,
            "details": details,
        }

        comparison = compute_what_changed(
            df_before=df_before,
            df_after=transformed_df,
            step_info=details,
            treat_empty_as_missing=treat_empty_as_missing
        )

        # Commit state
        self.working_df = transformed_df
        self.history.append(step_config)
        self.audit_records.append(comparison)
        self.last_comparison = comparison
        return comparison

    def undo_last_step(self) -> Optional[Dict[str, Any]]:
        """
        Undoes the most recently applied operation by replaying the plan from raw source.
        Returns the removed step details or None if history was empty.
        """
        if not self.history:
            return None

        removed_step = self.history.pop()
        self.audit_records.pop()

        # Replay remaining history deterministically from the immutable source
        replayed_df = self.source_df.copy()
        for step in self.history:
            replayed_df, _ = self._dispatch_operation(replayed_df, step["operation_type"], step["params"])

        self.working_df = replayed_df
        self.last_comparison = self.audit_records[-1] if self.audit_records else None
        return removed_step

    def reset_to_source(self) -> pd.DataFrame:
        """
        Discards all transformations and restores the working copy to the raw source.
        """
        self.working_df = self.source_df.copy()
        self.history.clear()
        self.audit_records.clear()
        self.last_comparison = None
        return self.working_df

    def _dispatch_operation(
        self,
        df: pd.DataFrame,
        op_type: str,
        params: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Dispatches operation calls to the allowlisted pure transformation registry.
        """
        # Column collision safety for generated features
        new_col = params.get("new_col_name") or params.get("output_col")
        if new_col and new_col in df.columns and not params.get("overwrite", False):
            raise ValueError(f"Column '{new_col}' already exists in DataFrame. Specify overwrite=True to replace.")

        if op_type in ("drop_missing_rows",):
            return op_drop_missing_rows(
                df,
                columns=params.get("columns"),
                how=params.get("how", "any"),
                treat_empty_as_missing=params.get("treat_empty_as_missing", True)
            )
        elif op_type in ("drop_columns",):
            return op_drop_columns(df, columns=params.get("columns", []))
        elif op_type in ("impute_missing", "impute_numeric"):
            strat = params.get("strategy", "mean")
            const_v = params.get("constant_value") or params.get("fill_value")
            return op_impute_missing(
                df,
                column=params.get("column", ""),
                strategy=strat,
                constant_value=const_v,
                treat_empty_as_missing=params.get("treat_empty_as_missing", True)
            )
        elif op_type == "impute_categorical_mode":
            return op_impute_missing(
                df,
                column=params.get("column", ""),
                strategy="mode",
                treat_empty_as_missing=params.get("treat_empty_as_missing", True)
            )
        elif op_type == "fill_constant":
            const_v = params.get("fill_value") or params.get("constant_value")
            return op_impute_missing(
                df,
                column=params.get("column", ""),
                strategy="constant",
                constant_value=const_v,
                treat_empty_as_missing=params.get("treat_empty_as_missing", True)
            )
        elif op_type in ("replace_whitespace_with_nan", "whitespace_to_nan"):
            return op_replace_whitespace_with_nan(df, columns=params.get("columns"))
        elif op_type == "remove_duplicates":
            keep_val = params.get("keep", "first")
            if keep_val == "none":
                keep_val = False
            return op_remove_duplicates(
                df,
                subset_columns=params.get("subset_columns") or params.get("columns"),
                keep=keep_val
            )
        elif op_type == "convert_type":
            return op_convert_type(
                df,
                column=params.get("column", ""),
                target_type=params.get("target_type", "string"),
                errors=params.get("errors", "coerce"),
                datetime_format=params.get("datetime_format")
            )
        elif op_type == "normalize_text":
            strip_w = params.get("strip_whitespace", True) if "strip_whitespace" in params else params.get("trim", True)
            c_style = params.get("case_style", "lower") if "case_style" in params else params.get("case", "lower")
            return op_normalize_text(
                df,
                column=params.get("column", ""),
                strip_whitespace=strip_w,
                case_style=c_style
            )
        elif op_type in ("feature_arithmetic", "numeric_arithmetic"):
            op = params.get("operation") or params.get("operator", "add")
            col_b = params.get("col_b") if "col_b" in params else params.get("col_b_or_scalar", 0)
            out_c = params.get("new_col_name") or params.get("output_col", "new_feature")
            return op_feature_arithmetic(
                df,
                col_a=params.get("col_a", ""),
                col_b_or_scalar=col_b,
                operation=op,
                new_col_name=out_c
            )
        elif op_type in ("feature_math", "math_transform"):
            func = params.get("math_func") or params.get("transform", "abs")
            out_c = params.get("new_col_name") or params.get("output_col", "new_feature")
            return op_feature_math(
                df,
                column=params.get("column", ""),
                math_func=func,
                new_col_name=out_c
            )
        elif op_type in ("feature_binning", "numeric_binning"):
            b_type = params.get("bin_type") or params.get("method", "quantile")
            out_c = params.get("new_col_name") or params.get("output_col", "binned_feature")
            return op_feature_binning(
                df,
                column=params.get("column", ""),
                bin_type=b_type,
                bins=params.get("bins", 4),
                labels=params.get("labels"),
                new_col_name=out_c
            )
        elif op_type in ("feature_datetime_extract", "datetime_extract"):
            comp = params.get("components")
            if not comp and params.get("part"):
                comp = [params["part"]]
            elif not comp:
                comp = ["year", "month"]
            out_p = params.get("prefix") or params.get("output_col")
            res_df, details = op_feature_datetime_extract(
                df,
                column=params.get("column", ""),
                components=comp,
                prefix=out_p
            )
            if params.get("output_col") and len(comp) == 1:
                old_col = f"{out_p}_{comp[0]}" if out_p else f"{params.get('column')}_{comp[0]}"
                if old_col in res_df.columns and old_col != params["output_col"]:
                    res_df = res_df.rename(columns={old_col: params["output_col"]})
            return res_df, details
        elif op_type in ("feature_date_diff", "date_difference"):
            start_c = params.get("col_start") or params.get("date_col_a", "")
            end_c = params.get("col_end") or params.get("date_col_b", "")
            out_c = params.get("new_col_name") or params.get("output_col", "date_diff_days")
            return op_feature_date_diff(
                df,
                col_start=start_c,
                col_end=end_c,
                new_col_name=out_c
            )
        elif op_type in ("feature_category_map", "categorical_map"):
            def_v = params.get("default_val") if "default_val" in params else params.get("default_value", "Other")
            out_c = params.get("new_col_name") or params.get("output_col", "mapped_feature")
            return op_feature_category_map(
                df,
                column=params.get("column", ""),
                mapping=params.get("mapping", {}),
                default_val=def_v,
                new_col_name=out_c
            )
        elif op_type in ("feature_binary_indicator", "binary_indicator"):
            out_c = params.get("new_col_name") or params.get("output_col", "is_target")
            return op_feature_binary_indicator(
                df,
                column=params.get("column", ""),
                target_value=params.get("target_value", ""),
                new_col_name=out_c
            )
        elif op_type in ("feature_one_hot", "one_hot_encode"):
            return op_feature_one_hot(
                df,
                column=params.get("column", ""),
                drop_first=params.get("drop_first", False),
                prefix=params.get("prefix"),
                max_categories=params.get("max_categories", 20)
            )
        elif op_type in ("feature_text_length",):
            out_c = params.get("new_col_name") or params.get("output_col", "text_length")
            return op_feature_text_length(df, column=params.get("column", ""), new_col_name=out_c)
        elif op_type in ("feature_word_count",):
            out_c = params.get("new_col_name") or params.get("output_col", "word_count")
            return op_feature_word_count(df, column=params.get("column", ""), new_col_name=out_c)
        elif op_type == "text_transform":
            tr = params.get("transform", "length")
            out_c = params.get("new_col_name") or params.get("output_col", "text_feature")
            if tr == "word_count":
                return op_feature_word_count(df, column=params.get("column", ""), new_col_name=out_c)
            elif tr == "length":
                return op_feature_text_length(df, column=params.get("column", ""), new_col_name=out_c)
            elif tr == "lower":
                return op_normalize_text(df, column=params.get("column", ""), strip_whitespace=False, case_style="lower")
            else:
                raise ValueError(f"Unsupported text transform: '{tr}'.")
        elif op_type == "netflix_recipe":
            rec_k = params.get("recipe_name") or params.get("recipe_key", "")
            return op_recipe_netflix_features(df, recipe_name=rec_k)
        else:
            raise ValueError(f"Unsupported or unallowlisted transformation operation: '{op_type}'.")

    def export_transformed_csv(self) -> str:
        """
        Exports the complete working DataFrame to CSV string without Pandas index.
        """
        buf = io.StringIO()
        self.working_df.to_csv(buf, index=False)
        return buf.getvalue()

    def export_cleaning_report_csv(self) -> str:
        """
        Generates a structured CSV report summarizing all applied transformations.
        """
        report_rows = []
        for audit in self.audit_records:
            info = audit["step_info"]
            report_rows.append({
                "Operation": info.get("operation", "Unknown"),
                "Rows Before": audit["rows_before"],
                "Rows After": audit["rows_after"],
                "Rows Delta": audit["rows_delta"],
                "Cols Added": ", ".join(audit["columns_added"]) if audit["columns_added"] else "None",
                "Cols Removed": ", ".join(audit["columns_removed"]) if audit["columns_removed"] else "None",
                "Missing Cells Delta": audit["missing_cells_delta"],
                "Duplicates Delta": audit["duplicates_delta"],
                "Details": str(info),
            })

        report_df = pd.DataFrame(report_rows)
        buf = io.StringIO()
        report_df.to_csv(buf, index=False)
        return buf.getvalue()

    def export_pipeline_json(self) -> str:
        """
        Exports the serializable ordered transformation configuration plan as JSON.
        """
        export_data = {
            "source_name": self.source_name,
            "source_identity": self.source_identity,
            "pipeline_steps": self.history,
            "steps": self.history,
        }
        return json.dumps(export_data, indent=2, default=str)


def export_working_csv(df: pd.DataFrame) -> str:
    """Exports a DataFrame to CSV string without Pandas index."""
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def export_cleaning_report(manager: WorkingDatasetManager) -> str:
    """Exports a comprehensive markdown report summarizing applied transformations."""
    lines = [
        "# DataLab: Clean & Transform Audit Report",
        f"**Dataset Source:** {manager.source_name}",
        f"**Source Identity:** `{manager.source_identity}`",
        f"**Source Dimensions:** {len(manager.source_df):,} rows × {manager.source_df.shape[1]} columns",
        f"**Current Working Dimensions:** {len(manager.working_df):,} rows × {manager.working_df.shape[1]} columns",
        f"**Total Operations Applied:** {len(manager.history)}",
        "",
        "> [!NOTE]",
        "> **Session Disclosure:** This audit report documents data transformations executed interactively",
        "> within the current session. The underlying raw source dataset remains immutable and unmodified.",
        "",
        "## Transformation History Ledger",
    ]
    if not manager.history:
        lines.append("No transformations applied in this session.")
    else:
        for i, step in enumerate(manager.history, 1):
            info = step.get("details", {})
            lines.append(f"### Step {i}: {info.get('operation', step.get('operation_type'))}")
            lines.append(f"- **Parameters:** `{step.get('params')}`")
            if i <= len(manager.audit_records):
                rec = manager.audit_records[i - 1]
                lines.append(f"- **Rows:** {rec['rows_before']:,} → {rec['rows_after']:,} (Δ {rec['rows_delta']:,})")
                lines.append(f"- **Columns Added:** {rec['columns_added'] or 'None'}")
                lines.append(f"- **Columns Removed:** {rec['columns_removed'] or 'None'}")
                lines.append(f"- **Missing Cells Delta:** {rec['missing_cells_delta']:,}")
            lines.append("")

    return "\n".join(lines)


class OperationResult:
    """Represents the outcome of a standalone cleaning operation."""
    def __init__(self, df: pd.DataFrame, success: bool, details: Optional[Dict[str, Any]] = None, error_message: str = ""):
        self.df = df
        self.success = success
        self.details = details or {}
        self.error_message = error_message


def apply_cleaning_operation(df: pd.DataFrame, config: Dict[str, Any]) -> OperationResult:
    """
    Pure functional helper to execute a single transformation operation config on a DataFrame.
    Returns an OperationResult with the resulting DataFrame, success boolean, and error message.
    """
    op_type = config.get("op_type") or config.get("operation_type")
    params = {k: v for k, v in config.items() if k not in ("op_type", "operation_type")}
    manager = WorkingDatasetManager(df, source_identity="temp_standalone")
    try:
        res_df, details = manager._dispatch_operation(df.copy(), op_type, params)
        return OperationResult(df=res_df, success=True, details=details)
    except Exception as e:
        return OperationResult(df=df.copy(), success=False, error_message=str(e))
