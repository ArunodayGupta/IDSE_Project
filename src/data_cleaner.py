"""
data_cleaner.py
Data hygiene, missing value handling, and string parsing pipeline for Netflix catalog data.

Processes raw Netflix data, captures before-and-after hygiene metrics,
resolves structural anomalies, and extracts normalized temporal and duration fields.
"""

from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np


def clean_netflix_data(raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Cleans raw Netflix catalog data and generates an audit hygiene report.

    Pipeline operations:
    1. Audits raw null counts and percentages across all columns.
    2. Tests and quantifies the 'Missing Director Hypothesis' (Movies vs. TV Shows).
    3. Drops rows missing critical identifying attributes: date_added, rating, duration.
    4. Imputes unlisted metadata (director -> 'Unlisted', cast -> 'Unlisted', country -> 'Unknown').
    5. Strips and parses date_added into datetime, year_added, month_added, and month_name.
    6. Parses duration into duration_min (for Movies) and seasons_count (for TV Shows).

    Args:
        raw_df: Raw DataFrame matching Kaggle netflix_titles schema.

    Returns:
        Tuple of (cleaned_df, hygiene_report)
    """
    df = raw_df.copy()
    initial_rows = len(df)

    # 1. Audit missing values BEFORE cleaning
    missing_before_series = df.isnull().sum()
    missing_before_pct = (missing_before_series / initial_rows) * 100
    missing_before_df = pd.DataFrame({
        "Column": missing_before_series.index,
        "Missing_Count": missing_before_series.values,
        "Missing_Pct": missing_before_pct.round(2).values
    })

    # 2. Missing Director Hypothesis calculation
    movie_mask = df["type"] == "Movie"
    tv_mask = df["type"] == "TV Show"

    movie_total = int(movie_mask.sum())
    tv_total = int(tv_mask.sum())

    movie_missing_dir = int(df.loc[movie_mask, "director"].isnull().sum())
    tv_missing_dir = int(df.loc[tv_mask, "director"].isnull().sum())

    movie_missing_dir_pct = round((movie_missing_dir / movie_total * 100), 2) if movie_total > 0 else 0.0
    tv_missing_dir_pct = round((tv_missing_dir / tv_total * 100), 2) if tv_total > 0 else 0.0

    director_hypothesis = {
        "movie_total": movie_total,
        "movie_missing_dir": movie_missing_dir,
        "movie_missing_dir_pct": movie_missing_dir_pct,
        "tv_total": tv_total,
        "tv_missing_dir": tv_missing_dir,
        "tv_missing_dir_pct": tv_missing_dir_pct,
        "explanation": (
            "Episodic TV productions employ showrunners, creators, and rotating episodic "
            "directors across seasons. As a consequence, catalog aggregators rarely attribute "
            "a single overarching director to TV series, causing a systemic ~70%+ null rate, "
            "whereas feature films almost universally credit a single auteur director."
        )
    }

    # 3. Drop rows with null date_added, rating, or duration (< 1-2% of dataset)
    critical_cols = ["date_added", "rating", "duration"]
    existing_critical = [c for c in critical_cols if c in df.columns]
    df = df.dropna(subset=existing_critical).copy()

    # 4. Impute categorical metadata
    if "director" in df.columns:
        df["director"] = df["director"].fillna("Unlisted")
    if "cast" in df.columns:
        df["cast"] = df["cast"].fillna("Unlisted")
    if "country" in df.columns:
        df["country"] = df["country"].fillna("Unknown")

    # 5. Date standardization & extraction
    if "date_added" in df.columns:
        # Strip leading/trailing whitespaces
        df["date_added_clean"] = df["date_added"].astype(str).str.strip()
        df["date_added_dt"] = pd.to_datetime(df["date_added_clean"], errors="coerce")

        # In case parsing fails for any malformed dates, drop those rows
        df = df.dropna(subset=["date_added_dt"]).copy()

        df["year_added"] = df["date_added_dt"].dt.year.astype(int)
        df["month_added"] = df["date_added_dt"].dt.month.astype(int)
        df["month_name"] = df["date_added_dt"].dt.strftime("%B")

    # 6. Duration parsing via regex (\d+)
    if "duration" in df.columns:
        duration_digits = df["duration"].astype(str).str.extract(r"(\d+)")[0].astype(float)

        # Populate duration_min exclusively for Movies
        df["duration_min"] = np.where(df["type"] == "Movie", duration_digits, np.nan)

        # Populate seasons_count exclusively for TV Shows
        df["seasons_count"] = np.where(df["type"] == "TV Show", duration_digits, np.nan)

    final_rows = len(df)
    dropped_rows = initial_rows - final_rows

    # 7. Audit missing values AFTER cleaning
    missing_after_series = df.isnull().sum()
    missing_after_pct = (missing_after_series / final_rows) * 100
    missing_after_df = pd.DataFrame({
        "Column": missing_after_series.index,
        "Missing_Count": missing_after_series.values,
        "Missing_Pct": missing_after_pct.round(2).values
    })

    hygiene_report = {
        "initial_rows": initial_rows,
        "final_rows": final_rows,
        "dropped_rows": dropped_rows,
        "dropped_pct": round((dropped_rows / initial_rows * 100), 2) if initial_rows > 0 else 0.0,
        "missing_before_df": missing_before_df,
        "missing_after_df": missing_after_df,
        "director_hypothesis": director_hypothesis
    }

    return df, hygiene_report


if __name__ == "__main__":
    from data_generator import ensure_data_exists
    csv_file = ensure_data_exists()
    raw = pd.read_csv(csv_file)
    cleaned, report = clean_netflix_data(raw)
    print("Cleaned DataFrame shape:", cleaned.shape)
    print("Dropped rows:", report["dropped_rows"])
    print("Director hypothesis:", report["director_hypothesis"])
