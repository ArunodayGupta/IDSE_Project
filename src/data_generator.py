"""
data_generator.py
Dataset verification module for Netflix Content Strategy EDA.

Verifies that the Kaggle netflix_titles.csv dataset is present in the data/ directory.
"""

import os
from pathlib import Path


def get_project_root() -> Path:
    """Returns the absolute Path to the project root directory."""
    return Path(__file__).resolve().parent.parent


def ensure_data_exists(data_path: str = "data/netflix_titles.csv") -> str:
    """
    Checks if netflix_titles.csv exists at the specified path or standard fallback locations.
    Resolves relative paths robustly against both the current working directory and the project root.
    Raises FileNotFoundError if missing, otherwise returns the validated path.

    Args:
        data_path: Relative or absolute path to netflix_titles.csv.

    Returns:
        str: Validated path to netflix_titles.csv.

    Raises:
        FileNotFoundError: If the dataset is not present.
    """
    # 1. Direct path check (as given or relative to cwd)
    if os.path.exists(data_path):
        return os.path.abspath(data_path)

    # 2. Check relative to project root
    root = get_project_root()
    candidate_root = root / data_path
    if candidate_root.exists():
        return str(candidate_root.resolve())

    # 3. Check data/ directory under project root
    candidate_data = root / "data" / "netflix_titles.csv"
    if candidate_data.exists():
        return str(candidate_data.resolve())

    # 4. Check project root directly
    candidate_direct = root / "netflix_titles.csv"
    if candidate_direct.exists():
        return str(candidate_direct.resolve())

    raise FileNotFoundError(
        f"Dataset not found at '{data_path}'. Please ensure 'netflix_titles.csv' is placed "
        f"in '{root / 'data'}' or the project root."
    )


if __name__ == "__main__":
    import pandas as pd
    path = ensure_data_exists()
    df_check = pd.read_csv(path)
    print(f"Dataset successfully verified at {path} with {len(df_check):,} rows.")
    print("Columns:", list(df_check.columns))
