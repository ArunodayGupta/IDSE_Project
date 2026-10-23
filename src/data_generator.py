"""
data_generator.py
Dataset verification module for Netflix Content Strategy EDA.

Verifies that the Kaggle netflix_titles.csv dataset is present in the data/ directory.
"""

import os


def ensure_data_exists(data_path: str = "data/netflix_titles.csv") -> str:
    """
    Checks if netflix_titles.csv exists at the specified path.
    Raises FileNotFoundError if missing, otherwise returns the path.

    Args:
        data_path: Relative or absolute path to netflix_titles.csv.

    Returns:
        str: Validated path to netflix_titles.csv.

    Raises:
        FileNotFoundError: If the dataset is not present.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError("Please place netflix_titles.csv in the data folder.")
    return data_path


if __name__ == "__main__":
    import pandas as pd
    path = ensure_data_exists()
    df_check = pd.read_csv(path)
    print(f"Dataset successfully verified at {path} with {len(df_check):,} rows.")
    print("Columns:", list(df_check.columns))
