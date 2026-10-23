"""
feature_engineer.py
Business metrics, lag calculations, audience tiering, and multi-label relational exploders.

Transforms cleaned Netflix catalog data into high-value strategic dimensions:
- Acquisition / Licensing Lag (Originals vs. Back-Catalog indicator)
- Content Origin Proxy categories
- Target Audience Maturity tiers
- Multi-label category exploders for genre and country distribution analysis
"""

from typing import List, Optional, Any
import pandas as pd
import numpy as np


# Audience tier mappings based on MPAA and TV Parental Guidelines
AUDIENCE_TIER_MAP = {
    # Kids
    "TV-Y": "Kids",
    "TV-Y7": "Kids",
    "TV-Y7-FV": "Kids",
    "G": "Kids",
    "TV-G": "Kids",
    # Family / Teens
    "PG": "Family / Teens",
    "TV-PG": "Family / Teens",
    "PG-13": "Family / Teens",
    "TV-14": "Family / Teens",
    # Mature / Adults
    "R": "Mature / Adults",
    "TV-MA": "Mature / Adults",
    "NC-17": "Mature / Adults",
    "NR": "Mature / Adults",
    "UR": "Mature / Adults",
}


def create_metadata_soup(df: pd.DataFrame) -> pd.Series:
    """
    Constructs a metadata soup column for NLP and content-based recommendation.

    Concatenates:
    - title: lowercased
    - director: lowercased, spaces removed from names (e.g., 'Tom Hanks' -> 'tomhanks')
    - cast: lowercased, spaces removed from names (e.g., 'Leonardo DiCaprio' -> 'leonardodicaprio')
    - listed_in: lowercased genres
    - description: lowercased synopsis

    Handles NaNs, empty values, and 'unlisted' sentinels cleanly.

    Args:
        df: DataFrame containing catalog features.

    Returns:
        pd.Series containing the concatenated metadata soup strings.
    """
    def _clean_name_tokens(val: Any) -> str:
        if pd.isna(val):
            return ""
        s = str(val).strip()
        if not s or s.lower() in ("unlisted", "unknown", "nan", "none"):
            return ""
        # Split names by comma, strip leading/trailing spaces, remove internal spaces, and lowercase
        names = [n.strip().replace(" ", "").lower() for n in s.split(",") if n.strip()]
        return " ".join(names)

    def _clean_text(val: Any) -> str:
        if pd.isna(val):
            return ""
        s = str(val).strip()
        if not s or s.lower() in ("nan", "none", "unknown"):
            return ""
        return s.lower()

    titles = df["title"].apply(_clean_text) if "title" in df.columns else pd.Series("", index=df.index)
    directors = df["director"].apply(_clean_name_tokens) if "director" in df.columns else pd.Series("", index=df.index)
    casts = df["cast"].apply(_clean_name_tokens) if "cast" in df.columns else pd.Series("", index=df.index)
    genres = df["listed_in"].apply(_clean_text) if "listed_in" in df.columns else pd.Series("", index=df.index)
    descriptions = df["description"].apply(_clean_text) if "description" in df.columns else pd.Series("", index=df.index)

    soup = (
        titles + " " +
        directors + " " +
        casts + " " +
        genres + " " +
        descriptions
    ).str.replace(r"\s+", " ", regex=True).str.strip()

    return soup


def engineer_features(cleaned_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes strategic business features on the cleaned Netflix dataset.

    Features engineered:
    1. acquisition_lag: difference in years between year_added and release_year (clipped at min 0).
    2. content_origin_proxy: categorizes lag into Same-Year (Original/Exclusive), Recent, or Back-Catalog.
    3. audience_tier: maps ratings into Kids, Family / Teens, and Mature / Adults.
    4. primary_genre: first listed genre for quick aggregation.
    5. primary_country: first listed production country.
    6. metadata_soup: lowercased, concatenated NLP soup of title, director, cast, genres, description.

    Args:
        cleaned_df: DataFrame output from data_cleaner.clean_netflix_data.

    Returns:
        pd.DataFrame with newly engineered columns.
    """
    df = cleaned_df.copy()

    # 1. Acquisition Lag (Licensing Lag in years)
    if "year_added" in df.columns and "release_year" in df.columns:
        lag = df["year_added"] - df["release_year"]
        # Handle rare edge cases where release_year > year_added by clipping to 0
        df["acquisition_lag"] = np.clip(lag, 0, None)

        # 2. Content Origin Proxy
        # 0 = Same-Year (High-probability Original or exclusive day-and-date release)
        # 1-3 = Recent theatrical/broadcast release
        # >3 = Studio Back-Catalog / Syndication
        conditions = [
            df["acquisition_lag"] == 0,
            (df["acquisition_lag"] >= 1) & (df["acquisition_lag"] <= 3),
            df["acquisition_lag"] > 3
        ]
        choices = [
            "Same-Year (Original/Exclusive Proxy)",
            "Recent Release (1-3 Yrs)",
            "Back-Catalog (>3 Yrs)"
        ]
        df["content_origin_proxy"] = np.select(conditions, choices, default="Back-Catalog (>3 Yrs)")

    # 3. Audience Maturity Tier
    if "rating" in df.columns:
        df["audience_tier"] = (
            df["rating"]
            .astype(str)
            .str.strip()
            .map(AUDIENCE_TIER_MAP)
            .fillna("Mature / Adults")
        )

    # 4. Primary genre and country helpers
    if "listed_in" in df.columns:
        df["primary_genre"] = (
            df["listed_in"]
            .astype(str)
            .str.split(",")
            .str[0]
            .str.strip()
        )

    if "country" in df.columns:
        df["primary_country"] = (
            df["country"]
            .astype(str)
            .str.split(",")
            .str[0]
            .str.strip()
        )

    # 5. Metadata soup for NLP Content Recommender Engine
    df["metadata_soup"] = create_metadata_soup(df)

    return df


def explode_column(
    df: pd.DataFrame,
    col_name: str,
    drop_values: Optional[List[str]] = None,
    output_col_name: Optional[str] = None
) -> pd.DataFrame:
    """
    Splits comma-separated string columns, strips whitespace, and explodes them
    into individual rows without altering the source dataframe.

    Useful for multi-label columns like 'listed_in' (genres) and 'country'.

    Args:
        df: Input DataFrame.
        col_name: Column name containing comma-separated strings.
        drop_values: List of tokens to exclude (default: ['Unknown', 'Unlisted']).
        output_col_name: Optional new column name. If None, overwrites col_name in the exploded slice.

    Returns:
        New exploded DataFrame with one label per row.
    """
    if col_name not in df.columns:
        raise ValueError(f"Column '{col_name}' not found in DataFrame.")

    if drop_values is None:
        drop_values = ["Unknown", "Unlisted", ""]

    target_col = output_col_name if output_col_name else col_name

    # Create an isolated copy to prevent SettingWithCopyWarning
    sub_df = df.copy()

    # Split on commas with optional following whitespace
    sub_df[target_col] = (
        sub_df[col_name]
        .astype(str)
        .str.split(r",\s*")
    )

    exploded = sub_df.explode(target_col)
    exploded[target_col] = exploded[target_col].astype(str).str.strip()

    # Filter out drop_values and empty strings
    mask = ~exploded[target_col].isin(drop_values) & (exploded[target_col] != "")
    return exploded[mask].copy()


if __name__ == "__main__":
    from data_generator import ensure_data_exists
    from data_cleaner import clean_netflix_data

    csv_path = ensure_data_exists()
    raw = pd.read_csv(csv_path)
    clean, _ = clean_netflix_data(raw)
    feat = engineer_features(clean)

    print("Engineered columns:", [c for c in feat.columns if c not in clean.columns])
    print(feat[["title", "release_year", "year_added", "acquisition_lag", "content_origin_proxy", "audience_tier"]].head(3))

    exploded_genres = explode_column(feat, "listed_in", output_col_name="genre")
    print("\nTop 5 exploded genres:")
    print(exploded_genres["genre"].value_counts().head(5))
