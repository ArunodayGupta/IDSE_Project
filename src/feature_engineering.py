"""
feature_engineering.py
Pre-release movie metadata feature engineering and leakage prevention for TMDB Box Office Prediction.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.data_loader import safe_parse_json, parse_release_date


# Major Hollywood / Global Film Studios for Studio Flagging
MAJOR_STUDIOS = [
    "warner bros",
    "universal pictures",
    "paramount",
    "twentieth century fox",
    "20th century fox",
    "columbia pictures",
    "walt disney",
    "marvel",
    "sony pictures",
    "new line cinema",
    "metro-goldwyn-mayer",
    "mgm",
    "tri-star",
    "dreamworks",
    "lionsgate"
]

# Prominent Genre List to Extract as Explicit Binary Indicators
CORE_GENRES = [
    ("genre_drama", "Drama"),
    ("genre_comedy", "Comedy"),
    ("genre_thriller", "Thriller"),
    ("genre_action", "Action"),
    ("genre_romance", "Romance"),
    ("genre_crime", "Crime"),
    ("genre_adventure", "Adventure"),
    ("genre_horror", "Horror"),
    ("genre_scifi", "Science Fiction"),
    ("genre_family", "Family"),
    ("genre_fantasy", "Fantasy"),
    ("genre_animation", "Animation"),
    ("genre_mystery", "Mystery"),
    ("genre_documentary", "Documentary"),
]

# Top languages retained directly; remaining mapped to 'other'
TOP_LANGUAGES = {"en", "fr", "es", "zh", "de", "hi", "ja", "it", "ko", "ru", "cn", "pt"}


def extract_director_name(crew_parsed: List[Dict[str, Any]]) -> str:
    """Extracts the first director name from parsed crew list, or 'Unknown'."""
    if not crew_parsed:
        return "Unknown"
    for member in crew_parsed:
        if isinstance(member, dict) and member.get("job") == "Director":
            name = member.get("name", "").strip()
            if name:
                return name
    return "Unknown"


def engineer_movie_features(
    df: pd.DataFrame,
    is_train: bool = True,
    stats: Optional[Dict[str, Any]] = None
) -> Tuple[pd.DataFrame, Optional[pd.Series], Dict[str, Any]]:
    """
    Transforms raw TMDB metadata into clean pre-release predictive features.
    
    Guarantees:
    1. Zero Target Leakage: Strictly omits 'revenue', 'popularity', 'vote_average',
       'vote_count', and post-release attributes.
    2. Zero Memory Leaks: Extracts compact numerical and low-cardinality features.
    3. Reproducible Imputation: Reuses training set median values for test/prediction.
    
    Returns:
        (X, y, stats) where y is None if target 'revenue' is absent or is_train=False.
    """
    data = df.copy()

    # 1. Target Extraction (if present)
    y: Optional[pd.Series] = None
    if "revenue" in data.columns and is_train:
        y = pd.to_numeric(data["revenue"], errors="coerce")
        valid_mask = y.notna() & (y > 0)
        data = data.loc[valid_mask].copy()
        y = y.loc[valid_mask]

    # Compute or retrieve reference statistics (median budget & runtime)
    if stats is None:
        non_zero_budgets = pd.to_numeric(data["budget"], errors="coerce")
        non_zero_budgets = non_zero_budgets[non_zero_budgets > 0]
        median_budget = float(non_zero_budgets.median()) if len(non_zero_budgets) > 0 else 16000000.0

        valid_runtimes = pd.to_numeric(data["runtime"], errors="coerce")
        valid_runtimes = valid_runtimes[valid_runtimes > 0]
        median_runtime = float(valid_runtimes.median()) if len(valid_runtimes) > 0 else 104.0

        stats = {
            "median_budget": median_budget,
            "median_runtime": median_runtime,
            "budget_p99": float(non_zero_budgets.quantile(0.99)) if len(non_zero_budgets) > 0 else 200000000.0,
            "runtime_min": float(valid_runtimes.min()) if len(valid_runtimes) > 0 else 40.0,
            "runtime_max": float(valid_runtimes.max()) if len(valid_runtimes) > 0 else 240.0,
        }
    else:
        median_budget = stats["median_budget"]
        median_runtime = stats["median_runtime"]

    # 2. Budget Features
    budget_numeric = pd.to_numeric(data["budget"], errors="coerce").fillna(0)
    budget_is_zero = (budget_numeric <= 0).astype(int)
    budget_imputed = np.where(budget_numeric > 0, budget_numeric, median_budget)
    log_budget = np.log1p(budget_imputed)

    # 3. Runtime Features
    runtime_numeric = pd.to_numeric(data["runtime"], errors="coerce")
    runtime_imputed = runtime_numeric.fillna(median_runtime)
    runtime_imputed = np.where(runtime_imputed > 0, runtime_imputed, median_runtime)

    # 4. Collection / Franchise
    has_collection = (
        data["belongs_to_collection"].notna()
        & (data["belongs_to_collection"].astype(str).str.strip() != "")
        & (data["belongs_to_collection"].astype(str).str.strip() != "[]")
    ).astype(int)

    # 5. Release Date Features
    parsed_dates = parse_release_date(data["release_date"])
    # Fallback for missing dates to median year 2000
    release_year = parsed_dates.dt.year.fillna(2000).astype(int)
    release_month = parsed_dates.dt.month.fillna(6).astype(int)
    release_quarter = parsed_dates.dt.quarter.fillna(2).astype(int)
    release_dayofweek = parsed_dates.dt.dayofweek.fillna(4).astype(int)
    release_is_weekend = release_dayofweek.isin([4, 5, 6]).astype(int)

    # 6. JSON Parsing: Genres
    parsed_genres = data["genres"].apply(safe_parse_json)
    genre_count = parsed_genres.apply(len)
    
    def get_primary_genre(g_list: List[Dict[str, Any]]) -> str:
        if g_list and isinstance(g_list[0], dict) and "name" in g_list[0]:
            return str(g_list[0]["name"])
        return "None"
    
    primary_genre = parsed_genres.apply(get_primary_genre)

    genre_flags: Dict[str, pd.Series] = {}
    for col_name, g_target in CORE_GENRES:
        genre_flags[col_name] = parsed_genres.apply(
            lambda glist: int(any(isinstance(g, dict) and g.get("name") == g_target for g in glist))
        )

    # 7. JSON Parsing: Production Companies
    parsed_companies = data["production_companies"].apply(safe_parse_json)
    company_count = parsed_companies.apply(len)

    def check_major_studio(c_list: List[Dict[str, Any]]) -> int:
        for c in c_list:
            if isinstance(c, dict):
                c_name = str(c.get("name", "")).lower()
                if any(studio in c_name for studio in MAJOR_STUDIOS):
                    return 1
        return 0

    is_major_studio = parsed_companies.apply(check_major_studio)

    # 8. JSON Parsing: Production Countries
    parsed_countries = data["production_countries"].apply(safe_parse_json)
    country_count = parsed_countries.apply(len)

    def get_primary_country(c_list: List[Dict[str, Any]]) -> str:
        if c_list and isinstance(c_list[0], dict) and "iso_3166_1" in c_list[0]:
            return str(c_list[0]["iso_3166_1"]).upper()
        return "OTHER"

    primary_country = parsed_countries.apply(get_primary_country)
    is_us_country = (primary_country == "US").astype(int)

    # 9. Spoken Languages
    parsed_languages = data["spoken_languages"].apply(safe_parse_json)
    language_count = parsed_languages.apply(len)

    # 10. Original Language
    raw_lang = data["original_language"].fillna("other").astype(str).str.lower()
    clean_lang = raw_lang.apply(lambda l: l if l in TOP_LANGUAGES else "other")
    is_english = (clean_lang == "en").astype(int)

    # 11. Cast & Crew Sizes
    parsed_cast = data["cast"].apply(safe_parse_json)
    cast_size = parsed_cast.apply(len)

    parsed_crew = data["crew"].apply(safe_parse_json)
    crew_size = parsed_crew.apply(len)

    # 12. Marketing / Pre-Release Artifact Indicators
    has_homepage = data["homepage"].notna().astype(int) if "homepage" in data.columns else pd.Series(0, index=data.index)
    has_tagline = data["tagline"].notna().astype(int) if "tagline" in data.columns else pd.Series(0, index=data.index)

    # Construct the final feature DataFrame
    features = pd.DataFrame({
        "budget": budget_imputed,
        "log_budget": log_budget,
        "budget_is_zero": budget_is_zero,
        "runtime": runtime_imputed,
        "has_collection": has_collection,
        "release_year": release_year,
        "release_month": release_month,
        "release_quarter": release_quarter,
        "release_dayofweek": release_dayofweek,
        "release_is_weekend": release_is_weekend,
        "genre_count": genre_count,
        "company_count": company_count,
        "is_major_studio": is_major_studio,
        "country_count": country_count,
        "is_us_country": is_us_country,
        "language_count": language_count,
        "is_english": is_english,
        "cast_size": cast_size,
        "crew_size": crew_size,
        "has_homepage": has_homepage,
        "has_tagline": has_tagline,
        "original_language": clean_lang,
        "primary_genre": primary_genre,
    }, index=data.index)

    # Attach explicit genre flags
    for col_name, series in genre_flags.items():
        features[col_name] = series

    return features, y, stats


def prepare_single_movie_features(
    movie_dict: Dict[str, Any],
    stats: Dict[str, Any]
) -> pd.DataFrame:
    """
    Transforms interactive prediction form user input into a single-row DataFrame
    matching the exact feature structure and preprocessing contract.
    """
    budget = float(movie_dict.get("budget", 0.0))
    median_budget = stats.get("median_budget", 16000000.0)
    budget_is_zero = 1 if budget <= 0 else 0
    budget_imputed = budget if budget > 0 else median_budget
    log_budget = np.log1p(budget_imputed)

    runtime = float(movie_dict.get("runtime", 100.0))
    median_runtime = stats.get("median_runtime", 104.0)
    runtime_imputed = runtime if runtime > 0 else median_runtime

    has_collection = 1 if movie_dict.get("belongs_to_collection", False) else 0

    rel_date = movie_dict.get("release_date")
    if hasattr(rel_date, "year"):
        release_year = rel_date.year
        release_month = rel_date.month
        release_quarter = (rel_date.month - 1) // 3 + 1
        release_dayofweek = rel_date.weekday()
    else:
        release_year = 2024
        release_month = 6
        release_quarter = 2
        release_dayofweek = 4
    release_is_weekend = 1 if release_dayofweek in [4, 5, 6] else 0

    selected_genres = [g.strip() for g in movie_dict.get("genres", [])]
    genre_count = len(selected_genres)
    primary_genre = selected_genres[0] if selected_genres else "None"

    genre_flags: Dict[str, int] = {}
    for col_name, g_target in CORE_GENRES:
        genre_flags[col_name] = 1 if g_target in selected_genres else 0

    company_count = int(movie_dict.get("company_count", 2))
    is_major = 1 if movie_dict.get("is_major_studio", False) else 0

    primary_country = str(movie_dict.get("production_country", "US")).upper()
    country_count = int(movie_dict.get("country_count", 1))
    is_us_country = 1 if primary_country in ("US", "UNITED STATES") else 0

    lang = str(movie_dict.get("original_language", "en")).lower()
    clean_lang = lang if lang in TOP_LANGUAGES else "other"
    is_english = 1 if clean_lang == "en" else 0
    language_count = int(movie_dict.get("language_count", 1))

    cast_size = int(movie_dict.get("cast_size", 15))
    crew_size = int(movie_dict.get("crew_size", 20))

    has_homepage = 1 if movie_dict.get("has_homepage", False) else 0
    has_tagline = 1 if movie_dict.get("has_tagline", False) else 0

    row_data = {
        "budget": budget_imputed,
        "log_budget": log_budget,
        "budget_is_zero": budget_is_zero,
        "runtime": runtime_imputed,
        "has_collection": has_collection,
        "release_year": release_year,
        "release_month": release_month,
        "release_quarter": release_quarter,
        "release_dayofweek": release_dayofweek,
        "release_is_weekend": release_is_weekend,
        "genre_count": genre_count,
        "company_count": company_count,
        "is_major_studio": is_major,
        "country_count": country_count,
        "is_us_country": is_us_country,
        "language_count": language_count,
        "is_english": is_english,
        "cast_size": cast_size,
        "crew_size": crew_size,
        "has_homepage": has_homepage,
        "has_tagline": has_tagline,
        "original_language": clean_lang,
        "primary_genre": primary_genre,
    }

    for col_name, val in genre_flags.items():
        row_data[col_name] = val

    return pd.DataFrame([row_data])
