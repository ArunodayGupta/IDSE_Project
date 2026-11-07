"""
feature_engineering.py
Pre-release movie metadata feature engineering and leakage prevention for TMDB Box Office Prediction.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.data_loader import safe_parse_json, parse_release_date


# Mandatory raw columns required for feature extraction
MANDATORY_RAW_COLUMNS = ["budget", "genres", "release_date", "runtime"]

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

# Canonical list of engineered feature column names in deterministic order
BASE_FEATURE_COLUMNS = [
    "budget",
    "log_budget",
    "budget_is_zero",
    "runtime",
    "has_collection",
    "release_year",
    "release_month",
    "release_quarter",
    "release_dayofweek",
    "release_is_weekend",
    "genre_count",
    "company_count",
    "is_major_studio",
    "country_count",
    "is_us_country",
    "language_count",
    "is_english",
    "cast_size",
    "crew_size",
    "has_homepage",
    "has_tagline",
    "original_language",
    "primary_genre",
]

ALL_FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + [col_name for col_name, _ in CORE_GENRES]


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


def validate_raw_schema(df: pd.DataFrame) -> None:
    """
    Validates that mandatory raw columns exist in the DataFrame.
    Raises ValueError with a clear, informative message if any mandatory column is missing.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("Input data must be a valid pandas DataFrame.")

    missing = [c for c in MANDATORY_RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input DataFrame is missing mandatory raw column(s): {missing}. "
            f"Required columns for movie revenue prediction: {MANDATORY_RAW_COLUMNS}."
        )


def compute_training_imputation_stats(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculates reference preprocessing and imputation statistics strictly from the training partition.
    Validation and test records must never influence these statistics.
    """
    # Budget statistics from strictly positive reported budgets
    non_zero_budgets = pd.to_numeric(data["budget"], errors="coerce")
    non_zero_budgets = non_zero_budgets[non_zero_budgets > 0]
    median_budget = float(non_zero_budgets.median()) if len(non_zero_budgets) > 0 else 16000000.0

    # Runtime statistics from strictly positive runtimes
    valid_runtimes = pd.to_numeric(data["runtime"], errors="coerce")
    valid_runtimes = valid_runtimes[valid_runtimes > 0]
    median_runtime = float(valid_runtimes.median()) if len(valid_runtimes) > 0 else 104.0

    # Release date statistics
    parsed_dates = parse_release_date(data["release_date"])
    valid_years = parsed_dates.dt.year.dropna()
    valid_months = parsed_dates.dt.month.dropna()
    valid_dows = parsed_dates.dt.dayofweek.dropna()

    median_year = int(valid_years.median()) if len(valid_years) > 0 else 2004
    median_month = int(valid_months.median()) if len(valid_months) > 0 else 6
    median_dow = int(valid_dows.median()) if len(valid_dows) > 0 else 4

    return {
        "median_budget": median_budget,
        "median_runtime": median_runtime,
        "budget_p99": float(non_zero_budgets.quantile(0.99)) if len(non_zero_budgets) > 0 else 200000000.0,
        "runtime_min": float(valid_runtimes.min()) if len(valid_runtimes) > 0 else 40.0,
        "runtime_max": float(valid_runtimes.max()) if len(valid_runtimes) > 0 else 240.0,
        "median_release_year": median_year,
        "median_release_month": median_month,
        "median_release_dayofweek": median_dow,
    }


def engineer_movie_features(
    df: pd.DataFrame,
    is_train: bool = True,
    stats: Optional[Dict[str, Any]] = None
) -> Tuple[pd.DataFrame, Optional[pd.Series], Dict[str, Any]]:
    """
    Transforms raw TMDB metadata into clean pre-release predictive features.
    
    Guarantees:
    1. Zero Target Leakage: Strictly omits 'revenue', 'popularity', 'vote_average',
       'vote_count', and post-release attributes from X.
    2. Zero Validation Leakage: When stats is provided, reuses existing training statistics.
       When stats is None and is_train=True, calculates statistics solely on training data.
    3. Missing-Value Handling: Treats blank strings, whitespace, and NaNs safely.
    4. Target Treatment Rationale: In TMDB box office prediction, legitimate theatrical
       revenues are positive (> 0). Unrecorded or non-positive revenues in training data
       cannot be log1p-transformed meaningfully and are excluded from training rows.
    
    Returns:
        (X, y, stats) where y is None if target 'revenue' is absent or is_train=False.
    """
    validate_raw_schema(df)
    data = df.copy()

    # 1. Target Extraction (if present and training partition)
    y: Optional[pd.Series] = None
    if "revenue" in data.columns and is_train:
        y_raw = pd.to_numeric(data["revenue"], errors="coerce")
        # Target Rationale: In TMDB, true box office returns are strictly positive.
        # Zero or negative revenues represent unrecorded/missing data.
        valid_mask = y_raw.notna() & (y_raw > 0)
        data = data.loc[valid_mask].copy()
        y = y_raw.loc[valid_mask]

    # 2. Imputation Statistics (computed on training data only or reused from stats)
    if stats is None:
        stats = compute_training_imputation_stats(data)

    median_budget = stats["median_budget"]
    median_runtime = stats["median_runtime"]
    median_year = stats.get("median_release_year", 2004)
    median_month = stats.get("median_release_month", 6)
    median_dow = stats.get("median_release_dayofweek", 4)

    # 3. Budget Features
    budget_numeric = pd.to_numeric(data["budget"], errors="coerce").fillna(0)
    budget_is_zero = (budget_numeric <= 0).astype(int)
    budget_imputed = np.where(budget_numeric > 0, budget_numeric, median_budget)
    log_budget = np.log1p(budget_imputed)

    # 4. Runtime Features
    runtime_numeric = pd.to_numeric(data["runtime"], errors="coerce")
    runtime_imputed = np.where(runtime_numeric.notna() & (runtime_numeric > 0), runtime_numeric, median_runtime)

    # 5. Collection / Franchise
    if "belongs_to_collection" in data.columns:
        has_collection = (
            data["belongs_to_collection"].notna()
            & (data["belongs_to_collection"].astype(str).str.strip() != "")
            & (data["belongs_to_collection"].astype(str).str.strip() != "[]")
            & (data["belongs_to_collection"].astype(str).str.strip() != "{}")
        ).astype(int)
    else:
        has_collection = pd.Series(0, index=data.index, dtype=int)

    # 6. Release Date Features
    parsed_dates = parse_release_date(data["release_date"])
    release_year = parsed_dates.dt.year.fillna(median_year).astype(int)
    release_month = parsed_dates.dt.month.fillna(median_month).astype(int)
    release_quarter = ((release_month - 1) // 3 + 1).astype(int)
    release_dayofweek = parsed_dates.dt.dayofweek.fillna(median_dow).astype(int)
    release_is_weekend = release_dayofweek.isin([4, 5, 6]).astype(int)

    # 7. JSON Parsing: Genres
    parsed_genres = data["genres"].apply(safe_parse_json)
    genre_count = parsed_genres.apply(len).astype(int)

    def get_primary_genre(g_list: List[Dict[str, Any]]) -> str:
        if g_list and isinstance(g_list[0], dict) and "name" in g_list[0]:
            name = str(g_list[0]["name"]).strip()
            return name if name else "None"
        return "None"

    primary_genre = parsed_genres.apply(get_primary_genre)

    genre_flags: Dict[str, pd.Series] = {}
    for col_name, g_target in CORE_GENRES:
        genre_flags[col_name] = parsed_genres.apply(
            lambda glist: int(any(isinstance(g, dict) and g.get("name") == g_target for g in glist))
        )

    # 8. JSON Parsing: Production Companies
    if "production_companies" in data.columns:
        parsed_companies = data["production_companies"].apply(safe_parse_json)
        company_count = parsed_companies.apply(len).astype(int)

        def check_major_studio(c_list: List[Dict[str, Any]]) -> int:
            for c in c_list:
                if isinstance(c, dict):
                    c_name = str(c.get("name", "")).lower()
                    if any(studio in c_name for studio in MAJOR_STUDIOS):
                        return 1
            return 0

        is_major_studio = parsed_companies.apply(check_major_studio).astype(int)
    else:
        company_count = pd.Series(0, index=data.index, dtype=int)
        is_major_studio = pd.Series(0, index=data.index, dtype=int)

    # 9. JSON Parsing: Production Countries
    if "production_countries" in data.columns:
        parsed_countries = data["production_countries"].apply(safe_parse_json)
        country_count = parsed_countries.apply(len).astype(int)

        def get_primary_country(c_list: List[Dict[str, Any]]) -> str:
            if c_list and isinstance(c_list[0], dict) and "iso_3166_1" in c_list[0]:
                code = str(c_list[0]["iso_3166_1"]).strip().upper()
                return code if code else "OTHER"
            return "OTHER"

        primary_country = parsed_countries.apply(get_primary_country)
        is_us_country = (primary_country == "US").astype(int)
    else:
        country_count = pd.Series(0, index=data.index, dtype=int)
        is_us_country = pd.Series(0, index=data.index, dtype=int)

    # 10. Spoken Languages
    if "spoken_languages" in data.columns:
        parsed_languages = data["spoken_languages"].apply(safe_parse_json)
        language_count = parsed_languages.apply(len).astype(int)
    else:
        language_count = pd.Series(0, index=data.index, dtype=int)

    # 11. Original Language
    if "original_language" in data.columns:
        raw_lang = data["original_language"].fillna("other").astype(str).str.strip().str.lower()
        clean_lang = raw_lang.apply(lambda l: l if l in TOP_LANGUAGES else "other")
        is_english = (clean_lang == "en").astype(int)
    else:
        clean_lang = pd.Series("other", index=data.index)
        is_english = pd.Series(0, index=data.index, dtype=int)

    # 12. Cast & Crew Sizes
    if "cast" in data.columns:
        parsed_cast = data["cast"].apply(safe_parse_json)
        cast_size = parsed_cast.apply(len).astype(int)
    else:
        cast_size = pd.Series(0, index=data.index, dtype=int)

    if "crew" in data.columns:
        parsed_crew = data["crew"].apply(safe_parse_json)
        crew_size = parsed_crew.apply(len).astype(int)
    else:
        crew_size = pd.Series(0, index=data.index, dtype=int)

    # 13. Pre-Release Marketing / Metadata Indicators (Treat blanks as missing)
    if "homepage" in data.columns:
        has_homepage = (
            data["homepage"].notna()
            & (data["homepage"].astype(str).str.strip() != "")
            & (data["homepage"].astype(str).str.strip().str.lower() != "nan")
        ).astype(int)
    else:
        has_homepage = pd.Series(0, index=data.index, dtype=int)

    if "tagline" in data.columns:
        has_tagline = (
            data["tagline"].notna()
            & (data["tagline"].astype(str).str.strip() != "")
            & (data["tagline"].astype(str).str.strip().str.lower() != "nan")
        ).astype(int)
    else:
        has_tagline = pd.Series(0, index=data.index, dtype=int)

    # Construct feature matrix using canonical column definitions
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

    # Attach binary genre flags
    for col_name, series in genre_flags.items():
        features[col_name] = series

    # Guarantee canonical column order
    features = features[ALL_FEATURE_COLUMNS]

    return features, y, stats


def prepare_single_movie_features(
    movie_dict: Dict[str, Any],
    stats: Dict[str, Any]
) -> pd.DataFrame:
    """
    Transforms interactive prediction form user input into a single-row DataFrame
    matching the exact feature structure, column names, column order, and preprocessing rules
    of engineer_movie_features().
    """
    median_budget = stats.get("median_budget", 16000000.0)
    median_runtime = stats.get("median_runtime", 104.0)
    median_year = stats.get("median_release_year", 2004)
    median_month = stats.get("median_release_month", 6)
    median_dow = stats.get("median_release_dayofweek", 4)

    # Budget
    budget_raw = float(movie_dict.get("budget", 0.0) or 0.0)
    budget_is_zero = 1 if budget_raw <= 0 else 0
    budget_imputed = budget_raw if budget_raw > 0 else median_budget
    log_budget = float(np.log1p(budget_imputed))

    # Runtime
    runtime_raw = float(movie_dict.get("runtime", 0.0) or 0.0)
    runtime_imputed = runtime_raw if runtime_raw > 0 else median_runtime

    # Collection
    has_collection = 1 if bool(movie_dict.get("belongs_to_collection", False)) else 0

    # Release Date
    rel_date = movie_dict.get("release_date")
    if rel_date is not None and hasattr(rel_date, "year"):
        release_year = int(rel_date.year)
        release_month = int(rel_date.month)
        release_quarter = int((rel_date.month - 1) // 3 + 1)
        release_dayofweek = int(rel_date.weekday())
    elif rel_date is not None and isinstance(rel_date, str) and rel_date.strip():
        parsed = parse_release_date(pd.Series([rel_date]))
        if pd.notna(parsed.iloc[0]):
            dt = parsed.iloc[0]
            release_year = int(dt.year)
            release_month = int(dt.month)
            release_quarter = int((dt.month - 1) // 3 + 1)
            release_dayofweek = int(dt.dayofweek)
        else:
            release_year = median_year
            release_month = median_month
            release_quarter = int((median_month - 1) // 3 + 1)
            release_dayofweek = median_dow
    else:
        # Documented consistent fallback using training partition statistics
        release_year = median_year
        release_month = median_month
        release_quarter = int((median_month - 1) // 3 + 1)
        release_dayofweek = median_dow

    release_is_weekend = 1 if release_dayofweek in (4, 5, 6) else 0

    # Genres
    raw_genres = movie_dict.get("genres", [])
    if isinstance(raw_genres, str):
        raw_genres = [g.strip() for g in raw_genres.split(",") if g.strip()]
    selected_genres = [str(g).strip() for g in raw_genres if str(g).strip()]
    genre_count = len(selected_genres)
    primary_genre = selected_genres[0] if selected_genres else "None"

    genre_flags: Dict[str, int] = {}
    for col_name, g_target in CORE_GENRES:
        genre_flags[col_name] = 1 if g_target in selected_genres else 0

    # Companies
    company_count = int(movie_dict.get("company_count", 1) or 1)
    is_major = 1 if bool(movie_dict.get("is_major_studio", False)) else 0

    # Countries
    primary_country = str(movie_dict.get("production_country", "US") or "OTHER").strip().upper()
    country_count = int(movie_dict.get("country_count", 1) or 1)
    is_us_country = 1 if primary_country == "US" else 0

    # Languages
    raw_lang = str(movie_dict.get("original_language", "en") or "other").strip().lower()
    clean_lang = raw_lang if raw_lang in TOP_LANGUAGES else "other"
    is_english = 1 if clean_lang == "en" else 0
    language_count = int(movie_dict.get("language_count", 1) or 1)

    # Cast & Crew
    cast_size = int(movie_dict.get("cast_size", 15) or 15)
    crew_size = int(movie_dict.get("crew_size", 20) or 20)

    # Marketing indicators
    has_homepage = 1 if bool(movie_dict.get("has_homepage", False)) else 0
    has_tagline = 1 if bool(movie_dict.get("has_tagline", False)) else 0

    row_data: Dict[str, Any] = {
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

    single_df = pd.DataFrame([row_data])
    return single_df[ALL_FEATURE_COLUMNS]
