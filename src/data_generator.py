"""
data_generator.py
Synthetic fallback dataset engine for Netflix Content Strategy EDA.

Generates a realistic 2,000-row dataset mimicking the exact schema, data anomalies,
and business patterns of the Kaggle Netflix Movies and TV Shows dataset.
Ensures zero FileNotFoundError exceptions and 100% schema parity with the real CSV.
"""

import os
import random
from datetime import datetime, timedelta
from typing import Optional, Tuple
import numpy as np
import pandas as pd

# Pools of realistic metadata for synthetic generation
SAMPLE_DIRECTORS = [
    "Martin Scorsese", "Christopher Nolan", "Ava DuVernay", "Bong Joon-ho",
    "Quentin Tarantino", "Guillermo del Toro", "David Fincher", "Greta Gerwig",
    "Spike Lee", "Denis Villeneuve", "Alfonso Cuarón", "Steven Spielberg",
    "Ridley Scott", "Taika Waititi", "Jordan Peele", "Chloe Zhao",
    "Anurag Kashyap", "Zoya Akhtar", "S.S. Rajamouli", "Park Chan-wook",
    "Luc Besson", "Pedro Almodóvar", "Guillermo del Toro", "Hirokazu Kore-eda",
    "Hayao Miyazaki", "Damien Chazelle", "Sam Mendes", "Sofia Coppola"
]

SAMPLE_ACTORS = [
    "Leonardo DiCaprio", "Meryl Streep", "Denzel Washington", "Viola Davis",
    "Shah Rukh Khan", "Deepika Padukone", "Song Kang-ho", "Bae Doona",
    "Pedro Pascal", "Millie Bobby Brown", "Bryan Cranston", "Olivia Colman",
    "Idris Elba", "Cillian Murphy", "Florence Pugh", "Mahershala Ali",
    "Christian Bale", "Scarlett Johansson", "Benedict Cumberbatch", "Dev Patel",
    "Priyanka Chopra", "Gong Yoo", "Anya Taylor-Joy", "Adam Driver"
]

SAMPLE_COUNTRIES = [
    "United States", "India", "United Kingdom", "South Korea", "Japan",
    "Canada", "France", "Germany", "Spain", "Mexico", "Australia",
    "Nigeria", "Brazil", "United States, United Kingdom",
    "United Kingdom, United States", "United States, Canada",
    "France, Belgium", "India, United States", "South Korea, United States"
]

MOVIE_GENRES = [
    "Dramas", "Comedies", "Action & Adventure", "Thrillers",
    "International Movies", "Documentaries", "Horror Movies",
    "Romantic Movies", "Sci-Fi & Fantasy", "Children & Family Movies",
    "Independent Movies", "Classic Movies", "Music & Musicals"
]

TV_GENRES = [
    "International TV Shows", "TV Dramas", "TV Comedies", "Crime TV Shows",
    "Romantic TV Shows", "Docuseries", "TV Action & Adventure",
    "Korean TV Shows", "Anime Series", "Kids' TV", "Reality TV",
    "British TV Shows", "Spanish-Language TV Shows"
]

RATINGS = [
    "TV-MA", "TV-14", "TV-PG", "R", "PG-13", "TV-Y", "TV-Y7", "PG", "G"
]
RATING_WEIGHTS = [0.38, 0.25, 0.12, 0.08, 0.06, 0.04, 0.03, 0.02, 0.02]

TITLE_PREFIXES = [
    "The Secret of", "Chronicles of", "Love in", "Beyond the", "Shadows over",
    "Midnight in", "Operation", "Tales from", "The Lost", "Return to",
    "Echoes of", "Last Call at", "Project", "Empire of", "Whispers in"
]

TITLE_NOUNS = [
    "Tokyo", "Brooklyn", "Seoul", "Mumbai", "London", "Horizon", "Paradise",
    "Silence", "Destiny", "Avalon", "Winter", "Summer", "Darkness", "Galaxy",
    "Heart", "Tomorrow", "Courage", "Kingdom", "Night", "Memory", "Labyrinth"
]

DESCRIPTIONS = [
    "A gripping tale of intrigue, betrayal, and redemption as old secrets resurface.",
    "When an unexpected crisis hits, an unlikely group must unite against impossible odds.",
    "An intimate exploration of love, family dynamics, and cultural heritage.",
    "A relentless detective pursues a master criminal playing a dangerous game of cat-and-mouse.",
    "Chronicles the lives, heartbreaks, and triumphs of young strivers chasing their dreams.",
    "A deep-dive investigative look into global events that transformed modern society.",
    "Stranded in an unfamiliar terrain, a crew confronts peril and existential challenges.",
    "A lighthearted romantic comedy where accidental encounters lead to life-altering romance.",
    "In a dystopian tomorrow, courageous rebels risk everything for truth and freedom."
]


def generate_synthetic_netflix_data(
    output_path: str = "data/netflix_titles.csv",
    n_samples: int = 2000,
    seed: int = 42,
    force: bool = False
) -> pd.DataFrame:
    """
    Generates a realistic synthetic Netflix dataset mirroring the exact schema,
    null distributions, date conventions, and duration formats of Kaggle's netflix_titles.csv.

    Args:
        output_path: Path where the CSV should be saved.
        n_samples: Number of rows to generate (default 2,000).
        seed: Random seed for deterministic generation.
        force: If True, regenerates even if file exists.

    Returns:
        pd.DataFrame containing the generated dataset.
    """
    if os.path.exists(output_path) and not force:
        return pd.read_csv(output_path)

    random.seed(seed)
    np.random.seed(seed)

    records = []
    base_date = datetime(2012, 1, 1)
    end_date = datetime(2024, 6, 30)
    total_days = (end_date - base_date).days

    for i in range(1, n_samples + 1):
        show_id = f"s{i}"

        # Strategic Shift: Before 2017, Netflix was heavily movie-dominated (~82%).
        # After 2017, TV Shows surged up to ~45% of additions.
        day_offset = random.randint(0, total_days)
        added_dt = base_date + timedelta(days=day_offset)
        added_year = added_dt.year

        tv_prob = 0.18 if added_year <= 2015 else (0.32 if added_year <= 2019 else 0.44)
        content_type = "TV Show" if random.random() < tv_prob else "Movie"

        # Title generation
        title = f"{random.choice(TITLE_PREFIXES)} {random.choice(TITLE_NOUNS)}"
        if random.random() < 0.2:
            title += f": Chapter {random.randint(1, 4)}"

        # Missing Director Hypothesis:
        # Real dataset bias: TV shows are rarely attributed to a single director (~75% null)
        # Movies usually have a listed director (~12% null)
        if content_type == "TV Show":
            director = np.nan if random.random() < 0.74 else random.choice(SAMPLE_DIRECTORS)
        else:
            director = np.nan if random.random() < 0.12 else random.choice(SAMPLE_DIRECTORS)

        # Cast with ~10% null
        if random.random() < 0.10:
            cast = np.nan
        else:
            cast_count = random.randint(1, 4)
            cast = ", ".join(random.sample(SAMPLE_ACTORS, cast_count))

        # Country with ~8% null
        if random.random() < 0.08:
            country = np.nan
        else:
            country = random.choice(SAMPLE_COUNTRIES)

        # date_added: formatted like "September 25, 2021" with ~1% null and occasional leading spaces
        if random.random() < 0.01:
            date_added_str = np.nan
        else:
            formatted_date = added_dt.strftime("%B %d, %Y")
            if random.random() < 0.05:
                formatted_date = "  " + formatted_date  # deliberate whitespace anomaly
            date_added_str = formatted_date

        # release_year:
        # Licensing Lag logic: In earlier years, back-catalog (lag 3-20 yrs) dominated.
        # In recent years, Originals / Same-Year releases (lag 0-1 yr) dominate.
        if added_year <= 2016:
            lag_weights = [0.20, 0.25, 0.55]  # same-year, 1-3 yrs, >3 yrs
        else:
            lag_weights = [0.52, 0.30, 0.18]

        chosen_lag_bucket = np.random.choice(["same", "recent", "catalog"], p=lag_weights)
        if chosen_lag_bucket == "same":
            lag = 0
        elif chosen_lag_bucket == "recent":
            lag = random.randint(1, 3)
        else:
            lag = random.randint(4, min(25, max(4, added_year - 1980)))

        release_year = max(1980, added_year - lag)

        # Rating with weighted distribution
        rating = np.random.choice(RATINGS, p=RATING_WEIGHTS)
        if random.random() < 0.005:
            rating = np.nan  # rare missing rating

        # Duration logic
        if content_type == "Movie":
            # Realistic runtime: bell curve centered around 98 min
            min_duration = int(np.clip(np.random.normal(102, 22), 65, 210))
            duration = f"{min_duration} min"
        else:
            # Steep drop-off: most shows only get 1 or 2 seasons
            season_weights = [0.65, 0.20, 0.08, 0.04, 0.015, 0.008, 0.004, 0.003]
            seasons = int(np.random.choice(range(1, 9), p=season_weights))
            duration = f"{seasons} Season" if seasons == 1 else f"{seasons} Seasons"

        # Genres (listed_in): 1 to 3 comma-separated genres
        genre_pool = TV_GENRES if content_type == "TV Show" else MOVIE_GENRES
        genre_count = random.randint(1, 3)
        listed_in = ", ".join(random.sample(genre_pool, min(genre_count, len(genre_pool))))

        # Description
        description = random.choice(DESCRIPTIONS)

        records.append({
            "show_id": show_id,
            "type": content_type,
            "title": title,
            "director": director,
            "cast": cast,
            "country": country,
            "date_added": date_added_str,
            "release_year": release_year,
            "rating": rating,
            "duration": duration,
            "listed_in": listed_in,
            "description": description
        })

    df = pd.DataFrame(records)

    # Ensure parent directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def ensure_data_exists(data_path: str = "data/netflix_titles.csv") -> str:
    """
    Checks if a real or synthetic dataset is present at data_path.
    If absent, triggers synthetic generation and returns the guaranteed file path.
    """
    if not os.path.exists(data_path):
        os.makedirs(os.path.dirname(os.path.abspath(data_path)), exist_ok=True)
        generate_synthetic_netflix_data(output_path=data_path, n_samples=2000)
    return data_path


if __name__ == "__main__":
    path = ensure_data_exists()
    df_check = pd.read_csv(path)
    print(f"Dataset successfully verified at {path} with {len(df_check)} rows.")
    print("Columns:", list(df_check.columns))
    print(df_check.head(2))
