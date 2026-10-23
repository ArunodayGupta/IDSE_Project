"""
Netflix Content Strategy & Catalog EDA
Source package containing data generation, cleaning, feature engineering, and visualization modules.
"""

from .data_generator import ensure_data_exists
from .data_cleaner import clean_netflix_data
from .feature_engineer import engineer_features, explode_column, create_metadata_soup
from .model_trainer import ContentRecommender
from .visualizations import (
    plot_catalog_growth,
    plot_acquisition_lag_trend,
    plot_top_countries_audience,
    plot_genre_cooccurrence,
    plot_duration_distributions,
    plot_missing_directors_comparison,
)

__all__ = [
    "ensure_data_exists",
    "clean_netflix_data",
    "engineer_features",
    "explode_column",
    "create_metadata_soup",
    "ContentRecommender",
    "plot_catalog_growth",
    "plot_acquisition_lag_trend",
    "plot_top_countries_audience",
    "plot_genre_cooccurrence",
    "plot_duration_distributions",
    "plot_missing_directors_comparison",
]
