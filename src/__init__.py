"""
DataLab: Movie Box Office Revenue Prediction
Core source package for data loading, preprocessing, feature engineering, modeling, and visualization.
"""

from .data_loader import (
    load_tmdb_train_data,
    load_tmdb_test_data,
    safe_parse_json,
    parse_release_date,
    resolve_file_path,
    get_dataset_availability,
)

from .feature_engineering import (
    engineer_movie_features,
    prepare_single_movie_features,
)

from .model_trainer import (
    train_movie_regressor,
    predict_single_movie,
    generate_kaggle_submission,
    compute_rmsle,
    MODEL_REGISTRY,
)

from .ui_theme import (
    inject_custom_theme,
    apply_figure_theme,
    render_page_header,
    render_section_header,
    render_kpi_row,
    render_metric_card,
    render_callout,
    render_badge,
    render_empty_state,
)

__all__ = [
    "load_tmdb_train_data",
    "load_tmdb_test_data",
    "safe_parse_json",
    "parse_release_date",
    "resolve_file_path",
    "get_dataset_availability",
    "engineer_movie_features",
    "prepare_single_movie_features",
    "train_movie_regressor",
    "predict_single_movie",
    "generate_kaggle_submission",
    "compute_rmsle",
    "MODEL_REGISTRY",
    "inject_custom_theme",
    "apply_figure_theme",
    "render_page_header",
    "render_section_header",
    "render_kpi_row",
    "render_metric_card",
    "render_callout",
    "render_badge",
    "render_empty_state",
]
