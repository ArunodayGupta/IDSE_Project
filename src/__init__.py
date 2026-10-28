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

from .data_explorer import (
    resolve_builtin_netflix_path,
    compute_dataset_identity,
    load_csv_from_file_or_bytes,
    get_missing_mask,
    classify_column_type,
    compute_overview_metrics,
    compute_column_schema,
    compute_missing_rankings,
    get_missingness_matrix,
    compute_numeric_stats,
    compute_categorical_stats,
    compute_datetime_stats,
    preview_datetime_parsing,
    apply_explorer_filters,
    paginate_dataframe,
    export_filtered_csv,
    export_profile_csv,
)
from .explorer_ui import render_explorer_tab
from .cleaning_lab import (
    WorkingDatasetManager,
    generate_data_quality_audit,
    compute_what_changed,
    apply_cleaning_operation,
    export_working_csv,
    export_cleaning_report,
)
from .cleaning_lab_ui import render_cleaning_lab_tab
from .model_lab import (
    detect_suggested_task,
    validate_modeling_inputs,
    build_preprocessor,
    train_and_evaluate_model,
    extract_feature_importances,
    compute_error_analysis,
    RecommenderArenaService,
)
from .model_lab_ui import render_model_lab_tab

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
    "resolve_builtin_netflix_path",
    "compute_dataset_identity",
    "load_csv_from_file_or_bytes",
    "get_missing_mask",
    "classify_column_type",
    "compute_overview_metrics",
    "compute_column_schema",
    "compute_missing_rankings",
    "get_missingness_matrix",
    "compute_numeric_stats",
    "compute_categorical_stats",
    "compute_datetime_stats",
    "preview_datetime_parsing",
    "apply_explorer_filters",
    "paginate_dataframe",
    "export_filtered_csv",
    "export_profile_csv",
    "render_explorer_tab",
    "WorkingDatasetManager",
    "generate_data_quality_audit",
    "compute_what_changed",
    "apply_cleaning_operation",
    "export_working_csv",
    "export_cleaning_report",
    "render_cleaning_lab_tab",
    "detect_suggested_task",
    "validate_modeling_inputs",
    "build_preprocessor",
    "train_and_evaluate_model",
    "extract_feature_importances",
    "compute_error_analysis",
    "RecommenderArenaService",
    "render_model_lab_tab",
]
