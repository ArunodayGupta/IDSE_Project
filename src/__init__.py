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
    validate_raw_schema,
)

from .model_trainer import (
    train_movie_regressor,
    predict_single_movie,
    generate_kaggle_submission,
    compute_rmsle,
    MODEL_REGISTRY,
)

from .visualizations import (
    plot_revenue_distribution,
    plot_budget_vs_revenue,
    plot_revenue_by_genre,
    plot_revenue_trend_over_time,
    plot_budget_distribution,
    plot_actual_vs_predicted,
    plot_residuals_distribution,
    plot_feature_importance,
    plot_missing_values_summary,
    format_currency_compact,
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
    "validate_raw_schema",
    "train_movie_regressor",
    "predict_single_movie",
    "generate_kaggle_submission",
    "compute_rmsle",
    "MODEL_REGISTRY",
    "plot_revenue_distribution",
    "plot_budget_vs_revenue",
    "plot_revenue_by_genre",
    "plot_revenue_trend_over_time",
    "plot_budget_distribution",
    "plot_actual_vs_predicted",
    "plot_residuals_distribution",
    "plot_feature_importance",
    "plot_missing_values_summary",
    "format_currency_compact",
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
