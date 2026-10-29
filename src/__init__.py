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
from .cleaning_lab import (
    WorkingDatasetManager,
    generate_data_quality_audit,
    compute_what_changed,
    apply_cleaning_operation,
    export_working_csv,
    export_cleaning_report,
    get_datalab_working_dataset,
)
from .model_lab import (
    detect_suggested_task,
    validate_modeling_inputs,
    build_preprocessor,
    train_and_evaluate_model,
    extract_feature_importances,
    compute_error_analysis,
    RecommenderArenaService,
    compute_model_config_fingerprint,
    compute_relevance_feedback_metrics,
    export_model_experiment_json,
    export_recommender_experiment_json,
)
from .operation_registry import (
    OPERATION_REGISTRY,
    get_operation,
    list_operations_by_category,
)
from .workflow_engine import (
    WorkflowStep,
    WorkflowPipeline,
    create_starter_workflow,
    list_starter_workflows,
)
from .missions import (
    MISSION_REGISTRY,
    get_mission,
    list_missions,
)
from .data_context import (
    get_active_dataset_context,
    resolve_modeling_dataset,
    register_working_dataset_manager,
)

# Lazy loader for UI rendering functions to avoid importing Streamlit when doing pure data operations
_UI_EXPORTS = {
    "render_explorer_tab": ("src.explorer_ui", "render_explorer_tab"),
    "render_cleaning_lab_tab": ("src.cleaning_lab_ui", "render_cleaning_lab_tab"),
    "render_workflow_studio_tab": ("src.workflow_ui", "render_workflow_studio_tab"),
    "render_model_lab_tab": ("src.model_lab_ui", "render_model_lab_tab"),
    "render_missions_tab": ("src.missions_ui", "render_missions_tab"),
}

def __getattr__(name: str):
    if name in _UI_EXPORTS:
        mod_name, func_name = _UI_EXPORTS[name]
        import importlib
        module = importlib.import_module(mod_name)
        return getattr(module, func_name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

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
    "WorkingDatasetManager",
    "generate_data_quality_audit",
    "compute_what_changed",
    "apply_cleaning_operation",
    "export_working_csv",
    "export_cleaning_report",
    "get_datalab_working_dataset",
    "detect_suggested_task",
    "validate_modeling_inputs",
    "build_preprocessor",
    "train_and_evaluate_model",
    "extract_feature_importances",
    "compute_error_analysis",
    "RecommenderArenaService",
    "compute_model_config_fingerprint",
    "compute_relevance_feedback_metrics",
    "export_model_experiment_json",
    "export_recommender_experiment_json",
    "OPERATION_REGISTRY",
    "get_operation",
    "list_operations_by_category",
    "WorkflowStep",
    "WorkflowPipeline",
    "create_starter_workflow",
    "list_starter_workflows",
    "MISSION_REGISTRY",
    "get_mission",
    "list_missions",
    "get_active_dataset_context",
    "resolve_modeling_dataset",
    "register_working_dataset_manager",
    "render_explorer_tab",
    "render_cleaning_lab_tab",
    "render_workflow_studio_tab",
    "render_model_lab_tab",
    "render_missions_tab",
]
