"""
operation_registry.py
Single canonical registry of all supported DataLab operations across
Clean & Transform (Phase 2), Workflow Studio (Phase 4), and Missions.

Each operation specification declares:
- op_type: Stable unique identifier
- name: Human-readable display label
- category: Grouping ("Cleaning", "Transformation", "Feature Engineering", "Domain Recipes", "Modeling", "Recommender")
- description: Concise operational summary
- explanation: Practical data science pedagogical explanation / trade-off
- params_schema: Expected parameters with types, defaults, and requirements
- mutates_data: Whether applying this changes the working DataFrame
- is_ml_step: Whether this operation trains or evaluates a model
- handler: Functional execution dispatch
"""

from typing import Dict, Any, List, Optional


class OperationSpec:
    """Specification and metadata for an allowlisted DataLab operation."""

    def __init__(
        self,
        op_type: str,
        name: str,
        category: str,
        description: str,
        explanation: str,
        params_schema: Dict[str, Any],
        mutates_data: bool = True,
        is_ml_step: bool = False,
        required_input_cols: Optional[List[str]] = None,
        output_col_param: Optional[str] = None
    ):
        self.op_type = op_type
        self.name = name
        self.category = category
        self.description = description
        self.explanation = explanation
        self.params_schema = params_schema
        self.mutates_data = mutates_data
        self.is_ml_step = is_ml_step
        self.required_input_cols = required_input_cols or []
        self.output_col_param = output_col_param

    def to_dict(self) -> Dict[str, Any]:
        return {
            "op_type": self.op_type,
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "explanation": self.explanation,
            "params_schema": self.params_schema,
            "mutates_data": self.mutates_data,
            "is_ml_step": self.is_ml_step,
            "required_input_cols": self.required_input_cols,
            "output_col_param": self.output_col_param,
        }


# Global Registry Dict: op_type -> OperationSpec
OPERATION_REGISTRY: Dict[str, OperationSpec] = {}


def register_operation(spec: OperationSpec) -> None:
    """Registers an operation specification in the global registry."""
    OPERATION_REGISTRY[spec.op_type] = spec


# ---------------------------------------------------------------------
# CLEANING OPERATIONS
# ---------------------------------------------------------------------
register_operation(OperationSpec(
    op_type="drop_missing_rows",
    name="Drop Missing Rows",
    category="Cleaning",
    description="Removes rows containing missing values across specified columns or all columns.",
    explanation="Dropping missing rows reduces sample size and can introduce survival bias if data is not Missing Completely At Random (MCAR). Use selectively.",
    params_schema={
        "columns": {"type": "list", "default": [], "label": "Subset Columns (empty = all)"},
        "how": {"type": "select", "options": ["any", "all"], "default": "any", "label": "Drop Condition"},
        "treat_empty_as_missing": {"type": "bool", "default": True, "label": "Treat Empty Strings as Missing"}
    }
))

register_operation(OperationSpec(
    op_type="impute_missing",
    name="Impute Missing Values",
    category="Cleaning",
    description="Fills missing values using mean, median, mode, or a constant value.",
    explanation="Imputation preserves total sample size but artificially deflates feature variance. Median is robust to extreme outliers, while constant fill makes missingness explicit.",
    params_schema={
        "column": {"type": "column", "required": True, "label": "Target Column"},
        "strategy": {"type": "select", "options": ["mean", "median", "mode", "constant"], "default": "median", "label": "Strategy"},
        "constant_value": {"type": "string", "default": "", "label": "Constant Fill Value (if strategy=constant)"},
        "treat_empty_as_missing": {"type": "bool", "default": True, "label": "Treat Empty Strings as Missing"}
    }
))

register_operation(OperationSpec(
    op_type="replace_whitespace_with_nan",
    name="Convert Whitespace to NaN",
    category="Cleaning",
    description="Replaces empty or whitespace-only strings with true nulls (np.nan).",
    explanation="Strings containing only blank spaces look empty but pass non-null checks in Pandas unless standardized to NaN.",
    params_schema={
        "columns": {"type": "list", "default": [], "label": "Target Columns (empty = all string cols)"}
    }
))

register_operation(OperationSpec(
    op_type="drop_columns",
    name="Drop Columns",
    category="Cleaning",
    description="Removes unnecessary, redundant, or leaking columns from the dataset.",
    explanation="Eliminating high-cardinality noise, identifiers, or target-leaking fields improves model generalization and reduces memory footprint.",
    params_schema={
        "columns": {"type": "list", "required": True, "label": "Columns to Drop"}
    }
))

register_operation(OperationSpec(
    op_type="remove_duplicates",
    name="Remove Duplicate Rows",
    category="Cleaning",
    description="Deduplicates identical records based on full rows or key subset columns.",
    explanation="Duplicate records can inflate evaluation metrics if duplicate instances leak into both train and test partitions.",
    params_schema={
        "subset_columns": {"type": "list", "default": [], "label": "Key Columns (empty = full row)"},
        "keep": {"type": "select", "options": ["first", "last", "none"], "default": "first", "label": "Keep Strategy"}
    }
))

# ---------------------------------------------------------------------
# TYPE CONVERSION & TEXT NORMALIZATION
# ---------------------------------------------------------------------
register_operation(OperationSpec(
    op_type="convert_type",
    name="Convert Data Type",
    category="Transformation",
    description="Explicitly casts a column to string, integer, float, datetime, or boolean.",
    explanation="Accurate types ensure numeric statistics, chronological sorting, and categorical encoding function correctly without type errors.",
    params_schema={
        "column": {"type": "column", "required": True, "label": "Target Column"},
        "target_type": {"type": "select", "options": ["string", "integer", "float", "datetime", "boolean"], "default": "string", "label": "Target Type"},
        "errors": {"type": "select", "options": ["coerce", "raise", "ignore"], "default": "coerce", "label": "Error Handling"},
        "datetime_format": {"type": "string", "default": "", "label": "Datetime Format (optional)"}
    }
))

register_operation(OperationSpec(
    op_type="normalize_text",
    name="Normalize Text",
    category="Transformation",
    description="Trims leading/trailing whitespace and standardizes casing (lower, upper, title).",
    explanation="Prevents identical categorical values from being split into distinct classes due to irregular spacing or capitalization differences.",
    params_schema={
        "column": {"type": "column", "required": True, "label": "Text Column"},
        "strip_whitespace": {"type": "bool", "default": True, "label": "Trim Whitespace"},
        "case_style": {"type": "select", "options": ["lower", "upper", "title", "preserve"], "default": "lower", "label": "Case Style"}
    }
))

# ---------------------------------------------------------------------
# FEATURE ENGINEERING OPERATIONS
# ---------------------------------------------------------------------
register_operation(OperationSpec(
    op_type="feature_arithmetic",
    name="Feature Arithmetic",
    category="Feature Engineering",
    description="Computes arithmetic combinations (add, subtract, multiply, divide) between columns or scalars.",
    explanation="Interactions and ratios (e.g. revenue per employee) often expose predictive non-linear signals directly to linear and tree models.",
    params_schema={
        "col_a": {"type": "column", "required": True, "label": "First Column (A)"},
        "col_b": {"type": "string", "required": True, "label": "Second Column or Scalar (B)"},
        "operation": {"type": "select", "options": ["add", "subtract", "multiply", "divide"], "default": "add", "label": "Operation"},
        "new_col_name": {"type": "string", "default": "new_feature", "label": "Output Column Name"}
    },
    output_col_param="new_col_name"
))

register_operation(OperationSpec(
    op_type="feature_math",
    name="Mathematical Transform",
    category="Feature Engineering",
    description="Applies mathematical functions (abs, log1p, sqrt, round) to numeric columns.",
    explanation="Log and square-root transformations compress right-skewed heavy-tailed distributions and stabilize variance for linear models.",
    params_schema={
        "column": {"type": "column", "required": True, "label": "Numeric Column"},
        "math_func": {"type": "select", "options": ["log1p", "sqrt", "abs", "round"], "default": "log1p", "label": "Function"},
        "new_col_name": {"type": "string", "default": "math_feature", "label": "Output Column Name"}
    },
    output_col_param="new_col_name"
))

register_operation(OperationSpec(
    op_type="feature_binning",
    name="Numeric Binning / Discretization",
    category="Feature Engineering",
    description="Discretizes continuous numeric features into quantile or equal-width bins.",
    explanation="Discretization simplifies non-linear relationships and isolates outliers into boundary buckets, helpful when relationships are non-monotonic.",
    params_schema={
        "column": {"type": "column", "required": True, "label": "Numeric Column"},
        "bin_type": {"type": "select", "options": ["quantile", "uniform"], "default": "quantile", "label": "Binning Method"},
        "bins": {"type": "int", "default": 4, "label": "Number of Bins"},
        "new_col_name": {"type": "string", "default": "binned_feature", "label": "Output Column Name"}
    },
    output_col_param="new_col_name"
))

register_operation(OperationSpec(
    op_type="feature_datetime_extract",
    name="Extract Datetime Components",
    category="Feature Engineering",
    description="Extracts temporal components (year, month, day, dayofweek, quarter) from parsed datetime columns.",
    explanation="Machine learning models cannot consume raw timestamps. Extracting cyclical and trend components exposes seasonal patterns.",
    params_schema={
        "column": {"type": "column", "required": True, "label": "Datetime Column"},
        "components": {"type": "multiselect", "options": ["year", "month", "day", "dayofweek", "quarter"], "default": ["year", "month"], "label": "Components"},
        "prefix": {"type": "string", "default": "", "label": "Column Prefix (optional)"}
    }
))

register_operation(OperationSpec(
    op_type="feature_date_diff",
    name="Date Difference (Days)",
    category="Feature Engineering",
    description="Calculates elapsed days between two datetime columns.",
    explanation="Measures duration intervals such as fulfillment time, customer tenure, or licensing delay.",
    params_schema={
        "col_start": {"type": "column", "required": True, "label": "Start Date Column"},
        "col_end": {"type": "column", "required": True, "label": "End Date Column"},
        "new_col_name": {"type": "string", "default": "date_diff_days", "label": "Output Column Name"}
    },
    output_col_param="new_col_name"
))

register_operation(OperationSpec(
    op_type="feature_category_map",
    name="Category Mapping & Consolidation",
    category="Feature Engineering",
    description="Maps known categorical levels to consolidated categories, grouping rare classes into 'Other'.",
    explanation="High-cardinality categoricals dilute statistical power. Grouping rare levels prevents overfitting on infrequent categories.",
    params_schema={
        "column": {"type": "column", "required": True, "label": "Categorical Column"},
        "mapping": {"type": "json", "default": {}, "label": "Mapping Dictionary {old: new}"},
        "default_val": {"type": "string", "default": "Other", "label": "Fallback Category"},
        "new_col_name": {"type": "string", "default": "mapped_feature", "label": "Output Column Name"}
    },
    output_col_param="new_col_name"
))

register_operation(OperationSpec(
    op_type="feature_binary_indicator",
    name="Binary Flag Indicator",
    category="Feature Engineering",
    description="Generates a 1.0 / 0.0 binary indicator indicating if a column matches a specific target value.",
    explanation="Binary flags isolate specific high-value conditions (e.g. is_domestic, has_discount) into clean numeric predictors.",
    params_schema={
        "column": {"type": "column", "required": True, "label": "Target Column"},
        "target_value": {"type": "string", "required": True, "label": "Target Value to Match"},
        "new_col_name": {"type": "string", "default": "is_target", "label": "Output Column Name"}
    },
    output_col_param="new_col_name"
))

register_operation(OperationSpec(
    op_type="feature_one_hot",
    name="One-Hot Encoding (Pandas)",
    category="Feature Engineering",
    description="Creates dummy 0/1 columns for the top most frequent categories of a column.",
    explanation="Converts unordered categorical variables into numeric columns suitable for linear models.",
    params_schema={
        "column": {"type": "column", "required": True, "label": "Categorical Column"},
        "max_categories": {"type": "int", "default": 10, "label": "Max Top Categories"},
        "drop_first": {"type": "bool", "default": False, "label": "Drop First (avoid multicollinearity)"},
        "prefix": {"type": "string", "default": "", "label": "Column Prefix (optional)"}
    }
))

register_operation(OperationSpec(
    op_type="feature_text_length",
    name="Text Character Length",
    category="Feature Engineering",
    description="Calculates character count of text fields.",
    explanation="Text length serves as a structural proxy for descriptive depth or complexity.",
    params_schema={
        "column": {"type": "column", "required": True, "label": "Text Column"},
        "new_col_name": {"type": "string", "default": "text_length", "label": "Output Column Name"}
    },
    output_col_param="new_col_name"
))

register_operation(OperationSpec(
    op_type="feature_word_count",
    name="Text Word Count",
    category="Feature Engineering",
    description="Calculates word count of text fields.",
    explanation="Word counts measure information density without relying on heavy NLP vectorizers.",
    params_schema={
        "column": {"type": "column", "required": True, "label": "Text Column"},
        "new_col_name": {"type": "string", "default": "word_count", "label": "Output Column Name"}
    },
    output_col_param="new_col_name"
))

# ---------------------------------------------------------------------
# DOMAIN RECIPES (NETFLIX)
# ---------------------------------------------------------------------
register_operation(OperationSpec(
    op_type="netflix_recipe",
    name="Netflix Feature Recipe",
    category="Domain Recipes",
    description="Applies verified Netflix domain feature engineering (acquisition_lag, audience_tier, metadata_soup).",
    explanation="Domain-specific transformations derived from catalog business logic. Requires compatible columns (e.g., release_year, year_added, rating).",
    params_schema={
        "recipe_name": {
            "type": "select",
            "options": ["acquisition_lag", "audience_tier", "content_origin_proxy", "primary_genre", "metadata_soup"],
            "default": "acquisition_lag",
            "label": "Recipe Key"
        }
    }
))

# ---------------------------------------------------------------------
# WORKFLOW ML & RECOMMENDER PIPELINE STEPS
# ---------------------------------------------------------------------
register_operation(OperationSpec(
    op_type="ml_supervised_train",
    name="Train Supervised Model",
    category="Modeling",
    description="Fits a leakage-free Scikit-Learn pipeline (imputation, scaling, one-hot encoding, and estimator) on a train split.",
    explanation="Supervised estimators learn mapping functions from input features to targets. Evaluating exclusively on held-out test splits guards against data leakage and overfitting.",
    params_schema={
        "target_col": {"type": "column", "required": True, "label": "Target Column"},
        "task_type": {"type": "select", "options": ["auto", "classification", "regression"], "default": "auto", "label": "Task Type"},
        "feature_cols": {"type": "multiselect", "default": [], "label": "Predictor Columns"},
        "test_size": {"type": "float", "default": 0.2, "label": "Test Split Ratio"},
        "random_state": {"type": "int", "default": 42, "label": "Random Seed"},
        "model_name": {"type": "select", "options": ["baseline", "logistic", "linear", "decision_tree", "random_forest"], "default": "logistic", "label": "Model Estimator"},
        "cv_folds": {"type": "int", "default": 0, "label": "Cross-Validation Folds (0 = test split only)"}
    },
    mutates_data=False,
    is_ml_step=True
))

register_operation(OperationSpec(
    op_type="recommender_query",
    name="Recommender Similarity Query",
    category="Recommender",
    description="Fits multi-field TF-IDF vector representations and queries top-K similar items on-demand using sparse cosine similarity.",
    explanation="Content-based recommenders match item descriptions and metadata in a high-dimensional vector space. Sparse on-demand similarity eliminates dense N x N matrix memory consumption.",
    params_schema={
        "title_col": {"type": "column", "default": "title", "label": "Title / Identifier Column"},
        "query_title": {"type": "string", "required": True, "label": "Query Item Title"},
        "text_fields": {"type": "multiselect", "default": [], "label": "Metadata Text Fields"},
        "top_k": {"type": "int", "default": 5, "label": "Top-K Recommendations"}
    },
    mutates_data=False,
    is_ml_step=True
))


def get_operation_spec(op_type: str) -> Optional[OperationSpec]:
    """Retrieves an OperationSpec by its stable op_type identifier."""
    # Check exact match or normalized match
    if op_type in OPERATION_REGISTRY:
        return OPERATION_REGISTRY[op_type]
    aliases = {
        "impute_numeric": "impute_missing",
        "impute_categorical_mode": "impute_missing",
        "fill_constant": "impute_missing",
        "whitespace_to_nan": "replace_whitespace_with_nan",
        "numeric_arithmetic": "feature_arithmetic",
        "math_transform": "feature_math",
        "numeric_binning": "feature_binning",
        "datetime_extract": "feature_datetime_extract",
        "date_difference": "feature_date_diff",
        "categorical_map": "feature_category_map",
        "binary_indicator": "feature_binary_indicator",
        "one_hot_encode": "feature_one_hot",
        "train_supervised_model": "ml_supervised_train",
    }
    canonical = aliases.get(op_type)
    return OPERATION_REGISTRY.get(canonical) if canonical else None


def list_operations_by_category() -> Dict[str, List[OperationSpec]]:
    """Returns a dictionary grouping operations by their category."""
    cats: Dict[str, List[OperationSpec]] = {}
    for spec in OPERATION_REGISTRY.values():
        cats.setdefault(spec.category, []).append(spec)
    return cats


# Convenient alias
get_operation = get_operation_spec
