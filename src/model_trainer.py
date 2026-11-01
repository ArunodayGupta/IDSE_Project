"""
model_trainer.py
Predictive modeling engine for TMDB Movie Box Office Worldwide Revenue.

Models supported:
1. Dummy Baseline (Median)
2. Random Forest Regressor
3. Extra Trees Regressor
4. HistGradientBoosting Regressor

Training strictly on log1p(revenue) with expm1 predictions clipped at 0 (nonnegative).
Primary evaluation metric: RMSLE (Root Mean Squared Logarithmic Error).
"""

from typing import Dict, Any, List, Tuple, Optional
import time
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor
)
from sklearn.metrics import mean_absolute_error, r2_score

from src.feature_engineering import engineer_movie_features


MODEL_REGISTRY = {
    "dummy": {
        "name": "Dummy Baseline",
        "description": "Baseline predicting median training revenue.",
        "class": DummyRegressor,
    },
    "random_forest": {
        "name": "Random Forest Regressor",
        "description": "Ensemble of randomized decision trees with bagging.",
        "class": RandomForestRegressor,
    },
    "extra_trees": {
        "name": "Extra Trees Regressor",
        "description": "Extremely randomized trees with random split thresholds.",
        "class": ExtraTreesRegressor,
    },
    "hist_gradient_boosting": {
        "name": "HistGradientBoosting Regressor",
        "description": "Histogram-based gradient boosting for fast non-linear fits.",
        "class": HistGradientBoostingRegressor,
    },
}


def compute_rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes Root Mean Squared Logarithmic Error (RMSLE).
    Ensures predictions are clipped >= 0 before log transform.
    """
    clipped_pred = np.clip(y_pred, 0, None)
    log_true = np.log1p(np.clip(y_true, 0, None))
    log_pred = np.log1p(clipped_pred)
    return float(np.sqrt(np.mean((log_pred - log_true) ** 2)))


def build_feature_preprocessor(feature_columns: List[str]) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """
    Builds a ColumnTransformer that scales numerical variables and one-hot encodes
    low-cardinality categorical variables.
    """
    cat_cols = ["original_language", "primary_genre"]
    # Ensure they exist in feature_columns
    cat_cols = [c for c in cat_cols if c in feature_columns]
    num_cols = [c for c in feature_columns if c not in cat_cols]

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=30)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop"
    )

    return preprocessor, num_cols, cat_cols


def create_estimator(model_key: str, hyperparams: Optional[Dict[str, Any]] = None, random_state: int = 42) -> Any:
    """Instantiates the selected model algorithm with validated hyperparameters."""
    params = hyperparams.copy() if hyperparams else {}

    if model_key == "dummy":
        return DummyRegressor(strategy="median")

    elif model_key == "random_forest":
        return RandomForestRegressor(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", 12),
            min_samples_split=params.get("min_samples_split", 4),
            random_state=random_state,
            n_jobs=-1
        )

    elif model_key == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", 12),
            min_samples_split=params.get("min_samples_split", 4),
            random_state=random_state,
            n_jobs=-1
        )

    elif model_key == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            max_iter=params.get("max_iter", 100),
            max_depth=params.get("max_depth", 6),
            learning_rate=params.get("learning_rate", 0.08),
            random_state=random_state
        )

    else:
        raise ValueError(f"Unknown model key '{model_key}'. Supported: {list(MODEL_REGISTRY.keys())}")


def extract_feature_importances(
    pipeline: Pipeline,
    num_cols: List[str],
    cat_cols: List[str]
) -> pd.DataFrame:
    """
    Extracts feature names and relative importance values from tree-based estimators.
    """
    estimator = pipeline.named_steps.get("estimator")
    preprocessor = pipeline.named_steps.get("preprocessor")

    if not hasattr(estimator, "feature_importances_"):
        return pd.DataFrame(columns=["feature", "importance"])

    importances = estimator.feature_importances_

    # Resolve one-hot encoded categorical feature names
    cat_feature_names: List[str] = []
    try:
        cat_pipe = preprocessor.named_transformers_.get("cat")
        if cat_pipe and hasattr(cat_pipe.named_steps.get("onehot"), "get_feature_names_out"):
            cat_feature_names = list(cat_pipe.named_steps["onehot"].get_feature_names_out(cat_cols))
    except Exception:
        cat_feature_names = [f"cat_{i}" for i in range(len(importances) - len(num_cols))]

    all_names = num_cols + cat_feature_names

    if len(all_names) != len(importances):
        # Truncate or pad safely
        all_names = all_names[:len(importances)] if len(all_names) > len(importances) else (
            all_names + [f"feature_{i}" for i in range(len(all_names), len(importances))]
        )

    df_imp = pd.DataFrame({
        "feature": all_names,
        "importance": importances
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)

    return df_imp


def train_movie_regressor(
    model_key: str,
    hyperparams: Optional[Dict[str, Any]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Trains a regression model on log1p(y_train), evaluates against y_val,
    and returns metrics, predictions, and diagnostic details.
    """
    preprocessor, num_cols, cat_cols = build_feature_preprocessor(list(X_train.columns))
    estimator = create_estimator(model_key, hyperparams, random_state=random_state)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("estimator", estimator)
    ])

    # Fit exclusively on log1p(y_train) to stabilize exponential variance
    y_train_log = np.log1p(np.clip(y_train.values, 0, None))

    t0 = time.time()
    pipeline.fit(X_train, y_train_log)
    fit_duration_sec = round(time.time() - t0, 3)

    # Predict on validation partition and transform back with expm1
    y_val_log_pred = pipeline.predict(X_val)
    y_val_pred = np.clip(np.expm1(y_val_log_pred), 0, None)
    y_val_arr = y_val.values

    # Validation Metrics
    rmsle = compute_rmsle(y_val_arr, y_val_pred)
    mae = float(mean_absolute_error(y_val_arr, y_val_pred))
    r2 = float(r2_score(y_val_arr, y_val_pred))

    # Feature Importance
    df_importances = extract_feature_importances(pipeline, num_cols, cat_cols)

    return {
        "model_key": model_key,
        "model_name": MODEL_REGISTRY.get(model_key, {}).get("name", model_key),
        "pipeline": pipeline,
        "hyperparams": hyperparams or {},
        "fit_duration_sec": fit_duration_sec,
        "rmsle": rmsle,
        "mae": mae,
        "r2": r2,
        "y_val_true": y_val_arr,
        "y_val_pred": y_val_pred,
        "residuals": y_val_arr - y_val_pred,
        "feature_importances": df_importances,
    }


def predict_single_movie(pipeline: Pipeline, single_row_df: pd.DataFrame) -> float:
    """
    Predicts worldwide box office revenue in USD for a single movie feature row.
    Guarantees non-negative prediction.
    """
    pred_log = pipeline.predict(single_row_df)[0]
    pred_revenue = float(np.clip(np.expm1(pred_log), 0, None))
    return pred_revenue


def generate_kaggle_submission(
    model_key: str,
    hyperparams: Optional[Dict[str, Any]],
    raw_train_df: pd.DataFrame,
    raw_test_df: pd.DataFrame,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Fits the selected model on the entire labeled training dataset (100%),
    predicts worldwide revenue for the test dataset, and returns a DataFrame
    formatted for Kaggle submission with columns: ['id', 'revenue'].
    Preserves original test IDs and guarantees non-negative revenues.
    """
    # 1. Feature Engineering on Full Training Data
    X_full, y_full, stats = engineer_movie_features(raw_train_df, is_train=True)

    # 2. Build Pipeline and Train on 100% of Training Data
    preprocessor, _, _ = build_feature_preprocessor(list(X_full.columns))
    estimator = create_estimator(model_key, hyperparams, random_state=random_state)
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("estimator", estimator)
    ])

    y_full_log = np.log1p(np.clip(y_full.values, 0, None))
    pipeline.fit(X_full, y_full_log)

    # 3. Feature Engineering on Test Data reusing Training Stats
    X_test, _, _ = engineer_movie_features(raw_test_df, is_train=False, stats=stats)

    # 4. Predict on Test Data
    test_pred_log = pipeline.predict(X_test)
    test_pred_revenue = np.clip(np.expm1(test_pred_log), 0, None)

    # 5. Format Submission
    submission = pd.DataFrame({
        "id": raw_test_df["id"].values,
        "revenue": np.round(test_pred_revenue, 2)
    })

    return submission
