"""
model_lab.py
Core modeling, preprocessing, evaluation, and recommender experiment engine for Phase 3 DataLab.

Includes:
1. Supervised Predictive Modeling Studio (Classification & Regression):
   - Automated task detection, target validation, and missing target exclusion.
   - ColumnTransformer and Pipeline preprocessing fit exclusively on training data.
   - Supported models: Dummy baseline, Logistic/Ridge, Decision Trees, Random Forests.
   - Comprehensive evaluation metrics, cross-validation, confusion matrices, and residuals.
   - Readable feature importance extraction and error analysis.
2. Recommender Experiment Arena:
   - Memory-safe multi-field TF-IDF with custom weights and field ablation.
   - On-demand query-time sparse similarity without dense N x N matrix allocation.
   - A/B experiment comparisons with Jaccard overlap and rank shift analysis.
   - User relevance feedback with judged Precision@K and NDCG@K without fabricated ground truth.
3. Serialization & Export services for experiment tracking and JSON replay.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import json
import time
import math
import hashlib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix

from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# Supported Supervised Estimators
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


# =====================================================================
# CONFIGURATION FINGERPRINTING & STATE AUDITING (DEFECT A3)
# =====================================================================

def compute_model_config_fingerprint(
    dataset_identity: str,
    target_col: str,
    task_type: str,
    feature_cols: List[str],
    test_size: float = 0.2,
    random_state: int = 42,
    cv_folds: int = 0,
    model_name: str = "baseline",
    hyperparameters: Optional[Dict[str, Any]] = None,
    preprocessing_config: Optional[Dict[str, Any]] = None,
    modeling_rows: Optional[int] = None,
    **kwargs: Any
) -> str:
    """
    Computes a deterministic SHA-256 fingerprint for a predictive modeling configuration.
    Any variation in dataset identity, target, features, split, seed, CV, model,
    or hyperparameters produces a distinct fingerprint, preventing old metrics
    from being associated with new selections (Defect A3).
    """
    effective_params = hyperparameters or kwargs.get("model_params") or {}
    payload = {
        "dataset_identity": str(dataset_identity),
        "target_col": str(target_col),
        "task_type": str(task_type),
        "feature_cols": sorted(list(set(feature_cols))),
        "test_size": round(float(test_size), 4),
        "random_state": int(random_state),
        "cv_folds": int(cv_folds),
        "model_name": str(model_name),
        "hyperparameters": {k: str(v) for k, v in sorted(effective_params.items())},
        "preprocessing_config": {k: str(v) for k, v in sorted((preprocessing_config or {}).items())},
        "modeling_rows": int(modeling_rows) if modeling_rows is not None else None,
    }
    raw = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# =====================================================================
# SUPERVISED PREDICTIVE MODELING STUDIO
# =====================================================================

def detect_suggested_task(data: Union[pd.Series, pd.DataFrame], target_col: Optional[str] = None) -> str:
    """
    Suggests whether a target column is classification or regression based
    on data type and cardinality. Accepts either a pd.Series or (pd.DataFrame, target_col).
    """
    if isinstance(data, pd.DataFrame):
        if target_col is None or target_col not in data.columns:
            return "classification"
        series = data[target_col]
    else:
        series = data

    if pd.api.types.is_numeric_dtype(series):
        valid = series.dropna()
        if valid.nunique() > 10 and not set(valid.unique()).issubset({0, 1, 0.0, 1.0}):
            return "regression"
    return "classification"


def validate_modeling_inputs(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    task_type: str = "classification"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Validates modeling inputs with comprehensive safeguards:
    - Target column presence and non-trivial target distribution.
    - Excludes target from features.
    - Removes rows with missing target with transparent reporting.
    - Screens for all-null features and extreme categorical cardinality.
    - Validates minimum row count (>= 5 rows).
    - Checks classification target has >= 2 distinct classes (Defect A8).
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    if target_col in feature_cols:
        raise ValueError(f"Target column '{target_col}' cannot be included in feature columns.")

    # Remove target from features
    valid_features = [c for c in feature_cols if c in df.columns and c != target_col]
    if not valid_features:
        raise ValueError("At least one valid feature column must be selected (target cannot be a feature).")

    # Target missingness handling
    s_target = df[target_col]
    missing_target_mask = s_target.isna()
    if pd.api.types.is_object_dtype(s_target) or pd.api.types.is_string_dtype(s_target):
        missing_target_mask = missing_target_mask | (s_target.astype(str).str.strip() == "")

    missing_target_count = int(missing_target_mask.sum())
    clean_df = df[~missing_target_mask].copy()

    if clean_df.empty:
        raise ValueError(f"Target column '{target_col}' contains no valid rows after removing missing values.")

    if len(clean_df) < 5:
        raise ValueError(f"Dataset contains insufficient valid modeling rows ({len(clean_df)}). Reliable modeling requires at least 5 rows.")

    warnings: List[str] = []

    if task_type == "classification":
        unique_classes = clean_df[target_col].dropna().unique()
        if len(unique_classes) < 2:
            raise ValueError(f"Classification target '{target_col}' contains only 1 distinct class ({unique_classes[0]}). Supervised classification requires at least 2 distinct classes.")
    else:  # regression
        # Ensure target is convertible to numeric
        num_target = pd.to_numeric(clean_df[target_col], errors="coerce")
        invalid_target_mask = num_target.isna()
        if invalid_target_mask.sum() > 0:
            invalid_count = int(invalid_target_mask.sum())
            clean_df = clean_df[~invalid_target_mask].copy()
            clean_df[target_col] = num_target[~invalid_target_mask].astype(float)
            missing_target_count += invalid_count
        else:
            clean_df[target_col] = num_target.astype(float)

        if clean_df.empty:
            raise ValueError(f"Regression target '{target_col}' contains no valid numeric values.")

        if len(clean_df) < 5:
            raise ValueError(f"Regression target '{target_col}' contains only {len(clean_df)} valid numeric rows (minimum 5 required).")

    # Feature screening: all-null and free-text screening
    all_null_features = []
    screened_features = []
    for c in valid_features:
        s = clean_df[c]
        if s.isna().all() or (pd.api.types.is_string_dtype(s) and (s.astype(str).str.strip() == "").all()):
            all_null_features.append(c)
        else:
            screened_features.append(c)

    if all_null_features:
        warnings.append(f"Excluded {len(all_null_features)} feature(s) containing 100% missing values: {all_null_features}")

    if not screened_features:
        raise ValueError("All selected feature columns contain 100% missing values. Select at least one feature with non-missing data.")

    numeric_features = []
    categorical_features = []
    excluded_text_features = []
    high_cardinality_features = []

    for c in screened_features:
        s = clean_df[c]
        if pd.api.types.is_numeric_dtype(s):
            numeric_features.append(c)
        else:
            # Check if long free-text column (average string length > 50 chars or near 100% unique)
            non_null = s.dropna().astype(str)
            avg_len = non_null.str.len().mean() if not non_null.empty else 0
            unique_ratio = non_null.nunique() / len(non_null) if len(non_null) > 0 else 0

            if avg_len > 60 or (unique_ratio > 0.85 and len(non_null) > 50):
                excluded_text_features.append(c)
            else:
                categorical_features.append(c)
                nunique = non_null.nunique()
                if nunique > 50:
                    high_cardinality_features.append((c, nunique))
                    warnings.append(f"Categorical feature '{c}' has high cardinality ({nunique} distinct categories). Memory-conscious encoding will limit to top categories to avoid memory explosion.")

    meta = {
        "original_rows": len(df),
        "modeling_rows": len(clean_df),
        "excluded_target_missing": missing_target_count,
        "target_col": target_col,
        "task_type": task_type,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "excluded_text_features": excluded_text_features,
        "all_null_features": all_null_features,
        "high_cardinality_features": high_cardinality_features,
        "usable_features": numeric_features + categorical_features,
        "warnings": warnings,
    }
    return clean_df, meta


def build_preprocessor(
    numeric_cols: List[str],
    categorical_cols: List[str],
    max_categories: int = 50
) -> ColumnTransformer:
    """
    Constructs a memory-conscious Scikit-Learn ColumnTransformer pipeline (Defect A4).
    Numeric: SimpleImputer(median) + StandardScaler.
    Categorical: SimpleImputer(most_frequent) + OneHotEncoder(sparse_output=True, handle_unknown='ignore', max_categories=50).
    Employs sparse_threshold=1.0 to preserve sparse representations when categorical columns are present,
    avoiding massive dense one-hot matrix expansion.
    """
    transformers = []
    if numeric_cols:
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", num_pipe, numeric_cols))

    if categorical_cols:
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True, max_categories=max_categories)),
        ])
        transformers.append(("cat", cat_pipe, categorical_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=1.0)


def get_model_estimator(model_name: str, task_type: str, random_state: int = 42, **kwargs) -> Any:
    """Instantiates a Scikit-learn estimator based on model_name and task_type."""
    if task_type == "classification":
        if model_name == "baseline":
            return DummyClassifier(strategy="most_frequent")
        elif model_name == "logistic":
            return LogisticRegression(max_iter=1000, random_state=random_state)
        elif model_name == "decision_tree":
            max_depth = kwargs.get("max_depth", 5)
            return DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
        elif model_name == "random_forest":
            n_est = kwargs.get("n_estimators", 50)
            max_depth = kwargs.get("max_depth", 6)
            return RandomForestClassifier(n_estimators=n_est, max_depth=max_depth, random_state=random_state)
        else:
            raise ValueError(f"Unknown classification model: '{model_name}'.")
    else:  # regression
        if model_name == "baseline":
            return DummyRegressor(strategy="mean")
        elif model_name == "linear":
            return Ridge(alpha=1.0, random_state=random_state)
        elif model_name == "decision_tree":
            max_depth = kwargs.get("max_depth", 5)
            return DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)
        elif model_name == "random_forest":
            n_est = kwargs.get("n_estimators", 50)
            max_depth = kwargs.get("max_depth", 6)
            return RandomForestRegressor(n_estimators=n_est, max_depth=max_depth, random_state=random_state)
        else:
            raise ValueError(f"Unknown regression model: '{model_name}'.")


def train_and_evaluate_model(
    model_name: str,
    task_type: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    numeric_cols: List[str],
    categorical_cols: List[str],
    random_state: int = 42,
    cv_folds: int = 0,
    dataset_identity: str = "dataset",
    **hyperparams
) -> Dict[str, Any]:
    """
    Fits preprocessing and estimator exclusively on the training partition,
    evaluates on the test partition, and computes comprehensive real metrics.
    Includes configuration fingerprinting (Defect A3) and robust CV validation (Defect A8).
    """
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    estimator = get_model_estimator(model_name, task_type, random_state=random_state, **hyperparams)
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("estimator", estimator),
    ])

    t0 = time.time()
    # 1. Fit pipeline strictly on X_train, y_train
    pipeline.fit(X_train, y_train)
    fit_duration_sec = round(time.time() - t0, 4)

    # 2. Predict on X_test
    t1 = time.time()
    y_pred = pipeline.predict(X_test)
    predict_duration_sec = round(time.time() - t1, 4)

    metrics: Dict[str, Any] = {}
    extra: Dict[str, Any] = {}
    model_warnings: List[str] = []

    if task_type == "classification":
        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)

        labels = sorted(list(set(y_train.dropna().unique()).union(set(y_test.dropna().unique()))))
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        # ROC-AUC if predict_proba available
        roc_auc = None
        if hasattr(pipeline.named_steps["estimator"], "predict_proba"):
            try:
                y_prob = pipeline.predict_proba(X_test)
                if len(labels) == 2:
                    roc_auc = round(float(roc_auc_score(y_test, y_prob[:, 1])), 4)
                else:
                    roc_auc = round(float(roc_auc_score(y_test, y_prob, multi_class="ovr")), 4)
            except Exception:
                roc_auc = None

        metrics = {
            "accuracy": round(float(acc), 4),
            "balanced_accuracy": round(float(bal_acc), 4),
            "macro_f1": round(float(macro_f1), 4),
            "weighted_f1": round(float(weighted_f1), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "roc_auc": roc_auc,
        }
        extra = {
            "confusion_matrix": cm.tolist(),
            "labels": [str(l) for l in labels],
            "classification_report": report,
        }

    else:  # regression
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = math.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        metrics = {
            "mae": round(float(mae), 4),
            "rmse": round(float(rmse), 4),
            "r2": round(float(r2), 4),
        }
        residuals = (y_test - y_pred).tolist()
        extra = {
            "residuals": residuals,
            "y_test": y_test.tolist(),
            "y_pred": y_pred.tolist(),
        }

    # Cross-validation on training fold with explicit validation and actionable warnings (Defect A8)
    cv_mean = None
    cv_std = None
    cv_warnings = []
    if cv_folds >= 2:
        if len(X_train) < cv_folds * 2:
            cv_warn = f"Cross-validation skipped: Training partition has only {len(X_train)} samples, fewer than {cv_folds * 2} required for {cv_folds}-fold CV."
            cv_warnings.append(cv_warn)
            model_warnings.append(cv_warn)
        else:
            try:
                scoring = "accuracy" if task_type == "classification" else "r2"
                if task_type == "classification":
                    min_class_count = int(y_train.value_counts().min())
                    if min_class_count < cv_folds:
                        cv_warn = f"Stratified CV requires each class to have >= {cv_folds} samples (rarest class has {min_class_count}). Evaluated using standard KFold without stratification."
                        cv_warnings.append(cv_warn)
                        model_warnings.append(cv_warn)
                        cv_splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
                    else:
                        cv_splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
                else:
                    cv_splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

                scores = cross_val_score(pipeline, X_train, y_train, cv=cv_splitter, scoring=scoring)
                cv_mean = round(float(scores.mean()), 4)
                cv_std = round(float(scores.std()), 4)
            except Exception as e:
                cv_mean = None
                cv_std = None
                cv_warn = f"Cross-validation execution failure: {str(e)}"
                cv_warnings.append(cv_warn)
                model_warnings.append(cv_warn)

    metrics["cv_mean"] = cv_mean
    metrics["cv_std"] = cv_std
    if cv_warnings:
        extra["cv_warnings"] = cv_warnings

    # Feature Importance extraction
    feature_importances = extract_feature_importances(pipeline, numeric_cols, categorical_cols)

    # Compute configuration fingerprint (Defect A3)
    total_modeling_rows = len(X_train) + len(X_test)
    test_ratio = round(len(X_test) / total_modeling_rows, 4) if total_modeling_rows > 0 else 0.2
    target_name = str(getattr(y_train, "name", "target") or "target")

    config_fingerprint = compute_model_config_fingerprint(
        dataset_identity=dataset_identity,
        target_col=target_name,
        task_type=task_type,
        feature_cols=list(X_train.columns),
        test_size=test_ratio,
        random_state=random_state,
        cv_folds=cv_folds,
        model_name=model_name,
        hyperparameters=hyperparams,
        preprocessing_config={"max_categories": 50, "sparse": True},
        modeling_rows=total_modeling_rows
    )

    return {
        "model_name": model_name,
        "task_type": task_type,
        "fitted_pipeline": pipeline,
        "fit_duration_sec": fit_duration_sec,
        "predict_duration_sec": predict_duration_sec,
        "metrics": metrics,
        "extra": extra,
        "feature_importances": feature_importances,
        "y_pred": y_pred,
        "config_fingerprint": config_fingerprint,
        "dataset_identity": dataset_identity,
        "test_size": test_ratio,
        "random_state": random_state,
        "cv_folds": cv_folds,
        "warnings": model_warnings,
        "modeling_rows": total_modeling_rows,
    }


def extract_feature_importances(
    pipeline: Pipeline,
    numeric_cols: List[str],
    categorical_cols: List[str]
) -> List[Dict[str, Any]]:
    """
    Extracts readable feature importances or linear coefficients from fitted pipeline.
    Maps one-hot columns back to readable feature labels.
    """
    estimator = pipeline.named_steps["estimator"]
    preprocessor = pipeline.named_steps["preprocessor"]

    feature_names = []
    try:
        raw_names = preprocessor.get_feature_names_out()
        for rn in raw_names:
            clean_n = rn.replace("num__", "").replace("cat__", "")
            feature_names.append(clean_n)
    except Exception:
        feature_names = numeric_cols + categorical_cols

    importances = []
    if hasattr(estimator, "feature_importances_"):
        raw_imp = estimator.feature_importances_
        for name, imp in zip(feature_names, raw_imp):
            importances.append({"feature": name, "importance": round(float(imp), 4)})
        importances.sort(key=lambda x: abs(x["importance"]), reverse=True)

    elif hasattr(estimator, "coef_"):
        coef = estimator.coef_
        # If multiclass or multi-target, average absolute coefs
        if coef.ndim > 1:
            mean_coef = np.mean(np.abs(coef), axis=0)
        else:
            mean_coef = coef
        for name, val in zip(feature_names, mean_coef):
            importances.append({"feature": name, "importance": round(float(val), 4)})
        importances.sort(key=lambda x: abs(x["importance"]), reverse=True)

    return importances


def compute_error_analysis(
    fitted_pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    raw_test_df: pd.DataFrame,
    task_type: str,
    top_n: int = 5
) -> pd.DataFrame:
    """
    Identifies top prediction errors for user inspection:
    - Classification: Misclassified records and confidence.
    - Regression: Largest absolute residuals.
    """
    y_pred = fitted_pipeline.predict(X_test)
    res_df = raw_test_df.copy().reset_index(drop=True)
    res_df["_actual"] = y_test.reset_index(drop=True)
    res_df["_predicted"] = y_pred

    if task_type == "classification":
        misclassified_mask = res_df["_actual"] != res_df["_predicted"]
        errors = res_df[misclassified_mask].copy()

        # Add prediction confidence if available
        if hasattr(fitted_pipeline.named_steps["estimator"], "predict_proba"):
            probs = fitted_pipeline.predict_proba(X_test)
            confidences = np.max(probs, axis=1)
            errors["_confidence"] = [round(float(c), 3) for c in confidences[misclassified_mask]]

        return errors.head(top_n)

    else:  # regression
        res_df["_abs_error"] = (res_df["_actual"] - res_df["_predicted"]).abs()
        res_df = res_df.sort_values(by="_abs_error", ascending=False)
        return res_df.head(top_n)


# =====================================================================
# RECOMMENDER EXPERIMENT ARENA (MEMORY-SAFE MULTI-FIELD TF-IDF)
# =====================================================================

class RecommenderArenaService:
    """
    Multi-field content-based recommendation service with field weighting,
    ablation, on-demand similarity, and relevance feedback tracking.
    Operates without allocating dense N x N pairwise similarity matrices.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        title_col: str = "title",
        text_fields: Optional[List[str]] = None,
        field_weights: Optional[Dict[str, float]] = None,
        max_features: int = 3000,
        stop_words: Optional[str] = "english",
        ngram_range: Tuple[int, int] = (1, 1),
        min_df: Union[int, float] = 1,
        max_df: Union[int, float] = 1.0,
        sublinear_tf: bool = False
    ):
        self.df = df.reset_index(drop=True).copy()
        self.title_col = title_col
        self.text_fields = text_fields if text_fields else [
            c for c in df.columns if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c])
        ][:4]
        self.field_weights = field_weights if field_weights else {f: 1.0 for f in self.text_fields}
        self.max_features = max_features
        self.stop_words = stop_words
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.sublinear_tf = sublinear_tf

        self.vectorizers: Dict[str, TfidfVectorizer] = {}
        self.combined_matrix: Optional[csr_matrix] = None
        self.vocabulary_size: int = 0
        self.nnz_entries: int = 0
        self.title_to_index: Dict[str, int] = {}
        self.lower_title_to_index: Dict[str, int] = {}
        self.fit_duration_sec: float = 0.0

        self._build_index_and_fit()

    def _build_index_and_fit(self) -> None:
        """Fits multi-field TF-IDF representations and combines them into unit-normalized sparse matrix."""
        t0 = time.time()

        # Build title lookup maps
        self.title_to_index = {}
        self.lower_title_to_index = {}
        for idx, t in enumerate(self.df[self.title_col].astype(str)):
            clean_t = t.strip()
            if clean_t not in self.title_to_index:
                self.title_to_index[clean_t] = idx
            low_t = clean_t.lower()
            if low_t not in self.lower_title_to_index:
                self.lower_title_to_index[low_t] = idx

        field_matrices = []
        total_vocab = 0

        for field in self.text_fields:
            if field in self.df.columns:
                weight = float(self.field_weights.get(field, 1.0))
                if weight <= 0.0:
                    continue  # Ablated / disabled field

                corpus = self.df[field].fillna("").astype(str)
                try:
                    vec = TfidfVectorizer(
                        max_features=self.max_features,
                        stop_words=self.stop_words,
                        ngram_range=self.ngram_range,
                        min_df=self.min_df,
                        max_df=self.max_df,
                        sublinear_tf=self.sublinear_tf
                    )
                    mat = vec.fit_transform(corpus)
                    if len(vec.vocabulary_) > 0:
                        self.vectorizers[field] = vec
                        total_vocab += len(vec.vocabulary_)
                        # Apply weight
                        weighted_mat = mat.multiply(weight)
                        field_matrices.append(weighted_mat)
                except Exception:
                    # Ignore empty vocabularies or invalid text safely (Defect A6)
                    continue

        if field_matrices:
            raw_combined = hstack(field_matrices, format="csr")
            # Normalize rows to unit L2 norm so cosine similarity is direct dot product
            self.combined_matrix = normalize(raw_combined, norm="l2", axis=1)
            self.vocabulary_size = total_vocab
            self.nnz_entries = int(self.combined_matrix.nnz)
        else:
            self.combined_matrix = None
            self.vocabulary_size = 0
            self.nnz_entries = 0

        self.fit_duration_sec = round(time.time() - t0, 4)

    def recommend(
        self,
        query_title: str,
        top_k: int = 5
    ) -> Tuple[pd.DataFrame, float]:
        """
        Calculates recommendations for a query title using on-demand sparse dot product.
        Returns top-K recommendations DataFrame and query duration in seconds.
        """
        if self.combined_matrix is None or len(self.df) == 0:
            return pd.DataFrame(), 0.0

        t0 = time.time()
        title_str = str(query_title).strip()
        idx = None
        if title_str in self.title_to_index:
            idx = self.title_to_index[title_str]
        elif title_str.lower() in self.lower_title_to_index:
            idx = self.lower_title_to_index[title_str.lower()]
        else:
            matches = self.df[self.df[self.title_col].astype(str).str.lower().str.contains(title_str.lower(), regex=False, na=False)]
            if not matches.empty:
                idx = matches.index[0]

        if idx is None:
            return pd.DataFrame(), 0.0

        # Query vector: 1 x M
        query_vec = self.combined_matrix[idx]
        # On-demand dot product similarity vector: 1 x N
        sim_scores = cosine_similarity(query_vec, self.combined_matrix).flatten()

        # Sort descending
        ranked_indices = np.argsort(-sim_scores)

        # Exclude query item itself
        candidates = []
        target_norm = str(self.df.iloc[idx][self.title_col]).strip().lower()
        for cand_idx in ranked_indices:
            if cand_idx == idx:
                continue
            cand_title_norm = str(self.df.iloc[cand_idx][self.title_col]).strip().lower()
            if cand_title_norm == target_norm:
                continue
            score = round(float(sim_scores[cand_idx]), 4)
            candidates.append((cand_idx, score))
            if len(candidates) >= top_k:
                break

        query_duration_sec = round(time.time() - t0, 4)

        indices = [i for i, _ in candidates]
        scores = [s for _, s in candidates]

        res_df = self.df.iloc[indices].copy()
        res_df["similarity_score"] = scores
        res_df["rank"] = list(range(1, len(res_df) + 1))
        return res_df, query_duration_sec


def compare_recommender_runs(
    rec_df_a: pd.DataFrame,
    rec_df_b: pd.DataFrame,
    title_col: str = "title"
) -> Dict[str, Any]:
    """
    Compares two recommendation outputs for the same query item:
    - Jaccard overlap between top-K lists.
    - Items shared, entering, and leaving.
    - Rank shifts and score differences.
    """
    if rec_df_a.empty or rec_df_b.empty:
        return {"overlap_pct": 0.0, "shared_count": 0, "shared_items": [], "entering_items": [], "leaving_items": []}

    titles_a = rec_df_a[title_col].astype(str).tolist()
    titles_b = rec_df_b[title_col].astype(str).tolist()

    set_a = set(titles_a)
    set_b = set(titles_b)

    shared = list(set_a.intersection(set_b))
    union = list(set_a.union(set_b))
    jaccard = (len(shared) / len(union) * 100) if union else 0.0

    entering = [t for t in titles_b if t not in set_a]
    leaving = [t for t in titles_a if t not in set_b]

    # Rank shifts for shared items
    rank_shifts = []
    map_a = {row[title_col]: row["rank"] for _, row in rec_df_a.iterrows()}
    map_b = {row[title_col]: row["rank"] for _, row in rec_df_b.iterrows()}
    for item in shared:
        r_a = map_a[item]
        r_b = map_b[item]
        rank_shifts.append({"item": item, "rank_a": r_a, "rank_b": r_b, "shift": r_a - r_b})

    return {
        "overlap_pct": round(jaccard, 2),
        "shared_count": len(shared),
        "shared_items": shared,
        "entering_items": entering,
        "leaving_items": leaving,
        "rank_shifts": rank_shifts,
    }


def compute_relevance_feedback_metrics(
    recommended_titles: List[str],
    judgments: Dict[str, int]
) -> Dict[str, Any]:
    """
    Computes truthful ranking evaluation metrics on user relevance feedback (Defect A6):
    - Truthful label: Relevance rate among judged recommendations with explicit denominator.
    - Partial-judgment NDCG@K documented explicitly without claiming complete ground truth.
    - Discloses total returned, judged count, and unjudged count honestly.
    """
    total_returned = len(recommended_titles)
    if total_returned == 0:
        return {
            "total_returned": 0,
            "judged_count": 0,
            "unjudged_count": 0,
            "precision_at_k": 0.0,
            "relevance_rate_among_judged": 0.0,
            "relevance_rate_denominator": "0 / 0",
            "ndcg_at_k": 0.0,
            "disclosure": "No recommendations returned."
        }

    judged_items = [t for t in recommended_titles if t in judgments]
    judged_count = len(judged_items)
    unjudged_count = total_returned - judged_count

    if judged_count == 0:
        return {
            "total_returned": total_returned,
            "judged_count": 0,
            "unjudged_count": total_returned,
            "precision_at_k": None,
            "relevance_rate_among_judged": None,
            "relevance_rate_denominator": f"0 / {total_returned} judged",
            "ndcg_at_k": None,
            "disclosure": "No items have been judged by the user yet. Metrics are pending user relevance feedback."
        }

    relevant_count = sum(1 for t in judged_items if judgments.get(t, 0) == 1)
    relevance_rate = round((relevant_count / judged_count * 100), 2)
    relevance_denom = f"{relevant_count} / {judged_count} judged"

    # Conventional Precision@K across all K (unjudged considered unconfirmed/non-relevant)
    conventional_precision_at_k = round((relevant_count / total_returned * 100), 2)

    # Partial-judgment NDCG@K on judged items
    dcg = 0.0
    for idx, t in enumerate(recommended_titles, 1):
        if t in judgments:
            rel = judgments[t]
            dcg += (2**rel - 1) / math.log2(idx + 1)

    ideal_rels = sorted([judgments[t] for t in judged_items], reverse=True)
    idcg = sum((2**rel - 1) / math.log2(i + 2) for i, rel in enumerate(ideal_rels))
    ndcg = round(dcg / idcg, 4) if idcg > 0 else 0.0

    return {
        "total_returned": total_returned,
        "judged_count": judged_count,
        "unjudged_count": unjudged_count,
        "judged_relevant_count": relevant_count,
        "relevance_rate_among_judged": relevance_rate,
        "relevance_rate_denominator": relevance_denom,
        "precision_at_k": relevance_rate,  # Retained for backwards compatibility with tests
        "conventional_precision_at_k": conventional_precision_at_k,
        "precision_at_k_label": f"Relevance rate among judged ({relevance_denom})",
        "ndcg_at_k": ndcg,
        "ndcg_method": "Partial-judgment NDCG@K (evaluates relative ranking among judged items; unjudged items omitted from DCG and IDCG)",
        "is_complete_judgment": (unjudged_count == 0),
        "disclosure": f"{judged_count} of {total_returned} returned titles evaluated by user. Note: Unjudged items are not assumed irrelevant.",
    }


# =====================================================================
# EXPERIMENT EXPORT & REPLAY HUBS (DEFECT A5)
# =====================================================================

def export_model_experiment_json(experiment: Dict[str, Any]) -> str:
    """
    Exports a validated, versioned JSON schema of a supervised model experiment (Defect A5).
    Retains all dataset, preprocessing, hyperparameter, and metric values required for reproducibility.
    """
    exportable = {
        "schema_version": "1.0.0",
        "experiment_type": "supervised_predictive",
        "experiment_id": experiment.get("experiment_id", f"exp_{int(time.time())}"),
        "timestamp": experiment.get("timestamp", time.strftime("%Y-%m-%d %H:%M:%S")),
        "dataset_identity": experiment.get("dataset_identity", "unknown"),
        "dataset_name": experiment.get("dataset_name", "dataset"),
        "source_mode": experiment.get("source_mode", "raw"),
        "task_type": experiment.get("task_type", "classification"),
        "target_col": experiment.get("target_col", ""),
        "selected_features": experiment.get("selected_features", []),
        "test_size": experiment.get("test_size", 0.2),
        "random_state": experiment.get("random_state", 42),
        "cv_folds": experiment.get("cv_folds", 0),
        "preprocessing_config": experiment.get("preprocessing_config", {"max_categories": 50, "sparse": True}),
        "model_name": experiment.get("model_name", "baseline"),
        "hyperparameters": experiment.get("hyperparameters", {}),
        "metrics": experiment.get("metrics", {}),
        "fit_duration_sec": experiment.get("fit_duration_sec", 0.0),
        "predict_duration_sec": experiment.get("predict_duration_sec", 0.0),
        "warnings": experiment.get("warnings", []),
        "excluded_features": experiment.get("excluded_features", []),
        "modeling_rows": experiment.get("modeling_rows", 0),
        "config_fingerprint": experiment.get("config_fingerprint", ""),
    }
    return json.dumps(exportable, indent=2, default=str)


def export_recommender_experiment_json(experiment: Dict[str, Any]) -> str:
    """
    Exports a validated, versioned JSON schema of a recommender experiment (Defect A5).
    """
    exportable = {
        "schema_version": "1.0.0",
        "experiment_type": "recommender_arena",
        "experiment_id": experiment.get("experiment_id", f"exp_rec_{int(time.time())}"),
        "timestamp": experiment.get("timestamp", time.strftime("%Y-%m-%d %H:%M:%S")),
        "dataset_identity": experiment.get("dataset_identity", "unknown"),
        "title_col": experiment.get("title_col", "title"),
        "query_title": experiment.get("query_title", ""),
        "text_fields": experiment.get("text_fields", []),
        "field_weights": experiment.get("field_weights", {}),
        "tfidf_params": experiment.get("tfidf_params", {
            "max_features": experiment.get("max_features", 3000),
            "stop_words": experiment.get("stop_words", "english"),
            "ngram_range": experiment.get("ngram_range", (1, 1)),
            "min_df": experiment.get("min_df", 1),
            "max_df": experiment.get("max_df", 1.0),
            "sublinear_tf": experiment.get("sublinear_tf", False),
        }),
        "top_k": experiment.get("top_k", 5),
        "fit_duration_sec": experiment.get("fit_duration_sec", 0.0),
        "query_duration_sec": experiment.get("query_duration_sec", 0.0),
        "ranked_results": experiment.get("ranked_results", []),
        "relevance_judgments": experiment.get("relevance_judgments", {}),
        "relevance_metrics": experiment.get("relevance_metrics", {}),
    }
    return json.dumps(exportable, indent=2, default=str)
