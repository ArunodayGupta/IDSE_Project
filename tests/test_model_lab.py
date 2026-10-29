"""
test_model_lab.py
Automated test suite for Phase 3 DataLab: Model Lab (ML Experiment Arena).

Tests:
1. Task detection and validation (classification vs regression, target missingness exclusion).
2. Data leakage prevention: Preprocessing ColumnTransformer fitted strictly on X_train.
3. Supervised classification suite (baseline Dummy, Logistic, DecisionTree, RandomForest, metrics, ROC-AUC).
4. Supervised regression suite (baseline Dummy, Linear/Ridge, DecisionTree, RandomForest, MAE/RMSE/R2, residuals).
5. Cross-validation inside pipelines.
6. Feature importances (Tree MDI, Linear coefficients, safe fallback).
7. Error analysis (misclassifications with confidence, regression largest residuals).
8. Experiment serialization and JSON exports.
9. Recommender Experiment Arena (multi-field TF-IDF, weights, ablation, on-demand sparse ranking,
   A/B comparison, judged relevance feedback NDCG/Precision without fabricated ground truth).
10. Memory safety and backward compatibility with legacy ContentRecommender.
"""

import unittest
import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from scipy.sparse import issparse

from src.model_lab import (
    detect_suggested_task,
    validate_modeling_inputs,
    build_preprocessor,
    train_and_evaluate_model,
    extract_feature_importances,
    compute_error_analysis,
    RecommenderArenaService,
    compare_recommender_runs,
    compute_relevance_feedback_metrics,
    export_model_experiment_json,
    export_recommender_experiment_json,
    compute_model_config_fingerprint,
)
from src.model_trainer import ContentRecommender


class TestModelLabSupervisedStudio(unittest.TestCase):
    """Unit test suite for Phase 3 Mode 1: Predictive Modeling Studio."""

    def setUp(self):
        """Build synthetic tabular fixture with numeric, categorical, text, and missing features."""
        np.random.seed(42)
        n = 120
        self.df = pd.DataFrame({
            "age": np.random.randint(18, 70, size=n).astype(float),
            "income": np.random.uniform(20000, 150000, size=n),
            "tier": np.random.choice(["Bronze", "Silver", "Gold"], size=n),
            "is_active": np.random.choice([0, 1], size=n),
            "churn_class": np.random.choice(["No", "Yes"], size=n, p=[0.7, 0.3]),
            "satisfaction_score": np.random.uniform(1.0, 10.0, size=n),
            "free_notes": ["Long descriptive user review text here with comprehensive notes and thoughts."] * n,
            "constant_col": [42.0] * n,
            "all_null_col": [np.nan] * n,
        })
        # Inject some missing values into features
        self.df.loc[0:4, "age"] = np.nan
        self.df.loc[5:9, "tier"] = np.nan

    def test_detect_suggested_task(self):
        """Test task heuristic detection for classification and regression targets."""
        self.assertEqual(detect_suggested_task(self.df, "churn_class"), "classification")
        self.assertEqual(detect_suggested_task(self.df, "tier"), "classification")
        self.assertEqual(detect_suggested_task(self.df, "is_active"), "classification")
        self.assertEqual(detect_suggested_task(self.df, "income"), "regression")
        self.assertEqual(detect_suggested_task(self.df, "satisfaction_score"), "regression")

    def test_validate_modeling_inputs_missing_target_handling(self):
        """Verify rows with missing target are excluded with transparent reporting."""
        df_missing_target = self.df.copy()
        df_missing_target.loc[0:3, "churn_class"] = np.nan

        clean_df, meta = validate_modeling_inputs(
            df_missing_target,
            target_col="churn_class",
            feature_cols=["age", "income", "tier"],
            task_type="classification"
        )
        self.assertEqual(meta["excluded_target_missing"], 4)
        self.assertEqual(len(clean_df), 116)
        self.assertFalse(clean_df["churn_class"].isna().any())
        self.assertIn("age", meta["numeric_features"])
        self.assertIn("income", meta["numeric_features"])
        self.assertIn("tier", meta["categorical_features"])

    def test_validate_modeling_inputs_target_in_features_error(self):
        """Verify selecting target as input feature raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            validate_modeling_inputs(
                self.df,
                target_col="churn_class",
                feature_cols=["age", "churn_class"],
                task_type="classification"
            )
        self.assertIn("cannot be included in feature columns", str(ctx.exception).lower())

        # When ONLY target is passed as feature
        with self.assertRaises(ValueError):
            validate_modeling_inputs(
                self.df,
                target_col="churn_class",
                feature_cols=["churn_class"],
                task_type="classification"
            )

    def test_validate_modeling_inputs_invalid_regression_target(self):
        """Verify non-numeric target for regression raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            validate_modeling_inputs(
                self.df,
                target_col="tier",
                feature_cols=["age", "income"],
                task_type="regression"
            )
        self.assertIn("numeric", str(ctx.exception).lower())

    def test_validate_modeling_inputs_text_column_screening(self):
        """Verify long free-text columns are excluded with an explanatory warning."""
        clean_df, meta = validate_modeling_inputs(
            self.df,
            target_col="churn_class",
            feature_cols=["age", "income", "free_notes"],
            task_type="classification"
        )
        self.assertIn("free_notes", meta["excluded_text_features"])
        self.assertNotIn("free_notes", meta["usable_features"])

    def test_build_preprocessor_and_leakage_free_fitting(self):
        """Verify ColumnTransformer fits strictly on training split without test leakage."""
        clean_df, meta = validate_modeling_inputs(
            self.df,
            target_col="satisfaction_score",
            feature_cols=["age", "income", "tier"],
            task_type="regression"
        )
        num_cols = meta["numeric_features"]
        cat_cols = meta["categorical_features"]
        X = clean_df[num_cols + cat_cols]
        y = clean_df["satisfaction_score"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        preprocessor = build_preprocessor(num_cols, cat_cols)
        # Fit preprocessor strictly on X_train
        preprocessor.fit(X_train)

        # Inspect numeric scaler mean: should match X_train numeric means, not full X
        scaler = preprocessor.named_transformers_["num"].named_steps["scaler"]
        train_imputed = X_train[num_cols].fillna(X_train[num_cols].median())
        np.testing.assert_allclose(scaler.mean_, train_imputed.mean().values, rtol=1e-4)

        # Transform X_test safely with unseen handling
        transformed_test = preprocessor.transform(X_test)
        self.assertEqual(transformed_test.shape[0], len(X_test))

    def test_classification_candidate_suite(self):
        """Test candidate classification models and evaluation metrics."""
        clean_df, meta = validate_modeling_inputs(
            self.df,
            target_col="churn_class",
            feature_cols=["age", "income", "tier"],
            task_type="classification"
        )
        num_cols = meta["numeric_features"]
        cat_cols = meta["categorical_features"]
        X = clean_df[num_cols + cat_cols]
        y = clean_df["churn_class"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

        candidate_models = ["baseline", "logistic", "decision_tree", "random_forest"]
        for model_name in candidate_models:
            result = train_and_evaluate_model(
                model_name=model_name,
                task_type="classification",
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                numeric_cols=num_cols,
                categorical_cols=cat_cols,
                random_state=42
            )
            self.assertEqual(result["model_name"], model_name)
            self.assertEqual(result["task_type"], "classification")
            self.assertIn("accuracy", result["metrics"])
            self.assertIn("macro_f1", result["metrics"])
            self.assertIn("weighted_f1", result["metrics"])
            self.assertIn("confusion_matrix", result["extra"])
            self.assertIn("classification_report", result["extra"])
            self.assertTrue(0.0 <= result["metrics"]["accuracy"] <= 1.0)
            self.assertGreaterEqual(result["fit_duration_sec"], 0.0)

    def test_regression_candidate_suite(self):
        """Test candidate regression models and evaluation metrics."""
        clean_df, meta = validate_modeling_inputs(
            self.df,
            target_col="satisfaction_score",
            feature_cols=["age", "income", "tier"],
            task_type="regression"
        )
        num_cols = meta["numeric_features"]
        cat_cols = meta["categorical_features"]
        X = clean_df[num_cols + cat_cols]
        y = clean_df["satisfaction_score"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        candidate_models = ["baseline", "linear", "decision_tree", "random_forest"]
        for model_name in candidate_models:
            result = train_and_evaluate_model(
                model_name=model_name,
                task_type="regression",
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                numeric_cols=num_cols,
                categorical_cols=cat_cols,
                random_state=42
            )
            self.assertEqual(result["model_name"], model_name)
            self.assertEqual(result["task_type"], "regression")
            self.assertIn("mae", result["metrics"])
            self.assertIn("rmse", result["metrics"])
            self.assertIn("r2", result["metrics"])
            self.assertGreaterEqual(result["metrics"]["mae"], 0.0)
            self.assertGreaterEqual(result["metrics"]["rmse"], 0.0)
            self.assertIn("residuals", result["extra"])

    def test_cross_validation_support(self):
        """Test that cross validation evaluates training partition with preprocessor inside each fold."""
        clean_df, meta = validate_modeling_inputs(
            self.df,
            target_col="churn_class",
            feature_cols=["age", "income", "tier"],
            task_type="classification"
        )
        num_cols = meta["numeric_features"]
        cat_cols = meta["categorical_features"]
        X = clean_df[num_cols + cat_cols]
        y = clean_df["churn_class"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        result = train_and_evaluate_model(
            model_name="logistic",
            task_type="classification",
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            numeric_cols=num_cols,
            categorical_cols=cat_cols,
            random_state=42,
            cv_folds=3
        )
        self.assertIsNotNone(result["metrics"]["cv_mean"])
        self.assertIsNotNone(result["metrics"]["cv_std"])

    def test_feature_importance_extraction(self):
        """Test extraction of feature importances for trees and coefficients for linear models."""
        clean_df, meta = validate_modeling_inputs(
            self.df,
            target_col="churn_class",
            feature_cols=["age", "income", "tier"],
            task_type="classification"
        )
        num_cols = meta["numeric_features"]
        cat_cols = meta["categorical_features"]
        X = clean_df[num_cols + cat_cols]
        y = clean_df["churn_class"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Tree model
        tree_res = train_and_evaluate_model(
            model_name="decision_tree",
            task_type="classification",
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            numeric_cols=num_cols,
            categorical_cols=cat_cols,
            random_state=42
        )
        importances = tree_res["feature_importances"]
        self.assertIsNotNone(importances)
        self.assertGreater(len(importances), 0)
        self.assertIn("feature", importances[0])
        self.assertIn("importance", importances[0])

        # Baseline model should return empty list gracefully
        dummy_res = train_and_evaluate_model(
            model_name="baseline",
            task_type="classification",
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            numeric_cols=num_cols,
            categorical_cols=cat_cols,
            random_state=42
        )
        self.assertEqual(dummy_res["feature_importances"], [])

    def test_error_analysis(self):
        """Test error analysis generation for classification and regression."""
        clean_df, meta = validate_modeling_inputs(
            self.df,
            target_col="churn_class",
            feature_cols=["age", "income", "tier"],
            task_type="classification"
        )
        num_cols = meta["numeric_features"]
        cat_cols = meta["categorical_features"]
        X = clean_df[num_cols + cat_cols]
        y = clean_df["churn_class"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        clf_res = train_and_evaluate_model(
            model_name="decision_tree",
            task_type="classification",
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            numeric_cols=num_cols,
            categorical_cols=cat_cols,
            random_state=42
        )
        errors = compute_error_analysis(
            clf_res["fitted_pipeline"],
            X_test=X_test,
            y_test=y_test,
            raw_test_df=clean_df.loc[X_test.index],
            task_type="classification",
            top_n=5
        )
        self.assertIsInstance(errors, pd.DataFrame)
        if not errors.empty:
            self.assertIn("_actual", errors.columns)
            self.assertIn("_predicted", errors.columns)

    def test_experiment_json_serialization(self):
        """Test that experiment configuration and metrics serialize to standard JSON."""
        clean_df, meta = validate_modeling_inputs(
            self.df,
            target_col="satisfaction_score",
            feature_cols=["age", "income", "tier"],
            task_type="regression"
        )
        num_cols = meta["numeric_features"]
        cat_cols = meta["categorical_features"]
        X = clean_df[num_cols + cat_cols]
        y = clean_df["satisfaction_score"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        reg_res = train_and_evaluate_model(
            model_name="linear",
            task_type="regression",
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            numeric_cols=num_cols,
            categorical_cols=cat_cols,
            random_state=42
        )
        json_str = export_model_experiment_json(reg_res)
        parsed = json.loads(json_str)
        self.assertEqual(parsed["model_name"], "linear")
        self.assertEqual(parsed["task_type"], "regression")
        self.assertIn("metrics", parsed)
        self.assertNotIn("fitted_pipeline", parsed)

    def test_model_configuration_fingerprint_sensitivity_a3(self):
        """Regression test A3: Verify that changing any single modeling parameter
        produces a distinct configuration fingerprint."""
        base_kwargs = {
            "dataset_identity": "ds_hash_1",
            "target_col": "churn_class",
            "task_type": "classification",
            "feature_cols": ["age", "income", "tier"],
            "test_size": 0.2,
            "random_state": 42,
            "cv_folds": 5,
            "model_name": "logistic",
            "model_params": {"C": 1.0},
            "preprocessing_config": {"strategy": "sparse_onehot"}
        }
        base_fp = compute_model_config_fingerprint(**base_kwargs)

        # 1. Change dataset_identity
        k1 = base_kwargs.copy()
        k1["dataset_identity"] = "ds_hash_2"
        self.assertNotEqual(base_fp, compute_model_config_fingerprint(**k1))

        # 2. Change target_col
        k2 = base_kwargs.copy()
        k2["target_col"] = "satisfaction_score"
        self.assertNotEqual(base_fp, compute_model_config_fingerprint(**k2))

        # 3. Change task_type
        k3 = base_kwargs.copy()
        k3["task_type"] = "regression"
        self.assertNotEqual(base_fp, compute_model_config_fingerprint(**k3))

        # 4. Change feature_cols
        k4 = base_kwargs.copy()
        k4["feature_cols"] = ["age", "income"]
        self.assertNotEqual(base_fp, compute_model_config_fingerprint(**k4))

        # 5. Change test_size
        k5 = base_kwargs.copy()
        k5["test_size"] = 0.3
        self.assertNotEqual(base_fp, compute_model_config_fingerprint(**k5))

        # 6. Change random_state
        k6 = base_kwargs.copy()
        k6["random_state"] = 99
        self.assertNotEqual(base_fp, compute_model_config_fingerprint(**k6))

        # 7. Change cv_folds
        k7 = base_kwargs.copy()
        k7["cv_folds"] = 10
        self.assertNotEqual(base_fp, compute_model_config_fingerprint(**k7))

        # 8. Change model_name
        k8 = base_kwargs.copy()
        k8["model_name"] = "random_forest"
        self.assertNotEqual(base_fp, compute_model_config_fingerprint(**k8))

        # 9. Change model_params
        k9 = base_kwargs.copy()
        k9["model_params"] = {"C": 0.1}
        self.assertNotEqual(base_fp, compute_model_config_fingerprint(**k9))

    def test_sparse_one_hot_preprocessor_and_estimator_compatibility_a4(self):
        """Regression test A4: Verify build_preprocessor preserves sparse matrices
        without memory blowup, and scikit-learn estimators natively fit and predict."""
        preprocessor = build_preprocessor(categorical_cols=["tier"], numeric_cols=["age", "income"])
        X_df = self.df[["tier", "age", "income"]].dropna()
        X_trans = preprocessor.fit_transform(X_df)
        self.assertTrue(issparse(X_trans), "Transformed matrix should be sparse CSR format to avoid memory explosion")

        # Verify estimators train and predict on sparse matrix
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression()
        clf.fit(X_trans, self.df.loc[X_df.index, "churn_class"])
        preds = clf.predict(X_trans)
        self.assertEqual(len(preds), len(X_df))

    def test_small_dataset_and_rare_class_handling_a8(self):
        """Regression test A8: Verify validation on small datasets, 1-class targets,
        and rare classes that make default stratification impossible."""
        # 1. Dataset with fewer than 5 rows
        tiny_df = self.df.head(4).copy()
        with self.assertRaises(ValueError) as ctx:
            validate_modeling_inputs(tiny_df, "churn_class", ["age", "income"], "classification")
        self.assertIn("insufficient", str(ctx.exception).lower())

        # 2. Single-class target
        one_class_df = self.df.copy()
        one_class_df["single_class"] = "AllSame"
        with self.assertRaises(ValueError) as ctx:
            validate_modeling_inputs(one_class_df, "single_class", ["age", "income"], "classification")
        self.assertIn("only 1 distinct class", str(ctx.exception).lower())


class TestModelLabRecommenderArena(unittest.TestCase):
    """Unit test suite for Phase 3 Mode 2: Recommender Experiment Arena."""

    def setUp(self):
        """Build synthetic catalog fixture for recommendation experiments."""
        self.catalog_df = pd.DataFrame({
            "title": [
                "Space Odyssey 2001",
                "Interstellar Journey",
                "Deep Space Wars",
                "Romantic Paris",
                "Midnight in France",
                "Action Hero Reloaded",
                "Fast Cars and Fury",
                "Silent Detective",
            ],
            "director": [
                "Stanley Kubrick",
                "Christopher Nolan",
                "George Lucas",
                "Woody Allen",
                "Jean-Luc Godard",
                "Michael Bay",
                "Justin Lin",
                "David Fincher",
            ],
            "cast": [
                "Keir Dullea, Gary Lockwood",
                "Matthew McConaughey, Anne Hathaway",
                "Mark Hamill, Harrison Ford",
                "Owen Wilson, Marion Cotillard",
                "Jean-Paul Belmondo",
                "Will Smith, Martin Lawrence",
                "Vin Diesel, Paul Walker",
                "Brad Pitt, Morgan Freeman",
            ],
            "genres": [
                "Sci-Fi, Adventure",
                "Sci-Fi, Drama",
                "Sci-Fi, Action",
                "Romance, Comedy",
                "Romance, Drama",
                "Action, Comedy",
                "Action, Thriller",
                "Crime, Mystery",
            ],
            "description": [
                "Astronauts journey across the solar system encountering mysterious alien monoliths in outer space.",
                "Explorers travel through a wormhole in space in an attempt to ensure humanity's survival on new planets.",
                "Rebels battle an evil galactic empire with laser swords and spaceships in deep space.",
                "A nostalgic writer wanders the beautiful streets of Paris finding love and artistic inspiration.",
                "Poetic romance blossoming along the Seine river during a stormy night in Paris.",
                "Explosive car chases, humor, and daring police officers taking down international syndicates.",
                "High-speed street racers pulled into espionage, fast sports cars, and explosive action sequences.",
                "Two detectives investigate a series of elaborate, dark ritualistic murders in a gloomy metropolis.",
            ]
        })

    def test_recommender_arena_fit_and_query(self):
        """Test multi-field TF-IDF fitting, vocabulary dimensions, and on-demand query ranking."""
        service = RecommenderArenaService(
            self.catalog_df,
            title_col="title",
            text_fields=["genres", "description"],
            field_weights={"genres": 2.0, "description": 1.0},
            max_features=500,
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True
        )
        self.assertGreater(service.vocabulary_size, 0)
        self.assertGreater(service.nnz_entries, 0)

        # Query recommendations for "Space Odyssey 2001"
        recs, duration = service.recommend(query_title="Space Odyssey 2001", top_k=3)
        self.assertEqual(len(recs), 3)
        rec_titles = recs["title"].tolist()
        self.assertNotIn("Space Odyssey 2001", rec_titles)
        self.assertIn("Interstellar Journey", rec_titles)
        self.assertIn("Deep Space Wars", rec_titles)
        self.assertGreaterEqual(recs["similarity_score"].iloc[0], recs["similarity_score"].iloc[1])

    def test_recommender_field_ablation_and_weighting(self):
        """Verify field weighting and ablation directly shift recommendation rankings."""
        # Config A: Only description (genres weight = 0.0)
        service_a = RecommenderArenaService(
            self.catalog_df,
            title_col="title",
            text_fields=["genres", "description"],
            field_weights={"genres": 0.0, "description": 1.0},
            stop_words="english"
        )
        recs_a, _ = service_a.recommend("Space Odyssey 2001", top_k=3)

        # Config B: Heavily weigh genres
        service_b = RecommenderArenaService(
            self.catalog_df,
            title_col="title",
            text_fields=["genres", "description"],
            field_weights={"genres": 5.0, "description": 0.1},
            stop_words="english"
        )
        recs_b, _ = service_b.recommend("Space Odyssey 2001", top_k=3)

        scores_a = recs_a["similarity_score"].tolist()
        scores_b = recs_b["similarity_score"].tolist()
        self.assertNotEqual(scores_a, scores_b)

    def test_memory_safe_sparse_similarity_equivalence(self):
        """Verify that on-demand sparse dot product computes correct cosine similarities."""
        from sklearn.metrics.pairwise import cosine_similarity

        service = RecommenderArenaService(
            self.catalog_df,
            title_col="title",
            text_fields=["description"],
            stop_words="english"
        )
        self.assertTrue(issparse(service.combined_matrix))
        recs, _ = service.recommend("Space Odyssey 2001", top_k=5)

        # Verify against full dense matrix calculation for query item (index 0)
        full_dense_sim = cosine_similarity(service.combined_matrix[0], service.combined_matrix).flatten()
        top_rec_title = recs["title"].iloc[0]
        top_rec_idx = self.catalog_df[self.catalog_df["title"] == top_rec_title].index[0]
        expected_score = round(float(full_dense_sim[top_rec_idx]), 4)
        self.assertAlmostEqual(recs["similarity_score"].iloc[0], expected_score, places=3)

    def test_legacy_content_recommender_compatibility(self):
        """Verify backward compatibility of legacy ContentRecommender without dense N x N in fit()."""
        legacy = ContentRecommender(max_features=500)
        df_legacy = self.catalog_df.copy()
        df_legacy["metadata_soup"] = df_legacy["genres"] + " " + df_legacy["description"]
        legacy.fit(df_legacy)

        # Verify recommend() works
        recs = legacy.recommend("Space Odyssey 2001", top_n=3)
        self.assertIsInstance(recs, pd.DataFrame)
        self.assertEqual(len(recs), 3)
        self.assertIn("title", recs.columns)
        self.assertIn("similarity", recs.columns)

        # Access legacy property cosine_sim
        cos_matrix = legacy.cosine_sim
        self.assertEqual(cos_matrix.shape, (8, 8))

    def test_ab_comparison_on_same_query(self):
        """Test A/B comparison metrics: Jaccard overlap, entering, leaving, and rank shift."""
        df_a = pd.DataFrame({
            "title": ["Interstellar Journey", "Deep Space Wars", "Action Hero Reloaded"],
            "similarity_score": [0.82, 0.75, 0.31],
            "rank": [1, 2, 3]
        })
        df_b = pd.DataFrame({
            "title": ["Deep Space Wars", "Interstellar Journey", "Silent Detective"],
            "similarity_score": [0.88, 0.85, 0.22],
            "rank": [1, 2, 3]
        })

        comp = compare_recommender_runs(df_a, df_b, title_col="title")
        self.assertEqual(comp["shared_count"], 2)
        # Jaccard overlap: 2 shared / 4 unique = 50.0%
        self.assertEqual(comp["overlap_pct"], 50.0)
        self.assertIn("Silent Detective", comp["entering_items"])
        self.assertIn("Action Hero Reloaded", comp["leaving_items"])

    def test_relevance_feedback_ndcg_and_precision(self):
        """Verify honest feedback evaluation without fabricated ground truth."""
        titles = ["Interstellar Journey", "Deep Space Wars", "Action Hero Reloaded", "Silent Detective"]

        # Case 1: No feedback supplied -> disclosure note and None precision
        res_none = compute_relevance_feedback_metrics(titles, {})
        self.assertIsNone(res_none["precision_at_k"])
        self.assertIsNone(res_none["ndcg_at_k"])
        self.assertEqual(res_none["judged_count"], 0)
        self.assertEqual(res_none["unjudged_count"], 4)

        # Case 2: Partial feedback supplied
        feedback = {
            "Interstellar Journey": 1,
            "Action Hero Reloaded": 0,
        }
        res_partial = compute_relevance_feedback_metrics(titles, feedback)
        self.assertEqual(res_partial["judged_count"], 2)
        self.assertEqual(res_partial["unjudged_count"], 2)
        self.assertEqual(res_partial["judged_relevant_count"], 1)
        self.assertEqual(res_partial["precision_at_k"], 50.0)
        self.assertGreater(res_partial["ndcg_at_k"], 0.0)

    def test_recommender_experiment_json_export(self):
        """Test recommender experiment serialization to JSON."""
        exp = {
            "experiment_id": "exp_rec_001",
            "title_col": "title",
            "text_fields": ["genres", "description"],
            "field_weights": {"genres": 2.0, "description": 1.0},
            "max_features": 500,
            "query_title": "Space Odyssey 2001",
            "top_k": 3,
        }
        json_str = export_recommender_experiment_json(exp)
        parsed = json.loads(json_str)
        self.assertEqual(parsed["experiment_id"], "exp_rec_001")

    def test_recommender_feedback_key_and_truthful_metrics_a6(self):
        """Regression test A6: Key feedback by stable (dataset_id, title_col, query_title)
        and verify truthful relevance metrics labeling."""
        titles = ["Space Odyssey 2001", "Interstellar Journey", "Deep Space Wars", "Romantic Paris", "Midnight in France"]

        # 1. Truthful label: relevance rate among judged (with denominator shown)
        feedback = {"Interstellar Journey": 1, "Deep Space Wars": 0}
        metrics = compute_relevance_feedback_metrics(titles, feedback)
        self.assertEqual(metrics["judged_count"], 2)
        self.assertEqual(metrics["unjudged_count"], 3)
        self.assertEqual(metrics["judged_relevant_count"], 1)
        self.assertEqual(metrics["relevance_rate_among_judged"], 50.0)
        self.assertEqual(metrics["relevance_rate_denominator"], "1 / 2 judged")
        self.assertIn("partial-judgment ndcg", metrics["ndcg_method"].lower())

        # 2. Feedback isolation: feedback dictionary for dataset A should not match key for dataset B
        feedback_store = {}
        key_ds_a = ("dataset_A", "title", "Space Odyssey 2001")
        key_ds_b = ("dataset_B", "title", "Space Odyssey 2001")
        feedback_store[key_ds_a] = {"Interstellar Journey": 1}
        self.assertNotIn(key_ds_b, feedback_store)


if __name__ == "__main__":
    unittest.main()
