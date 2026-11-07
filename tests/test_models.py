"""
test_models.py
Unit tests for model training, metrics calculation, single prediction, and Kaggle submission generation.
"""

import unittest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_loader import load_tmdb_train_data
from src.feature_engineering import engineer_movie_features, prepare_single_movie_features
from src.model_trainer import (
    train_movie_regressor,
    predict_single_movie,
    generate_kaggle_submission,
    compute_rmsle,
    MODEL_REGISTRY,
)


class TestModelPipeline(unittest.TestCase):
    """Verifies predictive modeling algorithms, evaluation metrics, and Kaggle submissions."""

    @classmethod
    def setUpClass(cls):
        """Loads a subset of TMDB training data with corrected split workflow for rapid test execution."""
        raw_df, _ = load_tmdb_train_data()
        cls.raw_df = raw_df
        # Use first 300 rows for rapid test execution
        labeled_subset = raw_df.head(300).copy()
        raw_train, raw_val = train_test_split(labeled_subset, test_size=0.25, random_state=42)

        # Correct workflow: engineer on raw_train, reuse stats for raw_val
        cls.X_train, cls.y_train, cls.stats = engineer_movie_features(raw_train, is_train=True, stats=None)
        cls.X_val, cls.y_val, _ = engineer_movie_features(raw_val, is_train=True, stats=cls.stats)

    def test_compute_rmsle(self):
        """Verifies RMSLE calculation formula against known sample values and edge cases."""
        y_true = np.array([1000.0, 50000.0, 1000000.0])
        y_pred = np.array([1100.0, 48000.0, 1050000.0])
        rmsle = compute_rmsle(y_true, y_pred)
        self.assertGreater(rmsle, 0.0)
        self.assertLess(rmsle, 0.5)

        # Zero error test
        self.assertAlmostEqual(compute_rmsle(y_true, y_true), 0.0, places=5)

        # Empty array test
        self.assertEqual(compute_rmsle(np.array([]), np.array([])), 0.0)

    def test_all_models_train_and_predict(self):
        """Verifies all four registered algorithms train, evaluate, and yield non-negative predictions."""
        for model_key in MODEL_REGISTRY.keys():
            res = train_movie_regressor(
                model_key=model_key,
                hyperparams={"n_estimators": 20, "max_depth": 5, "max_iter": 20},
                X_train=self.X_train,
                y_train=self.y_train,
                X_val=self.X_val,
                y_val=self.y_val,
                random_state=42
            )

            self.assertIn("rmsle", res)
            self.assertIn("mae", res)
            self.assertIn("r2", res)
            self.assertGreater(res["rmsle"], 0.0)
            self.assertGreaterEqual(res["mae"], 0.0)

            # Check non-negative predictions
            self.assertTrue(np.all(res["y_val_pred"] >= 0), f"Model {model_key} produced negative predictions.")

            # HistGradientBoosting should return empty or valid feature_importances without crashing
            self.assertIsInstance(res["feature_importances"], pd.DataFrame)

    def test_single_prediction_zero_budget(self):
        """Verifies single-movie prediction works correctly even when production budget is zero."""
        res = train_movie_regressor(
            model_key="random_forest",
            hyperparams={"n_estimators": 30, "max_depth": 6},
            X_train=self.X_train,
            y_train=self.y_train,
            X_val=self.X_val,
            y_val=self.y_val,
            random_state=42
        )

        # Single movie with 0 budget
        single_zero_budget = prepare_single_movie_features({
            "budget": 0,
            "runtime": 100,
            "genres": ["Drama"],
            "original_language": "en",
            "production_country": "US",
        }, self.stats)

        pred = predict_single_movie(res["pipeline"], single_zero_budget)
        self.assertIsInstance(pred, float)
        self.assertGreaterEqual(pred, 0.0)

    def test_kaggle_submission_structure(self):
        """Verifies Kaggle submission maintains required columns, preserves IDs, and has non-negative revenues."""
        test_slice = self.raw_df.head(15).copy().drop(columns=["revenue"])

        sub = generate_kaggle_submission(
            model_key="dummy",
            hyperparams={},
            raw_train_df=self.raw_df.head(200),
            raw_test_df=test_slice,
            random_state=42
        )

        self.assertEqual(list(sub.columns), ["id", "revenue"])
        self.assertEqual(len(sub), len(test_slice))
        self.assertEqual(list(sub["id"]), list(test_slice["id"]))
        self.assertTrue(np.all(sub["revenue"] >= 0))
        self.assertTrue(np.all(np.isfinite(sub["revenue"])))


if __name__ == "__main__":
    unittest.main()
