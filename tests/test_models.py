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
        """Loads a subset of TMDB training data for fast test execution."""
        raw_df, _ = load_tmdb_train_data()
        cls.raw_df = raw_df
        # Use first 300 rows for rapid test execution
        cls.sample_df = raw_df.head(300).copy()
        cls.X, cls.y, cls.stats = engineer_movie_features(cls.sample_df, is_train=True)
        cls.X_train, cls.X_val, cls.y_train, cls.y_val = train_test_split(
            cls.X, cls.y, test_size=0.25, random_state=42
        )

    def test_compute_rmsle(self):
        """Verifies RMSLE calculation formula."""
        y_true = np.array([1000.0, 50000.0, 1000000.0])
        y_pred = np.array([1100.0, 48000.0, 1050000.0])
        rmsle = compute_rmsle(y_true, y_pred)
        self.assertGreater(rmsle, 0.0)
        self.assertLess(rmsle, 0.5)

        # Zero error test
        self.assertAlmostEqual(compute_rmsle(y_true, y_true), 0.0, places=5)

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

    def test_single_prediction_non_negative(self):
        """Verifies interactive single-movie prediction produces a valid positive float."""
        res = train_movie_regressor(
            model_key="random_forest",
            hyperparams={"n_estimators": 30, "max_depth": 6},
            X_train=self.X_train,
            y_train=self.y_train,
            X_val=self.X_val,
            y_val=self.y_val,
            random_state=42
        )

        single_feature_df = prepare_single_movie_features({
            "budget": 50000000,
            "runtime": 120,
            "genres": ["Action", "Adventure"],
            "original_language": "en",
            "production_country": "US",
            "company_count": 3,
            "is_major_studio": True,
            "belongs_to_collection": True,
            "cast_size": 25,
            "crew_size": 40,
        }, self.stats)

        pred = predict_single_movie(res["pipeline"], single_feature_df)
        self.assertIsInstance(pred, float)
        self.assertGreater(pred, 0.0)

    def test_kaggle_submission_structure(self):
        """Verifies Kaggle submission maintains required columns, preserves IDs, and has positive revenues."""
        test_slice = self.sample_df.head(15).copy()
        test_slice = test_slice.drop(columns=["revenue"])

        sub = generate_kaggle_submission(
            model_key="dummy",
            hyperparams={},
            raw_train_df=self.sample_df,
            raw_test_df=test_slice,
            random_state=42
        )

        self.assertEqual(list(sub.columns), ["id", "revenue"])
        self.assertEqual(len(sub), len(test_slice))
        self.assertEqual(list(sub["id"]), list(test_slice["id"]))
        self.assertTrue(np.all(sub["revenue"] >= 0))


if __name__ == "__main__":
    unittest.main()
