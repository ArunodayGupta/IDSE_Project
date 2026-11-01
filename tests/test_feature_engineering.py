"""
test_feature_engineering.py
Unit tests for pre-release feature extraction, leakage prevention, and prediction input preparation.
"""

import unittest
import pandas as pd
import numpy as np

from src.feature_engineering import (
    engineer_movie_features,
    prepare_single_movie_features,
)


class TestFeatureEngineering(unittest.TestCase):
    """Verifies feature extraction, leakage prevention, and prediction pipeline consistency."""

    def setUp(self):
        """Construct synthetic test data with edge cases."""
        self.sample_df = pd.DataFrame({
            "id": [1, 2, 3],
            "title": ["Movie A", "Movie B", "Movie C"],
            "budget": [10000000, 0, 50000000],  # Includes 0 budget edge case
            "revenue": [25000000, 5000000, 120000000],
            "runtime": [110.0, 0.0, np.nan],  # Includes 0 and NaN runtime
            "release_date": ["5/23/83", "2/20/15", "8/6/04"],
            "genres": [
                "[{'id': 28, 'name': 'Action'}, {'id': 53, 'name': 'Thriller'}]",
                "[{'id': 35, 'name': 'Comedy'}]",
                "[]"
            ],
            "original_language": ["en", "fr", "unseen_lang"],
            "production_companies": [
                "[{'name': 'Warner Bros.', 'id': 6194}]",
                "[{'name': 'Indie Studio', 'id': 999}]",
                None
            ],
            "production_countries": [
                "[{'iso_3166_1': 'US', 'name': 'United States'}]",
                "[{'iso_3166_1': 'FR', 'name': 'France'}]",
                "[]"
            ],
            "spoken_languages": [
                "[{'iso_639_1': 'en', 'name': 'English'}]",
                "[{'iso_639_1': 'fr', 'name': 'Français'}]",
                None
            ],
            "cast": ["[{'name': 'Actor 1'}]", "[]", None],
            "crew": ["[{'job': 'Director', 'name': 'Director A'}]", None, "[]"],
            "belongs_to_collection": ["[{'name': 'Franchise 1'}]", None, ""],
            "homepage": ["http://example.com", None, None],
            "tagline": ["An epic adventure", None, "Classic"],
            "popularity": [15.5, 8.2, 22.1],  # Leaky post-release attribute
            "vote_average": [7.2, 6.1, 8.0],   # Leaky post-release attribute
            "vote_count": [1200, 450, 3100],   # Leaky post-release attribute
        })

    def test_target_leakage_prevention(self):
        """Strictly ensures post-release attributes and identifiers are never present in X."""
        X, y, stats = engineer_movie_features(self.sample_df, is_train=True)

        leaky_columns = [
            "revenue",
            "popularity",
            "vote_average",
            "vote_count",
            "id",
            "title",
            "homepage",
            "tagline",
            "overview"
        ]

        for col in leaky_columns:
            self.assertNotIn(col, X.columns, f"Leaky column '{col}' found in feature matrix X!")

        self.assertIsNotNone(y)
        self.assertEqual(len(X), len(y))

    def test_zero_imputation_and_transforms(self):
        """Verifies zero budget and runtime values are handled without NaNs."""
        X, y, stats = engineer_movie_features(self.sample_df, is_train=True)

        self.assertEqual(X["budget_is_zero"].iloc[1], 1)
        self.assertGreater(X["budget"].iloc[1], 0)  # Imputed with median
        self.assertGreater(X["runtime"].iloc[1], 0) # Imputed with median
        self.assertEqual(X.isna().sum().sum(), 0, "Feature matrix contains unexpected NaNs.")

    def test_single_prediction_features_consistency(self):
        """Verifies interactive prediction constructor produces exact matching columns."""
        X, y, stats = engineer_movie_features(self.sample_df, is_train=True)

        input_dict = {
            "budget": 35000000,
            "runtime": 105,
            "release_date": pd.to_datetime("2025-06-15"),
            "genres": ["Action", "Thriller"],
            "original_language": "en",
            "production_country": "US",
            "company_count": 2,
            "is_major_studio": True,
            "belongs_to_collection": False,
            "cast_size": 15,
            "crew_size": 25,
        }

        single_df = prepare_single_movie_features(input_dict, stats)

        self.assertEqual(list(single_df.columns), list(X.columns), "Single feature columns do not match X columns.")
        self.assertEqual(len(single_df), 1)
        self.assertEqual(single_df.isna().sum().sum(), 0)


if __name__ == "__main__":
    unittest.main()
