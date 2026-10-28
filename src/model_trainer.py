"""
model_trainer.py
NLP Content-Based Recommendation Engine using TF-IDF Vectorization and Cosine Similarity.

Fits on Netflix metadata soup features (title, director, cast, genres, description)
and calculates a cosine similarity matrix to recommend the most similar titles.
"""

from typing import List, Optional, Union, Dict, Any
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentRecommender:
    """
    NLP Content-Based Recommender Engine for Netflix titles.

    Utilizes TF-IDF vectorization with up to 5,000 features on the metadata_soup
    column (title, director, cast, genres, description) and computes pairwise
    cosine similarity to recommend top-N similar titles.
    """

    def __init__(self, stop_words: str = "english", max_features: int = 5000):
        self.stop_words = stop_words
        self.max_features = max_features
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self._cosine_sim: Optional[np.ndarray] = None
        self.df: Optional[pd.DataFrame] = None
        self.title_to_index: Dict[str, int] = {}
        self.lower_title_to_index: Dict[str, int] = {}
        self.is_fitted: bool = False

    @property
    def cosine_sim(self) -> Optional[np.ndarray]:
        """
        Backward-compatible property providing the full cosine similarity matrix
        only on explicit demand, avoiding unnecessary N-by-N memory allocation.
        """
        if self._cosine_sim is not None:
            return self._cosine_sim
        if self.is_fitted and self.tfidf_matrix is not None:
            self._cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
            return self._cosine_sim
        return None

    @cosine_sim.setter
    def cosine_sim(self, value: Optional[np.ndarray]) -> None:
        self._cosine_sim = value

    def fit(self, df: pd.DataFrame) -> "ContentRecommender":
        """
        Fits the TF-IDF vectorizer on the metadata_soup column and retains
        the sparse document-term matrix for memory-safe query-time similarity.

        Args:
            df: DataFrame containing at least 'title' and 'metadata_soup'
                (or raw catalog columns from which metadata_soup can be built).

        Returns:
            self: The fitted ContentRecommender instance.
        """
        if df is None or df.empty:
            self.df = pd.DataFrame()
            self.is_fitted = False
            return self

        # Reset index to guarantee 0-based integer alignment with similarity matrix
        self.df = df.reset_index(drop=True).copy()

        # If metadata_soup is missing, dynamically build it
        if "metadata_soup" not in self.df.columns:
            from .feature_engineer import create_metadata_soup
            self.df["metadata_soup"] = create_metadata_soup(self.df)

        corpus = self.df["metadata_soup"].fillna("").astype(str)

        # 1. Instantiate and fit TF-IDF Vectorizer
        self.vectorizer = TfidfVectorizer(
            stop_words=self.stop_words,
            max_features=self.max_features
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

        # 2. Reset full matrix cache to conserve memory (computed on-demand per query)
        self._cosine_sim = None

        # 3. Build fast O(1) title lookup index maps
        self.title_to_index = {}
        self.lower_title_to_index = {}
        for idx, title in enumerate(self.df["title"].astype(str)):
            cleaned = title.strip()
            if cleaned not in self.title_to_index:
                self.title_to_index[cleaned] = idx
            lowered = cleaned.lower()
            if lowered not in self.lower_title_to_index:
                self.lower_title_to_index[lowered] = idx

        self.is_fitted = True
        return self

    def recommend(
        self,
        title: str,
        top_n: int = 5,
        return_df: bool = True
    ) -> Union[pd.DataFrame, List[str]]:
        """
        Recommends the top_n most similar titles to the given title based on
        memory-safe sparse cosine similarity of metadata soup vectors.

        Args:
            title: Title string of the target movie or TV show.
            top_n: Number of recommendations to return (default: 5).
            return_df: If True, returns a DataFrame with similarity_score and metadata;
                       if False, returns a list of title strings.

        Returns:
            pd.DataFrame or List[str] with top_n recommended titles.
        """
        if not self.is_fitted or self.df is None or self.tfidf_matrix is None or len(self.df) == 0:
            return pd.DataFrame() if return_df else []

        # Find index of target title
        idx = None
        title_str = str(title).strip()
        if title_str in self.title_to_index:
            idx = self.title_to_index[title_str]
        elif title_str.lower() in self.lower_title_to_index:
            idx = self.lower_title_to_index[title_str.lower()]
        else:
            # Partial search fallback
            matches = self.df[self.df["title"].astype(str).str.lower().str.contains(title_str.lower(), regex=False, na=False)]
            if not matches.empty:
                idx = matches.index[0]

        if idx is None:
            return pd.DataFrame() if return_df else []

        # Memory-safe query vector dot product against catalog sparse matrix
        query_vec = self.tfidf_matrix[idx]
        sim_scores_array = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        sim_scores = list(enumerate(sim_scores_array))

        # Sort descending by similarity score
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        target_title_norm = str(self.df.iloc[idx]["title"]).strip().lower()

        # Exclude self and duplicate titles
        candidates = []
        for i, score in sim_scores:
            if i == idx:
                continue
            cand_title = str(self.df.iloc[i]["title"]).strip().lower()
            if cand_title == target_title_norm:
                continue
            candidates.append((i, float(score)))
            if len(candidates) >= top_n:
                break

        # Fallback if catalog is tiny
        if not candidates:
            candidates = [(i, float(score)) for i, score in sim_scores if i != idx][:top_n]

        indices = [i for i, _ in candidates]
        scores = [round(float(score), 4) for _, score in candidates]

        result_df = self.df.iloc[indices].copy()
        result_df["similarity_score"] = scores
        result_df["similarity"] = scores

        if not return_df:
            return result_df["title"].tolist()

        return result_df


if __name__ == "__main__":
    from data_generator import ensure_data_exists
    from data_cleaner import clean_netflix_data
    from feature_engineer import engineer_features

    csv_path = ensure_data_exists()
    raw = pd.read_csv(csv_path)
    clean, _ = clean_netflix_data(raw)
    feat = engineer_features(clean)

    recommender = ContentRecommender()
    recommender.fit(feat)

    sample_title = feat["title"].iloc[0]
    print(f"\n--- Recommendations for '{sample_title}' ---")
    recs = recommender.recommend(sample_title, top_n=5)
    print(recs[["title", "type", "rating", "similarity_score"]])
