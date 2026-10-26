"""Baseline recommendation models: Global Mean, Popularity, and User/Item Bias."""
import numpy as np
import pandas as pd


class GlobalMeanRecommender:
    """Predicts the overall mean rating for all items."""

    def __init__(self):
        self.mean_rating = 0.0

    def fit(self, train_ratings: pd.DataFrame):
        self.mean_rating = float(train_ratings["rating"].mean())
        return self

    def predict(self, user_ids: np.ndarray, movie_ids: np.ndarray) -> np.ndarray:
        return np.full(shape=len(user_ids), fill_value=self.mean_rating)


class PopularityRecommender:
    """Recommends the top-rated items with minimum vote thresholds (damped mean)."""

    def __init__(self, damping_factor: float = 25.0):
        self.damping_factor = damping_factor
        self.item_scores = None
        self.global_mean = 0.0

    def fit(self, train_ratings: pd.DataFrame):
        self.global_mean = float(train_ratings["rating"].mean())
        grouped = train_ratings.groupby("movie_id")["rating"].agg(["count", "mean"])
        
        # Damped mean / Bayesian average: (count * mean + C * global_mean) / (count + C)
        C = self.damping_factor
        grouped["damped_score"] = (grouped["count"] * grouped["mean"] + C * self.global_mean) / (grouped["count"] + C)
        self.item_scores = grouped["damped_score"].sort_values(ascending=False)
        return self

    def recommend(self, n: int = 10, exclude_seen: list | None = None) -> list:
        seen = set(exclude_seen or [])
        recs = [movie_id for movie_id in self.item_scores.index if movie_id not in seen]
        return recs[:n]
