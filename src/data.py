"""Data loading, cleaning and train/test splitting for MovieLens 1M."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "ml-1m"

OCCUPATIONS = {
    0: "other", 1: "academic/educator", 2: "artist", 3: "clerical/admin",
    4: "college/grad student", 5: "customer service", 6: "doctor/health care",
    7: "executive/managerial", 8: "farmer", 9: "homemaker", 10: "K-12 student",
    11: "lawyer", 12: "programmer", 13: "retired", 14: "sales/marketing",
    15: "scientist", 16: "self-employed", 17: "technician/engineer",
    18: "tradesman/craftsman", 19: "unemployed", 20: "writer",
}


def load_ratings(path: Path = RAW) -> pd.DataFrame:
    df = pd.read_csv(
        path / "ratings.dat", sep="::", engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"],
    )
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
    return df


def load_movies(path: Path = RAW) -> pd.DataFrame:
    df = pd.read_csv(
        path / "movies.dat", sep="::", engine="python",
        names=["movie_id", "title", "genres"], encoding="latin-1",
    )
    df["year"] = df["title"].str.extract(r"\((\d{4})\)\s*$").astype(float)
    df["genre_list"] = df["genres"].str.split("|")
    return df


def load_users(path: Path = RAW) -> pd.DataFrame:
    df = pd.read_csv(
        path / "users.dat", sep="::", engine="python",
        names=["user_id", "gender", "age", "occupation", "zip"],
    )
    df["occupation_name"] = df["occupation"].map(OCCUPATIONS)
    return df


def leave_last_n_out(ratings: pd.DataFrame, n_test: int = 5, n_val: int = 0):
    """Per-user temporal split: each user's latest `n_test` ratings go to test,
    the `n_val` before them to validation, the rest to train.

    This mimics real life (predict the future from the past) and avoids the
    data leakage of a plain random split.
    """
    df = ratings.sort_values(["user_id", "timestamp"]).copy()
    # rank 0 = most recent rating of that user
    df["rank"] = df.groupby("user_id")["timestamp"].rank(method="first", ascending=False) - 1
    test = df[df["rank"] < n_test]
    val = df[(df["rank"] >= n_test) & (df["rank"] < n_test + n_val)]
    train = df[df["rank"] >= n_test + n_val]
    drop = ["rank"]
    return train.drop(columns=drop), val.drop(columns=drop), test.drop(columns=drop)


def random_split(ratings: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    rng = np.random.default_rng(seed)
    mask = rng.random(len(ratings)) < test_size
    return ratings[~mask].copy(), ratings[mask].copy()


def to_matrix(ratings: pd.DataFrame, n_users: int | None = None, n_items: int | None = None):
    """Sparse user-item rating matrix (CSR) with 0-based contiguous indices.

    Returns (matrix, user_index, item_index) where the indices map original ids
    to row / column positions.
    """
    users = np.sort(ratings["user_id"].unique())
    items = np.sort(ratings["movie_id"].unique())
    u_idx = {u: i for i, u in enumerate(users)}
    i_idx = {m: i for i, m in enumerate(items)}
    rows = ratings["user_id"].map(u_idx).to_numpy()
    cols = ratings["movie_id"].map(i_idx).to_numpy()
    mat = sparse.csr_matrix(
        (ratings["rating"].to_numpy(dtype=np.float32), (rows, cols)),
        shape=(n_users or len(users), n_items or len(items)),
    )
    return mat, u_idx, i_idx
