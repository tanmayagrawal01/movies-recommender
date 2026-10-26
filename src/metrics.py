"""Evaluation metrics for rating prediction and top-K recommendation ranking."""
import numpy as np
import pandas as pd


# --- Rating Prediction Metrics ---

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


# --- Top-K Ranking Metrics ---

def precision_at_k(actual: list, predicted: list, k: int = 10) -> float:
    """Precision at K recommendations."""
    if not actual:
        return 0.0
    pred_k = predicted[:k]
    hits = len(set(pred_k) & set(actual))
    return hits / float(k)


def recall_at_k(actual: list, predicted: list, k: int = 10) -> float:
    """Recall at K recommendations."""
    if not actual:
        return 0.0
    pred_k = predicted[:k]
    hits = len(set(pred_k) & set(actual))
    return hits / float(len(actual))


def ndcg_at_k(actual: list, predicted: list, k: int = 10) -> float:
    """Normalized Discounted Cumulative Gain at K."""
    if not actual:
        return 0.0
    pred_k = predicted[:k]
    dcg = 0.0
    for i, item in enumerate(pred_k):
        if item in actual:
            dcg += 1.0 / np.log2(i + 2)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(actual), k)))
    if idcg == 0.0:
        return 0.0
    return float(dcg / idcg)


def mean_reciprocal_rank(actual: list, predicted: list) -> float:
    """Mean Reciprocal Rank (MRR)."""
    for i, item in enumerate(predicted):
        if item in actual:
            return 1.0 / (i + 1)
    return 0.0
