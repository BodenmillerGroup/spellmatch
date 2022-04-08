from functools import partial
from typing import Callable

import numpy as np


def precision(
    scores_arr: np.ndarray,
    assignment_arr_pred: np.ndarray,
    assignment_arr_true: np.ndarray,
) -> float:
    # of all predicted matches, what fraction is correct?
    tp = np.sum(assignment_arr_pred & assignment_arr_true)
    if tp == 0:
        return 0
    fp = np.sum(assignment_arr_pred & ~assignment_arr_true)
    return tp / (tp + fp)


def recall(
    scores_arr: np.ndarray,
    assignment_arr_pred: np.ndarray,
    assignment_arr_true: np.ndarray,
) -> float:
    # of all true matches, what fraction has been predicted correctly?
    tp = np.sum(assignment_arr_pred & assignment_arr_true)
    if tp == 0:
        return 0
    fn = np.sum(~assignment_arr_pred & assignment_arr_true)
    return tp / (tp + fn)


def f1score(
    scores_arr: np.ndarray,
    assignment_arr_pred: np.ndarray,
    assignment_arr_true: np.ndarray,
) -> float:
    # harmonic mean of precision and recall
    tp = np.sum(assignment_arr_pred & assignment_arr_true)
    if tp == 0:
        return 0.0
    fp = np.sum(assignment_arr_pred & ~assignment_arr_true)
    fn = np.sum(~assignment_arr_pred & assignment_arr_true)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def accuracy(
    scores_arr: np.ndarray,
    assignment_arr_pred: np.ndarray,
    assignment_arr_true: np.ndarray,
) -> float:
    # of all predictions (matches & mismatches), what fraction is correct?
    return np.mean(assignment_arr_pred == assignment_arr_true)


def uncertainty(
    scores_arr: np.ndarray,
    assignment_arr_pred: np.ndarray,
    assignment_arr_true: np.ndarray,
    aggr_fn: Callable[[np.ndarray], float],
    eps: float = 1e-9,
) -> float:
    fwd_scores_arr = scores_arr / (np.sum(scores_arr, axis=1, keepdims=True) + eps)
    rev_scores_arr = scores_arr / (np.sum(scores_arr, axis=0, keepdims=True) + eps)
    fwd_uncertainties = 1.0 - np.amax(fwd_scores_arr, axis=1)
    rev_uncertainties = 1.0 - np.amax(rev_scores_arr, axis=0)
    uncertainties = np.concatenate((fwd_uncertainties, rev_uncertainties))
    return float(aggr_fn(uncertainties))


def margin(
    scores_arr: np.ndarray,
    assignment_arr_pred: np.ndarray,
    assignment_arr_true: np.ndarray,
    aggr_fn: Callable[[np.ndarray], float],
    eps: float = 1e-9,
) -> float:
    fwd_scores_arr = scores_arr / (np.sum(scores_arr, axis=1, keepdims=True) + eps)
    rev_scores_arr = scores_arr / (np.sum(scores_arr, axis=0, keepdims=True) + eps)
    max2_fwd_scores_arr = -np.partition(-fwd_scores_arr, 1, axis=1)[:, :2]
    max2_rev_scores_arr = -np.partition(-rev_scores_arr, 1, axis=0)[:2, :]
    fwd_margins = max2_fwd_scores_arr[:, 0] - max2_fwd_scores_arr[:, 1]
    rev_margins = max2_rev_scores_arr[0, :] - max2_rev_scores_arr[1, :]
    margins = np.concatenate((fwd_margins, rev_margins))
    return float(aggr_fn(margins))


def entropy(
    scores_arr: np.ndarray,
    assignment_arr_pred: np.ndarray,
    assignment_arr_true: np.ndarray,
    aggr_fn: Callable[[np.ndarray], float],
    eps: float = 1e-9,
) -> float:
    fwd_scores_arr = scores_arr / (np.sum(scores_arr, axis=1, keepdims=True) + eps)
    rev_scores_arr = scores_arr / (np.sum(scores_arr, axis=0, keepdims=True) + eps)
    fwd_entropies = -np.sum(fwd_scores_arr * np.log(fwd_scores_arr + eps), axis=1)
    rev_entropies = -np.sum(rev_scores_arr * np.log(rev_scores_arr + eps), axis=0)
    entropies = np.concatenate((fwd_entropies, rev_entropies))
    return float(aggr_fn(entropies))


default_metrics = {
    "precision": precision,
    "recall": recall,
    "f1score": f1score,
    "accuracy": accuracy,
    "uncertainty_mean": partial(uncertainty, aggr_fn=np.mean),
    "uncertainty_std": partial(uncertainty, aggr_fn=np.std),
    "margin_mean": partial(margin, aggr_fn=np.mean),
    "margin_std": partial(margin, aggr_fn=np.std),
    "entropy_mean": partial(entropy, aggr_fn=np.mean),
    "entropy_std": partial(entropy, aggr_fn=np.std),
}
