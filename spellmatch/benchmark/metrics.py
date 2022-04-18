from collections.abc import Callable
from functools import partial

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
    # of all true matches, what fraction has been predicted?
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


def uncertainty(
    scores_arr: np.ndarray,
    assignment_arr_pred: np.ndarray,
    assignment_arr_true: np.ndarray,
    aggr_fn: Callable[[np.ndarray], float],
    normalize: bool = True,
) -> float:
    if normalize and scores_arr.max() != 0:
        scores_arr /= scores_arr.max()
    fwd_uncertainties = 1.0 - np.amax(scores_arr, axis=1)
    rev_uncertainties = 1.0 - np.amax(scores_arr, axis=0)
    uncertainties = np.concatenate((fwd_uncertainties, rev_uncertainties))
    return float(aggr_fn(uncertainties))


def margin(
    scores_arr: np.ndarray,
    assignment_arr_pred: np.ndarray,
    assignment_arr_true: np.ndarray,
    aggr_fn: Callable[[np.ndarray], float],
    normalize: bool = True,
) -> float:
    if normalize and scores_arr.max() != 0:
        scores_arr /= scores_arr.max()
    max2_fwd_scores = -np.partition(-scores_arr, 1, axis=1)[:, :2]
    max2_rev_scores = -np.partition(-scores_arr, 1, axis=0)[:2, :]
    fwd_margins = max2_fwd_scores[:, 0] - max2_fwd_scores[:, 1]
    rev_margins = max2_rev_scores[0, :] - max2_rev_scores[1, :]
    margins = np.concatenate((fwd_margins, rev_margins))
    return float(aggr_fn(margins))


def entropy(
    scores_arr: np.ndarray,
    assignment_arr_pred: np.ndarray,
    assignment_arr_true: np.ndarray,
    aggr_fn: Callable[[np.ndarray], float],
    normalize: bool = True,
) -> float:
    if normalize and scores_arr.max() != 0:
        scores_arr /= scores_arr.max()
    fwd_entropies = -np.sum(
        scores_arr * np.log(scores_arr, where=scores_arr != 0), axis=1
    )
    rev_entropies = -np.sum(
        scores_arr * np.log(scores_arr, where=scores_arr != 0), axis=0
    )
    entropies = np.concatenate((fwd_entropies, rev_entropies))
    return float(aggr_fn(entropies))


default_metrics = {
    "precision": precision,
    "recall": recall,
    "f1score": f1score,
    "uncertainty_mean": partial(uncertainty, aggr_fn=np.mean),
    "uncertainty_std": partial(uncertainty, aggr_fn=np.std),
    "margin_mean": partial(margin, aggr_fn=np.mean),
    "margin_std": partial(margin, aggr_fn=np.std),
    "entropy_mean": partial(entropy, aggr_fn=np.mean),
    "entropy_std": partial(entropy, aggr_fn=np.std),
}
