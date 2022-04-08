from typing import Callable

import numpy as np


def precision(
    scores: np.ndarray,
    assignment_mat_pred: np.ndarray,
    assignment_mat_true: np.ndarray,
) -> float:
    # Of all predicted matches, what fraction is correct?
    tp = np.sum(assignment_mat_pred & assignment_mat_true)
    if tp == 0:
        return 0
    fp = np.sum(assignment_mat_pred & ~assignment_mat_true)
    return tp / (tp + fp)


def recall(
    scores: np.ndarray,
    assignment_mat_pred: np.ndarray,
    assignment_mat_true: np.ndarray,
) -> float:
    # Of all true matches, what fraction has been predicted correctly?
    tp = np.sum(assignment_mat_pred & assignment_mat_true)
    if tp == 0:
        return 0
    fn = np.sum(~assignment_mat_pred & assignment_mat_true)
    return tp / (tp + fn)


def f1score(
    scores: np.ndarray,
    assignment_mat_pred: np.ndarray,
    assignment_mat_true: np.ndarray,
) -> float:
    # Harmonic mean of precision and recall
    tp = np.sum(assignment_mat_pred & assignment_mat_true)
    if tp == 0:
        return 0.0
    fp = np.sum(assignment_mat_pred & ~assignment_mat_true)
    fn = np.sum(~assignment_mat_pred & assignment_mat_true)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def accuracy(
    scores: np.ndarray,
    assignment_mat_pred: np.ndarray,
    assignment_mat_true: np.ndarray,
) -> float:
    # Of all predictions (matches & mismatches), what fraction is correct?
    return np.mean(assignment_mat_pred == assignment_mat_true)


def uncertainty(
    scores: np.ndarray,
    assignment_mat_pred: np.ndarray,
    assignment_mat_true: np.ndarray,
    aggr_fn: Callable[[np.ndarray], float],
    eps: float = 1e-9,
) -> float:
    fwd_scores = scores / (np.sum(scores, axis=1, keepdims=True) + eps)
    rev_scores = scores / (np.sum(scores, axis=0, keepdims=True) + eps)
    fwd_uncertainties = 1.0 - np.amax(fwd_scores, axis=1)
    rev_uncertainties = 1.0 - np.amax(rev_scores, axis=0)
    uncertainties = np.concatenate((fwd_uncertainties, rev_uncertainties))
    return float(aggr_fn(uncertainties))


def margin(
    scores: np.ndarray,
    assignment_mat_pred: np.ndarray,
    assignment_mat_true: np.ndarray,
    aggr_fn: Callable[[np.ndarray], float],
    eps: float = 1e-9,
) -> float:
    fwd_scores = scores / (np.sum(scores, axis=1, keepdims=True) + eps)
    rev_scores = scores / (np.sum(scores, axis=0, keepdims=True) + eps)
    max2_fwd_scores = -np.partition(-fwd_scores, 1, axis=1)[:, :2]
    max2_rev_scores = -np.partition(-rev_scores, 1, axis=0)[:2, :]
    fwd_margins = max2_fwd_scores[:, 0] - max2_fwd_scores[:, 1]
    rev_margins = max2_rev_scores[0, :] - max2_rev_scores[1, :]
    margins = np.concatenate((fwd_margins, rev_margins))
    return float(aggr_fn(margins))


def entropy(
    scores: np.ndarray,
    assignment_mat_pred: np.ndarray,
    assignment_mat_true: np.ndarray,
    aggr_fn: Callable[[np.ndarray], float],
    eps: float = 1e-9,
) -> float:
    fwd_scores = scores / (np.sum(scores, axis=1, keepdims=True) + eps)
    rev_scores = scores / (np.sum(scores, axis=0, keepdims=True) + eps)
    fwd_log_products = fwd_scores * np.log(fwd_scores + eps)
    rev_log_products = rev_scores * np.log(rev_scores + eps)
    fwd_entropies = -np.sum(fwd_log_products, axis=1)
    rev_entropies = -np.sum(rev_log_products, axis=0)
    entropies = np.concatenate((fwd_entropies, rev_entropies))
    return float(aggr_fn(entropies))
