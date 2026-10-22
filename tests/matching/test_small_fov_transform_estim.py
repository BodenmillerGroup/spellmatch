"""Regression test: transform_estim_k_best crashing on FOVs smaller than it.

`IterativePointsMatchingAlgorithm._update_transform` (`_algorithms.py`) picks
the k-best candidate matches via `np.argpartition(..., k_best - 1)`, which
requires `k_best <= array length`. With the default `transform_estim_k_best
= 50`, any FOV with fewer than 50 source points crashed outright with
`ValueError: kth(...) out of bounds`, confirmed on a real ROI with 33 source
and 16 target cells. Fixed by clamping k_best to the number of available
candidates in both the MAX_SCORE and MAX_MARGIN branches.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from skimage.transform import EuclideanTransform

from spellmatch.matching.algorithms._algorithms import IterativePointsMatchingAlgorithm


def _make_algo(transform_estim_type, k_best):
    algo = object.__new__(IterativePointsMatchingAlgorithm)
    algo.transform_estim_type = transform_estim_type
    algo.transform_estim_k_best = k_best
    algo.transform_type = EuclideanTransform
    return algo


def _make_scores(n1, n2, seed=0):
    rng = np.random.default_rng(seed)
    scores_data = rng.random((n1, n2)).astype(np.float32)
    scores = xr.DataArray(
        scores_data, coords={"source": np.arange(n1), "target": np.arange(n2)}
    )
    source_points = pd.DataFrame(rng.random((n1, 2)), columns=["x", "y"])
    target_points = pd.DataFrame(rng.random((n2, 2)), columns=["x", "y"])
    return source_points, target_points, scores


@pytest.mark.parametrize(
    "transform_estim_type",
    [
        IterativePointsMatchingAlgorithm.TransformEstimationType.MAX_SCORE,
        IterativePointsMatchingAlgorithm.TransformEstimationType.MAX_MARGIN,
    ],
)
def test_k_best_larger_than_available_points_no_longer_crashes(transform_estim_type):
    # Exact real-world shape that crashed: 33 source points, k_best=50 default.
    algo = _make_algo(transform_estim_type, k_best=50)
    source_points, target_points, scores = _make_scores(n1=33, n2=16)
    result = algo._update_transform(source_points, target_points, scores)
    assert result is not None


@pytest.mark.parametrize(
    "transform_estim_type",
    [
        IterativePointsMatchingAlgorithm.TransformEstimationType.MAX_SCORE,
        IterativePointsMatchingAlgorithm.TransformEstimationType.MAX_MARGIN,
    ],
)
def test_k_best_smaller_than_available_points_still_works_as_before(
    transform_estim_type,
):
    # Normal case (plenty of candidates) must be unaffected by the clamp.
    algo = _make_algo(transform_estim_type, k_best=10)
    source_points, target_points, scores = _make_scores(n1=100, n2=80)
    result = algo._update_transform(source_points, target_points, scores)
    assert result is not None
