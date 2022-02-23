import logging
from typing import Optional, Tuple, Type

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.neighbors import NearestNeighbors

from ..._spellmatch import hookimpl
from ...utils import transform_points
from ._algorithms import IterativePointsMatchingAlgorithm, MatchingAlgorithm

logger = logging.getLogger(__name__)


@hookimpl
def spellmatch_get_matching_algorithm(name: str) -> Optional[Type[MatchingAlgorithm]]:
    if name == "icp":
        return IterativeClosestPointsMatchingAlgorithm
    return None


class IterativeClosestPointsMatchingAlgorithm(IterativePointsMatchingAlgorithm):
    def __init__(
        self,
        *,
        max_iter: int,
        top_k_estim: int,
        max_dist: Optional[float] = None,
        min_change: Optional[float] = None,
        transform_type: str = "affine",
        points_feature: str = "centroid",
        intensities_feature: str = "intensity_mean",
    ) -> None:
        super(IterativeClosestPointsMatchingAlgorithm, self).__init__(
            max_iter, top_k_estim, transform_type, points_feature, intensities_feature
        )
        self.max_dist = max_dist
        self.min_change = min_change
        self._nn: Optional[NearestNeighbors] = None
        self._current_dists_mean: Optional[float] = None
        self._current_dists_std: Optional[float] = None
        self._last_dists_mean: Optional[float] = None
        self._last_dists_std: Optional[float] = None

    def _pre_match_points(
        self,
        source_points: pd.DataFrame,
        target_points: pd.DataFrame,
        source_intensities: Optional[pd.DataFrame] = None,
        target_intensities: Optional[pd.DataFrame] = None,
        source_shape: Optional[Tuple[int, int]] = None,
        target_shape: Optional[Tuple[int, int]] = None,
        transform: Optional[np.ndarray] = None,
    ) -> None:
        super(IterativeClosestPointsMatchingAlgorithm, self)._pre_match_points(
            source_points,
            target_points,
            source_intensities,
            target_intensities,
            source_shape,
            target_shape,
            transform,
        )
        self._nn = NearestNeighbors(n_neighbors=1).fit(target_points)

    def _post_match_points(
        self,
        scores: xr.DataArray,
        source_points: pd.DataFrame,
        target_points: pd.DataFrame,
        source_intensities: Optional[pd.DataFrame] = None,
        target_intensities: Optional[pd.DataFrame] = None,
        source_shape: Optional[Tuple[int, int]] = None,
        target_shape: Optional[Tuple[int, int]] = None,
        transform: Optional[np.ndarray] = None,
    ) -> xr.DataArray:
        scores = super(
            IterativeClosestPointsMatchingAlgorithm, self
        )._post_match_points(
            scores,
            source_points,
            target_points,
            source_intensities,
            target_intensities,
            source_shape,
            target_shape,
            transform,
        )
        self._nn = None
        return scores

    def _iter(
        self,
        iteration: int,
        source_points: pd.DataFrame,
        target_points: pd.DataFrame,
        source_intensities: Optional[pd.DataFrame] = None,
        target_intensities: Optional[pd.DataFrame] = None,
        source_shape: Optional[Tuple[int, int]] = None,
        target_shape: Optional[Tuple[int, int]] = None,
        transform: Optional[np.ndarray] = None,
    ) -> xr.DataArray:
        source_points = source_points
        if transform is not None:
            source_points = transform_points(source_points)
        source_indices = np.arange(len(source_points.index))
        dists, target_indices = self._nn.kneighbors(source_points.index)
        dists, target_indices = dists[:, 0], target_indices[:, 0]
        if self.max_dist:
            source_indices = source_indices[dists <= self.max_dist]
            target_indices = target_indices[dists <= self.max_dist]
            dists = dists[dists <= self.max_dist]
        self._current_dists_mean = np.mean(dists)
        self._current_dists_std = np.std(dists)
        source_name = source_points.index.name or "source"
        target_name = target_points.index.name or "target"
        scores_data = np.zeros((len(source_points.index), len(target_points.index)))
        scores_data[source_indices, target_indices] = 1.0
        return xr.DataArray(
            data=scores_data,
            coords={
                source_name: source_points.index.values,
                target_name: target_points.index.values,
            },
        )

    def _post_iter(
        self,
        iteration: int,
        scores: xr.DataArray,
        updated_transform: Optional[np.ndarray],
        source_points: pd.DataFrame,
        target_points: pd.DataFrame,
        source_intensities: Optional[pd.DataFrame] = None,
        target_intensities: Optional[pd.DataFrame] = None,
        source_shape: Optional[Tuple[int, int]] = None,
        target_shape: Optional[Tuple[int, int]] = None,
        transform: Optional[np.ndarray] = None,
    ) -> Tuple[xr.DataArray, np.ndarray, bool]:
        scores, updated_transform, stop = super(
            IterativeClosestPointsMatchingAlgorithm, self
        )._post_iter(
            iteration,
            scores,
            updated_transform,
            source_points,
            target_points,
            source_intensities,
            target_intensities,
            source_shape,
            target_shape,
            transform,
        )
        if (
            not stop
            and iteration > 0
            and self.min_change is not None
            and self._compute_dists_mean_change() < self.min_change
            and self._compute_dists_std_change() < self.min_change
        ):
            stop = True
        self._last_dists_mean = self._current_dists_mean
        self._last_dists_std = self._current_dists_std
        self._current_dists_mean = None
        self._current_dists_std = None
        return scores, updated_transform, stop

    def _compute_dists_mean_change(self) -> float:
        return np.abs(
            (self._current_dists_mean - self._last_dists_mean) / self._last_dists_mean
        )

    def _compute_dists_std_change(self) -> float:
        return np.abs(
            (self._current_dists_std - self._last_dists_std) / self._last_dists_std
        )
