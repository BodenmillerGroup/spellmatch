from typing import Optional, Type

import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import Polygon
from skimage.transform import ProjectiveTransform
from sklearn.neighbors import NearestNeighbors

from ..._spellmatch import hookimpl
from ._algorithms import IterativePointsMatchingAlgorithm, MaskMatchingAlgorithm


@hookimpl
def spellmatch_get_mask_matching_algorithm(
    name: str,
) -> Optional[Type[MaskMatchingAlgorithm]]:
    if name == "icp":
        return IterativeClosestPoints
    return None


class IterativeClosestPoints(IterativePointsMatchingAlgorithm):
    def __init__(
        self,
        *,
        max_dist: Optional[float] = None,
        min_change: Optional[float] = None,
        max_iter: int = 200,
        transform_type: str = "affine",
        transform_estim_top_k: int = 50,
        outlier_dist: Optional[float] = None,
        points_feature: str = "centroid",
        intensities_feature: str = "intensity_mean",
    ) -> None:
        super(IterativeClosestPoints, self).__init__(
            max_iter,
            transform_type,
            transform_estim_top_k,
            outlier_dist,
            points_feature,
            intensities_feature,
        )
        self.max_dist = max_dist
        self.min_change = min_change
        self._current_nn: Optional[NearestNeighbors] = None
        self._current_dists_mean: Optional[float] = None
        self._current_dists_std: Optional[float] = None
        self._last_dists_mean: Optional[float] = None
        self._last_dists_std: Optional[float] = None

    def match_points(
        self,
        source_points: pd.DataFrame,
        target_points: pd.DataFrame,
        source_bbox: Optional[Polygon] = None,
        target_bbox: Optional[Polygon] = None,
        source_intensities: Optional[pd.DataFrame] = None,
        target_intensities: Optional[pd.DataFrame] = None,
        transform: Optional[ProjectiveTransform] = None,
    ) -> xr.DataArray:
        self._current_nn = NearestNeighbors(n_neighbors=1).fit(target_points)
        self._last_dists_mean = None
        self._last_dists_std = None
        scores = super(IterativeClosestPoints, self).match_points(
            source_points,
            target_points,
            source_bbox,
            target_bbox,
            source_intensities,
            target_intensities,
            transform,
        )
        self._current_nn = None
        self._last_dists_mean = None
        self._last_dists_std = None
        return scores

    def _match_points(
        self,
        source_points: pd.DataFrame,
        target_points: pd.DataFrame,
        source_intensities: Optional[pd.DataFrame],
        target_intensities: Optional[pd.DataFrame],
    ) -> xr.DataArray:
        source_indices = np.arange(len(source_points.index))
        dists, target_indices = self._current_nn.kneighbors(source_points.index)
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
        scores = xr.DataArray(
            data=scores_data,
            coords={
                source_name: source_points.index.values,
                target_name: target_points.index.values,
            },
        )
        return scores

    def _post_iter(
        self,
        iteration: int,
        current_transform: Optional[ProjectiveTransform],
        current_scores: xr.DataArray,
        updated_transform: Optional[ProjectiveTransform],
    ) -> bool:
        stop = super(IterativeClosestPoints, self)._post_iter(
            iteration, current_transform, current_scores, updated_transform
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
        return stop

    def _compute_dists_mean_change(self) -> float:
        return np.abs(
            (self._current_dists_mean - self._last_dists_mean) / self._last_dists_mean
        )

    def _compute_dists_std_change(self) -> float:
        return np.abs(
            (self._current_dists_std - self._last_dists_std) / self._last_dists_std
        )
