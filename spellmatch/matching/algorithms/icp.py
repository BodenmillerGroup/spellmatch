from typing import Optional, Type, Union

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
    name: Optional[str],
) -> Union[Optional[Type["MaskMatchingAlgorithm"]], list[str]]:
    algorithms: dict[str, Type[MaskMatchingAlgorithm]] = {
        "icp": IterativeClosestPoints,
    }
    if name is not None:
        return algorithms.get(name)
    return list(algorithms.keys())


class IterativeClosestPoints(IterativePointsMatchingAlgorithm):
    def __init__(
        self,
        *,
        max_nn_dist: Optional[float] = None,
        min_change: Optional[float] = None,
        max_iter: int = 200,
        transform_type: str = "rigid",
        transform_estim_type: str = "max_score",
        transform_estim_top_k: int = 50,
        outlier_dist: Optional[float] = None,
        points_feature: str = "centroid",
        intensities_feature: str = "intensity_mean",
    ) -> None:
        super(IterativeClosestPoints, self).__init__(
            max_iter,
            transform_type,
            transform_estim_type,
            transform_estim_top_k,
            outlier_dist,
            points_feature,
            intensities_feature,
        )
        self.max_nn_dist = max_nn_dist
        self.min_change = min_change
        self._target_nn: Optional[NearestNeighbors] = None
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
        self._target_nn = NearestNeighbors(n_neighbors=1).fit(target_points)
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
        self._target_nn = None
        self._current_dists_mean = None
        self._current_dists_std = None
        return scores

    def _match_points(
        self,
        source_points: pd.DataFrame,
        target_points: pd.DataFrame,
        source_intensities: Optional[pd.DataFrame],
        target_intensities: Optional[pd.DataFrame],
    ) -> xr.DataArray:
        source_ind = np.arange(len(source_points.index))
        nn_dists, target_ind = self._target_nn.kneighbors(source_points.index)
        nn_dists, target_ind = nn_dists[:, 0], target_ind[:, 0]
        if self.max_nn_dist:
            source_ind = source_ind[nn_dists <= self.max_nn_dist]
            target_ind = target_ind[nn_dists <= self.max_nn_dist]
            nn_dists = nn_dists[nn_dists <= self.max_nn_dist]
        self._current_dists_mean = np.mean(nn_dists)
        self._current_dists_std = np.std(nn_dists)
        scores_data = np.zeros((len(source_points.index), len(target_points.index)))
        scores_data[source_ind, target_ind] = 1
        scores = xr.DataArray(
            data=scores_data,
            coords={
                source_points.index.name: source_points.index.to_numpy(),
                target_points.index.name: target_points.index.to_numpy(),
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
        return stop

    def _compute_dists_mean_change(self) -> float:
        return np.abs(
            (self._current_dists_mean - self._last_dists_mean) / self._last_dists_mean
        )

    def _compute_dists_std_change(self) -> float:
        return np.abs(
            (self._current_dists_std - self._last_dists_std) / self._last_dists_std
        )
