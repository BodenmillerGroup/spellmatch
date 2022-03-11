import logging
from typing import Optional, Type, Union

import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import Polygon
from skimage.transform import EuclideanTransform, ProjectiveTransform
from sklearn.neighbors import NearestNeighbors

from ..._spellmatch import hookimpl
from ...utils import describe_transform
from ._algorithms import IterativePointsMatchingAlgorithm, MaskMatchingAlgorithm

logger = logging.getLogger(__name__)


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
        points_feature: str = "centroid",
        intensities_feature: str = "intensity_mean",
        num_iter: int = 200,
        transform_type: Union[str, Type[ProjectiveTransform]] = EuclideanTransform,
        transform_estim_type: Union[
            str, IterativePointsMatchingAlgorithm.TransformEstimationType
        ] = IterativePointsMatchingAlgorithm.TransformEstimationType.MAX_SCORE,
        transform_estim_topn: Optional[int] = None,
        max_dist: Optional[float] = None,
        min_change: Optional[float] = None,
    ) -> None:
        super(IterativeClosestPoints, self).__init__(
            outlier_dist=max_dist,  # exclude points that anyway couldn't be assigned
            points_feature=points_feature,
            intensities_feature=intensities_feature,
            num_iter=num_iter,
            transform_type=transform_type,
            transform_estim_type=transform_estim_type,
            transform_estim_topn=transform_estim_topn,
        )
        self.max_dist = max_dist
        self.min_change = min_change
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
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(target_points.to_numpy())
        nn_dists, target_ind = nn.kneighbors(source_points.to_numpy())
        nn_dists, target_ind = nn_dists[:, 0], target_ind[:, 0]
        source_ind = np.arange(len(source_points.index))
        if self.max_dist is not None:
            source_ind = source_ind[nn_dists <= self.max_dist]
            target_ind = target_ind[nn_dists <= self.max_dist]
            nn_dists = nn_dists[nn_dists <= self.max_dist]
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
        transform_change = np.linalg.norm(
            current_transform.params - updated_transform.params
        )
        dists_mean_change = self._compute_distance_mean_change()
        dists_std_change = self._compute_distance_std_change()
        if (
            self.min_change is not None
            and dists_mean_change < self.min_change
            and dists_std_change < self.min_change
        ):
            stop = True
        logger.info(
            f"Iteration {iteration + 1}: "
            f"transform_change={transform_change:.6f} "
            f"({describe_transform(updated_transform)}), "
            f"dists_mean_change={dists_mean_change:.6f}, "
            f"dists_std_change={dists_std_change:.6f}, "
            f"stop={stop}"
        )
        self._last_dists_mean = self._current_dists_mean
        self._last_dists_std = self._current_dists_std
        return stop

    def _compute_distance_mean_change(self) -> float:
        if not self._last_dists_mean:
            return float("inf")
        return np.abs(
            (self._current_dists_mean - self._last_dists_mean) / self._last_dists_mean
        )

    def _compute_distance_std_change(self) -> float:
        if not self._last_dists_std:
            return float("inf")
        return np.abs(
            (self._current_dists_std - self._last_dists_std) / self._last_dists_std
        )
