import logging
from collections.abc import Callable, MutableMapping
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import Polygon
from skimage.transform import ProjectiveTransform
from sklearn.neighbors import NearestNeighbors

from ..._spellmatch import hookimpl
from ._algorithms import IterativePointsMatchingAlgorithm, MaskMatchingAlgorithm

logger = logging.getLogger(__name__)


@hookimpl
def spellmatch_get_mask_matching_algorithm(
    name: Optional[str],
) -> Union[Optional[type["MaskMatchingAlgorithm"]], list[str]]:
    algorithms = {
        "icp": IterativeClosestPoints,
    }
    if name is not None:
        return algorithms.get(name)
    return list(algorithms.keys())


class IterativeClosestPoints(IterativePointsMatchingAlgorithm):
    def __init__(
        self,
        *,
        point_feature: str = "centroid",
        intensity_feature: str = "intensity_mean",
        intensity_transform: Union[
            str, Callable[[np.ndarray], np.ndarray], None
        ] = None,
        transform_type: Union[str, type[ProjectiveTransform]] = "rigid",
        max_iter: int = 200,
        scores_tol: Optional[float] = None,
        transform_tol: Optional[float] = None,
        filter_outliers: bool = True,
        max_dist: Optional[float] = None,
        min_change: Optional[float] = None,
    ) -> None:
        super(IterativeClosestPoints, self).__init__(
            outlier_dist=max_dist if filter_outliers else None,
            point_feature=point_feature,
            intensity_feature=intensity_feature,
            intensity_transform=intensity_transform,
            transform_type=transform_type,
            transform_estim_type=self.TransformEstimationType.MAX_SCORE,
            transform_estim_k_best=None,
            max_iter=max_iter,
            scores_tol=scores_tol,
            transform_tol=transform_tol,
        )
        self.max_dist = max_dist
        self.min_change = min_change
        self._current_dists_mean: Optional[float] = None
        self._current_dists_std: Optional[float] = None
        self._last_dists_mean: Optional[float] = None
        self._last_dists_std: Optional[float] = None

    def match_points(
        self,
        source_name: str,
        target_name: str,
        source_points: pd.DataFrame,
        target_points: pd.DataFrame,
        source_bbox: Optional[Polygon] = None,
        target_bbox: Optional[Polygon] = None,
        source_intensities: Optional[pd.DataFrame] = None,
        target_intensities: Optional[pd.DataFrame] = None,
        prior_transform: Optional[ProjectiveTransform] = None,
        cache: Optional[MutableMapping[str, Any]] = None,
    ) -> tuple[dict[str, Any], xr.DataArray]:
        self._last_dists_mean = None
        self._last_dists_std = None
        info, scores = super(IterativeClosestPoints, self).match_points(
            source_name,
            target_name,
            source_points,
            target_points,
            source_bbox=source_bbox,
            target_bbox=target_bbox,
            source_intensities=source_intensities,
            target_intensities=target_intensities,
            prior_transform=prior_transform,
            cache=cache,
        )
        self._current_dists_mean = None
        self._current_dists_std = None
        return info, scores

    def _match_points(
        self,
        source_name: str,
        target_name: str,
        source_points: pd.DataFrame,
        target_points: pd.DataFrame,
        source_intensities: Optional[pd.DataFrame],
        target_intensities: Optional[pd.DataFrame],
        cache: Optional[MutableMapping[str, Any]],
    ) -> tuple[dict[str, Any], xr.DataArray]:
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(target_points.to_numpy())
        nn_dists, nn_ind = nn.kneighbors(source_points.to_numpy())
        dists, target_ind = nn_dists[:, 0], nn_ind[:, 0]
        source_ind = np.arange(len(source_points.index))
        if self.max_dist is not None:
            source_ind = source_ind[dists <= self.max_dist]
            target_ind = target_ind[dists <= self.max_dist]
            dists = dists[dists <= self.max_dist]
        self._current_dists_mean = np.mean(dists)
        self._current_dists_std = np.std(dists)
        scores_data = np.zeros((len(source_points.index), len(target_points.index)))
        scores_data[source_ind, target_ind] = 1
        scores = xr.DataArray(
            data=scores_data,
            coords={
                source_name: source_points.index.to_numpy(),
                target_name: target_points.index.to_numpy(),
            },
        )
        return {}, scores

    def _check_convergence(
        self,
        last_scores: Optional[xr.DataArray],
        current_scores: xr.DataArray,
        current_transform: Optional[ProjectiveTransform],
        updated_transform: Optional[ProjectiveTransform],
    ) -> bool:
        converged = super(IterativeClosestPoints, self)._check_convergence(
            last_scores, current_scores, current_transform, updated_transform
        )
        dists_mean_change = float("inf")
        dists_std_change = float("inf")
        if self._last_dists_mean is not None and self._last_dists_std is not None:
            dists_mean_change = np.abs(
                (self._current_dists_mean - self._last_dists_mean)
                / self._last_dists_mean
            )
            dists_std_change = np.abs(
                (self._current_dists_std - self._last_dists_std) / self._last_dists_std
            )
        logger.info(
            f"Distance change: mean={dists_mean_change:.9f}, SD={dists_std_change:.9f}"
        )
        if (
            self.min_change is not None
            and dists_mean_change < self.min_change
            and dists_std_change < self.min_change
        ):
            converged = True
        self._last_dists_mean = self._current_dists_mean
        self._last_dists_std = self._current_dists_std
        return converged
