import logging
from typing import Callable, Optional, Type, Union

import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import Polygon
from skimage.transform import EuclideanTransform, ProjectiveTransform
from sklearn.neighbors import NearestNeighbors

from ..._spellmatch import hookimpl
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
        point_feature: str = "centroid",
        intensity_feature: str = "intensity_mean",
        intensity_transform: Union[
            str, Callable[[np.ndarray], np.ndarray], None
        ] = None,
        transform_type: Union[str, Type[ProjectiveTransform]] = EuclideanTransform,
        max_iter: int = 200,
        scores_tol: Optional[float] = None,
        transform_tol: Optional[float] = None,
        max_dist: Optional[float] = None,
        min_change: Optional[float] = None,
    ) -> None:
        super(IterativeClosestPoints, self).__init__(
            outlier_dist=max_dist,
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
        source_points: pd.DataFrame,
        target_points: pd.DataFrame,
        source_bbox: Optional[Polygon] = None,
        target_bbox: Optional[Polygon] = None,
        source_intensities: Optional[pd.DataFrame] = None,
        target_intensities: Optional[pd.DataFrame] = None,
        prior_transform: Optional[ProjectiveTransform] = None,
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
            prior_transform,
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
        nn.fit(target_points)
        nn_dists, nn_ind = nn.kneighbors(source_points)
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
        converged: bool = False,
    ) -> bool:
        dists_mean_change = float("inf")
        if self._last_dists_mean is not None:
            dists_mean_change = np.abs(
                (self._current_dists_mean - self._last_dists_mean)
                / self._last_dists_mean
            )
        dists_std_change = float("inf")
        if self._last_dists_std is not None:
            dists_std_change = np.abs(
                (self._current_dists_std - self._last_dists_std) / self._last_dists_std
            )
        if (
            self.min_change is not None
            and dists_mean_change < self.min_change
            and dists_std_change < self.min_change
        ):
            converged = True
        logger.info(
            f"dists_mean_change={dists_mean_change:.6f}, "
            f"dists_std_change={dists_std_change:.6f}, "
            f"converged={converged}"
        )
        self._last_dists_mean = self._current_dists_mean
        self._last_dists_std = self._current_dists_std
        return super(IterativeClosestPoints, self)._post_iter(
            iteration,
            current_transform,
            current_scores,
            updated_transform,
            converged=converged,
        )
