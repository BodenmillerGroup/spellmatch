import logging
from typing import Optional

import numpy as np
import xarray as xr
from sklearn.neighbors import NearestNeighbors

from ._algorithms import IterativeMatchingAlgorithm

# TODO add criterion: distance mean/std change tolerance

logger = logging.getLogger(__name__)


class IterativeClosestPointsMatchingAlgorithm(IterativeMatchingAlgorithm):
    def __init__(
        self,
        *,
        num_iter: int,
        top_k_estim: int,
        max_dist: Optional[float] = None,
        transform_type: str = "affine",
    ) -> None:
        super(IterativeClosestPointsMatchingAlgorithm, self).__init__(
            num_iter, top_k_estim, transform_type
        )
        self.max_dist = max_dist
        self._current_target_nn: Optional[NearestNeighbors] = None

    def _pre_run(
        self,
        source_mask: xr.DataArray,
        target_mask: xr.DataArray,
        source_img: Optional[xr.DataArray] = None,
        target_img: Optional[xr.DataArray] = None,
        transform: Optional[np.ndarray] = None,
    ) -> None:
        super(IterativeClosestPointsMatchingAlgorithm, self)._pre_run(
            source_mask, target_mask, source_img, target_img, transform
        )
        self._current_target_nn = NearestNeighbors(n_neighbors=1).fit(
            self._current_target_centroids
        )

    def _iter(
        self,
        iteration: int,
        source_mask: xr.DataArray,
        target_mask: xr.DataArray,
        source_img: Optional[xr.DataArray] = None,
        target_img: Optional[xr.DataArray] = None,
        transform: Optional[np.ndarray] = None,
    ) -> xr.DataArray:
        source_indices = np.arange(len(self._current_source_regions))
        dists, target_indices = self._current_target_nn.kneighbors(
            self._current_source_centroids
        )
        dists, target_indices = dists[:, 0], target_indices[:, 0]
        if self.max_dist is not None:
            source_indices = source_indices[dists <= self.max_dist]
            target_indices = target_indices[dists <= self.max_dist]
        scores_data = np.zeros(
            (len(self._current_source_regions), len(self._current_target_regions))
        )
        scores_data[source_indices, target_indices] = 1.0
        return xr.DataArray(
            data=scores_data,
            coords={
                source_mask.name: [r.label for r in self._current_source_regions],
                target_mask.name: [r.label for r in self._current_target_regions],
            },
            dims=(source_mask.name, target_mask.name),
        )

    def _post_run(
        self,
        scores: xr.DataArray,
        source_mask: xr.DataArray,
        target_mask: xr.DataArray,
        source_img: Optional[xr.DataArray] = None,
        target_img: Optional[xr.DataArray] = None,
        transform: Optional[np.ndarray] = None,
    ) -> xr.DataArray:
        scores = super(IterativeClosestPointsMatchingAlgorithm, self)._post_run(
            scores, source_mask, target_mask, source_img, target_img, transform
        )
        self._current_target_nn = None
        return scores
