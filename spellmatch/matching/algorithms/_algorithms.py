import logging
from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple

import numpy as np
import xarray as xr
from skimage.measure import regionprops
from skimage.transform import (
    AffineTransform,
    EuclideanTransform,
    ProjectiveTransform,
    SimilarityTransform,
)

from ...utils import compute_centroids
from .._matching import SpellmatchMatchingError

logger = logging.getLogger(__name__)


class MatchingAlgorithm(ABC):
    def __init__(self) -> None:
        self._current_source_regions: Optional[list] = None
        self._current_target_regions: Optional[list] = None

    def __call__(
        self,
        source_mask: xr.DataArray,
        target_mask: xr.DataArray,
        source_img: Optional[xr.DataArray] = None,
        target_img: Optional[xr.DataArray] = None,
        transform: Optional[np.ndarray] = None,
        reverse: bool = False,
    ) -> xr.DataArray:
        if reverse:
            source_mask, target_mask = target_mask, source_mask
            source_img, target_img = target_img, source_img
            transform = np.linalg.inv(transform)
        self._pre_run(
            source_mask,
            target_mask,
            source_img=source_img,
            target_img=target_img,
            transform=transform,
        )
        scores = self._run(
            source_mask,
            target_mask,
            source_img=source_img,
            target_img=target_img,
            transform=transform,
        )
        scores = self._post_run(
            scores,
            source_mask,
            target_mask,
            source_img=source_img,
            target_img=target_img,
            transform=transform,
        )

    def _pre_run(
        self,
        source_mask: xr.DataArray,
        target_mask: xr.DataArray,
        source_img: Optional[xr.DataArray] = None,
        target_img: Optional[xr.DataArray] = None,
        transform: Optional[np.ndarray] = None,
    ) -> None:
        self._current_source_regions = regionprops(
            source_mask.to_numpy(),
            intensity_image=source_img.to_numpy() if source_img is not None else None,
        )
        self._current_target_regions = regionprops(
            target_mask.to_numpy(),
            intensity_image=target_img.to_numpy() if target_img is not None else None,
        )

    @abstractmethod
    def _run(
        self,
        source_mask: xr.DataArray,
        target_mask: xr.DataArray,
        source_img: Optional[xr.DataArray] = None,
        target_img: Optional[xr.DataArray] = None,
        transform: Optional[np.ndarray] = None,
    ) -> xr.DataArray:
        raise NotImplementedError()

    def _post_run(
        self,
        scores: xr.DataArray,
        source_mask: xr.DataArray,
        target_mask: xr.DataArray,
        source_img: Optional[xr.DataArray] = None,
        target_img: Optional[xr.DataArray] = None,
        transform: Optional[np.ndarray] = None,
    ) -> xr.DataArray:
        self._current_source_regions = None
        self._current_target_regions = None
        return scores


class IterativeMatchingAlgorithm(MatchingAlgorithm):
    IterationCallback = Callable[[int, xr.DataArray, np.ndarray], bool]
    TRANSFORM_TYPES: dict[str, ProjectiveTransform] = {
        "euclidean": EuclideanTransform,
        "similarity": SimilarityTransform,
        "affine": AffineTransform,
    }

    def __init__(
        self,
        num_iter: int,
        top_k_estim: int,
        transform_type: str,
    ) -> None:
        super(IterativeMatchingAlgorithm, self).__init__()
        self.num_iter = num_iter
        self.top_k_estim = top_k_estim
        self.transform_type = self.TRANSFORM_TYPES[transform_type]
        self.iter_callbacks: list[IterativeMatchingAlgorithm.IterationCallback] = []
        self._current_source_centroids: Optional[np.ndarray] = None
        self._current_target_centroids: Optional[np.ndarray] = None

    def _pre_run(
        self,
        source_mask: xr.DataArray,
        target_mask: xr.DataArray,
        source_img: Optional[xr.DataArray] = None,
        target_img: Optional[xr.DataArray] = None,
        transform: Optional[np.ndarray] = None,
    ) -> None:
        super(IterativeMatchingAlgorithm, self)._pre_run(
            source_mask, target_mask, source_img, target_img, transform
        )
        self._current_target_centroids = compute_centroids(
            target_mask, regions=self._current_target_regions
        )

    def _run(
        self,
        source_mask: xr.DataArray,
        target_mask: xr.DataArray,
        source_img: Optional[xr.DataArray] = None,
        target_img: Optional[xr.DataArray] = None,
        transform: Optional[np.ndarray] = None,
    ) -> xr.DataArray:
        current_transform = transform
        for iteration in range(self.num_iter):
            self._pre_iter(
                iteration,
                source_mask,
                target_mask,
                source_img=source_img,
                target_img=target_img,
                transform=current_transform,
            )
            scores = self._iter(
                iteration,
                source_mask,
                target_mask,
                source_img=source_img,
                target_img=target_img,
                transform=current_transform,
            )
            max_source_scores = np.amax(scores, axis=1)
            top_source_indices = np.argpartition(
                -max_source_scores, self.top_k_estim - 1
            )[: self.top_k_estim]
            top_target_indices = np.argmax(scores[top_source_indices, :], axis=1)
            top_source_indices = top_source_indices[max_source_scores > 0.0]
            top_target_indices = top_target_indices[max_source_scores > 0.0]
            updated_transform = self.transform_type()
            if updated_transform.estimate(
                self._current_source_centroids[top_source_indices],
                self._current_target_centroids[top_target_indices],
            ):
                updated_transform = updated_transform.params
            else:
                updated_transform = None
            scores, updated_transform, stop = self._post_iter(
                iteration,
                scores,
                updated_transform,
                source_mask,
                target_mask,
                source_img=source_img,
                target_img=target_img,
                transform=current_transform,
            )
            for callback in self.iter_callbacks:
                if not callback(iteration, scores, updated_transform):
                    stop = True
            if updated_transform is None or stop:
                break
            current_transform = updated_transform
        return scores

    def _pre_iter(
        self,
        iteration: int,
        source_mask: xr.DataArray,
        target_mask: xr.DataArray,
        source_img: Optional[xr.DataArray] = None,
        target_img: Optional[xr.DataArray] = None,
        transform: Optional[np.ndarray] = None,
    ) -> None:
        self._current_source_centroids = compute_centroids(
            source_mask, transform=transform
        )

    @abstractmethod
    def _iter(
        self,
        iteration: int,
        source_mask: xr.DataArray,
        target_mask: xr.DataArray,
        source_img: Optional[xr.DataArray] = None,
        target_img: Optional[xr.DataArray] = None,
        transform: Optional[np.ndarray] = None,
    ) -> xr.DataArray:
        raise NotImplementedError()

    def _post_iter(
        self,
        iteration: int,
        scores: xr.DataArray,
        updated_transform: Optional[np.ndarray],
        source_mask: xr.DataArray,
        target_mask: xr.DataArray,
        source_img: Optional[xr.DataArray] = None,
        target_img: Optional[xr.DataArray] = None,
        transform: Optional[np.ndarray] = None,
    ) -> Tuple[xr.DataArray, np.ndarray, bool]:
        self._current_source_centroids = None
        return scores, updated_transform, False

    def _post_run(
        self,
        scores: xr.DataArray,
        source_mask: xr.DataArray,
        target_mask: xr.DataArray,
        source_img: Optional[xr.DataArray] = None,
        target_img: Optional[xr.DataArray] = None,
        transform: Optional[np.ndarray] = None,
    ) -> xr.DataArray:
        scores = super(IterativeMatchingAlgorithm, self)._post_run(
            scores, source_mask, target_mask, source_img, target_img, transform
        )
        self._current_target_centroids = None
        return scores


class SpellmatchMatchingAlgorithmError(SpellmatchMatchingError):
    pass
