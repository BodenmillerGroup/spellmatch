import logging
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from skimage.measure import regionprops
from skimage.transform import (
    AffineTransform,
    EuclideanTransform,
    ProjectiveTransform,
    SimilarityTransform,
)

from ...utils import compute_intensities, compute_points
from .._matching import SpellmatchMatchingError

logger = logging.getLogger(__name__)


class MatchingAlgorithm(ABC):
    def match_masks(
        self,
        source_mask: xr.DataArray,
        target_mask: xr.DataArray,
        source_img: Optional[xr.DataArray] = None,
        target_img: Optional[xr.DataArray] = None,
        transform: Optional[np.ndarray] = None,
    ) -> xr.DataArray:
        self._pre_match_masks(
            source_mask,
            target_mask,
            source_img=source_img,
            target_img=target_img,
            transform=transform,
        )
        scores = self._match_masks(
            source_mask,
            target_mask,
            source_img=source_img,
            target_img=target_img,
            transform=transform,
        )
        scores = self._post_match_masks(
            scores,
            source_mask,
            target_mask,
            source_img=source_img,
            target_img=target_img,
            transform=transform,
        )
        return scores

    def _pre_match_masks(
        self,
        source_mask: xr.DataArray,
        target_mask: xr.DataArray,
        source_img: Optional[xr.DataArray] = None,
        target_img: Optional[xr.DataArray] = None,
        transform: Optional[np.ndarray] = None,
    ) -> None:
        pass

    @abstractmethod
    def _match_masks(
        self,
        source_mask: xr.DataArray,
        target_mask: xr.DataArray,
        source_img: Optional[xr.DataArray] = None,
        target_img: Optional[xr.DataArray] = None,
        transform: Optional[np.ndarray] = None,
    ) -> xr.DataArray:
        raise NotImplementedError()

    def _post_match_masks(
        self,
        scores: xr.DataArray,
        source_mask: xr.DataArray,
        target_mask: xr.DataArray,
        source_img: Optional[xr.DataArray] = None,
        target_img: Optional[xr.DataArray] = None,
        transform: Optional[np.ndarray] = None,
    ) -> xr.DataArray:
        return scores


class PointsMatchingAlgorithm(MatchingAlgorithm):
    def __init__(
        self,
        points_feature: str = "centroid",
        intensities_feature: str = "intensity_mean",
    ) -> None:
        super(PointsMatchingAlgorithm, self).__init__()
        self.points_feature = points_feature
        self.intensities_feature = intensities_feature
        self._current_source_mask: Optional[xr.DataArray] = None
        self._current_target_mask: Optional[xr.DataArray] = None
        self._current_source_img: Optional[xr.DataArray] = None
        self._current_target_img: Optional[xr.DataArray] = None
        self._current_source_regions: Optional[list] = None
        self._current_target_regions: Optional[list] = None

    def _pre_match_masks(
        self,
        source_mask: xr.DataArray,
        target_mask: xr.DataArray,
        source_img: Optional[xr.DataArray] = None,
        target_img: Optional[xr.DataArray] = None,
        transform: Optional[np.ndarray] = None,
    ) -> None:
        super(PointsMatchingAlgorithm, self)._pre_match_masks(
            source_mask, target_mask, source_img, target_img, transform
        )
        self._current_source_mask = source_mask
        self._current_target_mask = target_mask
        self._current_source_img = source_img
        self._current_target_img = target_img
        self._current_source_regions = regionprops(
            source_mask.to_numpy(),
            intensity_image=source_img.to_numpy() if source_img is not None else None,
        )
        self._current_target_regions = regionprops(
            target_mask.to_numpy(),
            intensity_image=target_img.to_numpy() if target_img is not None else None,
        )

    def _match_masks(
        self,
        source_mask: xr.DataArray,
        target_mask: xr.DataArray,
        source_img: Optional[xr.DataArray] = None,
        target_img: Optional[xr.DataArray] = None,
        transform: Optional[np.ndarray] = None,
    ) -> xr.DataArray:
        source_points = compute_points(
            source_mask,
            regions=self._current_source_regions,
            points_feature=self.points_feature,
        )
        target_points = compute_points(
            target_mask,
            regions=self._current_target_regions,
            points_feature=self.points_feature,
        )
        source_intensities = None
        if source_img is not None:
            source_intensities = compute_intensities(
                source_img,
                source_mask,
                regions=self._current_source_regions,
                intensities_feature=self.intensities_feature,
            )
        target_intensities = None
        if target_img is not None:
            target_intensities = compute_intensities(
                target_img,
                target_mask,
                regions=self._current_target_regions,
                intensities_feature=self.intensities_feature,
            )
        return self.match_points(
            source_points,
            target_points,
            source_intensities=source_intensities,
            target_intensities=target_intensities,
            source_shape=source_mask.shape,
            target_shape=target_mask.shape,
            transform=transform,
        )

    def _post_match_masks(
        self,
        scores: xr.DataArray,
        source_mask: xr.DataArray,
        target_mask: xr.DataArray,
        source_img: Optional[xr.DataArray] = None,
        target_img: Optional[xr.DataArray] = None,
        transform: Optional[np.ndarray] = None,
    ) -> xr.DataArray:
        scores = super(PointsMatchingAlgorithm, self)._post_match_masks(
            scores, source_mask, target_mask, source_img, target_img, transform
        )
        self._current_source_mask = None
        self._current_target_mask = None
        self._current_source_img = None
        self._current_target_img = None
        self._current_source_regions = None
        self._current_target_regions = None
        return scores

    def match_points(
        self,
        source_points: pd.DataFrame,
        target_points: pd.DataFrame,
        source_intensities: Optional[pd.DataFrame] = None,
        target_intensities: Optional[pd.DataFrame] = None,
        source_shape: Optional[Tuple[int, int]] = None,
        target_shape: Optional[Tuple[int, int]] = None,
        transform: Optional[np.ndarray] = None,
    ) -> xr.DataArray:
        self._pre_match_points(
            source_points,
            target_points,
            source_intensities=source_intensities,
            target_intensities=target_intensities,
            source_shape=source_shape,
            target_shape=target_shape,
            transform=transform,
        )
        scores = self._match_points(
            source_points,
            target_points,
            source_intensities=source_intensities,
            target_intensities=target_intensities,
            source_shape=source_shape,
            target_shape=target_shape,
            transform=transform,
        )
        scores = self._post_match_points(
            scores,
            source_points,
            target_points,
            source_intensities=source_intensities,
            target_intensities=target_intensities,
            source_shape=source_shape,
            target_shape=target_shape,
            transform=transform,
        )
        return scores

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
        pass

    @abstractmethod
    def _match_points(
        self,
        source_points: pd.DataFrame,
        target_points: pd.DataFrame,
        source_intensities: Optional[pd.DataFrame] = None,
        target_intensities: Optional[pd.DataFrame] = None,
        source_shape: Optional[Tuple[int, int]] = None,
        target_shape: Optional[Tuple[int, int]] = None,
        transform: Optional[np.ndarray] = None,
    ) -> xr.DataArray:
        raise NotImplementedError()

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
        return scores


class IterativePointsMatchingAlgorithm(PointsMatchingAlgorithm):
    TRANSFORM_TYPES: dict[str, ProjectiveTransform] = {
        "euclidean": EuclideanTransform,
        "similarity": SimilarityTransform,
        "affine": AffineTransform,
    }

    def __init__(
        self,
        max_iter: int,
        top_k_estim: int,
        transform_type: str = "affine",
        points_feature: str = "centroid",
        intensities_feature: str = "intensity_mean",
    ) -> None:
        super(IterativePointsMatchingAlgorithm, self).__init__(
            points_feature, intensities_feature
        )
        self.max_iter = max_iter
        self.top_k_estim = top_k_estim
        self.transform_type = self.TRANSFORM_TYPES[transform_type]

    def _match_points(
        self,
        source_points: pd.DataFrame,
        target_points: pd.DataFrame,
        source_intensities: Optional[pd.DataFrame] = None,
        target_intensities: Optional[pd.DataFrame] = None,
        source_shape: Optional[Tuple[int, int]] = None,
        target_shape: Optional[Tuple[int, int]] = None,
        transform: Optional[np.ndarray] = None,
    ) -> xr.DataArray:
        current_transform = transform
        for iteration in range(self.max_iter):
            self._pre_iter(
                iteration,
                source_points,
                target_points,
                source_intensities=source_intensities,
                target_intensities=target_intensities,
                source_shape=source_shape,
                target_shape=target_shape,
                transform=current_transform,
            )
            scores = self._iter(
                iteration,
                source_points,
                target_points,
                source_intensities=source_intensities,
                target_intensities=target_intensities,
                source_shape=source_shape,
                target_shape=target_shape,
                transform=current_transform,
            )
            updated_transform = self._update_transform()
            scores, updated_transform, stop = self._post_iter(
                iteration,
                scores,
                updated_transform,
                source_points,
                target_points,
                source_intensities=source_intensities,
                target_intensities=target_intensities,
                source_shape=source_shape,
                target_shape=target_shape,
                transform=current_transform,
            )
            if updated_transform is None or stop:
                break
            current_transform = updated_transform
        return scores

    def _pre_iter(
        self,
        iteration: int,
        source_points: pd.DataFrame,
        target_points: pd.DataFrame,
        source_shape: Optional[Tuple[int, int]] = None,
        target_shape: Optional[Tuple[int, int]] = None,
        source_intensities: Optional[pd.DataFrame] = None,
        target_intensities: Optional[pd.DataFrame] = None,
        transform: Optional[np.ndarray] = None,
    ) -> None:
        pass

    @abstractmethod
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
        raise NotImplementedError()

    def _update_transform(
        self,
        source_points: pd.DataFrame,
        target_points: pd.DataFrame,
        scores: xr.DataArray,
    ) -> Optional[np.ndarray]:
        max_source_scores = np.amax(scores, axis=1)
        top_source_indices = np.argpartition(-max_source_scores, self.top_k_estim - 1)[
            : self.top_k_estim
        ]
        top_target_indices = np.argmax(scores[top_source_indices, :], axis=1)
        top_source_indices = top_source_indices[max_source_scores > 0.0]
        top_target_indices = top_target_indices[max_source_scores > 0.0]
        updated_transform: ProjectiveTransform = self.transform_type()
        if updated_transform.estimate(
            source_points[top_source_indices], target_points[top_target_indices]
        ):
            return updated_transform.params
        return None

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
        return scores, updated_transform, False


class SpellmatchMatchingAlgorithmError(SpellmatchMatchingError):
    pass
