import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Callable, Optional, Type, Union

import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import Polygon
from skimage.measure import regionprops
from skimage.transform import (
    AffineTransform,
    EuclideanTransform,
    ProjectiveTransform,
    SimilarityTransform,
)

from ...utils import (
    compute_intensities,
    compute_points,
    create_bounding_box,
    create_graph,
    describe_transform,
    filter_outlier_points,
    get_function_by_name,
    restore_outlier_scores,
    transform_bounding_box,
    transform_points,
)
from .. import SpellmatchMatchingException

logger = logging.getLogger(__name__.rpartition(".")[0])


class _MaskMatchingMixin:
    def _init_mask_matching(self) -> None:
        pass

    def match_masks(
        self,
        source_mask: xr.DataArray,
        target_mask: xr.DataArray,
        source_img: Optional[xr.DataArray] = None,
        target_img: Optional[xr.DataArray] = None,
        prior_transform: Optional[ProjectiveTransform] = None,
    ) -> xr.DataArray:
        self._pre_match_masks(
            source_mask, target_mask, source_img, target_img, prior_transform
        )
        scores = self._match_masks(
            source_mask, target_mask, source_img, target_img, prior_transform
        )
        scores = self._post_match_masks(
            source_mask, target_mask, source_img, target_img, prior_transform, scores
        )
        return scores

    def _pre_match_masks(
        self,
        source_mask: xr.DataArray,
        target_mask: xr.DataArray,
        source_img: Optional[xr.DataArray],
        target_img: Optional[xr.DataArray],
        prior_transform: Optional[ProjectiveTransform],
    ) -> None:
        pass

    @abstractmethod
    def _match_masks(
        self,
        source_mask: xr.DataArray,
        target_mask: xr.DataArray,
        source_img: Optional[xr.DataArray],
        target_img: Optional[xr.DataArray],
        prior_transform: Optional[ProjectiveTransform],
    ) -> xr.DataArray:
        raise NotImplementedError()

    def _post_match_masks(
        self,
        source_mask: xr.DataArray,
        target_mask: xr.DataArray,
        source_img: Optional[xr.DataArray],
        target_img: Optional[xr.DataArray],
        prior_transform: Optional[ProjectiveTransform],
        scores: xr.DataArray,
    ) -> xr.DataArray:
        return scores


class _PointsMatchingMixin:
    def _init_points_matching(
        self,
        *,
        outlier_dist: Optional[float],
        point_feature: str,
        intensity_feature: str,
        intensity_transform: Union[str, Callable[[np.ndarray], np.ndarray], None],
    ) -> None:
        if isinstance(intensity_transform, str):
            try:
                intensity_transform = get_function_by_name(intensity_transform)
            except (ImportError, AttributeError) as e:
                raise SpellmatchMatchingAlgorithmException(
                    f"Failed to import function '{intensity_transform}': {e}"
                )
        self.outlier_dist = outlier_dist
        self.point_feature = point_feature
        self.intensity_feature = intensity_feature
        self.intensity_transform = intensity_transform

    def _match_points_from_masks(
        self,
        source_mask: xr.DataArray,
        target_mask: xr.DataArray,
        source_img: Optional[xr.DataArray],
        target_img: Optional[xr.DataArray],
        prior_transform: Optional[ProjectiveTransform],
    ) -> xr.DataArray:
        logger.info("Extracting points from masks")
        source_regions = regionprops(
            source_mask.to_numpy(),
            intensity_image=np.moveaxis(source_img.to_numpy(), 0, -1)
            if source_img is not None
            else None,
        )
        target_regions = regionprops(
            target_mask.to_numpy(),
            intensity_image=np.moveaxis(target_img.to_numpy(), 0, -1)
            if target_img is not None
            else None,
        )
        source_points = compute_points(
            source_mask, regions=source_regions, point_feature=self.point_feature
        )
        target_points = compute_points(
            target_mask, regions=target_regions, point_feature=self.point_feature
        )
        source_bbox = create_bounding_box(source_mask)
        target_bbox = create_bounding_box(target_mask)
        source_intensities = None
        if source_img is not None:
            source_intensities = compute_intensities(
                source_img,
                source_mask,
                regions=source_regions,
                intensity_feature=self.intensity_feature,
                intensity_transform=self.intensity_transform,
            )
        target_intensities = None
        if target_img is not None:
            target_intensities = compute_intensities(
                target_img,
                target_mask,
                regions=target_regions,
                intensity_feature=self.intensity_feature,
                intensity_transform=self.intensity_transform,
            )
        scores = self.match_points(
            source_mask.name or "source",
            target_mask.name or "target",
            source_points,
            target_points,
            source_bbox=source_bbox,
            target_bbox=target_bbox,
            source_intensities=source_intensities,
            target_intensities=target_intensities,
            prior_transform=prior_transform,
        )
        return scores

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
    ) -> xr.DataArray:
        transformed_source_points = source_points
        transformed_source_bbox = source_bbox
        if prior_transform is not None:
            transformed_source_points = transform_points(source_points, prior_transform)
            if source_bbox is not None:
                transformed_source_bbox = transform_bounding_box(
                    source_bbox, prior_transform
                )
        filtered_transformed_source_points = transformed_source_points
        filtered_target_points = target_points
        filtered_source_intensities = source_intensities
        filtered_target_intensities = target_intensities
        if self.outlier_dist is not None:
            if target_bbox is not None:
                filtered_transformed_source_points = filter_outlier_points(
                    transformed_source_points, target_bbox, self.outlier_dist
                )
                if source_intensities is not None:
                    filtered_source_intensities = source_intensities.loc[
                        filtered_transformed_source_points.index, :
                    ]
            if transformed_source_bbox is not None:
                filtered_target_points = filter_outlier_points(
                    target_points, transformed_source_bbox, self.outlier_dist
                )
                if target_intensities is not None:
                    filtered_target_intensities = target_intensities.loc[
                        filtered_target_points.index, :
                    ]
        self._pre_match_points(
            source_name,
            target_name,
            filtered_transformed_source_points,
            filtered_target_points,
            filtered_source_intensities,
            filtered_target_intensities,
        )
        filtered_scores = self._match_points(
            source_name,
            target_name,
            filtered_transformed_source_points,
            filtered_target_points,
            filtered_source_intensities,
            filtered_target_intensities,
        )
        filtered_scores = self._post_match_points(
            source_name,
            target_name,
            filtered_transformed_source_points,
            filtered_target_points,
            filtered_source_intensities,
            filtered_target_intensities,
            filtered_scores,
        )
        scores = filtered_scores
        if self.outlier_dist is not None:
            scores = restore_outlier_scores(
                source_points.index.to_numpy(),
                target_points.index.to_numpy(),
                filtered_scores,
            )
        return scores

    def _pre_match_points(
        self,
        source_name: str,
        target_name: str,
        source_points: pd.DataFrame,
        target_points: pd.DataFrame,
        source_intensities: Optional[pd.DataFrame],
        target_intensities: Optional[pd.DataFrame],
    ) -> None:
        pass

    @abstractmethod
    def _match_points(
        self,
        source_name: str,
        target_name: str,
        source_points: pd.DataFrame,
        target_points: pd.DataFrame,
        source_intensities: Optional[pd.DataFrame],
        target_intensities: Optional[pd.DataFrame],
    ) -> xr.DataArray:
        raise NotImplementedError()

    def _post_match_points(
        self,
        source_name: str,
        target_name: str,
        source_points: pd.DataFrame,
        target_points: pd.DataFrame,
        source_intensities: Optional[pd.DataFrame],
        target_intensities: Optional[pd.DataFrame],
        scores: xr.DataArray,
    ) -> xr.DataArray:
        return scores


class _GraphMatchingMixin:
    def _init_graph_matching(self, *, adj_radius: float) -> None:
        self.adj_radius = adj_radius

    def _match_graphs_from_points(
        self,
        source_name: str,
        target_name: str,
        source_points: pd.DataFrame,
        target_points: pd.DataFrame,
        source_intensities: Optional[pd.DataFrame],
        target_intensities: Optional[pd.DataFrame],
    ) -> xr.DataArray:
        logger.info("Constructing graphs from points")
        source_adj, source_dists = create_graph(
            source_name, source_points, self.adj_radius, "a", "b"
        )
        target_adj, target_dists = create_graph(
            target_name, target_points, self.adj_radius, "x", "y"
        )
        scores = self.match_graphs(
            source_adj,
            target_adj,
            source_dists=source_dists,
            target_dists=target_dists,
            source_intensities=source_intensities,
            target_intensities=target_intensities,
        )
        return scores

    def match_graphs(
        self,
        source_adj: xr.DataArray,
        target_adj: xr.DataArray,
        source_dists: Optional[xr.DataArray] = None,
        target_dists: Optional[xr.DataArray] = None,
        source_intensities: Optional[pd.DataFrame] = None,
        target_intensities: Optional[pd.DataFrame] = None,
    ) -> xr.DataArray:
        self._pre_match_graphs(
            source_adj,
            target_adj,
            source_dists,
            target_dists,
            source_intensities,
            target_intensities,
        )
        scores = self._match_graphs(
            source_adj,
            target_adj,
            source_dists,
            target_dists,
            source_intensities,
            target_intensities,
        )
        scores = self._post_match_graphs(
            source_adj,
            target_adj,
            source_dists,
            target_dists,
            source_intensities,
            target_intensities,
            scores,
        )
        return scores

    def _pre_match_graphs(
        self,
        source_adj: xr.DataArray,
        target_adj: xr.DataArray,
        source_dists: Optional[xr.DataArray],
        target_dists: Optional[xr.DataArray],
        source_intensities: Optional[pd.DataFrame],
        target_intensities: Optional[pd.DataFrame],
    ) -> None:
        pass

    @abstractmethod
    def _match_graphs(
        self,
        source_adj: xr.DataArray,
        target_adj: xr.DataArray,
        source_dists: Optional[xr.DataArray],
        target_dists: Optional[xr.DataArray],
        source_intensities: Optional[pd.DataFrame],
        target_intensities: Optional[pd.DataFrame],
    ) -> xr.DataArray:
        raise NotImplementedError()

    def _post_match_graphs(
        self,
        source_adj: xr.DataArray,
        target_adj: xr.DataArray,
        source_dists: Optional[xr.DataArray],
        target_dists: Optional[xr.DataArray],
        source_intensities: Optional[pd.DataFrame],
        target_intensities: Optional[pd.DataFrame],
        scores: xr.DataArray,
    ) -> xr.DataArray:
        return scores


class MatchingAlgorithm(ABC):
    pass


class MaskMatchingAlgorithm(MatchingAlgorithm, _MaskMatchingMixin):
    def __init__(self) -> None:
        super(MaskMatchingAlgorithm, self).__init__()
        self._init_mask_matching()


class PointsMatchingAlgorithm(MaskMatchingAlgorithm, _PointsMatchingMixin):
    def __init__(
        self,
        *,
        outlier_dist: Optional[float],
        point_feature: str,
        intensity_feature: str,
        intensity_transform: Union[str, Callable[[np.ndarray], np.ndarray], None],
    ) -> None:
        super(PointsMatchingAlgorithm, self).__init__()
        self._init_points_matching(
            outlier_dist=outlier_dist,
            point_feature=point_feature,
            intensity_feature=intensity_feature,
            intensity_transform=intensity_transform,
        )

    def _match_masks(
        self,
        source_mask: xr.DataArray,
        target_mask: xr.DataArray,
        source_img: Optional[xr.DataArray],
        target_img: Optional[xr.DataArray],
        prior_transform: Optional[ProjectiveTransform],
    ) -> xr.DataArray:
        return self._match_points_from_masks(
            source_mask, target_mask, source_img, target_img, prior_transform
        )


class IterativePointsMatchingAlgorithm(PointsMatchingAlgorithm):
    TRANSFORM_TYPES: dict[str, Type[ProjectiveTransform]] = {
        "rigid": EuclideanTransform,
        "similarity": SimilarityTransform,
        "affine": AffineTransform,
    }

    class TransformEstimationType(Enum):
        MAX_SCORE = "max_score"
        MAX_MARGIN = "max_margin"

    def __init__(
        self,
        *,
        outlier_dist: Optional[float],
        point_feature: str,
        intensity_feature: str,
        intensity_transform: Union[str, Callable[[np.ndarray], np.ndarray], None],
        transform_type: Union[str, Type[ProjectiveTransform]],
        transform_estim_type: Union[str, TransformEstimationType],
        transform_estim_k_best: Optional[int],
        max_iter: int,
        scores_tol: Optional[float],
        transform_tol: Optional[float],
        require_convergence: bool,
    ) -> None:
        if isinstance(transform_type, str):
            transform_type = self.TRANSFORM_TYPES[transform_type]
        if isinstance(transform_estim_type, str):
            transform_estim_type = self.TransformEstimationType(transform_estim_type)
        super(IterativePointsMatchingAlgorithm, self).__init__(
            outlier_dist=outlier_dist,
            point_feature=point_feature,
            intensity_feature=intensity_feature,
            intensity_transform=intensity_transform,
        )
        self.transform_type = transform_type
        self.transform_estim_type = transform_estim_type
        self.transform_estim_k_best = transform_estim_k_best
        self.max_iter = max_iter
        self.scores_tol = scores_tol
        self.transform_tol = transform_tol
        self.require_convergence = require_convergence
        self._last_scores: Optional[xr.DataArray] = None

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
    ) -> xr.DataArray:
        self._last_scores = None
        current_transform = prior_transform
        updated_transform = None
        converged = False
        for iteration in range(self.max_iter):
            self._pre_iter(iteration, current_transform)
            current_scores = super(IterativePointsMatchingAlgorithm, self).match_points(
                source_name,
                target_name,
                source_points,
                target_points,
                source_bbox=source_bbox,
                target_bbox=target_bbox,
                source_intensities=source_intensities,
                target_intensities=target_intensities,
                prior_transform=current_transform,
            )
            updated_transform = self._update_transform(
                source_points, target_points, current_scores
            )
            converged = self._post_iter(
                iteration, current_transform, current_scores, updated_transform
            )
            if updated_transform is None or converged:
                break
            current_transform = updated_transform
        if not converged:
            if self.require_convergence:
                raise SpellmatchMatchingAlgorithmException(
                    f"Iterative algorithm did not converge after {self.max_iter} "
                    f"iterations (updated_transform={updated_transform})"
                )
            else:
                logger.warning(
                    f"Iterative algorithm did not converge after {self.max_iter} "
                    f"iterations (updated_transform={updated_transform})"
                )
        return current_scores

    def _pre_iter(
        self, iteration: int, current_transform: Optional[ProjectiveTransform]
    ) -> None:
        logger.info(f"Iterative algorithm iteration {iteration + 1}")

    def _update_transform(
        self,
        source_points: pd.DataFrame,
        target_points: pd.DataFrame,
        scores: xr.DataArray,
    ) -> Optional[ProjectiveTransform]:
        if self.transform_estim_type == self.TransformEstimationType.MAX_SCORE:
            max_score_ind = np.argmax(scores.to_numpy(), axis=1)
            max_scores = np.take_along_axis(
                scores.to_numpy(), np.expand_dims(max_score_ind, axis=1), axis=1
            ).squeeze(axis=1)
            if self.transform_estim_k_best is not None:
                source_ind = np.argpartition(
                    -max_scores, self.transform_estim_k_best - 1
                )[: self.transform_estim_k_best]
            else:
                source_ind = np.arange(len(source_points.index))
            source_ind = source_ind[max_scores[source_ind] > 0]
            target_ind = max_score_ind[source_ind]
        elif self.transform_estim_type == self.TransformEstimationType.MAX_MARGIN:
            max2_score_ind = np.argpartition(-scores.to_numpy(), 1, axis=1)[:, :2]
            max2_scores = np.take_along_axis(scores.to_numpy(), max2_score_ind, axis=1)
            if self.transform_estim_k_best is not None:
                margins = max2_scores[:, 0] - max2_scores[:, 1]
                source_ind = np.argpartition(-margins, self.transform_estim_k_best - 1)[
                    : self.transform_estim_k_best
                ]
            else:
                source_ind = np.arange(len(source_points.index))
            source_ind = source_ind[max2_scores[source_ind, 0] > 0]
            target_ind = max2_score_ind[source_ind, 0]
        else:
            raise NotImplementedError()
        updated_transform = self.transform_type()
        if (
            len(source_ind) > 0
            and len(target_ind) > 0
            and updated_transform.estimate(
                source_points.iloc[source_ind, :].to_numpy(),
                target_points.iloc[target_ind, :].to_numpy(),
            )
        ):
            return updated_transform
        return None

    def _post_iter(
        self,
        iteration: int,
        current_transform: Optional[ProjectiveTransform],
        current_scores: xr.DataArray,
        updated_transform: Optional[ProjectiveTransform],
        converged: bool = False,
    ) -> bool:
        scores_loss = float("inf")
        if self._last_scores is not None:
            scores_loss = np.linalg.norm(current_scores - self._last_scores)
        if self.scores_tol is not None and scores_loss < self.scores_tol:
            converged = True
        transform_loss = float("inf")
        if current_transform is not None and updated_transform is not None:
            transform_loss = np.linalg.norm(
                updated_transform.params - current_transform.params
            )
        if self.transform_tol is not None and transform_loss < self.transform_tol:
            converged = True
        logger.info(
            f"scores_loss={scores_loss:.6f}, "
            f"transform_loss={transform_loss:.6f} "
            f"({describe_transform(updated_transform)}), "
            f"converged={converged}"
        )
        self._last_scores = current_scores
        return converged


class GraphMatchingAlgorithm(PointsMatchingAlgorithm, _GraphMatchingMixin):
    def __init__(
        self,
        *,
        point_feature: str,
        intensity_feature: str,
        intensity_transform: Union[str, Callable[[np.ndarray], np.ndarray], None],
        filter_outliers: bool,
        adj_radius: float,
    ) -> None:
        super(GraphMatchingAlgorithm, self).__init__(
            outlier_dist=adj_radius if filter_outliers else None,
            point_feature=point_feature,
            intensity_feature=intensity_feature,
            intensity_transform=intensity_transform,
        )
        self._init_graph_matching(adj_radius=adj_radius)

    def _match_points(
        self,
        source_name: str,
        target_name: str,
        source_points: pd.DataFrame,
        target_points: pd.DataFrame,
        source_intensities: Optional[pd.DataFrame],
        target_intensities: Optional[pd.DataFrame],
    ) -> xr.DataArray:
        return self._match_graphs_from_points(
            source_name,
            target_name,
            source_points,
            target_points,
            source_intensities,
            target_intensities,
        )


class IterativeGraphMatchingAlgorithm(
    IterativePointsMatchingAlgorithm, _GraphMatchingMixin
):
    def __init__(
        self,
        *,
        point_feature: str,
        intensity_feature: str,
        intensity_transform: Union[str, Callable[[np.ndarray], np.ndarray], None],
        transform_type: Union[str, Type[ProjectiveTransform]],
        transform_estim_type: Union[
            str, IterativePointsMatchingAlgorithm.TransformEstimationType
        ],
        transform_estim_k_best: Optional[int],
        max_iter: int,
        scores_tol: Optional[float],
        transform_tol: Optional[float],
        require_convergence: bool,
        filter_outliers: bool,
        adj_radius: float,
    ) -> None:
        super(IterativeGraphMatchingAlgorithm, self).__init__(
            outlier_dist=adj_radius if filter_outliers else None,
            point_feature=point_feature,
            intensity_feature=intensity_feature,
            intensity_transform=intensity_transform,
            transform_type=transform_type,
            transform_estim_type=transform_estim_type,
            transform_estim_k_best=transform_estim_k_best,
            max_iter=max_iter,
            scores_tol=scores_tol,
            transform_tol=transform_tol,
            require_convergence=require_convergence,
        )
        self._init_graph_matching(adj_radius=adj_radius)

    def _match_points(
        self,
        source_name: str,
        target_name: str,
        source_points: pd.DataFrame,
        target_points: pd.DataFrame,
        source_intensities: Optional[pd.DataFrame],
        target_intensities: Optional[pd.DataFrame],
    ) -> xr.DataArray:
        return self._match_graphs_from_points(
            source_name,
            target_name,
            source_points,
            target_points,
            source_intensities,
            target_intensities,
        )


class SpellmatchMatchingAlgorithmException(SpellmatchMatchingException):
    pass
