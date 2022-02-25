from abc import ABC, abstractmethod
from typing import Optional, Type

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
    filter_outlier_points,
    restore_outlier_scores,
    transform_bounding_box,
    transform_points,
)


# TODO matching logging


class _MaskMatchingMixin:
    def _init_mask_matching(self) -> None:
        pass

    def match_masks(
        self,
        source_mask: xr.DataArray,
        target_mask: xr.DataArray,
        source_img: Optional[xr.DataArray] = None,
        target_img: Optional[xr.DataArray] = None,
        transform: Optional[ProjectiveTransform] = None,
    ) -> xr.DataArray:
        return self._match_masks(
            source_mask, target_mask, source_img, target_img, transform
        )

    @abstractmethod
    def _match_masks(
        self,
        source_mask: xr.DataArray,
        target_mask: xr.DataArray,
        source_img: Optional[xr.DataArray],
        target_img: Optional[xr.DataArray],
        transform: Optional[ProjectiveTransform],
    ) -> xr.DataArray:
        raise NotImplementedError()


class _PointsMatchingMixin:
    def _init_points_matching(
        self,
        outlier_dist: Optional[float],
        points_feature: str,
        intensities_feature: str,
    ) -> None:
        self.outlier_dist = outlier_dist
        self.points_feature = points_feature
        self.intensities_feature = intensities_feature

    def _match_points_from_masks(
        self,
        source_mask: xr.DataArray,
        target_mask: xr.DataArray,
        source_img: Optional[xr.DataArray],
        target_img: Optional[xr.DataArray],
        transform: Optional[ProjectiveTransform],
    ) -> xr.DataArray:
        source_regions = regionprops(
            source_mask.to_numpy(),
            intensity_image=source_img.to_numpy() if source_img is not None else None,
        )
        target_regions = regionprops(
            target_mask.to_numpy(),
            intensity_image=target_img.to_numpy() if target_img is not None else None,
        )
        source_points = compute_points(
            source_mask, regions=source_regions, points_feature=self.points_feature
        )
        target_points = compute_points(
            target_mask, regions=target_regions, points_feature=self.points_feature
        )
        source_bbox = create_bounding_box(source_mask)
        target_bbox = create_bounding_box(target_mask)
        source_intensities = None
        if source_img is not None:
            source_intensities = compute_intensities(
                source_img,
                source_mask,
                regions=source_regions,
                intensities_feature=self.intensities_feature,
            )
        target_intensities = None
        if target_img is not None:
            target_intensities = compute_intensities(
                target_img,
                target_mask,
                regions=target_regions,
                intensities_feature=self.intensities_feature,
            )
        scores = self.match_points(
            source_points,
            target_points,
            source_bbox=source_bbox,
            target_bbox=target_bbox,
            source_intensities=source_intensities,
            target_intensities=target_intensities,
            transform=transform,
        )
        return scores

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
        transformed_source_points = source_points
        transformed_source_bbox = source_bbox
        if transform is not None:
            transformed_source_points = transform_points(source_points, transform)
            if source_bbox is not None:
                transformed_source_bbox = transform_bounding_box(source_bbox, transform)
        filtered_source_points = transformed_source_points
        filtered_target_points = target_points
        filtered_source_intensities = source_intensities
        filtered_target_intensities = target_intensities
        if self.outlier_dist is not None:
            if target_bbox is not None:
                filtered_source_points = filter_outlier_points(
                    transformed_source_points, target_bbox, self.outlier_dist
                )
                if source_intensities is not None:
                    filtered_source_intensities = source_intensities.loc[
                        filtered_source_points.index
                    ]

            if transformed_source_bbox is not None:
                filtered_target_points = filter_outlier_points(
                    target_points, transformed_source_bbox, self.outlier_dist
                )
                if target_intensities is not None:
                    filtered_target_intensities = target_intensities.loc[
                        filtered_target_points.index
                    ]
        filtered_scores = self._match_points(
            filtered_source_points,
            filtered_target_points,
            filtered_source_intensities,
            filtered_target_intensities,
        )
        scores = restore_outlier_scores(
            source_points.index, target_points.index, filtered_scores
        )
        return scores

    @abstractmethod
    def _match_points(
        self,
        source_points: pd.DataFrame,
        target_points: pd.DataFrame,
        source_intensities: Optional[pd.DataFrame],
        target_intensities: Optional[pd.DataFrame],
    ) -> xr.DataArray:
        raise NotImplementedError()


class _GraphMatchingMixin:
    def _init_graph_matching(self, adj_radius: float) -> None:
        self.adj_radius = adj_radius

    def _match_graphs_from_points(
        self,
        source_points: pd.DataFrame,
        target_points: pd.DataFrame,
        source_intensities: Optional[pd.DataFrame],
        target_intensities: Optional[pd.DataFrame],
    ) -> xr.DataArray:
        source_adj, source_dists = create_graph(
            source_points, self.adj_radius, "a", "b"
        )
        target_adj, target_dists = create_graph(
            target_points, self.adj_radius, "x", "y"
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
        return self._match_graphs(
            source_adj,
            target_adj,
            source_dists,
            target_dists,
            source_intensities,
            target_intensities,
        )

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


class MatchingAlgorithm(ABC):
    pass


class MaskMatchingAlgorithm(MatchingAlgorithm, _MaskMatchingMixin):
    def __init__(self) -> None:
        super(MaskMatchingAlgorithm, self).__init__()
        self._init_mask_matching()


class PointsMatchingAlgorithm(MaskMatchingAlgorithm, _PointsMatchingMixin):
    def __init__(
        self,
        outlier_dist: Optional[float],
        points_feature: str,
        intensities_feature: str,
    ) -> None:
        super(PointsMatchingAlgorithm, self).__init__()
        self._init_points_matching(outlier_dist, points_feature, intensities_feature)

    def _match_masks(
        self,
        source_mask: xr.DataArray,
        target_mask: xr.DataArray,
        source_img: Optional[xr.DataArray],
        target_img: Optional[xr.DataArray],
        transform: Optional[ProjectiveTransform],
    ) -> xr.DataArray:
        return self._match_points_from_masks(
            source_mask, target_mask, source_img, target_img, transform
        )


class IterativePointsMatchingAlgorithm(PointsMatchingAlgorithm):
    TRANSFORM_TYPES: dict[str, Type[ProjectiveTransform]] = {
        "euclidean": EuclideanTransform,
        "similarity": SimilarityTransform,
        "affine": AffineTransform,
    }

    def __init__(
        self,
        max_iter: int,
        transform_type: str,
        transform_estim_top_k: int,
        outlier_dist: Optional[float],
        points_feature: str,
        intensities_feature: str,
    ) -> None:
        super(IterativePointsMatchingAlgorithm, self).__init__(
            outlier_dist, points_feature, intensities_feature
        )
        self.max_iter = max_iter
        self.transform_type = self.TRANSFORM_TYPES[transform_type]
        self.transform_estim_top_k = transform_estim_top_k

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
        current_transform = transform
        for iteration in range(self.max_iter):
            self._pre_iter(iteration, current_transform)
            current_scores = super(IterativePointsMatchingAlgorithm, self).match_points(
                source_points,
                target_points,
                source_bbox=source_bbox,
                target_bbox=target_bbox,
                source_intensities=source_intensities,
                target_intensities=target_intensities,
                transform=current_transform,
            )
            updated_transform = self._update_transform(
                source_points, target_points, current_scores
            )
            stop = self._post_iter(
                iteration, current_transform, current_scores, updated_transform
            )
            if updated_transform is None or stop:
                break
            current_transform = updated_transform
        return current_scores

    def _pre_iter(
        self, iteration: int, current_transform: Optional[ProjectiveTransform]
    ) -> None:
        pass

    def _update_transform(
        self,
        source_points: pd.DataFrame,
        target_points: pd.DataFrame,
        scores: xr.DataArray,
    ) -> Optional[ProjectiveTransform]:
        max_source_scores = np.amax(scores, axis=1)
        top_source_indices = np.argpartition(
            -max_source_scores, self.transform_estim_top_k - 1
        )[: self.transform_estim_top_k]
        top_target_indices = np.argmax(scores[top_source_indices, :], axis=1)
        top_source_indices = top_source_indices[max_source_scores > 0.0]
        top_target_indices = top_target_indices[max_source_scores > 0.0]
        updated_transform = self.transform_type()
        if updated_transform.estimate(
            source_points[top_source_indices], target_points[top_target_indices]
        ):
            return updated_transform
        return None

    def _post_iter(
        self,
        iteration: int,
        current_transform: Optional[ProjectiveTransform],
        current_scores: xr.DataArray,
        updated_transform: Optional[ProjectiveTransform],
    ) -> bool:
        return False


class GraphMatchingAlgorithm(PointsMatchingAlgorithm, _GraphMatchingMixin):
    def __init__(
        self,
        adj_radius: float,
        outlier_dist: Optional[float],
        points_feature: str,
        intensities_feature: str,
    ) -> None:
        super(GraphMatchingAlgorithm, self).__init__(
            outlier_dist, points_feature, intensities_feature
        )
        self._init_graph_matching(adj_radius)

    def _match_points(
        self,
        source_points: pd.DataFrame,
        target_points: pd.DataFrame,
        source_intensities: Optional[pd.DataFrame],
        target_intensities: Optional[pd.DataFrame],
    ) -> xr.DataArray:
        return self._match_graphs_from_points(
            source_points, target_points, source_intensities, target_intensities
        )


class IterativeGraphMatchingAlgorithm(
    IterativePointsMatchingAlgorithm, _GraphMatchingMixin
):
    def __init__(
        self,
        adj_radius: float,
        max_iter: int,
        transform_type: str,
        transform_estim_top_k: int,
        outlier_dist: Optional[float],
        points_feature: str,
        intensities_feature: str,
    ) -> None:
        super(IterativeGraphMatchingAlgorithm, self).__init__(
            max_iter,
            transform_type,
            transform_estim_top_k,
            outlier_dist,
            points_feature,
            intensities_feature,
        )
        self._init_graph_matching(adj_radius)

    def _match_points(
        self,
        source_points: pd.DataFrame,
        target_points: pd.DataFrame,
        source_intensities: Optional[pd.DataFrame],
        target_intensities: Optional[pd.DataFrame],
    ) -> xr.DataArray:
        return self._match_graphs_from_points(
            source_points, target_points, source_intensities, target_intensities
        )
