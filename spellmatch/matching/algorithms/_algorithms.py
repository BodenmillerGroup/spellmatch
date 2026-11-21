import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, MutableMapping
from enum import Enum
from timeit import default_timer as timer
from typing import Any

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

from ...sparse_graph import SparseGraph
from ...utils import (
    compute_intensities,
    compute_max2,
    compute_points,
    create_bounding_box,
    describe_transform,
    filter_outlier_points,
    get_function_by_name,
    make_unique_names,
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
        source_img: xr.DataArray | None = None,
        target_img: xr.DataArray | None = None,
        prior_transform: ProjectiveTransform | None = None,
        cache: MutableMapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any], xr.DataArray]:
        self._pre_match_masks(
            source_mask, target_mask, source_img, target_img, prior_transform, cache
        )
        info, scores = self._match_masks(
            source_mask, target_mask, source_img, target_img, prior_transform, cache
        )
        info, scores = self._post_match_masks(
            source_mask,
            target_mask,
            source_img,
            target_img,
            prior_transform,
            cache,
            info,
            scores,
        )
        return info, scores

    def _pre_match_masks(
        self,
        source_mask: xr.DataArray,
        target_mask: xr.DataArray,
        source_img: xr.DataArray | None,
        target_img: xr.DataArray | None,
        prior_transform: ProjectiveTransform | None,
        cache: MutableMapping[str, Any] | None,
    ) -> None:
        pass

    @abstractmethod
    def _match_masks(
        self,
        source_mask: xr.DataArray,
        target_mask: xr.DataArray,
        source_img: xr.DataArray | None,
        target_img: xr.DataArray | None,
        prior_transform: ProjectiveTransform | None,
        cache: MutableMapping[str, Any] | None,
    ) -> tuple[dict[str, Any], xr.DataArray]:
        raise NotImplementedError()

    def _post_match_masks(
        self,
        source_mask: xr.DataArray,
        target_mask: xr.DataArray,
        source_img: xr.DataArray | None,
        target_img: xr.DataArray | None,
        prior_transform: ProjectiveTransform | None,
        cache: MutableMapping[str, Any] | None,
        info: dict[str, Any],
        scores: xr.DataArray,
    ) -> tuple[dict[str, Any], xr.DataArray]:
        return info, scores


class _PointsMatchingMixin:
    def _init_points_matching(
        self,
        *,
        outlier_dist: float | None,
        point_feature: str,
        intensity_feature: str,
        intensity_transform: str | Callable[[np.ndarray], np.ndarray] | None,
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
        source_img: xr.DataArray | None,
        target_img: xr.DataArray | None,
        prior_transform: ProjectiveTransform | None,
        cache: MutableMapping[str, Any] | None,
    ) -> tuple[dict[str, Any], xr.DataArray]:
        c = cache if cache is not None else {}
        if any(
            key not in c
            for key in [
                "source_points",
                "target_points",
                "source_intensities",
                "target_intensities",
            ]
        ):
            logger.debug("Extracting points from masks")
            source_regions = regionprops(
                source_mask.to_numpy(),
                intensity_image=np.moveaxis(
                    source_img.to_numpy(), 1 if source_img.ndim == 4 else 0, -1
                )
                if source_img is not None
                else None,
            )
            target_regions = regionprops(
                target_mask.to_numpy(),
                intensity_image=np.moveaxis(
                    target_img.to_numpy(), 1 if target_img.ndim == 4 else 0, -1
                )
                if target_img is not None
                else None,
            )
            c["source_points"] = compute_points(
                source_mask, regions=source_regions, point_feature=self.point_feature
            )
            c["target_points"] = compute_points(
                target_mask, regions=target_regions, point_feature=self.point_feature
            )
            c["source_intensities"] = None
            if source_img is not None:
                c["source_intensities"] = compute_intensities(
                    source_img,
                    source_mask,
                    regions=source_regions,
                    intensity_feature=self.intensity_feature,
                    intensity_transform=self.intensity_transform,
                )
            c["target_intensities"] = None
            if target_img is not None:
                c["target_intensities"] = compute_intensities(
                    target_img,
                    target_mask,
                    regions=target_regions,
                    intensity_feature=self.intensity_feature,
                    intensity_transform=self.intensity_transform,
                )
        # TODO add support for 3D bounding boxes
        source_bbox = None
        if source_mask.ndim == 2:
            source_bbox = create_bounding_box(source_mask)
        target_bbox = None
        if target_mask.ndim == 2:
            target_bbox = create_bounding_box(target_mask)
        info, scores = self.match_points(
            source_mask.name or "source",
            target_mask.name or "target",
            c["source_points"],
            c["target_points"],
            source_bbox=source_bbox,
            target_bbox=target_bbox,
            source_intensities=c["source_intensities"],
            target_intensities=c["target_intensities"],
            prior_transform=prior_transform,
            cache=cache,
        )
        return info, scores

    def match_points(
        self,
        source_name: str,
        target_name: str,
        source_points: pd.DataFrame,
        target_points: pd.DataFrame,
        source_bbox: Polygon | None = None,
        target_bbox: Polygon | None = None,
        source_intensities: pd.DataFrame | None = None,
        target_intensities: pd.DataFrame | None = None,
        prior_transform: ProjectiveTransform | None = None,
        cache: MutableMapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any], xr.DataArray]:
        source_name, target_name = make_unique_names(source_name, target_name)
        c = cache if cache is not None else {}
        c["prior_transform"] = prior_transform
        if "transformed_source_points" not in c:
            c["transformed_source_points"] = source_points
            if prior_transform is not None:
                logger.debug("Transforming source points")
                c["transformed_source_points"] = transform_points(
                    source_points, prior_transform
                )
        if any(
            key not in c
            for key in [
                "filtered_transformed_source_points",
                "filtered_target_points",
                "filtered_source_intensities",
                "filtered_target_intensities",
            ]
        ):
            c["filtered_transformed_source_points"] = c["transformed_source_points"]
            c["filtered_target_points"] = target_points
            c["filtered_source_intensities"] = _align_intensities(
                source_intensities, source_points, "source"
            )
            c["filtered_target_intensities"] = _align_intensities(
                target_intensities, target_points, "target"
            )
            if self.outlier_dist is not None:
                if source_bbox is None or target_bbox is None:
                    raise SpellmatchMatchingAlgorithmException(
                        "Outlier filtering requires bounding box information"
                    )
                c["filtered_transformed_source_points"] = filter_outlier_points(
                    c["transformed_source_points"], target_bbox, self.outlier_dist
                )
                if source_intensities is not None:
                    c["filtered_source_intensities"] = source_intensities.loc[
                        c["filtered_transformed_source_points"].index, :
                    ]
                transformed_source_bbox = source_bbox
                if prior_transform is not None:
                    transformed_source_bbox = transform_bounding_box(
                        source_bbox, prior_transform
                    )
                c["filtered_target_points"] = filter_outlier_points(
                    target_points, transformed_source_bbox, self.outlier_dist
                )
                if target_intensities is not None:
                    c["filtered_target_intensities"] = target_intensities.loc[
                        c["filtered_target_points"].index, :
                    ]
        self._pre_match_points(
            source_name,
            target_name,
            c["filtered_transformed_source_points"],
            c["filtered_target_points"],
            c["filtered_source_intensities"],
            c["filtered_target_intensities"],
            cache,
        )
        info, filtered_scores = self._match_points(
            source_name,
            target_name,
            c["filtered_transformed_source_points"],
            c["filtered_target_points"],
            c["filtered_source_intensities"],
            c["filtered_target_intensities"],
            cache,
        )
        info, filtered_scores = self._post_match_points(
            source_name,
            target_name,
            c["filtered_transformed_source_points"],
            c["filtered_target_points"],
            c["filtered_source_intensities"],
            c["filtered_target_intensities"],
            cache,
            info,
            filtered_scores,
        )
        scores = filtered_scores
        if self.outlier_dist is not None:
            scores = restore_outlier_scores(
                source_points.index.to_numpy(),
                target_points.index.to_numpy(),
                filtered_scores,
            )
        return info, scores

    def _pre_match_points(
        self,
        source_name: str,
        target_name: str,
        source_points: pd.DataFrame,
        target_points: pd.DataFrame,
        source_intensities: pd.DataFrame | None,
        target_intensities: pd.DataFrame | None,
        cache: MutableMapping[str, Any] | None,
    ) -> None:
        pass

    @abstractmethod
    def _match_points(
        self,
        source_name: str,
        target_name: str,
        source_points: pd.DataFrame,
        target_points: pd.DataFrame,
        source_intensities: pd.DataFrame | None,
        target_intensities: pd.DataFrame | None,
        cache: MutableMapping[str, Any] | None,
    ) -> tuple[dict[str, Any], xr.DataArray]:
        raise NotImplementedError()

    def _post_match_points(
        self,
        source_name: str,
        target_name: str,
        source_points: pd.DataFrame,
        target_points: pd.DataFrame,
        source_intensities: pd.DataFrame | None,
        target_intensities: pd.DataFrame | None,
        cache: MutableMapping[str, Any] | None,
        info: dict[str, Any],
        scores: xr.DataArray,
    ) -> tuple[dict[str, Any], xr.DataArray]:
        return info, scores


class _GraphMatchingMixin:
    def _init_graph_matching(self, *, adj_radius: float) -> None:
        self.adj_radius = adj_radius

    def _match_graphs_from_points(
        self,
        source_name: str,
        target_name: str,
        source_points: pd.DataFrame,
        target_points: pd.DataFrame,
        source_intensities: pd.DataFrame | None,
        target_intensities: pd.DataFrame | None,
        cache: MutableMapping[str, Any] | None,
    ) -> tuple[dict[str, Any], xr.DataArray]:
        c = cache if cache is not None else {}
        if any(key not in c for key in ["source_graph", "target_graph"]):
            logger.debug("Constructing graphs from points")
            c["source_graph"] = SparseGraph.from_points(
                source_name, source_points, self.adj_radius
            )
            c["target_graph"] = SparseGraph.from_points(
                target_name, target_points, self.adj_radius
            )
        info, scores = self.match_graphs(
            c["source_graph"],
            c["target_graph"],
            source_intensities=source_intensities,
            target_intensities=target_intensities,
            cache=cache,
        )
        return info, scores

    def match_graphs(
        self,
        source_adj: xr.DataArray | SparseGraph,
        target_adj: xr.DataArray | SparseGraph,
        source_dists: xr.DataArray | None = None,
        target_dists: xr.DataArray | None = None,
        source_intensities: pd.DataFrame | None = None,
        target_intensities: pd.DataFrame | None = None,
        cache: MutableMapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any], xr.DataArray]:
        """Match graphs, given as sparse graphs or dense adjacency/distance arrays."""
        source_graph = source_adj
        if not isinstance(source_graph, SparseGraph):
            source_graph = SparseGraph.from_dataarrays(source_adj, source_dists)
        target_graph = target_adj
        if not isinstance(target_graph, SparseGraph):
            target_graph = SparseGraph.from_dataarrays(target_adj, target_dists)
        self._pre_match_graphs(
            source_graph, target_graph, source_intensities, target_intensities, cache
        )
        info, scores = self._match_graphs(
            source_graph, target_graph, source_intensities, target_intensities, cache
        )
        info, scores = self._post_match_graphs(
            source_graph,
            target_graph,
            source_intensities,
            target_intensities,
            cache,
            info,
            scores,
        )
        return info, scores

    def _pre_match_graphs(
        self,
        source_graph: SparseGraph,
        target_graph: SparseGraph,
        source_intensities: pd.DataFrame | None,
        target_intensities: pd.DataFrame | None,
        cache: MutableMapping[str, Any] | None,
    ) -> None:
        pass

    @abstractmethod
    def _match_graphs(
        self,
        source_graph: SparseGraph,
        target_graph: SparseGraph,
        source_intensities: pd.DataFrame | None,
        target_intensities: pd.DataFrame | None,
        cache: MutableMapping[str, Any] | None,
    ) -> tuple[dict[str, Any], xr.DataArray]:
        raise NotImplementedError()

    def _post_match_graphs(
        self,
        source_graph: SparseGraph,
        target_graph: SparseGraph,
        source_intensities: pd.DataFrame | None,
        target_intensities: pd.DataFrame | None,
        cache: MutableMapping[str, Any] | None,
        info: dict[str, Any],
        scores: xr.DataArray,
    ) -> tuple[dict[str, Any], xr.DataArray]:
        return info, scores


class MatchingAlgorithm(ABC):
    pass


class MaskMatchingAlgorithm(MatchingAlgorithm, _MaskMatchingMixin):
    def __init__(self) -> None:
        super().__init__()
        self._init_mask_matching()


class PointsMatchingAlgorithm(MaskMatchingAlgorithm, _PointsMatchingMixin):
    def __init__(
        self,
        *,
        outlier_dist: float | None,
        point_feature: str,
        intensity_feature: str,
        intensity_transform: str | Callable[[np.ndarray], np.ndarray] | None,
    ) -> None:
        super().__init__()
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
        source_img: xr.DataArray | None,
        target_img: xr.DataArray | None,
        prior_transform: ProjectiveTransform | None,
        cache: MutableMapping[str, Any] | None,
    ) -> tuple[dict[str, Any], xr.DataArray]:
        return self._match_points_from_masks(
            source_mask,
            target_mask,
            source_img,
            target_img,
            prior_transform,
            cache,
        )


class IterativePointsMatchingAlgorithm(PointsMatchingAlgorithm):
    TRANSFORM_TYPES: dict[str, type[ProjectiveTransform]] = {
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
        outlier_dist: float | None,
        point_feature: str,
        intensity_feature: str,
        intensity_transform: str | Callable[[np.ndarray], np.ndarray] | None,
        transform_type: str | type[ProjectiveTransform],
        transform_estim_type: str | TransformEstimationType,
        transform_estim_k_best: int | None,
        max_iter: int,
        scores_tol: float | None,
        transform_tol: float | None,
    ) -> None:
        if isinstance(transform_type, str):
            transform_type = self.TRANSFORM_TYPES[transform_type]
        if isinstance(transform_estim_type, str):
            transform_estim_type = self.TransformEstimationType(transform_estim_type)
        super().__init__(
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

    def match_points(
        self,
        source_name: str,
        target_name: str,
        source_points: pd.DataFrame,
        target_points: pd.DataFrame,
        source_bbox: Polygon | None = None,
        target_bbox: Polygon | None = None,
        source_intensities: pd.DataFrame | None = None,
        target_intensities: pd.DataFrame | None = None,
        prior_transform: ProjectiveTransform | None = None,
        cache: MutableMapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any], xr.DataArray]:
        source_name, target_name = make_unique_names(source_name, target_name)
        seconds = 0
        converged = False
        last_scores = None
        current_transform = prior_transform
        iteration_cache = dict(cache) if cache is not None else {}
        if prior_transform is not None:
            logger.info(f"Initial transform: {describe_transform(prior_transform)}")
        else:
            logger.info("Initial transform: None")
        # transform under which the cached graphs were built
        graph_transform = current_transform
        for iteration in range(self.max_iter):
            start = timer()
            logger.info(f"Iterative algorithm iteration {iteration + 1}")
            distances_changed = False
            if iteration > 0 and not _preserves_distances(
                graph_transform, current_transform
            ):
                distances_changed = True
                graph_transform = current_transform
            self._prepare_iteration_cache(iteration, iteration_cache, distances_changed)
            info, scores = super().match_points(
                source_name,
                target_name,
                source_points,
                target_points,
                source_bbox=source_bbox,
                target_bbox=target_bbox,
                source_intensities=source_intensities,
                target_intensities=target_intensities,
                prior_transform=current_transform,
                cache=iteration_cache,
            )
            updated_transform = self._update_transform(
                source_points, target_points, scores
            )
            if updated_transform is None:
                break  # reason logged by _update_transform
            logger.info(f"Updated transform: {describe_transform(updated_transform)}")
            stop = timer()
            seconds += stop - start
            if self._check_convergence(
                last_scores, scores, current_transform, updated_transform
            ):
                converged = True
                logger.info(
                    f"Converged after {iteration + 1} iterations "
                    f"({seconds:.3f} seconds)"
                )
                break
            last_scores = scores
            current_transform = updated_transform
        if not converged:
            logger.warning(
                f"Iterative algorithm did not converge after {iteration + 1} "
                f"iterations ({seconds:.3f} seconds)"
            )
        info["iterations"] = iteration + 1
        info["converged"] = converged
        return info, scores

    def _prepare_iteration_cache(
        self, iteration: int, iteration_cache: dict[str, Any], distances_changed: bool
    ) -> None:
        if iteration > 0:
            # transform has been updated --> cached point transform/filtering is invalid
            for key in [
                "transformed_source_points",
                "filtered_transformed_source_points",
                "filtered_target_points",
                "filtered_source_intensities",
                "filtered_target_intensities",
            ]:
                iteration_cache.pop(key, None)

    def _update_transform(
        self,
        source_points: pd.DataFrame,
        target_points: pd.DataFrame,
        scores: xr.DataArray,
    ) -> ProjectiveTransform | None:
        scores_data = scores.to_numpy()
        max_score_ind = np.argmax(scores_data, axis=1)
        max_scores = np.take_along_axis(
            scores_data, np.expand_dims(max_score_ind, axis=1), axis=1
        ).squeeze(axis=1)
        if self.transform_estim_type == self.TransformEstimationType.MAX_SCORE:
            rankings = max_scores
        elif self.transform_estim_type == self.TransformEstimationType.MAX_MARGIN:
            max2_scores = compute_max2(scores_data, axis=1)
            rankings = max2_scores[:, 0] - max2_scores[:, 1]
        else:
            raise NotImplementedError()
        source_ind = np.arange(len(source_points.index))
        k_best = self.transform_estim_k_best
        if k_best is not None and k_best < len(source_ind):
            source_ind = np.argpartition(-rankings, k_best - 1)[:k_best]
        source_ind = source_ind[max_scores[source_ind] > 0]
        target_ind = max_score_ind[source_ind]
        if len(source_ind) == 0:
            logger.warning(
                "Failed to estimate transform: no cells with positive scores"
            )
            return None
        updated_transform = self.transform_type.from_estimate(
            source_points.iloc[source_ind, :].to_numpy(),
            target_points.iloc[target_ind, :].to_numpy(),
        )
        if not updated_transform:
            logger.warning(f"Failed to estimate transform: {updated_transform}")
            return None
        return updated_transform

    def _check_convergence(
        self,
        last_scores: xr.DataArray | None,
        current_scores: xr.DataArray,
        current_transform: ProjectiveTransform | None,
        updated_transform: ProjectiveTransform | None,
    ) -> bool:
        converged = False
        scores_loss = float("inf")
        if last_scores is not None:
            scores_loss = float(np.linalg.norm(current_scores - last_scores))
            current_scores_norm = float(np.linalg.norm(current_scores))
            if current_scores_norm > 0:
                scores_loss /= current_scores_norm
            if self.scores_tol is not None and scores_loss < self.scores_tol:
                converged = True
            logger.info(f"Scores loss (relative): {scores_loss:.9f}")
        transform_loss = float("inf")
        if current_transform is not None and updated_transform is not None:
            transform_loss = np.linalg.norm(
                updated_transform.params - current_transform.params
            )
            if self.transform_tol is not None and transform_loss < self.transform_tol:
                converged = True
            logger.info(f"Transform loss: {transform_loss:.9f}")
        return converged


class GraphMatchingAlgorithm(PointsMatchingAlgorithm, _GraphMatchingMixin):
    def __init__(
        self,
        *,
        point_feature: str,
        intensity_feature: str,
        intensity_transform: str | Callable[[np.ndarray], np.ndarray] | None,
        filter_outliers: bool,
        adj_radius: float,
    ) -> None:
        super().__init__(
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
        source_intensities: pd.DataFrame | None,
        target_intensities: pd.DataFrame | None,
        cache: MutableMapping[str, Any] | None,
    ) -> tuple[dict[str, Any], xr.DataArray]:
        return self._match_graphs_from_points(
            source_name,
            target_name,
            source_points,
            target_points,
            source_intensities,
            target_intensities,
            cache,
        )


class IterativeGraphMatchingAlgorithm(
    IterativePointsMatchingAlgorithm, _GraphMatchingMixin
):
    def __init__(
        self,
        *,
        point_feature: str,
        intensity_feature: str,
        intensity_transform: str | Callable[[np.ndarray], np.ndarray] | None,
        transform_type: str | type[ProjectiveTransform],
        transform_estim_type: str
        | IterativePointsMatchingAlgorithm.TransformEstimationType,
        transform_estim_k_best: int | None,
        max_iter: int,
        scores_tol: float | None,
        transform_tol: float | None,
        filter_outliers: bool,
        adj_radius: float,
    ) -> None:
        super().__init__(
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
        )
        self._init_graph_matching(adj_radius=adj_radius)

    def _prepare_iteration_cache(
        self, iteration: int, iteration_cache: dict[str, Any], distances_changed: bool
    ) -> None:
        super()._prepare_iteration_cache(iteration, iteration_cache, distances_changed)
        if iteration > 0 and (self.outlier_dist is not None or distances_changed):
            # point filtering or source point distances have been updated
            # --> cached graphs are invalid
            for key in ["source_graph", "target_graph"]:
                iteration_cache.pop(key, None)

    def _match_points(
        self,
        source_name: str,
        target_name: str,
        source_points: pd.DataFrame,
        target_points: pd.DataFrame,
        source_intensities: pd.DataFrame | None,
        target_intensities: pd.DataFrame | None,
        cache: MutableMapping[str, Any] | None,
    ) -> tuple[dict[str, Any], xr.DataArray]:
        return self._match_graphs_from_points(
            source_name,
            target_name,
            source_points,
            target_points,
            source_intensities,
            target_intensities,
            cache,
        )


def _align_intensities(
    intensities: pd.DataFrame | None, points: pd.DataFrame, name: str
) -> pd.DataFrame | None:
    if intensities is None:
        return None
    missing_labels = points.index.difference(intensities.index)
    if len(missing_labels) > 0:
        raise SpellmatchMatchingAlgorithmException(
            f"Missing {name} intensities for {len(missing_labels)} cells, e.g. "
            f"{', '.join(map(str, missing_labels[:5]))}"
        )
    return intensities.loc[points.index, :]


def _preserves_distances(
    transform1: ProjectiveTransform | None,
    transform2: ProjectiveTransform | None,
    atol: float = 1e-6,
) -> bool:
    """Check whether pairwise distances are the same under both transforms."""
    if transform1 is transform2:
        return True
    params1 = transform1.params if transform1 is not None else None
    params2 = transform2.params if transform2 is not None else None
    ndim = (params1 if params1 is not None else params2).shape[0] - 1
    if params1 is None:
        params1 = np.eye(ndim + 1)
    if params2 is None:
        params2 = np.eye(ndim + 1)
    if np.allclose(params1, params2, atol=atol):
        return True  # same transform (incl. projective transforms)
    for params in (params1, params2):
        if not np.allclose(params[-1, :], np.eye(ndim + 1)[-1, :], atol=atol):
            return False  # projective
    # pairwise distances under an affine transform with linear part L are determined
    # by the metric tensor L^T L (no inversion needed, also for singular L)
    linear1, linear2 = params1[:-1, :-1], params2[:-1, :-1]
    return np.allclose(linear1.T @ linear1, linear2.T @ linear2, atol=atol)


class SpellmatchMatchingAlgorithmException(SpellmatchMatchingException):
    pass
