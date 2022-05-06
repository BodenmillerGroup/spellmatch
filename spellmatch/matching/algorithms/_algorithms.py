import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, MutableMapping
from enum import Enum
from timeit import default_timer as timer
from typing import Any, Optional, Union

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
        cache: Optional[MutableMapping[str, Any]] = None,
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
        source_img: Optional[xr.DataArray],
        target_img: Optional[xr.DataArray],
        prior_transform: Optional[ProjectiveTransform],
        cache: Optional[MutableMapping[str, Any]],
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
        cache: Optional[MutableMapping[str, Any]],
    ) -> tuple[dict[str, Any], xr.DataArray]:
        raise NotImplementedError()

    def _post_match_masks(
        self,
        source_mask: xr.DataArray,
        target_mask: xr.DataArray,
        source_img: Optional[xr.DataArray],
        target_img: Optional[xr.DataArray],
        prior_transform: Optional[ProjectiveTransform],
        cache: Optional[MutableMapping[str, Any]],
        info: dict[str, Any],
        scores: xr.DataArray,
    ) -> tuple[dict[str, Any], xr.DataArray]:
        return info, scores


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
        cache: Optional[MutableMapping[str, Any]],
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
            logger.info("Extracting points from masks")
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
        source_bbox: Optional[Polygon] = None,
        target_bbox: Optional[Polygon] = None,
        source_intensities: Optional[pd.DataFrame] = None,
        target_intensities: Optional[pd.DataFrame] = None,
        prior_transform: Optional[ProjectiveTransform] = None,
        cache: Optional[MutableMapping[str, Any]] = None,
    ) -> tuple[dict[str, Any], xr.DataArray]:
        c = cache if cache is not None else {}
        if "transformed_source_points" not in c:
            c["transformed_source_points"] = source_points
            if prior_transform is not None:
                logger.info("Transforming source points")
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
            c["filtered_source_intensities"] = source_intensities
            c["filtered_target_intensities"] = target_intensities
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
                if source_bbox is not None:
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
        source_intensities: Optional[pd.DataFrame],
        target_intensities: Optional[pd.DataFrame],
        cache: Optional[MutableMapping[str, Any]],
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
        cache: Optional[MutableMapping[str, Any]],
    ) -> tuple[dict[str, Any], xr.DataArray]:
        raise NotImplementedError()

    def _post_match_points(
        self,
        source_name: str,
        target_name: str,
        source_points: pd.DataFrame,
        target_points: pd.DataFrame,
        source_intensities: Optional[pd.DataFrame],
        target_intensities: Optional[pd.DataFrame],
        cache: Optional[MutableMapping[str, Any]],
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
        source_intensities: Optional[pd.DataFrame],
        target_intensities: Optional[pd.DataFrame],
        cache: Optional[MutableMapping[str, Any]],
    ) -> tuple[dict[str, Any], xr.DataArray]:
        c = cache if cache is not None else {}
        if any(
            key not in c
            for key in ["source_adj", "target_adj", "source_dists", "target_dists"]
        ):
            logger.info("Constructing graphs from points")
            c["source_adj"], c["source_dists"] = create_graph(
                source_name, source_points, self.adj_radius, "a", "b"
            )
            c["target_adj"], c["target_dists"] = create_graph(
                target_name, target_points, self.adj_radius, "x", "y"
            )
        info, scores = self.match_graphs(
            c["source_adj"],
            c["target_adj"],
            source_dists=c["source_dists"],
            target_dists=c["target_dists"],
            source_intensities=source_intensities,
            target_intensities=target_intensities,
            cache=cache,
        )
        return info, scores

    def match_graphs(
        self,
        source_adj: xr.DataArray,
        target_adj: xr.DataArray,
        source_dists: Optional[xr.DataArray] = None,
        target_dists: Optional[xr.DataArray] = None,
        source_intensities: Optional[pd.DataFrame] = None,
        target_intensities: Optional[pd.DataFrame] = None,
        cache: Optional[MutableMapping[str, Any]] = None,
    ) -> tuple[dict[str, Any], xr.DataArray]:
        self._pre_match_graphs(
            source_adj,
            target_adj,
            source_dists,
            target_dists,
            source_intensities,
            target_intensities,
            cache,
        )
        info, scores = self._match_graphs(
            source_adj,
            target_adj,
            source_dists,
            target_dists,
            source_intensities,
            target_intensities,
            cache,
        )
        info, scores = self._post_match_graphs(
            source_adj,
            target_adj,
            source_dists,
            target_dists,
            source_intensities,
            target_intensities,
            cache,
            info,
            scores,
        )
        return info, scores

    def _pre_match_graphs(
        self,
        source_adj: xr.DataArray,
        target_adj: xr.DataArray,
        source_dists: Optional[xr.DataArray],
        target_dists: Optional[xr.DataArray],
        source_intensities: Optional[pd.DataFrame],
        target_intensities: Optional[pd.DataFrame],
        cache: Optional[MutableMapping[str, Any]],
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
        cache: Optional[MutableMapping[str, Any]],
    ) -> tuple[dict[str, Any], xr.DataArray]:
        raise NotImplementedError()

    def _post_match_graphs(
        self,
        source_adj: xr.DataArray,
        target_adj: xr.DataArray,
        source_dists: Optional[xr.DataArray],
        target_dists: Optional[xr.DataArray],
        source_intensities: Optional[pd.DataFrame],
        target_intensities: Optional[pd.DataFrame],
        cache: Optional[MutableMapping[str, Any]],
        info: dict[str, Any],
        scores: xr.DataArray,
    ) -> tuple[dict[str, Any], xr.DataArray]:
        return info, scores


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
        cache: Optional[MutableMapping[str, Any]],
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
        outlier_dist: Optional[float],
        point_feature: str,
        intensity_feature: str,
        intensity_transform: Union[str, Callable[[np.ndarray], np.ndarray], None],
        transform_type: Union[str, type[ProjectiveTransform]],
        transform_estim_type: Union[str, TransformEstimationType],
        transform_estim_k_best: Optional[int],
        max_iter: int,
        scores_tol: Optional[float],
        transform_tol: Optional[float],
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
        seconds = 0
        converged = False
        last_scores = None
        current_transform = prior_transform
        iteration_cache = dict(cache) if cache is not None else {}
        if prior_transform is not None:
            logger.info(f"Initial transform: {describe_transform(prior_transform)}")
        else:
            logger.info("Initial transform: None")
        for iteration in range(self.max_iter):
            start = timer()
            logger.info(f"Iterative algorithm iteration {iteration + 1}")
            self._prepare_iteration_cache(iteration, iteration_cache)
            info, scores = super(IterativePointsMatchingAlgorithm, self).match_points(
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
                logger.warning("Failed to fit updated transform")
                break
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
                f"Iterative algorithm did not converge after {self.max_iter} "
                f"iterations ({seconds:.3f} seconds)"
            )
        info["iterations"] = iteration + 1
        info["converged"] = converged
        return info, scores

    def _prepare_iteration_cache(
        self, iteration: int, iteration_cache: dict[str, Any]
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

    def _check_convergence(
        self,
        last_scores: Optional[xr.DataArray],
        current_scores: xr.DataArray,
        current_transform: Optional[ProjectiveTransform],
        updated_transform: Optional[ProjectiveTransform],
    ) -> bool:
        converged = False
        scores_loss = float("inf")
        if last_scores is not None:
            scores_loss = np.linalg.norm(current_scores - last_scores)
            if self.scores_tol is not None and scores_loss < self.scores_tol:
                converged = True
        logger.info(f"Scores loss: {scores_loss:.9f}")
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
        cache: Optional[MutableMapping[str, Any]],
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
        intensity_transform: Union[str, Callable[[np.ndarray], np.ndarray], None],
        transform_type: Union[str, type[ProjectiveTransform]],
        transform_estim_type: Union[
            str, IterativePointsMatchingAlgorithm.TransformEstimationType
        ],
        transform_estim_k_best: Optional[int],
        max_iter: int,
        scores_tol: Optional[float],
        transform_tol: Optional[float],
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
        )
        self._init_graph_matching(adj_radius=adj_radius)

    def _prepare_iteration_cache(
        self, iteration: int, iteration_cache: dict[str, Any]
    ) -> None:
        super(IterativeGraphMatchingAlgorithm, self)._prepare_iteration_cache(
            iteration, iteration_cache
        )
        if iteration > 0 and self.outlier_dist is not None:
            # point filtering has been updated --> cached graph is invalid
            for key in ["source_adj", "target_adj", "source_dists", "target_dists"]:
                iteration_cache.pop(key, None)

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
        return self._match_graphs_from_points(
            source_name,
            target_name,
            source_points,
            target_points,
            source_intensities,
            target_intensities,
            cache,
        )


class SpellmatchMatchingAlgorithmException(SpellmatchMatchingException):
    pass
