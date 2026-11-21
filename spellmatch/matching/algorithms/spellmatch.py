import logging
import warnings
from collections.abc import Callable, MutableMapping
from dataclasses import dataclass
from enum import Enum
from timeit import default_timer as timer
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import TruncatedSVD
from sklearn.exceptions import ConvergenceWarning
from sklearn.neighbors import NearestNeighbors

from ..._spellmatch import hookimpl
from ...sparse_graph import SparseGraph
from ...utils import make_unique_names
from ._algorithms import (
    IterativeGraphMatchingAlgorithm,
    MaskMatchingAlgorithm,
    SpellmatchMatchingAlgorithmException,
)

logger = logging.getLogger(__name__)


@hookimpl
def spellmatch_get_mask_matching_algorithm(
    name: str | None,
) -> type[MaskMatchingAlgorithm] | None | list[str]:
    algorithms = {
        "spellmatch": Spellmatch,
    }
    if name is not None:
        return algorithms.get(name)
    return list(algorithms.keys())


class Spellmatch(IterativeGraphMatchingAlgorithm):
    class SharedIntensityMetric(Enum):
        # Pearson correlation distance across the SVD embedding coordinates
        CORRELATION = "correlation"
        # cosine distance between the SVD embeddings (equivalent to the cosine distance
        # between the rank-k SVD reconstructions)
        COSINE = "cosine"
        # Pearson correlation distance across markers between the rank-k SVD
        # reconstructions (as in Zhu et al.)
        RECONSTRUCTED_CORRELATION = "reconstructed_correlation"

    class AllIntensityMetric(Enum):
        # Pearson correlation distance across the CCA embedding coordinates
        CORRELATION = "correlation"
        # cosine distance between the CCA embeddings
        COSINE = "cosine"

    def __init__(
        self,
        *,
        point_feature: str = "centroid",
        intensity_feature: str = "intensity_mean",
        intensity_transform: str | Callable[[np.ndarray], np.ndarray] | None = None,
        transform_type: str = "rigid",
        transform_estim_type: str = "max_score",
        transform_estim_k_best: int = 50,
        max_iter: int = 50,
        scores_tol: float | None = 1e-3,
        transform_tol: float | None = None,
        filter_outliers: bool = True,
        adj_radius: float = 15,
        alpha: float = 0.8,
        spatial_cdist_prior_thres: float | None = None,
        max_spatial_cdist: float | None = None,
        degree_weight: float = 0,
        degree_cdiff_thres: int = 3,
        intensity_weight: float = 0,
        intensity_interp_lmd: int | float = 11,
        intensity_interp_cca_n_components: int = 10,
        intensity_shared_pca_n_components: int = 5,
        intensity_shared_metric: str = "reconstructed_correlation",
        intensity_all_cca_fit_k_closest: int = 500,
        intensity_all_cca_fit_k_most_certain: int = 100,
        intensity_all_cca_n_components: int = 10,
        intensity_all_metric: str = "cosine",
        intensity_1d_cdiff_thres_sd: float = 2.0,
        distance_weight: float = 0,
        distance_cdiff_thres: float = 5,
        cca_max_iter: int = 500,
        cca_tol: float = 1e-6,
        opt_max_iter: int = 200,
        opt_tol: float = 1e-6,
        precision: str = "float32",
    ) -> None:
        super().__init__(
            point_feature=point_feature,
            intensity_feature=intensity_feature,
            intensity_transform=intensity_transform,
            transform_type=transform_type,
            transform_estim_type=transform_estim_type,
            transform_estim_k_best=transform_estim_k_best,
            max_iter=max_iter,
            scores_tol=scores_tol,
            transform_tol=transform_tol,
            filter_outliers=filter_outliers,
            adj_radius=adj_radius,
        )
        self.alpha = alpha
        self.spatial_cdist_prior_thres = spatial_cdist_prior_thres
        self.max_spatial_cdist = max_spatial_cdist
        self.degree_weight = degree_weight
        self.degree_cdiff_thres = degree_cdiff_thres
        self.intensity_weight = intensity_weight
        self.intensity_interp_lmd = intensity_interp_lmd
        self.intensity_interp_cca_n_components = intensity_interp_cca_n_components
        self.intensity_shared_pca_n_components = intensity_shared_pca_n_components
        self.intensity_shared_metric = self.SharedIntensityMetric(
            intensity_shared_metric
        )
        self.intensity_all_cca_fit_k_closest = intensity_all_cca_fit_k_closest
        self.intensity_all_cca_fit_k_most_certain = intensity_all_cca_fit_k_most_certain
        self.intensity_all_cca_n_components = intensity_all_cca_n_components
        self.intensity_all_metric = self.AllIntensityMetric(intensity_all_metric)
        self.intensity_1d_cdiff_thres_sd = intensity_1d_cdiff_thres_sd
        self.distance_weight = distance_weight
        self.distance_cdiff_thres = distance_cdiff_thres
        self.cca_max_iter = cca_max_iter
        self.cca_tol = cca_tol
        self.opt_max_iter = opt_max_iter
        self.opt_tol = opt_tol
        self.precision = np.dtype(precision).type
        self._current_source_points: pd.DataFrame | None = None
        self._current_target_points: pd.DataFrame | None = None
        self._warned: set[str] = set()

    def _warn_once(self, key: str, message: str) -> None:
        """Log a warning once per matching run (e.g., not for every lambda value)."""
        if key not in self._warned:
            self._warned.add(key)
            logger.warning(message)

    def _prepare_iteration_cache(
        self, iteration: int, iteration_cache: dict[str, Any], distances_changed: bool
    ) -> None:
        super()._prepare_iteration_cache(iteration, iteration_cache, distances_changed)
        if iteration == 0:
            self._warned.clear()
        if iteration > 0:
            # transform has been updated --> cached spatial cross-distance and
            # all-marker intensity cross-distance (fitted on nearest neighbors under
            # the current transform) are invalid
            iteration_cache.pop("spatial_cdist", None)
            iteration_cache.pop("intensity_cdist_all", None)
        if iteration > 0 and self.max_spatial_cdist is not None:
            # transform has been updated --> candidate cell pairs within
            # max_spatial_cdist (and thus the cached product graph) may have changed
            iteration_cache.pop("product_graph", None)
        if iteration > 0 and (self.outlier_dist is not None or distances_changed):
            # graphs have been updated --> cached degree cross-distance and product
            # graph are invalid
            iteration_cache.pop("degree_cdist", None)
            iteration_cache.pop("product_graph", None)
        if iteration > 0 and self.outlier_dist is not None:
            # point filtering has been updated --> cached cross-distances are invalid
            iteration_cache.pop("intensity_cdist_shared", None)

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
        self._current_source_points = source_points
        self._current_target_points = target_points
        info, scores = super()._match_graphs_from_points(
            source_name,
            target_name,
            source_points,
            target_points,
            source_intensities,
            target_intensities,
            cache,
        )
        self._current_source_points = None
        self._current_target_points = None
        return info, scores

    def _match_graphs(
        self,
        source_graph: SparseGraph,
        target_graph: SparseGraph,
        source_intensities: pd.DataFrame | None,
        target_intensities: pd.DataFrame | None,
        cache: MutableMapping[str, Any] | None,
    ) -> tuple[dict[str, Any], xr.DataArray]:
        c = cache if cache is not None else {}
        n1 = len(source_graph)
        n2 = len(target_graph)
        deg1 = source_graph.degrees.astype(self.precision)
        deg2 = target_graph.degrees.astype(self.precision)
        if "degree_cdist" not in c:
            c["degree_cdist"] = None
            if self.degree_weight > 0.0:
                logger.debug("Computing degree cross-distance")
                c["degree_cdist"] = self._compute_degree_cdist(deg1, deg2)
                assert c["degree_cdist"].dtype == self.precision
        if "intensity_cdist_shared" not in c:
            c["intensity_cdist_shared"] = None
            if self.intensity_weight > 0.0 and self.intensity_interp_lmd != 0.0:
                if source_intensities is None or target_intensities is None:
                    raise SpellmatchException(
                        "Intensities are required for computing their cross-distance"
                    )
                logger.debug("Computing intensity cross-distance (shared markers)")
                c["intensity_cdist_shared"] = self._compute_intensity_cdist_shared(
                    source_intensities, target_intensities
                )
                assert c["intensity_cdist_shared"].dtype == self.precision
        if "intensity_cdist_all" not in c:
            c["intensity_cdist_all"] = None
            if self.intensity_weight > 0.0 and self.intensity_interp_lmd != 1.0:
                if source_intensities is None or target_intensities is None:
                    raise SpellmatchException(
                        "Intensities are required for computing their cross-distance"
                    )
                if (
                    self._current_source_points is None
                    or self._current_target_points is None
                ):
                    raise SpellmatchException(
                        "Computing intensity cross-distances on all markers requires "
                        "running Spellmatch in point set registration mode"
                    )
                if c.get("prior_transform") is None:
                    logger.info(
                        "No spatial transform available, not using the intensity "
                        "cross-distance from all markers in this iteration"
                    )
                else:
                    logger.debug("Computing intensity cross-distance (all markers)")
                    c["intensity_cdist_all"] = self._compute_intensity_cdist_all(
                        self._current_source_points,
                        self._current_target_points,
                        source_intensities,
                        target_intensities,
                    )
                    assert c["intensity_cdist_all"].dtype == self.precision
        if self.distance_weight > 0 and (
            source_graph.dists is None or target_graph.dists is None
        ):
            raise SpellmatchException(
                "Distances are required for computing their cross-distance"
            )
        if "spatial_cdist" not in c:
            c["spatial_cdist"] = None
            if (
                self.alpha != 1.0 and self.spatial_cdist_prior_thres is not None
            ) or self.max_spatial_cdist is not None:
                if (
                    self._current_source_points is None
                    or self._current_target_points is None
                ):
                    raise SpellmatchException(
                        "Computing spatial cross-distance requires running Spellmatch "
                        "in point set registration or mask matching mode"
                    )
                if c.get("prior_transform") is None:
                    logger.info(
                        "No spatial transform available, not using the spatial prior "
                        "and the maximum spatial cross-distance in this iteration"
                    )
                else:
                    logger.debug("Computing spatial cross-distance")
                    c["spatial_cdist"] = np.asarray(
                        distance.cdist(
                            self._current_source_points.to_numpy(self.precision),
                            self._current_target_points.to_numpy(self.precision),
                        ),
                        dtype=self.precision,
                    )
        degree_cdist = c["degree_cdist"]
        intensity_cdist_shared = c["intensity_cdist_shared"]
        intensity_cdist_all = c["intensity_cdist_all"]
        spatial_cdist = c["spatial_cdist"]
        cell_pairs = None
        if self.max_spatial_cdist is not None and spatial_cdist is not None:
            cell_pairs = np.flatnonzero(spatial_cdist <= self.max_spatial_cdist)
            if len(cell_pairs) == 0:
                raise SpellmatchException(
                    "No candidate cell pairs within the maximum spatial cross-distance"
                )
            logger.info(
                f"Restricting to {len(cell_pairs)} of {n1 * n2} candidate cell pairs"
            )
            if degree_cdist is not None:
                degree_cdist = degree_cdist.ravel()[cell_pairs]
            if intensity_cdist_shared is not None:
                intensity_cdist_shared = intensity_cdist_shared.ravel()[cell_pairs]
            if intensity_cdist_all is not None:
                intensity_cdist_all = intensity_cdist_all.ravel()[cell_pairs]
            spatial_cdist = spatial_cdist.ravel()[cell_pairs]
        if "product_graph" not in c:
            logger.debug("Constructing product graph")
            c["product_graph"] = _ProductGraph.build(
                source_graph,
                target_graph,
                cell_pairs,
                self.precision,
                self.distance_cdiff_thres if self.distance_weight > 0 else None,
            )
        product_graph = c["product_graph"]
        if (
            intensity_cdist_shared is None
            or intensity_cdist_all is None
            or 0 <= self.intensity_interp_lmd <= 1
        ):
            # effective lambda (None if intensities are not used)
            if intensity_cdist_shared is not None and intensity_cdist_all is not None:
                lmd = self.intensity_interp_lmd
            elif intensity_cdist_shared is not None:
                lmd = 1.0  # shared markers only
            elif intensity_cdist_all is not None:
                lmd = 0.0  # all markers only
            else:
                lmd = None
            info, scores_data = self._match_graphs_for_lambda(
                n1,
                n2,
                cell_pairs,
                product_graph,
                degree_cdist,
                intensity_cdist_shared,
                intensity_cdist_all,
                spatial_cdist,
                lmd,
            )
        else:
            lmd = None
            info = None
            scores_data = None
            cancor_mean = None
            n_components = self.intensity_interp_cca_n_components
            n_source_features = len(source_intensities.columns)
            if n_components > n_source_features:
                self._warn_once(
                    "lambda_cca_source_features",
                    f"Requested number of intensity components for lambda estimation "
                    f"({n_components}) is larger than the number of source intensity "
                    f"features ({n_source_features}), continuing with "
                    f"{n_source_features} intensity components",
                )
                n_components = n_source_features
            n_target_features = len(target_intensities.columns)
            if n_components > n_target_features:
                self._warn_once(
                    "lambda_cca_target_features",
                    f"Requested number of intensity components for lambda estimation "
                    f"({n_components}) is larger than the number of target intensity "
                    f"features ({n_target_features}), continuing with "
                    f"{n_target_features} intensity components",
                )
                n_components = n_target_features
            for current_lmd in np.linspace(0, 1, int(self.intensity_interp_lmd)):
                logger.debug(f"Evaluating lambda={current_lmd:.3g}")
                current_info, current_scores_data = self._match_graphs_for_lambda(
                    n1,
                    n2,
                    cell_pairs,
                    product_graph,
                    degree_cdist,
                    intensity_cdist_shared,
                    intensity_cdist_all,
                    spatial_cdist,
                    current_lmd,
                )
                row_ind, col_ind = linear_sum_assignment(
                    current_scores_data, maximize=True
                )
                nonzero_mask = current_scores_data[row_ind, col_ind] > 0
                row_ind, col_ind = row_ind[nonzero_mask], col_ind[nonzero_mask]
                cca, cca_intensities1, cca_intensities2 = self._fit_cca(
                    n_components,
                    source_intensities.iloc[row_ind, :],
                    target_intensities.iloc[col_ind, :],
                )
                current_cancor_mean = np.mean(
                    np.diagonal(
                        np.corrcoef(cca_intensities1, cca_intensities2, rowvar=False),
                        offset=cca.n_components,
                    )
                )
                logger.debug(f"Canonical correlations mean={current_cancor_mean:.6f}")
                if cancor_mean is None or current_cancor_mean > cancor_mean:
                    lmd = current_lmd
                    info = current_info
                    scores_data = current_scores_data
                    cancor_mean = current_cancor_mean
            logger.info(f"Best lambda={lmd:.3g} (CC mean={cancor_mean:.6f})")
        source_name, target_name = make_unique_names(
            source_graph.name or "source", target_graph.name or "target"
        )
        scores = xr.DataArray(
            data=scores_data,
            coords={source_name: source_graph.labels, target_name: target_graph.labels},
        )
        return info, scores

    def _match_graphs_for_lambda(
        self,
        n1: int,
        n2: int,
        cell_pairs: np.ndarray | None,
        product_graph: "_ProductGraph",
        degree_cdist: np.ndarray | None,
        intensity_cdist_shared: np.ndarray | None,
        intensity_cdist_all: np.ndarray | None,
        spatial_cdist: np.ndarray | None,
        lmd: float | None,
    ) -> tuple[dict[str, Any], np.ndarray]:
        if intensity_cdist_shared is not None and intensity_cdist_all is not None:
            logger.debug("Combining intensity cross-distance matrices")
            intensity_cdist = np.asarray(
                self.precision(lmd) * intensity_cdist_shared
                + self.precision(1 - lmd) * intensity_cdist_all,
                dtype=self.precision,
            )
            assert intensity_cdist.dtype == self.precision
        elif intensity_cdist_shared is not None:
            intensity_cdist = intensity_cdist_shared
        elif intensity_cdist_all is not None:
            intensity_cdist = intensity_cdist_all
        else:
            intensity_cdist = None
        logger.debug("Initializing")
        # weighted cross-distances of the cell pairs (degree, intensity); unlike the
        # distance cross-distance (one value per pair of neighboring cell pairs), they
        # are applied as vector operations by _NormalizedSimilarityMatrix
        cell_pair_cdist = None
        total_weight = 0
        if self.degree_weight > 0:
            assert degree_cdist is not None
            weighted_degree_cdist = (
                self.precision(self.degree_weight) * degree_cdist.ravel()
            )
            cell_pair_cdist = (
                weighted_degree_cdist
                if cell_pair_cdist is None
                else cell_pair_cdist + weighted_degree_cdist
            )
            del weighted_degree_cdist
            total_weight += 2 * self.degree_weight
        if self.intensity_weight > 0 and intensity_cdist is not None:
            weighted_intensity_cdist = (
                self.precision(self.intensity_weight) * intensity_cdist.ravel()
            )
            cell_pair_cdist = (
                weighted_intensity_cdist
                if cell_pair_cdist is None
                else cell_pair_cdist + weighted_intensity_cdist
            )
            del weighted_intensity_cdist
            total_weight += 2 * self.intensity_weight
        if self.distance_weight > 0:
            total_weight += self.distance_weight
        if total_weight == 0:
            logger.info("No cross-distances used, matching graph topology only")
        w_tilde = _NormalizedSimilarityMatrix(
            product_graph,
            cell_pair_cdist,
            self.distance_weight,
            total_weight,
            self.precision,
        )
        n_cell_pairs = len(product_graph)
        if (
            self.alpha != 1.0
            and self.spatial_cdist_prior_thres is not None
            and spatial_cdist is not None
        ):
            h = np.ravel(
                1 - np.clip((spatial_cdist / self.spatial_cdist_prior_thres) ** 2, 0, 1)
            )
            if np.sum(h) == 0:
                raise SpellmatchException(
                    "No candidate cell pairs within the spatial prior threshold"
                )
            h = np.asarray(h / np.sum(h), dtype=self.precision)
        else:
            h = np.ones(n_cell_pairs, dtype=self.precision)
            h /= self.precision(n_cell_pairs)
        assert h.dtype == self.precision
        s = np.ones(n_cell_pairs, dtype=self.precision)
        s /= self.precision(n_cell_pairs)
        assert s.dtype == self.precision
        logger.debug("Optimizing")
        opt_converged = False
        seconds = 0
        for opt_iteration in range(self.opt_max_iter):
            start = timer()
            s_new: np.ndarray = (
                self.precision(self.alpha) * (w_tilde @ s)
                + self.precision(1.0 - self.alpha) * h
            )
            opt_loss = float(np.linalg.norm(s - s_new))
            assert s_new.dtype == self.precision
            s = s_new
            stop = timer()
            seconds += stop - start
            logger.debug(
                f"Optimizer iteration {opt_iteration:03d}: {opt_loss:.9f} "
                f"({stop - start:.3f} seconds)"
            )
            if opt_loss < self.opt_tol:
                opt_converged = True
                logger.debug(
                    f"Done after {opt_iteration + 1} iterations ({seconds:.3f} seconds)"
                )
                break
        if not opt_converged:
            self._warn_once(
                "opt_not_converged",
                f"Optimization did not converge after {self.opt_max_iter} iterations "
                f"({seconds:.3f} seconds, last loss: {opt_loss:.9f})",
            )
        info = {
            "lambda": lmd,
            "opt_iterations": opt_iteration + 1,
            "opt_converged": opt_converged,
        }
        scores_data = s
        if cell_pairs is not None:
            scores_data = np.zeros(n1 * n2, dtype=self.precision)
            scores_data[cell_pairs] = s
        return info, scores_data.reshape((n1, n2))

    def _compute_degree_cdist(self, deg1: np.ndarray, deg2: np.ndarray) -> np.ndarray:
        degree_cdist = np.subtract.outer(deg1, deg2).astype(self.precision, copy=False)
        degree_cdist /= self.precision(self.degree_cdiff_thres)
        np.square(degree_cdist, out=degree_cdist)
        np.clip(degree_cdist, 0, 1, out=degree_cdist)
        assert degree_cdist.dtype == self.precision
        return degree_cdist

    def _compute_intensity_cdist_shared(
        self,
        intensities1: pd.DataFrame,
        intensities2: pd.DataFrame,
    ) -> np.ndarray:
        columns = [c for c in intensities1.columns if c in intensities2.columns]
        intensities1 = intensities1[columns].astype(self.precision)
        intensities2 = intensities2[columns].astype(self.precision)
        intensities1_std = intensities1.std()
        intensities2_std = intensities2.std()
        constant_columns = [
            column
            for column in columns
            if not (intensities1_std[column] > 0 and intensities2_std[column] > 0)
        ]
        if len(constant_columns) > 0:
            self._warn_once(
                "constant_shared_markers",
                "Excluding shared markers without variance from computing the "
                f"intensity cross-distance: {', '.join(map(str, constant_columns))}",
            )
            intensities1 = intensities1.drop(columns=constant_columns)
            intensities2 = intensities2.drop(columns=constant_columns)
            intensities1_std = intensities1_std.drop(constant_columns)
            intensities2_std = intensities2_std.drop(constant_columns)
        if len(intensities1.columns) == 0 or len(intensities2.columns) == 0:
            raise SpellmatchException(
                "No shared markers with non-zero variance for computing the intensity "
                "cross-distance"
            )
        intensities1 = (intensities1 - intensities1.mean()) / intensities1_std
        intensities2 = (intensities2 - intensities2.mean()) / intensities2_std
        n_shared_features = len(intensities1.columns)
        metric = self.intensity_shared_metric
        if n_shared_features == 1:
            self._warn_once(
                "one_shared_marker",
                "Only one shared marker, using thresholded differences instead of "
                f"{metric.value} distances for the intensity cross-distance",
            )
            return self._compute_cdist_1d(
                intensities1.to_numpy(), intensities2.to_numpy()
            )
        elif n_shared_features == 2:
            if metric != self.SharedIntensityMetric.COSINE:
                self._warn_once(
                    "two_shared_markers",
                    "Only two shared markers, using cosine distances instead of "
                    f"{metric.value} distances for the intensity cross-distance",
                )
            return self._compute_cdist_2d(
                intensities1.to_numpy(), intensities2.to_numpy()
            )
        intensities_shared = pd.concat((intensities1, intensities2), ignore_index=True)
        n_components = self.intensity_shared_pca_n_components
        if n_components > n_shared_features:
            self._warn_once(
                "svd_components",
                "Requested number of intensity components for computing the intensity "
                f"cross-distance from shared markers ({n_components}) is larger than "
                f"the number of shared intensity features ({n_shared_features}), "
                f"continuing with {n_shared_features} intensity components",
            )
            n_components = n_shared_features
        svd = TruncatedSVD(
            n_components=n_components,
            algorithm="arpack"
            if n_components < n_shared_features
            else "randomized",  # ARPACK requires n_components < n_features
            random_state=0,  # reproducibility (randomized solver, ARPACK start vector)
        )
        svd.fit(intensities_shared)
        logger.debug(
            f"SVD: explained variance={np.sum(svd.explained_variance_ratio_):.3f} "
            f"{tuple(np.around(r, decimals=3) for r in svd.explained_variance_ratio_)}"
        )
        svd_intensities1 = svd.transform(intensities1)
        svd_intensities2 = svd.transform(intensities2)
        if n_components == 1:
            self._warn_once(
                "one_svd_component",
                "Only one SVD component, using thresholded differences instead of "
                f"{metric.value} distances for the intensity cross-distance",
            )
            return self._compute_cdist_1d(svd_intensities1, svd_intensities2)
        elif n_components == 2:
            if metric != self.SharedIntensityMetric.COSINE:
                self._warn_once(
                    "two_svd_components",
                    "Only two SVD components, using cosine distances instead of "
                    f"{metric.value} distances for the intensity cross-distance",
                )
            return self._compute_cdist_2d(svd_intensities1, svd_intensities2)
        else:
            if metric == self.SharedIntensityMetric.RECONSTRUCTED_CORRELATION:
                return self._compute_cdist(
                    svd.inverse_transform(svd_intensities1),
                    svd.inverse_transform(svd_intensities2),
                    "correlation",
                )
            return self._compute_cdist(svd_intensities1, svd_intensities2, metric.value)

    def _compute_cdist_1d(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """Cross-distance between one-dimensional cell embeddings, in [0, 1].

        The Pearson correlation is undefined and the cosine distance is binary (sign
        only) for one dimension; instead, the squared difference relative to
        intensity_1d_cdiff_thres_sd (pooled) standard deviations, clamped to [0, 1], is
        computed.
        """
        # threshold relative to the pooled standard deviation (much less sensitive to
        # outliers than, e.g., the maximum difference)
        thres = self.intensity_1d_cdiff_thres_sd * np.std(
            np.concatenate((x1[:, 0], x2[:, 0]))
        )
        cdist = np.subtract.outer(x1[:, 0], x2[:, 0]).astype(self.precision)
        if thres > 0:
            cdist /= self.precision(thres)
            np.square(cdist, out=cdist)
            np.clip(cdist, 0, 1, out=cdist)
        else:
            cdist[:] = 0  # constant values: no information
        return cdist

    def _compute_cdist_2d(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """Cross-distance between two-dimensional cell embeddings, in [0, 1].

        The Pearson correlation of two values is always +/-1; instead, the cosine
        distance (scaled to [0, 1]) is computed.
        """
        with np.errstate(invalid="ignore", divide="ignore"):
            cdist = 0.5 * distance.cdist(x1, x2, metric="cosine")
        # zero vectors (e.g. standardized intensities at the mean) have no direction;
        # assign the uninformative distance of orthogonal vectors
        cdist[np.isnan(cdist)] = 0.5
        return np.asarray(cdist, dtype=self.precision)

    def _compute_cdist(self, x1: np.ndarray, x2: np.ndarray, metric: str) -> np.ndarray:
        """Cross-distance between cell embeddings, in [0, 1].

        The specified (cosine or correlation) distance, which lies in [0, 2], is scaled
        to [0, 1].
        """
        cdist = 0.5 * distance.cdist(x1, x2, metric=metric)
        return np.asarray(cdist, dtype=self.precision)

    def _fit_cca(
        self, n_components: int, x: pd.DataFrame, y: pd.DataFrame
    ) -> tuple[CCA, np.ndarray, np.ndarray]:
        cca = CCA(
            n_components=n_components, max_iter=self.cca_max_iter, tol=self.cca_tol
        )
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always", ConvergenceWarning)
            cca_x, cca_y = cca.fit_transform(x, y)
        if any(issubclass(w.category, ConvergenceWarning) for w in caught_warnings):
            self._warn_once(
                "cca_not_converged",
                f"CCA did not converge after {self.cca_max_iter} iterations; consider "
                "increasing cca_max_iter",
            )
        for w in caught_warnings:
            if not issubclass(w.category, ConvergenceWarning):
                warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
        return cca, cca_x, cca_y

    def _compute_intensity_cdist_all(
        self,
        points1: pd.DataFrame,
        points2: pd.DataFrame,
        intensities1: pd.DataFrame,
        intensities2: pd.DataFrame,
    ) -> np.ndarray:
        intensities1 = intensities1.astype(self.precision)
        intensities2 = intensities2.astype(self.precision)
        if len(points2.index) < 2:
            raise SpellmatchException(
                "Computing intensity cross-distances on all markers requires at least "
                "two target cells"
            )
        k_closest = self.intensity_all_cca_fit_k_closest
        if k_closest > len(points1.index):
            self._warn_once(
                "cca_k_closest",
                "Requested number of closest cells for fitting the CCA "
                f"({k_closest}) is larger than the number of source cells "
                f"({len(points1.index)}), continuing with {len(points1.index)} cells",
            )
            k_closest = len(points1.index)
        k_most_certain = self.intensity_all_cca_fit_k_most_certain
        if k_most_certain > k_closest:
            self._warn_once(
                "cca_k_most_certain",
                "Requested number of most certain cells for fitting the CCA "
                f"({k_most_certain}) is larger than the number of closest cells "
                f"({k_closest}), continuing with {k_closest} cells",
            )
            k_most_certain = k_closest
        ind1 = np.arange(len(points1.index))
        nn2 = NearestNeighbors(n_neighbors=2)
        nn2.fit(points2)
        nn2_dists, nn2_ind = nn2.kneighbors(points1)
        closest_ind = np.argpartition(nn2_dists[:, 0], k_closest - 1)[:k_closest]
        ind1 = ind1[closest_ind]
        nn2_ind = nn2_ind[closest_ind, :]
        nn2_dists = nn2_dists[closest_ind, :]
        margins = nn2_dists[:, 1] - nn2_dists[:, 0]
        most_certain_ind = np.argpartition(-margins, k_most_certain - 1)[
            :k_most_certain
        ]
        ind1 = ind1[most_certain_ind]
        nn2_ind = nn2_ind[most_certain_ind, :]
        nn2_dists = nn2_dists[most_certain_ind, :]
        n_components = self.intensity_all_cca_n_components
        n_source_features = len(intensities1.columns)
        if n_components > n_source_features:
            self._warn_once(
                "cca_source_features",
                "Requested number of intensity components for computing the intensity "
                f"cross-distance from all markers ({n_components}) is larger than the "
                f"number of source intensity features ({n_source_features}), "
                f"continuing with {n_source_features} intensity components",
            )
            n_components = n_source_features
        n_target_features = len(intensities2.columns)
        if n_components > n_target_features:
            self._warn_once(
                "cca_target_features",
                "Requested number of intensity components for computing the intensity "
                f"cross-distance from all markers ({n_components}) is larger than the "
                f"number of target intensity features ({n_target_features}), "
                f"continuing with {n_target_features} intensity components",
            )
            n_components = n_target_features
        n_fit_cells = len(ind1)
        if n_components > n_fit_cells:
            self._warn_once(
                "cca_fit_cells",
                "Requested number of intensity components for computing the intensity "
                f"cross-distance from all markers ({n_components}) is larger than the "
                f"number of cells for fitting the CCA ({n_fit_cells}), "
                f"continuing with {n_fit_cells} intensity components",
            )
            n_components = n_fit_cells
        cca, cca_intensities1, cca_intensities2 = self._fit_cca(
            n_components,
            intensities1.iloc[ind1, :],
            intensities2.iloc[nn2_ind[:, 0], :],
        )
        cca_intensities1 = cca_intensities1.astype(self.precision)
        cca_intensities2 = cca_intensities2.astype(self.precision)
        cancor_mean = np.mean(
            np.diagonal(
                np.corrcoef(cca_intensities1, cca_intensities2, rowvar=False),
                offset=cca.n_components,
            )
        )
        logger.debug(f"CCA: canonical correlations mean={cancor_mean:.6f}")
        cca_intensities1, cca_intensities2 = cca.transform(intensities1, intensities2)
        cca_intensities1 = cca_intensities1.astype(self.precision)
        cca_intensities2 = cca_intensities2.astype(self.precision)
        metric = self.intensity_all_metric
        if n_components == 1:
            self._warn_once(
                "one_cca_component",
                "Only one CCA component, using thresholded differences instead of "
                f"{metric.value} distances for the intensity cross-distance",
            )
            return self._compute_cdist_1d(cca_intensities1, cca_intensities2)
        elif n_components == 2:
            if metric != self.AllIntensityMetric.COSINE:
                self._warn_once(
                    "two_cca_components",
                    "Only two CCA components, using cosine distances instead of "
                    f"{metric.value} distances for the intensity cross-distance",
                )
            return self._compute_cdist_2d(cca_intensities1, cca_intensities2)
        else:
            return self._compute_cdist(cca_intensities1, cca_intensities2, metric.value)


@dataclass(eq=False)
class _ProductGraph:
    """Product graph of two spatial cell graphs, restricted to candidate cell pairs.

    The adjacency A = (A1 ⊗ A2)[C, C] of the candidate cell pairs C is constructed
    directly (without the full Kronecker product), together with the distance
    cross-distance term on the same sparsity pattern. To save memory, both share their
    index arrays (indices, indptr); they must not be modified in place (e.g. by
    sort_indices or eliminate_zeros).
    """

    adj: sparse.csr_array
    distance_cdist: sparse.csr_array | None

    def __len__(self) -> int:
        return self.adj.shape[0]

    @property
    def degrees(self) -> np.ndarray:
        """Number of adjacent (candidate) cell pairs of each (candidate) cell pair."""
        return np.diff(self.adj.indptr)

    @staticmethod
    def build(
        graph1: SparseGraph,
        graph2: SparseGraph,
        cell_pairs: np.ndarray | None,
        precision: type,
        distance_cdiff_thres: float | None,
        chunk_size: int = 2**16,
    ) -> "_ProductGraph":
        """Build the product graph for the given cell pairs (all cell pairs if None).

        Cell pairs (i, j) and (k, l) are adjacent if i and k are adjacent in graph1 and
        j and l are adjacent in graph2. For each cell pair, all such neighbor pairs
        (k, l) are enumerated, and only those that are cell pairs themselves are kept.
        This is done in chunks of at most chunk_size neighbor pairs (but at least one
        cell pair per chunk) to bound temporary memory (2**16 was fastest in
        benchmarks, as the temporary arrays fit in the CPU cache).
        """
        n1, n2 = len(graph1), len(graph2)
        adj1, adj2 = graph1.adj, graph2.adj
        deg1, deg2 = graph1.degrees, graph2.degrees
        if cell_pairs is not None:
            n_cell_pairs = len(cell_pairs)
            ind1, ind2 = np.divmod(cell_pairs, n2)
        else:
            n_cell_pairs = n1 * n2
            ind1 = np.repeat(np.arange(n1), n2)
            ind2 = np.tile(np.arange(n2), n1)
        if distance_cdiff_thres is not None:
            assert graph1.dists is not None and graph2.dists is not None
            dists1 = graph1.dists.data.astype(precision)
            dists2 = graph2.dists.data.astype(precision)
        # number of neighbor pairs of each cell pair
        counts = deg1[ind1].astype(np.int64) * deg2[ind2]
        max_count = max(int(counts.max()), 1)
        max_index = max(n_cell_pairs, int(counts.sum()))
        index_dtype = np.int32 if max_index < np.iinfo(np.int32).max else np.int64
        chunk_len = max(chunk_size // max_count, 1)
        indices_chunks = []
        distance_cdist_chunks = []
        degrees = np.zeros(n_cell_pairs, dtype=np.int64)
        for chunk_start in range(0, n_cell_pairs, chunk_len):
            chunk_stop = min(chunk_start + chunk_len, n_cell_pairs)
            chunk_counts = counts[chunk_start:chunk_stop]
            # enumerate the neighbor pairs (k, l) of each cell pair (i, j), i.e. row:
            # the index of a neighbor pair within its row is decoded row-major into the
            # positions of the edges (i, k) and (j, l) in the CSR arrays of graph1 and
            # graph2, e.g. for neighbors k0, k1 of i and l0, l1, l2 of j into
            # (k0, l0), (k0, l1), (k0, l2), (k1, l0), (k1, l1), (k1, l2)
            rows = np.repeat(np.arange(chunk_start, chunk_stop), chunk_counts)
            neighbor_pair_ind = np.arange(len(rows)) - np.repeat(
                np.cumsum(chunk_counts) - chunk_counts, chunk_counts
            )
            row_deg2 = deg2[ind2[rows]]
            edges1 = adj1.indptr[ind1[rows]] + neighbor_pair_ind // row_deg2
            edges2 = adj2.indptr[ind2[rows]] + neighbor_pair_ind % row_deg2
            # neighbor pairs (k, l) as flat indices, and their product graph columns
            neighbor_cell_pairs = (
                adj1.indices[edges1].astype(np.int64) * n2 + adj2.indices[edges2]
            )
            if cell_pairs is None:
                cols = neighbor_cell_pairs
            else:
                # keep neighbor pairs that are cell pairs themselves
                cols = np.searchsorted(cell_pairs, neighbor_cell_pairs)
                np.minimum(cols, n_cell_pairs - 1, out=cols)
                keep = cell_pairs[cols] == neighbor_cell_pairs
                rows, cols, edges1, edges2 = (
                    rows[keep],
                    cols[keep],
                    edges1[keep],
                    edges2[keep],
                )
            degrees[chunk_start:chunk_stop] = np.bincount(
                rows - chunk_start, minlength=chunk_stop - chunk_start
            )
            indices_chunks.append(cols.astype(index_dtype))
            if distance_cdiff_thres is not None:
                chunk_distance_cdist = np.abs(dists1[edges1] - dists2[edges2])
                chunk_distance_cdist /= precision(distance_cdiff_thres)
                np.square(chunk_distance_cdist, out=chunk_distance_cdist)
                np.clip(chunk_distance_cdist, 0, 1, out=chunk_distance_cdist)
                distance_cdist_chunks.append(chunk_distance_cdist)
        indices = np.concatenate(indices_chunks)
        del indices_chunks
        distance_cdist_data = None
        if distance_cdiff_thres is not None:
            distance_cdist_data = np.concatenate(distance_cdist_chunks)
            del distance_cdist_chunks
        indptr = np.concatenate(([0], np.cumsum(degrees))).astype(index_dtype)
        adj = sparse.csr_array(
            (np.ones(len(indices), dtype=precision), indices, indptr),
            shape=(n_cell_pairs, n_cell_pairs),
        )
        distance_cdist = None
        if distance_cdist_data is not None:
            distance_cdist = sparse.csr_array(
                (distance_cdist_data, indices, indptr),
                shape=(n_cell_pairs, n_cell_pairs),
            )
        logger.debug(f"Product graph: {n_cell_pairs} cell pairs, {len(indices)} edges")
        return _ProductGraph(adj, distance_cdist)


class _NormalizedSimilarityMatrix:
    """Matrix-free W~ = D^(-1/2) (A - W) D^(-1/2).

    W = A * (c_v + c_w + w_d * M_vw) / T, where c are the weighted cross-distances of
    the cell pairs (degree, intensity), M the distance cross-distances between
    neighboring cell pairs with weight w_d, and T the total weight. Since the cell pair
    terms are separable, W~ s = D^(-1/2) (A - W) u with u = D^(-1/2) s and
    (A - W) u = (1 - c / T) * (A u) - A (c * u) / T - w_d / T * (M u).
    """

    def __init__(
        self,
        product_graph: _ProductGraph,
        cell_pair_cdist: np.ndarray | None,
        distance_weight: float,
        total_weight: float,
        precision: type,
    ) -> None:
        self.product_graph = product_graph
        deg = product_graph.degrees.astype(precision)
        self.deg_inv_sqrt = np.power(deg, -0.5, out=np.zeros_like(deg), where=deg != 0)
        self.rel_cell_pair_cdist = None
        if cell_pair_cdist is not None:
            self.rel_cell_pair_cdist = cell_pair_cdist / precision(total_weight)
        self.rel_distance_weight = None
        if product_graph.distance_cdist is not None:
            self.rel_distance_weight = precision(distance_weight / total_weight)

    def __matmul__(self, s: np.ndarray) -> np.ndarray:
        u = self.deg_inv_sqrt * s
        result = self.product_graph.adj @ u
        if self.rel_cell_pair_cdist is not None:
            result *= 1 - self.rel_cell_pair_cdist
            result -= self.product_graph.adj @ (self.rel_cell_pair_cdist * u)
        if self.rel_distance_weight is not None:
            result -= self.rel_distance_weight * (self.product_graph.distance_cdist @ u)
        return result * self.deg_inv_sqrt


class SpellmatchException(SpellmatchMatchingAlgorithmException):
    pass
