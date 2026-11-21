import logging
import warnings
from collections.abc import Callable, MutableMapping
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
        scores_tol: float | None = None,
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

    def _prepare_iteration_cache(
        self, iteration: int, iteration_cache: dict[str, Any], distances_changed: bool
    ) -> None:
        super()._prepare_iteration_cache(iteration, iteration_cache, distances_changed)
        if iteration > 0:
            # transform has been updated --> cached spatial cross-distance and
            # all-marker intensity cross-distance (fitted on nearest neighbors under
            # the current transform) are invalid
            iteration_cache.pop("spatial_cdist", None)
            iteration_cache.pop("intensity_cdist_all", None)
        if iteration > 0 and (self.outlier_dist is not None or distances_changed):
            # graphs have been updated --> cached graph cross-distances are invalid
            iteration_cache.pop("degree_cdist", None)
            iteration_cache.pop("distance_cdist_csr", None)
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
        source_adj: xr.DataArray,
        target_adj: xr.DataArray,
        source_dists: xr.DataArray | None,
        target_dists: xr.DataArray | None,
        source_intensities: pd.DataFrame | None,
        target_intensities: pd.DataFrame | None,
        cache: MutableMapping[str, Any] | None,
    ) -> tuple[dict[str, Any], xr.DataArray]:
        c = cache if cache is not None else {}
        n1 = len(source_adj)
        n2 = len(target_adj)
        adj1 = source_adj.to_numpy().astype(np.bool_)
        adj2 = target_adj.to_numpy().astype(np.bool_)
        adj_csr = sparse.csr_array(
            sparse.kron(sparse.csr_array(adj1), sparse.csr_array(adj2), format="csr")
        )
        deg1 = np.sum(adj1, axis=1, dtype=self.precision)
        deg2 = np.sum(adj2, axis=1, dtype=self.precision)
        if "degree_cdist" not in c:
            c["degree_cdist"] = None
            if self.degree_weight > 0.0:
                logger.info("Computing degree cross-distance")
                c["degree_cdist"] = self._compute_degree_cdist(deg1, deg2)
                assert c["degree_cdist"].dtype == self.precision
        if "intensity_cdist_shared" not in c:
            c["intensity_cdist_shared"] = None
            if self.intensity_weight > 0.0 and self.intensity_interp_lmd != 0.0:
                if source_intensities is None or target_intensities is None:
                    raise SpellmatchException(
                        "Intensities are required for computing their cross-distance"
                    )
                logger.info("Computing intensity cross-distance (shared markers)")
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
                logger.info("Computing intensity cross-distance (all markers)")
                c["intensity_cdist_all"] = self._compute_intensity_cdist_all(
                    self._current_source_points,
                    self._current_target_points,
                    source_intensities,
                    target_intensities,
                )
                assert c["intensity_cdist_all"].dtype == self.precision
        if "distance_cdist_csr" not in c:
            c["distance_cdist_csr"] = None
            if self.distance_weight > 0:
                if source_dists is None or target_dists is None:
                    raise SpellmatchException(
                        "Distances are required for computing their cross-distance"
                    )
                logger.info("Computing distance cross-distance")
                c["distance_cdist_csr"] = self._compute_distance_cdist(
                    adj1,
                    adj2,
                    source_dists.to_numpy().astype(self.precision),
                    target_dists.to_numpy().astype(self.precision),
                )
                assert c["distance_cdist_csr"].dtype == self.precision
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
                logger.info("Computing spatial cross-distance")
                c["spatial_cdist"] = np.asarray(
                    distance.cdist(
                        self._current_source_points.to_numpy().astype(self.precision),
                        self._current_target_points.to_numpy().astype(self.precision),
                    ),
                    dtype=self.precision,
                )
        degree_cdist = c["degree_cdist"]
        intensity_cdist_shared = c["intensity_cdist_shared"]
        intensity_cdist_all = c["intensity_cdist_all"]
        distance_cdist_csr = c["distance_cdist_csr"]
        spatial_cdist = c["spatial_cdist"]
        candidates = None
        if self.max_spatial_cdist is not None:
            assert spatial_cdist is not None
            candidates = np.flatnonzero(spatial_cdist <= self.max_spatial_cdist)
            if len(candidates) == 0:
                raise SpellmatchException(
                    "No candidate cell pairs within the maximum spatial cross-distance"
                )
            logger.info(
                f"Restricting to {len(candidates)} of {n1 * n2} candidate cell pairs"
            )
            adj_csr = adj_csr[candidates][:, candidates]
            if degree_cdist is not None:
                degree_cdist = degree_cdist.ravel()[candidates]
            if intensity_cdist_shared is not None:
                intensity_cdist_shared = intensity_cdist_shared.ravel()[candidates]
            if intensity_cdist_all is not None:
                intensity_cdist_all = intensity_cdist_all.ravel()[candidates]
            if distance_cdist_csr is not None:
                distance_cdist_csr = distance_cdist_csr[candidates][:, candidates]
            spatial_cdist = spatial_cdist.ravel()[candidates]
        if (
            intensity_cdist_shared is None
            or intensity_cdist_all is None
            or 0 <= self.intensity_interp_lmd <= 1
        ):
            info, scores_data = self._match_graphs_for_lambda(
                n1,
                n2,
                candidates,
                adj_csr,
                degree_cdist,
                intensity_cdist_shared,
                intensity_cdist_all,
                distance_cdist_csr,
                spatial_cdist,
                self.intensity_interp_lmd
                if intensity_cdist_shared is not None or intensity_cdist_all is not None
                else None,
            )
        else:
            lmd = None
            info = None
            scores_data = None
            cancor_mean = None
            n_components = self.intensity_interp_cca_n_components
            n_source_features = len(source_intensities.columns)
            if n_components > n_source_features:
                logger.warning(
                    f"Requested number of intensity components for lambda estimation "
                    f"({n_components}) is larger than the number of source intensity "
                    f"features ({n_source_features}), continuing with "
                    f"{n_source_features} intensity components"
                )
                n_components = n_source_features
            n_target_features = len(target_intensities.columns)
            if n_components > n_target_features:
                logger.warning(
                    f"Requested number of intensity components for lambda estimation "
                    f"({n_components}) is larger than the number of target intensity "
                    f"features ({n_target_features}), continuing with "
                    f"{n_target_features} intensity components"
                )
                n_components = n_target_features
            for current_lmd in np.linspace(0, 1, int(self.intensity_interp_lmd)):
                logger.info(f"Evaluating lambda={current_lmd:.3g}")
                current_info, current_scores_data = self._match_graphs_for_lambda(
                    n1,
                    n2,
                    candidates,
                    adj_csr,
                    degree_cdist,
                    intensity_cdist_shared,
                    intensity_cdist_all,
                    distance_cdist_csr,
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
                logger.info(f"Canonical correlations mean={current_cancor_mean:.6f}")
                if cancor_mean is None or current_cancor_mean > cancor_mean:
                    lmd = current_lmd
                    info = current_info
                    scores_data = current_scores_data
                    cancor_mean = current_cancor_mean
            logger.info(f"Best lambda={lmd:.3g} (CC mean={cancor_mean:.6f})")
        scores = xr.DataArray(
            data=scores_data,
            coords={
                source_adj.name or "source": source_adj.coords["a"].to_numpy(),
                target_adj.name or "target": target_adj.coords["x"].to_numpy(),
            },
        )
        return info, scores

    def _match_graphs_for_lambda(
        self,
        n1: int,
        n2: int,
        candidates: np.ndarray | None,
        adj_csr: sparse.csr_array,
        degree_cdist: np.ndarray | None,
        intensity_cdist_shared: np.ndarray | None,
        intensity_cdist_all: np.ndarray | None,
        distance_cdist_csr: sparse.csr_array | None,
        spatial_cdist: np.ndarray | None,
        lmd: float | None,
    ) -> tuple[dict[str, Any], np.ndarray]:
        if intensity_cdist_shared is not None and intensity_cdist_all is not None:
            logger.info("Combining intensity cross-distance matrices")
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
        logger.info("Initializing")
        n = adj_csr.shape[0]  # number of (candidate) cell pairs
        w_csr = sparse.csr_array((n, n), dtype=self.precision)
        total_weight = 0
        if self.degree_weight > 0:
            assert degree_cdist is not None
            degree_cdist = degree_cdist.ravel()
            w_csr += adj_csr * (
                self.precision(self.degree_weight) * degree_cdist[:, np.newaxis]
            )
            w_csr += adj_csr * (
                self.precision(self.degree_weight) * degree_cdist[np.newaxis, :]
            )
            total_weight += 2 * self.degree_weight
            assert w_csr.dtype == self.precision
        if self.intensity_weight > 0:
            assert intensity_cdist is not None
            intensity_cdist = intensity_cdist.ravel()
            w_csr += adj_csr * (
                self.precision(self.intensity_weight) * intensity_cdist[:, np.newaxis]
            )
            w_csr += adj_csr * (
                self.precision(self.intensity_weight) * intensity_cdist[np.newaxis, :]
            )
            total_weight += 2 * self.intensity_weight
            assert w_csr.dtype == self.precision
        if self.distance_weight > 0:
            assert distance_cdist_csr is not None
            w_csr += adj_csr * (
                self.precision(self.distance_weight) * distance_cdist_csr
            )
            total_weight += self.distance_weight
            assert w_csr.dtype == self.precision
        if total_weight > 0:
            w_csr /= self.precision(total_weight)
            assert w_csr.dtype == self.precision
        else:
            logger.info(
                "All cross-distance weights are zero, matching graph topology only"
            )
        # degrees of the (restricted) product graph, i.e., its row sums: in CSR format,
        # the entries of row i are stored at positions indptr[i] to indptr[i + 1] - 1,
        # so np.diff(indptr) is the number of entries per row; since all entries of the
        # adjacency matrix are one, this equals the row sums
        d = np.diff(adj_csr.indptr).astype(self.precision)
        d[d != 0] = d[d != 0] ** (-0.5)
        d_dia = sparse.dia_array((d, [0]), shape=(n, n))
        w_csr: sparse.csr_array = d_dia @ (adj_csr - w_csr) @ d_dia
        assert w_csr.dtype == self.precision
        del d, d_dia
        if self.alpha != 1.0 and self.spatial_cdist_prior_thres is not None:
            assert spatial_cdist is not None
            h = np.ravel(
                1 - np.clip((spatial_cdist / self.spatial_cdist_prior_thres) ** 2, 0, 1)
            )
            if np.sum(h) == 0:
                raise SpellmatchException(
                    "No candidate cell pairs within the spatial prior threshold"
                )
            h = np.asarray(h / np.sum(h), dtype=self.precision)[:, np.newaxis]
        else:
            h = np.ones((n, 1), dtype=self.precision)
            h /= self.precision(n)
        assert h.dtype == self.precision
        s = np.ones((n, 1), dtype=self.precision)
        s /= self.precision(n)
        assert s.dtype == self.precision
        logger.info("Optimizing")
        opt_converged = False
        seconds = 0
        for opt_iteration in range(self.opt_max_iter):
            start = timer()
            s_new: np.ndarray = (
                self.precision(self.alpha) * (w_csr @ s)
                + self.precision(1.0 - self.alpha) * h
            )
            opt_loss = float(np.linalg.norm(s[:, 0] - s_new[:, 0]))
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
                logger.info(
                    f"Done after {opt_iteration + 1} iterations ({seconds:.3f} seconds)"
                )
                break
        if not opt_converged:
            logger.warning(
                f"Optimization did not converge after {self.opt_max_iter} iterations "
                f"({seconds:.3f} seconds, last loss: {opt_loss:.9f})"
            )
        info = {
            "lambda": lmd,
            "opt_iterations": opt_iteration + 1,
            "opt_converged": opt_converged,
        }
        scores_data = s[:, 0]
        if candidates is not None:
            scores_data = np.zeros(n1 * n2, dtype=self.precision)
            scores_data[candidates] = s[:, 0]
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
            logger.warning(
                "Excluding shared markers without variance from computing the "
                f"intensity cross-distance: {', '.join(map(str, constant_columns))}"
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
            logger.warning(
                "Only one shared marker, using thresholded differences instead of "
                f"{metric.value} distances for the intensity cross-distance"
            )
            return self._compute_cdist_1d(
                intensities1.to_numpy(), intensities2.to_numpy()
            )
        elif n_shared_features == 2:
            if metric != self.SharedIntensityMetric.COSINE:
                logger.warning(
                    "Only two shared markers, using cosine distances instead of "
                    f"{metric.value} distances for the intensity cross-distance"
                )
            return self._compute_cdist_2d(
                intensities1.to_numpy(), intensities2.to_numpy()
            )
        intensities_shared = pd.concat((intensities1, intensities2), ignore_index=True)
        n_components = self.intensity_shared_pca_n_components
        if n_components > n_shared_features:
            logger.warning(
                "Requested number of intensity components for computing the intensity "
                f"cross-distance from shared markers ({n_components}) is larger than "
                f"the number of shared intensity features ({n_shared_features}), "
                f"continuing with {n_shared_features} intensity components"
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
            logger.warning(
                "Only one SVD component, using thresholded differences instead of "
                f"{metric.value} distances for the intensity cross-distance"
            )
            return self._compute_cdist_1d(svd_intensities1, svd_intensities2)
        elif n_components == 2:
            if metric != self.SharedIntensityMetric.COSINE:
                logger.warning(
                    "Only two SVD components, using cosine distances instead of "
                    f"{metric.value} distances for the intensity cross-distance"
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
            logger.warning(
                f"CCA did not converge after {self.cca_max_iter} iterations; consider "
                "increasing cca_max_iter"
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
            logger.warning(
                "Requested number of closest cells for fitting the CCA "
                f"({k_closest}) is larger than the number of source cells "
                f"({len(points1.index)}), continuing with {len(points1.index)} cells"
            )
            k_closest = len(points1.index)
        k_most_certain = self.intensity_all_cca_fit_k_most_certain
        if k_most_certain > k_closest:
            logger.warning(
                "Requested number of most certain cells for fitting the CCA "
                f"({k_most_certain}) is larger than the number of closest cells "
                f"({k_closest}), continuing with {k_closest} cells"
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
            logger.warning(
                "Requested number of intensity components for computing the intensity "
                f"cross-distance from all markers ({n_components}) is larger than the "
                f"number of source intensity features ({n_source_features}), "
                f"continuing with {n_source_features} intensity components"
            )
            n_components = n_source_features
        n_target_features = len(intensities2.columns)
        if n_components > n_target_features:
            logger.warning(
                "Requested number of intensity components for computing the intensity "
                f"cross-distance from all markers ({n_components}) is larger than the "
                f"number of target intensity features ({n_target_features}), "
                f"continuing with {n_target_features} intensity components"
            )
            n_components = n_target_features
        n_fit_cells = len(ind1)
        if n_components > n_fit_cells:
            logger.warning(
                "Requested number of intensity components for computing the intensity "
                f"cross-distance from all markers ({n_components}) is larger than the "
                f"number of cells for fitting the CCA ({n_fit_cells}), "
                f"continuing with {n_fit_cells} intensity components"
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
            logger.warning(
                "Only one CCA component, using thresholded differences instead of "
                f"{metric.value} distances for the intensity cross-distance"
            )
            return self._compute_cdist_1d(cca_intensities1, cca_intensities2)
        elif n_components == 2:
            if metric != self.AllIntensityMetric.COSINE:
                logger.warning(
                    "Only two CCA components, using cosine distances instead of "
                    f"{metric.value} distances for the intensity cross-distance"
                )
            return self._compute_cdist_2d(cca_intensities1, cca_intensities2)
        else:
            return self._compute_cdist(cca_intensities1, cca_intensities2, metric.value)

    def _compute_distance_cdist(
        self,
        adj1: np.ndarray,
        adj2: np.ndarray,
        dists1: np.ndarray,
        dists2: np.ndarray,
    ) -> sparse.csr_array:
        adj1_csr = sparse.csr_array(adj1)
        adj2_csr = sparse.csr_array(adj2)
        m_csr = sparse.csr_array(
            abs(
                # CSR for subtraction
                sparse.kron(sparse.csr_array(adj1 * dists1), adj2_csr, format="csr")
                - sparse.kron(adj1_csr, sparse.csr_array(adj2 * dists2), format="csr")
            )
        )
        m_csr.data = (m_csr.data / self.precision(self.distance_cdiff_thres)) ** 2
        np.clip(m_csr.data, 0, 1, out=m_csr.data)
        assert m_csr.dtype == self.precision
        return m_csr


class SpellmatchException(SpellmatchMatchingAlgorithmException):
    pass
