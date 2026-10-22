import logging
from collections.abc import Callable, MutableMapping
from timeit import default_timer as timer
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors

from ..._spellmatch import hookimpl
from .._kron_free import (
    DistanceTerms,
    build_distance_correction,
    distance_term_matvec,
    kron_matvec,
    outer_sum_matvec,
)
from ._algorithms import (
    IterativeGraphMatchingAlgorithm,
    MaskMatchingAlgorithm,
    SpellmatchMatchingAlgorithmException,
)

logger = logging.getLogger(__name__)


@hookimpl
def spellmatch_get_mask_matching_algorithm(
    name: Optional[str],
) -> Union[Optional[type[MaskMatchingAlgorithm]], list[str]]:
    algorithms = {
        "spellmatch": Spellmatch,
    }
    if name is not None:
        return algorithms.get(name)
    return list(algorithms.keys())


class Spellmatch(IterativeGraphMatchingAlgorithm):
    def __init__(
        self,
        *,
        point_feature: str = "centroid",
        intensity_feature: str = "intensity_mean",
        intensity_transform: Union[
            str, Callable[[np.ndarray], np.ndarray], None
        ] = None,
        transform_type: str = "rigid",
        transform_estim_type: str = "max_score",
        transform_estim_k_best: int = 50,
        max_iter: int = 50,
        scores_tol: Optional[float] = None,
        transform_tol: Optional[float] = None,
        filter_outliers: bool = True,
        adj_radius: float = 15,
        alpha: float = 0.8,
        spatial_cdist_prior_thres: Optional[float] = None,
        max_spatial_cdist: Optional[float] = None,
        degree_weight: float = 0,
        degree_cdiff_thres: int = 3,
        intensity_weight: float = 0,
        intensity_interp_lmd: Union[int, float] = 11,
        intensity_interp_cca_n_components: int = 10,
        intensity_shared_pca_n_components: int = 5,
        intensity_all_cca_fit_k_closest: int = 500,
        intensity_all_cca_fit_k_most_certain: int = 100,
        intensity_all_cca_n_components: int = 10,
        distance_weight: float = 0,
        distance_cdiff_thres: float = 5,
        cca_max_iter: int = 500,
        cca_tol: float = 1e-6,
        opt_max_iter: int = 200,
        opt_tol: float = 1e-6,
        precision: str = "float32",
    ) -> None:
        super(Spellmatch, self).__init__(
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
        self.intensity_all_cca_fit_k_closest = intensity_all_cca_fit_k_closest
        self.intensity_all_cca_fit_k_most_certain = intensity_all_cca_fit_k_most_certain
        self.intensity_all_cca_n_components = intensity_all_cca_n_components
        self.distance_weight = distance_weight
        self.distance_cdiff_thres = distance_cdiff_thres
        self.cca_max_iter = cca_max_iter
        self.cca_tol = cca_tol
        self.opt_max_iter = opt_max_iter
        self.opt_tol = opt_tol
        self.precision = np.dtype(precision).type
        self._current_source_points: Optional[pd.DataFrame] = None
        self._current_target_points: Optional[pd.DataFrame] = None

    def _prepare_iteration_cache(
        self, iteration: int, iteration_cache: dict[str, Any]
    ) -> None:
        super(Spellmatch, self)._prepare_iteration_cache(iteration, iteration_cache)
        if iteration > 0:
            # transform has been updated --> cached spatial cross-distance is invalid
            iteration_cache.pop("spatial_cdist", None)
        if iteration > 0 and self.outlier_dist is not None:
            # point filtering has been updated --> cached cross-distances are invalid
            iteration_cache.pop("degree_cdist", None)
            iteration_cache.pop("intensity_cdist_shared", None)
            iteration_cache.pop("intensity_cdist_all", None)
            iteration_cache.pop("distance_terms", None)
            iteration_cache.pop("spatial_cdist", None)

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
        self._current_source_points = source_points
        self._current_target_points = target_points
        info, scores = super(Spellmatch, self)._match_graphs_from_points(
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
        source_dists: Optional[xr.DataArray],
        target_dists: Optional[xr.DataArray],
        source_intensities: Optional[pd.DataFrame],
        target_intensities: Optional[pd.DataFrame],
        cache: Optional[MutableMapping[str, Any]],
    ) -> tuple[dict[str, Any], xr.DataArray]:
        c = cache if cache is not None else {}
        n1 = len(source_adj)
        n2 = len(target_adj)
        adj1 = source_adj.to_numpy().astype(np.bool_)
        adj2 = target_adj.to_numpy().astype(np.bool_)
        # The product graph adj_csr = sparse.kron(adj1, adj2) is never built:
        # it is an (n1*n2)x(n1*n2) matrix with nnz(adj1)*nnz(adj2) entries
        # and runs out of memory for realistic FOV sizes. adj1/adj2 are kept
        # small and square; every "adj_csr @ v" is computed exactly by
        # kron_matvec(adj1, adj2, v, n1, n2) instead.
        # The degree dtype must have headroom for the product deg1[i]*deg2[j]
        # below (numpy keeps the operands' dtype): with uint8, degrees above
        # ~16 wrapped around and broke the D^-1/2 normalization. uint16 (max
        # 65,535) leaves a wide margin at realistic adj_radius values.
        deg1 = np.sum(adj1, axis=1, dtype=np.uint16)
        deg2 = np.sum(adj2, axis=1, dtype=np.uint16)
        deg = np.asarray(deg1[:, np.newaxis] * deg2[np.newaxis, :])
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
        if "distance_terms" not in c:
            # None: distance term switched off (distance_weight == 0)
            c["distance_terms"] = None
            if self.distance_weight > 0:
                if source_dists is None or target_dists is None:
                    raise SpellmatchException(
                        "Distances are required for computing their cross-distance"
                    )
                logger.info("Computing distance cross-distance (matrix-free)")
                c["distance_terms"] = self._compute_distance_terms(
                    adj1,
                    adj2,
                    source_dists.to_numpy().astype(np.float64),
                    target_dists.to_numpy().astype(np.float64),
                    n1,
                    n2,
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
                logger.info("Computing spatial cross-distance")
                c["spatial_cdist"] = np.asarray(
                    distance.cdist(
                        self._current_source_points.to_numpy().astype(self.precision),
                        self._current_target_points.to_numpy().astype(self.precision),
                    ),
                    dtype=self.precision,
                )
        if (
            c["intensity_cdist_shared"] is None
            or c["intensity_cdist_all"] is None
            or 0 <= self.intensity_interp_lmd <= 1
        ):
            info, scores_data = self._match_graphs_for_lambda(
                n1,
                n2,
                adj1,
                adj2,
                deg,
                c["degree_cdist"],
                c["intensity_cdist_shared"],
                c["intensity_cdist_all"],
                c["distance_terms"],
                c["spatial_cdist"],
                self.intensity_interp_lmd,
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
            for current_lmd in np.linspace(0, 1, self.intensity_interp_lmd):
                logger.info(f"Evaluating lambda={current_lmd:.3g}")
                current_info, current_scores_data = self._match_graphs_for_lambda(
                    n1,
                    n2,
                    adj1,
                    adj2,
                    deg,
                    c["degree_cdist"],
                    c["intensity_cdist_shared"],
                    c["intensity_cdist_all"],
                    c["distance_terms"],
                    c["spatial_cdist"],
                    current_lmd,
                )
                row_ind, col_ind = linear_sum_assignment(
                    current_scores_data, maximize=True
                )
                nonzero_mask = current_scores_data[row_ind, col_ind] > 0
                row_ind, col_ind = row_ind[nonzero_mask], col_ind[nonzero_mask]
                cca = CCA(
                    n_components=n_components,
                    max_iter=self.cca_max_iter,
                    tol=self.cca_tol,
                )
                cca_intensities1, cca_intensities2 = cca.fit_transform(
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
        adj1: np.ndarray,
        adj2: np.ndarray,
        deg: np.ndarray,
        degree_cdist: Optional[np.ndarray],
        intensity_cdist_shared: Optional[np.ndarray],
        intensity_cdist_all: Optional[np.ndarray],
        distance_terms: Optional[DistanceTerms],
        spatial_cdist: Optional[np.ndarray],
        lmd: float,
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
        # The iteration needs w_csr @ v with
        #   w_csr = d_dia @ (adj_csr - combined) @ d_dia,
        # an (n1*n2)x(n1*n2) matrix like adj_csr. Instead of building it, we
        # build only the small ingredients each term needs and define
        # w_matvec(v) = w_csr @ v, computed fresh each call. This is an exact
        # reformulation, not an approximation. The distance term's expansion
        # is sensitive to cancellation, which is why it is computed in float64
        # regardless of self.precision.
        total_weight = 0.0
        if self.degree_weight > 0:
            total_weight += 2 * self.degree_weight
        if self.intensity_weight > 0:
            total_weight += 2 * self.intensity_weight
        if self.distance_weight > 0:
            total_weight += self.distance_weight

        u_degree = None
        if self.degree_weight > 0:
            assert degree_cdist is not None
            u_degree = np.asarray(degree_cdist.ravel(), dtype=np.float64)
        u_intensity = None
        if self.intensity_weight > 0:
            assert intensity_cdist is not None
            u_intensity = np.asarray(intensity_cdist.ravel(), dtype=np.float64)
        if self.distance_weight > 0:
            assert distance_terms is not None

        # float64 is required by the distance term.
        adj1_f64 = sparse.csr_array(adj1, dtype=np.float64)
        adj2_f64 = sparse.csr_array(adj2, dtype=np.float64)

        d = np.asarray(deg.flatten(), dtype=np.float64)
        d[d != 0] = d[d != 0] ** (-0.5)

        def w_matvec(v: np.ndarray) -> np.ndarray:
            s1 = d * v
            a_s1 = kron_matvec(adj1_f64, adj2_f64, s1, n1, n2)
            raw = np.zeros_like(a_s1)
            if self.degree_weight > 0:
                raw += self.degree_weight * outer_sum_matvec(
                    adj1_f64, adj2_f64, u_degree, s1, n1, n2, a_s=a_s1
                )
            if self.intensity_weight > 0:
                raw += self.intensity_weight * outer_sum_matvec(
                    adj1_f64, adj2_f64, u_intensity, s1, n1, n2, a_s=a_s1
                )
            if self.distance_weight > 0:
                raw += self.distance_weight * distance_term_matvec(
                    adj1_f64, adj2_f64, distance_terms, s1, n1, n2
                )
            if total_weight > 0:
                raw = raw / total_weight
            return np.asarray(d * (a_s1 - raw), dtype=self.precision)

        if self.alpha != 1.0 and self.spatial_cdist_prior_thres is not None:
            assert spatial_cdist is not None
            h = np.ravel(
                1 - np.clip((spatial_cdist / self.spatial_cdist_prior_thres) ** 2, 0, 1)
            )
            h = np.asarray(h / np.sum(h), dtype=self.precision)[:, np.newaxis]
        else:
            h = np.ones((n1 * n2, 1), dtype=self.precision)
            h /= self.precision(n1 * n2)
        assert h.dtype == self.precision
        if self.max_spatial_cdist is not None:
            assert spatial_cdist is not None
            s = np.ravel(spatial_cdist <= self.max_spatial_cdist)
            s = np.asarray(s / np.sum(s), dtype=self.precision)[:, np.newaxis]
        else:
            s = np.ones((n1 * n2, 1), dtype=self.precision)
            s /= self.precision(n1 * n2)
        assert s.dtype == self.precision
        logger.info("Optimizing")
        opt_converged = False
        seconds = 0
        for opt_iteration in range(self.opt_max_iter):
            start = timer()
            s_new: np.ndarray = (
                self.precision(self.alpha) * w_matvec(s[:, 0])[:, np.newaxis]
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
        return info, s[:, 0].reshape((n1, n2))

    def _compute_degree_cdist(self, deg1: np.ndarray, deg2: np.ndarray) -> np.ndarray:
        # deg1/deg2 are unsigned: subtract in a signed dtype, otherwise a
        # negative difference wraps around (e.g. 5 - 7 -> 65534) instead of
        # giving |5 - 7| = 2.
        degree_cdiff = np.abs(
            deg1.astype(np.int32)[:, np.newaxis] - deg2.astype(np.int32)[np.newaxis, :]
        )
        degree_cdist = (self.precision(degree_cdiff) / self.degree_cdiff_thres) ** 2
        np.clip(degree_cdist, 0, 1, out=degree_cdist)
        assert degree_cdist.dtype == self.precision
        return degree_cdist

    def _compute_intensity_cdist_shared(
        self,
        intensities1: pd.DataFrame,
        intensities2: pd.DataFrame,
    ) -> np.ndarray:
        intensities1 = intensities1.astype(self.precision)
        intensities2 = intensities2.astype(self.precision)
        intensities1 = (intensities1 - intensities1.mean()) / intensities1.std()
        intensities2 = (intensities2 - intensities2.mean()) / intensities2.std()
        intensities_shared = pd.concat(
            (intensities1, intensities2), join="inner", ignore_index=True
        )
        n_components = self.intensity_shared_pca_n_components
        n_shared_features = len(intensities_shared.columns)
        if n_components > n_shared_features:
            logger.warning(
                "Requested number of intensity components for computing the intensity "
                f"cross-distance from shared markers ({n_components}) is larger than "
                f"the number of shared intensity features ({n_shared_features}), "
                f"continuing with {n_shared_features} intensity components"
            )
            n_components = n_shared_features
        svd = TruncatedSVD(n_components=n_components, algorithm="arpack")
        svd.fit(intensities_shared)
        logger.debug(
            f"SVD: explained variance={np.sum(svd.explained_variance_ratio_):.3f} "
            f"{tuple(np.around(r, decimals=3) for r in svd.explained_variance_ratio_)}"
        )
        svd_intensities1 = svd.transform(intensities1[intensities_shared.columns])
        svd_intensities2 = svd.transform(intensities2[intensities_shared.columns])
        intensity_cdist_shared = 0.5 * np.asarray(
            distance.cdist(svd_intensities1, svd_intensities2, metric="correlation"),
            dtype=self.precision,
        )
        return np.asarray(intensity_cdist_shared, dtype=self.precision)

    def _compute_intensity_cdist_all(
        self,
        points1: pd.DataFrame,
        points2: pd.DataFrame,
        intensities1: pd.DataFrame,
        intensities2: pd.DataFrame,
    ) -> np.ndarray:
        intensities1 = intensities1.astype(self.precision)
        intensities2 = intensities2.astype(self.precision)
        ind1 = np.arange(len(points1.index))
        nn2 = NearestNeighbors(n_neighbors=2)
        nn2.fit(points2)
        nn2_dists, nn2_ind = nn2.kneighbors(points1)
        closest_ind = np.argpartition(
            nn2_dists[:, 0], self.intensity_all_cca_fit_k_closest - 1
        )[: self.intensity_all_cca_fit_k_closest]
        ind1 = ind1[closest_ind]
        nn2_ind = nn2_ind[closest_ind, :]
        nn2_dists = nn2_dists[closest_ind, :]
        margins = nn2_dists[:, 1] - nn2_dists[:, 0]
        most_certain_ind = np.argpartition(
            -margins, self.intensity_all_cca_fit_k_most_certain - 1
        )[: self.intensity_all_cca_fit_k_most_certain]
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
        cca = CCA(
            n_components=n_components, max_iter=self.cca_max_iter, tol=self.cca_tol
        )
        cca_intensities1, cca_intensities2 = cca.fit_transform(
            intensities1.iloc[ind1, :], intensities2.iloc[nn2_ind[:, 0], :]
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
        intensity_cdist_all = 0.5 * np.asarray(
            distance.cdist(cca_intensities1, cca_intensities2, metric="correlation"),
            dtype=self.precision,
        )
        return intensity_cdist_all

    def _compute_distance_terms(
        self,
        adj1: np.ndarray,
        adj2: np.ndarray,
        dists1: np.ndarray,
        dists2: np.ndarray,
        n1: int,
        n2: int,
    ) -> DistanceTerms:
        """Ingredients for the distance cross-distance, in matrix-free form.

        The distance cross-distance is ``clip(((kron(adj1*dists1,adj2) -
        kron(adj1,adj2*dists2))/thresh)**2, 0, 1)``, an ``(n1*n2)x(n1*n2)``
        matrix with ``nnz(adj1)*nnz(adj2)`` entries that runs out of memory
        for realistic FOV sizes if built explicitly (as the previous
        ``_compute_distance_cdist`` did).

        Returns ``DistanceTerms(d1, d2, thresh, correction)``: ``d1 = adj1 ⊙
        dists1``, ``d2 = adj2 ⊙ dists2`` are small (``n1xn1``/``n2xn2``)
        edge-masked distance matrices in float64, and ``correction`` is the
        sparse, upper-triangular-only ``relu(y-1)`` matrix (symmetric), or
        ``None`` if no edge pair is clipped. Combined with
        ``distance_term_matvec``, these reproduce ``clip(y,0,1) @ s`` exactly,
        without ever materializing the big object.
        """
        d1 = sparse.csr_array((adj1 * dists1).astype(np.float64))
        d2 = sparse.csr_array((adj2 * dists2).astype(np.float64))

        # |d1 - d2| <= max edge length, so if no edge is longer than the
        # threshold the clip can never fire and no correction is needed.
        max_edge = max(
            d1.max() if d1.nnz > 0 else 0.0, d2.max() if d2.nnz > 0 else 0.0
        )
        if max_edge <= self.distance_cdiff_thres:
            logger.info(
                "Distance-correction skipped: longest edge (%.3f) <= "
                "distance_cdiff_thres (%s), nothing is clipped",
                max_edge,
                self.distance_cdiff_thres,
            )
            return DistanceTerms(d1, d2, self.distance_cdiff_thres, correction=None)

        correction = build_distance_correction(
            d1, d2, self.distance_cdiff_thres, n1, n2
        )
        if correction is None:
            logger.info(
                "Distance-correction skipped: no edge pair differs by more than "
                "distance_cdiff_thres (%s), nothing is clipped",
                self.distance_cdiff_thres,
            )
        else:
            logger.info(
                "Built distance-correction matrix: %s nonzeros", f"{correction.nnz:,}"
            )
        return DistanceTerms(d1, d2, self.distance_cdiff_thres, correction)


class SpellmatchException(SpellmatchMatchingAlgorithmException):
    pass
