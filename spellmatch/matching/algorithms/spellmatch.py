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
            iteration_cache.pop("distance_cdist_csr", None)
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
        adj1 = source_adj.to_numpy().astype(np.bool8)
        adj2 = target_adj.to_numpy().astype(np.bool8)
        adj_csr = sparse.csr_array(sparse.kron(adj1, adj2, format="csr"))
        deg1 = np.sum(adj1, axis=1, dtype=np.uint8)
        deg2 = np.sum(adj2, axis=1, dtype=np.uint8)
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
        if (
            c["intensity_cdist_shared"] is None
            or c["intensity_cdist_all"] is None
            or 0 <= self.intensity_interp_lmd <= 1
        ):
            info, scores_data = self._match_graphs_for_lambda(
                n1,
                n2,
                adj_csr,
                deg,
                c["degree_cdist"],
                c["intensity_cdist_shared"],
                c["intensity_cdist_all"],
                c["distance_cdist_csr"],
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
                    adj_csr,
                    deg,
                    c["degree_cdist"],
                    c["intensity_cdist_shared"],
                    c["intensity_cdist_all"],
                    c["distance_cdist_csr"],
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
        adj_csr: sparse.csr_array,
        deg: np.ndarray,
        degree_cdist: Optional[np.ndarray],
        intensity_cdist_shared: Optional[np.ndarray],
        intensity_cdist_all: Optional[np.ndarray],
        distance_cdist_csr: Optional[sparse.csr_array],
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
        w_csr = sparse.csr_array((n1 * n2, n1 * n2), dtype=self.precision)
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
        d = np.asarray(deg.flatten(), dtype=self.precision)
        d[d != 0] = d[d != 0] ** (-0.5)
        d_dia = sparse.dia_array((d, [0]), shape=(n1 * n2, n1 * n2))
        w_csr: sparse.csr_array = d_dia @ (adj_csr - w_csr) @ d_dia
        assert w_csr.dtype == self.precision
        del d, d_dia
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
        return info, s[:, 0].reshape((n1, n2))

    def _compute_degree_cdist(self, deg1: np.ndarray, deg2: np.ndarray) -> np.ndarray:
        degree_cdiff = abs(deg1[:, np.newaxis] - deg2[np.newaxis, :])
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

    def _compute_distance_cdist(
        self,
        adj1: np.ndarray,
        adj2: np.ndarray,
        dists1: np.ndarray,
        dists2: np.ndarray,
    ) -> sparse.csr_array:
        m_csr = sparse.csr_array(
            abs(
                sparse.kron(adj1 * dists1, adj2, format="csr")  # CSR for subtraction
                - sparse.kron(adj1, adj2 * dists2, format="csr")  # CSR for subtraction
            )
        )
        m_csr.data = (m_csr.data / self.precision(self.distance_cdiff_thres)) ** 2
        np.clip(m_csr.data, 0, 1, out=m_csr.data)
        assert m_csr.dtype == self.precision
        return m_csr


class SpellmatchException(SpellmatchMatchingAlgorithmException):
    pass
