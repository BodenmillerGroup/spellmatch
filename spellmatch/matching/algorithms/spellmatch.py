import logging
from typing import Callable, Optional, Type, Union

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
) -> Union[Optional[Type[MaskMatchingAlgorithm]], list[str]]:
    algorithms: dict[str, Type[MaskMatchingAlgorithm]] = {
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
        max_iter: int = 10,
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
        opt_tol: float = 1e-9,
        precision=np.float32,
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
        self.precision = precision
        self._current_source_points: Optional[pd.DataFrame] = None
        self._current_target_points: Optional[pd.DataFrame] = None

    def _match_graphs_from_points(
        self,
        source_name: str,
        target_name: str,
        source_points: pd.DataFrame,
        target_points: pd.DataFrame,
        source_intensities: Optional[pd.DataFrame],
        target_intensities: Optional[pd.DataFrame],
    ) -> xr.DataArray:
        self._current_source_points = source_points
        self._current_target_points = target_points
        scores = super(Spellmatch, self)._match_graphs_from_points(
            source_name,
            target_name,
            source_points,
            target_points,
            source_intensities,
            target_intensities,
        )
        self._current_source_points = None
        self._current_target_points = None
        return scores

    def _match_graphs(
        self,
        source_adj: xr.DataArray,
        target_adj: xr.DataArray,
        source_dists: Optional[xr.DataArray],
        target_dists: Optional[xr.DataArray],
        source_intensities: Optional[pd.DataFrame],
        target_intensities: Optional[pd.DataFrame],
    ) -> xr.DataArray:
        n1 = len(source_adj)
        n2 = len(target_adj)
        adj1 = source_adj.to_numpy().astype(np.bool8)
        adj2 = target_adj.to_numpy().astype(np.bool8)
        deg1: np.ndarray = np.sum(adj1, axis=1, dtype=np.uint8)
        deg2: np.ndarray = np.sum(adj2, axis=1, dtype=np.uint8)
        adj = sparse.csr_array(sparse.kron(adj1, adj2, format="csr"), dtype=np.bool8)
        deg = np.asarray(deg1[:, np.newaxis] * deg2[np.newaxis, :], dtype=np.uint16)
        degree_cdist = None
        if self.degree_weight > 0:
            logger.info("Computing degree cross-distance")
            degree_cdist = self._compute_degree_cross_distance(deg1, deg2)
            assert degree_cdist.dtype == self.precision
        intensity_cdist_shared = None
        intensity_cdist_all = None
        if self.intensity_weight > 0:
            if source_intensities is None or target_intensities is None:
                raise SpellmatchException(
                    "Intensities are required for computing their cross-distance"
                )
            if self.intensity_interp_lmd != 0:
                logger.info("Computing intensity cross-distance (shared markers)")
                intensity_cdist_shared = self._compute_intensity_cross_distance_shared(
                    source_intensities, target_intensities
                )
                assert intensity_cdist_shared.dtype == self.precision
            if self.intensity_interp_lmd != 1:
                if (
                    self._current_source_points is None
                    or self._current_target_points is None
                ):
                    raise SpellmatchException(
                        "Computing intensity cross-distances on all markers requires "
                        "running Spellmatch in point set registration mode"
                    )
                logger.info("Computing intensity cross-distance (all markers)")
                intensity_cdist_all = self._compute_intensity_cross_distance_all(
                    self._current_source_points,
                    self._current_target_points,
                    source_intensities,
                    target_intensities,
                )
                assert intensity_cdist_all.dtype == self.precision
        distance_cdist = None
        if self.distance_weight > 0:
            if source_dists is None or target_dists is None:
                raise SpellmatchException(
                    "Distances are required for computing their cross-distance"
                )
            logger.info("Computing distance cross-distance")
            distance_cdist = self._compute_distance_cross_distance(
                adj1,
                adj2,
                source_dists.to_numpy().astype(self.precision),
                target_dists.to_numpy().astype(self.precision),
            )
            assert distance_cdist.dtype == self.precision
        spatial_cdist = None
        if (
            (self.alpha != 1.0 and self.spatial_cdist_prior_thres is not None)
            or self.max_spatial_cdist is not None
        ):
            if (
                self._current_source_points is None
                or self._current_target_points is None
            ):
                raise SpellmatchException(
                    "Computing spatial cross-distance requires running the Spellmatch "
                    "algorithm in point set registration or mask matching mode"
                )
            logger.info("Computing spatial cross-distance")
            spatial_cdist = np.asarray(
                distance.cdist(
                    self._current_source_points.to_numpy().astype(self.precision),
                    self._current_target_points.to_numpy().astype(self.precision),
                ),
                dtype=self.precision,
            )
        if (
            intensity_cdist_shared is None
            or intensity_cdist_all is None
            or 0 <= self.intensity_interp_lmd <= 1
        ):
            scores_data = self._match_graphs_for_lambda(
                n1,
                n2,
                adj,
                deg,
                degree_cdist,
                intensity_cdist_shared,
                intensity_cdist_all,
                distance_cdist,
                spatial_cdist,
                self.intensity_interp_lmd,
            )
        else:
            lmd = None
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
                current_scores_data = self._match_graphs_for_lambda(
                    n1,
                    n2,
                    adj,
                    deg,
                    degree_cdist,
                    intensity_cdist_shared,
                    intensity_cdist_all,
                    distance_cdist,
                    spatial_cdist,
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
        return scores

    def _match_graphs_for_lambda(
        self,
        n1: int,
        n2: int,
        adj: sparse.csr_array,
        deg: np.ndarray,
        degree_cdist: Optional[np.ndarray],
        intensity_cdist_shared: Optional[np.ndarray],
        intensity_cdist_all: Optional[np.ndarray],
        distance_cdist: Optional[sparse.csr_array],
        spatial_cdist: Optional[np.ndarray],
        lmd: float,
    ) -> np.ndarray:
        logger.info("Initializing")
        if intensity_cdist_shared is not None and intensity_cdist_all is not None:
            intensity_cdist = (
                lmd * intensity_cdist_shared + (1 - lmd) * intensity_cdist_all,
            )
            assert intensity_cdist.dtype == self.precision
        elif intensity_cdist_shared is not None:
            intensity_cdist = intensity_cdist_shared
        elif intensity_cdist_all is not None:
            intensity_cdist = intensity_cdist_all
        else:
            intensity_cdist = None
        w = sparse.csr_array((n1 * n2, n1 * n2), dtype=self.precision)
        total_weight = 0
        if self.degree_weight > 0:
            assert degree_cdist is not None
            degree_cdist = degree_cdist.ravel()
            w += self.degree_weight * (adj * degree_cdist[:, np.newaxis])
            w += self.degree_weight * (adj * degree_cdist[np.newaxis, :])
            total_weight += 2 * self.degree_weight
        if self.intensity_weight > 0:
            assert intensity_cdist is not None
            intensity_cdist = intensity_cdist.ravel()
            w += self.intensity_weight * (adj * intensity_cdist[:, np.newaxis])
            w += self.intensity_weight * (adj * intensity_cdist[np.newaxis, :])
            total_weight += 2 * self.intensity_weight
        if self.distance_weight > 0:
            assert distance_cdist is not None
            w += self.distance_weight * (adj * distance_cdist)
            total_weight += self.distance_weight
        if total_weight > 0:
            w /= total_weight
        assert w.dtype == self.precision
        d = np.asarray(deg.flatten(), dtype=self.precision)
        d[d != 0] = d[d != 0] ** (-0.5)
        d = sparse.dia_array((d, [0]), shape=(n1 * n2, n1 * n2))
        w: sparse.csr_array = d @ (adj - w) @ d
        assert w.dtype == self.precision
        del d
        if self.alpha != 1.0 and self.spatial_cdist_prior_thres is not None:
            assert spatial_cdist is not None
            h = np.ravel(
                1 - np.clip(spatial_cdist / self.spatial_cdist_prior_thres, 0, 1) ** 2
            )
            h = np.asarray(h / np.sum(h), dtype=self.precision)[:, np.newaxis]
        else:
            h = np.ones((n1 * n2, 1), dtype=self.precision)
            h /= n1 * n2  # does not change data type!
        assert h.dtype == self.precision
        if self.max_spatial_cdist is not None:
            assert spatial_cdist is not None
            s = np.ravel(spatial_cdist <= self.max_spatial_cdist)
            s = np.asarray(s / np.sum(s), dtype=self.precision)[:, np.newaxis]
        else:
            s = np.ones((n1 * n2, 1), dtype=self.precision)
            s /= n1 * n2  # does not change data type!
        assert s.dtype == self.precision
        logger.info("Optimizing")
        opt_converged = False
        for opt_iteration in range(self.opt_max_iter):
            s_new: np.ndarray = self.alpha * (w @ s) + (1.0 - self.alpha) * h
            opt_loss = np.linalg.norm(s[:, 0] - s_new[:, 0])
            assert s_new.dtype == self.precision
            s = s_new
            logger.debug(f"Optimizer iteration {opt_iteration:03d}: {opt_loss:.6f}")
            if opt_loss < self.opt_tol:
                opt_converged = True
                break
        if not opt_converged:
            logger.warning(
                f"Optimization did not converge after {self.opt_max_iter} iterations "
                f"(last loss: {opt_loss})"
            )
        logger.info(f"Done after {opt_iteration + 1} iterations")
        return s[:, 0].reshape((n1, n2))

    def _compute_degree_cross_distance(
        self, deg1: np.ndarray, deg2: np.ndarray
    ) -> np.ndarray:
        degree_cdiff = abs(deg1[:, np.newaxis] - deg2[np.newaxis, :])
        degree_cdist = np.clip(degree_cdiff / self.degree_cdiff_thres, 0, 1) ** 2
        return np.asarray(degree_cdist, dtype=self.precision)

    def _compute_intensity_cross_distance_shared(
        self,
        intensities1: pd.DataFrame,
        intensities2: pd.DataFrame,
    ) -> np.ndarray:
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
            f"SVD: explained variance={np.sum(svd.explained_variance_ratio_):.6f} "
            f"{tuple(np.around(r, decimals=6) for r in svd.explained_variance_ratio_)}"
        )
        svd_intensities1 = svd.transform(intensities1[intensities_shared.columns])
        svd_intensities2 = svd.transform(intensities2[intensities_shared.columns])
        intensity_cdist_shared = 0.5 * distance.cdist(
            svd_intensities1, svd_intensities2, metric="correlation"
        )
        return np.asarray(intensity_cdist_shared, dtype=self.precision)

    def _compute_intensity_cross_distance_all(
        self,
        points1: pd.DataFrame,
        points2: pd.DataFrame,
        intensities1: pd.DataFrame,
        intensities2: pd.DataFrame,
    ) -> np.ndarray:
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
        cancor_mean = np.mean(
            np.diagonal(
                np.corrcoef(cca_intensities1, cca_intensities2, rowvar=False),
                offset=cca.n_components,
            )
        )
        logger.debug(f"CCA: canonical correlations mean={cancor_mean:.6f}")
        cca_intensities1, cca_intensities2 = cca.transform(intensities1, intensities2)
        intensity_cdist_all = 0.5 * distance.cdist(
            cca_intensities1, cca_intensities2, metric="correlation"
        )
        return np.asarray(intensity_cdist_all, dtype=self.precision)

    def _compute_distance_cross_distance(
        self,
        adj1: np.ndarray,
        adj2: np.ndarray,
        dists1: np.ndarray,
        dists2: np.ndarray,
    ) -> sparse.csr_array:
        m: sparse.csr_array = abs(
            sparse.csr_array(
                sparse.kron(adj1 * dists1, adj2, format="csr"), dtype=self.precision
            )
            - sparse.csr_array(
                sparse.kron(adj1, adj2 * dists2, format="csr"), dtype=self.precision
            )
        )
        np.clip(
            m.data / self.distance_cdiff_thres,
            0,
            1,
            out=m.data,
            dtype=m.dtype,
        )
        m **= 2
        assert m.dtype == self.precision
        return m


class SpellmatchException(SpellmatchMatchingAlgorithmException):
    pass
