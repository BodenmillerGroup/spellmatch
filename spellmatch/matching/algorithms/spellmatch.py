import logging
from typing import Optional, Type, Union

import numpy as np
import pandas as pd
import xarray as xr
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA

from ..._spellmatch import hookimpl
from ._algorithms import (
    IterativeGraphMatchingAlgorithm,
    MaskMatchingAlgorithm,
    SpellmatchMatchingAlgorithmException,
)

logger = logging.getLogger(__name__)


@hookimpl
def spellmatch_get_mask_matching_algorithm(
    name: str,
) -> Optional[Type[MaskMatchingAlgorithm]]:
    if name == "spellmatch":
        return Spellmatch
    return None


class Spellmatch(IterativeGraphMatchingAlgorithm):
    def __init__(
        self,
        *,
        alpha: float = 0.8,
        degree_weight: float = 0,
        distance_weight: float = 0,
        intensities_weight: float = 1,
        degree_diff_thres: int = 3,
        intensities_lmd: Union[int, float] = 11,
        intensities_lmd_n_components: int = 10,
        intensities_lmd_max_iter: int = 500,
        intensities_lmd_tol: float = 1e-6,
        intensities_shared_n_components: Union[int, float, str] = 20,
        intensities_full_matching_top_k: int = 100,
        intensities_full_n_components: int = 20,
        intensities_full_max_iter: int = 500,
        intensities_full_tol: float = 1e-6,
        use_spatial_prior: bool = True,
        opt_max_iter: int = 100,
        opt_tol: float = 1e-9,
        adj_radius: float = 15,
        max_iter: int = 3,
        transform_type: str = "rigid",
        transform_estim_type: str = "max_margin",
        transform_estim_top_k: int = 50,
        exclude_outliers: bool = True,
        points_feature: str = "centroid",
        intensities_feature: str = "intensity_mean",
    ) -> None:
        super(Spellmatch, self).__init__(
            adj_radius,
            max_iter,
            transform_type,
            transform_estim_type,
            transform_estim_top_k,
            exclude_outliers,
            points_feature,
            intensities_feature,
        )
        self.alpha = alpha
        self.degree_weight = degree_weight
        self.distance_weight = distance_weight
        self.intensities_weight = intensities_weight
        self.degree_diff_thres = degree_diff_thres
        self.intensities_lmd = intensities_lmd
        self.intensities_lmd_n_components = intensities_lmd_n_components
        self.intensities_lmd_max_iter = intensities_lmd_max_iter
        self.intensities_lmd_tol = intensities_lmd_tol
        self.intensities_shared_n_components = intensities_shared_n_components
        self.intensities_full_matching_top_k = intensities_full_matching_top_k
        self.intensities_full_n_components = intensities_full_n_components
        self.intensities_full_max_iter = intensities_full_max_iter
        self.intensities_full_tol = intensities_full_tol
        self.use_spatial_prior = use_spatial_prior
        self.opt_max_iter = opt_max_iter
        self.opt_tol = opt_tol
        self._current_source_points: Optional[pd.DataFrame] = None
        self._current_target_points: Optional[pd.DataFrame] = None

    def _match_graphs_from_points(
        self,
        source_points: pd.DataFrame,
        target_points: pd.DataFrame,
        source_intensities: Optional[pd.DataFrame],
        target_intensities: Optional[pd.DataFrame],
    ) -> xr.DataArray:
        self._current_source_points = source_points
        self._current_target_points = target_points
        scores = super(Spellmatch, self)._match_graphs_from_points(
            source_points, target_points, source_intensities, target_intensities
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
        adj1 = source_adj.to_numpy()
        adj2 = target_adj.to_numpy()
        deg1: np.ndarray = np.sum(adj1, axis=0)
        deg2: np.ndarray = np.sum(adj2, axis=0)
        adj: sparse.spmatrix = sparse.kron(adj1, adj2)
        deg_cdist = None
        if self.degree_weight > 0:
            logger.info("Computing degree cross-distance")
            deg_cdist = self._compute_degree_cross_distances(deg1, deg2)
        spdist_cdists = None
        if self.distance_weight > 0:
            if source_dists is None or target_dists is None:
                raise SpellmatchException(
                    "Spatial distances are required for computing their cross-distances"
                )
            logger.info("Computing spatial distance cross-distances")
            spdist_cdists = self._compute_spatial_distance_cross_distances(
                adj1, adj2, source_dists.to_numpy(), target_dists.to_numpy()
            )
        intensities_cdists_shared = 0
        intensities_cdists_all = 0
        if self.intensities_weight > 0:
            if source_intensities is None or target_intensities is None:
                raise SpellmatchException(
                    "Intensities are required for computing their cross-distances"
                )
            if self.intensities_lmd != 0:
                logger.info("Computing intensities cross-distances (shared features)")
                intensities_cdists_shared = (
                    self._compute_intensities_cross_distances_shared(
                        source_intensities, target_intensities
                    )
                )
            if self.intensities_lmd != 1:
                if (
                    self._current_source_points is None
                    or self._current_target_points is None
                ):
                    raise SpellmatchException(
                        "Computing intensities cross-distances (all features) requires "
                        "running the Spellmatch algorithm in point registration mode"
                    )
                logger.info("Computing intensities cross-distances (all features)")
                intensities_cdists_all = self._compute_intensities_cross_distances_all(
                    self._current_source_points,
                    self._current_target_points,
                    source_intensities,
                    target_intensities,
                )
        if self.use_spatial_prior:
            if (
                self._current_source_points is None
                or self._current_target_points is None
            ):
                raise SpellmatchException(
                    "Computing a spatial cross-distance similarity prior requires "
                    "running the Spellmatch algorithm in point registration mode"
                )
            logger.info("Computing spatial cross-distance similarity prior")
            h = self._compute_spatial_cross_distance_similarity_prior(
                self._current_source_points, self._current_target_points
            )
        else:
            h = np.ones(n1 * n2) / (n1 * n2)
        best_s = None
        best_lmd = None
        best_mean_cancor = None
        if 0 <= self.intensities_lmd <= 1:
            lmds = [self.intensities_lmd]
        else:
            lmds = np.linspace(0, 1, self.intensities_lmd)
        for lmd in lmds:
            logger.info(f"Initializing (lambda={lmd})")
            intensities_cdist: np.ndarray = (
                lmd * intensities_cdists_shared + (1 - lmd) * intensities_cdists_all
            )
            w = sparse.csr_matrix((n1 * n2, n1 * n2))
            total_weight = 0
            if deg_cdist is not None:
                w += adj * (self.degree_weight * deg_cdist.ravel()[:, np.newaxis])
                w += adj * (self.degree_weight * deg_cdist.ravel()[np.newaxis, :])
                total_weight += 2 * self.degree_weight
            if spdist_cdists is not None:
                w += self.distance_weight * spdist_cdists
                total_weight += self.distance_weight
            if intensities_cdist is not None:
                w += adj * (
                    self.intensities_weight * intensities_cdist.ravel()[:, np.newaxis]
                )
                w += adj * (
                    self.intensities_weight * intensities_cdist.ravel()[np.newaxis, :]
                )
                total_weight += 2 * self.intensities_weight
            if total_weight > 0:
                w /= total_weight
            d = deg1[:, np.newaxis] * deg2[np.newaxis, :]
            d[d != 0] = d[d != 0] ** (-0.5)
            d = sparse.dia_matrix((d.ravel(), [0]), shape=(n1 * n2, n1 * n2))
            w: sparse.spmatrix = d @ (adj - w) @ d
            logger.info("Optimizing")
            s = np.ones(n1 * n2) / (n1 * n2)
            for opt_iter in range(self.opt_max_iter):
                s_new = self.alpha * w.dot(s) + (1 - self.alpha) * h
                loss = np.linalg.norm(s - s_new)
                logger.info(f"Iteration {opt_iter}: {loss:.6f}")
                s = s_new
                if loss < self.opt_tol:
                    break
            logger.info("Done")
            if len(lmds) > 1:
                row_ind, col_ind = linear_sum_assignment(
                    np.reshape(best_s, (n1, n2)), maximize=True
                )
                cca = CCA(
                    n_components=self.intensities_lmd_n_components,
                    max_iter=self.intensities_lmd_max_iter,
                    tol=self.intensities_lmd_tol,
                )
                cca1, cca2 = cca.fit_transform(
                    source_intensities.iloc[row_ind, :],
                    target_intensities.iloc[col_ind, :],
                )
                mean_cancor = np.mean(
                    np.diagonal(
                        np.corrcoef(cca1, cca2, rowvar=False), offset=cca.n_components
                    )
                )
                logger.info(
                    f"Mean canonical correlation of naive matching: {mean_cancor}"
                )
                if best_mean_cancor is None or mean_cancor > best_mean_cancor:
                    best_s = s
                    best_lmd = lmd
                    best_mean_cancor = mean_cancor
            else:
                best_s = s
                best_lmd = lmd
        if len(lmds) > 1:
            logger.info(
                f"Best lambda: {best_lmd} "
                f"(mean canonical correlation of naive matching: {best_mean_cancor})"
            )
        scores = xr.DataArray(
            data=np.reshape(best_s, (n1, n2)),
            coords={
                source_adj.name: source_adj.coords["a"].to_numpy(),
                target_adj.name: target_adj.coords["x"].to_numpy(),
            },
        )
        return scores

    def _compute_degree_cross_distances(
        self, deg1: np.ndarray, deg2: np.ndarray
    ) -> np.ndarray:
        deg_cdiffs = np.abs(deg1[:, np.newaxis] - deg2[np.newaxis, :])
        np.clip(deg_cdiffs / self.degree_diff_thres, 0, 1, out=deg_cdiffs)
        deg_cdists = deg_cdiffs**2
        return deg_cdists

    def _compute_spatial_distance_cross_distances(
        self,
        adj1: np.ndarray,
        adj2: np.ndarray,
        spdist1: np.ndarray,
        spdist2: np.ndarray,
    ) -> sparse.csr_matrix:
        m: sparse.csr_matrix = abs(
            sparse.kron(adj1 * spdist1, adj2, format="csr")
            - sparse.kron(adj1, adj2 * spdist2, format="csr")
        )
        np.clip(m.data / self.adj_radius, 0, 1, out=m.data)
        spdist_cdists = m**2
        return spdist_cdists

    def _compute_intensities_cross_distances_shared(
        self,
        intensities1: pd.DataFrame,
        intensities2: pd.DataFrame,
    ) -> np.ndarray:
        shared_intensities = pd.concat(
            (intensities1, intensities2), join="inner", ignore_index=True
        )
        n_components = min(
            self.intensities_shared_n_components, len(shared_intensities.columns)
        )
        # TODO use TruncatedSVD on centered and scaled intensities instead
        pca = PCA(n_components=n_components)
        pca.fit(shared_intensities)
        pca1: pd.DataFrame = pca.transform(intensities1[shared_intensities.columns])
        pca2: pd.DataFrame = pca.transform(intensities2[shared_intensities.columns])
        intensities_cdists_shared = distance.cdist(pca1, pca2, metric="correlation")
        return (intensities_cdists_shared + 1) / 2

    def _compute_intensities_cross_distances_all(
        self,
        points1: pd.DataFrame,
        points2: pd.DataFrame,
        intensities1: pd.DataFrame,
        intensities2: pd.DataFrame,
    ) -> np.ndarray:
        spatial_cdists = distance.cdist(points1, points2)
        source_2nn_ind = np.argpartition(-spatial_cdists, 1, axis=1)[:, :2]
        source_2nn_dists = np.take_along_axis(spatial_cdists, source_2nn_ind, axis=1)
        source_margins = source_2nn_dists[:, 0] - source_2nn_dists[:, 1]
        top_source_ind = np.argpartition(
            -source_margins, self.intensities_full_matching_top_k - 1
        )[: self.intensities_full_matching_top_k]
        top_target_ind = source_2nn_ind[top_source_ind, 0]
        n_components = min(
            self.intensities_full_n_components,
            len(intensities1.columns),
            len(intensities2.columns),
        )
        cca = CCA(
            n_components=n_components,
            max_iter=self.intensities_full_max_iter,
            tol=self.intensities_full_tol,
        )
        cca.fit(intensities1.iloc[top_source_ind], intensities2.iloc[top_target_ind])
        cca1: pd.DataFrame = cca.transform(intensities1)
        cca2: pd.DataFrame = cca.transform(intensities2)
        intensities_cdists_all = distance.cdist(cca1, cca2, metric="correlation")
        return (intensities_cdists_all + 1) / 2

    def _compute_spatial_cross_distance_similarity_prior(
        self, points1: pd.DataFrame, points2: pd.DataFrame
    ) -> np.ndarray:
        spatial_cdists = distance.cdist(points1, points2)
        clipped_spatial_cdists = np.clip(spatial_cdists / self.adj_radius, 0, 1)
        spatial_cdist_similarity = 1 - clipped_spatial_cdists**2
        spatial_cdist_similarity_prior = np.ravel(
            spatial_cdist_similarity / np.sum(spatial_cdist_similarity)
        )
        return spatial_cdist_similarity_prior


class SpellmatchException(SpellmatchMatchingAlgorithmException):
    pass
