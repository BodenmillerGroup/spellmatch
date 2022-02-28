import logging
from typing import Optional, Type

import numpy as np
import pandas as pd
import xarray as xr
from scipy import sparse
from scipy.spatial import distance
from skimage.transform import ProjectiveTransform

from ..._spellmatch import hookimpl
from ._algorithms import IterativeGraphMatchingAlgorithm, MaskMatchingAlgorithm

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
        alpha: float = 0.8,  # TODO
        degree_weight: float = 0,  # TODO
        distance_weight: float = 0,  # TODO
        intensity_weight: float = 0,  # TODO
        degree_diff_thres: int = 0,  # TODO
        distance_diff_thres: float = 0,  # TODO
        # TODO intensity params
        use_initial_spatial_prior: bool = False,
        use_updated_spatial_prior: bool = False,
        prior_distance_thres: float = 0,  # TODO
        opt_max_iter: int = 100,
        opt_tol: float = 1e-3,
        adj_radius: float = 10,
        max_iter: int = 20,
        transform_type: str = "rigid",
        transform_estim_type: str = "max_margin",
        transform_estim_top_k: int = 50,
        points_feature: str = "centroid",
        intensities_feature: str = "intensity_mean",
    ) -> None:
        super(Spellmatch, self).__init__(
            adj_radius,
            max_iter,
            transform_type,
            transform_estim_type,
            transform_estim_top_k,
            True,
            points_feature,
            intensities_feature,
        )
        self.alpha = alpha
        self.degree_weight = degree_weight
        self.distance_weight = distance_weight
        self.intensity_weight = intensity_weight
        self.degree_diff_thres = degree_diff_thres
        self.distance_diff_thres = distance_diff_thres
        self.use_initial_spatial_prior = use_initial_spatial_prior
        self.use_updated_spatial_prior = use_updated_spatial_prior
        self.prior_distance_thres = prior_distance_thres
        self.opt_max_iter = opt_max_iter
        self.opt_tol = opt_tol
        self._current_iteration: Optional[int] = None
        self._current_source_points: Optional[pd.DataFrame] = None
        self._current_target_points: Optional[pd.DataFrame] = None
        self._w: Optional[sparse.spmatrix] = None
        self._h: Optional[np.ndarray] = None
        self._s: Optional[np.ndarray] = None
        self._s_last: Optional[np.ndarray] = None

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

    def match_graphs(
        self,
        source_adj: xr.DataArray,
        target_adj: xr.DataArray,
        source_dists: Optional[xr.DataArray] = None,
        target_dists: Optional[xr.DataArray] = None,
        source_intensities: Optional[pd.DataFrame] = None,
        target_intensities: Optional[pd.DataFrame] = None,
    ) -> xr.DataArray:
        logger.info("Initializing")
        n1 = len(source_adj)
        n2 = len(target_adj)
        adj1 = source_adj.to_numpy()
        adj2 = target_adj.to_numpy()
        adj: sparse.spmatrix = sparse.kron(source_adj.to_numpy(), target_adj.to_numpy())
        deg1: np.ndarray = np.sum(adj1, axis=0)
        deg2: np.ndarray = np.sum(adj2, axis=0)
        deg = deg1[:, np.newaxis] * deg2[np.newaxis, :]
        w = sparse.csr_matrix((n1 * n2, n1 * n2))
        if self.degree_weight > 0:
            logger.info("Computing degree cross-distance")
            deg_cdist = self._compute_degree_cross_distance(deg1, deg2).ravel()
            w += adj * (0.5 * self.degree_weight * deg_cdist[:, np.newaxis])
            w += adj * (0.5 * self.degree_weight * deg_cdist[np.newaxis, :])
            del deg_cdist
        if (
            self.distance_weight > 0
            and source_dists is not None
            and target_dists is not None
        ):
            logger.info("Computing spatial distance cross-distance")
            spdist_cdist = self._compute_spatial_distance_cross_distance(
                adj1, adj2, source_dists.to_numpy(), target_dists.to_numpy()
            )
            w += self.distance_weight * spdist_cdist
            del spdist_cdist
        if self.intensity_weight > 0:
            logger.info("Computing intensity cross-distance")
            intensity_cdist = self._compute_intensity_cross_distance().ravel()
            w += adj * (0.5 * self.intensity_weight * intensity_cdist[:, np.newaxis])
            w += adj * (0.5 * self.intensity_weight * intensity_cdist[np.newaxis, :])
        w /= self.degree_weight + self.distance_weight + self.intensity_weight
        logging.info("Normalizing optimization matrix")
        d = deg.copy()
        d[d != 0] = d[d != 0] ** (-0.5)
        d = sparse.dia_matrix((d.ravel(), [0]), shape=(n1 * n2, n1 * n2))
        w = d @ (adj - w) @ d
        del d
        if (
            self.use_initial_spatial_prior
            and self._current_source_points is not None
            and self._current_target_points is not None
        ):
            logger.info("Initializing spatial cross-distance similarity prior")
            h = self._compute_spatial_cross_distance_similarity_prior(
                self._current_source_points.values, self._current_target_points.values
            )
        else:
            logger.info("Initializing uniform prior")
            h = np.ones(n1 * n2) / (n1 * n2)
        logger.info("Optimizing")
        self._w = w
        self._h = h
        self._s = None
        self._s_last = np.ones(n1 * n2) / (n1 * n2)
        scores = super(Spellmatch, self).match_graphs(
            source_adj,
            target_adj,
            source_dists=source_dists,
            target_dists=target_dists,
            source_intensities=source_intensities,
            target_intensities=target_intensities,
        )
        self._w = self._h = self._s = self._s_last = None
        logger.info("Done")
        return scores

    def _pre_iter(
        self, iteration: int, current_transform: Optional[ProjectiveTransform]
    ) -> None:
        super(Spellmatch, self)._pre_iter(iteration, current_transform)
        self._current_iteration = iteration

    def _match_graphs(
        self,
        source_adj: xr.DataArray,
        target_adj: xr.DataArray,
        source_dists: Optional[xr.DataArray],
        target_dists: Optional[xr.DataArray],
        source_intensities: Optional[pd.DataFrame],
        target_intensities: Optional[pd.DataFrame],
    ) -> xr.DataArray:
        if (
            self.use_updated_spatial_prior
            and self._current_iteration > 0
            and self._current_source_points is not None
            and self._current_target_points is not None
        ):
            self._h = self._compute_spatial_cross_distance_similarity_prior(
                self._current_source_points.values,
                self._current_target_points.values,
            )
        self._s = self.alpha * self._w.dot(self._s_last) + (1 - self.alpha) * self._h
        scores = xr.DataArray(
            data=np.reshape(self._s, (len(source_adj), len(target_adj))),
            coords={
                source_adj.name: source_adj.coords["a"].values,
                target_adj.name: target_adj.coords["x"].values,
            },
        )
        return scores

    def _post_iter(
        self,
        iteration: int,
        current_transform: Optional[ProjectiveTransform],
        current_scores: xr.DataArray,
        updated_transform: Optional[ProjectiveTransform],
    ) -> bool:
        stop = super(Spellmatch, self)._post_iter(
            iteration, current_transform, current_scores, updated_transform
        )
        loss = np.linalg.norm(self._s_last - self._s)
        logger.info(f"Iteration {iteration}: {loss:.6f}")
        self._s_last = self._s
        self._current_iteration = None
        return stop or loss < self.opt_tol

    def _compute_degree_cross_distance(
        self, deg1: np.ndarray, deg2: np.ndarray
    ) -> np.ndarray:
        deg_cdiff = np.abs(deg1[:, np.newaxis] - deg2[np.newaxis, :])
        np.clip(deg_cdiff / self.degree_diff_thres, 0, 1, out=deg_cdiff)
        deg_cdist = deg_cdiff**2
        return deg_cdist

    def _compute_spatial_distance_cross_distance(
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
        np.clip(m.data / self.distance_diff_thres, 0, 1, out=m.data)
        spdist_cdist = m**2
        return spdist_cdist

    def _compute_intensity_cross_distance(self) -> np.ndarray:
        pass  # TODO

    def _compute_spatial_cross_distance_similarity_prior(
        self, points1: np.ndarray, points2: np.ndarray
    ) -> np.ndarray:
        spatial_cdist = distance.cdist(points1, points2)
        clipped_spatial_cdist = np.clip(spatial_cdist / self.prior_distance_thres, 0, 1)
        spatial_cdist_similarity = 1 - clipped_spatial_cdist**2
        spatial_cdist_similarity_prior = np.ravel(
            spatial_cdist_similarity / np.sum(spatial_cdist_similarity)
        )
        return spatial_cdist_similarity_prior
