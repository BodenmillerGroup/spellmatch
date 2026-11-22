"""Matching algorithms based on the registration algorithms in ``_vendor``."""

import logging
from abc import abstractmethod
from collections.abc import Callable, MutableMapping
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from skimage.transform import ProjectiveTransform
from sklearn.neighbors import NearestNeighbors

from ...._spellmatch import hookimpl
from .._algorithms import MaskMatchingAlgorithm, PointsMatchingAlgorithm
from ._vendor import bcpd, cpd, l2dist_regs
from ._vendor.transformation import NormalizedTransformation, Transformation

try:
    import cupy as cp  # type: ignore
except ImportError:
    cp = None

logger = logging.getLogger(__name__.rpartition(".")[0])


def _cuda_available() -> bool:
    if cp is None:
        return False
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception as e:  # e.g. missing CUDA driver/runtime
        logger.debug(f"CUDA is not available: {e}")
        return False


def _get_normalization(
    source: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, float]:
    """Common centroid and scale (standard deviation) of both point sets."""
    points = np.concatenate((source, target))
    scale = float(np.std(points - np.mean(points, axis=0)))
    return np.mean(points, axis=0), scale if scale > 0 else 1.0


def _asnumpy(data: Any) -> np.ndarray:
    """Like cupy.asnumpy: copy CuPy arrays (GPU) to the host, if CuPy is installed."""
    if cp is not None and isinstance(data, cp.ndarray):
        return cp.asnumpy(data)
    return np.asarray(data)


@hookimpl
def spellmatch_get_mask_matching_algorithm(
    name: str | None,
) -> type["MaskMatchingAlgorithm"] | None | list[str]:
    algorithms = {
        "rigid_cpd": RigidCoherentPointDrift,
        "affine_cpd": AffineCoherentPointDrift,
        "nonrigid_cpd": NonRigidCoherentPointDrift,
        "combined_bcpd": CombinedBayesianCoherentPointDrift,
        "rigid_gmmreg": RigidGMMReg,
        "tps_gmmreg": TPSGMMReg,
        "rigid_svr": RigidSVR,
        "tps_svr": TPSSVR,
    }
    if name is not None:
        return algorithms.get(name)
    return list(algorithms.keys())


class _Probreg(PointsMatchingAlgorithm):
    def __init__(
        self,
        *,
        outlier_dist: float | None,
        point_feature: str,
        intensity_feature: str,
        intensity_transform: str | Callable[[np.ndarray], np.ndarray] | None,
        max_dist: float | None,
    ) -> None:
        super().__init__(
            outlier_dist=outlier_dist,
            point_feature=point_feature,
            intensity_feature=intensity_feature,
            intensity_transform=intensity_transform,
        )
        self.max_dist = max_dist
        self._current_iteration: int | None = None

    def _match_points(
        self,
        source_name: str,
        target_name: str,
        source_points: pd.DataFrame,
        target_points: pd.DataFrame,
        source_intensities: pd.DataFrame | None,
        target_intensities: pd.DataFrame | None,
        prior_transform: ProjectiveTransform | None,
        cache: MutableMapping[str, Any] | None,
    ) -> tuple[dict[str, Any], xr.DataArray]:
        self._current_iteration = 0
        transform = self._register_points(
            source_points.to_numpy(), target_points.to_numpy()
        )
        info = {"iterations": self._current_iteration}
        self._current_iteration = None
        source_ind = np.arange(len(source_points.index))
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(target_points.to_numpy())
        nn_dists, nn_ind = nn.kneighbors(
            _asnumpy(transform.transform(source_points.to_numpy()))
        )
        dists, target_ind = nn_dists[:, 0], nn_ind[:, 0]
        if self.max_dist:
            source_ind = source_ind[dists <= self.max_dist]
            target_ind = target_ind[dists <= self.max_dist]
            dists = dists[dists <= self.max_dist]
        scores_data = np.zeros((len(source_points.index), len(target_points.index)))
        scores_data[source_ind, target_ind] = 1
        scores = xr.DataArray(
            data=scores_data,
            coords={
                source_name: source_points.index.to_numpy(),
                target_name: target_points.index.to_numpy(),
            },
        )
        return info, scores

    @abstractmethod
    def _register_points(
        self, source: np.ndarray, target: np.ndarray
    ) -> Transformation:
        raise NotImplementedError()

    def _callback(self, transformation: Transformation) -> None:
        self._current_iteration += 1
        logger.info(f"Iteration {self._current_iteration:03d}")


class _CoherentPointDrift(_Probreg):
    def __init__(
        self,
        *,
        outlier_dist: float | None,
        point_feature: str,
        intensity_feature: str,
        intensity_transform: str | Callable[[np.ndarray], np.ndarray] | None,
        max_dist: float | None,
        w: float,
        maxiter: int,
        tol: float,
    ) -> None:
        super().__init__(
            outlier_dist=outlier_dist,
            point_feature=point_feature,
            intensity_feature=intensity_feature,
            intensity_transform=intensity_transform,
            max_dist=max_dist,
        )
        self.w = w
        self.maxiter = maxiter
        self.tol = tol

    def _register_points(
        self, source: np.ndarray, target: np.ndarray
    ) -> Transformation:
        instance = self._get_instance(source)
        instance.set_callbacks([self._instance_callback])
        result = instance.registration(
            target, w=self.w, maxiter=self.maxiter, tol=self.tol
        )
        return result.transformation

    @abstractmethod
    def _get_instance(self, source: np.ndarray) -> cpd.CoherentPointDrift:
        raise NotImplementedError()

    def _instance_callback(self, transformation: Transformation) -> None:
        self._callback(transformation)


class RigidCoherentPointDrift(_CoherentPointDrift):
    def __init__(
        self,
        *,
        outlier_dist: float | None = None,
        point_feature: str = "centroid",
        intensity_feature: str = "intensity_mean",
        intensity_transform: str | Callable[[np.ndarray], np.ndarray] | None = None,
        max_dist: float | None = None,
        w: float = 0,
        maxiter: int = 50,
        tol: float = 0.001,
        update_scale: bool = True,
    ) -> None:
        super().__init__(
            outlier_dist=outlier_dist,
            point_feature=point_feature,
            intensity_feature=intensity_feature,
            intensity_transform=intensity_transform,
            max_dist=max_dist,
            w=w,
            maxiter=maxiter,
            tol=tol,
        )
        self.update_scale = update_scale
        self.use_cuda = _cuda_available()

    def _get_instance(self, source: np.ndarray) -> cpd.CoherentPointDrift:
        return cpd.RigidCPD(
            source=source, update_scale=self.update_scale, use_cuda=self.use_cuda
        )


class AffineCoherentPointDrift(_CoherentPointDrift):
    def __init__(
        self,
        *,
        outlier_dist: float | None = None,
        point_feature: str = "centroid",
        intensity_feature: str = "intensity_mean",
        intensity_transform: str | Callable[[np.ndarray], np.ndarray] | None = None,
        max_dist: float | None = None,
        w: float = 0,
        maxiter: int = 50,
        tol: float = 0.001,
    ) -> None:
        super().__init__(
            outlier_dist=outlier_dist,
            point_feature=point_feature,
            intensity_feature=intensity_feature,
            intensity_transform=intensity_transform,
            max_dist=max_dist,
            w=w,
            maxiter=maxiter,
            tol=tol,
        )
        self.use_cuda = _cuda_available()

    def _get_instance(self, source: np.ndarray) -> cpd.CoherentPointDrift:
        return cpd.AffineCPD(source=source, use_cuda=self.use_cuda)


class NonRigidCoherentPointDrift(_CoherentPointDrift):
    def __init__(
        self,
        *,
        outlier_dist: float | None = None,
        point_feature: str = "centroid",
        intensity_feature: str = "intensity_mean",
        intensity_transform: str | Callable[[np.ndarray], np.ndarray] | None = None,
        max_dist: float | None = None,
        w: float = 0,
        maxiter: int = 50,
        tol: float = 0.001,
        beta: float = 2,
        lmd: float = 2,
    ) -> None:
        super().__init__(
            outlier_dist=outlier_dist,
            point_feature=point_feature,
            intensity_feature=intensity_feature,
            intensity_transform=intensity_transform,
            max_dist=max_dist,
            w=w,
            maxiter=maxiter,
            tol=tol,
        )
        self.beta = beta
        self.lmd = lmd
        self.use_cuda = _cuda_available()

    def _get_instance(self, source: np.ndarray) -> cpd.CoherentPointDrift:
        return cpd.NonRigidCPD(
            source=source, beta=self.beta, lmd=self.lmd, use_cuda=self.use_cuda
        )


class _BayesianCoherentPointDrift(_Probreg):
    def __init__(
        self,
        *,
        outlier_dist: float | None,
        point_feature: str,
        intensity_feature: str,
        intensity_transform: str | Callable[[np.ndarray], np.ndarray] | None,
        max_dist: float | None,
        w: float,
        maxiter: int,
        tol: float,
    ) -> None:
        super().__init__(
            outlier_dist=outlier_dist,
            point_feature=point_feature,
            intensity_feature=intensity_feature,
            intensity_transform=intensity_transform,
            max_dist=max_dist,
        )
        self.w = w
        self.maxiter = maxiter
        self.tol = tol

    def _register_points(
        self, source: np.ndarray, target: np.ndarray
    ) -> Transformation:
        # parameters are defined relative to the scale of the (normalized) point sets
        centroid, scale = _get_normalization(source, target)
        instance = self._get_instance((source - centroid) / scale)
        instance.set_callbacks([self._instance_callback])
        result_transformation = instance.registration(
            (target - centroid) / scale, w=self.w, maxiter=self.maxiter, tol=self.tol
        )
        return NormalizedTransformation(result_transformation, centroid, scale)

    @abstractmethod
    def _get_instance(self, source: np.ndarray) -> bcpd.CombinedBCPD:
        raise NotImplementedError()

    def _instance_callback(self, transformation: Transformation) -> None:
        self._callback(transformation)


class CombinedBayesianCoherentPointDrift(_BayesianCoherentPointDrift):
    def __init__(
        self,
        *,
        outlier_dist: float | None = None,
        point_feature: str = "centroid",
        intensity_feature: str = "intensity_mean",
        intensity_transform: str | Callable[[np.ndarray], np.ndarray] | None = None,
        max_dist: float | None = None,
        w: float = 0,
        maxiter: int = 50,
        tol: float = 0.001,
        lmd: float = 50,
        beta: float = 2,
        k: float = 1e20,
        gamma: float = 1,
    ) -> None:
        super().__init__(
            outlier_dist=outlier_dist,
            point_feature=point_feature,
            intensity_feature=intensity_feature,
            intensity_transform=intensity_transform,
            max_dist=max_dist,
            w=w,
            maxiter=maxiter,
            tol=tol,
        )
        self.lmd = lmd
        self.beta = beta
        self.k = k
        self.gamma = gamma

    def _get_instance(self, source: np.ndarray) -> bcpd.CombinedBCPD:
        return bcpd.CombinedBCPD(
            source, lmd=self.lmd, beta=self.beta, k=self.k, gamma=self.gamma
        )


class _L2DistReg(_Probreg):
    def __init__(
        self,
        *,
        outlier_dist: float | None,
        point_feature: str,
        intensity_feature: str,
        intensity_transform: str | Callable[[np.ndarray], np.ndarray] | None,
        max_dist: float | None,
        maxiter: int,
        tol: float,
        opt_maxiter: int,
        opt_tol: float,
    ) -> None:
        super().__init__(
            outlier_dist=outlier_dist,
            point_feature=point_feature,
            intensity_feature=intensity_feature,
            intensity_transform=intensity_transform,
            max_dist=max_dist,
        )
        self.maxiter = maxiter
        self.tol = tol
        self.opt_maxiter = opt_maxiter
        self.opt_tol = opt_tol

    def _register_points(
        self, source: np.ndarray, target: np.ndarray
    ) -> Transformation:
        # normalized point sets (the 2D thin plate spline kernel r^2 log(r) is
        # numerically unstable for unnormalized coordinates)
        centroid, scale = _get_normalization(source, target)
        instance = self._get_instance((source - centroid) / scale)
        instance.set_callbacks([self._instance_callback])
        result_transformation = instance.registration(
            (target - centroid) / scale,
            maxiter=self.maxiter,
            tol=self.tol,
            opt_maxiter=self.opt_maxiter,
            opt_tol=self.opt_tol,
        )
        return NormalizedTransformation(result_transformation, centroid, scale)

    @abstractmethod
    def _get_instance(self, source: np.ndarray) -> l2dist_regs.L2DistRegistration:
        raise NotImplementedError()

    def _instance_callback(self, transformation: Transformation) -> None:
        self._callback(transformation)


class RigidGMMReg(_L2DistReg):
    def __init__(
        self,
        *,
        outlier_dist: float | None = None,
        point_feature: str = "centroid",
        intensity_feature: str = "intensity_mean",
        intensity_transform: str | Callable[[np.ndarray], np.ndarray] | None = None,
        max_dist: float | None = None,
        maxiter: int = 1,
        tol: float = 1e-3,
        opt_maxiter: int = 50,
        opt_tol: float = 1e-3,
        sigma: float = 1,
        delta: float = 0.9,
        n_gmm_components: int = 800,
        use_estimated_sigma: bool = True,
    ) -> None:
        super().__init__(
            outlier_dist=outlier_dist,
            point_feature=point_feature,
            intensity_feature=intensity_feature,
            intensity_transform=intensity_transform,
            max_dist=max_dist,
            maxiter=maxiter,
            tol=tol,
            opt_maxiter=opt_maxiter,
            opt_tol=opt_tol,
        )
        self.sigma = sigma
        self.delta = delta
        self.n_gmm_components = n_gmm_components
        self.use_estimated_sigma = use_estimated_sigma

    def _get_instance(self, source: np.ndarray) -> l2dist_regs.L2DistRegistration:
        return l2dist_regs.RigidGMMReg(
            source,
            sigma=self.sigma,
            delta=self.delta,
            n_gmm_components=self.n_gmm_components,
            use_estimated_sigma=self.use_estimated_sigma,
        )


class TPSGMMReg(_L2DistReg):
    def __init__(
        self,
        *,
        outlier_dist: float | None = None,
        point_feature: str = "centroid",
        intensity_feature: str = "intensity_mean",
        intensity_transform: str | Callable[[np.ndarray], np.ndarray] | None = None,
        max_dist: float | None = None,
        maxiter: int = 1,
        tol: float = 1e-3,
        opt_maxiter: int = 50,
        opt_tol: float = 1e-3,
        sigma: float = 1,
        delta: float = 0.9,
        n_gmm_components: int = 800,
        alpha: float = 1,
        beta: float = 0.1,
        use_estimated_sigma: bool = True,
    ) -> None:
        super().__init__(
            outlier_dist=outlier_dist,
            point_feature=point_feature,
            intensity_feature=intensity_feature,
            intensity_transform=intensity_transform,
            max_dist=max_dist,
            maxiter=maxiter,
            tol=tol,
            opt_maxiter=opt_maxiter,
            opt_tol=opt_tol,
        )
        self.sigma = sigma
        self.delta = delta
        self.n_gmm_components = n_gmm_components
        self.alpha = alpha
        self.beta = beta
        self.use_estimated_sigma = use_estimated_sigma

    def _get_instance(self, source: np.ndarray) -> l2dist_regs.L2DistRegistration:
        return l2dist_regs.TPSGMMReg(
            source,
            sigma=self.sigma,
            delta=self.delta,
            n_gmm_components=self.n_gmm_components,
            alpha=self.alpha,
            beta=self.beta,
            use_estimated_sigma=self.use_estimated_sigma,
        )


class RigidSVR(_L2DistReg):
    def __init__(
        self,
        *,
        outlier_dist: float | None = None,
        point_feature: str = "centroid",
        intensity_feature: str = "intensity_mean",
        intensity_transform: str | Callable[[np.ndarray], np.ndarray] | None = None,
        max_dist: float | None = None,
        maxiter: int = 1,
        tol: float = 1e-3,
        opt_maxiter: int = 50,
        opt_tol: float = 1e-3,
        sigma: float = 1,
        delta: float = 0.9,
        gamma: float = 0.5,
        nu: float = 0.1,
        use_estimated_sigma: bool = True,
    ) -> None:
        super().__init__(
            outlier_dist=outlier_dist,
            point_feature=point_feature,
            intensity_feature=intensity_feature,
            intensity_transform=intensity_transform,
            max_dist=max_dist,
            maxiter=maxiter,
            tol=tol,
            opt_maxiter=opt_maxiter,
            opt_tol=opt_tol,
        )
        self.sigma = sigma
        self.delta = delta
        self.gamma = gamma
        self.nu = nu
        self.use_estimated_sigma = use_estimated_sigma

    def _get_instance(self, source: np.ndarray) -> l2dist_regs.L2DistRegistration:
        return l2dist_regs.RigidSVR(
            source,
            sigma=self.sigma,
            delta=self.delta,
            gamma=self.gamma,
            nu=self.nu,
            use_estimated_sigma=self.use_estimated_sigma,
        )


class TPSSVR(_L2DistReg):
    def __init__(
        self,
        *,
        outlier_dist: float | None = None,
        point_feature: str = "centroid",
        intensity_feature: str = "intensity_mean",
        intensity_transform: str | Callable[[np.ndarray], np.ndarray] | None = None,
        max_dist: float | None = None,
        maxiter: int = 1,
        tol: float = 1e-3,
        opt_maxiter: int = 50,
        opt_tol: float = 1e-3,
        sigma: float = 1,
        delta: float = 0.9,
        gamma: float = 0.5,
        nu: float = 0.1,
        alpha: float = 1,
        beta: float = 10,  # probreg: 0.1 (unstable for normalized point sets)
        use_estimated_sigma: bool = True,
    ) -> None:
        super().__init__(
            outlier_dist=outlier_dist,
            point_feature=point_feature,
            intensity_feature=intensity_feature,
            intensity_transform=intensity_transform,
            max_dist=max_dist,
            maxiter=maxiter,
            tol=tol,
            opt_maxiter=opt_maxiter,
            opt_tol=opt_tol,
        )
        self.sigma = sigma
        self.delta = delta
        self.gamma = gamma
        self.nu = nu
        self.alpha = alpha
        self.beta = beta
        self.use_estimated_sigma = use_estimated_sigma

    def _get_instance(self, source: np.ndarray) -> l2dist_regs.L2DistRegistration:
        return l2dist_regs.TPSSVR(
            source,
            sigma=self.sigma,
            delta=self.delta,
            gamma=self.gamma,
            nu=self.nu,
            alpha=self.alpha,
            beta=self.beta,
            use_estimated_sigma=self.use_estimated_sigma,
        )
