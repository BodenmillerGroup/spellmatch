# Adapted from probreg 0.3.8 (MIT License, Copyright (c) 2019 neka-nat), see LICENSE
# (l2dist_regs.py, cost_functions.py, features.py)
#
# Changes: exact Gauss transform, rigid cost function for 2D (rotation angle) in
# addition to 3D (quaternion), corrected gradients (L2 distance: factor 1/2; quaternion
# derivative of the rotation matrix), verified against finite differences.
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np
from scipy.optimize import minimize
from sklearn import mixture, svm

from . import kernels
from . import transformation as tf

logger = logging.getLogger(__name__)


class Feature(ABC):
    @abstractmethod
    def init(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def compute(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError()

    def annealing(self) -> None:
        pass


class GMM(Feature):
    """Feature points extraction using a Gaussian mixture model."""

    def __init__(self, n_gmm_components: int = 800) -> None:
        self._n_gmm_components = n_gmm_components

    def init(self) -> None:
        # fixed random state for reproducibility (not set in probreg)
        self._clf = mixture.GaussianMixture(
            n_components=self._n_gmm_components,
            covariance_type="spherical",
            random_state=0,
        )

    def compute(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # at most 80% of the points of each point set (probreg only limited the number
        # of components by the source point set, failing for smaller target point sets)
        n_components = min(self._n_gmm_components, int(data.shape[0] * 0.8))
        self._clf.set_params(n_components=n_components)
        self._clf.fit(data)
        return self._clf.means_, self._clf.weights_


class OneClassSVM(Feature):
    """Feature points extraction using a one-class SVM."""

    def __init__(
        self,
        dim: int,
        sigma: float,
        gamma: float = 0.5,
        nu: float = 0.05,
        delta: float = 10.0,
    ) -> None:
        self._dim = dim
        self._sigma = sigma
        self._gamma = gamma
        self._nu = nu
        self._delta = delta

    def init(self) -> None:
        self._clf = svm.OneClassSVM(nu=self._nu, kernel="rbf", gamma=self._gamma)

    def compute(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self._clf.fit(data)
        z = np.power(2.0 * np.pi * self._sigma**2, self._dim * 0.5)
        return self._clf.support_vectors_, self._clf.dual_coef_[0] * z

    def annealing(self) -> None:
        self._gamma *= self._delta


def compute_l2_dist(
    mu_source: np.ndarray,
    phi_source: np.ndarray,
    mu_target: np.ndarray,
    phi_target: np.ndarray,
    sigma: float,
) -> tuple[float, np.ndarray]:
    """L2 distance term and its gradient with respect to ``mu_source``."""
    z = np.power(2.0 * np.pi * sigma**2, mu_source.shape[1] * 0.5)
    h = np.sqrt(2.0) * sigma
    phi_j_e = kernels.gauss_transform(mu_target, mu_source, phi_target / z, h)
    phi_mu_j_e = kernels.gauss_transform(
        mu_target, mu_source, phi_target * mu_target.T / z, h
    ).T
    g = (phi_source * phi_j_e * mu_source.T - phi_source * phi_mu_j_e.T).T
    # probreg divided by 2 sigma^2, i.e., computed half of the gradient
    g /= sigma**2
    return float(-np.dot(phi_source, phi_j_e)), g


class CostFunction(ABC):
    @abstractmethod
    def to_transformation(self, theta: np.ndarray) -> tf.Transformation:
        raise NotImplementedError()

    @abstractmethod
    def initial(self) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def __call__(self, theta: np.ndarray, *args) -> tuple[float, np.ndarray]:
        raise NotImplementedError()


class RigidCostFunction(CostFunction):
    """Rigid transformation (rotation angle in 2D, quaternion in 3D, translation)."""

    def __init__(self, dim: int) -> None:
        if dim not in (2, 3):
            raise ValueError(f"Unsupported dimensionality: {dim}")
        self._dim = dim

    def _rotation(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Rotation matrix and its derivative with respect to the rotation params."""
        if self._dim == 2:
            c, s = np.cos(theta[0]), np.sin(theta[0])
            rot = np.array([[c, -s], [s, c]])
            d_rot = np.array([[[-s, -c], [c, -s]]])
            return rot, d_rot
        return kernels.quat2mat(theta[:4]), kernels.diff_rot_from_quaternion(theta[:4])

    def _n_rot_params(self) -> int:
        return 1 if self._dim == 2 else 4

    def to_transformation(self, theta: np.ndarray) -> tf.Transformation:
        rot, _ = self._rotation(theta)
        return tf.RigidTransformation(rot, theta[self._n_rot_params() :])

    def initial(self) -> np.ndarray:
        x0 = np.zeros(self._n_rot_params() + self._dim)
        if self._dim == 3:
            x0[0] = 1.0  # identity quaternion
        return x0

    def __call__(self, theta: np.ndarray, *args) -> tuple[float, np.ndarray]:
        mu_source, phi_source, mu_target, phi_target, sigma = args
        rot, d_rot = self._rotation(theta)
        t_mu_source = np.dot(mu_source, rot.T) + theta[self._n_rot_params() :]
        f, g = compute_l2_dist(t_mu_source, phi_source, mu_target, phi_target, sigma)
        gtm0 = np.dot(g.T, mu_source)
        grad = np.concatenate([(gtm0 * d_rot).sum(axis=(1, 2)), g.sum(axis=0)])
        return f, grad


class TPSCostFunction(CostFunction):
    def __init__(
        self, control_pts: np.ndarray, alpha: float = 1.0, beta: float = 0.1
    ) -> None:
        self._alpha = alpha
        self._beta = beta
        self._control_pts = control_pts

    def to_transformation(self, theta: np.ndarray) -> tf.TPSTransformation:
        dim = self._control_pts.shape[1]
        n_data = theta.shape[0] // dim
        n_a = dim * (dim + 1)
        a = theta[:n_a].reshape(dim + 1, dim)
        v = theta[n_a:].reshape(n_data - dim - 1, dim)
        return tf.TPSTransformation(a, v, self._control_pts)

    def initial(self) -> np.ndarray:
        dim = self._control_pts.shape[1]
        a = np.r_[np.zeros((1, dim)), np.identity(dim)]
        v = np.zeros((self._control_pts.shape[0] - dim - 1, dim))
        return np.r_[a, v].flatten()

    def __call__(self, theta: np.ndarray, *args) -> tuple[float, np.ndarray]:
        dim = self._control_pts.shape[1]
        mu_source, phi_source, mu_target, phi_target, sigma = args
        tf_obj = self.to_transformation(theta)
        basis, kernel = tf_obj.prepare(mu_source)
        t_mu_source = tf_obj.transform_basis(basis)
        bending = np.trace(np.dot(tf_obj.v.T, np.dot(kernel, tf_obj.v)))
        f1, g1 = compute_l2_dist(
            t_mu_source, phi_source, t_mu_source, phi_source, sigma
        )
        f2, g2 = compute_l2_dist(t_mu_source, phi_source, mu_target, phi_target, sigma)
        f = -f1 + 2.0 * f2
        g = -2.0 * g1 + 2.0 * g2
        grad = self._alpha * np.dot(basis.T, g)
        grad[dim + 1 :, :] += 2.0 * self._beta * np.dot(kernel, tf_obj.v)
        return self._alpha * f + self._beta * bending, grad.flatten()


class L2DistRegistration:
    """L2 distance registration.

    Point sets are represented as Gaussian mixtures, and the L2 distance between them
    is minimized. Note that the scale parameter sigma is estimated from (or has to be
    chosen according to) the scale of the point sets.
    """

    def __init__(
        self,
        source: np.ndarray,
        feature_gen: Feature,
        cost_fn: CostFunction,
        sigma: float = 1.0,
        delta: float = 0.9,
        use_estimated_sigma: bool = True,
    ) -> None:
        self._source = source
        self._feature_gen = feature_gen
        self._cost_fn = cost_fn
        self._sigma = sigma
        self._delta = delta
        self._use_estimated_sigma = use_estimated_sigma
        self._callbacks: list[Callable[[tf.Transformation], None]] = []
        if self._use_estimated_sigma:
            self._estimate_sigma(self._source)

    def set_callbacks(self, callbacks: list[Callable[[tf.Transformation], None]]):
        self._callbacks.extend(callbacks)

    def _estimate_sigma(self, data: np.ndarray) -> None:
        ndata, dim = data.shape
        data_hat = data - np.mean(data, axis=0)
        self._sigma = np.power(
            np.linalg.det(np.dot(data_hat.T, data_hat) / (ndata - 1)), 1.0 / (2.0 * dim)
        )

    def _annealing(self) -> None:
        self._sigma *= self._delta

    def _optimization_cb(self, x: np.ndarray) -> None:
        tf_result = self._cost_fn.to_transformation(x)
        for callback in self._callbacks:
            callback(tf_result)

    def registration(
        self,
        target: np.ndarray,
        maxiter: int = 1,
        tol: float = 1.0e-3,
        opt_maxiter: int = 50,
        opt_tol: float = 1.0e-3,
    ) -> tf.Transformation:
        f = None
        x_ini = self._cost_fn.initial()
        for _ in range(maxiter):
            self._feature_gen.init()
            mu_source, phi_source = self._feature_gen.compute(self._source)
            mu_target, phi_target = self._feature_gen.compute(target)
            args = (mu_source, phi_source, mu_target, phi_target, self._sigma)
            res = minimize(
                self._cost_fn,
                x_ini,
                args=args,
                method="BFGS",
                jac=True,
                tol=opt_tol,
                options={"maxiter": opt_maxiter},
                callback=self._optimization_cb,
            )
            self._annealing()
            self._feature_gen.annealing()
            if f is not None and abs(res.fun - f) < tol:
                break
            f = res.fun
            x_ini = res.x
        return self._cost_fn.to_transformation(res.x)


class RigidGMMReg(L2DistRegistration):
    def __init__(
        self,
        source: np.ndarray,
        sigma: float = 1.0,
        delta: float = 0.9,
        n_gmm_components: int = 800,
        use_estimated_sigma: bool = True,
    ) -> None:
        n_gmm_components = min(n_gmm_components, int(source.shape[0] * 0.8))
        super().__init__(
            source,
            GMM(n_gmm_components),
            RigidCostFunction(source.shape[1]),
            sigma,
            delta,
            use_estimated_sigma,
        )


class TPSGMMReg(L2DistRegistration):
    def __init__(
        self,
        source: np.ndarray,
        sigma: float = 1.0,
        delta: float = 0.9,
        n_gmm_components: int = 800,
        alpha: float = 1.0,
        beta: float = 0.1,
        use_estimated_sigma: bool = True,
    ) -> None:
        n_gmm_components = min(n_gmm_components, int(source.shape[0] * 0.8))
        super().__init__(
            source,
            GMM(n_gmm_components),
            TPSCostFunction(np.empty((0, source.shape[1])), alpha, beta),
            sigma,
            delta,
            use_estimated_sigma,
        )
        self._feature_gen.init()
        control_pts, _ = self._feature_gen.compute(source)
        self._cost_fn._control_pts = control_pts


class RigidSVR(L2DistRegistration):
    def __init__(
        self,
        source: np.ndarray,
        sigma: float = 1.0,
        delta: float = 0.9,
        gamma: float = 0.5,
        nu: float = 0.1,
        use_estimated_sigma: bool = True,
    ) -> None:
        super().__init__(
            source,
            OneClassSVM(source.shape[1], sigma, gamma, nu),
            RigidCostFunction(source.shape[1]),
            sigma,
            delta,
            use_estimated_sigma,
        )

    def _estimate_sigma(self, data: np.ndarray) -> None:
        super()._estimate_sigma(data)
        self._feature_gen._sigma = self._sigma
        self._feature_gen._gamma = 1.0 / (2.0 * np.square(self._sigma))


class TPSSVR(L2DistRegistration):
    def __init__(
        self,
        source: np.ndarray,
        sigma: float = 1.0,
        delta: float = 0.9,
        gamma: float = 0.5,
        nu: float = 0.1,
        alpha: float = 1.0,
        beta: float = 0.1,
        use_estimated_sigma: bool = True,
    ) -> None:
        super().__init__(
            source,
            OneClassSVM(source.shape[1], sigma, gamma, nu),
            TPSCostFunction(np.empty((0, source.shape[1])), alpha, beta),
            sigma,
            delta,
            use_estimated_sigma,
        )
        self._feature_gen.init()
        control_pts, _ = self._feature_gen.compute(source)
        self._cost_fn._control_pts = control_pts

    def _estimate_sigma(self, data: np.ndarray) -> None:
        super()._estimate_sigma(data)
        self._feature_gen._sigma = self._sigma
        self._feature_gen._gamma = 1.0 / (2.0 * np.square(self._sigma))
