# Adapted from probreg 0.3.8 (MIT License, Copyright (c) 2019 neka-nat), see LICENSE
#
# Revised following Hirose, "A Bayesian Formulation of Coherent Point Drift", IEEE
# TPAMI 43(7), 2021 (Fig. 2). Changes compared to probreg:
# - Covariance update Sigma = (lambda G^-1 + s^2 / sigma^2 diag(nu))^-1 (probreg used
#   s^2 / sigma^4), computed as G (lambda I + s^2 / sigma^2 diag(nu) G)^-1
# - Scale s = tr(R^T S_xu) / tr(S_uu) (probreg used tr(R S_xu))
# - Cell pairs with vanishing matching probabilities (nu_m -> 0, e.g. for partially
#   overlapping point sets) no longer produce NaN (x_hat is only used weighted by nu)
# - Gaussian kernel with width beta (as in Hirose, 2021; probreg used an inverse
#   multiquadric kernel without width parameter)
# - Outlier probability relative to the volume of the target point set (Hirose, 2021)
# - Convergence on the relative change of sigma^2
import logging
from collections.abc import Callable

import numpy as np
import scipy.special as spsp

from . import kernels
from . import transformation as tf

logger = logging.getLogger(__name__)


class CombinedBCPD:
    """Bayesian Coherent Point Drift (rigid and non-rigid transformation).

    Note that the point sets should be normalized (e.g. zero mean, unit variance), as
    the parameters (beta, lambda) are defined relative to the scale of the point sets.
    """

    def __init__(
        self,
        source: np.ndarray,
        lmd: float = 2.0,
        beta: float = 2.0,
        k: float = 1.0e20,
        gamma: float = 1.0,
    ) -> None:
        self._source = np.asarray(source, dtype=np.float64)
        self.lmd = lmd
        self.beta = beta
        self.k = k
        self.gamma = gamma
        self._callbacks: list[Callable[[tf.Transformation], None]] = []

    def set_callbacks(self, callbacks: list[Callable[[tf.Transformation], None]]):
        self._callbacks.extend(callbacks)

    def registration(
        self,
        target: np.ndarray,
        w: float = 0.0,
        maxiter: int = 50,
        tol: float = 0.001,
    ) -> tf.CombinedTransformation:
        y = self._source
        x = np.asarray(target, dtype=np.float64)
        m, dim = y.shape
        eps = np.finfo(np.float64).eps
        gmat = kernels.gaussian_kernel(y, y, self.beta)
        v = np.zeros_like(y)
        sigma_mat = np.identity(m)
        alpha = np.full(m, 1.0 / m)
        rot, t, scale = np.identity(dim), np.zeros(dim), 1.0
        sigma2 = self.gamma * kernels.squared_kernel_sum(y, x)
        volume = float(np.prod(np.ptp(x, axis=0)))
        outlier_density = w / volume if w > 0 and volume > 0 else 0.0
        transformation = tf.CombinedTransformation(rot, t, scale, v)
        for i in range(maxiter):
            # expectation step
            y_hat = transformation.transform(y)
            log_phi = -kernels.squared_distances(y_hat, x) / (2.0 * sigma2)
            log_phi -= scale**2 * dim * np.diag(sigma_mat)[:, None] / (2.0 * sigma2)
            phi = np.exp(log_phi) / (2.0 * np.pi * sigma2) ** (dim * 0.5)
            pmat = (1.0 - w) * alpha[:, None] * phi
            den = outlier_density + np.sum(pmat, axis=0)
            den[den == 0] = eps
            pmat /= den
            nu = np.sum(pmat, axis=1)
            nu_d = np.sum(pmat, axis=0)
            n_p = float(np.sum(nu))
            if n_p <= eps:
                logger.warning("BCPD: no point matched, stopping")
                break
            px = np.dot(pmat, x)
            x_hat = px / np.maximum(nu, eps)[:, None]
            # maximization step
            s2s2 = scale**2 / sigma2
            sigma_mat = np.linalg.solve(
                (self.lmd * np.identity(m) + s2s2 * nu[:, None] * gmat).T, gmat.T
            ).T
            residual = (x_hat - t) @ rot / scale - y
            v = s2s2 * sigma_mat @ (nu[:, None] * residual)
            u = y + v
            alpha = np.exp(spsp.psi(self.k + nu) - spsp.psi(self.k * m + n_p))
            x_m = np.sum(px, axis=0) / n_p
            sigma2_m = float(np.sum(nu * np.diag(sigma_mat))) / n_p
            u_m = np.sum(nu[:, None] * u, axis=0) / n_p
            u_hm = u - u_m
            s_xu = (px - nu[:, None] * x_m).T @ u_hm / n_p
            s_uu = (nu[:, None] * u_hm).T @ u_hm / n_p + sigma2_m * np.identity(dim)
            phi_mat, _, psi_h = np.linalg.svd(s_xu)
            c = np.ones(dim)
            c[-1] = np.linalg.det(phi_mat @ psi_h)
            rot = (phi_mat * c) @ psi_h
            scale = float(np.trace(rot.T @ s_xu) / np.trace(s_uu))
            t = x_m - scale * rot @ u_m
            transformation = tf.CombinedTransformation(rot, t, scale, v)
            y_hat = transformation.transform(y)
            last_sigma2 = sigma2
            sigma2 = (
                float(np.sum(nu_d * np.sum(x * x, axis=1)))
                - 2.0 * float(np.sum(px * y_hat))
                + float(np.sum(nu * np.sum(y_hat * y_hat, axis=1)))
            ) / (n_p * dim) + scale**2 * sigma2_m
            sigma2 = max(sigma2, eps)
            for callback in self._callbacks:
                callback(transformation)
            logger.debug(f"Iteration: {i}, sigma2: {sigma2}")
            if abs(sigma2 - last_sigma2) / last_sigma2 < tol:
                break
        return transformation
