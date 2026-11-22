# Adapted from probreg 0.3.8 (MIT License, Copyright (c) 2019 neka-nat), see LICENSE
#
# Changes: arbitrary dimensionality (instead of _N_DIM = 3), corrected variance update
# of rigid CPD without scale estimation, xp (numpy/cupy) used consistently, inputs
# converted to xp arrays, no color information or constrained non-rigid CPD.
import logging
from abc import ABC, abstractmethod
from collections import namedtuple
from collections.abc import Callable

import numpy as np

from . import kernels
from . import transformation as tf

logger = logging.getLogger(__name__)

EstepResult = namedtuple("EstepResult", ["pt1", "p1", "px", "n_p"])
MstepResult = namedtuple("MstepResult", ["transformation", "sigma2", "q"])


class CoherentPointDrift(ABC):
    """Coherent Point Drift (Myronenko & Song, 2010).

    The expectation step is implemented here, the maximization step depends on the
    transformation model and is implemented in subclasses.
    """

    def __init__(self, source: np.ndarray, use_cuda: bool = False) -> None:
        if use_cuda:
            import cupy as cp

            self.xp = cp
        else:
            self.xp = np
        self._source = self.xp.asarray(source)
        self._callbacks: list[Callable[[tf.Transformation], None]] = []

    def set_callbacks(self, callbacks: list[Callable[[tf.Transformation], None]]):
        self._callbacks.extend(callbacks)

    @abstractmethod
    def _initialize(self, target: np.ndarray) -> MstepResult:
        raise NotImplementedError()

    def expectation_step(
        self, t_source: np.ndarray, target: np.ndarray, sigma2: float, w: float = 0.0
    ) -> EstepResult:
        xp = self.xp
        dim = t_source.shape[1]
        pmat = kernels.squared_distances(t_source, target, xp=xp)
        pmat = xp.exp(-pmat / (2.0 * sigma2))
        c = (2.0 * np.pi * sigma2) ** (dim * 0.5)
        c *= w / (1.0 - w) * t_source.shape[0] / target.shape[0]
        den = xp.sum(pmat, axis=0)
        den[den == 0] = np.finfo(np.float32).eps
        den += c
        pmat = xp.divide(pmat, den)
        pt1 = xp.sum(pmat, axis=0)
        p1 = xp.sum(pmat, axis=1)
        px = xp.dot(pmat, target)
        return EstepResult(pt1, p1, px, float(xp.sum(p1)))

    @abstractmethod
    def maximization_step(
        self, target: np.ndarray, estep_res: EstepResult, sigma2_p: float
    ) -> MstepResult:
        raise NotImplementedError()

    def registration(
        self, target: np.ndarray, w: float = 0.0, maxiter: int = 50, tol: float = 0.001
    ) -> MstepResult:
        target = self.xp.asarray(target)
        res = self._initialize(target)
        q = res.q
        for i in range(maxiter):
            t_source = res.transformation.transform(self._source)
            estep_res = self.expectation_step(t_source, target, res.sigma2, w)
            res = self.maximization_step(target, estep_res, res.sigma2)
            for callback in self._callbacks:
                callback(res.transformation)
            logger.debug(f"Iteration: {i}, Criteria: {res.q}")
            if abs(res.q - q) < tol:
                break
            q = res.q
        return res


class RigidCPD(CoherentPointDrift):
    def __init__(
        self, source: np.ndarray, update_scale: bool = True, use_cuda: bool = False
    ) -> None:
        super().__init__(source, use_cuda=use_cuda)
        self._update_scale = update_scale

    def _initialize(self, target: np.ndarray) -> MstepResult:
        dim = self._source.shape[1]
        sigma2 = kernels.squared_kernel_sum(self._source, target, xp=self.xp)
        q = 1.0 + target.shape[0] * dim * 0.5 * np.log(sigma2)
        transformation = tf.RigidTransformation(
            self.xp.identity(dim), self.xp.zeros(dim), xp=self.xp
        )
        return MstepResult(transformation, sigma2, q)

    def maximization_step(
        self, target: np.ndarray, estep_res: EstepResult, sigma2_p: float
    ) -> MstepResult:
        xp = self.xp
        source = self._source
        pt1, p1, px, n_p = estep_res
        dim = source.shape[1]
        mu_x = xp.sum(px, axis=0) / n_p
        mu_y = xp.dot(source.T, p1) / n_p
        target_hat = target - mu_x
        source_hat = source - mu_y
        a = xp.dot(px.T, source_hat) - xp.outer(mu_x, xp.dot(p1.T, source_hat))
        u, _, vh = xp.linalg.svd(a, full_matrices=True)
        c = xp.ones(dim)
        c[-1] = xp.linalg.det(xp.dot(u, vh))
        rot = xp.dot(u * c, vh)
        tr_atr = float(xp.trace(xp.dot(a.T, rot)))
        tr_yp1y = float(xp.trace(xp.dot(source_hat.T * p1, source_hat)))
        scale = tr_atr / tr_yp1y if self._update_scale else 1.0
        t = mu_x - scale * xp.dot(rot, mu_y)
        tr_xp1x = float(xp.trace(xp.dot(target_hat.T * pt1, target_hat)))
        # Myronenko & Song (2010), Fig. 2: sigma2 = (tr_xp1x - s tr_atr) / (N_P D)
        # with the optimal scale s = tr_atr / tr_yp1y; for a fixed scale, the general
        # form (tr_xp1x - 2 s tr_atr + s^2 tr_yp1y) / (N_P D) applies (probreg used
        # tr_xp1x + tr_yp1y - s tr_atr, which overestimates sigma2 by tr_atr / (N_P D))
        sigma2 = (tr_xp1x - 2.0 * scale * tr_atr + scale**2 * tr_yp1y) / (n_p * dim)
        sigma2 = max(sigma2, np.finfo(np.float32).eps)
        q = (tr_xp1x - 2.0 * scale * tr_atr + (scale**2) * tr_yp1y) / (2.0 * sigma2)
        q += dim * n_p * 0.5 * np.log(sigma2)
        return MstepResult(tf.RigidTransformation(rot, t, scale, xp=xp), sigma2, q)


class AffineCPD(CoherentPointDrift):
    def _initialize(self, target: np.ndarray) -> MstepResult:
        dim = self._source.shape[1]
        sigma2 = kernels.squared_kernel_sum(self._source, target, xp=self.xp)
        q = 1.0 + target.shape[0] * dim * 0.5 * np.log(sigma2)
        transformation = tf.AffineTransformation(
            self.xp.identity(dim), self.xp.zeros(dim), xp=self.xp
        )
        return MstepResult(transformation, sigma2, q)

    def maximization_step(
        self, target: np.ndarray, estep_res: EstepResult, sigma2_p: float
    ) -> MstepResult:
        xp = self.xp
        source = self._source
        pt1, p1, px, n_p = estep_res
        dim = source.shape[1]
        mu_x = xp.sum(px, axis=0) / n_p
        mu_y = xp.dot(source.T, p1) / n_p
        target_hat = target - mu_x
        source_hat = source - mu_y
        a = xp.dot(px.T, source_hat) - xp.outer(mu_x, xp.dot(p1.T, source_hat))
        yp1y = xp.dot(source_hat.T * p1, source_hat)
        b = xp.linalg.solve(yp1y.T, a.T).T
        t = mu_x - xp.dot(b, mu_y)
        tr_xp1x = float(xp.trace(xp.dot(target_hat.T * pt1, target_hat)))
        tr_xpyb = float(xp.trace(xp.dot(a, b.T)))
        sigma2 = (tr_xp1x - tr_xpyb) / (n_p * dim)
        sigma2 = max(sigma2, np.finfo(np.float32).eps)
        q = (tr_xp1x - tr_xpyb) / (2.0 * sigma2)
        q += dim * n_p * 0.5 * np.log(sigma2)
        return MstepResult(tf.AffineTransformation(b, t, xp=xp), sigma2, q)


class NonRigidCPD(CoherentPointDrift):
    def __init__(
        self,
        source: np.ndarray,
        beta: float = 2.0,
        lmd: float = 2.0,
        use_cuda: bool = False,
    ) -> None:
        super().__init__(source, use_cuda=use_cuda)
        self._lmd = lmd
        self._tf_obj = tf.NonRigidTransformation(None, self._source, beta, xp=self.xp)

    def _initialize(self, target: np.ndarray) -> MstepResult:
        dim = self._source.shape[1]
        sigma2 = kernels.squared_kernel_sum(self._source, target, xp=self.xp)
        q = 1.0 + target.shape[0] * dim * 0.5 * np.log(sigma2)
        self._tf_obj.w = self.xp.zeros_like(self._source)
        return MstepResult(self._tf_obj, sigma2, q)

    def maximization_step(
        self, target: np.ndarray, estep_res: EstepResult, sigma2_p: float
    ) -> MstepResult:
        xp = self.xp
        source = self._source
        tf_obj = self._tf_obj
        pt1, p1, px, n_p = estep_res
        dim = source.shape[1]
        w = xp.linalg.solve(
            (p1 * tf_obj.g).T + self._lmd * sigma2_p * xp.identity(source.shape[0]),
            px - (source.T * p1).T,
        )
        t = source + xp.dot(tf_obj.g, w)
        tr_xp1x = float(xp.trace(xp.dot(target.T * pt1, target)))
        tr_pxt = float(xp.trace(xp.dot(px.T, t)))
        tr_tpt = float(xp.trace(xp.dot(t.T * p1, t)))
        sigma2 = (tr_xp1x - 2.0 * tr_pxt + tr_tpt) / (n_p * dim)
        sigma2 = max(sigma2, np.finfo(np.float32).eps)
        tf_obj.w = w
        return MstepResult(tf_obj, sigma2, sigma2)
