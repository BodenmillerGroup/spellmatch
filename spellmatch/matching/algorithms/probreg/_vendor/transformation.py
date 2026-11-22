# Adapted from probreg 0.3.8 (MIT License, Copyright (c) 2019 neka-nat), see LICENSE
from abc import ABC, abstractmethod
from types import ModuleType

import numpy as np

from . import kernels


class Transformation(ABC):
    def __init__(self, xp: ModuleType = np) -> None:
        self.xp = xp

    def transform(self, points: np.ndarray) -> np.ndarray:
        return self._transform(self.xp.asarray(points))

    @abstractmethod
    def _transform(self, points: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


class RigidTransformation(Transformation):
    def __init__(self, rot: np.ndarray, t: np.ndarray, scale: float = 1.0, xp=np):
        super().__init__(xp)
        self.rot = rot
        self.t = t
        self.scale = scale

    def _transform(self, points: np.ndarray) -> np.ndarray:
        return self.scale * self.xp.dot(points, self.rot.T) + self.t

    def inverse(self) -> "RigidTransformation":
        return RigidTransformation(
            self.rot.T,
            -self.xp.dot(self.rot.T, self.t) / self.scale,
            1.0 / self.scale,
            xp=self.xp,
        )


class AffineTransformation(Transformation):
    def __init__(self, b: np.ndarray, t: np.ndarray, xp=np):
        super().__init__(xp)
        self.b = b
        self.t = t

    def _transform(self, points: np.ndarray) -> np.ndarray:
        return self.xp.dot(points, self.b.T) + self.t


class NonRigidTransformation(Transformation):
    """Non-rigid transformation of the (source) points it was created for."""

    def __init__(self, w: np.ndarray | None, points: np.ndarray, beta: float, xp=np):
        super().__init__(xp)
        self.g = kernels.rbf_kernel(points, points, beta, xp=xp)
        self.w = w

    def _transform(self, points: np.ndarray) -> np.ndarray:
        return points + self.xp.dot(self.g, self.w)


class CombinedTransformation(Transformation):
    """Rigid transformation after non-rigid displacements of the (source) points."""

    def __init__(self, rot: np.ndarray, t: np.ndarray, scale: float, v: np.ndarray):
        super().__init__()
        self.rigid_trans = RigidTransformation(rot, t, scale)
        self.v = v

    def _transform(self, points: np.ndarray) -> np.ndarray:
        return self.rigid_trans.transform(points + self.v)


class TPSTransformation(Transformation):
    """Thin plate spline transformation."""

    def __init__(self, a: np.ndarray, v: np.ndarray, control_pts: np.ndarray):
        super().__init__()
        self.a = a
        self.v = v
        self.control_pts = control_pts

    def prepare(self, landmarks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        control_pts = self.control_pts
        m, d = landmarks.shape
        n, _ = control_pts.shape
        pm = np.c_[np.ones((m, 1)), landmarks]
        pn = np.c_[np.ones((n, 1)), control_pts]
        u, _, _ = np.linalg.svd(pn)
        pp = u[:, d + 1 :]
        kk = kernels.tps_kernel(control_pts, control_pts)
        uu = kernels.tps_kernel(landmarks, control_pts)
        basis = np.c_[pm, np.dot(uu, pp)]
        kernel = np.dot(pp.T, np.dot(kk, pp))
        return basis, kernel

    def transform_basis(self, basis: np.ndarray) -> np.ndarray:
        return np.dot(basis, np.r_[self.a, self.v])

    def _transform(self, points: np.ndarray) -> np.ndarray:
        basis, _ = self.prepare(points)
        return self.transform_basis(basis)


class NormalizedTransformation(Transformation):
    """Transformation estimated on normalized points: x -> T((x - c) / s) * s + c."""

    def __init__(
        self, transformation: Transformation, centroid: np.ndarray, scale: float
    ) -> None:
        super().__init__()
        self.transformation = transformation
        self.centroid = centroid
        self.scale = scale

    def _transform(self, points: np.ndarray) -> np.ndarray:
        normalized_points = (points - self.centroid) / self.scale
        return (
            self.transformation.transform(normalized_points) * self.scale
            + self.centroid
        )
