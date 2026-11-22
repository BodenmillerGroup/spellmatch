# Adapted from probreg 0.3.8 (MIT License, Copyright (c) 2019 neka-nat), see LICENSE
# (math_utils.py, cc/math_utils.cc, gauss_transform.py, se3_op.py, cupy_utils.py)
from types import ModuleType

import numpy as np
from scipy.spatial import distance


def squared_distances(x: np.ndarray, y: np.ndarray, xp: ModuleType = np) -> np.ndarray:
    if xp is np:
        return distance.cdist(x, y, "sqeuclidean")
    return xp.sum(xp.square(x[:, None, :] - y[None, :, :]), axis=2)


def squared_kernel_sum(x: np.ndarray, y: np.ndarray, xp: ModuleType = np) -> float:
    return float(squared_distances(x, y, xp=xp).sum()) / (
        x.shape[0] * x.shape[1] * y.shape[0]
    )


def rbf_kernel(
    x: np.ndarray, y: np.ndarray, beta: float, xp: ModuleType = np
) -> np.ndarray:
    return xp.exp(-squared_distances(x, y, xp=xp) / (2.0 * beta))


def gaussian_kernel(x: np.ndarray, y: np.ndarray, beta: float) -> np.ndarray:
    return np.exp(-squared_distances(x, y) / (2.0 * beta**2))


def tps_kernel(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    assert x.shape[1] == y.shape[1], "x and y must have same dimensions."
    diff2 = squared_distances(x, y)
    if x.shape[1] == 2:
        k = np.zeros_like(diff2)
        mask = diff2 > 1.0e-9
        k[mask] = diff2[mask] * np.log(np.sqrt(diff2[mask]))
        return k
    if x.shape[1] == 3:
        return -np.sqrt(diff2)
    raise ValueError(f"Invalid dimension of x: {x.shape[1]}")


def gauss_transform(
    source: np.ndarray, target: np.ndarray, weights: np.ndarray, h: float
) -> np.ndarray:
    """Exact Gauss transform.

    sum_j weights[..., j] * exp(-||target[i] - source[j]||^2 / h^2)
    """
    k = np.exp(-squared_distances(target, source) / (h * h))
    return np.dot(weights, k.T)


def _quat_k(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """K(q) with R(q) = I + 2 / |q|^2 K(q), and dK/dq (4, 3, 3)."""
    w, x, y, z = q
    k = np.array(
        [
            [-(y * y + z * z), x * y - w * z, x * z + w * y],
            [x * y + w * z, -(x * x + z * z), y * z - w * x],
            [x * z - w * y, y * z + w * x, -(x * x + y * y)],
        ]
    )
    d_k = np.array(
        [
            [[0, -z, y], [z, 0, -x], [-y, x, 0]],  # dK/dw
            [[0, y, z], [y, -2 * x, -w], [z, w, -2 * x]],  # dK/dx
            [[-2 * y, x, w], [x, 0, z], [-w, z, -2 * y]],  # dK/dy
            [[-2 * z, -w, x], [w, -2 * z, y], [x, y, 0]],  # dK/dz
        ],
        dtype=float,
    )
    return k, d_k


def quat2mat(q: np.ndarray) -> np.ndarray:
    """Rotation matrix of a (not necessarily normalized) quaternion (w, x, y, z)."""
    nq = float(np.sum(np.square(q)))
    if nq < np.finfo(np.float64).eps:
        return np.identity(3)
    k, _ = _quat_k(q)
    return np.identity(3) + 2.0 / nq * k


def diff_rot_from_quaternion(q: np.ndarray) -> np.ndarray:
    """Derivative of quat2mat with respect to the quaternion, (4, 3, 3).

    probreg's implementation was inconsistent with the normalization of quat2mat.
    """
    nq = float(np.sum(np.square(q)))
    k, d_k = _quat_k(q)
    return 2.0 / nq * d_k - (4.0 / nq**2) * q[:, None, None] * k[None, :, :]
