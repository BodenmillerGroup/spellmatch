"""Matrix-free replacements for the Kronecker-product-scale computations in
:class:`spellmatch.matching.algorithms.spellmatch.Spellmatch`.

The fixed-point iteration only ever needs ``W @ s`` for a changing vector
``s``, never the ``(n1*n2) x (n1*n2)`` matrix ``W`` itself, so every term is
rewritten as a sequence of ordinary ``n1 x n1`` / ``n2 x n2`` operations
instead of ever materializing ``A1 (x) A2``. Each identity used here is
checked against a dense ground truth in ``tests/matching/test_kron_free.py``.

Notation: ``a1``/``a2`` are the source/target adjacency matrices, ``d1``/``d2``
the same graphs with edge lengths as entries (``d1 = a1 ⊙ dists1``), and
``A = a1 kron a2`` the product graph over candidate pairs ``v = a*n2 + x``
(source cell ``a``, target cell ``x``).

Precision note: the distance term's expansion ``d1^2 - 2*d1*d2 + d2^2``
suffers measurable catastrophic cancellation in float32, so
``distance_term_matvec`` always computes in float64. ``kron_matvec`` and
``outer_sum_matvec`` compute in the dtype of their inputs; the caller
(``Spellmatch._match_graphs_for_lambda``) passes float64 and casts the result
to its target precision.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import sparse

logger = logging.getLogger(__name__)


def kron_matvec(
    x: np.ndarray, y: np.ndarray, v: np.ndarray, n1: int, n2: int
) -> np.ndarray:
    """Compute ``(x kron y) @ v`` without forming ``x kron y``.

    ``x`` is ``(n1, n1)``, ``y`` is ``(n2, n2)`` (dense or sparse), and ``v``
    has length ``n1*n2``. ``V`` is ``v`` reshaped row-major to ``(n1, n2)``,
    i.e. ``v[i*n2 + j] = V[i, j]``.

    Identity: ``(x kron y) @ v == vec(x @ V @ y.T)``.
    """
    v_mat = np.asarray(v).reshape(n1, n2)
    if sparse.issparse(x):
        out = x @ v_mat
    else:
        out = x.dot(v_mat)
    if sparse.issparse(y):
        out = out @ y.T
    else:
        out = out.dot(y.T)
    return np.asarray(out).reshape(n1 * n2)


def outer_sum_matvec(
    a1: np.ndarray,
    a2: np.ndarray,
    u: np.ndarray,
    s: np.ndarray,
    n1: int,
    n2: int,
    a_s: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute ``[A ⊙ (u 1^T + 1 u^T)] @ s`` where ``A = a1 kron a2``.

    Equivalent to the broadcast ``A * u[:, None] + A * u[None, :]`` on the
    explicit matrix, but in matrix-free form.

    ``a_s`` is ``A @ s``; pass it if the caller already has it (``w_matvec``
    always does) to skip recomputing one full ``kron_matvec`` per call.
    """
    if a_s is None:
        a_s = kron_matvec(a1, a2, s, n1, n2)
    return u * a_s + kron_matvec(a1, a2, u * s, n1, n2)


# Default number of correction entries generated per block in
# build_distance_correction. Block temporaries cost roughly 45-60 bytes per
# entry (measured), so 1e7 keeps them below 1 GB.
DEFAULT_BLOCK_NNZ = 10_000_000
_INT32_LIMIT = 2**31


def correction_index_dtype(n: int, nnz: int) -> type:
    """Index dtype of the correction matrix: int32 unless an index (up to
    ``n - 1``, with ``n = n1*n2``) or an ``indptr`` value (up to ``nnz``)
    would overflow it. scipy requires ``indices`` and ``indptr`` to share one
    dtype, so both limits apply to both arrays."""
    return np.int32 if n < _INT32_LIMIT and nnz < _INT32_LIMIT else np.int64


def _clipped(d1_val: np.ndarray, d2_val: np.ndarray, thresh: np.float64) -> np.ndarray:
    """The clip predicate ``y > 1``, computed with exactly the same arithmetic
    as the stored value ``y - 1`` so that counting and building agree."""
    return ((d1_val - d2_val) / thresh) ** 2 > 1.0


def _correction_layout(d1: sparse.spmatrix, d2: sparse.spmatrix, thresh: float):
    """Setup for building the correction matrix.

    Returns the source edges (``d1``, upper triangle ``a < b`` only) sorted by
    source cell ``a``, all directed target edges (``d2``, both directions)
    sorted by length, and per source edge the bounds ``lo``/``hi`` such that
    its clipped target edges are exactly the sorted positions ``[0, lo)`` and
    ``[hi, E2)``, where ``E2`` is the number of directed target edges.

    For a source edge of length ``d``, the predicate ``((d - e)/thresh)**2 > 1``
    is monotone in the sorted target edge length ``e`` on either side of ``d``
    (every floating-point step is monotone), so the clipped target edges
    always form a prefix and a suffix of the sorted order. ``searchsorted``
    finds the bounds up to rounding; the bounds are then moved one position
    at a time until the exact predicate holds on the inside and fails on the
    outside.
    """
    thresh = np.float64(thresh)

    d1_coo = sparse.coo_array(d1)
    upper = d1_coo.row < d1_coo.col
    e1_a = d1_coo.row[upper]
    e1_b = d1_coo.col[upper]
    e1_val = d1_coo.data[upper].astype(np.float64)
    order = np.argsort(e1_a, kind="stable")
    e1_a, e1_b, e1_val = e1_a[order], e1_b[order], e1_val[order]

    d2_coo = sparse.coo_array(d2)
    order = np.argsort(d2_coo.data, kind="stable")
    e2_x = d2_coo.row[order]
    e2_y = d2_coo.col[order]
    e2_val = d2_coo.data[order].astype(np.float64)
    n_e2 = len(e2_val)

    if len(e1_val) == 0 or n_e2 == 0:
        empty = np.zeros(len(e1_val), dtype=np.int64)
        return e1_a, e1_b, e1_val, e2_x, e2_y, e2_val, empty, empty + n_e2

    # Prefix [0, lo): clipped partners shorter than d.
    lo = np.searchsorted(e2_val, e1_val - thresh, side="left").astype(np.int64)
    while True:
        i = np.nonzero(lo > 0)[0]
        i = i[~_clipped(e1_val[i], e2_val[lo[i] - 1], thresh)]
        if len(i) == 0:
            break
        lo[i] -= 1
    while True:
        i = np.nonzero(lo < n_e2)[0]
        nxt = e2_val[lo[i]]
        i = i[(nxt < e1_val[i]) & _clipped(e1_val[i], nxt, thresh)]
        if len(i) == 0:
            break
        lo[i] += 1

    # Suffix [hi, E2): clipped partners longer than d.
    hi = np.searchsorted(e2_val, e1_val + thresh, side="right").astype(np.int64)
    while True:
        i = np.nonzero(hi < n_e2)[0]
        i = i[~_clipped(e1_val[i], e2_val[hi[i]], thresh)]
        if len(i) == 0:
            break
        hi[i] += 1
    while True:
        i = np.nonzero(hi > 0)[0]
        prv = e2_val[hi[i] - 1]
        i = i[(prv > e1_val[i]) & _clipped(e1_val[i], prv, thresh)]
        if len(i) == 0:
            break
        hi[i] -= 1

    return e1_a, e1_b, e1_val, e2_x, e2_y, e2_val, lo, hi


def build_distance_correction(
    d1: sparse.spmatrix,
    d2: sparse.spmatrix,
    thresh: float,
    n1: int,
    n2: int,
    block_nnz: int = DEFAULT_BLOCK_NNZ,
) -> Optional[sparse.csr_array]:
    """Build the sparse correction matrix ``relu(y-1)`` for the distance term,
    or return ``None`` if no edge pair is clipped (no correction needed).

    ``d1 = a1 ⊙ dists1``, ``d2 = a2 ⊙ dists2`` (edge-masked distances, i.e.
    zero everywhere the corresponding adjacency is zero). For a source edge
    ``(a, b)`` and a target edge ``(x, x')``, the entry at row
    ``v = a*n2 + x`` and column ``w = b*n2 + x'`` is ``y - 1`` with
    ``y = ((d1[a, b] - d2[x, x'])/thresh)**2``. Only entries where the clip
    actually fires (``y > 1``) are stored, and only the upper triangle
    (``v < w``): the full matrix is exactly symmetric with a structurally zero
    diagonal, so this loses no information and ``correction_matvec``
    reconstructs the full effect via ``U@s + U.T@s``.

    Iterating the *undirected* source edges once each (``a < b``) against
    *all directed* target edges visits every unordered pair ``{v, w}`` of
    product-graph nodes exactly once. Since ``a < b``,
    ``v = a*n2 + x < b*n2 <= w``, so every entry already lies in the upper
    triangle.

    Memory: the exact nnz is counted first (from ``_correction_layout``'s
    ``lo``/``hi``, without any edge-pair-scale allocation) and the CSR arrays
    are allocated once at their final size, with int32 indices whenever they
    fit (``correction_index_dtype``). Rows ``v = a*n2 + x`` of
    consecutive source cells ``a`` form a contiguous row range, so entries are
    generated in blocks of source cells (at most ``block_nnz`` entries each,
    unless a single cell has more), converted to CSR locally and copied into
    place. Peak memory is therefore the final matrix (12 B/nnz with int32)
    plus one block's temporaries -- not a global COO staging copy.
    """
    e1_a, e1_b, e1_val, e2_x, e2_y, e2_val, lo, hi = _correction_layout(d1, d2, thresh)
    thresh = np.float64(thresh)
    n = n1 * n2
    counts = lo + (len(e2_val) - hi)
    nnz = int(counts.sum())
    if nnz == 0:
        return None

    idx_dtype = correction_index_dtype(n, nnz)
    if idx_dtype is np.int64:
        logger.warning(
            "Distance-correction matrix needs int64 indices (n1*n2=%s, nnz=%s); "
            "it will use 16 instead of 12 bytes per entry",
            f"{n:,}",
            f"{nnz:,}",
        )

    indptr = np.empty(n + 1, dtype=idx_dtype)
    indices = np.empty(nnz, dtype=idx_dtype)
    data = np.empty(nnz, dtype=np.float64)
    indptr[0] = 0

    # Blocks of whole source cells: edges of cell a are contiguous (sorted by
    # a). Greedily close a block once it would exceed block_nnz entries.
    cell_starts = np.flatnonzero(np.r_[True, e1_a[1:] != e1_a[:-1]])
    cell_ends = np.r_[cell_starts[1:], len(e1_a)]
    cell_cum = np.cumsum(np.add.reduceat(counts, cell_starts))

    offset = 0
    filled_rows = 0  # indptr[1 : filled_rows + 1] is final
    first_cell = 0
    while first_cell < len(cell_starts):
        before = cell_cum[first_cell - 1] if first_cell > 0 else 0
        last_cell = int(np.searchsorted(cell_cum, before + block_nnz, side="right"))
        last_cell = max(last_cell, first_cell + 1)  # at least one cell per block
        e0, e1 = cell_starts[first_cell], cell_ends[last_cell - 1]
        a0, a1 = int(e1_a[e0]), int(e1_a[e1 - 1]) + 1
        first_cell = last_cell

        block_counts = counts[e0:e1]
        block_total = int(block_counts.sum())
        if block_total == 0:
            continue

        # Expand each edge's ranges [0, lo) and [hi, E2) into D2 positions.
        edge = np.repeat(np.arange(e0, e1), block_counts)
        pos = np.arange(block_total, dtype=np.int64)
        pos -= np.repeat(np.cumsum(block_counts) - block_counts, block_counts)
        edge_lo = lo[edge]
        j = pos + (hi[edge] - edge_lo) * (pos >= edge_lo)
        del pos, edge_lo

        # Same arithmetic as the predicate in _clipped: bit-identical values.
        y = ((e1_val[edge] - e2_val[j]) / thresh) ** 2
        vals = y - 1.0
        del y
        rows = (e1_a[edge].astype(np.int64) - a0) * n2 + e2_x[j]
        cols = e1_b[edge].astype(np.int64) * n2 + e2_y[j]
        del edge, j

        block = sparse.coo_array(
            (vals, (rows, cols)), shape=((a1 - a0) * n2, n)
        ).tocsr()
        del vals, rows, cols

        # Rows of cells without upper edges (between blocks) are empty.
        indptr[filled_rows + 1 : a0 * n2 + 1] = offset
        indptr[a0 * n2 + 1 : a1 * n2 + 1] = block.indptr[1:] + offset
        indices[offset : offset + block.nnz] = block.indices
        data[offset : offset + block.nnz] = block.data
        offset += block.nnz
        filled_rows = a1 * n2
        del block

    indptr[filled_rows + 1 :] = offset
    assert offset == nnz, f"correction count mismatch: built {offset}, counted {nnz}"
    return sparse.csr_array((data, indices, indptr), shape=(n, n))


def correction_matvec(correction: sparse.csr_array, s: np.ndarray) -> np.ndarray:
    """Apply an upper-triangular-only symmetric correction matrix to ``s``.

    ``correction`` stores only ``v < w``; the true (symmetric, zero-diagonal)
    matrix's action is ``U@s + U.T@s``.
    """
    return correction @ s + correction.T @ s


def _as_float64(m):
    """``m`` as float64, keeping it sparse if it is sparse."""
    if sparse.issparse(m):
        return m.astype(np.float64)
    return np.asarray(m, dtype=np.float64)


@dataclass(frozen=True)
class DistanceTerms:
    """Precomputed ingredients of the distance term, built once per FOV pair.

    ``d1 = adj1 ⊙ dists1`` and ``d2 = adj2 ⊙ dists2`` are the edge-masked
    distance matrices (float64), ``thresh`` is the ``distance_cdiff_thres``
    the correction was built with, and ``correction`` is the upper-triangular
    ``relu(y-1)`` matrix from ``build_distance_correction`` -- or ``None`` if
    no edge pair is clipped, in which case no correction is applied.
    """

    d1: sparse.csr_array
    d2: sparse.csr_array
    thresh: float
    correction: Optional[sparse.csr_array]


def distance_term_matvec(
    a1: np.ndarray,
    a2: np.ndarray,
    terms: DistanceTerms,
    s: np.ndarray,
    n1: int,
    n2: int,
) -> np.ndarray:
    """Compute the distance term ``(A ⊙ clip(y, 0, 1)) @ s`` without
    materializing any ``(n1*n2) x (n1*n2)`` object, where ``y`` is the squared
    scaled edge-length difference defined in ``build_distance_correction``.

    The unclipped ``(A ⊙ y) @ s`` is computed exactly via the three-term
    Kronecker expansion ``(d1² kron a2 - 2 d1 kron d2 + a1 kron d2²) @ s /
    thresh²``, in float64 (required, see the module docstring). The clip is
    applied as ``y - relu(y-1)``, with ``relu(y-1)`` supplied pre-built as
    ``terms.correction`` (``None``: nothing is clipped).
    """
    d1, d2, thresh = terms.d1, terms.d2, terms.thresh
    d1_f64, d2_f64 = _as_float64(d1), _as_float64(d2)
    a1_f64, a2_f64 = _as_float64(a1), _as_float64(a2)
    s_f64 = np.asarray(s, dtype=np.float64)

    d1_sq = d1_f64.multiply(d1_f64) if sparse.issparse(d1_f64) else d1_f64 * d1_f64
    d2_sq = d2_f64.multiply(d2_f64) if sparse.issparse(d2_f64) else d2_f64 * d2_f64

    thresh2 = np.float64(thresh) ** 2
    algebraic = (
        kron_matvec(d1_sq, a2_f64, s_f64, n1, n2)
        - 2.0 * kron_matvec(d1_f64, d2_f64, s_f64, n1, n2)
        + kron_matvec(a1_f64, d2_sq, s_f64, n1, n2)
    ) / thresh2

    if terms.correction is None:
        # Nothing clipped (e.g. distance_cdiff_thres >= longest edge).
        return algebraic
    corr = correction_matvec(terms.correction, s_f64)
    return algebraic - corr
