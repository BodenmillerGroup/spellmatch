"""Numerical correctness tests for spellmatch.matching._kron_free.

Every identity the matrix-free reformulation of Spellmatch's graph terms
relies on is checked against a dense, brute-force ground truth, not just
asserted.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from spellmatch.matching._kron_free import (
    DistanceTerms,
    build_distance_correction,
    distance_term_matvec,
    kron_matvec,
    outer_sum_matvec,
)


def _make_symmetric_adjacency(rng, n, p_edge):
    a = (rng.random((n, n)) < p_edge).astype(float)
    a = np.triu(a, 1)
    return a + a.T


@pytest.fixture
def small_graphs():
    rng = np.random.default_rng(0)
    n1, n2 = 40, 55
    a1 = _make_symmetric_adjacency(rng, n1, 0.15)
    a2 = _make_symmetric_adjacency(rng, n2, 0.15)
    dists1 = np.triu(rng.uniform(1, 15, (n1, n1)), 1)
    dists1 = dists1 + dists1.T
    dists2 = np.triu(rng.uniform(1, 15, (n2, n2)), 1)
    dists2 = dists2 + dists2.T
    d1 = a1 * dists1
    d2 = a2 * dists2
    return rng, n1, n2, a1, a2, d1, d2


def test_kron_matvec_matches_dense_kronecker_product(small_graphs):
    rng, n1, n2, a1, a2, _, _ = small_graphs
    s = rng.random(n1 * n2)
    expected = sparse.kron(a1, a2).toarray() @ s
    actual = kron_matvec(a1, a2, s, n1, n2)
    assert np.allclose(expected, actual)


def test_kron_matvec_handles_rectangular_smaller_or_larger_second_graph():
    # n1 != n2 in both directions -- guards against an accidental square-only
    # reshape/transpose bug.
    rng = np.random.default_rng(1)
    for n1, n2 in [(5, 30), (30, 5)]:
        a1 = _make_symmetric_adjacency(rng, n1, 0.3)
        a2 = _make_symmetric_adjacency(rng, n2, 0.3)
        s = rng.random(n1 * n2)
        expected = sparse.kron(a1, a2).toarray() @ s
        actual = kron_matvec(a1, a2, s, n1, n2)
        assert np.allclose(expected, actual)


def test_outer_sum_matvec_matches_dense_broadcast(small_graphs):
    rng, n1, n2, a1, a2, _, _ = small_graphs
    s = rng.random(n1 * n2)
    u = rng.random(n1 * n2)
    adj = sparse.kron(a1, a2).toarray()
    # The degree/intensity contribution to w_csr on the explicit matrix:
    # adj * u[:,None] + adj * u[None,:].
    expected = (adj * (u[:, None] + u[None, :])) @ s
    actual = outer_sum_matvec(a1, a2, u, s, n1, n2)
    assert np.allclose(expected, actual)


def test_distance_term_matvec_matches_exact_clipped_ground_truth(small_graphs):
    rng, n1, n2, a1, a2, d1, d2 = small_graphs
    s = rng.random(n1 * n2)
    thresh = 3.0  # small threshold to exercise the clip / correction path

    diff = sparse.kron(d1, a2).toarray() - sparse.kron(a1, d2).toarray()
    y_true = np.clip((diff / thresh) ** 2, 0, 1)
    expected = y_true @ s

    correction = build_distance_correction(
        sparse.csr_matrix(d1), sparse.csr_matrix(d2), thresh, n1, n2
    )
    terms = DistanceTerms(d1, d2, thresh, correction)
    actual = distance_term_matvec(a1, a2, terms, s, n1, n2)
    assert np.allclose(expected, actual, atol=1e-9)


def test_correction_matrix_avoids_double_counting_and_matches_upper_triangle(
    small_graphs,
):
    # This is the specific bug class caught during development: iterating
    # both directions of D1's edges against both directions of D2's edges
    # revisits every unordered candidate pair twice (once via (a,b)-(x,y),
    # once via its mirror (b,a)-(y,x)), which would double the stored
    # correction value if not handled. Verify nnz and values match a dense
    # brute-force upper-triangle computation exactly.
    rng, n1, n2, a1, a2, d1, d2 = small_graphs
    thresh = 3.0
    diff = sparse.kron(d1, a2).toarray() - sparse.kron(a1, d2).toarray()
    full_correction = np.maximum((diff / thresh) ** 2 - 1, 0)
    expected_upper = np.triu(full_correction, 1)

    correction = build_distance_correction(
        sparse.csr_matrix(d1), sparse.csr_matrix(d2), thresh, n1, n2
    )
    assert correction.nnz == int((expected_upper > 0).sum())
    dense_actual = correction.toarray()
    assert np.allclose(dense_actual, expected_upper)


def test_distance_term_matvec_combined_with_degree_term(small_graphs):
    # The matrix-free terms must remain exact when several weighted terms are
    # summed together; the clip is applied once, inside the distance term.
    rng, n1, n2, a1, a2, d1, d2 = small_graphs
    s = rng.random(n1 * n2)
    thresh = 3.0
    w_deg, w_dist = 1.0, 1.0

    deg1, deg2 = a1.sum(1), a2.sum(1)
    u_deg = np.abs(deg1[:, None] - deg2[None, :]).ravel()

    adj = sparse.kron(a1, a2).toarray()
    diff = sparse.kron(d1, a2).toarray() - sparse.kron(a1, d2).toarray()
    y_true = np.clip((diff / thresh) ** 2, 0, 1)
    w_full = (
        adj
        * (w_deg * (u_deg[:, None] + u_deg[None, :]) + w_dist * y_true)
        / (2 * w_deg + w_dist)
    )
    expected = w_full @ s

    correction = build_distance_correction(
        sparse.csr_matrix(d1), sparse.csr_matrix(d2), thresh, n1, n2
    )
    deg_term = outer_sum_matvec(a1, a2, u_deg, s, n1, n2)
    terms = DistanceTerms(d1, d2, thresh, correction)
    dist_term = distance_term_matvec(a1, a2, terms, s, n1, n2)
    actual = (w_deg * deg_term + w_dist * dist_term) / (2 * w_deg + w_dist)
    assert np.allclose(expected, actual, atol=1e-9)


def test_distance_term_needs_float64_when_edge_lengths_nearly_match():
    # Worst-case cancellation: every edge is ~10 px long, so for every edge
    # pair the expansion d1^2 - 2*d1*d2 + d2^2 subtracts numbers of ~100 to
    # get ~1e-6. distance_term_matvec (float64) must stay accurate, while
    # the same expansion in float32 is off by orders of magnitude -- which
    # is why the distance term is computed in float64.
    rng = np.random.default_rng(2)
    n1, n2 = 40, 45
    a1 = _make_symmetric_adjacency(rng, n1, 0.2)
    a2 = _make_symmetric_adjacency(rng, n2, 0.2)
    lengths1 = np.triu(10 + rng.normal(0, 1e-3, (n1, n1)), 1)
    lengths2 = np.triu(10 + rng.normal(0, 1e-3, (n2, n2)), 1)
    d1 = a1 * (lengths1 + lengths1.T)
    d2 = a2 * (lengths2 + lengths2.T)
    thresh = 5.0
    s = rng.random(n1 * n2)

    diff = sparse.kron(d1, a2).toarray() - sparse.kron(a1, d2).toarray()
    expected = np.clip((diff / thresh) ** 2, 0, 1) @ s

    def rel_err(actual):
        return np.abs(actual - expected).max() / np.abs(expected).max()

    correction = build_distance_correction(
        sparse.csr_array(d1), sparse.csr_array(d2), thresh, n1, n2
    )
    terms = DistanceTerms(d1, d2, thresh, correction)
    assert rel_err(distance_term_matvec(a1, a2, terms, s, n1, n2)) < 1e-6

    f32 = np.float32
    expansion_f32 = (
        kron_matvec(f32(d1 * d1), f32(a2), f32(s), n1, n2)
        - 2 * kron_matvec(f32(d1), f32(d2), f32(s), n1, n2)
        + kron_matvec(f32(a1), f32(d2 * d2), f32(s), n1, n2)
    ) / f32(thresh**2)
    assert rel_err(expansion_f32) > 1e-2


def _reference_build(d1, d2, thresh, n1, n2, chunk_size=2000):
    """A straightforward reference build (lists of COO chunks, concatenated,
    then converted), used as ground truth for build_distance_correction."""
    d1_coo = sparse.coo_array(d1)
    upper = d1_coo.row < d1_coo.col
    e1_a, e1_b = d1_coo.row[upper], d1_coo.col[upper]
    e1_val = d1_coo.data[upper].astype(np.float64)
    d2_coo = sparse.coo_array(d2)
    e2_x, e2_y = d2_coo.row, d2_coo.col
    e2_val = d2_coo.data.astype(np.float64)
    thresh = np.float64(thresh)
    rows, cols, vals = [], [], []
    for start in range(0, len(e1_val), chunk_size):
        stop = min(start + chunk_size, len(e1_val))
        diff = e1_val[start:stop, np.newaxis] - e2_val[np.newaxis, :]
        y = (diff / thresh) ** 2
        ci, cj = np.nonzero(y > 1.0)
        v = e1_a[start:stop][ci].astype(np.int64) * n2 + e2_x[cj]
        w = e1_b[start:stop][ci].astype(np.int64) * n2 + e2_y[cj]
        rows.append(np.minimum(v, w))
        cols.append(np.maximum(v, w))
        vals.append(y[ci, cj] - 1.0)
    n = n1 * n2
    if not rows:
        return sparse.csr_array((n, n), dtype=np.float64)
    return sparse.csr_array(
        sparse.coo_array(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(n, n),
        )
    )


def _radius_graph(rng, n, side, radius):
    """Distance-masked radius graph at real tissue density (degree ~7)."""
    from scipy.spatial import cKDTree

    tree = cKDTree(rng.uniform(0, side, (n, 2)))
    d = sparse.csr_array(tree.sparse_distance_matrix(tree, radius))
    d.eliminate_zeros()
    return d


def _assert_identical(actual, expected):
    assert actual.nnz == expected.nnz
    assert abs(actual - expected).max() == 0.0 if expected.nnz else actual.nnz == 0


@pytest.mark.parametrize("thresh", [2.0, 5.0, 10.0])
def test_build_matches_reference_bit_for_bit(thresh):
    rng = np.random.default_rng(3)
    n1, n2 = 300, 260
    d1 = _radius_graph(rng, n1, 165.0, 15.0)
    d2 = _radius_graph(rng, n2, 150.0, 15.0)
    actual = build_distance_correction(d1, d2, thresh, n1, n2)
    _assert_identical(actual, _reference_build(d1, d2, thresh, n1, n2))
    assert actual.indices.dtype == actual.indptr.dtype == np.int32


def test_build_is_independent_of_block_size():
    rng = np.random.default_rng(4)
    n1, n2 = 200, 180
    d1 = _radius_graph(rng, n1, 135.0, 15.0)
    d2 = _radius_graph(rng, n2, 125.0, 15.0)
    expected = _reference_build(d1, d2, 5.0, n1, n2)
    for block_nnz in [1, 37, 10_000, 10**9]:
        actual = build_distance_correction(d1, d2, 5.0, n1, n2, block_nnz=block_nnz)
        _assert_identical(actual, expected)


def test_build_handles_exact_boundary_ties():
    # Edge lengths on a grid, so |d1 - d2| == thresh (y == 1, not clipped)
    # occurs many times; the searchsorted bounds must land exactly on the
    # predicate boundary.
    rng = np.random.default_rng(5)
    n1, n2 = 30, 35
    a1 = _make_symmetric_adjacency(rng, n1, 0.3)
    a2 = _make_symmetric_adjacency(rng, n2, 0.3)
    lengths1 = np.triu(rng.integers(1, 13, (n1, n1)) * 0.5, 1)
    lengths2 = np.triu(rng.integers(1, 13, (n2, n2)) * 0.5, 1)
    d1 = sparse.csr_array(a1 * (lengths1 + lengths1.T))
    d2 = sparse.csr_array(a2 * (lengths2 + lengths2.T))
    for thresh in [0.5, 1.5, 2.5, 3.0]:
        actual = build_distance_correction(d1, d2, thresh, n1, n2, block_nnz=50)
        _assert_identical(actual, _reference_build(d1, d2, thresh, n1, n2))


def test_build_returns_none_when_nothing_is_clipped(small_graphs):
    rng, n1, n2, a1, a2, d1, d2 = small_graphs
    thresh = 15.0  # every edge length is < 15 in small_graphs
    correction = build_distance_correction(
        sparse.csr_array(d1), sparse.csr_array(d2), thresh, n1, n2
    )
    assert correction is None
    s = rng.random(n1 * n2)
    diff = sparse.kron(d1, a2).toarray() - sparse.kron(a1, d2).toarray()
    expected = ((diff / thresh) ** 2) @ s
    terms = DistanceTerms(d1, d2, thresh, correction)
    actual = distance_term_matvec(a1, a2, terms, s, n1, n2)
    assert np.allclose(expected, actual, atol=1e-9)


def test_build_falls_back_to_int64_indices(small_graphs, monkeypatch):
    import spellmatch.matching._kron_free as kf

    _, n1, n2, _, _, d1, d2 = small_graphs
    d1, d2 = sparse.csr_array(d1), sparse.csr_array(d2)
    monkeypatch.setattr(kf, "_INT32_LIMIT", 10)  # force the fallback
    actual = kf.build_distance_correction(d1, d2, 3.0, n1, n2)
    assert actual.indices.dtype == actual.indptr.dtype == np.int64
    _assert_identical(actual, _reference_build(d1, d2, 3.0, n1, n2))
