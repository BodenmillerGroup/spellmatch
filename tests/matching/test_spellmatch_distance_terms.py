"""Integration tests for Spellmatch._compute_distance_terms.

These exercise the real code path used during matching, not just the
underlying _kron_free functions (see test_kron_free.py for those).
"""

from __future__ import annotations

import numpy as np
import pytest

from spellmatch.matching.algorithms.spellmatch import Spellmatch


def _make_symmetric_adjacency(rng, n, p_edge):
    a = (rng.random((n, n)) < p_edge).astype(bool)
    a = np.triu(a, 1)
    return a | a.T


@pytest.fixture
def dense_graph_pair():
    rng = np.random.default_rng(2)
    n1, n2 = 150, 130
    adj1 = _make_symmetric_adjacency(rng, n1, 0.3)
    adj2 = _make_symmetric_adjacency(rng, n2, 0.3)
    dists1 = np.triu(rng.uniform(1, 15, (n1, n1)), 1)
    dists1 = dists1 + dists1.T
    dists2 = np.triu(rng.uniform(1, 15, (n2, n2)), 1)
    dists2 = dists2 + dists2.T
    return adj1, adj2, dists1, dists2, n1, n2


def test_correction_is_built_when_edge_pairs_are_clipped(dense_graph_pair):
    adj1, adj2, dists1, dists2, n1, n2 = dense_graph_pair
    algo = Spellmatch(distance_weight=1)
    terms = algo._compute_distance_terms(adj1, adj2, dists1, dists2, n1, n2)
    assert terms.d1.shape == (n1, n1)
    assert terms.d2.shape == (n2, n2)
    assert terms.thresh == algo.distance_cdiff_thres
    assert terms.correction.shape == (n1 * n2, n1 * n2)


def test_no_correction_when_no_edge_is_longer_than_thresh(dense_graph_pair):
    adj1, adj2, dists1, dists2, n1, n2 = dense_graph_pair
    # every edge is shorter than 15, so no edge pair can differ by more
    algo = Spellmatch(distance_weight=1, distance_cdiff_thres=15)
    terms = algo._compute_distance_terms(adj1, adj2, dists1, dists2, n1, n2)
    assert terms.correction is None


def test_no_correction_when_no_edge_pair_differs_by_more_than_thresh(
    dense_graph_pair,
):
    adj1, adj2, _, _, n1, n2 = dense_graph_pair
    rng = np.random.default_rng(3)
    # edges are longer than thresh (10-14 > 5), but no two differ by more than 4
    dists1 = np.triu(rng.uniform(10, 14, (n1, n1)), 1)
    dists1 = dists1 + dists1.T
    dists2 = np.triu(rng.uniform(10, 14, (n2, n2)), 1)
    dists2 = dists2 + dists2.T
    algo = Spellmatch(distance_weight=1, distance_cdiff_thres=5)
    terms = algo._compute_distance_terms(adj1, adj2, dists1, dists2, n1, n2)
    assert terms.correction is None
