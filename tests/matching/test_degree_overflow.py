"""End-to-end check of Spellmatch's graph matching against the explicit formula.

`Spellmatch._match_graphs` never builds the (n1*n2)x(n1*n2) product-graph
matrices; it computes ``w_csr @ s`` matrix-free (see ``_kron_free.py``). This
test runs it on small graphs and compares the scores with a dense float64
implementation of the explicit formulation

    W = D^-1/2 (A - combined) D^-1/2,   s <- alpha * W s + (1 - alpha) * h

with the degree and distance terms switched on.

The graphs are dense enough that degree products exceed 255. This guards the
uint8 degree-product overflow bug: degrees were summed as uint8, and
``deg1[:, None] * deg2[None, :]`` stays uint8 (numpy does not promote
same-dtype arithmetic), so e.g. 17*16=272 silently wrapped to 16. That
corrupted the D^-1/2 normalization for the affected candidate pairs, broke
the assumption that keeps the operator's spectral radius <= 1, and caused a
real NaN-divergence failure on IMC data. Degrees are now uint16.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from spellmatch.matching.algorithms.spellmatch import Spellmatch
from spellmatch.utils import create_graph


def _points(rng, n, side):
    return pd.DataFrame(
        rng.uniform(0, side, (n, 2)),
        columns=["x", "y"],
        index=pd.Index(np.arange(1, n + 1), name="cell"),
    )


def _reference_scores(adj1, adj2, dists1, dists2, algo):
    """Dense float64 implementation of the explicit Spellmatch iteration."""
    a1 = adj1.astype(np.float64)
    a2 = adj2.astype(np.float64)
    n = len(a1) * len(a2)
    adj = np.kron(a1, a2)
    deg1, deg2 = a1.sum(axis=1), a2.sum(axis=1)
    deg = np.kron(deg1, deg2)
    d = np.zeros(n)
    d[deg > 0] = deg[deg > 0] ** -0.5
    u = np.clip(
        (np.abs(deg1[:, None] - deg2[None, :]) / algo.degree_cdiff_thres) ** 2, 0, 1
    ).ravel()
    length_diff = np.kron(a1 * dists1, a2) - np.kron(a1, a2 * dists2)
    y = np.clip((length_diff / algo.distance_cdiff_thres) ** 2, 0, 1)
    combined = adj * (
        algo.degree_weight * (u[:, None] + u[None, :]) + algo.distance_weight * y
    )
    combined /= 2 * algo.degree_weight + algo.distance_weight
    w = d[:, None] * (adj - combined) * d[None, :]
    h = np.full(n, 1.0 / n)
    s = h.copy()
    for _ in range(algo.opt_max_iter):
        s_new = algo.alpha * (w @ s) + (1 - algo.alpha) * h
        converged = np.linalg.norm(s - s_new) < algo.opt_tol
        s = s_new
        if converged:
            break
    return s.reshape(len(a1), len(a2))


def test_match_graphs_matches_explicit_formula_with_large_degrees():
    rng = np.random.default_rng(0)
    # ~30 cells in a 25x25 px square with adj_radius=15: most cells have
    # 15-25 neighbors, so degree products exceed 255.
    source_adj, source_dists = create_graph(
        "source", _points(rng, 30, 25.0), 15.0, "a", "b"
    )
    target_adj, target_dists = create_graph(
        "target", _points(rng, 28, 25.0), 15.0, "x", "y"
    )
    adj1, adj2 = source_adj.to_numpy(), target_adj.to_numpy()
    assert adj1.sum(axis=1).max() * adj2.sum(axis=1).max() > 255

    algo = Spellmatch(
        adj_radius=15.0,
        degree_weight=1.0,
        distance_weight=1.0,
        distance_cdiff_thres=5.0,
        precision="float64",
        opt_max_iter=500,
        opt_tol=1e-12,
    )
    _, scores = algo._match_graphs(
        source_adj, target_adj, source_dists, target_dists, None, None, {}
    )

    expected = _reference_scores(
        adj1, adj2, source_dists.to_numpy(), target_dists.to_numpy(), algo
    )
    assert np.all(np.isfinite(scores.to_numpy()))
    np.testing.assert_allclose(scores.to_numpy(), expected, rtol=1e-8, atol=0)
