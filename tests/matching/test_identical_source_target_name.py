"""Dimension names on the final scores DataArray must be the real
source/target identifiers, not generic placeholders -- and identical
source/target names are prevented earlier (in cli_match), not something
these internals handle.

icp.py, probreg.py and spellmatch.py build their final scores xr.DataArray
as `coords={source_name: ..., target_name: ...}` (source_adj.name/
target_adj.name for spellmatch.py), using the mask's own identifier (its
filename stem, via `source_mask.name or "source"` in _algorithms.py). This
is deliberate: downstream outputs -- in particular the assignment CSV's
column headers, written via assignment.py's `scores.dims[0]: ...,
scores.dims[1]: ...` -- show the real file names this way, not a generic
"source"/"target" label.

Whenever source and target happen to share that identifier (e.g. matching
two panels of the same physical tissue core, both labeled "A3"), the dict
literal collapses to a single key, leaving xr.DataArray unable to infer two
dimensions from one coordinate:

    ValueError: coordinate A3 has dimensions ('A3',), but these are not a
    subset of the DataArray dimensions ('dim_0', 'dim_1')

Observed in a real pipeline run where every source/target pair shared the
same identifier: every pair failed this way, across all three algorithms
(cpd, icp, spellmatch).

This is prevented one layer up: `cli_match` refuses identical
source_mask/target_mask filenames before any matching runs (see
tests/test_cli_match_validation.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spellmatch.matching.algorithms.icp import IterativeClosestPoints
from spellmatch.matching.algorithms.spellmatch import Spellmatch


@pytest.fixture
def synthetic_points():
    rng = np.random.default_rng(0)
    n = 20
    source = pd.DataFrame(rng.uniform(0, 50, (n, 2)), columns=["x", "y"])
    target = pd.DataFrame(
        source.to_numpy() + rng.normal(0, 0.5, (n, 2)), columns=["x", "y"]
    )
    return source, target


def test_icp_match_points_uses_real_names_as_dims(synthetic_points):
    source, target = synthetic_points
    algo = IterativeClosestPoints(max_dist=None)
    info, scores = algo._match_points(
        "source_mask", "target_mask", source, target, None, None, {}
    )
    assert scores.dims == ("source_mask", "target_mask")
    assert scores.shape == (len(source), len(target))


def test_spellmatch_match_graphs_uses_real_names_as_dims(synthetic_points):
    source, target = synthetic_points
    algo = Spellmatch(adj_radius=15, max_iter=1, opt_max_iter=20)
    info, scores = algo._match_graphs_from_points(
        "source_mask", "target_mask", source, target, None, None, {}
    )
    assert scores.dims == ("source_mask", "target_mask")
    assert scores.shape == (len(source), len(target))
