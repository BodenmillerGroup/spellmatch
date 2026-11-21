from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr
from scipy import sparse
from sklearn.neighbors import radius_neighbors_graph


@dataclass(eq=False)
class SparseGraph:
    """Spatial cell graph with sparse adjacency and neighbor distances.

    ``adj`` (bool) and ``dists`` share the same sparsity pattern (CSR), i.e.
    ``dists.data`` holds the distances of all adjacent cell pairs (including distances
    of zero).
    """

    name: str | None
    labels: np.ndarray
    adj: sparse.csr_array
    dists: sparse.csr_array | None

    def __len__(self) -> int:
        return len(self.labels)

    @property
    def degrees(self) -> np.ndarray:
        """Number of neighbors of each cell.

        In CSR format, the neighbors of cell i are stored at positions indptr[i] to
        indptr[i + 1] - 1, so the differences of indptr are the row sums of the
        adjacency matrix.
        """
        return np.diff(self.adj.indptr)

    @staticmethod
    def from_points(
        name: str | None, points: pd.DataFrame, adj_radius: float
    ) -> "SparseGraph":
        """Create a spatial cell graph (cells within ``adj_radius`` are adjacent)."""
        dists = sparse.csr_array(
            radius_neighbors_graph(points.to_numpy(), adj_radius, mode="distance")
        )
        adj = sparse.csr_array(
            (np.ones(dists.nnz, dtype=bool), dists.indices.copy(), dists.indptr.copy()),
            shape=dists.shape,
        )
        return SparseGraph(name, points.index.to_numpy(), adj, dists)

    @staticmethod
    def from_dataarrays(
        adj: xr.DataArray, dists: xr.DataArray | None = None
    ) -> "SparseGraph":
        adj_csr = sparse.csr_array(adj.to_numpy().astype(bool))
        dists_csr = None
        if dists is not None:
            # distances of adjacent cell pairs, on the sparsity pattern of the adjacency
            dists_data = dists.to_numpy()[adj_csr.nonzero()].astype(np.float64)
            dists_csr = sparse.csr_array(
                (dists_data, adj_csr.indices.copy(), adj_csr.indptr.copy()),
                shape=adj_csr.shape,
            )
        labels = adj.coords[adj.dims[0]].to_numpy()
        return SparseGraph(adj.name, labels, adj_csr, dists_csr)
