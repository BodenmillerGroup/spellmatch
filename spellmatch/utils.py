import logging
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr
from skimage.measure import regionprops
from skimage.transform import ProjectiveTransform

logger = logging.getLogger(__name__)


def compute_centroids(
    mask: xr.DataArray, transform: Optional[np.ndarray] = None, regions=None
) -> pd.DataFrame:
    if regions is None:
        regions = regionprops(mask.to_numpy())
    centroids = np.array([r["centroid"] for r in regions])
    centroids -= 0.5 * np.array([mask.shape])
    if "scale" in mask.attrs:
        centroids *= mask.attrs["scale"]
    if transform is not None:
        centroids = ProjectiveTransform(matrix=transform)(centroids)
    return pd.DataFrame(data=centroids, index=[r["label"] for r in regions])
