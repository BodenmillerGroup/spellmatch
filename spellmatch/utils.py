import logging
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr
from skimage.measure import regionprops

logger = logging.getLogger(__name__)


def compute_points(
    mask: xr.DataArray, regions: Optional[list] = None, points_feature: str = "centroid"
) -> pd.DataFrame:
    if regions is None:
        regions = regionprops(mask.to_numpy())
    points = np.array([r[points_feature] for r in regions])
    points -= 0.5 * np.array([mask.shape])
    if "scale" in mask.attrs:
        points *= mask.attrs["scale"]
    return pd.DataFrame(
        data=points,
        index=pd.Index(data=[r["label"] for r in regions], name=mask.name),
        columns=["y", "x"],
    )


def compute_intensities(
    img: xr.DataArray,
    mask: xr.DataArray,
    regions: Optional[list] = None,
    intensities_feature: str = "intensity_mean",
) -> pd.DataFrame:
    if regions is None:
        regions = regionprops(mask.to_numpy(), intensity_image=img.to_numpy())
    return pd.DataFrame(
        data=np.array([r[intensities_feature] for r in regions]),
        index=pd.Index(data=[r["label"] for r in regions], name=mask.name),
        columns=img.coords.get("c"),
    )
