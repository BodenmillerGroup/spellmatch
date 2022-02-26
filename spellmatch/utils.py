from typing import Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import distance
from shapely.geometry import Point, Polygon
from skimage.measure import regionprops
from skimage.transform import ProjectiveTransform


def create_bounding_box(mask: xr.DataArray) -> Polygon:
    shell = np.array(
        [
            [0.5 * mask.shape[-1], -0.5 * mask.shape[-2]],
            [0.5 * mask.shape[-1], 0.5 * mask.shape[-2]],
            [-0.5 * mask.shape[-1], 0.5 * mask.shape[-2]],
            [-0.5 * mask.shape[-1], -0.5 * mask.shape[-2]],
        ]
    )
    if "scale" in mask.attrs:
        shell *= mask.attrs["scale"]
    return Polygon(shell=shell)


def compute_points(
    mask: xr.DataArray, regions: Optional[list] = None, points_feature: str = "centroid"
) -> pd.DataFrame:
    if regions is None:
        regions = regionprops(mask.to_numpy())
    points = np.array([r[points_feature][::-1] for r in regions])
    points += [[-0.5 * mask.shape[-1] + 0.5, -0.5 * mask.shape[-2] + 0.5]]
    if "scale" in mask.attrs:
        points *= mask.attrs["scale"]
    return pd.DataFrame(
        data=points,
        index=pd.Index(data=[r["label"] for r in regions], name=mask.name),
        columns=["x", "y"],
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


def create_graph(
    points: pd.DataFrame, adj_radius: float, xdim: str, ydim: str
) -> Tuple[xr.DataArray, xr.DataArray]:
    dists = xr.DataArray(
        data=distance.squareform(distance.pdist(points)),
        coords={xdim: points.index, ydim: points.index},
        name=points.index.name,
    )
    adj = dists <= adj_radius
    np.fill_diagonal(adj, False)
    return adj, dists


def transform_bounding_box(bbox: Polygon, transform: ProjectiveTransform) -> Polygon:
    return Polygon(transform(np.asarray(bbox.exterior.coords)))


def transform_points(
    points: pd.DataFrame, transform: ProjectiveTransform
) -> pd.DataFrame:
    return pd.DataFrame(
        data=transform(points.values),
        index=points.index.copy(),
        columns=points.columns.copy(),
    )


def filter_outlier_points(
    points: pd.DataFrame, bbox: Polygon, outlier_dist: float
) -> pd.DataFrame:
    if outlier_dist > 0.0:
        bbox = bbox.buffer(outlier_dist)
    filtered_mask = [Point(point).within(bbox) for point in points.values]
    return points[filtered_mask]


def restore_outlier_scores(
    source_index: pd.Index, target_index: pd.Index, filtered_scores: xr.DataArray
) -> xr.DataArray:
    scores = xr.DataArray(
        data=np.zeros((len(source_index), len(target_index))),
        coords={
            source_index.name: source_index.values,
            target_index.name: target_index.values,
        },
    )
    scores.loc[filtered_scores.coords] = filtered_scores
    return scores
