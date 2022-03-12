from typing import TYPE_CHECKING, Callable, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import gaussian_filter, median_filter
from scipy.spatial import distance
from shapely.geometry import Point, Polygon
from skimage.measure import regionprops
from skimage.transform import AffineTransform, ProjectiveTransform

if TYPE_CHECKING:
    import pyqtgraph as pg
    from qtpy.QtCore import QEventLoop


def describe_image(img: xr.DataArray) -> str:
    return f"{np.dtype(img.dtype).name} {img.shape[1:]}, {img.shape[0]} channels"


def describe_mask(mask: xr.DataArray) -> str:
    return f"{np.dtype(mask.dtype).name} {mask.shape}, {len(np.unique(mask)) - 1} cells"


def describe_transform(transform: ProjectiveTransform) -> str:
    if type(transform) is ProjectiveTransform:
        transform = AffineTransform(matrix=transform.params)
    transform_infos = []
    if hasattr(transform, "scale"):
        if np.isscalar(transform.scale):
            transform_infos.append(f"scale: {transform.scale:.3f}")
        else:
            transform_infos.append(
                f"scale: sx={transform.scale[0]:.3f} sy={transform.scale[1]:.3f}"
            )
    if hasattr(transform, "rotation"):
        transform_infos.append(f"ccw rotation: {180 * transform.rotation / np.pi:.3f}°")
    if hasattr(transform, "shear"):
        transform_infos.append(f"ccw shear: {180 * transform.shear / np.pi:.3f}°")
    if hasattr(transform, "translation"):
        transform_infos.append(
            "translation: "
            f"tx={transform.translation[0]:.3f} ty={transform.translation[1]:.3f}"
        )
    return ", ".join(transform_infos)


def describe_scores(scores: xr.DataArray) -> str:
    max2_scores = -np.partition(-scores.to_numpy(), 1, axis=-1)[:, :2]
    mean_score = np.mean(max2_scores[:, 0])
    mean_margin = np.mean(max2_scores[:, 0] - max2_scores[:, 1])
    return f"mean score: {mean_score}, mean margin: {mean_margin}"


def describe_assignment(
    assignment: pd.DataFrame, recovered: Optional[float] = None
) -> str:
    text = f"{len(assignment.index)} cell pairs"
    if recovered is not None:
        text += f", recovered {recovered:.0%}"
    return text


def preprocess_image(
    img: xr.DataArray,
    median_filter_size: Optional[int] = None,
    clipping_quantile: Optional[float] = None,
    transform_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    gaussian_filter_sigma: Optional[float] = None,
    inplace=False,
) -> Optional[xr.DataArray]:
    if not inplace:
        img = img.copy()
    if median_filter_size is not None:
        img[:] = median_filter(img.to_numpy(), size=median_filter_size)
    if clipping_quantile is not None:
        clipping_max = np.quantile(img.to_numpy(), clipping_quantile)
        img[:] = np.clip(img.to_numpy(), None, clipping_max)
    if transform_func is not None:
        img[:] = transform_func(img.to_numpy())
    if gaussian_filter_sigma is not None:
        img[:] = gaussian_filter(img.to_numpy(), gaussian_filter_sigma)
    if not inplace:
        return img
    return None


def compute_points(
    mask: xr.DataArray, regions: Optional[list] = None, points_feature: str = "centroid"
) -> pd.DataFrame:
    if regions is None:
        regions = regionprops(mask.to_numpy())
    points = (
        np.array([region[points_feature] for region in regions])
        - 0.5 * np.array([mask.shape])
        + 0.5
    )[:, ::-1]
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
        intensity_image = np.moveaxis(img.to_numpy(), 0, -1)
        regions = regionprops(mask.to_numpy(), intensity_image=intensity_image)
    return pd.DataFrame(
        data=np.array([r[intensities_feature] for r in regions]),
        index=pd.Index(data=[r["label"] for r in regions], name=mask.name),
        columns=img.coords.get("c"),
    )


def create_bounding_box(mask: xr.DataArray) -> Polygon:
    bbox_shell = np.array(
        [
            [0.5 * mask.shape[-1], -0.5 * mask.shape[-2]],
            [0.5 * mask.shape[-1], 0.5 * mask.shape[-2]],
            [-0.5 * mask.shape[-1], 0.5 * mask.shape[-2]],
            [-0.5 * mask.shape[-1], -0.5 * mask.shape[-2]],
        ]
    )
    if "scale" in mask.attrs:
        bbox_shell *= mask.attrs["scale"]
    return Polygon(shell=bbox_shell)


def transform_points(
    points: pd.DataFrame, transform: ProjectiveTransform
) -> pd.DataFrame:
    return pd.DataFrame(
        data=transform(points.to_numpy()),
        index=points.index.copy(),
        columns=points.columns.copy(),
    )


def transform_bounding_box(bbox: Polygon, transform: ProjectiveTransform) -> Polygon:
    return Polygon(shell=transform(np.asarray(bbox.exterior.coords)))


def filter_outlier_points(
    points: pd.DataFrame, bbox: Polygon, outlier_dist: float
) -> pd.DataFrame:
    if outlier_dist > 0:
        bbox = bbox.buffer(outlier_dist)
    filtered_mask = [Point(point).within(bbox) for point in points.to_numpy()]
    return points[filtered_mask]


def restore_outlier_scores(
    source_index: pd.Index, target_index: pd.Index, filtered_scores: xr.DataArray
) -> xr.DataArray:
    scores = xr.DataArray(
        data=np.zeros((len(source_index), len(target_index))),
        coords={
            source_index.name: source_index.to_numpy(),
            target_index.name: target_index.to_numpy(),
        },
    )
    scores.loc[filtered_scores.coords] = filtered_scores
    return scores


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


def show_image(
    img: Optional[np.ndarray], window_title: Optional[str] = None
) -> Tuple["pg.ImageView", "QEventLoop"]:
    import pyqtgraph as pg
    from qtpy.QtCore import QEventLoop, Qt

    pg.setConfigOption("imageAxisOrder", "row-major")
    pg.mkQApp()
    imv = pg.ImageView()
    imv_loop = QEventLoop()
    imv.destroyed.connect(imv_loop.quit)
    imv.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
    if img is not None:
        imv.setImage(img)
    if window_title is not None:
        imv.setWindowTitle(window_title)
    imv.show()
    return imv, imv_loop
