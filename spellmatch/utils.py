import builtins
import importlib
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Optional

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
    dtype_name = np.dtype(img.dtype).name
    if img.ndim == 2:
        return f"{dtype_name} {img.shape}, 1 channel"
    if img.ndim == 3:
        return f"{dtype_name} {img.shape[1:]}, {img.shape[0]} channels"
    if img.ndim == 4:
        shape = tuple(x for i, x in enumerate(img.shape) if i != 1)
        return f"{dtype_name} {shape}, {img.shape[1]} channels"
    raise ValueError(f"Unsupported number of dimensions: {img.name}")


def describe_mask(mask: xr.DataArray) -> str:
    dtype_name = np.dtype(mask.dtype).name
    return f"{dtype_name} {mask.shape}, {len(np.unique(mask)) - 1} cells"


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
    if hasattr(transform, "rotation") and transform.dimensionality == 2:
        transform_infos.append(f"ccw rotation: {180 * transform.rotation / np.pi:.3f}°")
    if hasattr(transform, "shear") and transform.dimensionality == 2:
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
    return f"mean score: {mean_score:.6f}, mean margin: {mean_margin:.6f}"


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
    if gaussian_filter_sigma is not None:
        img[:] = gaussian_filter(img.to_numpy(), gaussian_filter_sigma)
    if not inplace:
        return img
    return None


def compute_points(
    mask: xr.DataArray, regions: Optional[list] = None, point_feature: str = "centroid"
) -> pd.DataFrame:
    if regions is None:
        regions = regionprops(mask.to_numpy())
    points = (
        np.array([region[point_feature] for region in regions])
        - 0.5 * np.array([mask.shape])
        + 0.5
    )[:, ::-1]
    if "scale" in mask.attrs:
        points *= np.expand_dims(mask.attrs["scale"], 0)
    return pd.DataFrame(
        data=points,
        index=pd.Index(data=[r["label"] for r in regions], name="cell"),
        columns=["x", "y"] if mask.ndim == 2 else ["x", "y", "z"],
    )


def compute_intensities(
    img: xr.DataArray,
    mask: xr.DataArray,
    regions: Optional[list] = None,
    intensity_feature: str = "intensity_mean",
    intensity_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> pd.DataFrame:
    if regions is None:
        intensity_image = np.moveaxis(img.to_numpy(), 1 if img.ndim == 4 else 0, -1)
        regions = regionprops(mask.to_numpy(), intensity_image=intensity_image)
    intensities_data = np.array([r[intensity_feature] for r in regions])
    if intensity_transform is not None:
        intensities_data = intensity_transform(intensities_data)
    return pd.DataFrame(
        data=intensities_data,
        index=pd.Index(data=[r["label"] for r in regions], name="cell"),
        columns=img.coords.get("c"),
    )


def create_bounding_box(mask: xr.DataArray) -> Polygon:
    if mask.ndim != 2:
        # TODO add support for 3D bounding boxes
        raise NotImplementedError("3D bounding boxes are not supported")
    bbox_shell = np.array(
        [
            [0.5 * mask.shape[-1], -0.5 * mask.shape[-2]],
            [0.5 * mask.shape[-1], 0.5 * mask.shape[-2]],
            [-0.5 * mask.shape[-1], 0.5 * mask.shape[-2]],
            [-0.5 * mask.shape[-1], -0.5 * mask.shape[-2]],
        ]
    )
    if "scale" in mask.attrs:
        bbox_shell *= np.expand_dims(mask.attrs["scale"], 0)
    return Polygon(shell=bbox_shell)


def transform_points(
    points: pd.DataFrame, transform: ProjectiveTransform, inplace: bool = False
) -> Optional[pd.DataFrame]:
    if not inplace:
        points = points.copy()
    points[:] = transform(points.to_numpy())
    if not inplace:
        return points


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
    source_labels: np.ndarray, target_labels: np.ndarray, filtered_scores: xr.DataArray
) -> xr.DataArray:
    scores = xr.DataArray(
        data=np.zeros((len(source_labels), len(target_labels))),
        coords={
            filtered_scores.dims[0]: source_labels,
            filtered_scores.dims[1]: target_labels,
        },
    )
    scores.loc[filtered_scores.coords] = filtered_scores
    return scores


def create_graph(
    name: str, points: pd.DataFrame, adj_radius: float, xdim: str, ydim: str
) -> tuple[xr.DataArray, xr.DataArray]:
    dists_data = distance.squareform(distance.pdist(points.to_numpy()))
    dists = xr.DataArray(
        data=dists_data,
        coords={xdim: points.index.to_numpy(), ydim: points.index.to_numpy()},
        name=name,
    )
    adj_data = dists_data <= adj_radius
    np.fill_diagonal(adj_data, False)
    adj = dists.copy(data=adj_data)
    return adj, dists


def show_image(
    img: Optional[np.ndarray], window_title: Optional[str] = None
) -> tuple["pg.ImageView", "QEventLoop"]:
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


def get_function_by_name(name: str) -> Any:
    parts = name.rsplit(sep=".", maxsplit=1)
    if len(parts) == 1:
        module_name = "builtins"
        function_name = parts
        module = builtins
    else:
        module_name, function_name = parts
        module = importlib.import_module(module_name)
    return getattr(module, function_name)
