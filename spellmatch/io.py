from os import PathLike
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import tifffile
import xarray as xr
from skimage.transform import ProjectiveTransform

from ._spellmatch import SpellmatchException


def read_panel(
    panel_file: Union[str, PathLike], panel_name_col: str = "name"
) -> pd.DataFrame:
    panel_file = Path(panel_file)
    panel = pd.read_csv(panel_file)
    if panel_name_col not in panel:
        raise SpellmatchIOException(
            f"Column '{panel_name_col}' is missing in panel {panel_file.name}"
        )
    return panel


def read_image(
    img_file: Union[str, PathLike],
    panel: Optional[pd.DataFrame] = None,
    panel_name_col: str = "name",
    panel_keep_col: str = "keep",
    scale: float = 1,
    zscale: Optional[float] = None,
) -> xr.DataArray:
    img_file = Path(img_file)
    img: np.ndarray = tifffile.imread(img_file)
    if img.ndim == 2:
        dims = ("y", "x")
        scale = (scale, scale)
        coords = None
    elif img.ndim in (3, 4):
        if panel is None:
            raise SpellmatchIOException(
                f"No panel provided for multi-channel image {img_file.name}"
            )
        if panel_name_col not in panel:
            raise SpellmatchIOException(
                f"Column '{panel_name_col}' is missing in panel"
                f" for image {img_file.name}"
            )
        channel_axis = 1 if img.ndim == 4 else 0
        if len(panel.index) != img.shape[channel_axis]:
            raise SpellmatchIOException(
                f"Panel contains {len(panel.index)} channels, "
                f"but {img_file.name} has {img.shape[channel_axis]} channels"
            )
        channel_names = panel[panel_name_col]
        if panel_keep_col in panel:
            if img.ndim == 3:
                img = img[panel[panel_keep_col] == 1, :, :]
            else:
                img = img[:, panel[panel_keep_col] == 1, :, :]
            channel_names = channel_names[panel[panel_keep_col] == 1]
        dupl_channel_names = channel_names.loc[channel_names.duplicated()]
        if len(dupl_channel_names) > 0:
            raise SpellmatchIOException(
                f"Duplicated channel names in panel: {dupl_channel_names.tolist()}"
            )
        if img.ndim == 3:
            dims = ("c", "y", "x")
            scale = (scale, scale)
        else:
            dims = ("z", "c", "y", "x")
            scale = (zscale or scale, scale, scale)
        coords = {"c": channel_names.tolist()}
    else:
        raise SpellmatchIOException(
            f"{img_file.name} has shape {img.shape}, expected two to four dimensions"
        )
    return xr.DataArray(
        data=img, coords=coords, dims=dims, name=img_file.name, attrs={"scale": scale}
    )


def read_mask(
    mask_file: Union[str, PathLike], scale: float = 1, zscale: Optional[float] = None,
) -> xr.DataArray:
    mask_file = Path(mask_file)
    mask: np.ndarray = tifffile.imread(mask_file)
    if mask.ndim not in (2, 3):
        raise SpellmatchIOException(
            f"{mask_file.name} has shape {mask.shape}, expected two or three dimensions"
        )
    if mask.ndim == 2:
        dims = ("y", "x")
        scale = (scale, scale)
    else:
        dims = ("z", "y", "x")
        scale = (zscale or scale, scale, scale)
    return xr.DataArray(
        data=mask, dims=dims, name=mask_file.name, attrs={"scale": scale}
    )


def read_transform(transform_file: Union[str, PathLike]) -> ProjectiveTransform:
    transform_file = Path(transform_file)
    transform_matrix: np.ndarray = np.load(transform_file, allow_pickle=False)
    if transform_matrix.shape not in ((3, 3), (4, 4)):
        raise SpellmatchIOException(
            f"Transform {transform_file.name} has shape {transform_matrix.shape}, "
            "expected (3, 3) or (4, 4)"
        )
    return ProjectiveTransform(matrix=transform_matrix)


def write_transform(
    transform_file: Union[str, PathLike], transform: ProjectiveTransform
) -> None:
    transform_file = Path(transform_file)
    np.save(transform_file, transform.params, allow_pickle=False)


def read_scores(scores_file: Union[str, PathLike]) -> xr.DataArray:
    return xr.open_dataarray(scores_file, engine="scipy")


def write_scores(scores_file: Union[str, PathLike], scores: xr.DataArray) -> None:
    scores.to_netcdf(scores_file, format="NETCDF3_CLASSIC", engine="scipy")


def read_assignment(assignment_file: Union[str, PathLike]) -> pd.DataFrame:
    assignment_file = Path(assignment_file)
    assignment = pd.read_csv(assignment_file)
    if len(assignment.columns) != 2:
        raise SpellmatchIOException(
            f"Malformed assignment file: {assignment_file.name}"
        )
    return assignment


def write_assignment(
    assignment_file: Union[str, PathLike], assignment: pd.DataFrame
) -> None:
    assignment_file = Path(assignment_file)
    assert len(assignment.columns) == 2
    assignment.to_csv(assignment_file, index=False)


class SpellmatchIOException(SpellmatchException):
    pass
