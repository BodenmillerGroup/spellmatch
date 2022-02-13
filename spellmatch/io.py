from os import PathLike
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import tifffile
import xarray as xr
from skimage.transform import AffineTransform


def read_panel(panel_file: Union[str, PathLike]) -> pd.DataFrame:
    return pd.read_csv(panel_file)


def read_image(
    img_file: Union[str, PathLike],
    panel: Optional[pd.DataFrame] = None,
    panel_name_col: str = "name",
    panel_keep_col: str = "keep",
) -> xr.DataArray:
    img_file = Path(img_file)
    img: np.ndarray = tifffile.imread(img_file)
    if img.ndim != 3:
        raise ValueError(
            f"{img_file.name} has shape {img.shape}, expected 3 dimensions (c, y, x)"
        )
    channel_names = None
    if panel is not None:
        if len(panel.index) != img.shape[0]:
            raise ValueError(
                f"Panel has {len(panel.index)} channels,"
                f" but {img_file.name} has {img.shape[0]}"
            )
        if panel_name_col not in panel:
            raise ValueError(f"Column '{panel_name_col}' is missing in panel")
        channel_names = panel[panel_name_col]
        if panel_keep_col in panel:
            img = img[panel[panel_keep_col] == 1]
            channel_names = channel_names[panel[panel_keep_col] == 1]
    return xr.DataArray(
        data=img,
        coords={"c": channel_names} if channel_names is not None else None,
        dims=["c", "y", "x"],
        name=img_file.name,
    )


def read_mask(mask_file: Union[str, PathLike]) -> np.ndarray:
    mask_file = Path(mask_file)
    mask: np.ndarray = tifffile.imread(mask_file)
    if mask.ndim != 2:
        raise ValueError(
            f"{mask_file.name} has shape {mask.shape}, expected 2 dimensions (y, x)"
        )
    return mask


def read_label_pairs(label_pairs_file: Union[str, PathLike]) -> pd.DataFrame:
    return pd.read_csv(label_pairs_file, usecols=["Source", "Target"])


def write_label_pairs(
    label_pairs_file: Union[str, PathLike], label_pairs: pd.DataFrame
) -> None:
    assert len(label_pairs.columns) == 2
    assert label_pairs.columns[0] == "Source"
    assert label_pairs.columns[1] == "Target"
    label_pairs.to_csv(label_pairs_file, index=False)


def read_transform(transform_file: Union[str, PathLike]) -> AffineTransform:
    transform_params = np.load(transform_file, allow_pickle=False)
    return AffineTransform(matrix=transform_params)


def write_transform(
    transform_file: Union[str, PathLike], transform: AffineTransform
) -> None:
    np.save(transform_file, transform, allow_pickle=False)
