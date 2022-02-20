import logging
from os import PathLike
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import tifffile
import xarray as xr

from ._spellmatch import SpellmatchError


_DEFAULT_PANEL_NAME_COL = "name"
_DEFAULT_PANEL_KEEP_COL = "keep"

logger = logging.getLogger(__name__)


def read_panel(
    panel_file: Union[str, PathLike], panel_name_col: str = _DEFAULT_PANEL_NAME_COL
) -> pd.DataFrame:
    panel_file = Path(panel_file)
    panel = pd.read_csv(panel_file)
    if panel_name_col not in panel:
        raise SpellmatchIOError(
            f"Column '{panel_name_col}' is missing in panel {panel_file.name}"
        )
    return panel


def read_image(
    img_file: Union[str, PathLike],
    panel: Optional[pd.DataFrame] = None,
    panel_name_col: str = _DEFAULT_PANEL_NAME_COL,
    panel_keep_col: str = _DEFAULT_PANEL_KEEP_COL,
    scale: float = 1.0,
) -> xr.DataArray:
    img_file = Path(img_file)
    img: np.ndarray = tifffile.imread(img_file)
    if img.ndim == 2:
        dims = ("y", "x")
        coords = None
    elif img.ndim == 3:
        if panel is None:
            raise SpellmatchIOError(
                f"No panel provided for multi-channel image {img_file.name}"
            )
        if panel_name_col not in panel:
            raise SpellmatchIOError(f"Column '{panel_name_col}' is missing in panel")
        if len(panel.index) != img.shape[0]:
            raise SpellmatchIOError(
                f"Panel contains {len(panel.index)} channels, "
                f"but {img_file.name} has {img.shape[0]} channels"
            )
        channel_names = panel[panel_name_col]
        if panel_keep_col in panel:
            img = img[panel[panel_keep_col] == 1]
            channel_names = channel_names[panel[panel_keep_col] == 1]
        dupl_channel_names = channel_names.loc[channel_names.duplicated()]
        if len(dupl_channel_names) > 0:
            raise SpellmatchIOError(
                f"Duplicated channel names in panel: {dupl_channel_names.tolist()}"
            )
        dims = ("c", "y", "x")
        coords = {"c": channel_names.tolist()}
    else:
        raise SpellmatchIOError(
            f"{img_file.name} has shape {img.shape}, expected two or three dimensions"
        )
    return xr.DataArray(
        data=img, coords=coords, dims=dims, name=img_file.name, attrs={"scale": scale}
    )


def read_mask(mask_file: Union[str, PathLike], scale: float = 1.0) -> xr.DataArray:
    mask_file = Path(mask_file)
    mask: np.ndarray = tifffile.imread(mask_file)
    if mask.ndim != 2:
        raise SpellmatchIOError(
            f"{mask_file.name} has shape {mask.shape}, expected 2 dimensions (y, x)"
        )
    return xr.DataArray(
        data=mask, dims=("y", "x"), name=mask_file.name, attrs={"scale": scale}
    )


def read_cell_pairs(cell_pairs_file: Union[str, PathLike]) -> pd.DataFrame:
    cell_pairs_file = Path(cell_pairs_file)
    return pd.read_csv(cell_pairs_file, usecols=["Source", "Target"])


def write_cell_pairs(
    cell_pairs_file: Union[str, PathLike], cell_pairs: pd.DataFrame
) -> None:
    cell_pairs_file = Path(cell_pairs_file)
    assert len(cell_pairs.columns) == 2
    assert cell_pairs.columns[0] == "Source"
    assert cell_pairs.columns[1] == "Target"
    cell_pairs.to_csv(cell_pairs_file, index=False)


def read_transform(transform_file: Union[str, PathLike]) -> np.ndarray:
    transform_file = Path(transform_file)
    transform: np.ndarray = np.load(transform_file, allow_pickle=False)
    if transform.shape != (3, 3):
        raise SpellmatchIOError(
            f"Transform {transform_file.name} has shape {transform.shape}, "
            "expected (3, 3)"
        )
    return transform


def write_transform(
    transform_file: Union[str, PathLike], transform: np.ndarray
) -> None:
    transform_file = Path(transform_file)
    assert transform.shape == (3, 3)
    np.save(transform_file, transform, allow_pickle=False)


def read_scores(scores_file: Union[str, PathLike]) -> xr.DataArray:
    return xr.open_dataarray(scores_file, engine="scipy")


def write_scores(scores_file: Union[str, PathLike], scores: xr.DataArray) -> None:
    scores.to_netcdf(scores_file, format="NETCDF3_CLASSIC", engine="scipy")


class SpellmatchIOError(SpellmatchError):
    pass
