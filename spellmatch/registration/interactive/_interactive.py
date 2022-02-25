from typing import Optional, Tuple, Type, Union

import numpy as np
import pandas as pd
import xarray as xr
from napari.layers import Image, Labels
from napari.viewer import Viewer
from qtpy.QtWidgets import QApplication
from skimage.transform import AffineTransform, ProjectiveTransform

from ...utils import compute_points
from .._registration import SpellmatchRegistrationError
from ._qt import QPointMatchingDialog


def align_masks(
    source_mask: xr.DataArray,
    target_mask: xr.DataArray,
    source_img: Optional[Union[np.ndarray, xr.DataArray]] = None,
    target_img: Optional[Union[np.ndarray, xr.DataArray]] = None,
    transform_type: Type[ProjectiveTransform] = AffineTransform,
    cell_pairs: Optional[pd.DataFrame] = None,
) -> Optional[Tuple[pd.DataFrame, Optional[ProjectiveTransform]]]:
    if source_img is not None and source_img.shape[:-2] != source_mask.shape:
        raise SpellmatchInteractiveRegistrationError(
            f"Source image has shape {source_img.shape}, "
            f"but source mask has shape {source_mask.shape}"
        )
    if target_img is not None and target_img.shape[:-2] != target_mask.shape:
        raise SpellmatchInteractiveRegistrationError(
            f"Source image has shape {target_img.shape}, "
            f"but source mask has shape {target_mask.shape}"
        )
    app = QApplication([])

    source_viewer, source_mask_layer, _ = _create_viewer(
        "Source", source_mask, img=source_img
    )
    target_viewer, target_mask_layer, _ = _create_viewer(
        "Target", target_mask, img=target_img
    )

    point_matching_dialog = QPointMatchingDialog(
        compute_points(source_mask),
        compute_points(target_mask),
        transform_type=transform_type,
        label_pairs_columns=(
            cell_pairs.columns.tolist()
            if cell_pairs is not None
            else ("Source", "Target")
        ),
    )
    if cell_pairs is not None:
        point_matching_dialog.label_pairs = cell_pairs

    def on_layer_mouse_drag(mask_layer: Labels, event) -> None:
        selected_label = mask_layer.get_value(event.position, world=True)
        if selected_label != 0:
            mask_layer.metadata["selected_label"] = selected_label
            source_label = source_mask_layer.metadata.get("selected_label")
            target_label = target_mask_layer.metadata.get("selected_label")
            if source_label is not None and target_label is not None:
                point_matching_dialog.append_label_pair(source_label, target_label)
                source_mask_layer.metadata["selected_label"] = None
                target_mask_layer.metadata["selected_label"] = None
        else:
            mask_layer.metadata["selected_label"] = None

    source_mask_layer.mouse_drag_callbacks.append(on_layer_mouse_drag)
    target_mask_layer.mouse_drag_callbacks.append(on_layer_mouse_drag)
    point_matching_dialog.finished.connect(lambda _: app.exit())

    source_viewer.show()
    target_viewer.show()
    point_matching_dialog.show()
    point_matching_dialog.raise_()
    point_matching_dialog.activateWindow()

    app.exec()
    result = point_matching_dialog.result()
    if result == QPointMatchingDialog.DialogCode.Accepted:
        return (
            point_matching_dialog.label_pairs,
            point_matching_dialog.transform,
        )
    return None


def _create_viewer(
    title: str, mask: xr.DataArray, img: Optional[xr.DataArray] = None
) -> Tuple[Viewer, Labels, Optional[list[Image]]]:
    if img is not None and img.name is not None:
        title += f": {img.name}"
    viewer = Viewer(title=title, axis_labels=("y", "x"), show=False)
    img_layers = None
    if img is not None:
        img_data = img.data
        img_name = img.name
        img_channel_axis = None
        if "c" in img.dims:
            img_data = img.data[::-1]
            if "c" in img.coords:
                img_name = img.coords["c"][::-1]
            img_channel_axis = img.dims.index("c")
        img_scale = None
        if "scale" in img.attrs:
            img_scale = (img.attrs["scale"], img.attrs["scale"])
        img_layers = viewer.add_image(
            data=img_data,
            channel_axis=img_channel_axis,
            rgb=False,
            colormap="gray",
            name=img_name,
            scale=img_scale,
            translate=-0.5 * np.array(img.shape[1:]),
            visible=False,
        )
    mask_scale = None
    if "scale" in mask.attrs:
        mask_scale = (mask.attrs["scale"], mask.attrs["scale"])
    mask_layer = viewer.add_labels(
        data=mask,
        name=mask.name,
        scale=mask_scale,
        translate=-0.5 * np.array(mask.shape),
    )
    return viewer, mask_layer, img_layers


class SpellmatchInteractiveRegistrationError(SpellmatchRegistrationError):
    pass
