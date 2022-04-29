from typing import Optional, Union

import numpy as np
import pandas as pd
import xarray as xr
from napari.layers import Image, Labels
from napari.viewer import Viewer
from qtpy.QtCore import QEventLoop
from skimage.transform import AffineTransform, ProjectiveTransform

from ...utils import compute_points
from .._registration import SpellmatchRegistrationException
from ._qt import QInteractiveRegistrationDialog


def register_interactive(
    source_mask: xr.DataArray,
    target_mask: xr.DataArray,
    source_img: Optional[Union[np.ndarray, xr.DataArray]] = None,
    target_img: Optional[Union[np.ndarray, xr.DataArray]] = None,
    transform_type: type[ProjectiveTransform] = AffineTransform,
    assignment: Optional[pd.DataFrame] = None,
) -> Optional[tuple[pd.DataFrame, Optional[ProjectiveTransform]]]:
    if source_mask.ndim != 2 or target_mask.ndim != 2:
        raise NotImplementedError("3D registration is not supported")
    if source_img is not None and source_img.shape[-2:] != source_mask.shape:
        raise SpellmatchInteractiveRegistrationException(
            f"Source image has shape {source_img.shape}, "
            f"but source mask has shape {source_mask.shape}"
        )
    if target_img is not None and target_img.shape[-2:] != target_mask.shape:
        raise SpellmatchInteractiveRegistrationException(
            f"Source image has shape {target_img.shape}, "
            f"but source mask has shape {target_mask.shape}"
        )

    source_viewer, source_mask_layer, _ = _create_viewer(
        "Source", source_mask, img=source_img
    )
    target_viewer, target_mask_layer, _ = _create_viewer(
        "Target", target_mask, img=target_img
    )

    point_matching_dialog = QInteractiveRegistrationDialog(
        compute_points(source_mask),
        compute_points(target_mask),
        transform_type=transform_type,
        label_pairs_columns=(
            assignment.columns.tolist()
            if assignment is not None
            else (source_mask.name or "source", target_mask.name or "target")
        ),
    )
    if assignment is not None:
        point_matching_dialog.label_pairs = assignment

    def on_mask_layer_mouse_drag(mask_layer: Labels, event) -> None:
        selected_label = mask_layer.get_value(event.position, world=True)
        if selected_label:
            mask_layer.selected_label = selected_label
            if source_mask_layer.selected_label and target_mask_layer.selected_label:
                point_matching_dialog.append_label_pair(
                    source_mask_layer.selected_label, target_mask_layer.selected_label
                )
                source_mask_layer.selected_label = 0
                target_mask_layer.selected_label = 0
        else:
            mask_layer.selected_label = 0
        yield
        while event.type == "mouse_move":
            mask_layer.selected_label = 0
            yield

    point_matching_event_loop = QEventLoop()
    source_mask_layer.mouse_drag_callbacks.append(on_mask_layer_mouse_drag)
    target_mask_layer.mouse_drag_callbacks.append(on_mask_layer_mouse_drag)
    point_matching_dialog.finished.connect(lambda _: point_matching_event_loop.quit())

    source_viewer.show()
    target_viewer.show()
    point_matching_dialog.show()
    point_matching_dialog.raise_()
    point_matching_dialog.activateWindow()
    point_matching_event_loop.exec()
    source_viewer.close()
    target_viewer.close()

    result = point_matching_dialog.result()
    if result == QInteractiveRegistrationDialog.DialogCode.Accepted:
        return (
            point_matching_dialog.label_pairs,
            point_matching_dialog.transform,
        )
    return None


def _create_viewer(
    title: str, mask: xr.DataArray, img: Optional[xr.DataArray] = None
) -> tuple[Viewer, Labels, Optional[list[Image]]]:
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
                img_name = img.coords["c"].to_numpy()[::-1]
            img_channel_axis = img.dims.index("c")
        img_layers = viewer.add_image(
            data=img_data,
            channel_axis=img_channel_axis,
            rgb=False,
            colormap="gray",
            name=img_name,
            scale=img.attrs["scale"],
            translate=-0.5 * np.array(img.shape[1:]),
            visible=False,
        )
    mask_layer = viewer.add_labels(
        data=mask,
        name=mask.name,
        scale=mask.attrs["scale"],
        translate=-0.5 * np.array(mask.shape),
    )
    mask_layer.contour = 1
    mask_layer.selected_label = 0
    mask_layer.show_selected_label = True
    return viewer, mask_layer, img_layers


class SpellmatchInteractiveRegistrationException(SpellmatchRegistrationException):
    pass
