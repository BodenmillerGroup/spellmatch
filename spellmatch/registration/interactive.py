from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from napari.layers import Image, Labels
from napari.viewer import Viewer
from qtpy.QtCore import QModelIndex, QPoint, Qt
from qtpy.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLineEdit,
    QMenu,
    QStyle,
    QTableView,
    QVBoxLayout,
    QWidget,
)
from skimage.measure import regionprops
from skimage.transform import AffineTransform

from ..qt import QPandasTableModel


class QMaskRegistrationDialog(QDialog):
    def __init__(
        self,
        source_centroids: dict[int, Tuple[float, float]],
        target_centroids: dict[int, Tuple[float, float]],
        label_pairs: Optional[pd.DataFrame] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super(QMaskRegistrationDialog, self).__init__(parent=parent)
        self._source_centroids = source_centroids
        self._target_centroids = target_centroids
        self._transform = None
        layout = QVBoxLayout()
        self.setLayout(layout)
        self.setWindowTitle("Mask registration")
        self._label_pairs_table_model = QPandasTableModel(columns=["Source", "Target"])
        if label_pairs is not None:
            self._label_pairs_table_model.table = label_pairs
        self._label_pairs_table_model.dataChanged.connect(
            lambda top_left, bottom_right, roles: self.update_transform()
        )
        self._label_pairs_table_model.rowsInserted.connect(
            lambda parent, first, last: self.update_transform()
        )
        self._label_pairs_table_model.rowsRemoved.connect(
            lambda parent, first, last: self.update_transform()
        )
        self._label_pairs_table_model.modelReset.connect(
            lambda: self.update_transform()
        )
        self._label_pairs_table_view = QTableView(parent=self)
        self._label_pairs_table_view.setModel(self._label_pairs_table_model)
        self._label_pairs_table_view.setSelectionBehavior(
            QTableView.SelectionBehavior.SelectRows
        )
        self._label_pairs_table_view.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu
        )
        self._label_pairs_table_view.customContextMenuRequested.connect(
            self._on_label_pairs_table_view_custom_context_menu_requested
        )
        layout.addWidget(self._label_pairs_table_view)
        status_form_layout = QFormLayout()
        self._transform_rmsd_line_edit = QLineEdit(parent=self)
        self._transform_rmsd_line_edit.setReadOnly(True)
        status_form_layout.addRow("Transform RMSD:", self._transform_rmsd_line_edit)
        layout.addLayout(status_form_layout)
        self._button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        self._button_box.button(QDialogButtonBox.StandardButton.Save).setEnabled(False)
        self._button_box.accepted.connect(self.accept)
        self._button_box.rejected.connect(self.reject)
        layout.addWidget(self._button_box)

    def keyPressEvent(self, event) -> None:
        if event.key() not in (Qt.Key.Key_Enter, Qt.Key.Key_Return):
            super(QMaskRegistrationDialog, self).keyPressEvent(event)

    def append_label_pair(self, source_label: int, target_label: int) -> None:
        self._label_pairs_table_model.append(
            pd.DataFrame(data=[[source_label, target_label]])
        )

    def update_transform(self) -> None:
        src = np.array(
            [self._source_centroids[label] for label in self.label_pairs.iloc[:, 0]]
        )
        dst = np.array(
            [self._target_centroids[label] for label in self.label_pairs.iloc[:, 1]]
        )
        transform = AffineTransform()
        if transform.estimate(src, dst):
            self._transform = transform
            rmsd = np.mean(transform.residuals(src, dst) ** 2) ** 0.5
            self._transform_rmsd_line_edit.setText(f"{rmsd:.6f}")
        else:
            self._transform = None
            self._transform_rmsd_line_edit.setText("")
        self._button_box.button(QDialogButtonBox.StandardButton.Save).setEnabled(
            self._transform is not None
        )

    def _on_label_pairs_table_model_data_changed(
        self,
        topLeft: QModelIndex,
        bottomRight: QModelIndex,
        roles: Sequence[Qt.ItemDataRole],
    ) -> None:
        self.update_transform()

    def _on_label_pairs_table_view_custom_context_menu_requested(
        self, pos: QPoint
    ) -> None:
        index = self._label_pairs_table_view.indexAt(pos)
        if index.isValid():
            menu = QMenu(parent=self._label_pairs_table_view)
            del_action = menu.addAction(
                menu.style().standardIcon(QStyle.StandardPixmap.SP_DialogCloseButton),
                "Delete",
            )
            if menu.exec(self._label_pairs_table_view.mapToGlobal(pos)) == del_action:
                self._label_pairs_table_model.removeRow(index.row(), QModelIndex())

    @property
    def label_pairs(self) -> pd.DataFrame:
        return self._label_pairs_table_model.table

    @property
    def transform(self) -> Optional[AffineTransform]:
        return self._transform


def register_masks(
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    source_img: Optional[xr.DataArray] = None,
    target_img: Optional[xr.DataArray] = None,
    label_pairs: Optional[pd.DataFrame] = None,
) -> Optional[Tuple[pd.DataFrame, AffineTransform]]:
    app = QApplication([])

    source_viewer, source_mask_layer, _ = _create_viewer(
        "Source", source_mask, img=source_img
    )
    target_viewer, target_mask_layer, _ = _create_viewer(
        "Target", target_mask, img=target_img
    )
    status_dialog = QMaskRegistrationDialog(
        {rp.label: rp.centroid for rp in regionprops(source_mask)},
        {rp.label: rp.centroid for rp in regionprops(target_mask)},
        label_pairs=label_pairs,
    )

    def on_layer_mouse_drag(mask_layer: Labels, event) -> None:
        selected_label = mask_layer.get_value(event.position, world=True)
        if selected_label != 0:
            mask_layer.metadata["selected_label"] = selected_label
            source_label = source_mask_layer.metadata.get("selected_label")
            target_label = target_mask_layer.metadata.get("selected_label")
            if source_label is not None and target_label is not None:
                status_dialog.append_label_pair(source_label, target_label)
                source_mask_layer.metadata["selected_label"] = None
                target_mask_layer.metadata["selected_label"] = None
        else:
            mask_layer.metadata["selected_label"] = None

    source_mask_layer.mouse_drag_callbacks.append(on_layer_mouse_drag)
    target_mask_layer.mouse_drag_callbacks.append(on_layer_mouse_drag)
    status_dialog.finished.connect(lambda _: app.exit())

    source_viewer.show()
    target_viewer.show()
    status_dialog.show()
    status_dialog.raise_()
    status_dialog.activateWindow()

    app.exec()
    if status_dialog.result() == QMaskRegistrationDialog.DialogCode.Accepted:
        return status_dialog.label_pairs, status_dialog.transform
    return None


def _create_viewer(
    title: str, mask: np.ndarray, img: Optional[xr.DataArray] = None
) -> Tuple[Viewer, Labels, Optional[list[Image]]]:
    if img is not None:
        title += f": {img.name}"
    viewer = Viewer(title=title, axis_labels=("y", "x"), show=False)
    img_layers = None
    if img is not None:
        channel_names = None
        if "c" in img.coords:
            channel_names = img.coords["c"].values.tolist()
        img_layers = viewer.add_image(
            data=img.data[::-1],
            channel_axis=0,
            rgb=False,
            colormap="gray",
            name=channel_names[::-1],
            visible=False,
        )
    mask_layer = viewer.add_labels(data=mask)
    return viewer, mask_layer, img_layers
