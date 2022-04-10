from typing import Optional

import numpy as np
import pandas as pd
from qtpy.QtCore import QModelIndex, QPoint, Qt
from qtpy.QtWidgets import (
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
from skimage.transform import AffineTransform, ProjectiveTransform

from ._pandas_table_model import QPandasTableModel


class QInteractiveRegistrationDialog(QDialog):
    def __init__(
        self,
        source_points: pd.DataFrame,
        target_points: pd.DataFrame,
        transform_type: type[ProjectiveTransform] = AffineTransform,
        label_pairs_columns: tuple[str, str] = ("Source", "Target"),
        parent: Optional[QWidget] = None,
    ) -> None:
        super(QInteractiveRegistrationDialog, self).__init__(parent=parent)
        self._source_points = source_points
        self._target_points = target_points
        self._transform_type = transform_type
        self._transform = None
        layout = QVBoxLayout()
        self.setLayout(layout)
        self.setWindowTitle("Interactive registration")
        self._label_pairs_table_model = QPandasTableModel(columns=label_pairs_columns)
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

    def append_label_pair(self, source_label: int, target_label: int) -> None:
        self._label_pairs_table_model.append(
            pd.DataFrame(data=[[source_label, target_label]])
        )

    def update_transform(self) -> None:
        transform = self._transform_type()
        src = np.array(
            [
                self._source_points.loc[label, :].to_numpy()
                for label in self.label_pairs.iloc[:, 0]
            ]
        )
        dst = np.array(
            [
                self._target_points.loc[label, :].to_numpy()
                for label in self.label_pairs.iloc[:, 1]
            ]
        )
        if len(src) > 0 and len(dst) > 0 and transform.estimate(src, dst):
            self._transform = transform
            transform_rmsd = np.mean(transform.residuals(src, dst) ** 2) ** 0.5
            self._transform_rmsd_line_edit.setText(f"{transform_rmsd:.6f}")
        else:
            self._transform = None
            self._transform_rmsd_line_edit.setText("")
        self._button_box.button(QDialogButtonBox.StandardButton.Save).setEnabled(
            self._transform is not None
        )

    def keyPressEvent(self, event) -> None:
        if event.key() not in (Qt.Key.Key_Enter, Qt.Key.Key_Return):
            super(QInteractiveRegistrationDialog, self).keyPressEvent(event)

    def _on_label_pairs_table_view_custom_context_menu_requested(
        self, pos: QPoint
    ) -> None:
        index = self._label_pairs_table_view.indexAt(pos)
        if index.isValid():
            menu = QMenu(parent=self._label_pairs_table_view)
            global_pos = self._label_pairs_table_view.mapToGlobal(pos)
            delete_action = menu.addAction(
                menu.style().standardIcon(QStyle.StandardPixmap.SP_DialogCloseButton),
                "Delete",
            )
            if menu.exec(global_pos) == delete_action:
                self._label_pairs_table_model.removeRow(index.row(), QModelIndex())

    @property
    def label_pairs(self) -> pd.DataFrame:
        return self._label_pairs_table_model.table

    @label_pairs.setter
    def label_pairs(self, label_pairs: pd.DataFrame) -> None:
        self._label_pairs_table_model.table = label_pairs

    @property
    def transform(self) -> Optional[ProjectiveTransform]:
        return self._transform
