from typing import Optional, Sequence

import pandas as pd
from qtpy.QtCore import QAbstractTableModel, QModelIndex, QObject, Qt


class QPandasTableModel(QAbstractTableModel):
    def __init__(
        self,
        columns: Optional[Sequence[str]] = None,
        editable: bool = False,
        parent: Optional[QObject] = None,
    ) -> None:
        super(QPandasTableModel, self).__init__(parent)
        self._table = pd.DataFrame(columns=columns)
        self._editable = editable

    def rowCount(self, parent: QModelIndex) -> int:
        if parent.isValid():
            return 0
        return len(self._table.index)

    def columnCount(self, parent: QModelIndex) -> int:
        if parent.isValid():
            return 0
        return len(self._table.columns)

    def data(self, index: QModelIndex, role: Qt.ItemDataRole) -> object:
        if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole):
            if 0 <= index.row() < len(self._table.index) and 0 <= index.column() < len(
                self._table.columns
            ):
                return self._table.values[index.row(), index.column()].item()
        return None

    def headerData(
        self, section: int, orientation: Qt.Orientation, role: Qt.ItemDataRole
    ) -> object:
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal and 0 <= section < len(
                self._table.columns
            ):
                return self._table.columns[section]
            if orientation == Qt.Orientation.Vertical and 0 <= section < len(
                self._table.index
            ):
                return self._table.index[section]
        return None

    def setData(self, index: QModelIndex, value: object, role: Qt.ItemDataRole) -> bool:
        if (
            role == Qt.ItemDataRole.EditRole
            and 0 <= index.row() < len(self._table.index)
            and 0 <= index.column() < len(self._table.columns)
        ):
            self._table.iloc[index.row(), index.column()] = value
            self.dataChanged.emit(index, index, [role])
            return True
        return False

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        flags = super(QPandasTableModel, self).flags(index)
        if self._editable:
            flags |= Qt.ItemFlag.ItemIsEditable
        return flags

    def insertRows(self, row: int, count: int, parent: QModelIndex) -> bool:
        if parent.isValid():
            return False
        objs = None
        df = pd.DataFrame(index=range(count), columns=self._table.columns)
        if row == 0:
            objs = (df, self._table)
        elif 0 < row < len(self._table.index):
            objs = (self._table.iloc[:row], df, self._table.iloc[row:])
        elif row == len(self._table.index):
            objs = (self._table, df)
        if objs is not None:
            self.beginInsertRows(parent, row, row + count - 1)
            self._table = pd.concat(objs, ignore_index=True)
            self.endInsertRows()
            return True
        return False

    def removeRows(self, row: int, count: int, parent: QModelIndex) -> bool:
        if parent.isValid():
            return False
        if 0 <= row < row + count <= len(self._table.index):
            self.beginRemoveRows(parent, row, row + count - 1)
            self._table.drop(index=self._table.index[row : row + count], inplace=True)
            self._table.reset_index(drop=True, inplace=True)
            self.endRemoveRows()
            return True
        return False

    def append(self, table: pd.DataFrame) -> None:
        row = self.rowCount(QModelIndex())
        self.beginInsertRows(QModelIndex(), row, row + len(table.index) - 1)
        table = pd.DataFrame(data=table.values, columns=self._table.columns)
        self._table = pd.concat((self._table, table), ignore_index=True)
        self.endInsertRows()

    @property
    def table(self) -> pd.DataFrame:
        return self._table

    @table.setter
    def table(self, table: pd.DataFrame) -> None:
        self.beginResetModel()
        self._table = pd.DataFrame(data=table.values, columns=self._table.columns)
        self.endResetModel()
