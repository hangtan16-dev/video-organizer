"""
Collections manager dialog.

A "collection" is a named saved filter preset (filter text + sort key/direction).
Collections appear as quick-access items in the folder tree sidebar.

This dialog lets the user:
- View all saved collections
- Create new collections from the current filter state
- Edit collection names and filter strings
- Delete collections
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QPushButton, QLabel, QLineEdit, QDialogButtonBox, QInputDialog,
    QMessageBox, QSplitter, QGroupBox, QFormLayout
)
from PyQt6.QtCore import Qt, pyqtSignal


class CollectionsDialog(QDialog):
    """Manage named filter collections."""

    # Emitted when the user clicks Apply or double-clicks a collection.
    # Payload: {'name': str, 'filter_text': str,
    #           'sort_key': str, 'sort_asc': bool}
    collection_selected = pyqtSignal(dict)

    def __init__(self, cache, current_filter: str = '',
                 current_sort_key: str = 'name', current_sort_asc: bool = True,
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle("Collections")
        self.setMinimumSize(480, 360)

        self._cache            = cache
        self._current_filter   = current_filter
        self._current_sort_key = current_sort_key
        self._current_sort_asc = current_sort_asc

        self.setStyleSheet("""
            QDialog { background: #1e1e1e; color: #ddd; }
            QLabel { color: #ddd; }
            QListWidget {
                background: #252525; color: #ddd;
                border: 1px solid #3a3a3a; border-radius: 3px;
            }
            QListWidget::item:selected { background: #3a6fc4; color: white; }
            QListWidget::item:hover    { background: #2e2e2e; }
            QLineEdit {
                background: #2e2e2e; color: #ddd;
                border: 1px solid #3a3a3a; border-radius: 3px; padding: 4px;
            }
            QPushButton {
                background: #2e2e2e; color: #ddd;
                border: 1px solid #3a3a3a; border-radius: 3px; padding: 4px 10px;
            }
            QPushButton:hover { background: #3a3a3a; }
            QPushButton:disabled { color: #555; border-color: #2a2a2a; }
            QGroupBox {
                color: #ccc; border: 1px solid #3a3a3a;
                border-radius: 4px; margin-top: 8px; padding-top: 4px;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; color: #888; }
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # ── Current filter info ───────────────────────────────────────────
        if current_filter:
            info = QLabel(f'Current filter: "{current_filter}"')
            info.setStyleSheet("color: #888; font-style: italic;")
            layout.addWidget(info)

        # ── Save current filter as new collection ─────────────────────────
        save_group = QGroupBox("Save current filter as collection")
        save_layout = QHBoxLayout(save_group)
        save_layout.setContentsMargins(8, 4, 8, 8)
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("Collection name…")
        self._name_edit.returnPressed.connect(self._save_current)
        save_btn = QPushButton("Save")
        save_btn.setStyleSheet(
            "QPushButton { background: #3a6fc4; color: white; border: none; "
            "border-radius: 3px; padding: 4px 12px; }"
            "QPushButton:hover { background: #4a7fd4; }"
        )
        save_btn.clicked.connect(self._save_current)
        save_layout.addWidget(self._name_edit)
        save_layout.addWidget(save_btn)
        layout.addWidget(save_group)

        # ── Saved collections list ────────────────────────────────────────
        layout.addWidget(QLabel("Saved collections:"))
        self._list = QListWidget()
        self._list.itemDoubleClicked.connect(self._apply_selected)
        layout.addWidget(self._list)

        # ── Action buttons ────────────────────────────────────────────────
        btn_row = QHBoxLayout()

        apply_btn = QPushButton("Apply")
        apply_btn.setStyleSheet(
            "QPushButton { background: #3a6fc4; color: white; border: none; "
            "border-radius: 3px; padding: 4px 12px; }"
            "QPushButton:hover { background: #4a7fd4; }"
        )
        apply_btn.clicked.connect(self._apply_selected)

        rename_btn = QPushButton("Rename…")
        rename_btn.clicked.connect(self._rename_selected)

        delete_btn = QPushButton("Delete")
        delete_btn.clicked.connect(self._delete_selected)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)

        btn_row.addWidget(apply_btn)
        btn_row.addWidget(rename_btn)
        btn_row.addWidget(delete_btn)
        btn_row.addStretch()
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

        self._reload_list()

    # ── List management ───────────────────────────────────────────────────

    def _reload_list(self):
        self._list.clear()
        collections = []
        if hasattr(self._cache, 'get_collections'):
            try:
                collections = self._cache.get_collections() or []
            except Exception:
                pass

        for col in collections:
            filter_text = col.get('filter_text', '')
            sort_key    = col.get('sort_key', 'name')
            sort_dir    = '↑' if col.get('sort_asc', True) else '↓'
            label = f"★  {col['name']}"
            if filter_text:
                label += f'  —  "{filter_text}"'
            label += f"  [{sort_key} {sort_dir}]"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, col)
            self._list.addItem(item)

        if not collections:
            placeholder = QListWidgetItem("No collections saved yet.")
            placeholder.setFlags(Qt.ItemFlag.NoItemFlags)
            placeholder.setForeground(
                self._list.palette().color(self._list.foregroundRole()).darker(200)
            )
            self._list.addItem(placeholder)

    def _current_collection(self):
        """Return the data dict for the currently selected item, or None."""
        item = self._list.currentItem()
        if not item:
            return None
        return item.data(Qt.ItemDataRole.UserRole)

    # ── Slot implementations ──────────────────────────────────────────────

    def _save_current(self):
        name = self._name_edit.text().strip()
        if not name:
            self._name_edit.setFocus()
            return
        if hasattr(self._cache, 'save_collection'):
            try:
                self._cache.save_collection(
                    name,
                    self._current_filter,
                    self._current_sort_key,
                    self._current_sort_asc,
                )
            except Exception as exc:
                QMessageBox.warning(self, "Save failed", str(exc))
                return
        self._name_edit.clear()
        self._reload_list()

    def _apply_selected(self):
        col = self._current_collection()
        if not col:
            return
        self.collection_selected.emit(col)
        self.accept()

    def _rename_selected(self):
        col = self._current_collection()
        if not col:
            return
        new_name, ok = QInputDialog.getText(
            self, "Rename collection",
            "New name:", text=col.get('name', '')
        )
        if not ok or not new_name.strip():
            return
        new_name = new_name.strip()
        if hasattr(self._cache, 'rename_collection'):
            try:
                self._cache.rename_collection(col['name'], new_name)
            except Exception as exc:
                QMessageBox.warning(self, "Rename failed", str(exc))
                return
        elif hasattr(self._cache, 'delete_collection') and hasattr(self._cache, 'save_collection'):
            # Fallback: delete + re-save with the new name
            try:
                self._cache.delete_collection(col['name'])
                self._cache.save_collection(
                    new_name,
                    col.get('filter_text', ''),
                    col.get('sort_key', 'name'),
                    col.get('sort_asc', True),
                )
            except Exception as exc:
                QMessageBox.warning(self, "Rename failed", str(exc))
                return
        self._reload_list()

    def _delete_selected(self):
        col = self._current_collection()
        if not col:
            return
        reply = QMessageBox.question(
            self, "Delete collection",
            f"Delete collection \"{col['name']}\"?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        if hasattr(self._cache, 'delete_collection'):
            try:
                self._cache.delete_collection(col['name'])
            except Exception as exc:
                QMessageBox.warning(self, "Delete failed", str(exc))
                return
        self._reload_list()
