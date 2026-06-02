"""
Batch rename dialog.

Template variables:
  {name}       — filename without extension
  {ext}        — file extension including dot  (e.g. .mp4)
  {index:03d}  — 1-based index, zero-padded to 3 digits (or any width)

Usage:
    dlg = BatchRenameDialog(paths, parent=window)
    dlg.renames_done.connect(handler)   # receives list of old paths
    dlg.exec()
"""

import os
import re

# Windows-reserved filename characters and reserved device names
_INVALID_CHARS = '<>:"/\\|?*'
_RESERVED_NAMES = frozenset({
    'CON', 'PRN', 'AUX', 'NUL',
    *(f'COM{i}' for i in range(1, 10)),
    *(f'LPT{i}' for i in range(1, 10)),
})


def _validate_filename(name: str) -> str:
    """Return error message describing why name is invalid, or '' if valid."""
    if not name or not name.strip():
        return "name is empty"
    if len(name) > 255:
        return "name exceeds 255 characters"
    bad = [c for c in _INVALID_CHARS if c in name]
    if bad:
        return f"name contains invalid characters: {''.join(set(bad))}"
    # Trailing dots/spaces are illegal on Windows
    if name.endswith(' ') or name.endswith('.'):
        return "name cannot end with a space or dot"
    # Reserved device names (case-insensitive, with or without extension)
    stem = os.path.splitext(name)[0].upper()
    if stem in _RESERVED_NAMES:
        return f"'{stem}' is a reserved Windows name"
    return ''


from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QTableWidget, QTableWidgetItem, QHeaderView,
    QDialogButtonBox, QMessageBox,
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer


class BatchRenameDialog(QDialog):
    renames_done = pyqtSignal(list)   # list of original paths that were renamed

    def __init__(self, paths: list, parent=None):
        super().__init__(parent)
        self._paths = list(paths)
        self.setWindowTitle("Batch Rename")
        self.setMinimumSize(640, 420)
        self.setStyleSheet("""
            QDialog { background: #1e1e1e; color: #ddd; }
            QLabel  { color: #ddd; }
            QLineEdit { background: #2e2e2e; color: #ddd; border: 1px solid #3a3a3a;
                        border-radius: 3px; padding: 4px 6px; }
            QTableWidget { background: #2e2e2e; color: #ccc; gridline-color: #3a3a3a;
                           border: 1px solid #3a3a3a; }
            QHeaderView::section { background: #252525; color: #aaa;
                                   border: none; padding: 4px; }
            QPushButton { background: #3a3a3a; color: #ddd; border: 1px solid #555;
                          border-radius: 3px; padding: 4px 12px; min-width: 70px; }
            QPushButton:hover { background: #484848; }
            QPushButton:disabled { color: #666; background: #2a2a2a; }
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # Template row
        tmpl_row = QHBoxLayout()
        tmpl_row.addWidget(QLabel("Template:"))
        self._tmpl_edit = QLineEdit("{name}{ext}")
        self._tmpl_edit.setPlaceholderText("{name}{ext}")
        tmpl_row.addWidget(self._tmpl_edit)
        layout.addLayout(tmpl_row)

        # Hint label
        hint = QLabel(
            "Variables:  {name}  {ext}  {index:03d}\n"
            "Example:  Movie_{index:03d}{ext}  →  Movie_001.mp4"
        )
        hint.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(hint)

        # Preview table
        self._table = QTableWidget(len(self._paths), 2, self)
        self._table.setHorizontalHeaderLabels(["Original", "New Name"])
        self._table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        self._table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        layout.addWidget(self._table, 1)

        # Buttons
        btn_row = QHBoxLayout()
        self._btn_rename = QPushButton("Rename")
        self._btn_rename.setStyleSheet(
            "QPushButton { background: #3a6fc4; color: #fff; border: none; }"
            "QPushButton:hover { background: #4a7fd4; }"
            "QPushButton:disabled { background: #2a4a84; color: #888; }"
        )
        self._btn_rename.clicked.connect(self._do_rename)
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.reject)
        btn_row.addStretch()
        btn_row.addWidget(self._btn_rename)
        btn_row.addWidget(btn_close)
        layout.addLayout(btn_row)

        # Debounce template changes
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(200)
        self._preview_timer.timeout.connect(self._refresh_preview)

        self._tmpl_edit.textChanged.connect(lambda _: self._preview_timer.start())
        self._refresh_preview()

    # ── preview ────────────────────────────────────────────────────────────────
    def _apply_template(self, template: str, path: str, index: int) -> str:
        """Apply template string to one path. Returns new filename (not full path)."""
        name, ext = os.path.splitext(os.path.basename(path))
        # Replace {index:Nd} style patterns
        def _replace_index(m):
            fmt = m.group(1) or 'd'
            try:
                return format(index, fmt)
            except (ValueError, TypeError):
                return str(index)

        result = re.sub(r'\{index(?::([^}]*))?\}', _replace_index, template)
        result = result.replace('{name}', name)
        result = result.replace('{ext}', ext)
        return result

    def _compute_new_names(self) -> list[str]:
        """Return list of new filenames for each path (with conflict resolution)."""
        template = self._tmpl_edit.text() or '{name}{ext}'
        new_names: list[str] = []
        used: dict[str, int] = {}  # filename -> count already used
        for i, path in enumerate(self._paths, start=1):
            base = self._apply_template(template, path, i)
            # Handle empty result
            if not base:
                base = os.path.basename(path)
            # Ensure extension present if template produced none and original had one
            _, orig_ext = os.path.splitext(path)
            _, new_ext  = os.path.splitext(base)
            if orig_ext and not new_ext:
                base += orig_ext

            # Conflict resolution: append _2, _3 if same name already used
            candidate = base
            count = used.get(base.lower(), 0)
            if count > 0:
                root, ext = os.path.splitext(base)
                candidate = f"{root}_{count + 1}{ext}"
            used[base.lower()] = used.get(base.lower(), 0) + 1
            new_names.append(candidate)
        return new_names

    def _refresh_preview(self):
        new_names = self._compute_new_names()
        any_invalid = False
        for row, (path, new_name) in enumerate(zip(self._paths, new_names)):
            orig_item = QTableWidgetItem(os.path.basename(path))
            orig_item.setForeground(Qt.GlobalColor.gray)
            new_item  = QTableWidgetItem(new_name)
            is_same = (os.path.basename(path) == new_name)

            err = _validate_filename(new_name)
            if err:
                any_invalid = True
                new_item.setForeground(Qt.GlobalColor.red)
                new_item.setToolTip(f"Invalid: {err}")
            else:
                new_item.setForeground(
                    Qt.GlobalColor.gray if is_same else Qt.GlobalColor.white
                )
            self._table.setItem(row, 0, orig_item)
            self._table.setItem(row, 1, new_item)

        # Enable rename only if at least one name changes AND none are invalid
        has_change = any(os.path.basename(p) != n
                         for p, n in zip(self._paths, new_names))
        self._btn_rename.setEnabled(has_change and not any_invalid)

    # ── rename ─────────────────────────────────────────────────────────────────
    def _do_rename(self):
        new_names = self._compute_new_names()
        renamed: list[str] = []
        errors:  list[str] = []

        for path, new_name in zip(self._paths, new_names):
            if os.path.basename(path) == new_name:
                continue  # nothing to do
            err = _validate_filename(new_name)
            if err:
                errors.append(f"{os.path.basename(path)} → {new_name}: {err}")
                continue
            parent = os.path.dirname(path)
            dest   = os.path.join(parent, new_name)
            try:
                os.rename(path, dest)
                renamed.append(path)
            except OSError as exc:
                errors.append(f"{os.path.basename(path)}: {exc}")

        if errors:
            QMessageBox.warning(
                self, "Rename errors",
                "Some files could not be renamed:\n" + "\n".join(errors)
            )

        if renamed:
            self.renames_done.emit(renamed)

        if not errors:
            self.accept()
