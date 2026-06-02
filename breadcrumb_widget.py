"""
Breadcrumb navigation widget.

Shows the current folder path as clickable path segments separated by ›.
"""

import os

from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QPushButton, QLabel, QScrollArea, QSizePolicy,
)
from PyQt6.QtCore import Qt, pyqtSignal


class BreadcrumbWidget(QWidget):
    navigate_requested = pyqtSignal(str)   # emitted when a segment is clicked
    bookmark_requested = pyqtSignal(str)   # emitted when pin button is clicked

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_path = ''
        self._is_bookmarked = False

        self.setFixedHeight(30)
        self.setStyleSheet("background: #252525;")

        outer = QHBoxLayout(self)
        outer.setContentsMargins(4, 2, 4, 2)
        outer.setSpacing(0)

        # Scrollable area for the crumb buttons
        self._scroll = QScrollArea(self)
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setStyleSheet(
            "QScrollArea { background: transparent; border: none; }"
        )
        self._scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        self._crumb_widget = QWidget()
        self._crumb_widget.setStyleSheet("background: transparent;")
        self._crumb_layout = QHBoxLayout(self._crumb_widget)
        self._crumb_layout.setContentsMargins(0, 0, 0, 0)
        self._crumb_layout.setSpacing(0)
        self._crumb_layout.addStretch()

        self._scroll.setWidget(self._crumb_widget)
        outer.addWidget(self._scroll)

    def set_path(self, path: str):
        """Rebuild breadcrumb segments from path."""
        self._current_path = path
        self._rebuild_crumbs(path)

    def set_bookmarked(self, bookmarked: bool):
        """Kept for API compatibility. The pin/bookmark button was removed, so
        there is no visual to update — just record the state."""
        self._is_bookmarked = bookmarked

    def _rebuild_crumbs(self, path: str):
        """Remove existing crumb buttons and add new ones for the given path."""
        # Clear existing widgets
        while self._crumb_layout.count() > 1:  # keep the trailing stretch
            item = self._crumb_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not path:
            return

        path = os.path.normpath(path)

        # Build list of (label, full_path) segments
        segments: list[tuple[str, str]] = []
        if os.name == 'nt':
            drive, rest = os.path.splitdrive(path)
            drive_root = drive + '\\'
            segments.append((drive_root, drive_root))
            parts = [p for p in rest.split(os.sep) if p]
            accumulated = drive_root
            for part in parts:
                accumulated = os.path.join(accumulated, part)
                segments.append((part, accumulated))
        else:
            parts = [p for p in path.split('/') if p]
            segments.append(('/', '/'))
            accumulated = '/'
            for part in parts:
                accumulated = os.path.join(accumulated, part)
                segments.append((part, accumulated))

        btn_style = (
            "QPushButton { background: transparent; border: none; color: #aaa; "
            "              padding: 2px 4px; font-size: 12px; }"
            "QPushButton:hover { color: #fff; text-decoration: underline; }"
        )
        sep_style = "QLabel { color: #555; background: transparent; padding: 0 2px; }"

        insert_pos = 0
        for i, (label, full_path) in enumerate(segments):
            btn = QPushButton(label, self._crumb_widget)
            btn.setStyleSheet(btn_style)
            btn.setFlat(True)
            btn.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Preferred)
            # Capture full_path in closure
            _fp = full_path
            btn.clicked.connect(lambda checked=False, fp=_fp: self.navigate_requested.emit(fp))
            self._crumb_layout.insertWidget(insert_pos, btn)
            insert_pos += 1

            if i < len(segments) - 1:
                sep = QLabel("›", self._crumb_widget)
                sep.setStyleSheet(sep_style)
                self._crumb_layout.insertWidget(insert_pos, sep)
                insert_pos += 1

        # Scroll to the end to show the deepest segment
        self._scroll.horizontalScrollBar().setValue(
            self._scroll.horizontalScrollBar().maximum()
        )
