"""
Folder tile widget for the thumbnail grid.

Shows a system folder icon, folder name, and item count.
Exposes the same public API as VideoThumbnailWidget so ThumbnailGridWidget
can treat both uniformly:  set_checked / is_checked / cleanup / signals.

Signals
───────
checked_changed    (path: str, checked: bool)
delete_requested   (path: str)
move_requested     (path: str)
rename_requested   (path: str)
copy_requested     (path: str)
navigate_requested (path: str)   double-click or context-menu "Open folder"
"""

import datetime
import os

from PyQt6.QtWidgets import (
    QFrame, QLabel, QCheckBox, QMenu, QApplication, QStyle,
)
from PyQt6.QtCore import Qt, pyqtSignal, QPoint
from PyQt6.QtGui import QFont

# Re-use the same bottom-bar height as VideoThumbnailWidget so mixed rows
# (folders + videos) all have a consistent height.
from video_thumbnail_widget import BOTTOM_H

_CB_SIZE    = 22
_CB_MARGIN  = 5
_ICON_MAX   = 80      # max folder-icon size in px before scaling to fit cell


class FolderThumbnailWidget(QFrame):
    checked_changed    = pyqtSignal(str, bool)   # path, is_checked
    delete_requested   = pyqtSignal(str)
    move_requested     = pyqtSignal(str)
    rename_requested   = pyqtSignal(str)
    copy_requested     = pyqtSignal(str)
    navigate_requested = pyqtSignal(str)          # open / double-click

    def __init__(self, folder_path: str, child_count: int = 0,
                 video_count: int = 0, parent=None):
        super().__init__(parent)
        self.folder_path  = folder_path
        self._child_count = child_count
        self._video_count = video_count

        has_videos = video_count > 0

        self.setFrameShape(QFrame.Shape.Box)
        self.setLineWidth(1)
        self.setMouseTracking(True)

        sm_font = QFont()
        sm_font.setPointSize(8)

        # ── icon area ─────────────────────────────────────────────────────────
        # Background: slightly lighter when videos are present so the folder
        # "pops"; darker/greyed when the folder has no video files.
        icon_bg = "#2a2a2a" if has_videos else "#1e1e1e"
        self._icon_area = QLabel(self)
        self._icon_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._icon_area.setStyleSheet(f"background: {icon_bg};")

        # Grab the platform folder icon once; cache the full-size pixmap.
        # Use the open-folder icon when videos are present for extra clarity.
        style    = QApplication.style()
        sp       = (QStyle.StandardPixmap.SP_DirOpenIcon if has_videos
                    else QStyle.StandardPixmap.SP_DirIcon)
        icon     = style.standardIcon(sp)
        self._src_pixmap     = icon.pixmap(_ICON_MAX, _ICON_MAX)
        self._last_icon_size = (-1, -1)   # (w, h) of last scaled version

        # ── bottom bar ────────────────────────────────────────────────────────
        name = os.path.basename(folder_path)

        # Video-count badge shown in the bottom-right corner.
        # Green when videos are present; muted orange-red when none.
        if has_videos:
            count_text  = f"🎬 {video_count}"
            count_style = "color: #6abf69; background: #111; padding-right: 4px;"
        else:
            count_text  = "no videos"
            count_style = "color: #b06040; background: #111; padding-right: 4px;"

        self._name_label = QLabel(f"📁  {name}", self)
        self._name_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        self._name_label.setStyleSheet(
            "color: #ccc; background: #111; padding-left: 4px;"
        )
        self._name_label.setFont(sm_font)
        self._name_label.setToolTip(folder_path)

        self._count_label = QLabel(count_text, self)
        self._count_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self._count_label.setStyleSheet(count_style)
        self._count_label.setFont(sm_font)

        # ── Feature 7: meta label (size + date modified) ──────────────────────
        self._meta_label = QLabel("", self)
        self._meta_label.setFont(sm_font)
        self._meta_label.setStyleSheet(
            "color: #aaa; background: rgba(0,0,0,160); padding-left: 4px; font-size: 7px;"
        )
        self._meta_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        self._meta_label.hide()

        # ── Feature 19: NFO badge ─────────────────────────────────────────────
        self._nfo_badge = QLabel("NFO", self)
        self._nfo_badge.setStyleSheet(
            "background: rgba(180,120,60,200); color: white; "
            "border-radius: 3px; padding: 1px 3px; font-size: 9px;"
        )
        self._nfo_badge.hide()

        # ── checkbox (top-right, identical position to VideoThumbnailWidget) ──
        self._checkbox = QCheckBox(self)
        self._checkbox.setStyleSheet(
            "QCheckBox::indicator { width: 18px; height: 18px; }"
            "QCheckBox { background: rgba(0,0,0,140); border-radius: 3px; }"
        )
        self._checkbox.stateChanged.connect(self._on_check_changed)
        # Feature 3: connect checkbox state change to frame style update
        self._checkbox.stateChanged.connect(lambda state: self._update_frame_style())

        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)

        # Apply initial frame style (Feature 3)
        self._update_frame_style()

    # ── public API (mirrors VideoThumbnailWidget) ───────────────────────────────
    def set_checked(self, checked: bool):
        self._checkbox.setChecked(checked)
        self._update_frame_style()   # Feature 3

    def is_checked(self) -> bool:
        return self._checkbox.isChecked()

    def cleanup(self):
        """No-op — folders have no background threads."""
        pass

    # ── Feature 3: selection highlight ring ───────────────────────────────────
    def _update_frame_style(self):
        if self._checkbox.isChecked():
            self.setStyleSheet("QFrame { border: 2px solid #3a6fc4; border-radius: 2px; }")
        else:
            self.setStyleSheet("QFrame { border: 1px solid #3a3a3a; border-radius: 2px; }")

    # ── Feature 7: folder size and date modified ───────────────────────────────
    def set_folder_size(self, size_bytes: int):
        self._folder_size_bytes = size_bytes
        self._update_meta_label()

    def set_mtime(self, mtime: float):
        self._mtime = mtime
        self._update_meta_label()

    def _update_meta_label(self):
        parts = []
        if hasattr(self, '_folder_size_bytes') and self._folder_size_bytes >= 0:
            parts.append(self._human_size(self._folder_size_bytes))
        if hasattr(self, '_mtime') and self._mtime > 0:
            dt = datetime.datetime.fromtimestamp(self._mtime)
            parts.append(dt.strftime('%Y-%m-%d'))
        self._meta_label.setText('  '.join(parts))
        self._meta_label.setVisible(bool(parts))

    @staticmethod
    def _human_size(n: int) -> str:
        for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
            if n < 1024:
                return f"{n:.1f} {unit}" if unit != 'B' else f"{n} B"
            n /= 1024
        return f"{n:.1f} PB"

    # ── Feature 19: NFO badge ─────────────────────────────────────────────────
    def set_has_nfo(self, has_nfo: bool):
        self._nfo_badge.setVisible(has_nfo)

    # ── Feature 22: focus ring ─────────────────────────────────────────────────
    def set_focused(self, focused: bool):
        if focused:
            self.setStyleSheet("QFrame { border: 2px solid #5a9fd4; border-radius: 2px; }")
        else:
            self._update_frame_style()

    # ── layout ─────────────────────────────────────────────────────────────────
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._relayout()

    def _relayout(self):
        w, h    = self.width(), self.height()
        icon_h  = max(1, h - BOTTOM_H)
        name_y  = icon_h

        # Icon area fills everything above the bottom bar
        self._icon_area.setGeometry(0, 0, w, icon_h)

        # Scale folder icon to fit, with a comfortable margin; cache the result
        max_iw = max(1, w - 24)
        max_ih = max(1, icon_h - 16)
        if (max_iw, max_ih) != self._last_icon_size:
            scaled = self._src_pixmap.scaled(
                min(max_iw, _ICON_MAX * 2), min(max_ih, _ICON_MAX * 2),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self._icon_area.setPixmap(scaled)
            self._last_icon_size = (max_iw, max_ih)

        # Bottom bar: name on left (65 %), count on right (35 %)
        name_w = max(10, int(w * 0.65))
        self._name_label.setGeometry(0,      name_y, name_w,     BOTTOM_H)
        self._count_label.setGeometry(name_w, name_y, w - name_w, BOTTOM_H)

        # Feature 7: meta label — semi-transparent overlay at bottom of icon area
        self._meta_label.setGeometry(4, icon_h - 18, w - 8, 16)

        # Feature 19: NFO badge — top-left of icon area
        self._nfo_badge.setGeometry(4, 4, 30, 14)
        self._nfo_badge.raise_()

        # Checkbox — top-right corner
        self._checkbox.setGeometry(
            w - _CB_SIZE - _CB_MARGIN, _CB_MARGIN, _CB_SIZE, _CB_SIZE
        )
        self._checkbox.raise_()

    # ── mouse interaction ───────────────────────────────────────────────────────
    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.navigate_requested.emit(self.folder_path)
        super().mouseDoubleClickEvent(event)

    def mousePressEvent(self, event):
        """Single click on the icon area toggles the checkbox."""
        if event.button() == Qt.MouseButton.LeftButton:
            if event.position().y() < self.height() - BOTTOM_H:
                self._checkbox.setChecked(not self._checkbox.isChecked())
        super().mousePressEvent(event)

    def _on_check_changed(self, state: int):
        self.checked_changed.emit(
            self.folder_path,
            state == Qt.CheckState.Checked.value,
        )

    def _show_context_menu(self, pos: QPoint):
        menu     = QMenu(self)
        act_open = menu.addAction("Open folder")
        menu.addSeparator()
        act_rename = menu.addAction("Rename…")
        act_copy   = menu.addAction("Copy to…")
        act_move   = menu.addAction("Move to…")
        act_del    = menu.addAction("Delete")
        action = menu.exec(self.mapToGlobal(pos))
        if action == act_open:
            self.navigate_requested.emit(self.folder_path)
        elif action == act_rename:
            self.rename_requested.emit(self.folder_path)
        elif action == act_copy:
            self.copy_requested.emit(self.folder_path)
        elif action == act_move:
            self.move_requested.emit(self.folder_path)
        elif action == act_del:
            self.delete_requested.emit(self.folder_path)
