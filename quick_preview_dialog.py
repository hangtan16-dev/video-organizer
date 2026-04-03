"""
Quick preview overlay — full-screen thumbnail view with arrow key navigation.

Triggered by Spacebar in the grid. Shows the thumbnail at maximum size
with the filename, rating, and basic metadata. Arrow keys navigate between
items. Escape or Space closes.
"""

import os
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QPushButton, QWidget
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QPixmap, QFont, QColor, QPalette


def _fmt_duration(secs):
    """Format seconds as H:MM:SS or M:SS."""
    if not secs:
        return ''
    s = int(secs)
    h, r = divmod(s, 3600)
    m, sec = divmod(r, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


class QuickPreviewDialog(QDialog):
    """
    Full-window image viewer. Shows video thumbnail at maximum size.
    Left/Right arrows navigate between items.
    Escape or Space closes.
    """

    def __init__(self, items: list, start_idx: int, parent=None):
        """
        Parameters
        ----------
        items : list of dict
            Each dict: {'path': str, 'pixmap': QPixmap|None,
                        'rating': int, 'duration': float}
        start_idx : int
            Index of the initially shown item.
        """
        super().__init__(parent)
        self._items = items
        self._idx = max(0, min(start_idx, len(items) - 1))

        # Window setup
        self.setWindowTitle("Quick Preview")
        self.setWindowState(Qt.WindowState.WindowMaximized)
        self.setStyleSheet("QDialog { background: #000; }")

        # Make sure keyboard events reach us
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # ── Image label (fills most of the window) ────────────────────────
        self._img_label = QLabel(self)
        self._img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._img_label.setStyleSheet("background: transparent;")

        # ── Counter label (top-right) ─────────────────────────────────────
        self._counter_label = QLabel(self)
        self._counter_label.setStyleSheet(
            "color: white; background: rgba(0,0,0,160); "
            "border-radius: 4px; padding: 4px 8px; font-size: 13px;"
        )
        self._counter_label.setAlignment(Qt.AlignmentFlag.AlignRight)

        # ── Bottom info bar ───────────────────────────────────────────────
        self._info_bar = QWidget(self)
        self._info_bar.setStyleSheet("background: rgba(0,0,0,180); border-radius: 6px;")

        info_layout = QVBoxLayout(self._info_bar)
        info_layout.setContentsMargins(12, 6, 12, 6)
        info_layout.setSpacing(2)

        self._filename_label = QLabel(self._info_bar)
        self._filename_label.setStyleSheet("color: white; font-size: 15px; font-weight: bold;")
        self._filename_label.setAlignment(Qt.AlignmentFlag.AlignLeft)

        self._meta_label = QLabel(self._info_bar)
        self._meta_label.setStyleSheet("color: #aaa; font-size: 12px;")
        self._meta_label.setAlignment(Qt.AlignmentFlag.AlignLeft)

        self._rating_label = QLabel(self._info_bar)
        self._rating_label.setStyleSheet("color: #f5c542; font-size: 16px;")
        self._rating_label.setAlignment(Qt.AlignmentFlag.AlignLeft)

        self._hint_label = QLabel("Press  ←  →  to navigate   •   Esc  to close",
                                   self._info_bar)
        self._hint_label.setStyleSheet("color: #555; font-size: 10px;")
        self._hint_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        info_layout.addWidget(self._filename_label)
        info_layout.addWidget(self._rating_label)
        info_layout.addWidget(self._meta_label)
        info_layout.addWidget(self._hint_label)

        # ── Navigation buttons ────────────────────────────────────────────
        nav_style = (
            "QPushButton { background: rgba(255,255,255,40); color: white; "
            "border: none; border-radius: 4px; font-size: 32px; } "
            "QPushButton:hover { background: rgba(255,255,255,80); }"
        )

        self._btn_prev = QPushButton("‹", self)
        self._btn_prev.setFixedSize(48, 80)
        self._btn_prev.setStyleSheet(nav_style)
        self._btn_prev.clicked.connect(self._go_prev)

        self._btn_next = QPushButton("›", self)
        self._btn_next.setFixedSize(48, 80)
        self._btn_next.setStyleSheet(nav_style)
        self._btn_next.clicked.connect(self._go_next)

        # Initial display
        self._show_item(self._idx)

    # ── Layout ────────────────────────────────────────────────────────────

    def resizeEvent(self, event):
        super().resizeEvent(event)
        w, h = self.width(), self.height()

        # Image fills the whole dialog
        self._img_label.setGeometry(0, 0, w, h)

        # Counter: top-right corner
        self._counter_label.adjustSize()
        cw = self._counter_label.width() + 16
        self._counter_label.setGeometry(w - cw - 8, 8, cw, 28)

        # Info bar: bottom, full width with 12px margin
        bar_h = 90
        self._info_bar.setGeometry(12, h - bar_h - 12, w - 24, bar_h)

        # Nav buttons: vertically centered on left and right edges
        mid_y = h // 2 - 40
        self._btn_prev.move(8, mid_y)
        self._btn_next.move(w - 56, mid_y)

        # Refresh scaled image for new size
        self._update_image()

    # ── Navigation ────────────────────────────────────────────────────────

    def _go_prev(self):
        if self._idx > 0:
            self._show_item(self._idx - 1)

    def _go_next(self):
        if self._idx < len(self._items) - 1:
            self._show_item(self._idx + 1)

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key.Key_Left:
            self._go_prev()
        elif key == Qt.Key.Key_Right:
            self._go_next()
        elif key in (Qt.Key.Key_Escape, Qt.Key.Key_Space):
            self.accept()
        else:
            super().keyPressEvent(event)

    # ── Display ───────────────────────────────────────────────────────────

    def _show_item(self, idx: int):
        self._idx = idx
        item = self._items[idx]
        self._current_pixmap = item.get('pixmap')

        # Counter
        self._counter_label.setText(f"{idx + 1} / {len(self._items)}")
        self._counter_label.adjustSize()
        # Reposition counter after text change
        w = self.width()
        cw = self._counter_label.width() + 16
        self._counter_label.setGeometry(w - cw - 8, 8, cw, 28)

        # Filename
        path = item.get('path', '')
        self._filename_label.setText(os.path.basename(path))

        # Rating
        rating = item.get('rating', 0)
        stars = '★' * rating + '☆' * (5 - rating)
        self._rating_label.setText(stars if rating else '')
        self._rating_label.setVisible(bool(rating))

        # Duration / meta
        dur = item.get('duration', 0)
        dur_str = _fmt_duration(dur)
        self._meta_label.setText(dur_str if dur_str else '')

        # Nav button visibility
        self._btn_prev.setVisible(idx > 0)
        self._btn_next.setVisible(idx < len(self._items) - 1)

        self._update_image()

    def _update_image(self):
        """Scale the current pixmap to fit the available area."""
        pixmap = self._current_pixmap if hasattr(self, '_current_pixmap') else None
        w = max(self.width(), 100)
        h = max(self.height() - 120, 100)   # leave room for info bar

        if pixmap and not pixmap.isNull():
            scaled = pixmap.scaled(
                QSize(w, h),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self._img_label.setPixmap(scaled)
        else:
            self._img_label.clear()
            self._img_label.setText("No thumbnail available")
            self._img_label.setStyleSheet(
                "color: #555; font-size: 18px; background: transparent;"
            )
