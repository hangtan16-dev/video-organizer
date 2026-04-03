"""
Tag editor dialog — lets the user view and edit tags for one or more videos.

Features
────────
• Shows all existing tags as clickable chips (click to remove)
• Text field to add new tags (Enter or comma to confirm)
• "Common tags" quick-add buttons populated from CacheManager.get_all_tags()
• OK saves, Cancel discards
• Supports editing tags for multiple files at once (shows union, marks common tags)
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton,
    QLabel, QScrollArea, QWidget, QDialogButtonBox, QFrame
)
from PyQt6.QtCore import Qt


_PREDEFINED_TAGS = ["Watched", "Favorite", "4K", "HDR", "To Watch", "Archive"]


class _FlowWidget(QWidget):
    """Simple widget that lays out children left-to-right with wrapping."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._buttons = []

    def add_chip(self, text, on_click=None, color='#3a5080'):
        btn = QPushButton(text, self)
        btn.setStyleSheet(
            f"QPushButton {{ background: {color}; color: white; border: none; "
            f"border-radius: 10px; padding: 2px 8px; font-size: 10px; }}"
            f"QPushButton:hover {{ background: #4a7fd4; }}"
        )
        if on_click:
            btn.clicked.connect(on_click)
        self._buttons.append(btn)
        btn.show()
        self._relayout()
        return btn

    def clear_chips(self):
        for btn in self._buttons:
            btn.deleteLater()
        self._buttons.clear()
        self.setFixedHeight(40)

    def _relayout(self):
        x, y, row_h = 4, 4, 0
        max_w = self.width() if self.width() > 10 else 400
        for btn in self._buttons:
            btn.adjustSize()
            bw = btn.sizeHint().width() + 4
            bh = btn.sizeHint().height() + 4
            if x + bw > max_w and x > 4:
                x = 4
                y += row_h + 4
                row_h = 0
            btn.setGeometry(x, y, bw - 4, bh - 4)
            x += bw
            row_h = max(row_h, bh)
        self.setFixedHeight(max(y + row_h + 8, 40))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._relayout()


class TagDialog(QDialog):
    """Edit tags for one or more video files."""

    def __init__(self, video_paths: list, cache, parent=None):
        super().__init__(parent)
        self._paths = list(video_paths)
        self._cache = cache

        n = len(self._paths)
        title = "Edit Tags" if n == 1 else f"Edit Tags — {n} files"
        self.setWindowTitle(title)
        self.setMinimumWidth(480)
        self.setMinimumHeight(340)

        self.setStyleSheet("""
            QDialog        { background: #1e1e1e; color: #ddd; }
            QLabel         { color: #ddd; }
            QLineEdit      { background: #2e2e2e; color: #ddd; border: 1px solid #3a3a3a;
                             border-radius: 3px; padding: 4px; }
            QPushButton    { background: #2e2e2e; color: #ddd; border: 1px solid #3a3a3a;
                             border-radius: 3px; padding: 4px 10px; }
            QPushButton:hover { background: #3a3a3a; }
            QScrollArea    { background: #252525; border: 1px solid #3a3a3a; border-radius: 3px; }
            QDialogButtonBox QPushButton { padding: 4px 16px; }
        """)

        # Collect current tags (union across all selected files)
        self._current_tags: set = set()
        for path in self._paths:
            existing = cache.get_tags(path) if hasattr(cache, 'get_tags') else []
            self._current_tags.update(existing)

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # ── Current tags ──────────────────────────────────────────────────
        layout.addWidget(QLabel("Current tags:"))

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedHeight(90)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self._chip_container = _FlowWidget()
        self._chip_container.setStyleSheet("background: #252525;")
        scroll.setWidget(self._chip_container)
        layout.addWidget(scroll)

        # ── Separator ─────────────────────────────────────────────────────
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #3a3a3a;")
        layout.addWidget(sep)

        # ── Add tag ───────────────────────────────────────────────────────
        layout.addWidget(QLabel("Add tag:"))
        add_row = QHBoxLayout()
        self._tag_input = QLineEdit()
        self._tag_input.setPlaceholderText("Type tag and press Enter…")
        self._tag_input.returnPressed.connect(self._add_tag)
        add_btn = QPushButton("Add")
        add_btn.setStyleSheet(
            "QPushButton { background: #3a6fc4; color: white; border: none; "
            "border-radius: 3px; padding: 4px 12px; }"
            "QPushButton:hover { background: #4a7fd4; }"
        )
        add_btn.clicked.connect(self._add_tag)
        add_row.addWidget(self._tag_input)
        add_row.addWidget(add_btn)
        layout.addLayout(add_row)

        # ── Quick add ─────────────────────────────────────────────────────
        layout.addWidget(QLabel("Quick add:"))
        self._quick_container = _FlowWidget()
        self._quick_container.setStyleSheet("background: #1e1e1e;")
        self._quick_container.setMinimumHeight(40)
        layout.addWidget(self._quick_container)

        # ── OK / Cancel ───────────────────────────────────────────────────
        btn_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btn_box.button(QDialogButtonBox.StandardButton.Ok).setStyleSheet(
            "QPushButton { background: #3a6fc4; color: white; border: none; "
            "border-radius: 3px; padding: 4px 16px; }"
        )
        btn_box.accepted.connect(self._save)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

        self._populate_quick_tags()
        self._refresh_chips()

    # ── Internal helpers ──────────────────────────────────────────────────

    def _populate_quick_tags(self):
        """Populate quick-add buttons from cache or fall back to predefined list."""
        all_tags = []
        if hasattr(self._cache, 'get_all_tags'):
            try:
                all_tags = self._cache.get_all_tags() or []
            except Exception:
                pass
        if not all_tags:
            all_tags = _PREDEFINED_TAGS

        for tag in all_tags[:20]:
            t = tag  # capture
            self._quick_container.add_chip(
                tag,
                on_click=lambda checked, tg=t: self._quick_add(tg),
                color='#2e2e2e'
            )
        # Override the text color for quick tags to #888
        for btn in self._quick_container._buttons:
            btn.setStyleSheet(
                "QPushButton { background: #2e2e2e; color: #888; border: 1px solid #3a3a3a; "
                "border-radius: 10px; padding: 2px 8px; font-size: 10px; }"
                "QPushButton:hover { background: #3a3a3a; color: #ddd; }"
            )

    def _refresh_chips(self):
        """Rebuild the current-tags chip display."""
        self._chip_container.clear_chips()
        for tag in sorted(self._current_tags):
            t = tag  # capture
            self._chip_container.add_chip(
                f"{tag}  ×",
                on_click=lambda checked, tg=t: self._remove_tag(tg),
                color='#3a5080'
            )

    def _add_tag(self):
        text = self._tag_input.text()
        parts = [p.strip() for p in text.split(',')]
        for part in parts:
            if part:
                self._current_tags.add(part)
        self._tag_input.clear()
        self._refresh_chips()

    def _quick_add(self, tag: str):
        self._current_tags.add(tag)
        self._refresh_chips()

    def _remove_tag(self, tag: str):
        self._current_tags.discard(tag)
        self._refresh_chips()

    def _save(self):
        tag_list = sorted(self._current_tags)
        for path in self._paths:
            if hasattr(self._cache, 'set_tags'):
                try:
                    self._cache.set_tags(path, tag_list)
                except Exception:
                    pass
        self.accept()
