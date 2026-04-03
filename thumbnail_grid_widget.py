"""
Scrollable grid of VideoThumbnailWidget instances — VIRTUAL SCROLLING.

Performance improvements over v2
──────────────────────────────────
• Virtual scrolling: only widgets inside the visible viewport ± _BUFFER_PX
  are alive as Qt objects.  For 1 000+ videos ≈ 20-40 live widgets at any
  time instead of 1 000+.
• All file/video state lives in lightweight _Item dataclasses (no Qt objects).
• _layout_cache pre-computes every item's (x, y, w, h) over plain Python
  tuples; scroll-range queries are O(n) without touching any widget.
• Batch loading (30 items per event-loop tick) so the main thread never
  stalls during folder scan.
• Thumbnail generation is queued for ALL items during batch load, not just
  visible ones, so off-screen videos are still processed in the background.
"""

import os
from dataclasses import dataclass, field

from PyQt6.QtWidgets import QScrollArea, QWidget, QLabel, QApplication
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QMimeData, QUrl
from PyQt6.QtGui import QImage, QPixmap, QKeyEvent, QFont, QDrag

from video_thumbnail_widget import VideoThumbnailWidget, BOTTOM_H
from folder_thumbnail_widget import FolderThumbnailWidget
from thumbnail_generator import ThumbnailGeneratorService
from app_settings import AppSettings
from cache_manager import CacheManager

_SPACING    = 8
_BATCH_SIZE = 30      # items added per event-loop tick during folder load
_BUFFER_PX  = 600     # pixels above/below visible viewport to keep widgets alive

# Sentinel for filtered-out items in layout cache
_FILTERED_SENTINEL = (-1, -1, 0, 0)


def _count_videos_deep(path: str, exts: frozenset, max_depth: int,
                       _depth: int = 1) -> int:
    """Recursively count video files up to max_depth levels below path."""
    count = 0
    try:
        for e in os.scandir(path):
            if e.name.startswith('.'):
                continue
            if e.is_file() and os.path.splitext(e.name)[1].lower() in exts:
                count += 1
            elif e.is_dir(follow_symlinks=False) and _depth < max_depth:
                count += _count_videos_deep(e.path, exts, max_depth, _depth + 1)
    except (PermissionError, OSError):
        pass
    return count


def _scan_folder(path: str) -> 'tuple[int, int]':
    """Return (direct_child_count, video_count_up_to_3_levels_deep)."""
    try:
        entries = [e for e in os.scandir(path) if not e.name.startswith('.')]
        total   = len(entries)
    except (PermissionError, OSError):
        return 0, 0
    exts   = frozenset(AppSettings.VIDEO_EXTENSIONS)
    videos = _count_videos_deep(path, exts, max_depth=3)
    return total, videos


# ── lightweight data record (no Qt objects in constructor) ─────────────────────
@dataclass
class _Item:
    path:         str
    is_folder:    bool   = False
    child_count:  int    = 0       # meaningful only when is_folder=True
    video_count:  int    = 0       # video files directly inside (folders only)
    seek_time:    float  = 0.0
    aspect_ratio: float  = 16 / 9
    duration:     float  = 0.0
    pixmap:       object = None    # QPixmap once thumbnail is ready; else None
    checked:      bool   = False
    mtime:        float  = 0.0     # os.stat().st_mtime — used for sort
    size:         int    = 0       # os.stat().st_size  — used for sort (0 for folders)
    rating:       int    = 0       # 0-5 star rating
    tags:         list   = field(default_factory=list)
    is_watched:   bool   = False
    nfo_exists:   bool   = False
    sub_exists:   bool   = False
    folder_size:  int    = -1      # -1 = not yet scanned
    is_focused:   bool   = False


# ══════════════════════════════════════════════════════════════════════════════
class ThumbnailGridWidget(QScrollArea):
    selection_changed      = pyqtSignal(int)   # emits number of checked items
    status_message         = pyqtSignal(str)   # emits status-bar text
    navigate_requested     = pyqtSignal(str)   # user double-clicked / opened a subfolder
    delete_key_pressed     = pyqtSignal()      # Delete key pressed
    undo_requested         = pyqtSignal()      # Ctrl+Z pressed
    rename_key_pressed     = pyqtSignal()      # F2 pressed
    rename_requested       = pyqtSignal(str)   # rename context menu on a widget
    copy_requested         = pyqtSignal(str)   # copy context menu on a widget
    open_requested         = pyqtSignal(str)   # open a video file (double-click or context menu)
    quick_preview_requested = pyqtSignal(int)  # spacebar on focused item — emits item index
    files_dropped          = pyqtSignal(list)  # files/folders dropped onto the grid
    watch_toggled          = pyqtSignal(str, bool)  # (path, watched) re-emitted from widget

    def __init__(self, generator: ThumbnailGeneratorService,
                 settings: AppSettings,
                 cache: CacheManager,
                 parent=None):
        super().__init__(parent)
        self._generator = generator
        self._settings  = settings
        self._cache     = cache

        # ── sort state (kept in sync with AppSettings) ──────────────────────
        self._sort_key: str  = settings.sort_key   # 'name'|'modified'|'size'|'type'|'rating'
        self._sort_asc: bool = settings.sort_asc

        # ── filter state ─────────────────────────────────────────────────────
        self._filter_text: str = ''

        # ── virtual-scroll data model ────────────────────────────────────────
        self._items:        list[_Item]                    = []
        self._path_to_idx:  dict[str, int]                 = {}
        # (x, y, w, h) geometry for every item index; (-1,-1,0,0) for filtered-out
        self._layout_cache: list[tuple[int, int, int, int]] = []
        # Only items whose geometry overlaps the buffered viewport are alive here
        self._active:       dict[int, object]  = {}   # VideoThumbnailWidget | FolderThumbnailWidget

        # Batch-load state
        self._pending_items: list[_Item] = []
        self._batch_idx:     int         = 0
        self._total_count:   int         = 0

        # ── keyboard focus navigation ────────────────────────────────────────
        self._focused_idx: int = -1

        # ── scroll position memory ────────────────────────────────────────────
        self._folder_scroll:       dict[str, int] = {}
        self._current_folder_path: str            = ''

        # ── drag start position (drag-from support) ──────────────────────────
        self._drag_start_pos = None

        # ── virtual canvas ───────────────────────────────────────────────────
        self._container = QWidget()
        self._container.setStyleSheet("background: #1e1e1e;")
        self.setWidget(self._container)
        # We manage the container size ourselves (setWidgetResizable(False))
        self.setWidgetResizable(False)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setStyleSheet("QScrollArea { background: #1e1e1e; border: none; }")

        # Focus policy for keyboard shortcuts
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # ── empty-state overlay label ─────────────────────────────────────────
        self._empty_label = QLabel(self._container)
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setStyleSheet("color: #555; background: transparent;")
        self._empty_label.setWordWrap(True)
        font = QFont()
        font.setPointSize(13)
        self._empty_label.setFont(font)
        self._empty_label.hide()

        # ── drag and drop ─────────────────────────────────────────────────────
        self.setAcceptDrops(True)

        # ── signal connections ───────────────────────────────────────────────
        self._generator.thumbnail_ready.connect(self._on_thumbnail_ready)
        self.verticalScrollBar().valueChanged.connect(self._on_scroll)

        # Debounce resize (60 ms) and scroll-end (60 ms) before calling layout
        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.timeout.connect(self._full_relayout)

        self._scroll_timer = QTimer(self)
        self._scroll_timer.setSingleShot(True)
        self._scroll_timer.timeout.connect(self._update_active_widgets)

    # ── public API ─────────────────────────────────────────────────────────────
    def load_folder(self, folder_path: str):
        """Start loading subfolders then video files from folder_path (non-blocking)."""
        # Save scroll position for the folder we're leaving
        if self._current_folder_path and self._items:
            self._folder_scroll[self._current_folder_path] = self.verticalScrollBar().value()

        self._current_folder_path = folder_path

        self._generator.cancel_all()
        self._clear_all()

        if not folder_path or not os.path.isdir(folder_path):
            self.status_message.emit("No folder selected")
            return

        exts = AppSettings.VIDEO_EXTENSIONS
        try:
            all_entries = list(os.scandir(folder_path))
        except PermissionError:
            self.status_message.emit(f"Permission denied: {folder_path}")
            return

        raw_dirs = [
            e for e in all_entries
            if e.is_dir(follow_symlinks=False) and not e.name.startswith('.')
        ]
        raw_files = [
            e for e in all_entries
            if e.is_file() and os.path.splitext(e.name)[1].lower() in exts
        ]

        pending: list[_Item] = []
        for d in raw_dirs:
            total, videos = _scan_folder(d.path)
            try:
                st = d.stat()
                mtime = st.st_mtime
            except OSError:
                mtime = 0.0
            pending.append(_Item(path=d.path, is_folder=True,
                                 child_count=total, video_count=videos,
                                 mtime=mtime))
        for f in raw_files:
            seek = self._cache.get_seek_override(f.path) or self._settings.seek_time
            try:
                st = f.stat()
                mtime, size = st.st_mtime, st.st_size
            except OSError:
                mtime, size = 0.0, 0
            rating = self._cache.get_rating(f.path)
            pending.append(_Item(path=f.path, seek_time=seek,
                                 mtime=mtime, size=size, rating=rating))

        # Load watched/tag/sidecar state for file items
        for item in pending:
            if not item.is_folder:
                item.is_watched = self._cache.is_watched(item.path)
                item.tags = self._cache.get_tags(item.path)
                base = os.path.splitext(item.path)[0]
                item.sub_exists = any(
                    os.path.exists(base + ext)
                    for ext in ('.srt', '.sub', '.ass', '.vtt', '.smi')
                )
                item.nfo_exists = os.path.exists(base + '.nfo')

        pending = self._apply_sort(pending)

        self._pending_items = pending
        self._total_count   = len(pending)

        if not pending:
            self.status_message.emit("Empty folder")
            self._empty_label.setText(
                "📂\n\nNo videos in this folder\n\nTry a parent folder or clear the filter"
            )
            vp_w = self.viewport().width()
            vp_h = self.viewport().height()
            self._empty_label.setGeometry(0, 0, vp_w, vp_h)
            self._empty_label.show()
            return

        # Items exist — ensure empty label is hidden
        self._empty_label.hide()

        n_dirs, n_files = len(raw_dirs), len(raw_files)
        parts = []
        if n_dirs:  parts.append(f"{n_dirs} folder{'s' if n_dirs != 1 else ''}")
        if n_files: parts.append(f"{n_files} video{'s' if n_files != 1 else ''}")
        self.status_message.emit(f"Loading {', '.join(parts)}…")
        self.selection_changed.emit(0)
        QTimer.singleShot(0, lambda: self._create_next_batch(folder_path))

    def set_filter(self, text: str):
        """Filter visible items to those whose name contains text (case-insensitive)."""
        self._filter_text = text.lower().strip()
        self._full_relayout()
        self.status_message.emit(self._items_summary())

    def set_sort(self, key: str, asc: bool):
        """
        Change the sort order and immediately re-sort the displayed items.

        Folders are always grouped before files; within each group the chosen
        key is applied.  Runs entirely on the main thread — no disk I/O.
        """
        if self._sort_key == key and self._sort_asc == asc:
            return
        self._sort_key = key
        self._sort_asc = asc

        if not self._items and not self._pending_items:
            return

        # Flush live widget state into _Item records before destroying widgets
        self._sync_active_state()

        # Tear down all active widgets (their grid positions are about to change).
        # cleanup() must be called first to stop timers and playback threads before
        # the widget is unparented — otherwise timers fire on a parentless widget.
        for w in self._active.values():
            if hasattr(w, 'cleanup'):
                w.cleanup()
            w.setParent(None)
            w.deleteLater()
        self._active.clear()

        # Re-sort already-loaded items and any not-yet-displayed pending items
        self._items        = self._apply_sort(self._items)
        self._pending_items = self._apply_sort(self._pending_items)

        self._rebuild_index()
        self._full_relayout()

    # ── public state-update methods ────────────────────────────────────────────
    def update_item_watched(self, path: str, is_watched: bool):
        """Called when user marks a video as watched/unwatched."""
        idx = self._path_to_idx.get(path)
        if idx is not None:
            self._items[idx].is_watched = is_watched
            w = self._active.get(idx)
            if w and hasattr(w, 'set_watched'):
                w.set_watched(is_watched)

    def update_item_tags(self, path: str, tags: list):
        """Called when user edits tags."""
        idx = self._path_to_idx.get(path)
        if idx is not None:
            self._items[idx].tags = tags
            w = self._active.get(idx)
            if w and hasattr(w, 'set_tags'):
                w.set_tags(tags)

    def update_folder_size(self, path: str, size_bytes: int):
        """Called by FolderSizeScanner."""
        idx = self._path_to_idx.get(path)
        if idx is not None:
            self._items[idx].folder_size = size_bytes
            w = self._active.get(idx)
            if w and hasattr(w, 'set_folder_size'):
                w.set_folder_size(size_bytes)

    def get_all_items(self) -> list:
        """Return a copy of all _Item records (for export)."""
        return list(self._items)

    def get_video_items(self) -> list:
        """Return only non-folder _Item records."""
        return [i for i in self._items if not i.is_folder]

    # ── helpers ─────────────────────────────────────────────────────────────────
    def _apply_sort(self, items: 'list[_Item]') -> 'list[_Item]':
        """
        Return a new sorted list with folders first, then files.
        Each group is sorted by self._sort_key / self._sort_asc.
        """
        key  = self._sort_key
        desc = not self._sort_asc

        def _key(item: _Item):
            if key == 'modified':
                return item.mtime
            if key == 'size':
                return item.size
            if key == 'type':
                ext  = os.path.splitext(item.path)[1].lower()
                name = os.path.basename(item.path).lower()
                return (ext, name)
            if key == 'rating':
                return item.rating
            # default: name
            return os.path.basename(item.path).lower()

        folders = sorted([i for i in items if     i.is_folder], key=_key, reverse=desc)
        files   = sorted([i for i in items if not i.is_folder], key=_key, reverse=desc)
        return folders + files

    def get_checked_paths(self) -> list[str]:
        """Return paths of all checked items (syncs live widget state first)."""
        self._sync_active_state()
        return [item.path for item in self._items if item.checked]

    def get_all_video_paths(self) -> list[str]:
        """Return all video (non-folder) paths currently loaded."""
        return [item.path for item in self._items if not item.is_folder]

    def select_all(self):
        for item in self._items:
            item.checked = True
        for w in self._active.values():
            w.set_checked(True)
        self.selection_changed.emit(len(self._items))

    def deselect_all(self):
        for item in self._items:
            item.checked = False
        for w in self._active.values():
            w.set_checked(False)
        self.selection_changed.emit(0)

    def remove_paths(self, paths: list[str]):
        """Remove videos from the grid (called after successful delete/move)."""
        self._sync_active_state()
        path_set = set(paths)

        # Destroy every live widget — indices will shift after list rebuild
        for w in self._active.values():
            w.cleanup()
            w.setParent(None)
            w.deleteLater()
        self._active.clear()

        # Rebuild the items list and index
        self._items = [item for item in self._items if item.path not in path_set]
        self._rebuild_index()
        self._full_relayout()

        self.status_message.emit(self._items_summary())
        self.selection_changed.emit(len(self.get_checked_paths()))

    def set_thumbnails_per_row(self, n: int):
        self._settings.thumbnails_per_row = n
        self._full_relayout()

    # ── keyboard shortcuts ─────────────────────────────────────────────────────
    def keyPressEvent(self, event: QKeyEvent):
        key  = event.key()
        mods = event.modifiers()

        if key == Qt.Key.Key_Delete:
            self.delete_key_pressed.emit()
        elif key == Qt.Key.Key_A and mods & Qt.KeyboardModifier.ControlModifier:
            self.select_all()
        elif key == Qt.Key.Key_Escape:
            self.deselect_all()
        elif key == Qt.Key.Key_Z and mods & Qt.KeyboardModifier.ControlModifier:
            self.undo_requested.emit()
        elif key == Qt.Key.Key_F2:
            self.rename_key_pressed.emit()
        elif key == Qt.Key.Key_Right:
            self._move_focus(1)
        elif key == Qt.Key.Key_Left:
            self._move_focus(-1)
        elif key == Qt.Key.Key_Down:
            self._move_focus(self._settings.thumbnails_per_row)
        elif key == Qt.Key.Key_Up:
            self._move_focus(-self._settings.thumbnails_per_row)
        elif key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._activate_focused()
        elif key == Qt.Key.Key_Space:
            self._preview_focused()
        else:
            super().keyPressEvent(event)

    # ── keyboard navigation helpers ────────────────────────────────────────────
    def _move_focus(self, delta: int):
        """Move keyboard focus by delta items, skipping filtered-out ones."""
        if not self._items:
            return
        # Find valid (non-filtered) indices
        valid = [i for i, g in enumerate(self._layout_cache) if g != _FILTERED_SENTINEL]
        if not valid:
            return
        if self._focused_idx not in valid:
            self._focused_idx = valid[0]
        else:
            pos = valid.index(self._focused_idx)
            pos = max(0, min(len(valid) - 1, pos + delta))
            self._focused_idx = valid[pos]
        # Scroll to make focused item visible
        self._scroll_to_focused()
        self._update_active_widgets()  # repaint focus ring

    def _scroll_to_focused(self):
        if self._focused_idx < 0 or self._focused_idx >= len(self._layout_cache):
            return
        g = self._layout_cache[self._focused_idx]
        if g == _FILTERED_SENTINEL:
            return
        x, y, w, h = g
        sv   = self.verticalScrollBar().value()
        vp_h = self.viewport().height()
        if y < sv:
            self.verticalScrollBar().setValue(y - 8)
        elif y + h > sv + vp_h:
            self.verticalScrollBar().setValue(y + h - vp_h + 8)

    def _activate_focused(self):
        """Enter/navigate focused item."""
        if self._focused_idx < 0:
            return
        item = self._items[self._focused_idx]
        if item.is_folder:
            self.navigate_requested.emit(item.path)
        else:
            self.open_requested.emit(item.path)

    def _preview_focused(self):
        """Spacebar: emit quick preview signal for focused item."""
        if self._focused_idx < 0 or self._focused_idx >= len(self._items):
            return
        item = self._items[self._focused_idx]
        if not item.is_folder:
            self.quick_preview_requested.emit(self._focused_idx)

    # ── drag and drop ──────────────────────────────────────────────────────────
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            paths = [url.toLocalFile() for url in event.mimeData().urls()]
            self.files_dropped.emit(paths)
            event.acceptProposedAction()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start_pos = event.position().toPoint()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.MouseButton.LeftButton and
                self._drag_start_pos is not None):
            dist = (event.position().toPoint() - self._drag_start_pos).manhattanLength()
            if dist > QApplication.startDragDistance():
                self._start_drag()
        super().mouseMoveEvent(event)

    def _start_drag(self):
        checked = self.get_checked_paths()
        if not checked:
            return
        drag = QDrag(self)
        mime = QMimeData()
        urls = [QUrl.fromLocalFile(p) for p in checked]
        mime.setUrls(urls)
        drag.setMimeData(mime)
        drag.exec(Qt.DropAction.CopyAction | Qt.DropAction.MoveAction)

    # ── filter helper ─────────────────────────────────────────────────────────
    def _item_matches_filter(self, item: '_Item') -> bool:
        """Return True if the item should be SHOWN (not filtered out)."""
        text = self._filter_text.lower()
        if not text:
            return True
        name = os.path.basename(item.path).lower()
        # tag: prefix searches tags
        if text.startswith('tag:'):
            tag_query = text[4:].strip()
            return tag_query in [t.lower() for t in item.tags]
        # watched: filter
        if text == 'watched':
            return item.is_watched
        if text in ('unwatch', 'unwatched'):
            return not item.is_watched
        # rated: filter
        if text == 'rated':
            return item.rating > 0
        if text == 'unrated':
            return item.rating == 0
        # Default: filename substring
        return text in name

    # ── batched item creation ───────────────────────────────────────────────────
    def _create_next_batch(self, folder_path: str):
        """Add up to _BATCH_SIZE items, schedule next batch, yield to event loop."""
        batch = self._pending_items[self._batch_idx: self._batch_idx + _BATCH_SIZE]
        if not batch:
            return

        for item in batch:
            self._path_to_idx[item.path] = len(self._items)
            self._items.append(item)
            # Queue thumbnail generation for video files only
            if not item.is_folder:
                self._generator.request_thumbnail(item.path, item.seek_time)

        self._batch_idx += _BATCH_SIZE
        # Recompute layout + create/destroy widgets for current viewport
        self._full_relayout()

        if self._batch_idx < len(self._pending_items):
            QTimer.singleShot(0, lambda: self._create_next_batch(folder_path))
        else:
            self.status_message.emit(self._items_summary())
            # Restore saved scroll position for this folder
            saved = self._folder_scroll.get(folder_path, 0)
            if saved > 0:
                QTimer.singleShot(50, lambda: self.verticalScrollBar().setValue(saved))

    # ── layout ─────────────────────────────────────────────────────────────────
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._resize_timer.start(60)

    def _is_item_filtered(self, item: '_Item') -> bool:
        """Return True if this item should be hidden by the current filter."""
        return not self._item_matches_filter(item)

    def _full_relayout(self):
        """Recompute every item's geometry, resize the canvas, refresh widgets."""
        n = max(1, self._settings.thumbnails_per_row)
        vp_w = self.viewport().width()
        vp_h = self.viewport().height()
        cell_w = max(80, (vp_w - _SPACING * (n + 1)) // n)

        cache: list[tuple[int, int, int, int]] = []
        x, y  = _SPACING, _SPACING
        col   = 0
        row_h = 0

        for item in self._items:
            if self._is_item_filtered(item):
                # Filtered-out items get sentinel; do not advance layout position
                cache.append(_FILTERED_SENTINEL)
                continue

            h = max(40, round(cell_w / item.aspect_ratio)) + BOTTOM_H
            cache.append((x, y, cell_w, h))
            row_h = max(row_h, h)
            col += 1
            if col >= n:
                col = 0
                x   = _SPACING
                y  += row_h + _SPACING
                row_h = 0
            else:
                x += cell_w + _SPACING

        if row_h:          # last partial row
            y += row_h + _SPACING

        self._layout_cache = cache

        # Resize virtual canvas to fit all items
        self._container.resize(vp_w, max(y, vp_h))

        # Empty-state overlay for filter
        visible_count = sum(1 for g in self._layout_cache if g != _FILTERED_SENTINEL)
        if visible_count == 0 and self._items:
            self._empty_label.setText(
                "🔍\n\nNo files match the filter\n\nClear the filter to show all files"
            )
            self._empty_label.setGeometry(0, 0, vp_w, vp_h)
            self._empty_label.show()
        else:
            self._empty_label.hide()

        self._update_active_widgets()

    def _buffered_visible_range(self) -> tuple[int, int]:
        """Return (y_top, y_bottom) of the buffered visible region."""
        sv   = self.verticalScrollBar().value()
        vp_h = self.viewport().height()
        return sv - _BUFFER_PX, sv + vp_h + _BUFFER_PX

    def _update_active_widgets(self):
        """
        Create widgets for items entering the buffered viewport;
        destroy and save state for items leaving it.
        """
        if not self._layout_cache:
            return

        y_top, y_bot = self._buffered_visible_range()

        # Which item indices should currently have live widgets?
        should_be_active: set[int] = set()
        for idx, (x, y, w, h) in enumerate(self._layout_cache):
            if (x, y, w, h) == _FILTERED_SENTINEL:
                continue  # filtered-out item
            if y + h >= y_top and y <= y_bot:
                should_be_active.add(idx)

        # ── destroy widgets that have scrolled out of range ──────────────────
        to_remove = [idx for idx in self._active if idx not in should_be_active]
        for idx in to_remove:
            widget = self._active.pop(idx)
            item   = self._items[idx]
            item.checked = widget.is_checked()
            if isinstance(widget, VideoThumbnailWidget):
                item.seek_time = widget.seek_time
            widget.cleanup()
            widget.setParent(None)
            widget.deleteLater()

        # ── create / update widgets for items entering range ─────────────────
        for idx in should_be_active:
            x, y, cw, ch = self._layout_cache[idx]

            if idx in self._active:
                # Already alive — just sync geometry (may have changed on resize)
                self._active[idx].setGeometry(x, y, cw, ch)
                continue

            item = self._items[idx]
            if item.is_folder:
                widget = FolderThumbnailWidget(item.path, item.child_count,
                                              item.video_count, self._container)
                widget.checked_changed.connect(self._on_check_changed)
                widget.delete_requested.connect(self._delete_single)
                widget.move_requested.connect(self._move_single)
                widget.rename_requested.connect(self._rename_single)
                widget.copy_requested.connect(self._copy_single)
                widget.navigate_requested.connect(self.navigate_requested)
            else:
                widget = VideoThumbnailWidget(item.path, item.seek_time, self._container)
                widget.checked_changed.connect(self._on_check_changed)
                widget.seek_requested.connect(self._on_seek_requested)
                widget.open_requested.connect(self._open_file)
                widget.delete_requested.connect(self._delete_single)
                widget.move_requested.connect(self._move_single)
                widget.rename_requested.connect(self._rename_single)
                widget.copy_requested.connect(self._copy_single)
                widget.rating_changed.connect(self._on_rating_changed)
                if hasattr(widget, 'watch_toggled'):
                    widget.watch_toggled.connect(self.watch_toggled)
                if item.pixmap is not None:
                    widget.set_thumbnail(item.pixmap, item.duration)
                widget.set_rating(item.rating)

            # Restore saved state
            widget.set_checked(item.checked)

            # Pass extended state to video widgets
            if not item.is_folder:
                if hasattr(widget, 'set_watched'):
                    widget.set_watched(item.is_watched)
                if hasattr(widget, 'set_sidecar_files'):
                    widget.set_sidecar_files(item.sub_exists, item.nfo_exists)
                if hasattr(widget, 'set_focused'):
                    widget.set_focused(item.is_focused)

            # Pass extended state to folder widgets
            if item.is_folder:
                if hasattr(widget, 'set_folder_size') and item.folder_size >= 0:
                    widget.set_folder_size(item.folder_size)
                if hasattr(widget, 'set_mtime'):
                    widget.set_mtime(item.mtime)
                if hasattr(widget, 'set_has_nfo'):
                    widget.set_has_nfo(item.nfo_exists)

            widget.setGeometry(x, y, cw, ch)
            widget.show()
            self._active[idx] = widget

        # Apply focus ring to focused widget
        for idx, widget in self._active.items():
            is_focused = (idx == self._focused_idx)
            if hasattr(widget, 'set_focused'):
                widget.set_focused(is_focused)

        # Hide any widgets that are now filtered
        for idx in list(self._active.keys()):
            if idx not in should_be_active:
                pass  # already handled by destruction above

    # ── scroll handling ─────────────────────────────────────────────────────────
    def _on_scroll(self, _value: int):
        # Debounce: give the user 60 ms to finish a scroll gesture
        self._scroll_timer.start(60)

    # ── thumbnail ready ─────────────────────────────────────────────────────────
    def _on_thumbnail_ready(self, video_path: str, seek_time: float,
                            qimage: object, duration: float):
        idx = self._path_to_idx.get(video_path)
        if idx is None:
            return

        item = self._items[idx]

        # Drop stale result (seek time changed after this request was queued)
        if abs(item.seek_time - seek_time) > 0.001:
            return

        # Convert to QPixmap on the main thread (cheap)
        if isinstance(qimage, QImage) and not qimage.isNull():
            pix = QPixmap.fromImage(qimage)
        elif isinstance(qimage, QPixmap) and not qimage.isNull():
            pix = qimage
        else:
            return

        # Update the item record (persists across virtual-scroll widget recycling)
        item.pixmap   = pix
        item.duration = duration
        if pix.height() > 0:
            item.aspect_ratio = pix.width() / pix.height()

        # Push to live widget if one exists for this item right now
        widget = self._active.get(idx)
        if widget is not None:
            widget.set_thumbnail(pix, duration)

        # Aspect ratio may have changed → debounce a full relayout
        self._resize_timer.start(120)

    # ── slots from individual widgets ───────────────────────────────────────────
    def _on_check_changed(self, path: str, checked: bool):
        idx = self._path_to_idx.get(path)
        if idx is not None:
            self._items[idx].checked = checked
        # Count directly — state was just updated above; no need to sync widgets
        self.selection_changed.emit(sum(1 for item in self._items if item.checked))

    def _on_seek_requested(self, video_path: str, seek_time: float):
        idx = self._path_to_idx.get(video_path)
        if idx is not None:
            item = self._items[idx]
            item.seek_time = seek_time
            item.pixmap    = None
        self._generator.regenerate_thumbnail(video_path, seek_time)

    def _on_rating_changed(self, path: str, rating: int):
        idx = self._path_to_idx.get(path)
        if idx is not None:
            self._items[idx].rating = rating
        self._cache.set_rating(path, rating)

    def _open_file(self, video_path: str):
        # Emit the signal so MainWindow can decide whether to use the player or startfile
        self.open_requested.emit(video_path)

    def _mark_item_checked(self, video_path: str):
        """Mark a single item as checked (used by delete/move context-menu actions)."""
        idx = self._path_to_idx.get(video_path)
        if idx is not None:
            self._items[idx].checked = True
            w = self._active.get(idx)
            if w:
                w.set_checked(True)
        self.selection_changed.emit(sum(1 for item in self._items if item.checked))

    def _delete_single(self, video_path: str):
        self._mark_item_checked(video_path)

    def _move_single(self, video_path: str):
        self._mark_item_checked(video_path)

    def _rename_single(self, path: str):
        self._mark_item_checked(path)
        self.rename_requested.emit(path)

    def _copy_single(self, path: str):
        self._mark_item_checked(path)
        self.copy_requested.emit(path)

    def _sync_active_state(self):
        """Copy live widget state back to the corresponding _Item records."""
        for idx, widget in self._active.items():
            item = self._items[idx]
            item.checked = widget.is_checked()
            if isinstance(widget, VideoThumbnailWidget):
                item.seek_time = widget.seek_time

    def _items_summary(self) -> str:
        """Build a status-bar string like '3 folders, 12 videos'."""
        filter_active = bool(self._filter_text)
        n_dirs  = sum(1 for it in self._items if it.is_folder
                      and not self._is_item_filtered(it))
        n_files = sum(1 for it in self._items if not it.is_folder
                      and not self._is_item_filtered(it))
        parts: list[str] = []
        if n_dirs:  parts.append(f"{n_dirs} folder{'s' if n_dirs != 1 else ''}")
        if n_files: parts.append(f"{n_files} video{'s' if n_files != 1 else ''}")
        result = ', '.join(parts) if parts else "Empty folder"
        if filter_active:
            shown = n_dirs + n_files
            result += f" ({shown} shown)"
        return result

    def _rebuild_index(self):
        """Rebuild _path_to_idx after the items list has been modified."""
        self._path_to_idx = {item.path: i for i, item in enumerate(self._items)}

    def _clear_all(self):
        """Destroy all live widgets and reset the entire data model."""
        for widget in self._active.values():
            widget.cleanup()
            widget.setParent(None)
            widget.deleteLater()
        self._active.clear()
        self._items.clear()
        self._path_to_idx.clear()
        self._layout_cache.clear()
        self._pending_items = []
        self._batch_idx     = 0
        self._total_count   = 0
        self._focused_idx   = -1
        self._empty_label.hide()
        vp_w = self.viewport().width()
        vp_h = self.viewport().height()
        self._container.resize(vp_w, vp_h)
