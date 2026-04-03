"""
Main application window.

Layout:
  ┌─────────────────────────────────────────────┐
  │  Toolbar                                    │
  ├──────────────┬──────────────────────────────┤
  │  Folder tree │  Breadcrumb + Filter + Grid  │
  │              │                              │
  ├──────────────┴──────────────────────────────┤
  │  [Video player panel — hidden by default]   │
  ├──────────────────────────────────────────────┤
  │  Status bar                                 │
  └─────────────────────────────────────────────┘

Uses a custom QTreeWidget for the folder tree (avoids QFileSystemModel,
which was reorganised across PyQt6 versions).
"""

import os
import shutil
import string
from collections import deque
from dataclasses import dataclass, field

from PyQt6.QtWidgets import (
    QMainWindow, QSplitter, QTreeWidget, QTreeWidgetItem,
    QToolBar, QStatusBar, QMessageBox, QFileDialog,
    QWidget, QLabel, QSpinBox, QComboBox, QToolButton, QHBoxLayout,
    QVBoxLayout, QAbstractItemView, QLineEdit, QInputDialog,
    QProgressDialog, QSlider, QMenu,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QAction

from app_settings import AppSettings
from cache_manager import CacheManager
import metadata_dialog as _metadata_dialog
from thumbnail_generator import ThumbnailGeneratorService
from thumbnail_grid_widget import ThumbnailGridWidget
from settings_dialog import SettingsDialog
from breadcrumb_widget import BreadcrumbWidget
from video_player_widget import VideoPlayerWidget
from batch_rename_dialog import BatchRenameDialog
from duplicate_finder_dialog import DuplicateFinderDialog
from bulk_meta_worker import BulkMetaWorker

try:
    from tag_dialog import TagDialog
    _HAS_TAG_DIALOG = True
except ImportError:
    _HAS_TAG_DIALOG = False

try:
    from quick_preview_dialog import QuickPreviewDialog
    _HAS_QUICK_PREVIEW = True
except ImportError:
    _HAS_QUICK_PREVIEW = False

try:
    from folder_size_scanner import FolderSizeScanner
    _HAS_FOLDER_SCANNER = True
except ImportError:
    _HAS_FOLDER_SCANNER = False

try:
    from export_dialog import ExportDialog
    _HAS_EXPORT = True
except ImportError:
    _HAS_EXPORT = False

try:
    from collections_dialog import CollectionsDialog
    _HAS_COLLECTIONS = True
except ImportError:
    _HAS_COLLECTIONS = False

try:
    import send2trash
    _HAS_SEND2TRASH = True
except ImportError:
    _HAS_SEND2TRASH = False

# Sentinel child used to mark "not yet expanded"
_PLACEHOLDER = "__placeholder__"
_BOOKMARK_ROLE = Qt.ItemDataRole.UserRole + 1   # flag to distinguish bookmark items


@dataclass
class _UndoOp:
    kind:       str          # 'move' | 'delete'
    orig_paths: list         # original source paths
    dest:       str = ''     # destination folder (for 'move')


class _FolderTree(QTreeWidget):
    """Lazy-loading folder tree that works with Python's os module."""

    def __init__(self, bookmarks: list = None, parent=None):
        super().__init__(parent)
        self.setHeaderHidden(True)
        self.setColumnCount(1)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setAnimated(True)
        self.itemExpanded.connect(self._on_expanded)
        self._populate_roots()
        if bookmarks:
            for b in bookmarks:
                self._add_bookmark_item(b)

    # ------------------------------------------------------------------ build
    def _populate_roots(self):
        if os.name == "nt":
            # Windows: add every available drive letter
            for letter in string.ascii_uppercase:
                drive = f"{letter}:\\"
                if os.path.exists(drive):
                    item = QTreeWidgetItem([drive])
                    item.setData(0, Qt.ItemDataRole.UserRole, drive)
                    self._add_placeholder(item)
                    self.addTopLevelItem(item)
        else:
            # Unix: single root
            item = QTreeWidgetItem(["/"])
            item.setData(0, Qt.ItemDataRole.UserRole, "/")
            self._add_placeholder(item)
            self.addTopLevelItem(item)

    @staticmethod
    def _add_placeholder(item: QTreeWidgetItem):
        ph = QTreeWidgetItem([_PLACEHOLDER])
        item.addChild(ph)

    # ------------------------------------------------------------------ bookmarks
    def _add_bookmark_item(self, path: str) -> QTreeWidgetItem:
        name = os.path.basename(path) or path
        item = QTreeWidgetItem([f"★  {name}"])
        item.setData(0, Qt.ItemDataRole.UserRole, path)
        item.setData(0, _BOOKMARK_ROLE, True)
        item.setToolTip(0, path)
        # Bookmarks are not expandable
        self.addTopLevelItem(item)
        return item

    def add_bookmark(self, path: str):
        """Add a bookmark item to the tree (if not already present)."""
        for i in range(self.topLevelItemCount()):
            item = self.topLevelItem(i)
            if (item.data(0, _BOOKMARK_ROLE) and
                    item.data(0, Qt.ItemDataRole.UserRole) == path):
                return
        self._add_bookmark_item(path)

    def reload_bookmarks(self, bookmarks: list):
        """Remove all bookmark items and re-add them from the given list."""
        to_remove = []
        for i in range(self.topLevelItemCount()):
            item = self.topLevelItem(i)
            if item.data(0, _BOOKMARK_ROLE):
                to_remove.append(item)
        for item in to_remove:
            idx = self.indexOfTopLevelItem(item)
            if idx >= 0:
                self.takeTopLevelItem(idx)
        for path in bookmarks:
            self._add_bookmark_item(path)

    # ------------------------------------------------------------------ network paths
    def add_network_path(self, path: str):
        """Add a network/UNC path as a top-level expandable item."""
        name = path if len(path) <= 40 else f"…{path[-38:]}"
        item = QTreeWidgetItem([f"🌐  {name}"])
        item.setData(0, Qt.ItemDataRole.UserRole, path)
        item.setToolTip(0, path)
        self._add_placeholder(item)
        self.addTopLevelItem(item)

    # ------------------------------------------------------------------ lazy load
    def _on_expanded(self, item: QTreeWidgetItem):
        # Bookmark items are not expandable — skip
        if item.data(0, _BOOKMARK_ROLE):
            return
        # Only load if the single child is still the placeholder
        if item.childCount() == 1 and item.child(0).text(0) == _PLACEHOLDER:
            item.takeChild(0)
            path = item.data(0, Qt.ItemDataRole.UserRole)
            self._load_children(item, path)

    def _load_children(self, item: QTreeWidgetItem, path: str):
        try:
            entries = sorted(
                (e for e in os.scandir(path)
                 if e.is_dir(follow_symlinks=False)
                 and not e.name.startswith('.')),
                key=lambda e: e.name.lower()
            )
        except (PermissionError, OSError):
            return
        for entry in entries:
            child = QTreeWidgetItem([entry.name])
            child.setData(0, Qt.ItemDataRole.UserRole, entry.path)
            self._add_placeholder(child)
            item.addChild(child)

    # ------------------------------------------------------------------ navigation
    def select_path(self, folder_path: str):
        """Expand the tree down to folder_path and select it."""
        folder_path = os.path.normpath(folder_path)
        if os.name == "nt":
            drive = os.path.splitdrive(folder_path)[0] + "\\"
            parts = folder_path[len(drive):].split(os.sep)
            parts = [p for p in parts if p]
            self._expand_path(drive, parts)
        else:
            parts = [p for p in folder_path.split("/") if p]
            self._expand_path("/", parts)

    def _expand_path(self, root_path: str, parts: list[str]):
        # Find root item (skip bookmark / network items)
        root_item = None
        for i in range(self.topLevelItemCount()):
            item = self.topLevelItem(i)
            if item.data(0, _BOOKMARK_ROLE):
                continue
            data = item.data(0, Qt.ItemDataRole.UserRole)
            if data and os.path.normpath(data) == os.path.normpath(root_path):
                root_item = item
                break
        if root_item is None:
            return

        current = root_item
        self._ensure_expanded(current)
        for part in parts:
            found = None
            for i in range(current.childCount()):
                child = current.child(i)
                if child.text(0).lower() == part.lower():
                    found = child
                    break
            if found is None:
                break
            self._ensure_expanded(found)
            current = found

        self.setCurrentItem(current)
        self.scrollToItem(current)

    def _ensure_expanded(self, item: QTreeWidgetItem):
        if not item.isExpanded():
            item.setExpanded(True)
            # _on_expanded fires synchronously above


class _MoveWorker(QThread):
    """
    Moves (or copies) files/folders on a background thread.

    Cross-drive strategy
    ─────────────────────
    os.rename() is tried first (atomic, instant on the same drive/filesystem).
    If that raises OSError (different drive), we fall back to:
      1. shutil.copy2()  – full copy with metadata
      2. size verification  – ensures the copy is complete before touching source
      3. source deletion (skipped when copy_only=True) with retries.
    """

    _DELETE_RETRIES     = 5
    _DELETE_RETRY_DELAY = 0.5   # seconds between delete retries

    item_done = pyqtSignal(str, bool, str)   # (src_path, success, error_msg)
    all_done  = pyqtSignal(list, list)        # (moved_paths, error_strings)

    def __init__(self, paths: list, dest: str, copy_only: bool = False, parent=None):
        super().__init__(parent)
        self._paths     = paths
        self._dest      = dest
        self._copy_only = copy_only

    def run(self):
        moved, errors = [], []
        for p in self._paths:
            err = self._move_one(p)
            if err:
                errors.append(f"{os.path.basename(p)}: {err}")
                self.item_done.emit(p, False, err)
            else:
                moved.append(p)
                self.item_done.emit(p, True, '')
        self.all_done.emit(moved, errors)

    def _move_one(self, src: str) -> str:
        """Move (or copy) src into self._dest.  Returns '' on success, error string on failure."""
        dst = os.path.join(self._dest, os.path.basename(src))

        if self._copy_only:
            try:
                if os.path.isdir(src):
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)
                return ''
            except Exception as exc:
                return f"copy failed: {exc}"

        # ── same-filesystem: atomic rename ───────────────────────────────────
        try:
            os.rename(src, dst)
            return ''
        except OSError:
            pass   # different drive — fall through to copy+delete

        # ── cross-drive: copy, verify, then delete source ────────────────────
        try:
            if os.path.isdir(src):
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
        except Exception as exc:
            return f"copy failed: {exc}"

        # Verify destination size matches source before we touch the source
        try:
            if not os.path.isdir(src):
                if os.path.getsize(dst) != os.path.getsize(src):
                    try:
                        os.unlink(dst)
                    except OSError:
                        pass
                    return "destination size mismatch after copy; source not deleted"
        except OSError:
            pass   # if we can't stat, continue and try to delete anyway

        # Delete source with retries (handles lingering cv2.VideoCapture handles)
        import time
        last_err = ''
        for attempt in range(self._DELETE_RETRIES):
            try:
                if os.path.isdir(src):
                    shutil.rmtree(src)
                else:
                    os.unlink(src)
                return ''   # success
            except OSError as exc:
                last_err = str(exc)
                if attempt < self._DELETE_RETRIES - 1:
                    time.sleep(self._DELETE_RETRY_DELAY)

        return (f"file was copied to destination but source could not be deleted "
                f"({last_err})")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self._settings = AppSettings()
        self._cache = CacheManager(self._settings.db_path, self._settings.cache_dir)
        _metadata_dialog.set_cache_manager(self._cache)
        _metadata_dialog.set_custom_search_urls(self._settings.custom_search_urls)
        self._generator = ThumbnailGeneratorService(self._cache, self)
        self._current_folder: str = ""

        # Undo stack
        self._undo_stack: deque = deque(maxlen=10)
        self._pending_move_orig_paths: list = []
        self._pending_move_dest: str = ""

        # Background workers (keep refs to prevent GC)
        self._move_worker: _MoveWorker | None = None
        self._bulk_meta_worker: BulkMetaWorker | None = None
        self._folder_scanner = None

        self.setWindowTitle("Video Organizer")
        self.setMinimumSize(900, 600)
        self._apply_dark_theme()

        self._build_toolbar()
        self._build_central()
        self._build_status_bar()

        # Restore geometry
        geom = self._settings.window_geometry
        if geom:
            self.restoreGeometry(geom)
        splitter_state = self._settings.splitter_state
        if splitter_state:
            self._splitter.restoreState(splitter_state)

        # Restore player geometry
        player_geom = self._settings.player_geometry
        if player_geom:
            self._player_panel.restoreGeometry(player_geom)

        # Open last folder
        last = self._settings.last_folder
        if last and os.path.isdir(last):
            self._navigate_to(last)

    # ------------------------------------------------------------------ build UI
    def _apply_dark_theme(self):
        self.setStyleSheet("""
            QMainWindow, QWidget { background: #1e1e1e; color: #ddd; }
            QToolBar { background: #252525; border-bottom: 1px solid #333; spacing: 4px; }
            QToolButton { background: #2e2e2e; border: 1px solid #3a3a3a;
                          border-radius: 3px; padding: 4px 8px; color: #ddd; }
            QToolButton:hover { background: #3a3a3a; }
            QToolButton:pressed { background: #444; }
            QToolButton:checked { background: #3a6fc4; color: #fff; }
            QTreeWidget { background: #252525; color: #ccc; border: none;
                          selection-background-color: #3a6fc4; outline: none; }
            QTreeWidget::item { padding: 2px; }
            QTreeWidget::item:hover { background: #2e2e2e; }
            QTreeWidget::item:selected { background: #3a6fc4; color: #fff; }
            QScrollBar:vertical { background: #252525; width: 10px; border: none; }
            QScrollBar::handle:vertical { background: #444; border-radius: 4px; min-height: 20px; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
            QScrollBar:horizontal { background: #252525; height: 10px; border: none; }
            QScrollBar::handle:horizontal { background: #444; border-radius: 4px; }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
            QStatusBar { background: #252525; color: #aaa; }
            QSplitter::handle { background: #333; width: 3px; }
            QSpinBox { background: #2e2e2e; color: #ddd; border: 1px solid #3a3a3a;
                       border-radius: 3px; padding: 2px 22px 2px 6px; }
            QSpinBox::up-button   { width: 18px; background: #3a3a3a;
                                    border-left: 1px solid #555;
                                    border-bottom: 1px solid #555; }
            QSpinBox::down-button { width: 18px; background: #3a3a3a;
                                    border-left: 1px solid #555; }
            QSpinBox::up-button:hover, QSpinBox::down-button:hover { background: #505050; }
            QSpinBox::up-arrow   { width: 7px; height: 7px;
                                   border-left: 4px solid transparent;
                                   border-right: 4px solid transparent;
                                   border-bottom: 5px solid #ccc; image: none; }
            QSpinBox::down-arrow { width: 7px; height: 7px;
                                   border-left: 4px solid transparent;
                                   border-right: 4px solid transparent;
                                   border-top: 5px solid #ccc; image: none; }
            QLabel { color: #ddd; }
            QLineEdit { background: #2e2e2e; color: #ddd; border: 1px solid #3a3a3a;
                        border-radius: 3px; padding: 3px 6px; }
            QMessageBox { background: #2a2a2a; color: #ddd; }
            QMessageBox QPushButton { background: #3a3a3a; color: #ddd; border: 1px solid #555;
                                      border-radius: 3px; padding: 4px 12px; min-width: 60px; }
            QMessageBox QPushButton:hover { background: #484848; }
            QProgressDialog { background: #2a2a2a; color: #ddd; }
            QProgressDialog QPushButton { background: #3a3a3a; color: #ddd;
                                          border: 1px solid #555; border-radius: 3px;
                                          padding: 4px 12px; min-width: 60px; }
            QMenu { background: #2a2a2a; color: #ddd; border: 1px solid #444; }
            QMenu::item { padding: 5px 20px 5px 12px; }
            QMenu::item:selected { background: #3a6fc4; }
            QMenu::separator { height: 1px; background: #444; margin: 3px 0; }
        """)

    def _build_toolbar(self):
        tb = QToolBar("Main", self)
        tb.setMovable(False)
        tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.addToolBar(tb)

        # ── Selection group ──────────────────────────────────────────────────
        self._act_select_all = QAction("Select All", self)
        self._act_select_all.triggered.connect(self._on_select_all)
        tb.addAction(self._act_select_all)

        self._act_deselect = QAction("Deselect All", self)
        self._act_deselect.triggered.connect(self._on_deselect_all)
        tb.addAction(self._act_deselect)

        tb.addSeparator()

        # ── File operations group ────────────────────────────────────────────
        self._act_move = QAction("Move Selected…", self)
        self._act_move.triggered.connect(self._on_move)
        self._act_move.setEnabled(False)
        tb.addAction(self._act_move)

        self._act_copy_sel = QAction("Copy Selected…", self)
        self._act_copy_sel.triggered.connect(self._on_copy)
        self._act_copy_sel.setEnabled(False)
        tb.addAction(self._act_copy_sel)

        self._act_delete = QAction("Delete Selected", self)
        self._act_delete.triggered.connect(self._on_delete)
        self._act_delete.setEnabled(False)
        tb.addAction(self._act_delete)

        self._act_undo = QAction("Undo", self)
        self._act_undo.setShortcut("Ctrl+Z")
        self._act_undo.triggered.connect(self._on_undo)
        self._act_undo.setEnabled(False)
        tb.addAction(self._act_undo)

        tb.addSeparator()

        # ── Rename group ─────────────────────────────────────────────────────
        self._act_batch_rename = QAction("Rename…", self)
        self._act_batch_rename.triggered.connect(self._on_batch_rename)
        self._act_batch_rename.setEnabled(False)
        tb.addAction(self._act_batch_rename)

        tb.addSeparator()

        # ── View group: columns slider + sort ────────────────────────────────
        view_widget = QWidget()
        view_layout = QHBoxLayout(view_widget)
        view_layout.setContentsMargins(4, 0, 4, 0)
        view_layout.setSpacing(4)

        cols_lbl = QLabel("Columns:")
        cols_lbl.setStyleSheet("color: #ccc;")
        view_layout.addWidget(cols_lbl)

        self._cols_slider = QSlider(Qt.Orientation.Horizontal)
        self._cols_slider.setRange(1, 8)
        self._cols_slider.setValue(self._settings.thumbnails_per_row)
        self._cols_slider.setFixedWidth(120)
        self._cols_slider.setStyleSheet("""
            QSlider::groove:horizontal { height: 4px; background: #3a3a3a; border-radius: 2px; }
            QSlider::sub-page:horizontal { background: #3a6fc4; border-radius: 2px; }
            QSlider::handle:horizontal { width: 14px; height: 14px; margin: -5px 0;
                background: #7ab8e8; border-radius: 7px; }
        """)
        self._cols_slider.valueChanged.connect(self._on_cols_changed)
        view_layout.addWidget(self._cols_slider)

        self._cols_label = QLabel(str(self._settings.thumbnails_per_row))
        self._cols_label.setStyleSheet("color: #888; min-width: 16px;")
        self._cols_slider.valueChanged.connect(lambda v: self._cols_label.setText(str(v)))
        view_layout.addWidget(self._cols_label)

        # Sort controls
        sort_sep = QLabel("|")
        sort_sep.setStyleSheet("color: #444; padding: 0 4px;")
        view_layout.addWidget(sort_sep)

        sort_lbl = QLabel("Sort:")
        sort_lbl.setStyleSheet("color: #ccc;")
        view_layout.addWidget(sort_lbl)

        self._sort_combo = QComboBox()
        self._sort_combo.addItems(["Name", "Date Modified", "Size", "Type", "Rating"])
        self._sort_combo.setMinimumWidth(110)
        self._sort_combo.setStyleSheet(
            "QComboBox { background:#2e2e2e; color:#ddd; border:1px solid #3a3a3a;"
            "            border-radius:3px; padding:2px 6px; }"
            "QComboBox::drop-down { border:none; width:18px; }"
            "QComboBox QAbstractItemView { background:#2e2e2e; color:#ddd;"
            "                              selection-background-color:#3a6fc4; }"
        )
        _key_to_label = {'name': 'Name', 'modified': 'Date Modified',
                         'size': 'Size',  'type': 'Type', 'rating': 'Rating'}
        saved_label = _key_to_label.get(self._settings.sort_key, 'Name')
        self._sort_combo.setCurrentText(saved_label)
        view_layout.addWidget(self._sort_combo)

        self._sort_dir_btn = QToolButton()
        self._sort_dir_btn.setCheckable(True)
        self._sort_dir_btn.setChecked(self._settings.sort_asc)
        self._sort_dir_btn.setText("↑ Asc" if self._settings.sort_asc else "↓ Desc")
        self._sort_dir_btn.setStyleSheet(
            "QToolButton { background:#2e2e2e; border:1px solid #3a3a3a;"
            "              border-radius:3px; padding:3px 8px; color:#ddd; }"
            "QToolButton:hover  { background:#3a3a3a; }"
            "QToolButton:checked { background:#3a6fc4; color:#fff; border-color:#2a5fc4; }"
        )
        view_layout.addWidget(self._sort_dir_btn)

        tb.addWidget(view_widget)

        # Connect sort controls
        self._sort_combo.currentTextChanged.connect(self._on_sort_changed)
        self._sort_dir_btn.toggled.connect(self._on_sort_dir_toggled)

        tb.addSeparator()

        # ── Navigate group ───────────────────────────────────────────────────
        self._act_up = QAction("⬆ Up", self)
        self._act_up.setToolTip("Go up one level in the folder hierarchy")
        self._act_up.triggered.connect(self._on_go_up)
        self._act_up.setEnabled(False)
        tb.addAction(self._act_up)

        tb.addSeparator()

        # ── Tools menu button ────────────────────────────────────────────────
        self._tools_btn = QToolButton()
        self._tools_btn.setText("Tools ▾")
        self._tools_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self._tools_btn.setStyleSheet(
            "QToolButton { background:#2e2e2e; border:1px solid #3a3a3a;"
            "              border-radius:3px; padding:4px 10px; color:#ddd; }"
            "QToolButton:hover { background:#3a3a3a; }"
            "QToolButton::menu-indicator { image: none; }"
        )

        tools_menu = QMenu(self._tools_btn)
        self._act_find_dupes = tools_menu.addAction("Find Duplicates…")
        self._act_find_dupes.triggered.connect(self._on_find_duplicates)

        self._act_fetch_meta = tools_menu.addAction("⬇ Fetch Metadata")
        self._act_fetch_meta.triggered.connect(self._on_fetch_all_metadata)

        self._act_export = tools_menu.addAction("📊 Export Library…")
        self._act_export.triggered.connect(self._on_export)
        self._act_export.setEnabled(_HAS_EXPORT)

        self._act_collections = tools_menu.addAction("★ Collections…")
        self._act_collections.triggered.connect(self._on_collections)
        self._act_collections.setEnabled(_HAS_COLLECTIONS)

        self._act_edit_tags = tools_menu.addAction("🏷 Edit Tags…")
        self._act_edit_tags.triggered.connect(self._on_edit_tags)
        self._act_edit_tags.setEnabled(False)

        self._act_scan_sizes = tools_menu.addAction("📁 Scan Folder Sizes")
        self._act_scan_sizes.triggered.connect(self._on_scan_folder_sizes)
        self._act_scan_sizes.setEnabled(_HAS_FOLDER_SCANNER)

        tools_menu.addSeparator()

        self._act_network = tools_menu.addAction("🌐 Add Network Location…")
        self._act_network.triggered.connect(self._on_add_network_path)

        self._tools_btn.setMenu(tools_menu)
        tb.addWidget(self._tools_btn)

        tb.addSeparator()

        # ── Settings / Open ──────────────────────────────────────────────────
        act_settings = QAction("Settings", self)
        act_settings.triggered.connect(self._on_settings)
        tb.addAction(act_settings)

        act_open_folder = QAction("Open Folder…", self)
        act_open_folder.triggered.connect(self._on_open_folder)
        tb.addAction(act_open_folder)

        tb.addSeparator()

        # ── Player toggle ────────────────────────────────────────────────────
        self._act_player = QAction("▶ Player", self)
        self._act_player.setCheckable(True)
        self._act_player.setChecked(False)
        self._act_player.toggled.connect(self._on_player_toggled)
        tb.addAction(self._act_player)

    def _build_central(self):
        # ── Horizontal splitter: [tree | right panel] ────────────────────────
        self._splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.setCentralWidget(self._splitter)

        # Left: custom folder tree
        self._tree = _FolderTree(bookmarks=self._settings.bookmarks)
        self._tree.setMinimumWidth(180)
        self._tree.setMaximumWidth(400)
        self._tree.currentItemChanged.connect(self._on_tree_item_changed)
        self._splitter.addWidget(self._tree)

        # Right: breadcrumb + filter + grid in a QVBoxLayout container
        right_widget = QWidget(self)
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        # Breadcrumb
        self._breadcrumb = BreadcrumbWidget(right_widget)
        self._breadcrumb.navigate_requested.connect(self._navigate_to)
        self._breadcrumb.bookmark_requested.connect(self._on_bookmark_toggle)
        right_layout.addWidget(self._breadcrumb)

        # Filter bar
        self._filter_edit = QLineEdit(right_widget)
        self._filter_edit.setPlaceholderText("🔍  Filter files…")
        self._filter_edit.setStyleSheet(
            "QLineEdit { background: #1e1e1e; color: #ddd; border: none;"
            "            border-bottom: 1px solid #333; padding: 4px 8px;"
            "            border-radius: 0; }"
        )
        self._filter_edit.textChanged.connect(self._on_filter_changed)
        right_layout.addWidget(self._filter_edit)

        # Thumbnail grid
        self._grid = ThumbnailGridWidget(self._generator, self._settings, self._cache, right_widget)
        self._grid.selection_changed.connect(self._on_selection_changed)
        self._grid.status_message.connect(self._on_grid_status)
        self._grid.navigate_requested.connect(self._navigate_to)
        self._grid.delete_key_pressed.connect(self._on_delete)
        self._grid.undo_requested.connect(self._on_undo)
        self._grid.rename_key_pressed.connect(self._on_rename_key)
        self._grid.rename_requested.connect(self._on_rename_single)
        self._grid.copy_requested.connect(self._on_copy_single)
        self._grid.open_requested.connect(self._on_open_video)
        self._grid.files_dropped.connect(self._on_files_dropped)
        self._grid.quick_preview_requested.connect(self._on_quick_preview)
        if hasattr(self._grid, 'watch_toggled'):
            self._grid.watch_toggled.connect(self._on_watch_toggled)
        right_layout.addWidget(self._grid)

        self._splitter.addWidget(right_widget)

        self._splitter.setStretchFactor(0, 0)
        self._splitter.setStretchFactor(1, 1)
        self._splitter.setSizes([220, 680])

        # Network paths from settings
        for np_path in self._settings.network_paths:
            self._tree.add_network_path(np_path)

        # ── Floating player window (not embedded — independent popup) ─────────
        self._player_panel = VideoPlayerWidget(self)   # parent=self for GC only
        self._player_panel.closed.connect(self._on_player_closed)

    # ── Open video signal handler ─────────────────────────────────────────────
    @pyqtSlot(str)
    def _on_open_video(self, path: str):
        """Open a video: use the popup player if it is open, otherwise startfile."""
        if self._act_player.isChecked():
            self._player_panel.play(path)
            self._player_panel.show()
            self._player_panel.raise_()
            self._player_panel.activateWindow()
        else:
            try:
                os.startfile(path)
            except AttributeError:
                import subprocess
                subprocess.Popen(['xdg-open', path])

    def _on_player_toggled(self, checked: bool):
        """Show or hide the floating player window."""
        if checked:
            self._player_panel.show()
            self._player_panel.raise_()
            self._player_panel.activateWindow()
        else:
            self._settings.player_geometry = self._player_panel.saveGeometry()
            self._player_panel.stop()
            self._player_panel.hide()

    @pyqtSlot()
    def _on_player_closed(self):
        """Called when the user clicks the X on the player window."""
        self._settings.player_geometry = self._player_panel.saveGeometry()
        # Un-check the toolbar button without re-triggering _on_player_toggled
        self._act_player.blockSignals(True)
        self._act_player.setChecked(False)
        self._act_player.blockSignals(False)

    def _build_status_bar(self):
        self._status = QStatusBar(self)
        self.setStatusBar(self._status)
        self._status_label = QLabel("")
        self._status.addWidget(self._status_label)
        self._sel_label = QLabel("")
        self._status.addPermanentWidget(self._sel_label)
        self._disk_label = QLabel("")
        self._disk_label.setStyleSheet("color: #888; padding-right: 6px;")
        self._status.addPermanentWidget(self._disk_label)

    # ------------------------------------------------------------------ navigation
    def _navigate_to(self, folder_path: str):
        self._current_folder = folder_path
        self._settings.last_folder = folder_path
        self._tree.select_path(folder_path)
        self._grid.load_folder(folder_path)
        self.setWindowTitle(f"Video Organizer – {folder_path}")
        self._breadcrumb.set_path(folder_path)
        # Update bookmark button state
        bms = self._settings.bookmarks
        self._breadcrumb.set_bookmarked(folder_path in bms)
        # Enable "Up" only when there is a true parent (not already at a root/drive)
        parent = os.path.dirname(folder_path)
        self._act_up.setEnabled(bool(parent) and parent != folder_path)

    # ------------------------------------------------------------------ toolbar slots
    @pyqtSlot()
    def _on_select_all(self):
        self._grid.select_all()

    @pyqtSlot()
    def _on_deselect_all(self):
        self._grid.deselect_all()

    @pyqtSlot()
    def _on_move(self):
        paths = self._grid.get_checked_paths()
        if not paths:
            return
        dest = QFileDialog.getExistingDirectory(
            self, "Move files to…", self._current_folder or os.path.expanduser("~")
        )
        if not dest:
            return

        # Cancel thumbnail generation so cv2.VideoCapture handles are released
        # before we try to delete source files on cross-drive moves.
        self._generator.cancel_all()

        # Disable move/delete buttons while the operation is in progress
        self._act_move.setEnabled(False)
        self._act_delete.setEnabled(False)
        self._status_label.setText(f"Moving {len(paths)} item(s)…")

        # Save original paths for undo
        self._pending_move_orig_paths = list(paths)
        self._pending_move_dest = dest

        self._move_worker = _MoveWorker(paths, dest, copy_only=False, parent=self)
        self._move_worker.item_done.connect(self._on_move_item_done)
        self._move_worker.all_done.connect(self._on_move_all_done)
        self._move_worker.start()

    @pyqtSlot()
    def _on_copy(self):
        """Copy selected items to a destination folder."""
        paths = self._grid.get_checked_paths()
        if not paths:
            return
        dest = QFileDialog.getExistingDirectory(
            self, "Copy files to…", self._current_folder or os.path.expanduser("~")
        )
        if not dest:
            return

        self._generator.cancel_all()
        self._act_copy_sel.setEnabled(False)
        self._status_label.setText(f"Copying {len(paths)} item(s)…")

        self._move_worker = _MoveWorker(paths, dest, copy_only=True, parent=self)
        self._move_worker.item_done.connect(self._on_copy_item_done)
        self._move_worker.all_done.connect(self._on_copy_all_done)
        self._move_worker.start()

    @pyqtSlot(str)
    def _on_copy_single(self, path: str):
        """Copy a single item (from context menu)."""
        dest = QFileDialog.getExistingDirectory(
            self, "Copy file to…", self._current_folder or os.path.expanduser("~")
        )
        if not dest:
            return
        self._generator.cancel_all()
        self._status_label.setText(f"Copying {os.path.basename(path)}…")
        self._move_worker = _MoveWorker([path], dest, copy_only=True, parent=self)
        self._move_worker.item_done.connect(self._on_copy_item_done)
        self._move_worker.all_done.connect(self._on_copy_all_done)
        self._move_worker.start()

    @pyqtSlot(str, bool, str)
    def _on_copy_item_done(self, path: str, success: bool, _err: str):
        self._status_label.setText(
            f"Copied: {os.path.basename(path)}" if success
            else f"Copy failed: {os.path.basename(path)}"
        )

    @pyqtSlot(list, list)
    def _on_copy_all_done(self, copied: list, errors: list):
        self._move_worker = None
        self._on_selection_changed(len(self._grid.get_checked_paths()))
        n = len(copied)
        self._status_label.setText(f"Copied {n} item(s).")
        if errors:
            QMessageBox.warning(self, "Copy – errors",
                                "Some items could not be copied:\n" + "\n".join(errors))

    @pyqtSlot(str, bool, str)
    def _on_move_item_done(self, path: str, success: bool, _err: str):
        """Update status bar as each item finishes."""
        self._status_label.setText(
            f"Moved: {os.path.basename(path)}" if success
            else f"Failed: {os.path.basename(path)}"
        )

    @pyqtSlot(list, list)
    def _on_move_all_done(self, moved: list, errors: list):
        """Called on the main thread when all moves are complete."""
        for p in moved:
            if os.path.isfile(p):   # skip cache invalidation for folders
                self._cache.invalidate(p)
        self._grid.remove_paths(moved)
        self._move_worker = None    # release QThread ref so it can be GC'd

        # Push undo operation
        if moved:
            dest = self._pending_move_dest
            moved_set = set(moved)
            orig_paths = [p for p in self._pending_move_orig_paths if p in moved_set]
            self._undo_stack.append(_UndoOp(
                kind='move',
                orig_paths=orig_paths,
                dest=dest,
            ))
            self._act_undo.setEnabled(True)

        # Re-enable buttons (selection_changed will set the correct enabled state)
        self._on_selection_changed(len(self._grid.get_checked_paths()))
        if errors:
            QMessageBox.warning(self, "Move – errors",
                                "Some items could not be moved:\n" + "\n".join(errors))

    @pyqtSlot()
    def _on_delete(self):
        paths = self._grid.get_checked_paths()
        if not paths:
            return
        names = "\n".join(os.path.basename(p) for p in paths[:20])
        suffix = f"\n…and {len(paths) - 20} more" if len(paths) > 20 else ""
        verb = "recycle" if _HAS_SEND2TRASH else "permanently delete"
        reply = QMessageBox.question(
            self, "Confirm delete",
            f"Are you sure you want to {verb} {len(paths)} item(s)?\n\n{names}{suffix}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        errors, deleted = [], []
        for p in paths:
            try:
                is_file = os.path.isfile(p)
                if _HAS_SEND2TRASH:
                    send2trash.send2trash(p)
                else:
                    if os.path.isdir(p):
                        shutil.rmtree(p)
                    else:
                        os.remove(p)
                if is_file:
                    self._cache.invalidate(p)
                deleted.append(p)
            except Exception as e:
                errors.append(f"{os.path.basename(p)}: {e}")
        self._grid.remove_paths(deleted)

        # Push undo
        if deleted:
            self._undo_stack.append(_UndoOp(kind='delete', orig_paths=deleted))
            self._act_undo.setEnabled(True)

        if errors:
            QMessageBox.warning(self, "Delete – errors",
                                "Some items could not be deleted:\n" + "\n".join(errors))

    @pyqtSlot()
    def _on_undo(self):
        if not self._undo_stack:
            return
        op = self._undo_stack.pop()
        self._act_undo.setEnabled(bool(self._undo_stack))

        if op.kind == 'delete':
            QMessageBox.information(
                self, "Undo Delete",
                "Items were sent to the Recycle Bin.\n"
                "Use File Explorer to restore deleted items."
            )
        elif op.kind == 'move':
            # Files are now at dest/basename(orig_path); move them back
            dest_paths = [
                os.path.join(op.dest, os.path.basename(p))
                for p in op.orig_paths
            ]
            if op.orig_paths:
                back_dest = os.path.dirname(op.orig_paths[0])
            else:
                return
            # Only move files that actually exist at dest
            to_move = [p for p in dest_paths if os.path.exists(p)]
            if not to_move:
                QMessageBox.information(self, "Undo", "No files to undo.")
                return
            self._generator.cancel_all()
            self._status_label.setText(f"Undoing move of {len(to_move)} item(s)…")
            self._move_worker = _MoveWorker(to_move, back_dest, copy_only=False, parent=self)
            self._move_worker.item_done.connect(self._on_move_item_done)
            self._move_worker.all_done.connect(self._on_undo_move_all_done)
            self._move_worker.start()

    @pyqtSlot(list, list)
    def _on_undo_move_all_done(self, moved: list, errors: list):
        self._move_worker = None
        self._status_label.setText(f"Undo complete: {len(moved)} item(s) restored.")
        if errors:
            QMessageBox.warning(self, "Undo – errors",
                                "Some items could not be moved back:\n" + "\n".join(errors))
        if self._current_folder:
            self._grid.load_folder(self._current_folder)

    @pyqtSlot(int)
    def _on_cols_changed(self, value: int):
        self._grid.set_thumbnails_per_row(value)
        self._settings.thumbnails_per_row = value

    # ── sort slots ────────────────────────────────────────────────────────────────
    _LABEL_TO_KEY = {
        'Name': 'name', 'Date Modified': 'modified',
        'Size': 'size',  'Type': 'type', 'Rating': 'rating',
    }

    def _on_sort_changed(self, label: str):
        key = self._LABEL_TO_KEY.get(label, 'name')
        asc = self._sort_dir_btn.isChecked()
        self._settings.sort_key = key
        self._grid.set_sort(key, asc)

    def _on_sort_dir_toggled(self, checked: bool):
        self._sort_dir_btn.setText("↑ Asc" if checked else "↓ Desc")
        key = self._LABEL_TO_KEY.get(self._sort_combo.currentText(), 'name')
        self._settings.sort_asc = checked
        self._grid.set_sort(key, checked)

    @pyqtSlot()
    def _on_settings(self):
        dlg = SettingsDialog(self._settings, self)
        dlg.settings_changed.connect(self._apply_settings)
        dlg.exec()

    @pyqtSlot()
    def _on_go_up(self):
        if not self._current_folder:
            return
        parent = os.path.dirname(self._current_folder)
        if parent and parent != self._current_folder:
            self._navigate_to(parent)

    @pyqtSlot()
    def _on_open_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Open folder", self._current_folder or os.path.expanduser("~")
        )
        if folder:
            self._navigate_to(folder)

    def _apply_settings(self):
        self._cols_slider.setValue(self._settings.thumbnails_per_row)
        _metadata_dialog.set_custom_search_urls(self._settings.custom_search_urls)
        if self._current_folder:
            self._grid.load_folder(self._current_folder)

    # ── filter slot ─────────────────────────────────────────────────────────────
    @pyqtSlot(str)
    def _on_filter_changed(self, text: str):
        self._grid.set_filter(text)

    # ── batch rename ─────────────────────────────────────────────────────────────
    @pyqtSlot()
    def _on_batch_rename(self):
        paths = self._grid.get_checked_paths()
        if not paths:
            return
        dlg = BatchRenameDialog(paths, self)
        dlg.renames_done.connect(self._on_renames_done)
        dlg.exec()

    @pyqtSlot(list)
    def _on_renames_done(self, old_paths: list):
        """Refresh folder after batch rename."""
        if self._current_folder:
            self._grid.load_folder(self._current_folder)
        self._status_label.setText(f"Renamed {len(old_paths)} item(s).")

    @pyqtSlot()
    def _on_rename_key(self):
        """F2: open batch rename for checked items."""
        self._on_batch_rename()

    @pyqtSlot(str)
    def _on_rename_single(self, path: str):
        """Rename a single file via simple input dialog."""
        old_name = os.path.basename(path)
        new_name, ok = QInputDialog.getText(
            self, "Rename", "New name:", text=old_name
        )
        if not ok or not new_name.strip() or new_name == old_name:
            return
        dest = os.path.join(os.path.dirname(path), new_name.strip())
        try:
            os.rename(path, dest)
            if self._current_folder:
                self._grid.load_folder(self._current_folder)
            self._status_label.setText(f"Renamed to {new_name}")
        except OSError as exc:
            QMessageBox.warning(self, "Rename error", str(exc))

    # ── bookmarks ─────────────────────────────────────────────────────────────
    @pyqtSlot(str)
    def _on_bookmark_toggle(self, path: str):
        bms = self._settings.bookmarks
        if path in bms:
            bms.remove(path)
            self._breadcrumb.set_bookmarked(False)
        else:
            bms.append(path)
            self._breadcrumb.set_bookmarked(True)
        self._settings.bookmarks = bms
        self._tree.reload_bookmarks(bms)

    # ── duplicate finder ──────────────────────────────────────────────────────
    @pyqtSlot()
    def _on_find_duplicates(self):
        folder = self._current_folder or os.path.expanduser("~")
        dlg = DuplicateFinderDialog(folder, self._cache, self)
        dlg.exec()

    # ── bulk metadata fetch ───────────────────────────────────────────────────
    @pyqtSlot()
    def _on_fetch_all_metadata(self):
        paths = self._grid.get_all_video_paths()
        if not paths:
            self._status_label.setText("No videos to fetch metadata for.")
            return

        progress = QProgressDialog(
            "Fetching metadata…", "Cancel", 0, len(paths), self
        )
        progress.setWindowTitle("Fetch Metadata")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)

        worker = BulkMetaWorker(paths, self)
        self._bulk_meta_worker = worker

        @pyqtSlot(str, int, int)
        def _on_progress(current_file, done, total):
            if progress.wasCanceled():
                worker.stop()
                return
            progress.setValue(done)
            progress.setLabelText(f"Fetching: {os.path.basename(current_file)}")

        @pyqtSlot(str, dict)
        def _on_item_done(path, data):
            if data:
                self._cache.save_video_metadata(path, data)

        @pyqtSlot(int, int)
        def _on_all_done(successes, failures):
            progress.setValue(len(paths))
            progress.close()
            self._status_label.setText(
                f"Fetched metadata for {successes}/{successes + failures} videos."
            )
            self._bulk_meta_worker = None

        worker.progress.connect(_on_progress)
        worker.item_done.connect(_on_item_done)
        worker.all_done.connect(_on_all_done)
        progress.canceled.connect(worker.stop)
        worker.start()

    # ── network paths ─────────────────────────────────────────────────────────
    @pyqtSlot()
    def _on_add_network_path(self):
        path, ok = QInputDialog.getText(
            self, "Add Network Path",
            "Enter UNC or network path (e.g. \\\\server\\share):"
        )
        if not ok or not path.strip():
            return
        path = path.strip()
        if not os.path.isdir(path):
            QMessageBox.warning(self, "Invalid path",
                                f"Path is not accessible:\n{path}")
            return
        paths = self._settings.network_paths
        if path not in paths:
            paths.append(path)
            self._settings.network_paths = paths
        self._tree.add_network_path(path)

    # ── drag-and-drop (files_dropped) ─────────────────────────────────────────
    @pyqtSlot(list)
    def _on_files_dropped(self, paths: list):
        """Handle files/folders dropped onto the grid."""
        video_exts = AppSettings.VIDEO_EXTENSIONS
        valid = []
        for p in paths:
            if os.path.isdir(p):
                valid.append(p)
            elif os.path.isfile(p) and os.path.splitext(p)[1].lower() in video_exts:
                valid.append(p)
        if not valid:
            return
        if not self._current_folder:
            return
        # Ask: move or copy?
        reply = QMessageBox.question(
            self, "Drop files",
            f"Move or copy {len(valid)} item(s) to the current folder?\n\n"
            f"Current folder: {self._current_folder}\n\n"
            f"Yes = Move,  No = Copy,  Cancel = Abort",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No |
            QMessageBox.StandardButton.Cancel,
        )
        if reply == QMessageBox.StandardButton.Cancel:
            return
        copy_only = (reply == QMessageBox.StandardButton.No)

        self._generator.cancel_all()
        self._pending_move_orig_paths = list(valid)
        self._pending_move_dest = self._current_folder
        self._move_worker = _MoveWorker(valid, self._current_folder,
                                        copy_only=copy_only, parent=self)
        self._move_worker.item_done.connect(self._on_move_item_done)
        if copy_only:
            self._move_worker.all_done.connect(self._on_copy_all_done)
        else:
            self._move_worker.all_done.connect(self._on_move_all_done)
        self._move_worker.start()

    # ── quick preview (Feature 16) ────────────────────────────────────────────
    @pyqtSlot(int)
    def _on_quick_preview(self, idx: int):
        if not _HAS_QUICK_PREVIEW:
            return
        all_items   = self._grid.get_all_items()
        video_items = self._grid.get_video_items()
        preview_items = [
            {
                'path':     item.path,
                'pixmap':   item.pixmap,
                'rating':   item.rating,
                'duration': item.duration,
            }
            for item in video_items
        ]
        # Map full-list idx to video-only list index
        if idx < len(all_items):
            target_path = all_items[idx].path
            video_idx = next(
                (i for i, it in enumerate(video_items) if it.path == target_path), 0
            )
        else:
            video_idx = 0
        dlg = QuickPreviewDialog(preview_items, video_idx, self)
        dlg.exec()

    # ── watch toggled (Feature 17) ────────────────────────────────────────────
    @pyqtSlot(str, bool)
    def _on_watch_toggled(self, path: str, watched: bool):
        self._cache.set_watched(path, watched)
        self._grid.update_item_watched(path, watched)

    # ── tag editing (Feature 12) ──────────────────────────────────────────────
    @pyqtSlot()
    def _on_edit_tags(self):
        if not _HAS_TAG_DIALOG:
            return
        paths = self._grid.get_checked_paths()
        if not paths:
            return
        video_paths = [p for p in paths if os.path.isfile(p)]
        if not video_paths:
            return
        dlg = TagDialog(video_paths, self._cache, self)
        if dlg.exec() == dlg.DialogCode.Accepted:
            for p in video_paths:
                tags = self._cache.get_tags(p)
                self._grid.update_item_tags(p, tags)

    # ── folder size scanner (Feature 14) ──────────────────────────────────────
    @pyqtSlot()
    def _on_scan_folder_sizes(self):
        if not _HAS_FOLDER_SCANNER:
            return
        folder_items = [item for item in self._grid.get_all_items() if item.is_folder]
        if not folder_items:
            self._status_label.setText("No subfolders to scan.")
            return
        paths = [item.path for item in folder_items]
        self._status_label.setText(f"Scanning {len(paths)} folder sizes…")
        scanner = FolderSizeScanner(paths, self)
        scanner.folder_done.connect(self._on_folder_size_done)
        scanner.all_done.connect(lambda: self._status_label.setText("Folder scan complete."))
        scanner.start()
        self._folder_scanner = scanner  # keep reference

    @pyqtSlot(str, int)
    def _on_folder_size_done(self, path: str, size: int):
        self._grid.update_folder_size(path, size)

    # ── export (Feature 20) ───────────────────────────────────────────────────
    @pyqtSlot()
    def _on_export(self):
        if not _HAS_EXPORT:
            return
        items = self._grid.get_all_items()
        if not items:
            return
        dlg = ExportDialog(items, self._cache, self)
        dlg.exec()

    # ── collections (Feature 18) ──────────────────────────────────────────────
    @pyqtSlot()
    def _on_collections(self):
        if not _HAS_COLLECTIONS:
            return
        filter_text = self._filter_edit.text()
        sort_key = self._settings.sort_key
        sort_asc = self._settings.sort_asc
        dlg = CollectionsDialog(self._cache, filter_text, sort_key, sort_asc, self)
        dlg.collection_selected.connect(self._apply_collection)
        dlg.exec()

    @pyqtSlot(dict)
    def _apply_collection(self, col: dict):
        self._filter_edit.setText(col.get('filter_text', ''))
        label_map = {
            'name': 'Name', 'modified': 'Date Modified',
            'size': 'Size', 'type': 'Type', 'rating': 'Rating',
        }
        label = label_map.get(col.get('sort_key', 'name'), 'Name')
        self._sort_combo.setCurrentText(label)
        self._sort_dir_btn.setChecked(col.get('sort_asc', True))

    # ── status bar helpers ─────────────────────────────────────────────────────
    def _update_disk_summary(self):
        items = self._grid.get_all_items()
        n_folders = sum(1 for i in items if i.is_folder)
        n_videos  = sum(1 for i in items if not i.is_folder)
        total_size = sum(i.size for i in items if not i.is_folder)
        parts = []
        if n_folders:   parts.append(f"📁 {n_folders}")
        if n_videos:    parts.append(f"🎬 {n_videos}")
        if total_size > 0: parts.append(self._human_size(total_size))
        self._disk_label.setText("  ".join(parts))

    @staticmethod
    def _human_size(n: int) -> str:
        for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
            if n < 1024:
                return f"{n} B" if unit == 'B' else f"{n:.1f} {unit}"
            n /= 1024
        return f"{n:.1f} PB"

    # ------------------------------------------------------------------ tree slot
    def _on_tree_item_changed(self, current: QTreeWidgetItem, _previous):
        if current is None:
            return
        path = current.data(0, Qt.ItemDataRole.UserRole)
        if path and os.path.isdir(path) and path != self._current_folder:
            self._navigate_to(path)

    # ------------------------------------------------------------------ grid signals
    @pyqtSlot(int)
    def _on_selection_changed(self, count: int):
        self._sel_label.setText(f"{count} selected" if count else "")
        self._act_move.setEnabled(count > 0)
        self._act_copy_sel.setEnabled(count > 0)
        self._act_delete.setEnabled(count > 0)
        self._act_batch_rename.setEnabled(count > 0)
        self._act_edit_tags.setEnabled(count > 0 and _HAS_TAG_DIALOG)

    @pyqtSlot(str)
    def _on_grid_status(self, msg: str):
        self._status_label.setText(msg)
        self._update_disk_summary()

    # ------------------------------------------------------------------ window events
    def closeEvent(self, event):
        self._settings.window_geometry = self.saveGeometry()
        self._settings.splitter_state = self._splitter.saveState()
        self._generator.cancel_all()
        if self._player_panel is not None:
            if self._player_panel.isVisible():
                self._settings.player_geometry = self._player_panel.saveGeometry()
            self._player_panel.stop()
            self._player_panel.hide()   # prevent it outliving the main window
        super().closeEvent(event)
