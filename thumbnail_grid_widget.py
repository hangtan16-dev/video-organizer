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
import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass, field

from PyQt6.QtWidgets import QScrollArea, QWidget, QLabel, QApplication
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QMimeData, QUrl, QThread, QObject
from PyQt6.QtGui import QImage, QPixmap, QKeyEvent, QFont, QDrag

from video_thumbnail_widget import VideoThumbnailWidget, BOTTOM_H
from folder_thumbnail_widget import FolderThumbnailWidget
from thumbnail_generator import ThumbnailGeneratorService
from app_settings import AppSettings
from cache_manager import CacheManager

_logger = logging.getLogger(__name__)

_SPACING          = 8
_BATCH_SIZE       = 30    # items added per event-loop tick during folder load
_BUFFER_PX        = 600   # pixels above/below visible viewport to keep widgets alive
_PIXMAP_CACHE_MAX = 150   # max QPixmaps kept in RAM (≈ 150 × ~1.4 MB ≈ 210 MB cap)
_THUMB_DISPLAY_MAX_W = 800  # width thumbnails are scaled to in RAM / on reload
_FOLDER_SCROLL_MAX = 200  # max remembered scroll positions

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
    thumbnail_failed: bool = False  # cv2 couldn't read this file last time


# ── Background folder scanner ────────────────────────────────────────────────
# The big win: every os.scandir / os.stat / os.path.exists call happens here,
# not on the GUI thread. For 900+ files on an HDD that's ~7 disk hits per file
# (one stat + six sidecar candidates) ≈ 6000 syscalls — synchronously that's
# tens of seconds. We do it off-thread and emit the finished _Item list.
#
# Cancellation works by setting `cancelled` — the loop checks it between
# stat calls, so the worker bails out quickly when the user navigates away.

_SIDECAR_EXTS = ('.srt', '.sub', '.ass', '.vtt', '.smi')

# Module-level strong refs to running scan threads. Keeps the QThread + worker
# alive so Python GC can't free their wrappers before `finished` has actually
# fired and Qt has finalised the OS thread. Entries are removed automatically
# by the `finished` signal handler set up in ThumbnailGridWidget.__init__.
_ACTIVE_SCAN_THREADS: 'set[tuple]' = set()


class _FolderScanWorker(QObject):
    """Runs on a dedicated QThread. Lives for the lifetime of the grid widget.

    `do_scan` is a slot — the grid emits its `_scan_request` signal across a
    queued connection, which Qt routes here on the worker thread. Each new
    scan call cancels any in-progress scan via the lock-protected token.
    """
    scan_done = pyqtSignal(object)   # emits a dict (see _scan_impl) or None on cancel

    def __init__(self):
        super().__init__()
        self._lock = threading.Lock()
        self._current_token: int = 0      # incremented per scan; old scans abort
        self._cancelled = False           # flipped to True when a new scan starts

    def cancel(self):
        with self._lock:
            self._cancelled = True

    def do_scan(self, folder_path: str, recursive: bool,
                exts: object, default_seek: float, token: int):
        """Slot — runs on the worker thread (queued connection from grid)."""
        # Cancel any in-flight scan; this one is now current.
        with self._lock:
            self._cancelled = False
            self._current_token = token

        try:
            result = self._scan_impl(folder_path, bool(recursive),
                                     frozenset(exts), float(default_seek), int(token))
        except Exception:
            _logger.exception("Folder scan crashed for %s", folder_path)
            result = None
        # Always emit so the grid knows the scan is done (even on cancel/error).
        self.scan_done.emit(result)

    def _is_cancelled(self, token: int) -> bool:
        with self._lock:
            return self._cancelled or token != self._current_token

    def _scan_impl(self, folder_path, recursive, exts,
                   default_seek, token):
        if not folder_path or not os.path.isdir(folder_path):
            return {'folder_path': folder_path,
                    'raw_dirs': [], 'raw_files': [],
                    'token': token, 'error': 'not-a-folder'}

        raw_dirs_data: list[tuple[str, float, int, int]] = []   # (path, mtime, child_count, video_count)
        # Each file entry: (path, mtime, size, sub_exists, nfo_exists)
        raw_files_data: list[tuple[str, float, int, bool, bool]] = []

        try:
            if recursive:
                # Walk subtree, capturing files; no folder cards in recursive mode.
                for entry in _iter_videos_recursive_safe(folder_path, exts):
                    if self._is_cancelled(token):
                        return None
                    try:
                        st = entry.stat()
                        mtime, size = st.st_mtime, st.st_size
                    except OSError:
                        mtime, size = 0.0, 0
                    base = os.path.splitext(entry.path)[0]
                    sub_exists = any(os.path.exists(base + e) for e in _SIDECAR_EXTS)
                    nfo_exists = os.path.exists(base + '.nfo')
                    raw_files_data.append((entry.path, mtime, size,
                                           sub_exists, nfo_exists))
            else:
                try:
                    all_entries = list(os.scandir(folder_path))
                except PermissionError:
                    return {'folder_path': folder_path,
                            'raw_dirs': [], 'raw_files': [],
                            'token': token, 'error': 'permission-denied'}

                # Split entries; do NOT call stat() during the scandir loop
                # because that would force another syscall.
                dir_entries  = []
                file_entries = []
                for e in all_entries:
                    if self._is_cancelled(token):
                        return None
                    try:
                        if e.is_dir(follow_symlinks=False):
                            if not e.name.startswith('.'):
                                dir_entries.append(e)
                        elif e.is_file():
                            if os.path.splitext(e.name)[1].lower() in exts:
                                file_entries.append(e)
                    except OSError:
                        continue

                for d in dir_entries:
                    if self._is_cancelled(token):
                        return None
                    try:
                        st = d.stat()
                        mtime = st.st_mtime
                    except OSError:
                        mtime = 0.0
                    # Folder child / video count (3-level deep scandir scan).
                    # Doing this here keeps it off the GUI thread.
                    total, videos = _scan_folder(d.path)
                    raw_dirs_data.append((d.path, mtime, total, videos))

                for f in file_entries:
                    if self._is_cancelled(token):
                        return None
                    try:
                        st = f.stat()
                        mtime, size = st.st_mtime, st.st_size
                    except OSError:
                        mtime, size = 0.0, 0
                    base = os.path.splitext(f.path)[0]
                    sub_exists = any(os.path.exists(base + e) for e in _SIDECAR_EXTS)
                    nfo_exists = os.path.exists(base + '.nfo')
                    raw_files_data.append((f.path, mtime, size,
                                           sub_exists, nfo_exists))
        except OSError:
            _logger.exception("Folder scan OSError on %s", folder_path)
            return {'folder_path': folder_path,
                    'raw_dirs': [], 'raw_files': [],
                    'token': token, 'error': 'os-error'}

        if self._is_cancelled(token):
            return None

        # Return raw filesystem data. The main thread (phase 2) does the
        # bulk DB lookups (fast, indexed by path) and builds final _Item
        # records. Splitting it this way keeps the worker dependency-free
        # — no CacheManager reference, so we don't have to worry about
        # SQLite connection thread-affinity.
        # raw_dirs_data: list[(path, mtime)]
        # raw_files_data: list[(path, mtime, size, sub_exists, nfo_exists)]
        return {'folder_path': folder_path,
                'raw_dirs':  raw_dirs_data,
                'raw_files': raw_files_data,
                'token': token,
                'error': None}


def _iter_videos_recursive_safe(root, exts):
    """Standalone copy of ThumbnailGridWidget._iter_videos_recursive so the
    background worker doesn't need a widget reference."""
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as it:
                for entry in it:
                    try:
                        if entry.name.startswith('.'):
                            continue
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(entry.path)
                        elif entry.is_file(follow_symlinks=False):
                            if os.path.splitext(entry.name)[1].lower() in exts:
                                yield entry
                    except OSError:
                        continue
        except (PermissionError, FileNotFoundError, OSError):
            continue


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
    # Internal: marshal a scan request onto the worker thread (queued connection).
    # Args: folder_path, recursive, exts, default_seek, token
    _scan_request          = pyqtSignal(str, bool, object, float, int)

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

        # ── recursive view (show all videos in folder + all subfolders) ─────
        self._recursive: bool = settings.recursive_view

        # ── filter state ─────────────────────────────────────────────────────
        self._filter_text: str = ''
        # Star-rating filter: None = show all; 0-5 = show ONLY videos whose
        # rating equals this value (0 = unrated/zero-star). Exact match, NOT
        # ">=". Folders are exempt (always shown) so navigation still works.
        self._rating_filter: 'int | None' = None

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

        # ── LRU pixmap cache (bounded) ───────────────────────────────────────
        # Stores (QPixmap, duration) keyed by video path.
        # Capped at _PIXMAP_CACHE_MAX so memory stays bounded regardless of
        # how many videos are in a folder.
        self._pixmap_cache: OrderedDict[str, tuple[object, float]] = OrderedDict()

        # ── scroll position memory (bounded) ─────────────────────────────────
        self._folder_scroll:       dict[str, int] = {}
        self._current_folder_path: str            = ''
        # True once the user scrolls after a folder load — used to NOT yank the
        # scroll back to the remembered position if they've already taken over.
        self._user_scrolled_since_load: bool      = False

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

        # Per-path retry counter for TRANSIENT thumbnail failures (disk
        # contention / read-timeouts during a mass regen + scroll). Bounded so
        # a genuinely unreadable file can't retry forever. Cleared on success.
        self._thumb_retry: 'dict[str, int]' = {}

        # ── signal connections ───────────────────────────────────────────────
        self._generator.thumbnail_ready.connect(self._on_thumbnail_ready)
        self._generator.thumbnail_failed.connect(self._on_thumbnail_failed)
        self.verticalScrollBar().valueChanged.connect(self._on_scroll)

        # Debounce resize (60 ms) and scroll-end (60 ms) before calling layout
        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.timeout.connect(self._full_relayout)

        self._scroll_timer = QTimer(self)
        self._scroll_timer.setSingleShot(True)
        self._scroll_timer.timeout.connect(self._update_active_widgets)

        # ── background folder-scan thread ───────────────────────────────────
        # All os.scandir / os.stat / os.path.exists work happens here so the
        # main thread can keep painting while we walk a slow HDD folder.
        #
        # Lifecycle: we deliberately DO NOT parent the QThread to `self`,
        # because Qt's parent-child destruction would tear the thread down
        # while its event loop is still running. Instead:
        #   * widget.destroyed → quit() the thread's event loop
        #   * thread.finished → deleteLater on both worker and thread
        #   * Module-level set keeps a strong ref so Python GC can't kill
        #     the wrappers before the C++ side has cleanly stopped.
        self._scan_thread = QThread()
        self._scan_thread.setObjectName("FolderScanThread")
        self._scan_worker = _FolderScanWorker()
        self._scan_worker.moveToThread(self._scan_thread)
        # Queued: signal emitted on main thread → slot runs on worker thread.
        self._scan_request.connect(
            self._scan_worker.do_scan, Qt.ConnectionType.QueuedConnection)
        # Queued: signal emitted on worker thread → slot runs on main thread.
        self._scan_worker.scan_done.connect(
            self._on_scan_done, Qt.ConnectionType.QueuedConnection)
        _ACTIVE_SCAN_THREADS.add((self._scan_thread, self._scan_worker))
        # When the widget dies, quit the thread loop; when the thread
        # finishes, delete the C++ objects and drop our strong ref.
        #
        # The destroyed→quit lambda must be DEFENSIVE: by the time it
        # fires, the QThread C++ object may already be gone (because
        # `finished → deleteLater` ran earlier in the same shutdown
        # sequence). Calling quit() on a freed wrapper raises
        # `RuntimeError: wrapped C/C++ object of type QThread has been
        # deleted`, which propagates as an unhandled exception into Qt's
        # signal cleanup and has been observed to crash the process
        # (access violation, faulting module varies by which DLL Qt is
        # tearing down at the moment).
        thread_ref = self._scan_thread
        worker_ref = self._scan_worker
        def _safe_quit(*_):
            try:
                if thread_ref.isRunning():
                    thread_ref.quit()
            except RuntimeError:
                pass   # Underlying C++ QThread already deleted — nothing to do
        self.destroyed.connect(_safe_quit)
        self._scan_thread.finished.connect(thread_ref.deleteLater)
        self._scan_thread.finished.connect(worker_ref.deleteLater)
        self._scan_thread.finished.connect(
            lambda: _ACTIVE_SCAN_THREADS.discard((thread_ref, worker_ref)))
        self._scan_thread.start()
        # Monotonic token so a stale scan result for an old folder is ignored.
        self._scan_token: int = 0
        # Folder we're currently scanning (cleared when result arrives).
        self._scanning_folder: str = ''

    # ── public API ─────────────────────────────────────────────────────────────
    def load_folder(self, folder_path: str):
        """Kick off folder load (non-blocking).

        All filesystem I/O — scandir, stat, sidecar existence checks — happens
        on the dedicated `_scan_thread`. The main thread returns immediately
        after showing a "Scanning…" status; `_on_scan_done` builds the UI when
        the worker emits a result. This keeps the GUI responsive on slow HDDs
        even for folders with hundreds of files."""
        # Save scroll position for the folder we're leaving (cap dict size)
        if self._current_folder_path and self._items:
            self._folder_scroll[self._current_folder_path] = self.verticalScrollBar().value()
            if len(self._folder_scroll) > _FOLDER_SCROLL_MAX:
                # Drop the oldest entry
                self._folder_scroll.pop(next(iter(self._folder_scroll)))

        self._current_folder_path = folder_path
        self._scanning_folder     = folder_path
        self._user_scrolled_since_load = False

        # Cancel any in-flight scan first so it short-circuits quickly.
        self._scan_worker.cancel()
        self._generator.cancel_all()
        self._clear_all()
        # Reset the disk coordinator to a clean idle state. All previews and
        # thumbnail workers for the OLD folder have just been stopped/cancelled
        # above, so any foreground hold or background count left over is stale.
        # Belt-and-suspenders against a gate leak (e.g. a preview thread that
        # was force-killed without running its end_foreground finally).
        try:
            from disk_coordinator import COORDINATOR
            COORDINATOR.reset()
        except Exception:
            pass

        if not folder_path or not os.path.isdir(folder_path):
            self.status_message.emit("No folder selected")
            return

        # Two-phase load to keep the GUI thread responsive:
        #   phase 1 (worker thread): walk filesystem, collect tuples
        #   phase 2 (main thread, on scan_done): bulk-query DB, build _Items
        default_seek = self._settings.seek_time
        exts         = AppSettings.VIDEO_EXTENSIONS

        self._scan_token += 1
        token = self._scan_token

        self.status_message.emit("Scanning folder…")

        # Cross-thread marshal: the signal is queued-connected to the worker,
        # so this returns immediately and the heavy I/O runs on _scan_thread.
        self._scan_request.emit(folder_path, bool(self._recursive),
                                exts, float(default_seek), int(token))

    def _on_scan_done(self, result):
        """Phase 2 — runs on main thread after the worker finishes scanning.

        The worker handed us cheap dataclass-friendly tuples; here we do the
        fast DB lookups (all indexed-by-path queries hit the SQLite WAL cache)
        and build the final _Item list, then start the batched widget build.
        """
        # Stale or cancelled result.
        if not result:
            return
        # Token check guards against late results from an aborted scan.
        if result.get('token') != self._scan_token:
            return
        if result.get('folder_path') != self._scanning_folder:
            return

        self._scanning_folder = ''
        err = result.get('error')
        if err == 'permission-denied':
            self.status_message.emit(f"Permission denied: {result['folder_path']}")
            return
        if err == 'not-a-folder':
            self.status_message.emit("No folder selected")
            return

        raw_dirs  = result['raw_dirs']
        raw_files = result['raw_files']

        # Phase 2: bulk DB lookups (5 indexed queries, microseconds).
        file_paths = [f[0] for f in raw_files]
        try:
            seek_overrides = self._cache.get_seek_overrides_bulk(file_paths)
            ratings_map    = self._cache.get_ratings_bulk(file_paths)
            watched_set    = self._cache.get_watched_bulk(file_paths)
            tags_map       = self._cache.get_tags_bulk(file_paths)
            failed_set     = self._cache.get_failed_paths_bulk(file_paths)
        except Exception:
            _logger.exception("Bulk DB lookup failed in _on_scan_done")
            seek_overrides, ratings_map, watched_set = {}, {}, set()
            tags_map, failed_set = {}, set()
        default_seek = self._settings.seek_time

        pending: list[_Item] = []
        for (dpath, dmtime, dchild, dvideo) in raw_dirs:
            pending.append(_Item(path=dpath, is_folder=True,
                                 child_count=dchild, video_count=dvideo,
                                 mtime=dmtime))
        for (fpath, fmtime, fsize, fsub, fnfo) in raw_files:
            seek = seek_overrides.get(fpath, default_seek)
            pending.append(_Item(
                path=fpath, seek_time=seek, mtime=fmtime, size=fsize,
                rating=ratings_map.get(fpath, 0),
                tags=tags_map.get(fpath, []),
                is_watched=(fpath in watched_set),
                sub_exists=fsub,
                nfo_exists=fnfo,
                thumbnail_failed=(fpath in failed_set),
            ))

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

        self._empty_label.hide()

        n_dirs, n_files = len(raw_dirs), len(raw_files)
        parts = []
        if n_dirs:  parts.append(f"{n_dirs} folder{'s' if n_dirs != 1 else ''}")
        if n_files: parts.append(f"{n_files} video{'s' if n_files != 1 else ''}")
        self.status_message.emit(f"Loading {', '.join(parts)}…")
        self.selection_changed.emit(0)
        folder_path = result['folder_path']
        QTimer.singleShot(0, lambda: self._create_next_batch(folder_path))

    def set_filter(self, text: str):
        """Filter visible items to those whose name contains text (case-insensitive)."""
        self._filter_text = text.lower().strip()
        self._full_relayout()
        self.status_message.emit(self._items_summary())

    def set_rating_filter(self, stars: 'int | None'):
        """Show ONLY videos whose rating equals `stars` (exact match).
        `stars` 0 = unrated/zero-star, 1-5 = that many stars, None = show all
        (filter off). Folders are exempt — they stay visible so the user can
        still navigate while a rating filter is active. ANDed with the text
        filter."""
        if stars is not None:
            stars = max(0, min(5, int(stars)))
        if stars == self._rating_filter:
            return
        self._rating_filter = stars
        self._full_relayout()
        self.status_message.emit(self._items_summary())

    def set_recursive(self, on: bool):
        """Toggle recursive folder traversal. When on, the grid shows every
        video file inside the current folder AND all of its descendants;
        subfolders themselves are hidden. When off, the grid shows only
        direct children (current default behaviour)."""
        on = bool(on)
        if on == self._recursive:
            return
        self._recursive = on
        self._settings.recursive_view = on
        if self._current_folder_path:
            self.load_folder(self._current_folder_path)

    def is_recursive(self) -> bool:
        return self._recursive

    @staticmethod
    def _iter_videos_recursive(root: str, exts: 'set[str] | frozenset[str]'):
        """Yield os.DirEntry-like objects for every video file under root.

        Uses os.scandir at each level (much faster than os.walk on Windows),
        skips hidden directories (leading dot) and symlink loops, and
        swallows per-folder PermissionError so an unreadable subdirectory
        doesn't abort the whole scan."""
        # We yield real DirEntry objects so callers can keep using .path / .stat()
        stack = [root]
        while stack:
            current = stack.pop()
            try:
                with os.scandir(current) as it:
                    for entry in it:
                        try:
                            if entry.name.startswith('.'):
                                continue
                            if entry.is_dir(follow_symlinks=False):
                                stack.append(entry.path)
                            elif entry.is_file(follow_symlinks=False):
                                if os.path.splitext(entry.name)[1].lower() in exts:
                                    yield entry
                        except OSError:
                            continue   # skip individual unreadable entries
            except (PermissionError, FileNotFoundError, OSError):
                continue   # whole subtree unreadable — skip it

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
        """Return True if the item should be SHOWN (not filtered out).
        Combines the rating filter AND the text filter."""
        # Star-rating filter (exact match). Applies to VIDEOS only; folders are
        # always shown so the user can navigate while filtering by rating.
        if self._rating_filter is not None and not item.is_folder:
            if item.rating != self._rating_filter:
                return False

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
            # Queue thumbnail generation for video files only.
            # Skip files we already know cv2 can't decode — saves CPU AND
            # avoids triggering the FFmpeg stderr noise on every reopen.
            if not item.is_folder and not item.thumbnail_failed:
                self._generator.request_thumbnail(item.path, item.seek_time)

        self._batch_idx += _BATCH_SIZE
        # Recompute layout + create/destroy widgets for current viewport
        self._full_relayout()

        if self._batch_idx < len(self._pending_items):
            QTimer.singleShot(0, lambda: self._create_next_batch(folder_path))
        else:
            self.status_message.emit(self._items_summary())
            # Restore saved scroll position for this folder — but only if the
            # user hasn't already started scrolling during the load.
            saved = self._folder_scroll.get(folder_path, 0)
            if saved > 0:
                QTimer.singleShot(50, lambda: self._restore_folder_scroll(saved))

    # ── layout ─────────────────────────────────────────────────────────────────
    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Resize, like scroll, is a foreground UI activity — suspend disk
        # work so the relayout stays smooth (non-blocking).
        try:
            from disk_coordinator import COORDINATOR
            COORDINATOR.note_ui_activity()
        except Exception:
            pass
        self._resize_timer.start(60)

    def _is_item_filtered(self, item: '_Item') -> bool:
        """Return True if this item should be hidden by the current filter."""
        return not self._item_matches_filter(item)

    def _full_relayout(self):
        """Recompute every item's geometry, resize the canvas, refresh widgets.

        A scroll ANCHOR (the item under the top of the viewport) is captured
        before re-flowing and re-pinned afterwards, so row-height changes — e.g.
        a thumbnail arriving and replacing the 16:9 placeholder aspect ratio for
        an item ABOVE the viewport — don't shift what the user is looking at.
        Without this the grid visibly "jumps" (usually upward, since VR
        thumbnails are taller than 16:9) while scrolling a folder whose
        thumbnails are still generating."""
        bar = self.verticalScrollBar()
        anchor_idx, anchor_delta = self._capture_scroll_anchor(bar.value())

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

        # Re-pin the scroll anchor so this relayout doesn't move the content
        # currently under the viewport (no-op when scrolled to the very top).
        self._restore_scroll_anchor(anchor_idx, anchor_delta)

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

    def _capture_scroll_anchor(self, sv: int) -> 'tuple[int | None, int]':
        """Identify the item at the top of the viewport and how far the viewport
        top sits below that item's top, from the CURRENT layout. Returns
        (anchor_idx, delta). (None, 0) when at the top or empty — no pinning
        needed. The layout cache is row-major (increasing y), so we take the
        last item whose top is at/above the scroll value and stop."""
        if sv <= 0 or not self._layout_cache:
            return None, 0
        anchor_idx = None
        anchor_y = 0
        for idx, rect in enumerate(self._layout_cache):
            if rect == _FILTERED_SENTINEL:
                continue
            y = rect[1]
            if y <= sv:
                anchor_idx, anchor_y = idx, y
            else:
                break
        if anchor_idx is None:
            return None, 0
        return anchor_idx, sv - anchor_y

    def _restore_scroll_anchor(self, anchor_idx: 'int | None', anchor_delta: int):
        """Re-pin the captured anchor item to the same viewport offset after a
        relayout — WITHOUT emitting valueChanged, so it isn't mistaken for a
        user scroll (which would stop previews / re-trigger work)."""
        if anchor_idx is None or anchor_idx >= len(self._layout_cache):
            return
        rect = self._layout_cache[anchor_idx]
        if rect == _FILTERED_SENTINEL:
            return
        bar = self.verticalScrollBar()
        target = max(0, rect[1] + anchor_delta)
        if target == bar.value():
            return
        blocked = bar.blockSignals(True)
        bar.setValue(target)
        bar.blockSignals(blocked)

    def _restore_folder_scroll(self, saved: int):
        """Return to the scroll position last seen in this folder — unless the
        user has already taken over scrolling since the load began (in which
        case yanking them back would itself be the 'jumps on its own' glitch)."""
        if self._user_scrolled_since_load:
            return
        bar = self.verticalScrollBar()
        blocked = bar.blockSignals(True)
        bar.setValue(saved)
        bar.blockSignals(blocked)
        self._update_active_widgets()

    def _buffered_visible_range(self) -> tuple[int, int]:
        """Return (y_top, y_bottom) of the buffered visible region."""
        sv   = self.verticalScrollBar().value()
        vp_h = self.viewport().height()
        return sv - _BUFFER_PX, sv + vp_h + _BUFFER_PX

    def _load_disk_thumbnail(self, item) -> 'tuple[object, float] | None':
        """Synchronously load `item`'s thumbnail from the on-disk JPEG cache,
        scaled to display width. Returns (QPixmap, duration) or None.

        Used when a widget is (re)created but its pixmap was evicted from the
        bounded RAM LRU (after the user scrolled past many videos): the JPEG
        still exists on disk, so reload it directly instead of flashing back to
        "Generating…" or re-queueing a decode. This is a SMALL, BOUNDED read
        (a ~200 KB JPEG, scaled-decoded to ≤800 px in a few ms) — NOT the
        multi-GB video read the off-GUI-thread design exists to avoid — so doing
        it on the GUI thread keeps the thumbnail instant with no perceptible
        hitch."""
        try:
            disk = self._cache.get_thumbnail_path(item.path, item.seek_time)
            if not disk:
                return None
            from PyQt6.QtGui import QImageReader, QPixmap
            from PyQt6.QtCore import QSize
            reader = QImageReader(disk)
            reader.setAutoTransform(True)
            sz = reader.size()
            if sz.isValid() and sz.width() > _THUMB_DISPLAY_MAX_W:
                h = max(1, round(sz.height() * _THUMB_DISPLAY_MAX_W / sz.width()))
                reader.setScaledSize(QSize(_THUMB_DISPLAY_MAX_W, h))
            img = reader.read()
            if img.isNull():
                return None
            pix = QPixmap.fromImage(img)
            if pix.isNull():
                return None
            dur = self._cache.get_duration(item.path) or item.duration
            return (pix, dur)
        except Exception:
            return None

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
                # Pass file size + settings so the widget can adapt hover
                # preview strategy (skip / throttle) for large VR files.
                widget = VideoThumbnailWidget(
                    item.path, item.seek_time, self._container,
                    file_size=item.size, settings=self._settings,
                )
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
                cached = self._pixmap_cache.get(item.path)
                if cached is None:
                    # RAM-LRU miss — the thumbnail may already exist on disk
                    # (generated earlier, then evicted from the bounded RAM LRU
                    # when the user scrolled past many videos). Reload it from
                    # the on-disk JPEG cache so it reappears INSTANTLY instead of
                    # flashing back to "Generating…" or being regenerated. This
                    # is the fix for "thumbnail reverts to Generating after
                    # scrolling away and back".
                    disk = self._load_disk_thumbnail(item)
                    if disk is not None:
                        self._pixmap_cache[item.path] = disk
                        if len(self._pixmap_cache) > _PIXMAP_CACHE_MAX:
                            self._pixmap_cache.popitem(last=False)
                        cached = disk
                if cached is not None:
                    self._pixmap_cache.move_to_end(item.path)   # mark recently used
                    widget.set_thumbnail(cached[0], cached[1])
                elif item.thumbnail_failed and hasattr(widget, 'set_thumbnail_failed'):
                    # Already-known-bad file: stop the shimmer immediately
                    # rather than pretending we're loading it.
                    widget.set_thumbnail_failed("Cannot read frame")
                widget.set_rating(item.rating)
                # In recursive mode, show the relative path in the tooltip so
                # the user can see which subfolder each video lives in.
                if self._recursive and self._current_folder_path:
                    try:
                        rel = os.path.relpath(item.path, self._current_folder_path)
                        widget.setToolTip(rel)
                    except ValueError:
                        widget.setToolTip(item.path)

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
        # The user has taken over scrolling — don't auto-restore to the
        # remembered folder position out from under them.
        self._user_scrolled_since_load = True
        # Scrolling is a foreground UI activity and must take priority over
        # disk work. The instant the user scrolls, tell the disk coordinator
        # to abort in-flight thumbnail generation and keep background parked
        # (NON-BLOCKING — never waits, never touches disk on the GUI thread).
        # Without this, 3 thumbnail workers + a preview saturating the HDD
        # make scrolling stutter/freeze. See disk_coordinator.note_ui_activity.
        try:
            from disk_coordinator import COORDINATOR
            COORDINATOR.note_ui_activity()
        except Exception:
            pass
        # Also signal-stop any running hover previews so they stop competing
        # for the disk. Non-blocking: stop() just sets a flag + closes a file.
        self._stop_all_previews()
        # Debounce: give the user 60 ms to finish a scroll gesture
        self._scroll_timer.start(60)

    def _stop_all_previews(self):
        """Stop the (single) hover preview. Non-blocking. Routes through the
        global preview manager so there's one authority over the preview
        thread's lifecycle."""
        try:
            from pyav_play_thread import PREVIEW_MANAGER
            PREVIEW_MANAGER.cancel_all()
        except Exception:
            pass
        # Clear each widget's stale handle so it doesn't think it's playing.
        for widget in list(self._active.values()):
            try:
                if getattr(widget, '_play_thread', None) is not None:
                    widget._play_thread = None
            except (RuntimeError, AttributeError):
                pass

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

        # A result arrived → the file is readable after all. Clear any
        # failed-state + retry counter so recycled widgets show the thumbnail,
        # not a stale "Cannot read frame".
        item.thumbnail_failed = False
        self._thumb_retry.pop(video_path, None)

        # Update lightweight item metadata only (no pixmap stored in _Item)
        item.duration = duration
        if pix.height() > 0:
            item.aspect_ratio = pix.width() / pix.height()

        # ── LRU cache: insert / refresh, evict oldest when over limit ─────────
        self._pixmap_cache.pop(video_path, None)
        self._pixmap_cache[video_path] = (pix, duration)
        if len(self._pixmap_cache) > _PIXMAP_CACHE_MAX:
            self._pixmap_cache.popitem(last=False)   # evict least-recently-used

        # Push to live widget if one exists for this item right now
        widget = self._active.get(idx)
        if widget is not None:
            widget.set_thumbnail(pix, duration)

        # Aspect ratio may have changed → debounce a full relayout
        self._resize_timer.start(120)

    # Reasons that are TRANSIENT (the file opened fine; a frame just couldn't
    # be read in time) and therefore worth auto-retrying rather than showing a
    # permanent "Cannot read frame". Hard failures (e.g. "Cannot open video")
    # are NOT here — those stay failed.
    _RETRYABLE_THUMB_REASONS = ("Could not read frame", "Timeout")
    _MAX_THUMB_RETRIES = 5

    def _on_thumbnail_failed(self, video_path: str, reason: str):
        """Generator couldn't produce a thumbnail.

        Transient failures (disk contention / read-timeouts during a mass
        regen + scroll) are AUTO-RETRIED with exponential backoff instead of
        being left blank with 'Cannot read frame' forever — the file opened
        fine, we just lost the race for the disk. Only after several failed
        retries (or for a hard open failure) do we give up: cache the failure
        and show the placeholder."""
        idx = self._path_to_idx.get(video_path)
        if idx is None:
            return
        item   = self._items[idx]
        widget = self._active.get(idx)

        if reason in self._RETRYABLE_THUMB_REASONS:
            n = self._thumb_retry.get(video_path, 0)
            if n < self._MAX_THUMB_RETRIES:
                self._thumb_retry[video_path] = n + 1
                # Keep it retryable (not permanently failed). Show the
                # placeholder meanwhile; a successful retry replaces it.
                item.thumbnail_failed = False
                if widget is not None and hasattr(widget, 'set_thumbnail_failed'):
                    widget.set_thumbnail_failed(reason)
                from PyQt6.QtCore import QTimer
                # Backoff 2,4,8,16,30s — long enough to outlast the disk
                # contention from a folder-wide regen before we give up.
                delay_ms = min(30_000, int(2_000 * (2 ** n)))
                QTimer.singleShot(
                    delay_ms, lambda p=video_path: self._retry_thumbnail(p))
                return
            # Retries exhausted → treat as a genuine failure: cache it under a
            # PERMANENT reason (NOT the transient one, which the cache ignores)
            # so we don't keep hammering a file the disk truly can't serve.
            try:
                self._cache.mark_thumbnail_failed(
                    video_path, "Repeated read failures")
            except Exception:
                pass

        item.thumbnail_failed = True
        if widget is not None and hasattr(widget, 'set_thumbnail_failed'):
            widget.set_thumbnail_failed(reason)

    def _retry_thumbnail(self, video_path: str):
        """Re-queue thumbnail generation for a transiently-failed file (see
        _on_thumbnail_failed). Runs on the GUI thread via a one-shot timer."""
        idx = self._path_to_idx.get(video_path)
        if idx is None:
            return                      # folder changed — item gone
        item = self._items[idx]
        if item.is_folder or not os.path.exists(item.path):
            return
        # Clear any cached failure (belt-and-suspenders) and re-request. The
        # generator dedups by path, so a still-pending worker won't double-run.
        try:
            self._cache.clear_thumbnail_failure(video_path)
        except Exception:
            pass
        item.thumbnail_failed = False
        try:
            self._generator.request_thumbnail(video_path, item.seek_time)
        except Exception:
            _logger.exception("thumbnail retry failed for %s", video_path)

    # ── slots from individual widgets ───────────────────────────────────────────
    def _on_check_changed(self, path: str, checked: bool):
        idx = self._path_to_idx.get(path)
        if idx is not None:
            self._items[idx].checked = checked
        # Count directly — state was just updated above; no need to sync widgets
        self.selection_changed.emit(sum(1 for item in self._items if item.checked))

    def _on_seek_requested(self, video_path: str, seek_time: float):
        """Persist the user's chosen seek_time and regenerate the static
        thumbnail at the new position.

        This is now SAFE to do on every seek because the regeneration
        runs inside a disk-coordinator background_section — it can never
        overlap a foreground hover preview. If the user is still
        interacting, the coordinator makes the regen wait its turn; it
        never thrashes the disk against playback. (Before the coordinator
        existed, this path was the cause of the "freeze after a few
        videos" hang, so it was disabled; the coordinator lets us turn it
        back on.)

        Debounced 600 ms so dragging the slider back and forth only
        spawns ONE regen worker after the user lands on a position.
        """
        idx = self._path_to_idx.get(video_path)
        if idx is not None:
            self._items[idx].seek_time = seek_time
            self._items[idx].thumbnail_failed = False
        self._pixmap_cache.pop(video_path, None)
        if not hasattr(self, '_seek_thumb_debounce'):
            from PyQt6.QtCore import QTimer
            self._seek_thumb_debounce = QTimer(self)
            self._seek_thumb_debounce.setSingleShot(True)
            self._seek_thumb_pending: 'tuple[str, float] | None' = None
            self._seek_thumb_debounce.timeout.connect(
                self._fire_pending_seek_thumb)
        self._seek_thumb_pending = (video_path, seek_time)
        self._seek_thumb_debounce.start(600)

    def _fire_pending_seek_thumb(self):
        pending = getattr(self, '_seek_thumb_pending', None)
        if pending is None:
            return
        self._seek_thumb_pending = None
        try:
            self._generator.regenerate_thumbnail(*pending)
        except Exception:
            _logger.exception("seek thumbnail regen failed for %s", pending[0])

    def _on_rating_changed(self, path: str, rating: int):
        # CRITICAL: use set_rating_ASYNC, not the blocking variant.
        # Synchronous set_rating can block the main thread for multiple
        # seconds under SQLite contention from thumbnail-generator
        # workers (reproduced: max 6s, mean 940ms with 8 background
        # writers on HDD).  That's the click-stars-app-freezes hang.
        # The async path queues the write to a single dedicated thread.
        try:
            idx = self._path_to_idx.get(path)
            if idx is not None:
                self._items[idx].rating = int(rating)
            self._cache.set_rating_async(path, int(rating))
        except Exception:
            import logging
            logging.getLogger(__name__).exception(
                "rating change handler failed for %s rating=%r", path, rating
            )

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
        filter_active = bool(self._filter_text) or self._rating_filter is not None
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

    def shutdown_all_widgets(self):
        """Block until every active hover-playback thread has stopped.
        Call from closeEvent before destroying the window — prevents
        'QThread: Destroyed while thread is still running'.

        Two-phase to avoid sequential 2s timeouts:
          1) Signal stop() on every widget's play_thread (non-blocking)
          2) Then wait briefly on each (since stop signals are already in
             flight, almost all threads have already exited by the time
             we reach their wait() call)
        """
        # Phase 1: signal-stop everyone in parallel
        for widget in self._active.values():
            try:
                if hasattr(widget, '_play_thread') and widget._play_thread is not None:
                    # Just signal; do NOT wait. stop() is fast.
                    widget._play_thread.stop()
            except Exception:
                _logger.exception("shutdown signal failed for a widget")
        # Phase 2: drain — each wait_for_shutdown should now return almost
        # immediately because the thread already started shutting down.
        # Use 500ms per widget so even pathological cases (10+ live widgets,
        # one taking the full timeout) cap at ≤5s total.
        for widget in self._active.values():
            if hasattr(widget, 'wait_for_shutdown'):
                widget.wait_for_shutdown(500)
            elif hasattr(widget, 'cleanup'):
                widget.cleanup()
        # Stop the folder-scan thread cleanly. Safe to call multiple times.
        try:
            if self._scan_worker is not None:
                self._scan_worker.cancel()
            t = self._scan_thread
            if t is not None and t.isRunning():
                t.quit()
                if not t.wait(2000):
                    _logger.warning("Folder scan thread did not exit in 2s")
        except RuntimeError:
            # Underlying C++ object already deleted — nothing to do.
            pass
        except Exception:
            _logger.exception("Error during _scan_thread shutdown")

    def pause_all_playback(self):
        """Stop and WAIT for every hover-playback thread to release its
        cv2.VideoCapture handle. Visual state is preserved. Call before
        move/copy/delete so Windows doesn't refuse with a sharing violation."""
        for widget in self._active.values():
            if hasattr(widget, 'pause_playback_and_wait'):
                widget.pause_playback_and_wait(1500)

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
        self._pixmap_cache.clear()   # release all cached QPixmaps immediately
        self._pending_items = []
        self._batch_idx     = 0
        self._total_count   = 0
        self._focused_idx   = -1
        self._empty_label.hide()
        vp_w = self.viewport().width()
        vp_h = self.viewport().height()
        self._container.resize(vp_w, vp_h)
