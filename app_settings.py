import os
from PyQt6.QtCore import QSettings


class AppSettings:
    DEFAULT_SEEK_TIME = 5.0
    DEFAULT_THUMBNAILS_PER_ROW = 3
    THUMBNAIL_CACHE_HEIGHT = 1080
    VIDEO_EXTENSIONS = {
        '.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm',
        '.m4v', '.ts', '.mts', '.m2ts', '.3gp', '.divx', '.xvid',
        '.rmvb', '.rm', '.vob', '.mpg', '.mpeg', '.mxf', '.f4v',
        '.asf', '.ogv', '.dv', '.qt', '.yuv', '.amv', '.nsv',
    }

    def __init__(self):
        # The QSettings storage location is taken from env vars so tests
        # can redirect to an isolated namespace WITHOUT polluting the
        # user's real Windows registry / macOS plist / Linux config.
        # In production these are unset → defaults to 'VideoOrganizer'.
        org = os.environ.get('VIDEO_ORGANIZER_QSETTINGS_ORG', 'VideoOrganizer')
        app = os.environ.get('VIDEO_ORGANIZER_QSETTINGS_APP', 'VideoOrganizer')
        self._settings = QSettings(org, app)
        app_data = os.environ.get('APPDATA', os.path.expanduser('~'))
        self._app_dir = os.path.join(app_data, 'VideoOrganizer')
        self._cache_dir = os.path.join(self._app_dir, 'thumbnails')
        os.makedirs(self._cache_dir, exist_ok=True)
        os.makedirs(self._app_dir, exist_ok=True)

    def sync(self):
        """Flush pending settings writes to disk. Call before a hard exit
        (os._exit) so geometry/splitter/last-folder etc. actually persist —
        os._exit skips QSettings' normal destructor-time sync."""
        try:
            self._settings.sync()
        except Exception:
            pass

    @property
    def seek_time(self) -> float:
        return float(self._settings.value('seek_time', self.DEFAULT_SEEK_TIME))

    @seek_time.setter
    def seek_time(self, value: float):
        self._settings.setValue('seek_time', float(value))

    @property
    def thumbnails_per_row(self) -> int:
        return int(self._settings.value('thumbnails_per_row', self.DEFAULT_THUMBNAILS_PER_ROW))

    @thumbnails_per_row.setter
    def thumbnails_per_row(self, value: int):
        self._settings.setValue('thumbnails_per_row', int(value))

    @property
    def cache_dir(self) -> str:
        return self._cache_dir

    @property
    def db_path(self) -> str:
        return os.path.join(self._app_dir, 'cache.db')

    @property
    def custom_search_urls(self) -> list:
        """Return the list of custom search URL bases (one per line in settings)."""
        raw = str(self._settings.value('custom_search_urls', ''))
        # Backward-compat: migrate old single-URL setting on first read
        if not raw:
            old = str(self._settings.value('custom_search_url', ''))
            if old.strip():
                raw = old.strip()
        return [u.strip() for u in raw.splitlines() if u.strip()]

    @custom_search_urls.setter
    def custom_search_urls(self, urls: list):
        self._settings.setValue('custom_search_urls', '\n'.join(u.strip() for u in urls if u.strip()))

    # ── sort preferences ─────────────────────────────────────────────────────
    _VALID_SORT_KEYS = ('name', 'modified', 'size', 'type', 'rating')

    @property
    def sort_key(self) -> str:
        v = str(self._settings.value('sort_key', 'name'))
        return v if v in self._VALID_SORT_KEYS else 'name'

    @sort_key.setter
    def sort_key(self, value: str):
        self._settings.setValue('sort_key', value)

    @property
    def sort_asc(self) -> bool:
        return self._settings.value('sort_asc', True, type=bool)

    @sort_asc.setter
    def sort_asc(self, value: bool):
        self._settings.setValue('sort_asc', bool(value))

    @property
    def last_folder(self) -> str:
        return str(self._settings.value('last_folder', os.path.expanduser('~')))

    @last_folder.setter
    def last_folder(self, value: str):
        self._settings.setValue('last_folder', value)

    @property
    def window_geometry(self):
        return self._settings.value('window_geometry')

    @window_geometry.setter
    def window_geometry(self, value):
        self._settings.setValue('window_geometry', value)

    @property
    def splitter_state(self):
        return self._settings.value('splitter_state')

    @splitter_state.setter
    def splitter_state(self, value):
        self._settings.setValue('splitter_state', value)

    # ── folder bookmarks ──────────────────────────────────────────────────────
    @property
    def bookmarks(self) -> list:
        raw = str(self._settings.value('bookmarks', ''))
        return [b.strip() for b in raw.splitlines()
                if b.strip() and os.path.isdir(b.strip())]

    @bookmarks.setter
    def bookmarks(self, value: list):
        self._settings.setValue('bookmarks', '\n'.join(str(v) for v in value))

    # ── network / UNC paths ───────────────────────────────────────────────────
    @property
    def network_paths(self) -> list:
        raw = str(self._settings.value('network_paths', ''))
        return [p.strip() for p in raw.splitlines() if p.strip()]

    @network_paths.setter
    def network_paths(self, value: list):
        self._settings.setValue('network_paths', '\n'.join(value))

    # ── theme ────────────────────────────────────────────────────────────────
    _VALID_THEMES = ('dark', 'light', 'system')

    @property
    def theme(self) -> str:
        v = str(self._settings.value('theme', 'dark'))
        return v if v in self._VALID_THEMES else 'dark'

    @theme.setter
    def theme(self, value: str):
        self._settings.setValue('theme', value)

    # ── Large-file / VR performance tuning ───────────────────────────────────
    # Files above this size will NOT trigger hover preview — opening cv2 on a
    # multi-GB file on HDD takes 1–2 s, which makes the UI feel laggy when the
    # mouse just brushes past. The static thumbnail still displays; the user
    # can double-click to play in the panel instead.
    @property
    def hover_preview_max_gb(self) -> float:
        return float(self._settings.value('hover_preview_max_gb', 4.0))

    @hover_preview_max_gb.setter
    def hover_preview_max_gb(self, value: float):
        self._settings.setValue('hover_preview_max_gb', float(value))

    # ── Deprecated knobs (kept for QSettings back-compat) ────────────────────
    # Hover preview now ALWAYS runs at the video's native FPS — these two
    # settings are no-ops in the current code path but are preserved so a
    # user's existing QSettings file with these values won't break.
    # Future re-introduction of throttling can read them again.
    @property
    def hover_preview_fps_large(self) -> int:
        return int(self._settings.value('hover_preview_fps_large', 8))

    @hover_preview_fps_large.setter
    def hover_preview_fps_large(self, value: int):
        self._settings.setValue('hover_preview_fps_large', max(1, int(value)))

    @property
    def large_file_threshold_mb(self) -> int:
        return int(self._settings.value('large_file_threshold_mb', 500))

    @large_file_threshold_mb.setter
    def large_file_threshold_mb(self, value: int):
        self._settings.setValue('large_file_threshold_mb', max(0, int(value)))

    # Enable FFmpeg hardware acceleration when available (NVDEC / D3D11 / VAAPI).
    # Default is False because the standard pip opencv-python wheel on Windows
    # does NOT actually ship working HW decode — it prints noisy stderr errors
    # and SW fallback can be unreliable. video_capture_helper auto-detects the
    # prebuilt wheel and disables HW preemptively even if this setting is True,
    # so flipping this on only matters for custom OpenCV builds with real
    # D3D11VA / NVDEC / VAAPI support compiled in.
    @property
    def use_hw_accel(self) -> bool:
        return self._settings.value('use_hw_accel', False, type=bool)

    @use_hw_accel.setter
    def use_hw_accel(self, value: bool):
        self._settings.setValue('use_hw_accel', bool(value))

    # Number of CPU threads for software decode. 0 = auto (use all cores).
    # Applied via the OPENCV_FFMPEG_CAPTURE_OPTIONS env var in main.py.
    @property
    def cpu_decode_threads(self) -> int:
        return int(self._settings.value('cpu_decode_threads', 0))

    @cpu_decode_threads.setter
    def cpu_decode_threads(self, value: int):
        self._settings.setValue('cpu_decode_threads', max(0, int(value)))

    # ── recursive folder traversal ───────────────────────────────────────────
    @property
    def recursive_view(self) -> bool:
        """When True, the grid shows every video inside the current folder
        AND all subfolders, recursively (subfolders themselves are hidden)."""
        return self._settings.value('recursive_view', False, type=bool)

    @recursive_view.setter
    def recursive_view(self, value: bool):
        self._settings.setValue('recursive_view', bool(value))

    # ── view mode ────────────────────────────────────────────────────────────
    @property
    def view_mode(self) -> str:
        """'grid' or 'list'"""
        v = str(self._settings.value('view_mode', 'grid'))
        return v if v in ('grid', 'list') else 'grid'

    @view_mode.setter
    def view_mode(self, value: str):
        self._settings.setValue('view_mode', value)

    # ── columns slider ────────────────────────────────────────────────────────
    @property
    def thumbnail_scale(self) -> int:
        """Column count, 1-10. Used by the thumbnail size slider."""
        return int(self._settings.value('thumbnail_scale',
                                         self.DEFAULT_THUMBNAILS_PER_ROW))

    @thumbnail_scale.setter
    def thumbnail_scale(self, value: int):
        self._settings.setValue('thumbnail_scale', int(value))

    # ── player geometry ────────────────────────────────────────────────────────
    @property
    def player_geometry(self):
        return self._settings.value('player_geometry')

    @player_geometry.setter
    def player_geometry(self, value):
        self._settings.setValue('player_geometry', value)
