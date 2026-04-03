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
        self._settings = QSettings('VideoOrganizer', 'VideoOrganizer')
        app_data = os.environ.get('APPDATA', os.path.expanduser('~'))
        self._app_dir = os.path.join(app_data, 'VideoOrganizer')
        self._cache_dir = os.path.join(self._app_dir, 'thumbnails')
        os.makedirs(self._cache_dir, exist_ok=True)
        os.makedirs(self._app_dir, exist_ok=True)

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
