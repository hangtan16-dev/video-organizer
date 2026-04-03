"""
Background folder size scanner.

Scans selected folders recursively to compute total disk usage,
then emits results one folder at a time so the UI can update
incrementally without freezing.
"""

import os
from PyQt6.QtCore import QThread, pyqtSignal


class FolderSizeScanner(QThread):
    """
    Recursively computes the disk usage of a list of folder paths.

    Emits folder_done(path, size_bytes) for each folder as it completes.
    Emits all_done() when finished.
    Stoppable via stop().

    Usage
    -----
    scanner = FolderSizeScanner(['/path/a', '/path/b'])
    scanner.folder_done.connect(on_folder_done)
    scanner.all_done.connect(on_all_done)
    scanner.start()
    # later: scanner.stop()
    """

    folder_done = pyqtSignal(str, int)   # (folder_path, total_bytes)
    all_done    = pyqtSignal()

    def __init__(self, folder_paths: list, parent=None):
        super().__init__(parent)
        self._paths = list(folder_paths)
        self._stop  = False

    def stop(self):
        """Request the scanner to stop after the current folder completes."""
        self._stop = True

    def run(self):
        for path in self._paths:
            if self._stop:
                break
            size = self._get_size(path)
            self.folder_done.emit(path, size)
        self.all_done.emit()

    def _get_size(self, path: str) -> int:
        """Recursively compute total size of all files under path."""
        total = 0
        try:
            for entry in os.scandir(path):
                if self._stop:
                    break
                if entry.is_file(follow_symlinks=False):
                    try:
                        total += entry.stat().st_size
                    except OSError:
                        pass
                elif entry.is_dir(follow_symlinks=False):
                    total += self._get_size(entry.path)
        except (PermissionError, OSError):
            pass
        return total

    # ── Convenience helpers ───────────────────────────────────────────────

    @staticmethod
    def format_size(n: int) -> str:
        """Return a human-readable size string (e.g. '1.4 GB')."""
        if n < 0:
            return '—'
        for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
            if n < 1024:
                return f"{n:.1f} {unit}" if unit != 'B' else f"{n} B"
            n /= 1024
        return f"{n:.1f} PB"
