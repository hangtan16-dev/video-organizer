"""
Bulk metadata fetch worker.

Iterates over a list of video paths, attempts online metadata lookup for each
using the same backend chain as MetadataDialog (TMDB → OMDb → Wikipedia → DDG),
and emits progress signals so the caller can update a QProgressDialog.
"""

from PyQt6.QtCore import QThread, pyqtSignal

from metadata_dialog import _filename_to_queries, _try_all_backends


class BulkMetaWorker(QThread):
    progress  = pyqtSignal(str, int, int)   # current_file, done, total
    item_done = pyqtSignal(str, dict)        # path, metadata dict (may be empty)
    all_done  = pyqtSignal(int, int)         # successes, failures

    def __init__(self, paths: list, parent=None):
        super().__init__(parent)
        self._paths = list(paths)
        self._stop  = False

    def stop(self):
        self._stop = True

    def run(self):
        total     = len(self._paths)
        successes = 0
        failures  = 0

        for i, path in enumerate(self._paths):
            if self._stop:
                break

            self.progress.emit(path, i, total)

            result: dict = {}
            try:
                candidates = _filename_to_queries(path)
                for candidate in candidates:
                    if self._stop:
                        break
                    result = _try_all_backends(candidate)
                    if result.get('Title') or result.get('Summary'):
                        break
            except Exception:
                result = {}

            self.item_done.emit(path, result)

            if result.get('Title') or result.get('Summary'):
                successes += 1
            else:
                failures += 1

        self.all_done.emit(successes, failures)
