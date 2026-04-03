"""
Duplicate video finder dialog.

Algorithm:
  1. Recursively scan folder for video files.
  2. Group by file size (exact duplicates cannot have different sizes).
  3. For groups with 2+ files, extract one frame at 50% duration, resize
     to 16×16 grayscale, and compare pairwise L2 distance.
  4. Pairs with distance < threshold are presented as duplicate sets.
"""

import os

import cv2
import numpy as np

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QProgressBar, QTreeWidget, QTreeWidgetItem, QMessageBox,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from app_settings import AppSettings
from cache_manager import CacheManager

try:
    import send2trash as _send2trash
    _HAS_SEND2TRASH = True
except ImportError:
    _HAS_SEND2TRASH = False

_HASH_SIZE  = 16       # resize to 16×16 for perceptual hash
_THRESHOLD  = 500.0    # L2 distance threshold (tunable)


def _fmt_size(n: int) -> str:
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if n < 1024:
            return f"{n:.1f} {unit}" if unit != 'B' else f"{n} B"
        n /= 1024
    return f"{n:.1f} PB"


def _extract_fingerprint(path: str) -> np.ndarray | None:
    """Extract a 256-element float32 fingerprint from the middle frame."""
    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            cap.release()
            return None
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total > 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (_HASH_SIZE, _HASH_SIZE),
                           interpolation=cv2.INTER_LINEAR)
        return small.flatten().astype(np.float32)
    except Exception:
        return None


class _ScanWorker(QThread):
    progress = pyqtSignal(str, int, int)              # current_file, done, total
    found_duplicates = pyqtSignal(list)               # list of lists of paths
    finished_scan    = pyqtSignal(int, int)           # total_files, duplicate_sets

    def __init__(self, folder_path: str):
        super().__init__()
        self._folder = folder_path
        self._stop   = False

    def stop(self):
        self._stop = True

    def run(self):
        exts = AppSettings.VIDEO_EXTENSIONS
        all_files: list[str] = []

        # Recursive scan
        for root, dirs, files in os.walk(self._folder):
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            for f in files:
                if os.path.splitext(f)[1].lower() in exts:
                    all_files.append(os.path.join(root, f))

        total = len(all_files)

        # Group by size
        size_groups: dict[int, list[str]] = {}
        for path in all_files:
            try:
                sz = os.path.getsize(path)
            except OSError:
                continue
            size_groups.setdefault(sz, []).append(path)

        # Candidates: groups with 2+ files of same size
        candidates = [grp for grp in size_groups.values() if len(grp) >= 2]

        done = 0
        duplicate_sets: list[list[str]] = []

        for group in candidates:
            if self._stop:
                break
            fingerprints: dict[str, np.ndarray] = {}
            for path in group:
                if self._stop:
                    break
                done += 1
                self.progress.emit(os.path.basename(path), done, total)
                fp = _extract_fingerprint(path)
                if fp is not None:
                    fingerprints[path] = fp

            # Pairwise comparison within size group
            paths_with_fp = list(fingerprints.keys())
            visited: set[str] = set()
            for i, p1 in enumerate(paths_with_fp):
                if p1 in visited:
                    continue
                cluster = [p1]
                for p2 in paths_with_fp[i + 1:]:
                    if p2 in visited:
                        continue
                    dist = float(np.linalg.norm(fingerprints[p1] - fingerprints[p2]))
                    if dist < _THRESHOLD:
                        cluster.append(p2)
                        visited.add(p2)
                if len(cluster) >= 2:
                    visited.add(p1)
                    duplicate_sets.append(cluster)

        self.found_duplicates.emit(duplicate_sets)
        self.finished_scan.emit(total, len(duplicate_sets))


class DuplicateFinderDialog(QDialog):
    def __init__(self, folder_path: str, cache: CacheManager, parent=None):
        super().__init__(parent)
        self._folder = folder_path
        self._cache  = cache
        self._worker: _ScanWorker | None = None

        self.setWindowTitle("Find Duplicate Videos")
        self.setMinimumSize(700, 500)
        self.setStyleSheet("""
            QDialog { background: #1e1e1e; color: #ddd; }
            QLabel  { color: #ddd; }
            QProgressBar { background: #2e2e2e; border: 1px solid #3a3a3a;
                           border-radius: 3px; color: #ddd; text-align: center; }
            QProgressBar::chunk { background: #3a6fc4; border-radius: 2px; }
            QTreeWidget { background: #2e2e2e; color: #ccc; border: 1px solid #3a3a3a;
                          gridline-color: #3a3a3a; }
            QTreeWidget::item { padding: 2px; }
            QTreeWidget::item:hover { background: #3a3a3a; }
            QTreeWidget::item:selected { background: #3a6fc4; }
            QHeaderView::section { background: #252525; color: #aaa;
                                   border: none; padding: 4px; }
            QPushButton { background: #3a3a3a; color: #ddd; border: 1px solid #555;
                          border-radius: 3px; padding: 4px 12px; min-width: 70px; }
            QPushButton:hover { background: #484848; }
            QPushButton:disabled { color: #666; background: #2a2a2a; }
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        self._status_label = QLabel(f"Scanning: {folder_path}", self)
        self._status_label.setStyleSheet("color: #aaa; font-size: 11px;")
        layout.addWidget(self._status_label)

        self._progress = QProgressBar(self)
        self._progress.setRange(0, 0)   # indeterminate initially
        layout.addWidget(self._progress)

        self._tree = QTreeWidget(self)
        self._tree.setColumnCount(3)
        self._tree.setHeaderLabels(["File", "Size", "Path"])
        self._tree.header().setStretchLastSection(True)
        self._tree.setColumnWidth(0, 220)
        self._tree.setColumnWidth(1, 80)
        layout.addWidget(self._tree, 1)

        btn_row = QHBoxLayout()
        self._btn_delete = QPushButton("Delete Checked")
        self._btn_delete.setEnabled(False)
        self._btn_delete.setStyleSheet(
            "QPushButton { background: #8b2020; color: #fff; border: none; }"
            "QPushButton:hover { background: #a03030; }"
            "QPushButton:disabled { background: #2a2a2a; color: #666; border: 1px solid #555; }"
        )
        self._btn_delete.clicked.connect(self._on_delete_checked)
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self._on_close)
        btn_row.addWidget(self._btn_delete)
        btn_row.addStretch()
        btn_row.addWidget(btn_close)
        layout.addLayout(btn_row)

        # Start scan
        self._worker = _ScanWorker(folder_path)
        self._worker.progress.connect(self._on_progress)
        self._worker.found_duplicates.connect(self._on_duplicates)
        self._worker.finished_scan.connect(self._on_scan_done)
        self._worker.start()

    def _on_progress(self, current_file: str, done: int, total: int):
        self._progress.setRange(0, max(1, total))
        self._progress.setValue(done)
        self._status_label.setText(f"Scanning ({done}/{total}): {current_file}")

    def _on_duplicates(self, duplicate_sets: list):
        self._tree.clear()
        for set_idx, group in enumerate(duplicate_sets, start=1):
            group_item = QTreeWidgetItem(self._tree,
                                         [f"Duplicate Set {set_idx}", "", ""])
            group_item.setExpanded(True)
            group_item.setFlags(group_item.flags() & ~Qt.ItemFlag.ItemIsUserCheckable)

            for i, path in enumerate(group):
                try:
                    sz = _fmt_size(os.path.getsize(path))
                except OSError:
                    sz = "?"
                child = QTreeWidgetItem(group_item, [
                    os.path.basename(path), sz, os.path.dirname(path)
                ])
                child.setData(0, Qt.ItemDataRole.UserRole, path)
                child.setFlags(child.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                # Pre-check all but the first (keep first)
                child.setCheckState(0, Qt.CheckState.Checked if i > 0
                                       else Qt.CheckState.Unchecked)

        if duplicate_sets:
            self._btn_delete.setEnabled(True)

    def _on_scan_done(self, total_files: int, n_sets: int):
        self._progress.setRange(0, 1)
        self._progress.setValue(1)
        self._status_label.setText(
            f"Scan complete: {total_files} files checked, "
            f"{n_sets} duplicate set{'s' if n_sets != 1 else ''} found."
        )

    def _on_delete_checked(self):
        to_delete: list[str] = []
        for i in range(self._tree.topLevelItemCount()):
            group_item = self._tree.topLevelItem(i)
            for j in range(group_item.childCount()):
                child = group_item.child(j)
                if child.checkState(0) == Qt.CheckState.Checked:
                    path = child.data(0, Qt.ItemDataRole.UserRole)
                    if path:
                        to_delete.append(path)

        if not to_delete:
            QMessageBox.information(self, "Nothing checked",
                                    "No items are checked for deletion.")
            return

        reply = QMessageBox.question(
            self, "Confirm delete",
            f"Delete {len(to_delete)} checked item(s)?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        errors: list[str] = []
        deleted: list[str] = []
        for path in to_delete:
            try:
                if _HAS_SEND2TRASH:
                    _send2trash.send2trash(path)
                else:
                    os.remove(path)
                self._cache.invalidate(path)
                deleted.append(path)
            except Exception as exc:
                errors.append(f"{os.path.basename(path)}: {exc}")

        if errors:
            QMessageBox.warning(self, "Delete errors",
                                "Some files could not be deleted:\n" + "\n".join(errors))

        # Remove deleted items from tree
        deleted_set = set(deleted)
        for i in range(self._tree.topLevelItemCount()):
            group_item = self._tree.topLevelItem(i)
            for j in range(group_item.childCount() - 1, -1, -1):
                child = group_item.child(j)
                path = child.data(0, Qt.ItemDataRole.UserRole)
                if path in deleted_set:
                    group_item.removeChild(child)

        if not errors and deleted:
            self._status_label.setText(
                f"Deleted {len(deleted)} file(s). "
                f"Remaining tree shows still-present duplicates."
            )

    def _on_close(self):
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait(2000)
        self.reject()

    def closeEvent(self, event):
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait(2000)
        super().closeEvent(event)
