"""
Background thumbnail generation service.

KEY DESIGN — why the old version froze the UI
─────────────────────────────────────────────
The old code called get_thumbnail_path() (SQLite + disk stat) on the main
thread for every video, then immediately emitted the signal, which caused
_on_thumbnail_ready to run synchronously and load a QPixmap (JPEG decode)
for every cached file — all on the main thread.  For 100+ cached files that
blocks the event loop for multiple seconds.

NEW DESIGN
──────────
• request_thumbnail() does NO disk/SQLite work — it just queues a worker.
• Every worker runs entirely on a thread-pool thread:
    1. Check SQLite cache  (background thread)
    2. If cached → load JPEG with cv2, convert BGR→RGB→QImage  (background)
    3. If not cached → open video, seek, extract frame, scale, cache  (background)
    4. Scale frame down to ≤ _EMIT_MAX_WIDTH wide before converting to QImage
    5. Emit (video_path, seek_time, QImage, duration_secs) signal

• The on-disk cache always stores full THUMBNAIL_HEIGHT (1080 px) quality.
• The emitted QImage is small (≤ 800 px wide) to minimise cross-thread copy
  overhead and the per-video memory held in _Item.pixmap for virtual scroll.
• The main thread only receives a ready-made QImage and calls
  QPixmap.fromImage(qimage) which is essentially free.
"""

import cv2
import numpy as np

from PyQt6.QtCore import QRunnable, QThreadPool, QObject, pyqtSignal, QMutex, QMutexLocker
from PyQt6.QtGui import QImage

from cache_manager import CacheManager

THUMBNAIL_HEIGHT = 1080
# Maximum width of the QImage emitted across the thread boundary.
# The full-resolution frame is still written to the on-disk JPEG cache.
_EMIT_MAX_WIDTH  = 800


# ──────────────────────────────────────────────────────────────────────────────
# Signals object — created on the main thread, used from worker threads.
# Using `object` as the QImage slot type so Qt does not try to copy it
# through the meta-type system; Python's ref-count keeps it alive in the queue.
# ──────────────────────────────────────────────────────────────────────────────
class _WorkerSignals(QObject):
    # video_path, seek_time, QImage (as object), duration_secs
    thumbnail_ready = pyqtSignal(str, float, object, float)
    error = pyqtSignal(str, str)


def _frame_to_qimage(frame_bgr: np.ndarray) -> QImage:
    """Convert a BGR numpy frame to an RGB QImage that owns its data."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = frame_rgb.shape
    # tobytes() makes a Python bytes copy; QImage wraps it without another copy.
    # We keep `raw` alive by storing it inside the QImage via PyQt6's mechanism.
    raw = frame_rgb.tobytes()
    img = QImage(raw, w, h, ch * w, QImage.Format.Format_RGB888)
    # .copy() makes QImage own the pixel buffer independently of `raw`
    return img.copy()


def _scale_for_emit(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Scale frame down so its width is at most _EMIT_MAX_WIDTH.

    This reduces the size of the QImage passed across the thread boundary
    and the per-video pixmap stored in the virtual-scroll _Item records.
    The full-resolution frame is written to the disk cache before this call.
    """
    h, w = frame_bgr.shape[:2]
    if w <= _EMIT_MAX_WIDTH:
        return frame_bgr
    scale = _EMIT_MAX_WIDTH / w
    return cv2.resize(
        frame_bgr,
        (int(w * scale), int(h * scale)),
        interpolation=cv2.INTER_AREA,
    )


class ThumbnailWorker(QRunnable):
    def __init__(self, video_path: str, seek_time: float, cache_manager: CacheManager):
        super().__init__()
        self.setAutoDelete(True)
        self.video_path = video_path
        self.seek_time = seek_time
        self.cache_manager = cache_manager
        self.signals = _WorkerSignals()   # lives on main thread; queued connections auto-applied
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        if self._cancelled:
            return
        try:
            self._do_work()
        except Exception as e:
            self.signals.error.emit(self.video_path, str(e))

    def _do_work(self):
        # ── 1. Check cache (background thread — no main-thread stall) ──────
        cached_path = self.cache_manager.get_thumbnail_path(self.video_path, self.seek_time)
        duration = self.cache_manager.get_duration(self.video_path)

        if cached_path:
            # Load the cached JPEG in the background and convert to QImage
            frame = cv2.imread(cached_path)
            if frame is not None:
                if duration <= 0:
                    duration = self._read_duration(self.video_path)
                # Scale down before cross-thread emit (cache still stores 1080p)
                qimg = _frame_to_qimage(_scale_for_emit(frame))
                self.signals.thumbnail_ready.emit(
                    self.video_path, self.seek_time, qimg, duration
                )
                return
            # Cached file was missing — fall through and regenerate

        if self._cancelled:
            return

        # ── 2. Open the video ───────────────────────────────────────────────
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.signals.error.emit(self.video_path, "Cannot open video")
            return

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if fps > 0 and fps < 1000:
                duration = total_frames / fps
            else:
                fps = 25.0
                duration = 0.0

            if total_frames > 0 and fps > 0:
                target = int(self.seek_time * fps)
                target = max(0, min(target, total_frames - 1))
                cap.set(cv2.CAP_PROP_POS_FRAMES, target)

            ret, frame = cap.read()
        finally:
            cap.release()

        if not ret or frame is None:
            self.signals.error.emit(self.video_path, "Could not read frame")
            return

        if self._cancelled:
            return

        # ── 3. Scale to 1080p height ────────────────────────────────────────
        orig_h, orig_w = frame.shape[:2]
        if orig_h > 0 and orig_h != THUMBNAIL_HEIGHT:
            scale = THUMBNAIL_HEIGHT / orig_h
            frame = cv2.resize(
                frame,
                (int(orig_w * scale), THUMBNAIL_HEIGHT),
                interpolation=cv2.INTER_LANCZOS4,
            )

        if self._cancelled:
            return

        # ── 4. Store full-resolution frame to cache (1080p JPEG) ─────────────
        self.cache_manager.store_thumbnail(
            self.video_path, self.seek_time, frame, duration
        )

        # ── 5. Scale down for emit, convert to QImage, and emit ──────────────
        # The cache stores 1080p; we only send ≤800px wide across the thread
        # boundary so cross-thread copy and _Item.pixmap memory stay small.
        qimg = _frame_to_qimage(_scale_for_emit(frame))
        self.signals.thumbnail_ready.emit(self.video_path, self.seek_time, qimg, duration)

    @staticmethod
    def _read_duration(video_path: str) -> float:
        """Quick metadata-only read to get video duration."""
        cap = None
        try:
            cap = cv2.VideoCapture(video_path)
            fps    = cap.get(cv2.CAP_PROP_FPS)
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            if fps > 0 and fps < 1000 and frames > 0:
                return frames / fps
        except Exception:
            pass
        finally:
            if cap is not None:
                cap.release()
        return 0.0


# ──────────────────────────────────────────────────────────────────────────────
class ThumbnailGeneratorService(QObject):
    """
    Manages a thread pool of ThumbnailWorkers.
    Emits thumbnail_ready(video_path, seek_time, QImage, duration) on the
    main thread when a thumbnail is ready.
    """
    # video_path, seek_time, QImage (object), duration_secs
    thumbnail_ready = pyqtSignal(str, float, object, float)

    def __init__(self, cache_manager: CacheManager, parent=None):
        super().__init__(parent)
        self._cache = cache_manager
        self._pool = QThreadPool.globalInstance()
        self._pool.setMaxThreadCount(max(4, self._pool.maxThreadCount()))
        self._mutex = QMutex()
        self._pending: dict[str, ThumbnailWorker] = {}

    def request_thumbnail(self, video_path: str, seek_time: float):
        """Queue thumbnail generation. Does NO work on the calling thread."""
        with QMutexLocker(self._mutex):
            if video_path in self._pending:
                return
            worker = ThumbnailWorker(video_path, seek_time, self._cache)
            worker.signals.thumbnail_ready.connect(self._on_ready)
            worker.signals.error.connect(self._on_error)
            self._pending[video_path] = worker
        self._pool.start(worker)

    def regenerate_thumbnail(self, video_path: str, seek_time: float):
        """Cancel any in-flight worker, invalidate cache, regenerate."""
        # Cache ops are independently thread-safe (each call uses its own connection);
        # do them before acquiring the mutex so the critical section stays short.
        self._cache.invalidate(video_path)
        self._cache.set_seek_override(video_path, seek_time)

        with QMutexLocker(self._mutex):
            old = self._pending.pop(video_path, None)
            if old:
                old.cancel()
            worker = ThumbnailWorker(video_path, seek_time, self._cache)
            worker.signals.thumbnail_ready.connect(self._on_ready)
            worker.signals.error.connect(self._on_error)
            self._pending[video_path] = worker
        self._pool.start(worker)

    def cancel_all(self):
        with QMutexLocker(self._mutex):
            for w in self._pending.values():
                w.cancel()
            self._pending.clear()

    # ── slots (called on main thread via queued connection) ─────────────────
    def _on_ready(self, video_path: str, seek_time: float,
                  qimage: object, duration: float):
        with QMutexLocker(self._mutex):
            self._pending.pop(video_path, None)
        self.thumbnail_ready.emit(video_path, seek_time, qimage, duration)

    def _on_error(self, video_path: str, msg: str):
        print(f"[ThumbnailGen] {video_path}: {msg}")
        with QMutexLocker(self._mutex):
            self._pending.pop(video_path, None)
