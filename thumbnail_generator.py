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

import os
import time as _time
import cv2
import numpy as np

# Belt-and-suspenders: silence cv2's own FFmpeg log channel for builds that
# ignore the OPENCV_FFMPEG_LOGLEVEL env var. main.py already sets the env
# var before this module loads, but on some Windows OpenCV builds only the
# Python API call works.
try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
except (AttributeError, Exception):
    pass

# Prefer PyAV for thumbnail extraction — benchmarked 8× faster than cv2 on
# 8K HEVC files (461 ms vs 3727 ms).  Falls back to cv2 transparently.
try:
    import av
    _HAS_PYAV = True
    try:
        # Deadlock fix (see pyav_play_thread.py for the full analysis):
        # restore libav's C log callback so decoder threads never call
        # PyGILState_Ensure from av_log — which can deadlock against the GIL
        # held by a thread blocked inside avcodec. Keep the C output quiet.
        av.logging.set_level(av.logging.FATAL)
        av.logging.restore_default_callback()
        av.logging.set_libav_level(av.logging.FATAL)
    except Exception:
        pass
except ImportError:
    _HAS_PYAV = False

from PyQt6.QtCore import QRunnable, QThreadPool, QObject, pyqtSignal, QMutex, QMutexLocker, QTimer
from PyQt6.QtGui import QImage

from cache_manager import CacheManager
from app_logger import get_logger
log = get_logger(__name__)

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
    # Emitted from run()'s finally for EVERY worker exit — success, error, OR
    # cancellation — carrying the worker itself. The service uses it to ALWAYS
    # clear _pending (so a cancelled worker, which emits neither ready nor
    # error, can't leave the path stuck pending → "Generating…" forever) and to
    # re-queue a preview-preempted worker. Carries the worker (object) so the
    # service can identity-check it against the current _pending entry.
    finished = pyqtSignal(object)


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
        self.signals = _WorkerSignals()
        self._cancelled = False
        # PyAV `av.open(...)` is given an explicit Python file object instead
        # of a path. That way `force_close` from another thread can close the
        # file — libav sees an I/O error on its next read and the worker-side
        # decode loop returns cleanly. We must NOT touch self._container from
        # another thread: closing a PyAV container while another thread is
        # inside container.decode(stream) is a use-after-free → Windows
        # access violation (observed in the user's "folder switch mid-scan"
        # workflow).
        self._container = None
        self._file_obj = None
        self._started_at: float = 0.0   # monotonic time when run() started
        # Set True if PyAV's bounded av.open() gave up (disk stuck/contended,
        # not a codec problem). _do_work then SKIPS the unbounded cv2 fallback
        # and emits a transient error so the grid retries later — otherwise the
        # worker grinds >30 s, blows the watchdog, and (serialized) stalls every
        # other thumbnail.
        self._open_timed_out = False
        # When this worker is cancelled, should the service re-queue the
        # thumbnail? True for a preview PREEMPTION (the thumbnail still needs
        # generating once the disk frees); False for cancel_all (delete /
        # shutdown — don't spawn fresh work). Set by force_close(requeue=...).
        self._requeue_on_cancel = True

    def cancel(self):
        """Soft cancel — flag the worker.  The decode call won't be
        interrupted but post-decode work is skipped."""
        self._cancelled = True

    def force_close(self, requeue: bool = True):
        """Hard cancel — flag the worker so its decode loop bails on the next
        frame, and close the underlying Python file object (if any) so libav
        sees EOF on its next read. The worker thread unwinds on its own; we do
        NOT touch the PyAV container from here.

        `requeue` records whether the service should regenerate this thumbnail
        after the worker exits: True for a preview preemption (the disk was
        yanked away mid-decode but the thumbnail is still wanted), False for
        cancel_all (the folder is being torn down / a file deleted).

        Why only the file object, not the container: closing the container from
        a different thread while the worker is inside container.decode(stream)
        crashes the process (access violation in libavcodec — observed during
        folder-switch mid-scan). file.close() is thread-safe.
        """
        self._cancelled = True
        self._requeue_on_cancel = requeue
        f = self._file_obj
        if f is not None:
            try:
                f.close()
            except Exception:
                pass
        # Note: self._container is NOT touched here. The worker thread's
        # `finally` block in _extract_frame_pyav closes it safely.

    def run(self):
        # NOTE: _started_at is deliberately NOT set here. The watchdog uses it
        # to force-close workers stuck in a slow DECODE (holding a file
        # handle). A worker can sit PARKED in background_section for a long
        # time while the user previews (the disk coordinator holds background
        # work back) — that is NOT a stuck decode and must NOT be timed out.
        # So _started_at is set only once we're inside background_section and
        # about to actually open/decode the file (see _do_work).
        try:
            if not self._cancelled:
                self._do_work()
        except Exception as e:
            log.error("Worker exception [%s]: %s", os.path.basename(self.video_path),
                      e, exc_info=True)
            self.signals.error.emit(self.video_path, str(e))
        finally:
            self._container = None
            # ALWAYS notify the service so it clears _pending — even when
            # _do_work returned early due to cancellation (which emits neither
            # ready nor error). Without this a preview-preempted worker leaves
            # its path stuck in _pending and request_thumbnail() dedups every
            # future attempt → the card is stuck "Generating…" forever.
            try:
                self.signals.finished.emit(self)
            except RuntimeError:
                pass   # signals object already gone (interpreter teardown)

    def _do_work(self):
        # ── 0. Skip files we already know are unreadable ───────────────────
        # Critical: this MUST come before any cv2 call so we don't reopen
        # corrupted files and trigger another flood of FFmpeg stderr noise.
        if self.cache_manager.is_thumbnail_failed(self.video_path):
            self.signals.error.emit(self.video_path, "Cached failure")
            return

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

        # ── 2-5. Open source video, extract a frame, cache + emit ──────────
        # This is the ONLY part that opens the multi-GB source file, so it
        # runs inside a background_section. The disk coordinator guarantees
        # this never overlaps a foreground preview/seek: if the user starts
        # hovering, this section finishes its current file then the worker
        # parks before starting the next. See disk_coordinator.py.
        from disk_coordinator import COORDINATOR
        label = f"thumb:{id(self)}"
        with COORDINATOR.background_section(label, self.video_path,
                                            on_yield=self.force_close):
            if self._cancelled:
                return
            # We now hold background disk access and are about to open/decode —
            # START the watchdog clock here (NOT in run()), so time spent
            # PARKED waiting for a preview to finish never counts as a decode
            # "timeout". This was the bug: parked workers were force-closed at
            # 30 s → thumbnail_failed("Timeout") → retry → re-park → repeat,
            # so cards stayed stuck "Generating…" the whole time the user
            # browsed/previewed.
            self._started_at = _time.monotonic()
            # Prefer PyAV (8x faster than cv2 on 8K HEVC). Falls back to cv2 —
            # EXCEPT when PyAV's bounded open timed out (disk stuck): cv2 has no
            # timeout and would block the same way, grinding past the 30 s
            # watchdog and stalling the serialized queue. In that case treat it
            # as transient and let the grid retry when the disk frees up.
            frame, duration = self._extract_frame_pyav()
            if frame is None and self._open_timed_out:
                if self._cancelled:
                    return
                self.signals.error.emit(self.video_path, "Could not read frame")
                return
            if frame is None:
                if self._cancelled:
                    return
                frame, duration = self._extract_frame_cv2(duration)
            if frame is None:
                if self._cancelled:
                    return
                # TRANSIENT failure: the file OPENED fine but we couldn't grab
                # a frame here — almost always disk contention or a read
                # timeout during a mass regen + scroll, NOT a corrupt file
                # (corrupt/unsupported files fail at OPEN → "Cannot open
                # video", which IS cached). Do NOT mark this a permanent
                # failure; emit a retryable error so the grid re-attempts it
                # once the disk frees up. (mark_thumbnail_failed here was the
                # bug: one contended read left the thumbnail permanently blank
                # with "Cannot read frame" and it was never regenerated.)
                self.signals.error.emit(self.video_path, "Could not read frame")
                return

            if self._cancelled:
                return

            # Scale to 1080p height
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

            # Store full-resolution frame to cache (1080p JPEG)
            _t_store = _time.monotonic()
            self.cache_manager.store_thumbnail(
                self.video_path, self.seek_time, frame, duration
            )
            _store_dt = _time.monotonic() - _t_store
            if _store_dt > 3.0:
                log.warning("Slow thumbnail STORE [%s]: %.1fs (SQLite write-lock "
                            "contention?)", os.path.basename(self.video_path), _store_dt)

        # ── 6. Emit (no disk I/O — outside the background section) ──────────
        # The cache stores 1080p; we only send ≤800px wide across the thread
        # boundary so cross-thread copy and _Item.pixmap memory stay small.
        qimg = _frame_to_qimage(_scale_for_emit(frame))
        self.signals.thumbnail_ready.emit(self.video_path, self.seek_time, qimg, duration)

    def _extract_frame_pyav(self) -> 'tuple[np.ndarray | None, float]':
        """Extract a single frame at seek_time using PyAV.  Returns
        (BGR ndarray, duration_seconds) or (None, 0.0) on failure.

        ~8× faster than cv2 on 8K HEVC.

        Thread-safety: we open the file ourselves as a Python file object
        and pass it to av.open(). force_close() (called from another
        thread) closes the file, which makes libav's next read fail and
        the decode loop exit cleanly INSIDE THIS THREAD. The PyAV
        container is never touched from another thread.
        """
        if not _HAS_PYAV:
            return None, 0.0
        container = None
        try:
            if self._cancelled:
                return None, 0.0

            # NATIVE path open (NOT a Python file object). A file-object open
            # routes every libav read through a GIL-holding Python callback
            # (pyio); on huge files that starves the GUI thread and freezes
            # the UI. av.open(path) reads in C and releases the GIL. `timeout`
            # bounds a stuck open. The caller (_do_work) holds a coordinator
            # background_section, so no foreground preview is on the disk now.
            # Spin-up/contention-tolerant open timeout (was 4 s): on an HDD a
            # cold or busy open can exceed 4 s, after which _do_work falls back
            # to the UNBOUNDED cv2 path → a worker stuck >30 s → watchdog kill →
            # retry → churn, which (with serialized background decode) stalls
            # every OTHER thumbnail too. Native av.open releases the GIL and
            # honours this bound, so a generous value is safe.
            _t_open = _time.monotonic()
            try:
                container = av.open(self.video_path, timeout=(12.0, 6.0))
            except Exception:
                _open_dt = _time.monotonic() - _t_open
                if _open_dt > 9.0:
                    # Open consumed (most of) the bounded timeout then failed →
                    # the disk is stuck/contended, NOT a codec issue. cv2 would
                    # block the same way (no timeout), so flag it: _do_work will
                    # SKIP the cv2 fallback and emit a transient error → the grid
                    # retries once the disk frees up. This is what stops a worker
                    # grinding >30 s, blowing the watchdog, and (serialized)
                    # stalling every OTHER thumbnail behind it.
                    self._open_timed_out = True
                    log.warning("Thumbnail OPEN gave up [%s] after %.1fs — "
                                "transient, will retry",
                                os.path.basename(self.video_path), _open_dt)
                raise
            _open_dt = _time.monotonic() - _t_open
            if _open_dt > 5.0:
                log.warning("Slow thumbnail OPEN [%s]: av.open took %.1fs",
                            os.path.basename(self.video_path), _open_dt)
            self._container = container
            try:
                stream = container.streams.video[0]
                ctx = stream.codec_context
                # SLICE threading (NOT FRAME). We grab a single KEYFRAME (see the
                # seek below) rather than forward-decoding the whole GOP, so SLICE
                # just parallelizes decoding that one intra-coded 8K frame across
                # cores — no lookahead pipeline needed.
                #
                # Why NOT FRAME: FRAME spawns a lookahead pipeline whose worker
                # threads container.close() does NOT join → across 86 files they
                # piled up to ~90 OS threads + 2.2 GB of in-flight 8K buffers and
                # crashed the process (native exit 127). SLICE workers are joined
                # on close. (FRAME workers also call av_log → PyGILState_Ensure, a
                # GIL deadlock — also fixed globally via restore_default_callback.)
                ctx.thread_type = 'SLICE'
                ctx.thread_count = max(4, min(16, int((os.cpu_count() or 8) * 0.5)))

                # Compute duration from stream metadata
                duration = 0.0
                if stream.duration and stream.time_base:
                    duration = float(stream.duration * stream.time_base)
                elif container.duration:
                    duration = container.duration / 1_000_000.0

                # Grab the nearest KEYFRAME at/after seek_time and use it as-is.
                # We decode ~ONE frame instead of forward-decoding the whole GOP
                # (~300 frames) to the EXACT timestamp: on an 8K VR file that's
                # ~0.3 s vs ~5-25 s. (SLICE-threaded forward-decode is ~5x slower
                # than the preview's FRAME threading, which we can't use here.)
                # The still lands on a keyframe within one GOP of seek_time —
                # still past the intro, which is the whole point of seek_time.
                # Prefer the keyframe AT/AFTER seek_time (backward=False); fall
                # back to the one before it when seek_time is near end-of-file.
                if stream.time_base:
                    target_pts = int(self.seek_time / stream.time_base)
                else:
                    target_pts = int(self.seek_time * 1_000_000)
                _decode_start = _time.monotonic()
                best_frame = None
                for _backward in (False, True):
                    try:
                        container.seek(target_pts, stream=stream,
                                       backward=_backward)
                    except Exception:
                        pass   # some containers don't support this seek flag
                    for frame in container.decode(stream):
                        if self._cancelled:
                            return None, 0.0
                        best_frame = frame
                        break   # the keyframe we sought to IS the thumbnail
                    if best_frame is not None:
                        break

                _decode_dt = _time.monotonic() - _decode_start
                if _decode_dt > 5.0:
                    log.warning("Slow thumbnail keyframe decode [%s]: %.1fs",
                                os.path.basename(self.video_path), _decode_dt)
                if best_frame is None:
                    return None, 0.0

                # Convert to BGR ndarray (matches cv2 convention for the
                # downstream resize / store_thumbnail code path)
                arr = best_frame.to_ndarray(format='bgr24')
                return arr, duration
            finally:
                # Close container ONLY from this (worker) thread.
                try:
                    if self._container is container:
                        self._container = None
                    container.close()
                except Exception:
                    pass
        except Exception as e:
            log.debug("PyAV thumbnail extract failed [%s]: %s",
                      os.path.basename(self.video_path), e)
            return None, 0.0

    def _extract_frame_cv2(self, prev_duration: float) -> 'tuple[np.ndarray | None, float]':
        """cv2 fallback for files PyAV can't handle.  Same return contract.

        No lock here: the caller (_do_work) already holds a
        background_section from the disk coordinator, so cv2's
        VideoCapture open cannot overlap a foreground preview/seek.
        """
        from video_capture_helper import open_capture
        cap = open_capture(self.video_path, hw_accel=True)
        if cap is None:
            self.cache_manager.mark_thumbnail_failed(
                self.video_path, "Cannot open video"
            )
            return None, prev_duration
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if fps > 0 and fps < 1000:
                duration = total_frames / fps
            else:
                fps = 25.0
                duration = prev_duration
            if total_frames > 0 and fps > 0:
                target = int(self.seek_time * fps)
                target = max(0, min(target, total_frames - 1))
                cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            ret, frame = cap.read()
        finally:
            cap.release()
        return (frame if ret else None), duration

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
    thumbnail_ready  = pyqtSignal(str, float, object, float)
    # video_path, reason — emitted when generation gives up (e.g. corrupt file)
    thumbnail_failed = pyqtSignal(str, str)

    # Maximum seconds a single thumbnail worker may run before the
    # watchdog force-closes its file handle.  Tuned for 8K HEVC on HDD:
    # PyAV typically finishes in <1 s, but pathological files can hang
    # cv2 for tens of seconds.  30 s gives ample headroom while ensuring
    # the user can delete the file within reasonable time after cancel.
    WORKER_TIMEOUT_SEC = 30

    def __init__(self, cache_manager: CacheManager, parent=None):
        super().__init__(parent)
        self._cache = cache_manager
        # Dedicated pool, NOT the global one — lets us pick the worker
        # count without affecting Qt's own thread usage.  2 workers is
        # the sweet spot for HDD: enough parallelism to overlap network
        # I/O with decode, but not so many that head seeks thrash.
        self._pool = QThreadPool()
        # 3 concurrent thumbnail workers (per the coordinator design). The
        # disk coordinator guarantees these never overlap a foreground
        # preview/seek, so 3 is safe: at most 3 short background reads, never
        # competing with hover playback. Bounded head movement on HDD.
        self._pool.setMaxThreadCount(3)
        self._mutex = QMutex()
        self._pending: dict[str, ThumbnailWorker] = {}

        # Watchdog: every few seconds, force-close any worker that's
        # been running longer than WORKER_TIMEOUT_SEC.  Without this,
        # cv2 reading a corrupted/huge file can hang for minutes, holding
        # the file handle open and blocking the user from deleting it.
        self._watchdog = QTimer(self)
        self._watchdog.setInterval(5000)
        self._watchdog.timeout.connect(self._enforce_timeouts)
        self._watchdog.start()

    def request_thumbnail(self, video_path: str, seek_time: float,
                          priority: int = 0):
        """Queue thumbnail generation. Does NO work on the calling thread.

        `priority` maps to QThreadPool's run priority (higher → sooner).
        Preview-preempted re-queues use a higher priority so a video the user
        just looked at regenerates before the cold backlog (see _on_finished).
        """
        with QMutexLocker(self._mutex):
            if video_path in self._pending:
                return
            worker = ThumbnailWorker(video_path, seek_time, self._cache)
            worker.signals.thumbnail_ready.connect(self._on_ready)
            worker.signals.error.connect(self._on_error)
            worker.signals.finished.connect(self._on_finished)
            self._pending[video_path] = worker
        self._pool.start(worker, priority)

    def regenerate_thumbnail(self, video_path: str, seek_time: float):
        """Cancel any in-flight worker, invalidate cache, regenerate.

        Called from the GUI thread when the user drags the seek slider.
        The SQLite writes (`invalidate`, `set_seek_override`,
        `clear_thumbnail_failure`) can block under contention when many
        thumbnail workers are also writing — that was the user-reported
        "stuck in generating" symptom. We queue them through the cache's
        background writer so the GUI thread never waits on SQLite. The
        order of the queued writes is preserved (single-threaded writer).
        The worker itself doesn't depend on these writes finishing first:
        seek_time is passed by value, and a new JPEG path won't collide
        with the old one even if invalidate runs slightly late.
        """
        try:
            self._cache._writer.submit(self._cache.invalidate, video_path)
            self._cache._writer.submit(
                self._cache.set_seek_override, video_path, seek_time)
            self._cache._writer.submit(
                self._cache.clear_thumbnail_failure, video_path)
        except AttributeError:
            # Fallback if the background writer isn't available for some
            # reason (unit tests, older CacheManager): do them inline.
            self._cache.invalidate(video_path)
            self._cache.set_seek_override(video_path, seek_time)
            self._cache.clear_thumbnail_failure(video_path)

        with QMutexLocker(self._mutex):
            old = self._pending.pop(video_path, None)
            if old:
                old.cancel()
            worker = ThumbnailWorker(video_path, seek_time, self._cache)
            worker.signals.thumbnail_ready.connect(self._on_ready)
            worker.signals.error.connect(self._on_error)
            worker.signals.finished.connect(self._on_finished)
            self._pending[video_path] = worker
        self._pool.start(worker)

    def cancel_all(self):
        """Cancel all queued/running workers and wait for the pool to drain.

        IMPORTANT: also force-closes the PyAV container of each running
        worker so that any in-flight decode immediately fails — this is
        what unblocks the user from deleting a file whose thumbnail
        worker is mid-read.  Without force-close, cv2.read() on a huge
        file holds the OS file handle for tens of seconds and the
        subsequent send2trash call fails with sharing violation."""
        with QMutexLocker(self._mutex):
            workers = list(self._pending.values())
            self._pending.clear()
        for w in workers:
            # requeue=False: the folder is being torn down / a file deleted —
            # do NOT regenerate these (and _pending is already cleared above).
            w.force_close(requeue=False)   # cancel flag + release file handle
        # Wait for the pool, but with a longer budget — PyAV's close()
        # makes pending decodes raise on the worker thread, which then
        # returns quickly via the except clause in run().
        self._pool.waitForDone(5000)

    def _enforce_timeouts(self):
        """Called by the watchdog every 5 s.  Any worker that's been
        running longer than WORKER_TIMEOUT_SEC gets force-closed so it
        releases the OS file handle."""
        now = _time.monotonic()
        timed_out: list = []
        with QMutexLocker(self._mutex):
            for path, w in list(self._pending.items()):
                if w._started_at > 0 and (now - w._started_at) > self.WORKER_TIMEOUT_SEC:
                    timed_out.append((path, w))
                    self._pending.pop(path, None)
        for path, w in timed_out:
            log.warning("Thumbnail worker timeout (>%ds) for %s — force-closing",
                        self.WORKER_TIMEOUT_SEC, os.path.basename(path))
            # requeue=False: retry for a timed-out worker is driven by the
            # thumbnail_failed("Timeout") signal below (the grid's retry), and
            # _pending was already popped above — so _on_finished must not also
            # re-queue it.
            w.force_close(requeue=False)
            # Transient: a slow/contended decode that blew the worker timeout
            # (common during a mass regen on an HDD). Do NOT cache it as a
            # permanent failure — let the grid retry it once the disk frees up.
            self.thumbnail_failed.emit(path, "Timeout")

    # ── slots (called on main thread via queued connection) ─────────────────
    def _on_ready(self, video_path: str, seek_time: float,
                  qimage: object, duration: float):
        with QMutexLocker(self._mutex):
            self._pending.pop(video_path, None)
        self.thumbnail_ready.emit(video_path, seek_time, qimage, duration)

    def _on_error(self, video_path: str, msg: str):
        # "Cached failure" is expected after the first try fails — log only
        # at debug level so it doesn't spam the warning log on every scroll.
        if msg == "Cached failure":
            log.debug("Thumbnail skip (cached failure): %s",
                      os.path.basename(video_path))
        else:
            log.warning("Thumbnail error [%s]: %s",
                        os.path.basename(video_path), msg)
        with QMutexLocker(self._mutex):
            self._pending.pop(video_path, None)
        self.thumbnail_failed.emit(video_path, msg)

    def _on_finished(self, worker: 'ThumbnailWorker'):
        """Terminal handler for EVERY worker exit (success / error / cancel),
        emitted from run()'s finally. Its job is to guarantee _pending never
        leaks: a CANCELLED worker emits no ready/error, so without this its
        path would stay pending and request_thumbnail() would dedup every
        retry → the card is stuck "Generating…".

        Uses an IDENTITY check so a stale finish (e.g. an old worker that
        regenerate_thumbnail already replaced, or one cancel_all already
        cleared) can't disturb a newer worker for the same path. A worker that
        was cancelled to free the disk for a hover preview is RE-QUEUED so the
        thumbnail regenerates once the disk frees again."""
        path = worker.video_path
        with QMutexLocker(self._mutex):
            if self._pending.get(path) is not worker:
                return   # superseded / already cleared — nothing to do
            self._pending.pop(path, None)
        if worker._cancelled and worker._requeue_on_cancel:
            # Preempted by a foreground preview/seek mid-decode — the thumbnail
            # is still wanted, and it's a video the user JUST looked at. Re-queue
            # at HIGH priority so it regenerates ahead of the cold backlog the
            # moment the disk frees, instead of landing at the back of a long
            # queue (which left just-browsed cards stuck "Generating…"). The
            # disk coordinator still makes it wait until foreground is done.
            self.request_thumbnail(path, worker.seek_time, priority=1)
