"""
PyAV-based video playback thread — drop-in replacement for _VideoPlayThread.

Why
───
The pip `opencv-python` wheel ships an FFmpeg without working HW decode
for HEVC on Windows.  PyAV (https://pyav.org) is a thin Python binding
for FFmpeg whose pip wheel includes a more complete build: D3D11VA on
Windows, VideoToolbox on macOS, VAAPI on Linux.  When the GPU is
available, hardware decode drops per-frame cost from 50–200 ms (CPU
HEVC) to ~5 ms.

Even WITHOUT HW accel, PyAV's SW decoder uses the same AVX2 + multi-
threaded libavcodec as OpenCV — so this module is at minimum a
pass-through-quality replacement with cleaner failure modes (no
"[hevc] Failed setup for format d3d11" stderr spam).

API parity with `video_thumbnail_widget._VideoPlayThread`
─────────────────────────────────────────────────────────
- signal  `frame_ready(QImage)`
- method  `stop()`                      → request abort at next frame
- method  `request_seek(sec)`           → seek before next decode
- method  `set_display_size(w, h)`      → live-update target reformat size
- Qt signal `finished`                  → auto-handled by qthread_registry

Threading
─────────
QThread.  All FFmpeg calls happen on the worker; only the emitted
QImage crosses the thread boundary (via Qt queued connection).
"""
import os
import platform
import threading as _threading
import time as _time

import numpy as np

import vr_unwarp as vu

try:
    import av
    HAS_PYAV = True
    _PYAV_IMPORT_ERROR = None
    # Silence PyAV's libav logger.  This is SEPARATE from OpenCV's FFmpeg
    # logging — PyAV ships its own libav, so OPENCV_FFMPEG_LOGLEVEL doesn't
    # affect it.  Without this, multi-GB HEVC files print
    # `[matroska,webm @ ...] Unsupported encoding type` / EBML errors to
    # stderr directly from C code, regardless of Python's logging config.
    try:
        # PyAV's logging levels: 0 quiet, 8 panic, 16 fatal, 24 error,
        # 32 warning, 40 info.  "Unsupported encoding type" is a WARNING
        # that fires per-stream for HEVC HDR DV files even when decode
        # works — pure noise.  Set to FATAL so only catastrophic errors
        # surface (and they'll be raised as Python exceptions anyway).
        av.logging.set_level(av.logging.FATAL)
        # CRITICAL (deadlock fix): remove PyAV's *Python* log callback and
        # restore libav's built-in C callback. By default PyAV forwards every
        # libav log line to Python, which means a libav FRAME-decoder WORKER
        # thread that emits a log (e.g. the h264 "num_reorder_frames" warning)
        # calls PyGILState_Ensure to acquire the GIL — while the main decode
        # thread holds the GIL blocked inside avcodec waiting for that very
        # worker. That is a hard GIL⊗decoder deadlock that froze the whole app
        # mid-playback (observed: every thread parked in PyEval_RestoreThread,
        # one libav worker stuck in av_log→PyGILState_Ensure). set_level alone
        # does NOT prevent it — the callback still grabs the GIL. Restoring the
        # C callback means workers log in pure C (no GIL), so the deadlock is
        # structurally impossible. set_libav_level keeps that C output quiet.
        try:
            av.logging.restore_default_callback()
            av.logging.set_libav_level(av.logging.FATAL)
        except Exception:
            pass
    except Exception:
        pass
except ImportError as e:
    HAS_PYAV = False
    _PYAV_IMPORT_ERROR = str(e)

from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QImage

from app_logger import get_logger
from qthread_registry import install
log = get_logger(__name__)


def is_available() -> bool:
    """True if `import av` succeeded — caller can use _PyAVPlayThread."""
    return HAS_PYAV


# Timeout passed to av.open() so it never blocks indefinitely. libav respects
# this for both demux-init reads and subsequent I/O. Tuple is
# (open_timeout, read_timeout). Open: 6s — long enough for the seek-index
# read of a 30 GB MKV on HDD even under file system contention, but short
# enough that a wedged worker dies before piling up. Read: 2s — once decode
# is going, any single read taking >2s indicates a problem.
# libav open/read timeouts (seconds). These MUST tolerate an HDD waking from
# a low-power/parked state: a spun-down drive can take ~5 s to spin up, during
# which the very first read blocks. A 2 s read timeout aborted the open
# mid-spin-up and the preview/thumbnail failed for no real reason. These run
# on a WORKER thread and release the GIL while blocked, so a generous bound
# never freezes the GUI — it just lets a cold drive finish waking up.
_PYAV_OPEN_TIMEOUT_S = 20.0
_PYAV_READ_TIMEOUT_S = 15.0

# Internal libav decoder thread count for hover preview. The default 75%-of-
# cores (24 on a 32-core CPU) was great for FPS but disastrous for cleanup:
# closing the file mid-decode left ~24 orphaned decoder threads per play
# thread. With 4 threads + SLICE threading, the per-stop leak is bounded
# and the per-frame decode is still well under 100 ms for 8K HEVC.
_HOVER_THREAD_COUNT = 4

# Max seconds a preview thread waits for FOREGROUND disk access (i.e. for
# in-flight background thumbnail workers to finish their current file and
# park) before proceeding in degraded mode. Sized to cover the worst-case
# drain: a thumbnail worker mid-av.open on a huge MKV bounded by
# _PYAV_OPEN_TIMEOUT_S. See disk_coordinator.DiskAccessCoordinator.
_PYAV_FG_WAIT_S = 8.0

class _PreviewManager:
    """Serializes hover-preview threads so AT MOST ONE is ever alive.

    Why this exists (the framework invariant): a preview thread decodes via
    PyAV using a Python file object (pyio). Two PyAV decodes running on
    Python file objects concurrently can deadlock on the GIL (a PyAV
    reentrancy hazard) — and even short of a deadlock, two threads reading
    different multi-GB files thrash the HDD. The disk coordinator guarantees
    only one FOREGROUND *holder*, but a thread that was told to stop keeps
    touching the disk until its decode loop next checks the flag (which can
    be hundreds of ms while decoding an 8K frame). During that window a
    newly-started preview overlaps the dying one → the observed wedge.

    The manager removes the overlap entirely: starting a new preview signals
    the current one to stop and DEFERS the new start until the old thread's
    `finished` signal fires. Result: the new thread's first disk touch only
    happens after the old thread has fully exited. Never two at once.

    All methods run on the GUI thread (submit/cancel from widget handlers,
    _on_finished from the queued `finished` signal). Non-blocking — the GUI
    never waits for a thread to die; it just gets a callback when it does.
    """
    def __init__(self):
        self._current = None        # the started, running thread (or None)
        self._pending = None        # a built-but-not-started thread (or None)

    def submit(self, thread) -> None:
        """Request that `thread` (already constructed + signal-connected, but
        NOT started) become the one live preview. Supersedes any pending
        request. Starts immediately if nothing is running; otherwise stops
        the current one and starts this when it finishes."""
        if self._pending is not None and self._pending is not thread:
            self._discard_pending(self._pending)   # superseded before it launched
        self._pending = thread
        cur = self._current
        if cur is not None:
            try:
                running = cur.isRunning()
            except RuntimeError:
                running = False
            if running:
                try:
                    cur.stop()          # finished → _on_finished → _launch
                except (RuntimeError, AttributeError):
                    pass
                return
        self._launch()

    def _launch(self) -> None:
        t = self._pending
        self._pending = None
        if t is None:
            return
        self._current = t
        try:
            t.finished.connect(lambda bound=t: self._on_finished(bound),
                               Qt.ConnectionType.QueuedConnection)
            t.start()
        except (RuntimeError, AttributeError):
            self._current = None

    def _on_finished(self, thread) -> None:
        if self._current is thread:
            self._current = None
        if self._pending is not None and self._current is None:
            self._launch()

    def cancel(self, thread=None) -> None:
        """Stop the preview. If `thread` is given, only act if it is the
        current/pending one (so a widget leaving doesn't kill another
        widget's preview). If None, cancel unconditionally."""
        if thread is not None:
            if self._pending is thread:
                self._discard_pending(self._pending)
                self._pending = None
            if self._current is not thread:
                return
        else:
            if self._pending is not None:
                self._discard_pending(self._pending)
            self._pending = None
        cur = self._current
        if cur is not None:
            try:
                cur.stop()
            except (RuntimeError, AttributeError):
                pass

    def cancel_all(self) -> None:
        """Drop any pending and stop the current — used on scroll/teardown."""
        if self._pending is not None:
            self._discard_pending(self._pending)
        self._pending = None
        cur = self._current
        if cur is not None:
            try:
                cur.stop()
            except (RuntimeError, AttributeError):
                pass

    @staticmethod
    def _discard_pending(t) -> None:
        """Clean up a preview thread that was built + registered but is being
        superseded/cancelled BEFORE it ever start()ed. It never runs, so it
        would otherwise linger in the strong-ref registries forever (it never
        finishes → the reaper, which now reaps only FINISHED threads, never
        touches it). It is not running, so deleting it now is safe — and removing
        it from the registries means a never-started thread can't be revived."""
        if t is None:
            return
        try:
            from video_thumbnail_widget import _running_play_threads
            _running_play_threads.discard(t)
        except Exception:
            pass
        try:
            import qthread_registry
            qthread_registry.unregister(t)
        except Exception:
            pass
        try:
            if not t.isRunning() and not t.isFinished():
                t.deleteLater()
        except (RuntimeError, AttributeError):
            pass


# Process-wide singleton: at most ONE preview thread alive across all widgets.
PREVIEW_MANAGER = _PreviewManager()


def import_error() -> str:
    return _PYAV_IMPORT_ERROR or ""


# Per-platform list of HW accel methods to attempt, in priority order.
# We try them sequentially; PyAV raises on init failure so we can detect
# and move on without printing the noisy [hevc] errors that cv2 produces.
def _hw_methods() -> 'list[str]':
    sysname = platform.system()
    if sysname == 'Windows':
        return ['d3d11va', 'dxva2', 'cuda']
    if sysname == 'Linux':
        return ['vaapi', 'cuda', 'vdpau']
    if sysname == 'Darwin':
        return ['videotoolbox']
    return []


class _PyAVPlayThread(QThread):
    """Plays a video file via PyAV's libavcodec binding.

    Tries HW decode first (when `hw_accel=True`); falls back to multi-
    threaded SW decode automatically on HW init failure.  Either way,
    decode runs on this QThread — never on the Qt main thread.
    """
    frame_ready = pyqtSignal(object)   # QImage

    def __init__(self, video_path: str, start_sec: float,
                 display_w: int, display_h: int,
                 *, hw_accel: bool = True, target_fps: float = 0.0,
                 thread_type: str = 'AUTO',
                 skip_frame: 'str | None' = None):
        """thread_type:
            'SLICE'  - parallel slice decode within one frame; safer for
                       frequent seeking (no lookahead buffer to flush).
            'FRAME'  - lookahead-based parallel frame decode.  Much
                       higher throughput for continuous playback —
                       benchmarked at 73 fps for 8K 60fps HEVC where
                       SLICE caps at ~19 fps on the same 24-thread CPU.
                       Use for hover preview.
            'AUTO'   - libavcodec picks (often picks SLICE for HEVC).

        skip_frame: optional libavcodec frame-discard level.  Accepted
            values: None / 'NONE' / 'DEFAULT' / 'NONREF' / 'BIDIR' /
            'NONINTRA' / 'NONKEY' / 'ALL'.  Use as a fallback if even
            FRAME threading can't sustain native FPS — drops frames
            during decode rather than wasting cycles on them.
        """
        super().__init__()
        if not HAS_PYAV:
            raise RuntimeError(
                "PyAV is not installed. `pip install av` to enable the "
                "PyAV-based player thread."
            )
        self._path        = video_path
        self._start_sec   = start_sec
        self._disp_w      = display_w
        self._disp_h      = display_h
        self._stop_flag   = False
        # VR → 2D: detected lazily from the first frame's full size ('?' = not yet
        # checked; None = not VR; 'left' etc. = un-warp this eye).
        self._vr_eye      = '?'
        self._vr_unwarper = None
        self._seek_to: 'float | None' = None
        # Video timestamp (seconds) of the most recently EMITTED frame — lets
        # callers/tests observe true playback progress (for the real-time
        # frame-dropping pacing) without inspecting QImages.
        self._playback_vt: 'float | None' = None
        self._hw_accel    = hw_accel
        self._target_fps  = float(target_fps)
        self._thread_type = thread_type
        self._skip_frame  = skip_frame
        self.active_hwaccel: 'str | None' = None
        # PyAV `av.open(path)` is NOT interruptible — it blocks the worker
        # thread inside libav file-open for as long as the disk takes (10–30 s
        # on HDD for a 30 GB MKV). When the user rapidly hovers different
        # widgets, stop() sets a flag the in-flight thread can't observe
        # until av.open returns, and we accumulate stuck threads.
        #
        # Fix: open the file ourselves as a Python file object and pass it
        # to av.open(). stop() closes the file object — libav's next read
        # then fails with EOF/EIO, av.open or container.decode raises, and
        # the worker thread exits within its own try/except. We never touch
        # the PyAV container from another thread.
        self._file_obj: 'object | None' = None
        # True only while inside av.open(). stop() async-closes the file ONLY
        # in this window — closing it mid-DECODE would orphan libav's internal
        # decoder threads (they can't be joined once their IO is yanked),
        # leaving zombie C++ threads that block process/Qt shutdown. When
        # playing, stop() just sets the flag and lets the decode loop exit so
        # container.close() in the finally joins those threads cleanly.
        self._opening = False
        # Stable foreground label, set here (not in _play) so stop() can
        # release the disk-coordinator gate even if it's called before _play
        # has started running.
        self._fg_label = f"preview:{id(self)}"
        install(self)

    # ── thread-safe attribute setters (GIL-safe for ints / floats) ───────────
    def set_display_size(self, w: int, h: int):
        self._disp_w = w
        self._disp_h = h

    def request_seek(self, sec: float):
        self._seek_to = sec

    def stop(self):
        """Signal the play thread to stop. NON-BLOCKING and safe to call
        from the GUI thread.

        Sets _stop_flag — the decode loop checks it between frames and exits
        within one frame. A thread stuck inside a slow av.open() will finish
        the open (bounded by the av.open `timeout`) and then immediately see
        the flag and exit. We do NOT need to forcibly abort the open: with
        NATIVE av.open(path) the open releases the GIL during its reads, so a
        slow open never wedges the GUI — it just delays this one preview,
        and the PREVIEW_MANAGER waits for us on a worker thread, never on the
        GUI thread.

        Also releases the disk-coordinator foreground gate immediately
        (label-aware, so a stale release is a no-op) so the next preview and
        the thumbnail workers aren't blocked behind a slow teardown."""
        self._stop_flag = True
        # Cancel any pending seek so a thread told to stop does NOT waste time
        # doing a deep keyframe seek on a multi-GB file that it will never get
        # to display. Without this, a hover→seek→leave/switch sequence makes
        # the dying thread run a slow backward-seek before it can exit, which
        # both delays the next preview AND widens the teardown race window.
        self._seek_to = None
        try:
            from disk_coordinator import COORDINATOR
            COORDINATOR.end_foreground(self._fg_label)
        except Exception:
            pass

    # Backwards-compat alias.
    force_close = stop

    # ── open / HW accel selection ────────────────────────────────────────────
    def _open_container(self):
        """Open the container; try HW codecs first, fall back to SW.
        Returns (container, stream).  Raises on hard open failure.

        PyAV 17+ accepts an `hwaccel=HWAccel(...)` kwarg directly on av.open(),
        and `allow_software_fallback=True` lets libavcodec switch to SW
        decode per-frame if HW init fails — so we get robust playback
        even when the HW path is partially broken (e.g. an unsupported
        codec profile).
        """
        if self._hw_accel:
            for method in _hw_methods():
                if self._stop_flag:
                    raise RuntimeError("stopped before open")
                try:
                    container, stream = self._open_with_hw(method)
                    log.debug("PyAV: opened %s with HW accel '%s'",
                             os.path.basename(self._path), method)
                    self.active_hwaccel = method
                    return container, stream
                except Exception as e:
                    log.debug("PyAV: HW '%s' failed: %s", method, e)
                    continue
            log.debug("PyAV: all HW backends failed, falling back to SW")

        return self._open_sw()

    def _open_with_hw(self, method: str):
        """Try to open with the named HW backend.  Raises on init failure.

        Uses `allow_software_fallback=True` so even if a SPECIFIC frame
        can't be decoded via HW, libavcodec transparently does that frame
        via SW.  This is much more robust than per-cap probing.
        """
        hw = av.codec.hwaccel.HWAccel(
            device_type=method,
            allow_software_fallback=True,
        )
        # NATIVE path open (NOT a Python file object). This is critical: with
        # a Python file object, libav routes every read through a Python
        # callback (pyio) that holds the GIL — on huge files with deep seeks
        # that means thousands of GIL-grabbing callbacks during the open,
        # which STARVES the GUI thread and freezes the UI. av.open(path) reads
        # natively in C and releases the GIL, so the GUI stays responsive even
        # while a 27 GB file opens. `timeout` bounds a truly-stuck open.
        if self._stop_flag:
            raise RuntimeError("stopped before open")
        container = av.open(self._path, hwaccel=hw,
                            timeout=(_PYAV_OPEN_TIMEOUT_S, _PYAV_READ_TIMEOUT_S))
        try:
            stream = container.streams.video[0]
            # Also configure thread params for the SW fallback path
            self._configure_codec_sw(stream.codec_context)
            # Probe: demux+decode a single packet to surface init failures
            # NOW rather than partway through playback.
            iterator = container.demux(stream)
            for packet in iterator:
                frames = list(packet.decode())
                if frames:
                    container.seek(0, stream=stream, backward=True)
                    return container, stream
            raise RuntimeError("no decodable packets in stream")
        except Exception:
            try:
                container.close()
            except Exception:
                pass
            raise

    def _open_sw(self):
        """Plain software-decode open via NATIVE path (not a Python file
        object) so libav reads in C and releases the GIL — see _open_with_hw
        for why pyio (file-object) opens freeze the GUI on huge files."""
        if self._stop_flag:
            raise RuntimeError("stopped before SW open")
        container = av.open(self._path,
                            timeout=(_PYAV_OPEN_TIMEOUT_S, _PYAV_READ_TIMEOUT_S))
        stream = container.streams.video[0]
        self._configure_codec_sw(stream.codec_context)
        log.debug("PyAV: opened %s with SW decode",
                 os.path.basename(self._path))
        return container, stream

    def _configure_codec_sw(self, ctx):
        """Configure codec context for fastest SW decode.

        Sets thread_count to the same 75%-of-cores value used by cv2's
        FFmpeg backend (or the user's explicit cpu_decode_threads override).
        libavcodec internally caps at 16 per codec instance; over-asking
        is a no-op.

        thread_type comes from __init__.  'FRAME' is dramatically faster
        for HEVC on multi-core CPUs (8K 60fps HEVC: 73 fps vs 19 fps with
        'SLICE') because of lookahead-based parallelism.  Use 'FRAME'
        for hover preview; 'SLICE' for the player panel where the user
        may seek frequently.
        """
        try:
            # Read config via getattr with defaults so a partially-initialised
            # instance never crashes here (and silently skips configuration).
            thread_type = getattr(self, '_thread_type', None) or 'SLICE'
            skip_frame  = getattr(self, '_skip_frame', None)
            # For hover preview (thread_type='FRAME'), we cap the decoder
            # at a small thread count. This is the only way to keep the
            # per-stop leak bounded — closing the file mid-decode orphans
            # whatever decoder threads libav spawned, so fewer threads =
            # less leak. The player panel still gets the full 75%-of-
            # cores allocation (via SLICE threading) for max FPS.
            if thread_type == 'FRAME':
                n = _HOVER_THREAD_COUNT
            else:
                try:
                    from main import compute_decode_threads
                    n = compute_decode_threads()
                except Exception:
                    n = max(1, int((os.cpu_count() or 4) * 0.75))
            ctx.thread_count = n
            ctx.thread_type = thread_type
            if skip_frame is not None:
                try:
                    ctx.skip_frame = skip_frame
                except (ValueError, Exception) as e:
                    log.debug("skip_frame=%r rejected: %s", skip_frame, e)
        except Exception:
            pass

    # ── run loop ─────────────────────────────────────────────────────────────
    def run(self):
        try:
            self._play()
        except Exception as e:
            log.error("PyAV play thread error: %s", e, exc_info=True)

    def _play(self):
        # ── Acquire FOREGROUND disk access ──────────────────────────────
        # This blocks THIS worker thread (never the GUI) until every
        # background thumbnail worker has finished its current file and
        # parked. Once we hold foreground, no thumbnail worker touches the
        # disk, so hover playback gets the drive head to itself — no thrash.
        # Released in the finally below.
        from disk_coordinator import COORDINATOR
        self._fg_label = f"preview:{id(self)}"
        self._holds_foreground = COORDINATOR.begin_foreground(
            self._fg_label, self._path, timeout_s=_PYAV_FG_WAIT_S)
        if not self._holds_foreground:
            # Degraded: a background worker is wedged. Proceed anyway so the
            # user still gets a preview attempt (the coordinator already
            # logged a warning). Most of the time this branch never runs.
            log.debug("PyAV: starting preview WITHOUT exclusive disk on %s",
                      os.path.basename(self._path))
        try:
            self._play_inner()
        finally:
            try:
                COORDINATOR.end_foreground(self._fg_label)
            except Exception:
                pass

    def _play_inner(self):
        try:
            # _opening gates whether stop() may async-close the file (see stop).
            self._opening = True
            try:
                container, stream = self._open_container()
            finally:
                self._opening = False
        except Exception as e:
            # Common case when stop() interrupts av.open() via file close —
            # log at debug level so the noise doesn't drown real errors.
            if self._stop_flag:
                log.debug("PyAV: open interrupted by stop() on %s",
                          os.path.basename(self._path))
            else:
                log.error("PyAV: could not open %s: %s",
                          os.path.basename(self._path), e)
            return

        try:
            avg_rate = stream.average_rate
            native_fps = float(avg_rate) if avg_rate else 25.0
            if not (0 < native_fps < 300):
                native_fps = 25.0
            effective_fps = (self._target_fps if self._target_fps > 0
                             else native_fps)
            frame_interval = 1.0 / effective_fps   # seconds per frame

            # Initial seek (if start_sec > 0)
            if self._start_sec > 0:
                self._seek_container(container, stream, self._start_sec)

            iterator = container.decode(stream)

            # Adaptive output resolution (shrink if behind on FPS). Smaller
            # frames are cheaper to reformat AND to marshal across the thread,
            # so shrinking directly raises the sustainable real-time rate.
            scale_factor   = 1.0
            min_scale      = 0.5
            frames_emitted = 0
            recent_frames  = []

            # ── Real-time presentation clock with frame DROPPING ───────────
            # The decoder is much faster than the GUI can DISPLAY 8K-sourced
            # frames: decode ~95 fps, but cross-thread QImage delivery caps the
            # GUI at ~25 fps. If we reformat+emit EVERY decoded frame the
            # preview plays in SLOW MOTION — it shows all 60 fps of content but
            # only ~25 reach the screen each second (~0.4× speed). Fix: anchor
            # wall-clock to the stream PTS and only spend the expensive
            # reformat+emit on a frame that is DUE. Frames that are already late
            # (GUI can't keep up) are DROPPED cheaply — we still pull them from
            # the decoder (needed for P-frame references) but skip the costly
            # scale/convert/emit. Result: playback tracks REAL TIME at whatever
            # rate the GUI sustains, instead of running slow. Anchors reset on
            # every seek / EOF-loop.
            anchor_wall = None     # perf_counter at this segment's first frame
            anchor_vt   = None     # that frame's video timestamp (seconds)
            last_emit_t = 0.0      # wall time of the last EMITTED frame
            tb          = stream.time_base
            # Rolling (EMA) cost of one next(iterator) decode. Frame DROPPING
            # only helps when the DECODER outruns real time and the GUI delivery
            # is the bottleneck — then skipping the costly reformat+emit lets us
            # present a fresher frame on schedule. If the DECODE itself can't
            # keep up (e.g. 8K 60fps HEVC that SW-decodes at only ~28 fps on
            # this CPU), dropping decoded frames makes the preview choppy WITHOUT
            # making it any more real-time. So we detect "decode can't keep up"
            # and then emit every frame (smooth, even if slower than 1×) instead.
            dec_ema     = frame_interval

            while not self._stop_flag:
                # Honor a pending seek BEFORE pulling the next frame — but not
                # if we've been told to stop in the meantime (stop() also
                # clears _seek_to; this guard closes the tiny window where the
                # flag flips between the while-check and here).
                if self._seek_to is not None and not self._stop_flag:
                    sec = self._seek_to
                    self._seek_to = None
                    self._seek_container(container, stream, sec)
                    iterator = container.decode(stream)
                    anchor_wall = None   # re-anchor real-time clock after seek
                    continue

                try:
                    _dec_t0 = _time.perf_counter()
                    frame = next(iterator)
                    dec_ema = 0.9 * dec_ema + 0.1 * (_time.perf_counter() - _dec_t0)
                except StopIteration:
                    # EOF → loop back to start_sec
                    self._seek_container(container, stream, self._start_sec)
                    iterator = container.decode(stream)
                    anchor_wall = None   # re-anchor after loop
                    continue
                except av.error.FFmpegError as e:
                    log.warning("PyAV decode error: %s", e)
                    break

                if self._stop_flag:
                    break

                # Real-time gate: decide whether THIS frame is due, late, or
                # early — using its presentation timestamp vs wall clock.
                vt = (float(frame.pts * tb)
                      if (frame.pts is not None and tb) else None)
                now = _time.perf_counter()
                if anchor_wall is None or vt is None:
                    anchor_wall = now
                    anchor_vt = vt if vt is not None else 0.0
                if vt is not None:
                    video_elapsed = vt - anchor_vt
                    wall_elapsed  = now - anchor_wall
                    behind = wall_elapsed - video_elapsed     # >0 ⇒ running late
                    # DROP a late frame (skip reformat+emit) to catch up to real
                    # time — but ONLY when the decoder is outrunning real time
                    # (otherwise dropping just adds choppiness, see dec_ema note
                    # above), and always refresh at least ~5 fps so a heavily
                    # behind preview never looks frozen.
                    decode_keeps_up = dec_ema < frame_interval
                    if (decode_keeps_up and behind > frame_interval
                            and (now - last_emit_t) < 0.20):
                        continue
                    # Early frame → sleep until it is actually due (paces to
                    # real time without spinning the GIL).
                    ahead = video_elapsed - wall_elapsed
                    if ahead > 0.002:
                        self.msleep(int(ahead * 1000))

                qimg = self._frame_to_qimage_scaled(frame, scale_factor)
                if qimg is not None and not self._stop_flag:
                    self.frame_ready.emit(qimg)
                    if vt is not None:
                        self._playback_vt = vt
                    last_emit_t = _time.perf_counter()
                    frames_emitted += 1
                    recent_frames.append(last_emit_t)
                    while recent_frames and (last_emit_t - recent_frames[0]) > 1.0:
                        recent_frames.pop(0)

                # Dynamic resource allocation: shrink output when the EMIT rate
                # is below native (display can't keep up), restore as it
                # recovers. Re-evaluated once per ~30 emitted frames.
                if frames_emitted >= 30 and len(recent_frames) >= 2:
                    span = recent_frames[-1] - recent_frames[0]
                    observed = (len(recent_frames) - 1) / span if span > 0 else 0
                    target = 1.0 / frame_interval
                    if observed < target * 0.95 and scale_factor > min_scale:
                        scale_factor = max(min_scale, scale_factor - 0.10)
                    elif observed > target * 0.99 and scale_factor < 1.0:
                        scale_factor = min(1.0, scale_factor + 0.05)
                    frames_emitted = 0
        finally:
            try:
                container.close()
            except Exception:
                pass
            # Release the underlying Python file object. stop() may have
            # already closed it from another thread — close() is idempotent
            # so the redundant call is harmless.
            f = self._file_obj
            self._file_obj = None
            if f is not None:
                try:
                    f.close()
                except Exception:
                    pass

    # ── frame conversion ─────────────────────────────────────────────────────
    def _frame_to_qimage_scaled(self, frame, scale_factor: float) -> 'QImage | None':
        """Like _frame_to_qimage but with an additional `scale_factor`
        applied to the display size — used by the adaptive resolution
        path when we're falling behind native FPS."""
        if scale_factor >= 1.0:
            return self._frame_to_qimage(frame)
        save_w, save_h = self._disp_w, self._disp_h
        try:
            self._disp_w = max(1, int(save_w * scale_factor))
            self._disp_h = max(1, int(save_h * scale_factor))
            return self._frame_to_qimage(frame)
        finally:
            self._disp_w, self._disp_h = save_w, save_h

    def _maybe_unwarp(self, arr, src_w, src_h):
        """If the source is a side-by-side VR frame, return a flat 16:9 un-warped
        eye; otherwise return arr unchanged. Detection + unwarper build happen
        once; any failure falls back to the raw frame."""
        try:
            eye = self._vr_eye
        except (AttributeError, RuntimeError):
            return arr           # not fully initialised (unit-test __new__) → no-op
        if eye == '?':
            eye = vu.detect_stereo_eye(src_w, src_h)
            if eye is not None and not vu.halves_look_stereo(arr):
                eye = None        # 2:1 but a 2D scene (halves differ) → not VR
            self._vr_eye = eye
            if eye is not None:
                try:
                    from vr_frame import for_path
                    self._vr_unwarper = for_path(self._path, eye=eye)
                except Exception as e:
                    log.debug("preview unwarper init failed: %s", e)
                    self._vr_eye = None
        if getattr(self, '_vr_unwarper', None) is None:
            return arr
        try:
            return np.ascontiguousarray(self._vr_unwarper.apply(arr))
        except Exception as e:
            log.debug("preview un-warp failed: %s", e)
            return arr

    def _frame_to_qimage(self, frame) -> 'QImage | None':
        """Scale-and-convert PyAV frame to RGB888 QImage at display size.
        Uses libswscale via PyAV's reformat() (SIMD-accelerated)."""
        dw, dh = self._disp_w, self._disp_h
        fw, fh = frame.width, frame.height
        if dw <= 0 or dh <= 0 or fw <= 0 or fh <= 0:
            return None

        if fw > dw or fh > dh:
            scale = min(dw / fw, dh / fh)
            nw = max(1, int(fw * scale))
            nh = max(1, int(fh * scale))
        else:
            nw, nh = fw, fh

        try:
            reformatted = frame.reformat(width=nw, height=nh, format='rgb24')
            arr = reformatted.to_ndarray()
        except Exception as e:
            log.debug("PyAV reformat failed: %s; using to_ndarray fallback", e)
            try:
                arr = frame.to_ndarray(format='rgb24')
                if arr.shape[1] > nw or arr.shape[0] > nh:
                    import cv2
                    arr = cv2.resize(arr, (nw, nh), interpolation=cv2.INTER_LINEAR)
            except Exception:
                return None

        # VR side-by-side → flatten to one un-warped eye (same look as the
        # player + the static thumbnail). Detected once from the FULL source
        # size; the remap is cached inside the unwarper.
        arr = self._maybe_unwarp(arr, fw, fh)

        # Build a FULLY SELF-CONTAINED QImage (deep copy of the pixels).
        #
        # The previous "one copy" optimization built a QImage that REFERENCED
        # a Python bytes buffer (qimg._raw_buffer = buf) and shipped it across
        # a queued (cross-thread) signal. That is a memory-safety hazard: when
        # PyQt marshals the QImage for the queued call it may shallow-copy the
        # QImage (implicitly shared), and if the original Python qimg (the only
        # thing keeping `buf` alive) is garbage-collected before the GUI thread
        # processes the event, the GUI reads freed memory — manifesting as
        # flaky hangs / access violations under rapid previewing. `.copy()`
        # makes the QImage own its pixels, so it is safe across threads no
        # matter what PyQt does with object lifetimes. Frames are display-sized
        # (tiny — e.g. 48×24 up to a few hundred px), so the copy is cheap.
        rh, rw, ch = arr.shape
        buf = arr.tobytes()
        qimg = QImage(buf, rw, rh, ch * rw,
                      QImage.Format.Format_RGB888).copy()
        return qimg

    # ── seek ─────────────────────────────────────────────────────────────────
    def _seek_container(self, container, stream, sec: float):
        """Seek to `sec` (seconds) using PTS in the stream's time base.
        `backward=True` lands on the nearest keyframe ≤ target — FFmpeg
        will decode forward from there to reach the target frame."""
        try:
            tb = stream.time_base
            if tb:
                target_pts = int(sec / tb)
            else:
                target_pts = int(sec * 1_000_000)   # microseconds fallback
            container.seek(target_pts, stream=stream, backward=True)
        except Exception as e:
            log.warning("PyAV: seek to %.2fs failed: %s", sec, e)
