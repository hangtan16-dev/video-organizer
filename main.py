"""Entry point for the Video Organizer application."""

import sys
import os
import traceback

# ── Silence FFmpeg / libav stderr spam BEFORE cv2 is imported anywhere ──
# OpenCV ships its own libavformat which prints "Unsupported encoding type"
# / "File ended prematurely" / "[hevc] Failed setup for format d3d11"
# directly to stderr from C code — those messages can't be intercepted by
# Python's logging module.  Setting these env vars before the first cv2
# import suppresses them at the source.
#
# Use unconditional assignment (not setdefault) because if these had been
# accidentally set to a verbose value upstream, suppression wouldn't work.
os.environ['OPENCV_FFMPEG_LOGLEVEL'] = '-8'      # quiet (FFmpeg AV_LOG_QUIET)
os.environ['OPENCV_LOG_LEVEL']       = 'OFF'
os.environ['AV_LOG_FORCE_NOCOLOR']   = '1'
# Belt-and-suspenders for direct FFmpeg consumers
os.environ['AV_LOG_LEVEL']           = 'quiet'

# ── Multi-threaded CPU decode via FFmpeg codec options ─────────────────────
# OPENCV_FFMPEG_CAPTURE_OPTIONS is parsed by cv2's FFmpeg backend when it
# opens a VideoCapture.  Syntax: "key1;value1|key2;value2".
#
#   threads;0          → use all available cores for decoding (auto)
#   thread_type;slice  → slice-level parallelism (safe with frequent seeks;
#                        avoids the lookahead buffer that "frame" parallelism
#                        uses, which can desync after a seek)
#
# Big win for large H.264/H.265 4K/8K VR files on multi-core CPUs:
# slice-parallel decode can cut per-frame decode time by 2–4× depending on
# how many slices the encoder produced.
#
# Must be set BEFORE the first `import cv2`, which happens transitively
# via main_window → thumbnail_generator → cv2.
os.environ.setdefault(
    'OPENCV_FFMPEG_CAPTURE_OPTIONS',
    'threads;0|thread_type;slice',
)

# ── Force the FFmpeg multimedia backend for the in-app full-screen player ──
# Qt 6.5+ ships a bundled-FFmpeg QMediaPlayer backend with D3D11VA HW decode
# AND built-in HEVC/H.265, so 8K VR HEVC plays smoothly without the Windows
# "HEVC Video Extensions" store codec. Set explicitly (before QApplication) so
# playback is deterministic regardless of any system default.
os.environ.setdefault('QT_MEDIA_BACKEND', 'ffmpeg')

# Ensure the script's directory is on sys.path so all local modules resolve.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtCore import Qt, QtMsgType, qInstallMessageHandler
from PyQt6.QtGui import QPalette, QColor

from app_logger import get_logger, log_path
from main_window import MainWindow

log = get_logger(__name__)


# ── Qt message handler: routes qDebug/qWarning/qCritical into Python logging ─
def _qt_message_handler(msg_type: QtMsgType, context, message: str):
    if msg_type == QtMsgType.QtDebugMsg:
        log.debug("[Qt] %s", message)
    elif msg_type == QtMsgType.QtInfoMsg:
        log.info("[Qt] %s", message)
    elif msg_type == QtMsgType.QtWarningMsg:
        log.warning("[Qt] %s", message)
    elif msg_type in (QtMsgType.QtCriticalMsg, QtMsgType.QtFatalMsg):
        log.critical("[Qt] %s", message)


# ── Global Python exception hook ─────────────────────────────────────────────
def _excepthook(exc_type, exc_value, exc_tb):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_tb)
        return
    msg = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    log.critical("Unhandled exception:\n%s", msg)
    # Show a user-facing dialog if the app is running
    app = QApplication.instance()
    if app is not None:
        QMessageBox.critical(
            None,
            "Unexpected Error",
            f"The application encountered an unexpected error and may be unstable.\n\n"
            f"{exc_type.__name__}: {exc_value}\n\n"
            f"Full details written to:\n{log_path()}",
        )


def compute_decode_threads(s=None) -> int:
    """Return how many CPU threads to assign to FFmpeg / libavcodec decode.

    Policy:
      - If the user set `cpu_decode_threads > 0` explicitly, use that.
      - Otherwise, use 75% of available logical cores (rounded down,
        at least 1).
    libavcodec internally caps a single codec instance at 16 threads,
    so on big machines (>21 cores) the cap dominates.  We pass the
    higher number anyway in case a future libavcodec lifts the limit.
    """
    if s is None:
        try:
            from app_settings import AppSettings
            s = AppSettings()
        except Exception:
            s = None
    explicit = (s.cpu_decode_threads if s is not None else 0)
    if explicit and explicit > 0:
        return int(explicit)
    return max(1, int((os.cpu_count() or 4) * 0.75))


def _apply_decode_settings():
    """Set OPENCV_FFMPEG_CAPTURE_OPTIONS with the explicit thread count
    BEFORE any cv2.VideoCapture is opened (the env var is read once
    when FFmpeg first opens a stream)."""
    try:
        n = compute_decode_threads()
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = (
            f'threads;{n}|thread_type;slice'
        )
        log.info("CPU decode threads: %d (of %d logical cores)",
                 n, os.cpu_count() or 0)
    except Exception as e:
        log.warning("Could not apply decode-threads setting: %s", e)


def _heal_bad_settings():
    """Detect and reset persisted settings that don't make sense — typically
    a residue of a test that wrote to the user's registry, or a hand-edit
    gone wrong.  Resetting an out-of-band value to the default is always
    safer than leaving it in a state that breaks hover preview entirely."""
    try:
        from PyQt6.QtCore import QSettings
        s = QSettings('VideoOrganizer', 'VideoOrganizer')
        # Min sensible values for each hover-preview knob. Anything below
        # these would make the feature unusable; reset to defaults.
        constraints = (
            ('hover_preview_max_gb',     1.0,   100.0,  'hover_preview_max_gb'),
            ('large_file_threshold_mb', 50,   1000000,  'large_file_threshold_mb'),
            ('hover_preview_fps_large',  2,        60,  'hover_preview_fps_large'),
        )
        healed = []
        for key, lo, hi, friendly in constraints:
            raw = s.value(key)
            if raw is None:
                continue
            try:
                val = float(raw)
            except (ValueError, TypeError):
                s.remove(key)
                healed.append(f'{friendly} (unparseable)')
                continue
            if not (lo <= val <= hi):
                s.remove(key)
                healed.append(f'{friendly} ({val} out of [{lo}, {hi}])')
        if healed:
            s.sync()
            log.warning("Reset out-of-range settings to defaults: %s",
                        ", ".join(healed))
    except Exception as e:
        log.warning("Could not heal settings: %s", e)


def _expected_decode_threads(opts: str) -> str:
    """Best-effort: explain what `threads;X` means in human terms so the user
    can see in the log how much parallelism they're getting."""
    try:
        parts = dict(p.split(';') for p in opts.split('|') if ';' in p)
        n = parts.get('threads', '?')
        if n == '0':
            ncpu = os.cpu_count() or 1
            # FFmpeg's auto budget for video codecs: min(ncpu * 1.5, 16)
            est = min(int(ncpu * 1.5), 16)
            return f"auto (≈{est} threads on this {ncpu}-core CPU; FFmpeg caps at 16)"
        return f"{n} threads (explicit)"
    except Exception:
        return "unknown"


_CRASH_FILE = None   # keep the crash-dump file handle alive for the app's life


def _enable_crash_dumps():
    """Dump the native, all-thread C traceback to crash.log on a hard crash
    (segfault / abort) — e.g. the use-after-free that can follow the Qt warning
    "QThread: Destroyed while thread is still running". app.log only captures
    that *warning* (via the Qt message handler), NOT the native stack that shows
    WHICH thread died — this fills the gap so the next crash is diagnosable."""
    global _CRASH_FILE
    try:
        import faulthandler
        import datetime
        crash_dir = os.path.dirname(log_path())
        os.makedirs(crash_dir, exist_ok=True)
        crash_path = os.path.join(crash_dir, 'crash.log')
        _CRASH_FILE = open(crash_path, 'a', buffering=1, encoding='utf-8')
        _CRASH_FILE.write(
            f"\n===== session {datetime.datetime.now():%Y-%m-%d %H:%M:%S} =====\n")
        _CRASH_FILE.flush()
        faulthandler.enable(file=_CRASH_FILE, all_threads=True)
        log.info("Native crash dumps enabled → %s", crash_path)
    except Exception as e:
        log.warning("Could not enable crash dumps: %s", e)


def main():
    sys.excepthook = _excepthook
    qInstallMessageHandler(_qt_message_handler)
    _enable_crash_dumps()
    log.info("=" * 60)
    log.info("Video Organizer starting  (log: %s)", log_path())
    # Request 1 ms Windows scheduler resolution so QThread.msleep is
    # accurate to ~1 ms.  Without this, native-FPS playback caps at ~88%
    # because msleep rounds up to the default 15.6 ms granularity.
    from hires_timer import request_high_resolution_timer
    request_high_resolution_timer()
    _heal_bad_settings()       # repair any settings stuck at nonsense values
    _apply_decode_settings()   # apply user override BEFORE we log the final values
    opts = os.environ.get('OPENCV_FFMPEG_CAPTURE_OPTIONS', '(default)')
    log.info("FFmpeg capture options: %s", opts)
    log.info("Decode parallelism:     %s", _expected_decode_threads(opts))
    log.info("Logical CPU cores:      %d", os.cpu_count() or 0)
    log.info("Note: slice-type parallelism is also bounded by slices/frame "
             "in the encoded stream (typically 1-8 for H.264/H.265).")
    _log_simd_status()


def _log_simd_status():
    """Log the SIMD instruction sets OpenCV was compiled with AND which of
    those the local CPU actually supports.  Both OpenCV and the bundled
    FFmpeg use runtime dispatch via CPUID, so what's compiled-in AND
    supported is what gets executed."""
    try:
        import cv2
        # Stable IDs from OpenCV's cvdef.h
        # 1=SSE, 2=SSE2, 4=SSSE3, 5=SSE4.1, 7=SSE4.2,
        # 10=AVX, 11=AVX2, 12=FMA3, 13=AVX-512F, 14=AVX-512BW
        wanted = [(10, "AVX"), (11, "AVX2"), (12, "FMA3"),
                  (13, "AVX-512F"), (14, "AVX-512BW")]
        available = []
        for code, name in wanted:
            try:
                if cv2.checkHardwareSupport(code):
                    available.append(name)
            except Exception:
                pass
        compiled_in = cv2.getCPUFeaturesLine()
        log.info("OpenCV SIMD compiled-in:  %s", compiled_in)
        log.info("OpenCV SIMD active CPU:   %s",
                 ", ".join(available) if available else "(none of AVX/AVX2/FMA3/AVX-512)")
        log.info("FFmpeg H.264/H.265 decode uses AVX2 SIMD via runtime "
                 "dispatch (av_get_cpu_flags) when AVX2 is in the active set.")
    except Exception as e:
        log.warning("Could not probe SIMD status: %s", e)

    # High-DPI support
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("Video Organizer")
    app.setOrganizationName("VideoOrganizer")
    app.setApplicationVersion("1.0.0")

    # Dark palette fallback (in case stylesheet misses something)
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(30, 30, 30))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(221, 221, 221))
    palette.setColor(QPalette.ColorRole.Base, QColor(37, 37, 37))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(45, 45, 45))
    palette.setColor(QPalette.ColorRole.Text, QColor(221, 221, 221))
    palette.setColor(QPalette.ColorRole.Button, QColor(46, 46, 46))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(221, 221, 221))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(58, 111, 196))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)

    try:
        window = MainWindow()
        window.show()
        log.info("MainWindow shown")
    except Exception:
        log.critical("Failed to create MainWindow", exc_info=True)
        raise

    # Final safety net for clean shutdown: whatever triggers the quit (window
    # close, a Quit action, app.quit()), drain every registered QThread before
    # the interpreter tears their Python wrappers down. Otherwise a still-
    # running thread (e.g. a hover preview the user never moved away from) is
    # destroyed mid-run → "QThread: Destroyed while thread is still running",
    # which aborts the process. MainWindow.closeEvent already does this for the
    # normal window-close path; aboutToQuit also covers paths that bypass it.
    def _drain_threads_on_quit():
        try:
            from pyav_play_thread import PREVIEW_MANAGER
            PREVIEW_MANAGER.cancel_all()
        except Exception:
            pass
        # Best-effort cache flush in case this quit path bypassed closeEvent
        # (e.g. app.quit() from a menu) — closeEvent already flushed+closed it on
        # the normal window-close path, so this is a guarded no-op there.
        try:
            win = window
            if win is not None and getattr(win, '_cache', None) is not None:
                win._cache.flush_writes(timeout=2.0)
                win._cache.close()
        except Exception:
            pass
        stuck = 0
        try:
            import qthread_registry
            stuck = qthread_registry.wait_all(3000)
        except Exception:
            log.warning("aboutToQuit thread drain failed", exc_info=True)
        # ALWAYS hard-exit (see MainWindow.closeEvent for the full rationale):
        # even with 0 stuck, a QThread finishing in its finish() window — or a
        # timer arming a new preview — can be destroyed mid-run during Qt's
        # teardown and abort with "QThread: Destroyed while thread is still
        # running". State is already persisted, so skip the C++ destructors.
        # (On the normal window-close path closeEvent already os._exit'd before
        # this runs; this covers app.quit()/menu-quit paths that bypass it.)
        if stuck:
            log.warning("%d background thread(s) still running at quit — hard exit",
                        stuck)
        try:
            if window is not None and getattr(window, '_settings', None) is not None:
                window._settings.sync()
        except Exception:
            pass
        try:
            import logging
            logging.shutdown()
        except Exception:
            pass
        os._exit(0)
    app.aboutToQuit.connect(_drain_threads_on_quit)

    code = app.exec()
    log.info("Application exiting with code %d", code)
    sys.exit(code)


if __name__ == "__main__":
    main()
