"""
Individual video thumbnail widget.

Performance notes
─────────────────
• _hover_delay (500 ms): video capture is NOT opened until the mouse rests
  on the thumbnail for 500 ms, preventing I/O churn during scroll.

• _VideoPlayThread: frame decode runs on a background QThread.
  For large VR videos (4K–8K H.264/H.265) cv2.VideoCapture.read() can take
  50–200 ms per frame — far longer than one 30 fps frame interval (33 ms).
  Moving it off the main thread keeps the UI smooth regardless of resolution.

• Frame pre-scaling: the thread scales each decoded frame DOWN to the display
  label size with cv2.INTER_LINEAR immediately after decode, before the
  BGR→RGB copy.  For 8K VR (7680×3840 → ~400×225) this shrinks the copy
  from ~88 MB to ~270 KB — roughly a 325× reduction before any Qt work.

• _display_pixmap cache: the static source pixmap (_full_pixmap) is scaled
  to the exact label size once per size change (SmoothTransformation) and
  cached.  Subsequent paints blit the pre-scaled pixmap — no scaling at all.

Layout (manual geometry via resizeEvent):
  ┌──────────────────────────────┐
  │                              │ [☑] ← checkbox overlay
  │         image / video        │
  │                              │
  ├──────────────────────────────┤
  │  ○━━━━━━━━━━━━━━━━━━━━━━━━━  │  ← seek slider
  ├──────────────────────────────┤
  │  filename.mp4    0:05 / 1:23 │  ← name + time label
  ├──────────────────────────────┤
  │  ★★★☆☆                       │  ← star rating
  └──────────────────────────────┘
"""

import os
import time as _time

import cv2
import numpy as np

import vr_unwarp as vu
from vr_frame import for_path as _make_frame_unwarper
from app_logger import get_logger
log = get_logger(__name__)

from PyQt6.QtWidgets import (
    QFrame, QLabel, QCheckBox, QSlider, QMenu, QDialog,
    QVBoxLayout, QHBoxLayout, QDoubleSpinBox, QDialogButtonBox, QWidget,
)
from metadata_dialog import MetadataDialog
from PyQt6.QtCore import (
    Qt, QThread, QTimer, QSize, pyqtSignal, QPoint, QObject, pyqtSlot,
    QEvent, QRect,
)
from PyQt6.QtGui import QPixmap, QImage, QFont, QPainter, QColor, QCursor


class _NoWheelSlider(QSlider):
    """A QSlider that IGNORES the mouse wheel.

    Seeking is intended to be deliberate (click / drag the handle). The
    default QSlider nudges its value on every wheel tick, which made scrolling
    the thumbnail grid accidentally re-seek whatever card was under the
    cursor. Ignoring the wheel lets the event bubble up to the grid's scroll
    area so the wheel just scrolls the list, as expected."""

    def wheelEvent(self, event):   # noqa: N802 (Qt naming)
        event.ignore()


# ── geometry constants (exported so the grid can compute heights) ─────────────
_SLIDER_H = 32   # taller slider row → bigger click target for the seek bar
_NAME_H   = 24
_RATING_H = 16
BOTTOM_H  = _SLIDER_H + _NAME_H + _RATING_H   # 62 px  (public export)

_CB_SIZE   = 24
_CB_MARGIN = 5

_HOVER_DELAY_MS   = 500   # ms of dwell before video playback starts
_SEEK_DEBOUNCE_MS = 400   # ms after last slider move before regenerating thumbnail

# ── Global play-thread registry ───────────────────────────────────────────────
# Holds a Python reference to every _VideoPlayThread that is still running.
# Without this, the reference count can drop to zero while the C++ thread is
# still inside cv2.VideoCapture.read(), causing Qt to destroy the QThread and
# crash with "QThread: Destroyed while thread '' is still running".
# Removal is handled on the GUI thread by _play_thread_reaper (below) — NOT by
# a bare lambda connected to `finished`, which would run as a DirectConnection
# *inside* the worker's QThreadPrivate::finish() (holding the QThread mutex and
# needing the GIL) and could deadlock the GUI thread's deleteLater→~QThread::
# wait(). See qthread_registry for the full analysis of that GIL ⊗ mutex
# deadlock — it is what froze the app during heavy big-file browsing.
_running_play_threads: set = set()


class _PlayThreadReaper(QObject):
    """Removes finished play threads from `_running_play_threads`, always on
    the GUI thread (queued), so cleanup never runs inside the worker's
    finish() critical section. C++ object deletion is handled separately and
    safely by qthread_registry.install() (called in every play thread's
    __init__)."""

    @pyqtSlot()
    def reap(self) -> None:
        for t in list(_running_play_threads):
            # Only drop threads that RAN and FINISHED — not ones merely "not
            # running". A preview thread handed to PREVIEW_MANAGER may be built
            # but not yet started (its start is deferred until the current
            # preview exits); isRunning() is False for it, but it will start
            # later. Dropping it here (and letting the registry deleteLater it)
            # is the use-after-free that aborts with "QThread: Destroyed while
            # thread is still running". isFinished() is True only post-run().
            try:
                done = t.isFinished()
            except RuntimeError:
                done = True   # C++ object already gone
            if done:
                _running_play_threads.discard(t)


# Created on the importing (GUI) thread, so its slot runs there.
_play_thread_reaper = _PlayThreadReaper()


def _fmt_time(secs: float) -> str:
    s = int(max(0.0, secs))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


# ── background video playback thread ──────────────────────────────────────────
class _VideoPlayThread(QThread):
    """
    Decodes video frames on a background thread and emits QImages at native FPS.

    • cap.read() is blocking and can take 50-200 ms for large VR files.
      Running it here keeps the main thread free to handle UI events.
    • Each frame is scaled to the display size with cv2.INTER_LINEAR immediately
      after decode — this is the cheapest place to downscale.
    • Timing: sleeps (frame_ms – decode_ms) per iteration to match native FPS.
      If decode takes longer than one frame interval, the next frame reads
      immediately (no artificial slowdown; natural catch-up).
    • Loops from start_sec when EOF is reached.
    • Responds to request_seek() and set_display_size() from the main thread
      (plain attribute writes — safe under CPython's GIL).
    """
    frame_ready = pyqtSignal(object)   # QImage (thread-safe)

    def __init__(self, video_path: str, start_sec: float,
                 display_w: int, display_h: int,
                 *,
                 hw_accel: bool = True,
                 target_fps: float = 0.0):
        """target_fps=0 means "use the file's native FPS".  Setting it to a
        lower value (e.g. 8) skips frames on decode, which is how we keep
        hover preview responsive for large VR files on HDD."""
        super().__init__()             # no Qt parent — lifecycle via finished→deleteLater
        self._path       = video_path
        self._start_sec  = start_sec
        self._disp_w     = display_w
        self._disp_h     = display_h
        self._stop_flag  = False
        self._seek_to: float | None = None   # written by main thread; GIL-safe
        self._hw_accel   = hw_accel
        self._target_fps = float(target_fps)
        from qthread_registry import install
        install(self)

    def set_display_size(self, w: int, h: int):
        """Update target display size (main thread → thread; GIL-safe for ints)."""
        self._disp_w = w
        self._disp_h = h

    def request_seek(self, sec: float):
        """Ask the thread to seek to sec on its next frame iteration (GIL-safe)."""
        self._seek_to = sec

    def stop(self):
        """Signal the thread to exit on its next frame boundary."""
        self._stop_flag = True

    def run(self):
        from video_capture_helper import open_capture
        cap = open_capture(self._path, hw_accel=self._hw_accel)
        if cap is None:
            return

        native_fps = cap.get(cv2.CAP_PROP_FPS)
        if not (0 < native_fps < 300):
            native_fps = 25.0
        effective_fps = self._target_fps if self._target_fps > 0 else native_fps
        frame_interval = 1.0 / effective_fps   # seconds per frame

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total > 0 and native_fps > 0:
            start_f = max(0, min(int(self._start_sec * native_fps), total - 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

        # ── Adaptive output resolution ────────────────────────────────────
        # If we fall behind native FPS, shrink the emit buffer to reduce
        # per-frame reformat cost.  Smaller output → faster cvtColor and
        # tobytes() copy → more headroom for decode.
        scale_factor    = 1.0   # 1.0 = full requested display size
        min_scale       = 0.5   # never shrink past 50% of display size
        frames_emitted  = 0
        recent_frames   = []    # rolling perf_counter timestamps of recent emits

        # ── Absolute-deadline pacing ──────────────────────────────────────
        # Instead of computing sleep_ms from the previous frame's elapsed
        # time (which drifts when sleeps over-run), schedule each frame
        # against an absolute deadline.  Drift gets corrected automatically:
        # if frame N took too long, frame N+1's deadline is still on time.
        next_deadline = _time.perf_counter() + frame_interval

        try:
            while not self._stop_flag:
                # Apply any pending seek requested from the main thread
                if self._seek_to is not None:
                    seek_sec      = self._seek_to
                    self._seek_to = None
                    if total > 0 and native_fps > 0:
                        cap.set(cv2.CAP_PROP_POS_FRAMES,
                                max(0, min(int(seek_sec * native_fps), total - 1)))
                    next_deadline = _time.perf_counter() + frame_interval

                ret, frame = cap.read()
                if not ret:
                    # EOF — loop from start_sec
                    if total > 0 and native_fps > 0:
                        cap.set(cv2.CAP_PROP_POS_FRAMES,
                                max(0, int(self._start_sec * native_fps)))
                    ret, frame = cap.read()
                    if not ret:
                        break

                # Adaptive scale: shrink display target if behind on FPS
                dw = int(self._disp_w * scale_factor)
                dh = int(self._disp_h * scale_factor)
                if dw > 0 and dh > 0:
                    fh, fw = frame.shape[:2]
                    if fw > dw or fh > dh:
                        s   = min(dw / fw, dh / fh)
                        nw  = max(1, int(fw * s))
                        nh  = max(1, int(fh * s))
                        frame = cv2.resize(frame, (nw, nh),
                                           interpolation=cv2.INTER_LINEAR)

                rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rh, rw, ch = rgb.shape
                raw  = rgb.tobytes()
                qimg = QImage(raw, rw, rh, ch * rw,
                              QImage.Format.Format_RGB888).copy()

                if not self._stop_flag:
                    self.frame_ready.emit(qimg)
                    frames_emitted += 1
                    now = _time.perf_counter()
                    recent_frames.append(now)
                    # Keep ~1 second of history
                    while recent_frames and (now - recent_frames[0]) > 1.0:
                        recent_frames.pop(0)

                # ── Dynamic resource allocation ───────────────────────────
                # After ~30 emitted frames (≈1 sec at 24 fps), check if we
                # held native FPS. If we're below 95% of target, shrink the
                # output resolution by 10% (down to min_scale). If we're
                # comfortably at native, restore size by 10%.
                if frames_emitted >= 30 and len(recent_frames) >= 2:
                    span = recent_frames[-1] - recent_frames[0]
                    observed = (len(recent_frames) - 1) / span if span > 0 else 0
                    target   = 1.0 / frame_interval
                    if observed < target * 0.95 and scale_factor > min_scale:
                        scale_factor = max(min_scale, scale_factor - 0.10)
                    elif observed > target * 0.99 and scale_factor < 1.0:
                        scale_factor = min(1.0, scale_factor + 0.05)
                    frames_emitted = 0   # reset window

                # Absolute-deadline sleep: hits native FPS even when individual
                # frames overrun, because the next deadline doesn't shift.
                now = _time.perf_counter()
                remaining = next_deadline - now
                if remaining > 0.002:
                    # Use msleep for the bulk of the wait (cheap), then
                    # busy-wait the last 1 ms for accuracy. Requires
                    # timeBeginPeriod(1) at app startup or this still
                    # rounds up to ~15 ms on Windows.
                    sleep_ms = int(remaining * 1000) - 1
                    if sleep_ms > 0:
                        self.msleep(sleep_ms)
                    while _time.perf_counter() < next_deadline:
                        pass
                # Advance deadline by exactly one frame interval, regardless
                # of overrun, so drift is corrected naturally.
                next_deadline += frame_interval
                # If we've fallen MULTIPLE frames behind, snap forward so we
                # don't try to "catch up" by decoding back-to-back forever.
                catchup = _time.perf_counter() - next_deadline
                if catchup > frame_interval * 3:
                    next_deadline = _time.perf_counter() + frame_interval
        finally:
            cap.release()


# ── seek dialog ───────────────────────────────────────────────────────────────
class _SeekDialog(QDialog):
    def __init__(self, current: float, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Set thumbnail time")
        self.setFixedSize(290, 105)
        layout = QVBoxLayout(self)
        row = QHBoxLayout()
        row.addWidget(QLabel("Seek time (seconds):"))
        self._spin = QDoubleSpinBox()
        self._spin.setRange(0.0, 999999.0)
        self._spin.setDecimals(2)
        self._spin.setSingleStep(1.0)
        self._spin.setValue(current)
        row.addWidget(self._spin)
        layout.addLayout(row)
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    @property
    def seek_time(self) -> float:
        return self._spin.value()


# ── star rating widget ─────────────────────────────────────────────────────────
class _StarsWidget(QWidget):
    """Displays 5 clickable star characters for rating (0–5)."""
    rating_changed = pyqtSignal(int)

    _FILLED = '★'
    _EMPTY  = '☆'

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rating = 0
        font = QFont()
        font.setPointSize(9)
        self.setFont(font)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(_RATING_H)

    def set_rating(self, r: int):
        self._rating = max(0, min(5, r))
        self.update()

    def get_rating(self) -> int:
        return self._rating

    def paintEvent(self, event):
        """
        Draw 5 stars evenly spread across the full widget width.
        Each star is centred inside its own equal-sized zone (width / 5),
        so the visual position and the click-detection zone always match.
        """
        painter = QPainter(self)
        painter.setFont(self.font())
        zone_w   = max(1, self.width() / 5)
        filled_c = QColor('#f0c040')
        empty_c  = QColor('#555')
        for i in range(5):
            painter.setPen(filled_c if i < self._rating else empty_c)
            # drawText into each zone rect so Qt centres the glyph for us
            painter.drawText(
                int(i * zone_w), 0,
                int(zone_w), self.height(),
                Qt.AlignmentFlag.AlignCenter,
                self._FILLED if i < self._rating else self._EMPTY,
            )
        painter.end()

    def mousePressEvent(self, event):
        # Defense-in-depth: any failure inside the click handler must not
        # crash the app.  This handler triggers a signal chain that writes
        # to SQLite and can race with cv2/PyAV signals from the parent
        # widget's play thread — if anything raises here, we want a log
        # entry, not a process crash.
        try:
            if event.button() == Qt.MouseButton.LeftButton:
                w = max(1.0, float(self.width()))
                zone_w = max(1.0, w / 5.0)
                # Which of the 5 equal zones did the click land in?
                x = float(event.position().x())
                clicked = int(x / zone_w) + 1
                clicked = max(1, min(5, clicked))
                # Clicking the active star again clears the rating
                if clicked == self._rating:
                    clicked = 0
                self._rating = clicked
                self.update()
                self.rating_changed.emit(int(self._rating))
        except Exception:
            import logging
            logging.getLogger(__name__).exception(
                "_StarsWidget click handler raised — swallowed to prevent crash"
            )
        try:
            event.accept()
        except Exception:
            pass


# ── main widget ───────────────────────────────────────────────────────────────
class VideoThumbnailWidget(QFrame):
    checked_changed  = pyqtSignal(str, bool)
    seek_requested   = pyqtSignal(str, float)
    open_requested   = pyqtSignal(str)
    delete_requested = pyqtSignal(str)
    move_requested   = pyqtSignal(str)
    rename_requested = pyqtSignal(str)
    copy_requested   = pyqtSignal(str)
    rating_changed   = pyqtSignal(str, int)
    watch_toggled    = pyqtSignal(str, bool)   # Feature 17

    def __init__(self, video_path: str, seek_time: float, parent=None,
                 *, file_size: int = 0, settings=None):
        """file_size (bytes) and settings (AppSettings) let the widget decide
        whether to attempt hover playback at all, or at what FPS, for large
        VR files where opening cv2 is itself a 1–2 s operation."""
        super().__init__(parent)
        self.video_path    = video_path
        self._seek_time    = seek_time
        # Seek time the static thumbnail currently reflects. Initialise it to
        # the CURRENT seek_time so an incidental hover (e.g. the cursor brushing
        # a card while the user scrolls) does NOT regenerate the thumbnail on
        # hover-end. Without this it defaulted to None → the FIRST hover-end on
        # every card saw "None != seek_time" → emitted seek_requested → the grid
        # regenerated the thumbnail ("Generating…"). Only a REAL seek (slider
        # drag changing _seek_time) should persist+regenerate on hover-end.
        self._seek_time_at_last_thumb = seek_time
        self._duration     = 0.0
        self._aspect_ratio: float = 16 / 9
        self._file_size    = int(file_size or 0)
        self._settings     = settings   # AppSettings or None

        # ── pixmap cache ─────────────────────────────────────────────────────
        self._full_pixmap:    QPixmap | None = None
        self._display_pixmap: QPixmap | None = None
        self._last_display_size = QSize()

        # ── hover / playback ──────────────────────────────────────────────────
        self._hovering    = False
        self._play_thread: _VideoPlayThread | None = None

        # Delay before starting playback — prevents I/O during scroll
        self._hover_delay = QTimer(self)
        self._hover_delay.setSingleShot(True)
        self._hover_delay.setInterval(_HOVER_DELAY_MS)
        self._hover_delay.timeout.connect(self._start_playback)

        # ── seek debounce ─────────────────────────────────────────────────────
        self._seek_debounce = QTimer(self)
        self._seek_debounce.setSingleShot(True)
        self._seek_debounce.setInterval(_SEEK_DEBOUNCE_MS)
        self._seek_debounce.timeout.connect(self._apply_seek)

        # ── chapter positions (Feature 24) ───────────────────────────────────
        self._chapter_positions: list[float] = []

        # ── Qt setup ──────────────────────────────────────────────────────────
        self.setFrameShape(QFrame.Shape.Box)
        self.setLineWidth(2)   # Feature 3: increased from 1 to 2
        self.setMouseTracking(True)

        sm_font = QFont()
        sm_font.setPointSize(8)

        self._image_label = QLabel(self)
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setStyleSheet("background: #1a1a1a;")
        self._image_label.setMouseTracking(True)
        # Stretch pixmap to fill the label.  When the play loop's adaptive
        # resolution downscales the emit buffer (e.g. 800x450 → 400x225 to
        # save decode time on slow files), the small pixmap is stretched
        # by Qt to fill the label area — the user sees full-size video,
        # not a small video centered with empty borders.  No aspect-ratio
        # distortion because the label is sized to match the video's
        # aspect via VideoThumbnailWidget.required_height(width).
        self._image_label.setScaledContents(True)
        # Hover preview is gated on the mouse being over the THUMBNAIL IMAGE
        # specifically (not the whole card — e.g. the seek bar / name / rating
        # below it don't count). We watch the image label's Enter/Leave via an
        # event filter rather than the widget's own enterEvent/leaveEvent.
        self._image_label.installEventFilter(self)

        # ── Feature 4: loading label with shimmer animation ───────────────────
        self._loading_label = QLabel("Generating…", self)
        self._loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._loading_label.setStyleSheet("color: #888; background: transparent;")
        self._loading_label.setFont(sm_font)
        # CRITICAL: this overlay sits ON TOP of the image label and, while a
        # thumbnail is still generating, covers the centre of the card. If it
        # ate mouse events, the image label's Enter never fired → hover
        # preview & live-seek were DEAD on every "Generating…" card. Make it
        # mouse-transparent so the cursor passes through to the image label.
        self._loading_label.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

        self._shimmer_timer = QTimer(self)
        self._shimmer_timer.setInterval(400)
        self._shimmer_timer.timeout.connect(self._shimmer_step)
        self._shimmer_step_n = 0
        self._shimmer_timer.start()

        # Belt-and-suspenders timeout: if the thumbnail generator goes
        # silent (worker hung, signal lost, file unreadable in a way we
        # didn't anticipate), the shimmer can run forever giving the
        # user no feedback.  After 45 s, give up and show "Cannot read".
        # The user can still right-click to retry.  This is separate
        # from — and longer than — the service-level watchdog (30 s),
        # so under normal conditions the worker's `thumbnail_failed`
        # signal arrives first and pre-empts this fallback.
        self._shimmer_max_timer = QTimer(self)
        self._shimmer_max_timer.setSingleShot(True)
        self._shimmer_max_timer.setInterval(45_000)
        self._shimmer_max_timer.timeout.connect(self._on_shimmer_timeout)
        self._shimmer_max_timer.start()

        self._checkbox = QCheckBox(self)
        # A blue border makes the checkbox clearly visible — it is now the ONLY
        # way to select a card (clicking the thumbnail no longer selects), so it
        # must stand out against any thumbnail.
        self._checkbox.setStyleSheet(
            "QCheckBox::indicator { width: 18px; height: 18px; }"
            "QCheckBox { background: rgba(0,0,0,150); border: 2px solid #5a9fd4;"
            " border-radius: 4px; padding: 1px; }"
        )
        self._checkbox.stateChanged.connect(self._on_check_changed)
        # Feature 3: connect checkbox to frame style update
        self.checked_changed.connect(lambda path, checked: self._update_frame_style())

        self._seek_slider = _NoWheelSlider(Qt.Orientation.Horizontal, self)
        self._seek_slider.setRange(0, 3600)
        self._seek_slider.setValue(int(seek_time))
        # Thick groove + large handle so the seek bar is an easy click target.
        self._seek_slider.setStyleSheet("""
            QSlider { background: transparent; }
            QSlider::groove:horizontal {
                height: 8px; background: #3a3a3a; border-radius: 4px; }
            QSlider::sub-page:horizontal {
                background: #5a9fd4; border-radius: 4px; }
            QSlider::handle:horizontal {
                width: 18px; height: 18px; margin: -5px 0;
                background: #7ab8e8; border-radius: 9px; }
            QSlider::handle:horizontal:hover { background: #a8d4f5; }
        """)
        self._seek_slider.valueChanged.connect(self._on_slider_value_changed)
        self._seek_slider.sliderReleased.connect(self._on_slider_released)

        self._name_label = QLabel(os.path.basename(video_path), self)
        self._name_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        self._name_label.setStyleSheet(
            "color: #ccc; background: #111; padding-left: 4px;"
        )
        self._name_label.setFont(sm_font)
        self._name_label.setToolTip(os.path.basename(video_path))

        self._time_label = QLabel(_fmt_time(seek_time), self)
        self._time_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self._time_label.setStyleSheet(
            "color: #888; background: #111; padding-right: 4px;"
        )
        self._time_label.setFont(sm_font)

        self._stars_widget = _StarsWidget(self)
        self._stars_widget.rating_changed.connect(self._on_star_rating_changed)

        # Children must NOT take keyboard focus. A focused child (e.g. the
        # checkbox right after you click it) makes the enclosing QScrollArea
        # auto-scroll to keep that widget visible — so when you then scroll away,
        # the view is yanked back to it: the "scroll jumps back up" glitch
        # (reproduced: focusing a card's checkbox then scrolling teleported the
        # view by tens of thousands of px). Mouse clicks still work with NoFocus;
        # only the focus that drives QScrollArea.ensureWidgetVisible is removed.
        # The grid itself keeps StrongFocus for keyboard navigation.
        for _w in (self._checkbox, self._seek_slider, self._stars_widget):
            _w.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        # ── Feature 17: watched badge ─────────────────────────────────────────
        self._watched_badge = QLabel("✓", self)
        self._watched_badge.setStyleSheet(
            "background: rgba(60,180,60,200); color: white; "
            "border-radius: 3px; padding: 1px 4px; font-size: 10px; font-weight: bold;"
        )
        self._watched_badge.setFont(sm_font)
        self._watched_badge.adjustSize()
        self._watched_badge.hide()
        self._is_watched = False

        # ── Feature 19: sidecar badges ────────────────────────────────────────
        self._sub_badge = QLabel("CC", self)
        self._sub_badge.setStyleSheet(
            "background: rgba(90,120,200,200); color: white; "
            "border-radius: 3px; padding: 1px 3px; font-size: 9px;"
        )
        self._sub_badge.hide()

        self._nfo_badge = QLabel("NFO", self)
        self._nfo_badge.setStyleSheet(
            "background: rgba(180,120,60,200); color: white; "
            "border-radius: 3px; padding: 1px 3px; font-size: 9px;"
        )
        self._nfo_badge.hide()

        # Decorative overlays must not eat hover events meant for the image
        # label (same reason as _loading_label above). The checkbox is left
        # interactive on purpose — it needs clicks.
        for _badge in (self._watched_badge, self._sub_badge, self._nfo_badge):
            _badge.setAttribute(
                Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)

        # Apply initial frame style (Feature 3)
        self._update_frame_style()

    # ── public API ─────────────────────────────────────────────────────────────
    @property
    def seek_time(self) -> float:
        return self._seek_time

    def _on_shimmer_timeout(self):
        """45 s widget-level fallback: if no thumbnail or failure signal
        arrived by now, give up so the user isn't stuck staring at an
        infinite 'Generating…' shimmer."""
        if self._full_pixmap is None:
            self.set_thumbnail_failed("Timed out — try Set thumbnail time…")

    def set_thumbnail_failed(self, reason: str = ''):
        """Called when the generator gives up on this file. Stops the shimmer
        and displays a 'cannot read' placeholder so the user knows it isn't
        still loading."""
        self._shimmer_timer.stop()
        self._shimmer_max_timer.stop()
        self._loading_label.setText("⚠  Cannot read frame")
        self._loading_label.setStyleSheet(
            "color: #c66; background: rgba(0,0,0,140); "
            "padding: 4px 8px; border-radius: 3px;"
        )
        self._loading_label.setToolTip(reason or "Thumbnail generation failed")
        self._loading_label.show()

    def set_thumbnail(self, image: object, duration: float):
        """Accept QImage (from generator signal) or QPixmap (from item cache)."""
        if isinstance(image, QImage) and not image.isNull():
            pix = QPixmap.fromImage(image)
        elif isinstance(image, QPixmap) and not image.isNull():
            pix = image
        else:
            return

        # VR side-by-side thumbnail → show the flat (one-eye, un-warped) view,
        # matching the in-app player. Detected by the (aspect-preserving)
        # thumbnail shape; done once, cheap on a small image. Any failure falls
        # back to the raw thumbnail so a normal card is never broken.
        if vu.is_sbs_aspect(pix.width(), pix.height()):
            flat = self._unwarp_thumb_pixmap(pix)
            if flat is not None and not flat.isNull():
                pix = flat

        self._full_pixmap = pix
        self._invalidate_display_cache()

        if pix.height() > 0:
            self._aspect_ratio = pix.width() / pix.height()

        if duration > 0:
            self._duration = duration
            self._seek_slider.setRange(0, max(1, int(duration)))

        self._loading_label.hide()
        self._shimmer_timer.stop()       # stop the dot animation
        self._shimmer_max_timer.stop()   # cancel the 45 s fallback
        self._update_time_label()

        if not self._hovering:
            self._show_static_thumbnail()

    def _unwarp_thumb_pixmap(self, pix):
        """Crop to one eye + reproject a side-by-side VR thumbnail to a flat 16:9
        QPixmap (stereographic 220°), the same as the fullscreen player. Returns
        None on any error (caller keeps the raw thumbnail)."""
        try:
            img = pix.toImage().convertToFormat(QImage.Format.Format_RGB888)
            w, h, bpl = img.width(), img.height(), img.bytesPerLine()
            ptr = img.constBits(); ptr.setsize(h * bpl)
            arr = (np.frombuffer(ptr, np.uint8).reshape(h, bpl)[:, :w * 3]
                   .reshape(h, w, 3))
            if getattr(self, '_thumb_unwarper', None) is None:
                self._thumb_unwarper = _make_frame_unwarper(self.video_path, eye='left')
            flat = np.ascontiguousarray(self._thumb_unwarper.apply(arr))
            oh, ow = flat.shape[:2]
            qi = QImage(flat.data, ow, oh, ow * 3, QImage.Format.Format_RGB888).copy()
            return QPixmap.fromImage(qi)
        except Exception as e:
            log.debug("VR thumbnail un-warp failed: %s", e)
            return None

    def set_checked(self, checked: bool):
        self._checkbox.setChecked(checked)
        self._update_frame_style()   # Feature 3

    def is_checked(self) -> bool:
        return self._checkbox.isChecked()

    def set_rating(self, r: int):
        self._stars_widget.set_rating(r)

    def get_rating(self) -> int:
        return self._stars_widget.get_rating()

    def required_height(self, width: int) -> int:
        return max(40, round(width / self._aspect_ratio)) + BOTTOM_H

    def cleanup(self):
        """Stop all timers, playback thread, and release pixmap memory.
        Non-blocking: the play thread is signalled to stop but not waited on
        (safe for normal scroll-out recycling on the main thread)."""
        self._hover_delay.stop()
        self._seek_debounce.stop()
        self._shimmer_timer.stop()
        self._shimmer_max_timer.stop()
        self._stop_playback()
        # Drop pixmap references immediately so Qt can release the backing store
        # before Python GC gets around to collecting this widget.
        self._full_pixmap    = None
        self._display_pixmap = None
        self._image_label.clear()

    def wait_for_shutdown(self, timeout_ms: int = 2000):
        """Full shutdown for app exit: stop timers + WAIT for play thread to exit.
        Blocks for up to timeout_ms ms to prevent 'QThread destroyed while running'."""
        self._hover_delay.stop()
        self._seek_debounce.stop()
        self._shimmer_timer.stop()
        self._shimmer_max_timer.stop()
        self._wait_for_playback(timeout_ms)
        self._full_pixmap    = None
        self._display_pixmap = None
        self._image_label.clear()

    def pause_playback_and_wait(self, timeout_ms: int = 1500):
        """Stop hover playback and BLOCK until the cv2.VideoCapture handle is
        released. Used before file move/copy/delete so the OS doesn't refuse
        with a sharing violation. Visual state (pixmap, label) is preserved."""
        self._hover_delay.stop()
        self._wait_for_playback(timeout_ms)
        # Restore the static thumbnail so the user sees something sensible
        if self._full_pixmap is not None:
            self._show_static_thumbnail()

    def _wait_for_playback(self, timeout_ms: int):
        """Internal: stop the play thread and wait for it to exit.

        Every Qt call here is guarded: by the time this runs (app shutdown,
        or pre-move handle release), the play thread's C++ object may already
        have been deleteLater'd, in which case any method call raises
        RuntimeError ('wrapped C/C++ object ... has been deleted'). An
        unguarded raise here propagates into Qt's teardown and crashes the
        process with an access violation (observed at shutdown)."""
        if self._play_thread is None:
            return
        thread            = self._play_thread
        self._play_thread = None
        try:
            thread.frame_ready.disconnect(self._on_play_frame)
        except (RuntimeError, TypeError):
            pass
        try:
            thread.stop()
        except (RuntimeError, AttributeError):
            return   # C++ object gone — nothing to wait on
        try:
            thread.wait(timeout_ms)   # block until run() returns (or timeout)
        except (RuntimeError, AttributeError):
            pass

    # ── Feature 3: selection highlight ring ───────────────────────────────────
    def _update_frame_style(self):
        # No blue selection ring around the whole card — selection is shown by
        # the checkbox (which has its own blue border). Keep a neutral border
        # whether or not the item is checked.
        self.setStyleSheet("QFrame { border: 1px solid #3a3a3a; border-radius: 2px; }")

    # ── Feature 4: shimmer animation ──────────────────────────────────────────
    def _shimmer_step(self):
        dots = ['', '·', '··', '···'][self._shimmer_step_n % 4]
        self._loading_label.setText(f"Generating{dots}")
        self._shimmer_step_n += 1

    # ── Feature 17: watch history badge ───────────────────────────────────────
    def set_watched(self, watched: bool):
        self._is_watched = watched
        if watched:
            self._watched_badge.show()
            self._watched_badge.raise_()
        else:
            self._watched_badge.hide()

    def is_watched(self) -> bool:
        return self._is_watched

    # ── Feature 19: sidecar file badges ───────────────────────────────────────
    def set_sidecar_files(self, has_sub: bool, has_nfo: bool):
        self._sub_badge.setVisible(has_sub)
        self._nfo_badge.setVisible(has_nfo)

    # ── Feature 22: focus ring ─────────────────────────────────────────────────
    def set_focused(self, focused: bool):
        if focused:
            if self._checkbox.isChecked():
                self.setStyleSheet("QFrame { border: 2px solid #7ab8e8; border-radius: 2px; }")
            else:
                self.setStyleSheet("QFrame { border: 2px solid #5a9fd4; border-radius: 2px; }")
        else:
            self._update_frame_style()

    # ── Feature 24: chapter/scene marks ───────────────────────────────────────
    def set_chapter_positions(self, positions: list[float]):
        """Set chapter/scene change positions (in seconds) to display on seek slider."""
        self._chapter_positions = positions
        if positions and self._duration > 0:
            tip = "Scenes: " + ", ".join(f"{_fmt_time(p)}" for p in positions[:10])
            self._seek_slider.setToolTip(tip)

    # ── layout ─────────────────────────────────────────────────────────────────
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._relayout()

    def _relayout(self):
        w, h     = self.width(), self.height()
        image_h  = max(1, h - BOTTOM_H)
        slider_y = image_h
        name_y   = image_h + _SLIDER_H
        rating_y = image_h + _SLIDER_H + _NAME_H

        self._image_label.setGeometry(0, 0, w, image_h)
        self._loading_label.setGeometry(0, 0, w, image_h)
        self._seek_slider.setGeometry(4, slider_y + 3, w - 8, _SLIDER_H - 6)

        name_w = max(10, int(w * 0.65))
        self._name_label.setGeometry(0,      name_y, name_w,     _NAME_H)
        self._time_label.setGeometry(name_w, name_y, w - name_w, _NAME_H)

        self._stars_widget.setGeometry(4, rating_y, w - 8, _RATING_H)

        self._checkbox.setGeometry(
            w - _CB_SIZE - _CB_MARGIN, _CB_MARGIN, _CB_SIZE, _CB_SIZE
        )
        self._checkbox.raise_()

        # Feature 17: watched badge — bottom-left of image area
        self._watched_badge.setGeometry(4, image_h - 20, 24, 16)
        if self._is_watched:
            self._watched_badge.raise_()

        # Feature 19: sidecar badges — top-left of image area
        self._sub_badge.setGeometry(4, 4, 24, 14)
        self._nfo_badge.setGeometry(32, 4, 30, 14)
        self._sub_badge.raise_()
        self._nfo_badge.raise_()

        # If label size changed, invalidate display cache and update thread size
        new_size = self._image_label.size()
        if new_size != self._last_display_size:
            self._invalidate_display_cache()
            if self._play_thread is not None:
                self._play_thread.set_display_size(new_size.width(), new_size.height())

        if self._full_pixmap and not self._hovering:
            self._show_static_thumbnail()

    # ── pixmap display helpers ─────────────────────────────────────────────────
    def _invalidate_display_cache(self):
        """Clear both halves of the scaled-pixmap cache consistently."""
        self._display_pixmap    = None
        self._last_display_size = QSize()

    def _show_static_thumbnail(self):
        if not self._full_pixmap:
            self._image_label.clear()
            return
        size = self._image_label.size()
        if size.isEmpty():
            return
        if self._display_pixmap is None or size != self._last_display_size:
            self._display_pixmap = self._full_pixmap.scaled(
                size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self._last_display_size = size
        self._image_label.setPixmap(self._display_pixmap)

    def _update_time_label(self):
        cur = _fmt_time(self._seek_time)
        if self._duration > 0:
            self._time_label.setText(f"{cur} / {_fmt_time(self._duration)}")
        else:
            self._time_label.setText(cur)

    # ── hover / playback ───────────────────────────────────────────────────────
    def eventFilter(self, obj, event):
        # Hover preview is driven by the IMAGE LABEL's Enter/Leave so it only
        # runs while the cursor is over the thumbnail image — NOT the seek bar,
        # name or rating below it (per the user's request).
        if obj is self._image_label:
            et = event.type()
            if et == QEvent.Type.Enter:
                self._begin_thumb_hover()
            elif et == QEvent.Type.Leave:
                # The checkbox / badges sit ON TOP of the image; moving onto
                # one fires Leave on the label even though the cursor is still
                # over the thumbnail. Only end hover when truly off the image.
                if not self._cursor_over_image():
                    self._end_thumb_hover()
        return super().eventFilter(obj, event)

    def _cursor_over_image(self) -> bool:
        """True if the global mouse cursor is within the image label's rect."""
        lbl = self._image_label
        try:
            top_left = lbl.mapToGlobal(lbl.rect().topLeft())
            return QRect(top_left, lbl.rect().size()).contains(QCursor.pos())
        except RuntimeError:
            return False

    def enterEvent(self, event):
        # Hover is started by the image-label event filter, NOT here — entering
        # the card over the seek bar must not start a preview.
        super().enterEvent(event)

    def leaveEvent(self, event):
        # Safety net: leaving the whole card (e.g. exiting from over the
        # checkbox overlay, where the label's Leave doesn't fire) must still
        # tear down any preview.
        super().leaveEvent(event)
        self._end_thumb_hover()

    def _begin_thumb_hover(self):
        """Cursor entered the thumbnail image → arm the 500 ms hover preview."""
        if self._hovering:
            return
        self._hovering = True
        self._hover_delay.start()

    def _end_thumb_hover(self):
        """Cursor left the thumbnail image (or the card) → stop the preview,
        persist any seek made during hover, restore the static thumbnail."""
        if not self._hovering:
            return
        self._hovering = False
        self._hover_delay.stop()
        self._stop_playback()
        # If the user moved the slider while hovering, _apply_seek deferred the
        # static thumbnail regen to avoid competing with playback. Now that
        # hover is done, persist the new seek time as the card image.
        if getattr(self, '_seek_time_at_last_thumb', None) != self._seek_time:
            self._seek_time_at_last_thumb = self._seek_time
            self.seek_requested.emit(self.video_path, self._seek_time)
        self._show_static_thumbnail()
        self._image_label.setStyleSheet("background: #1a1a1a;")  # reset border

    def _start_playback(self):
        """Called by _hover_delay after 500 ms of dwell.

        Hover preview runs at the video's NATIVE FPS for all files, at
        all sizes.  Uses PyAV with FRAME threading when available — this
        is dramatically faster than slice threading for HEVC on multi-
        core CPUs (8K 60fps HEVC: 73 fps with FRAME vs 19 fps with SLICE
        on a 24-thread CPU).  Falls back to cv2 if PyAV isn't installed.

        Opening on a multi-GB MKV on HDD takes 1–2 s (the demuxer has to
        read the seek index at end-of-file).  During that delay the
        static thumbnail stays visible and the image label border lights
        up cyan so the user knows hover IS active.
        """
        if not self._hovering or not os.path.exists(self.video_path):
            return

        # ── Reuse the live preview for a same-video re-hover/seek ──────────
        # If a preview thread for THIS video is already running (e.g. the user
        # nudged the seek slider, or hovered out and straight back in), do NOT
        # tear it down and build a new one. Rebuilding serialises through
        # PREVIEW_MANAGER — the dying thread must fully exit before the new one
        # can start — which drops every frame for seconds AND churns the
        # thread-teardown path. Instead just re-seek the live thread; it keeps
        # decoding and emitting at the new position with no gap. (request_seek
        # is a GIL-safe attribute set; the decode loop honours it next frame.)
        existing = self._play_thread
        if existing is not None:
            try:
                reusable = (existing.isRunning()
                            and getattr(existing, '_path', None) == self.video_path
                            and not getattr(existing, '_stop_flag', True))
            except RuntimeError:
                reusable = False   # C++ object gone — fall through to rebuild
            if reusable:
                try:
                    existing.request_seek(self._seek_time)
                except (RuntimeError, AttributeError):
                    pass
                else:
                    # Keep the cyan "hover active" border; nothing else to do.
                    self._image_label.setStyleSheet(
                        "background: #1a1a1a; border: 2px solid #5a9fd4;")
                    return

        # NOTE: we no longer manually stop the previous/other play threads
        # here. The global PREVIEW_MANAGER.submit() (below) enforces the
        # "≤1 preview thread alive" invariant: it stops whatever is currently
        # running and defers OUR start until that thread has fully exited.
        # This is what prevents two concurrent PyAV pyio decodes (the GIL
        # deadlock that wedged the UI on big-file browsing).

        use_hw_accel = True
        if self._settings is not None:
            use_hw_accel = self._settings.use_hw_accel

        target_fps = 0.0   # native FPS

        lbl = self._image_label

        # Prefer PyAV with FRAME threading when available.  Huge win for
        # HEVC: 4× more throughput than cv2 / slice threading on a
        # 24-thread CPU (benchmarked 73 fps on 8K 60fps HEVC).
        #
        # IMPORTANT: hw_accel=False for hover.  PyAV's D3D11VA path keeps
        # frames in GPU memory; converting to RGB24 via reformat() does
        # a GPU→CPU copy that costs more than the decode itself.
        # Pure multi-threaded SW decode + libswscale beats HW for our
        # hover-preview use case (small output, frequent reformat).
        thread = None
        try:
            from pyav_play_thread import _PyAVPlayThread, is_available as _pyav_avail
            if _pyav_avail():
                thread = _PyAVPlayThread(
                    self.video_path, self._seek_time,
                    lbl.width(), lbl.height(),
                    hw_accel=False,          # SW + FRAME threading is faster here
                    target_fps=target_fps,
                    thread_type='FRAME',
                )
        except Exception:
            thread = None

        if thread is None:
            thread = _VideoPlayThread(
                self.video_path, self._seek_time,
                lbl.width(), lbl.height(),
                hw_accel=use_hw_accel,
                target_fps=target_fps,
            )

        _running_play_threads.add(thread)
        # Remove from the registry on the GUI thread (QUEUED), never via a
        # Direct-connected lambda inside finish() — see _PlayThreadReaper and
        # qthread_registry for the GIL ⊗ QThread-mutex deadlock that caused.
        # C++ deletion (deleteLater) is handled by qthread_registry.install()
        # in the thread's __init__, so we deliberately do NOT connect it again
        # here (a second deleteLater + the dying-thread race is what wedged).
        thread.finished.connect(
            _play_thread_reaper.reap, Qt.ConnectionType.UniqueConnection)
        thread.frame_ready.connect(self._on_play_frame)
        self._play_thread = thread
        # Do NOT start the thread directly. Hand it to the global preview
        # manager, which guarantees at most ONE preview thread is ALIVE at
        # a time across all widgets: if another preview is still running
        # (e.g. the one we're switching away from, or this widget's own
        # previous preview after a seek), the manager stops it and starts
        # ours only once it has fully exited. This prevents two concurrent
        # PyAV pyio decodes (which deadlock on the GIL) — the root cause of
        # the multi-GB-file browse wedge.
        from pyav_play_thread import PREVIEW_MANAGER
        PREVIEW_MANAGER.submit(thread)
        # Cyan border: hover is active. If a 1-2 s open is in progress for
        # a huge file, this border is the only visual feedback until the
        # first decoded frame arrives.
        self._image_label.setStyleSheet("background: #1a1a1a; border: 2px solid #5a9fd4;")

    def _stop_playback(self):
        """Non-blocking stop: route through the global preview manager so the
        thread is stopped exactly once and never two run concurrently."""
        if self._play_thread is not None:
            thread            = self._play_thread
            self._play_thread = None
            # Disconnect BEFORE stop so no stale frame signals reach us.
            try:
                thread.frame_ready.disconnect(self._on_play_frame)
            except (RuntimeError, TypeError):
                pass
            # Ask the manager to stop it (only acts if it's the current/
            # pending preview). Non-blocking; the thread self-destructs when
            # its run() returns. DO NOT call thread.wait() here (GUI thread).
            try:
                from pyav_play_thread import PREVIEW_MANAGER
                PREVIEW_MANAGER.cancel(thread)
            except Exception:
                try:
                    thread.stop()
                except (RuntimeError, AttributeError):
                    pass
        # Feature 8: reset image label border after stopping
        self._image_label.setStyleSheet("background: #1a1a1a;")

    def _on_play_frame(self, qimage: object):
        """Receive a decoded, pre-scaled frame from _VideoPlayThread."""
        if isinstance(qimage, QImage) and not qimage.isNull():
            self._image_label.setPixmap(QPixmap.fromImage(qimage))

    # ── seek slider ────────────────────────────────────────────────────────────
    def _on_slider_value_changed(self, value: int):
        self._seek_time = float(value)
        self._update_time_label()
        # Forward seek to background thread if currently playing
        if self._play_thread is not None:
            self._play_thread.request_seek(self._seek_time)
        self._seek_debounce.start()

    def _on_slider_released(self):
        self._seek_debounce.stop()
        self._apply_seek()

    def _apply_seek(self):
        """Commit the current seek position.

        Two cases:
          - If user is HOVERING (actively previewing): just update the
            play_thread's seek position. DO NOT trigger a static thumbnail
            regen — that would spawn a thumbnail worker which competes for
            file I/O with the hover playback. Multi-GB MKV reads on HDD
            cannot interleave; the user-reported "freeze" was thumbnail
            workers and play_threads thrashing the disk simultaneously.
          - If user is NOT hovering (slider released after navigating
            away): trigger the static thumbnail regen to persist the new
            seek time as the card's still image.
        """
        self._seek_debounce.stop()
        if self._hovering:
            # Live re-seek: tell the play_thread to seek; preserve playback.
            if self._play_thread is not None:
                try:
                    self._play_thread.request_seek(self._seek_time)
                except (RuntimeError, AttributeError):
                    pass
            # No thumbnail regen, no clearing of static thumbnail — the
            # play_thread keeps emitting frames at the new position.
            return
        # Not hovering: stop any leftover playback, regen static thumbnail.
        if self._play_thread is not None:
            self._stop_playback()
        self._loading_label.show()
        self._loading_label.setText("Generating…")
        self._loading_label.setStyleSheet("color: #888; background: transparent;")
        self._loading_label.setToolTip("")
        self._shimmer_timer.start()
        self._shimmer_step_n = 0
        self._shimmer_max_timer.start()
        self._image_label.clear()
        self._full_pixmap = None
        self._invalidate_display_cache()
        self.seek_requested.emit(self.video_path, self._seek_time)
        # The static thumbnail now reflects this seek_time — record it so a
        # later incidental hover-end doesn't redundantly regenerate it.
        self._seek_time_at_last_thumb = self._seek_time

    # ── star rating ────────────────────────────────────────────────────────────
    def _on_star_rating_changed(self, rating: int):
        # Defensive: if anything in the grid's handler (cache write etc.)
        # raises during emit, don't propagate the exception back to Qt's
        # event dispatcher where it would terminate the app.
        try:
            self.rating_changed.emit(self.video_path, int(rating))
        except Exception:
            import logging
            logging.getLogger(__name__).exception(
                "rating_changed emit failed for %s", self.video_path
            )

    # ── context menu ───────────────────────────────────────────────────────────
    def _show_context_menu(self, pos: QPoint):
        menu      = QMenu(self)
        act_props = menu.addAction("Properties…")
        act_seek  = menu.addAction("Set thumbnail time…")
        act_open  = menu.addAction("Open with default player")
        menu.addSeparator()
        # Feature 17: watch toggle
        act_watch = menu.addAction("Mark as Unwatched" if self._is_watched else "Mark as Watched")
        menu.addSeparator()
        act_rename = menu.addAction("Rename…")
        act_copy   = menu.addAction("Copy to…")
        act_move   = menu.addAction("Move to…")
        act_del    = menu.addAction("Delete")
        action    = menu.exec(self.mapToGlobal(pos))
        if action == act_props:
            # Parent to the top-level window, NOT to self (the thumbnail widget).
            # The virtual-scroll engine can destroy thumbnail widgets while the
            # dialog is open; parenting to the main window prevents that from
            # killing the dialog mid-execution.
            dlg = MetadataDialog(self.video_path, self.window())
            dlg.exec()
        elif action == act_seek:
            self._prompt_seek_time()
        elif action == act_open:
            self.open_requested.emit(self.video_path)
        elif action == act_watch:
            # Feature 17: toggle watched state
            self._is_watched = not self._is_watched
            self.set_watched(self._is_watched)
            self.watch_toggled.emit(self.video_path, self._is_watched)
        elif action == act_rename:
            self.rename_requested.emit(self.video_path)
        elif action == act_copy:
            self.copy_requested.emit(self.video_path)
        elif action == act_move:
            self.move_requested.emit(self.video_path)
        elif action == act_del:
            self.delete_requested.emit(self.video_path)

    def _prompt_seek_time(self):
        dlg = _SeekDialog(self._seek_time, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._seek_time = dlg.seek_time
            self._seek_slider.blockSignals(True)
            self._seek_slider.setValue(int(self._seek_time))
            self._seek_slider.blockSignals(False)
            self._update_time_label()
            self._apply_seek()   # shared helper — no duplicate block

    # ── checkbox ───────────────────────────────────────────────────────────────
    def _on_check_changed(self, state: int):
        self.checked_changed.emit(
            self.video_path,
            state == Qt.CheckState.Checked.value,
        )
        self._update_frame_style()   # Feature 3

    def mousePressEvent(self, event):
        # Clicking the thumbnail does NOT select it and does NOT draw a border.
        # Selection is ONLY via the checkbox in the top-right corner (clicking
        # the checkbox itself toggles it). Double-click still opens the file.
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if event.position().y() < self.height() - BOTTOM_H:
                self.open_requested.emit(self.video_path)
        super().mouseDoubleClickEvent(event)
