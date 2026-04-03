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

from PyQt6.QtWidgets import (
    QFrame, QLabel, QCheckBox, QSlider, QMenu, QDialog,
    QVBoxLayout, QHBoxLayout, QDoubleSpinBox, QDialogButtonBox, QWidget,
)
from metadata_dialog import MetadataDialog
from PyQt6.QtCore import Qt, QThread, QTimer, QSize, pyqtSignal, QPoint
from PyQt6.QtGui import QPixmap, QImage, QFont, QPainter, QColor


# ── geometry constants (exported so the grid can compute heights) ─────────────
_SLIDER_H = 22
_NAME_H   = 24
_RATING_H = 16
BOTTOM_H  = _SLIDER_H + _NAME_H + _RATING_H   # 62 px  (public export)

_CB_SIZE   = 22
_CB_MARGIN = 5

_HOVER_DELAY_MS   = 500   # ms of dwell before video playback starts
_SEEK_DEBOUNCE_MS = 400   # ms after last slider move before regenerating thumbnail


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
                 display_w: int, display_h: int):
        super().__init__()             # no Qt parent — lifecycle via finished→deleteLater
        self._path      = video_path
        self._start_sec = start_sec
        self._disp_w    = display_w
        self._disp_h    = display_h
        self._stop_flag = False
        self._seek_to: float | None = None   # written by main thread; GIL-safe

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
        cap = cv2.VideoCapture(self._path)
        if not cap.isOpened():
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not (0 < fps < 300):
            fps = 25.0
        frame_ms = 1000.0 / fps

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total > 0 and fps > 0:
            start_f = max(0, min(int(self._start_sec * fps), total - 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

        try:
            while not self._stop_flag:
                t0 = _time.perf_counter()

                # Apply any pending seek requested from the main thread
                if self._seek_to is not None:
                    seek_sec      = self._seek_to
                    self._seek_to = None
                    if total > 0 and fps > 0:
                        cap.set(cv2.CAP_PROP_POS_FRAMES,
                                max(0, min(int(seek_sec * fps), total - 1)))

                ret, frame = cap.read()
                if not ret:
                    # EOF — loop from start_sec
                    if total > 0 and fps > 0:
                        cap.set(cv2.CAP_PROP_POS_FRAMES,
                                max(0, int(self._start_sec * fps)))
                    ret, frame = cap.read()
                    if not ret:
                        break

                # ── Scale down with OpenCV BEFORE BGR→RGB copy ───────────────
                # For 8K VR (7680×3840 → ~400×225) the copy shrinks from
                # ~88 MB to ~270 KB, making the QImage creation ~325× cheaper.
                dw, dh = self._disp_w, self._disp_h
                if dw > 0 and dh > 0:
                    fh, fw = frame.shape[:2]
                    if fw > dw or fh > dh:
                        scale = min(dw / fw, dh / fh)
                        nw    = max(1, int(fw * scale))
                        nh    = max(1, int(fh * scale))
                        frame = cv2.resize(frame, (nw, nh),
                                           interpolation=cv2.INTER_LINEAR)

                # ── BGR → RGB → QImage (owned buffer, thread-safe) ───────────
                rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rh, rw, ch = rgb.shape
                raw  = rgb.tobytes()          # Python bytes: safe cross-thread
                qimg = QImage(raw, rw, rh, ch * rw,
                              QImage.Format.Format_RGB888).copy()

                if not self._stop_flag:
                    self.frame_ready.emit(qimg)

                # ── Sleep for the remainder of the frame interval ─────────────
                elapsed_ms = (_time.perf_counter() - t0) * 1000.0
                sleep_ms   = int(frame_ms - elapsed_ms)
                if sleep_ms > 1:
                    self.msleep(sleep_ms)
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
        if event.button() == Qt.MouseButton.LeftButton:
            zone_w  = max(1, self.width() / 5)
            # Which of the 5 equal zones did the click land in?
            clicked = int(event.position().x() / zone_w) + 1
            clicked = max(1, min(5, clicked))
            # Clicking the active star again clears the rating
            if clicked == self._rating:
                clicked = 0
            self._rating = clicked
            self.update()
            self.rating_changed.emit(self._rating)
        event.accept()   # stop propagation — parent must not see this click


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

    def __init__(self, video_path: str, seek_time: float, parent=None):
        super().__init__(parent)
        self.video_path    = video_path
        self._seek_time    = seek_time
        self._duration     = 0.0
        self._aspect_ratio: float = 16 / 9

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

        # ── Feature 4: loading label with shimmer animation ───────────────────
        self._loading_label = QLabel("Generating…", self)
        self._loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._loading_label.setStyleSheet("color: #888; background: transparent;")
        self._loading_label.setFont(sm_font)

        self._shimmer_timer = QTimer(self)
        self._shimmer_timer.setInterval(400)
        self._shimmer_timer.timeout.connect(self._shimmer_step)
        self._shimmer_step_n = 0
        self._shimmer_timer.start()

        self._checkbox = QCheckBox(self)
        self._checkbox.setStyleSheet(
            "QCheckBox::indicator { width: 18px; height: 18px; }"
            "QCheckBox { background: rgba(0,0,0,140); border-radius: 3px; }"
        )
        self._checkbox.stateChanged.connect(self._on_check_changed)
        # Feature 3: connect checkbox to frame style update
        self.checked_changed.connect(lambda path, checked: self._update_frame_style())

        self._seek_slider = QSlider(Qt.Orientation.Horizontal, self)
        self._seek_slider.setRange(0, 3600)
        self._seek_slider.setValue(int(seek_time))
        self._seek_slider.setStyleSheet("""
            QSlider { background: transparent; }
            QSlider::groove:horizontal {
                height: 4px; background: #3a3a3a; border-radius: 2px; }
            QSlider::sub-page:horizontal {
                background: #5a9fd4; border-radius: 2px; }
            QSlider::handle:horizontal {
                width: 12px; height: 12px; margin: -4px 0;
                background: #7ab8e8; border-radius: 6px; }
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

        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)

        # Apply initial frame style (Feature 3)
        self._update_frame_style()

    # ── public API ─────────────────────────────────────────────────────────────
    @property
    def seek_time(self) -> float:
        return self._seek_time

    def set_thumbnail(self, image: object, duration: float):
        """Accept QImage (from generator signal) or QPixmap (from item cache)."""
        if isinstance(image, QImage) and not image.isNull():
            pix = QPixmap.fromImage(image)
        elif isinstance(image, QPixmap) and not image.isNull():
            pix = image
        else:
            return

        self._full_pixmap = pix
        self._invalidate_display_cache()

        if pix.height() > 0:
            self._aspect_ratio = pix.width() / pix.height()

        if duration > 0:
            self._duration = duration
            self._seek_slider.setRange(0, max(1, int(duration)))

        self._loading_label.hide()
        self._shimmer_timer.stop()   # Feature 4: stop shimmer when thumbnail arrives
        self._update_time_label()

        if not self._hovering:
            self._show_static_thumbnail()

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
        """Stop all timers and playback thread. Call before destroying the widget."""
        self._hover_delay.stop()
        self._seek_debounce.stop()
        self._shimmer_timer.stop()   # Feature 4
        self._stop_playback()

    # ── Feature 3: selection highlight ring ───────────────────────────────────
    def _update_frame_style(self):
        if self._checkbox.isChecked():
            self.setStyleSheet("QFrame { border: 2px solid #3a6fc4; border-radius: 2px; }")
        else:
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
    def enterEvent(self, event):
        super().enterEvent(event)
        self._hovering = True
        self._hover_delay.start()

    def leaveEvent(self, event):
        super().leaveEvent(event)
        self._hovering = False
        self._hover_delay.stop()
        self._stop_playback()
        self._show_static_thumbnail()
        self._image_label.setStyleSheet("background: #1a1a1a;")  # Feature 8: reset border

    def _start_playback(self):
        """Called by _hover_delay after 500 ms of dwell."""
        if not self._hovering or not os.path.exists(self.video_path):
            return
        lbl    = self._image_label
        thread = _VideoPlayThread(
            self.video_path, self._seek_time,
            lbl.width(), lbl.height(),
        )
        thread.frame_ready.connect(self._on_play_frame)
        thread.finished.connect(thread.deleteLater)
        self._play_thread = thread
        thread.start()
        # Feature 8: hover border during playback
        self._image_label.setStyleSheet("background: #1a1a1a; border: 2px solid #5a9fd4;")

    def _stop_playback(self):
        """Non-blocking stop: signal thread to exit; it self-destructs when done."""
        if self._play_thread is not None:
            thread             = self._play_thread
            self._play_thread  = None
            # Disconnect BEFORE stop() so no stale frame signals reach this widget
            try:
                thread.frame_ready.disconnect(self._on_play_frame)
            except RuntimeError:
                pass
            thread.stop()
            # Thread self-destructs via finished→deleteLater once run() returns
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
        """Commit the current seek position: clear thumbnail, request a new one."""
        self._seek_debounce.stop()
        self._loading_label.show()
        self._shimmer_timer.start()   # Feature 4: restart shimmer on re-seek
        self._shimmer_step_n = 0
        self._image_label.clear()
        self._full_pixmap = None
        self._invalidate_display_cache()
        self.seek_requested.emit(self.video_path, self._seek_time)

    # ── star rating ────────────────────────────────────────────────────────────
    def _on_star_rating_changed(self, rating: int):
        self.rating_changed.emit(self.video_path, rating)

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
        if event.button() == Qt.MouseButton.LeftButton:
            if event.position().y() < self.height() - BOTTOM_H:
                self._checkbox.setChecked(not self._checkbox.isChecked())
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if event.position().y() < self.height() - BOTTOM_H:
                self.open_requested.emit(self.video_path)
        super().mouseDoubleClickEvent(event)
