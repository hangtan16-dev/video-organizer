"""
Built-in video player panel widget.

Decodes frames on a background thread.  Prefers PyAV when available
(better seeking and a more complete FFmpeg build for HW decode on
Windows); falls back to OpenCV's `_VideoPlayThread` if PyAV isn't
installed.

No audio support (OpenCV / PyAV decode video only here — the volume
slider is shown but greyed out).
"""

import os

import cv2

from PyQt6.QtWidgets import (
    QWidget, QLabel, QSlider, QPushButton, QHBoxLayout, QVBoxLayout,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QPixmap, QImage, QFont

from video_thumbnail_widget import (
    _VideoPlayThread, _running_play_threads, _play_thread_reaper,
)
from app_logger import get_logger

# Prefer PyAV when installed — it has working HW decode on Windows where
# OpenCV's bundled FFmpeg does not.  This import is best-effort; if `av`
# isn't installed the widget transparently uses _VideoPlayThread.
try:
    from pyav_play_thread import _PyAVPlayThread, is_available as _pyav_available
except Exception:
    _PyAVPlayThread = None
    def _pyav_available():
        return False

log = get_logger(__name__)


class VideoPlayerWidget(QWidget):
    """
    Floating video player window.

    Shown as an independent, resizable, closable top-level window.
    Emits ``closed`` when the user closes it so the toolbar button can
    un-check itself automatically.
    """

    closed = pyqtSignal()   # emitted from closeEvent

    def __init__(self, parent=None):
        # Qt.WindowType.Window  → own title bar + resize handles + close button
        # parent kept for memory management (GC), but the window is independent
        super().__init__(parent, Qt.WindowType.Window)
        self.setWindowTitle("Video Player")
        self.setMinimumSize(480, 360)
        self.resize(800, 520)
        self._video_path: str = ''
        self._duration: float = 0.0
        self._play_thread: _VideoPlayThread | None = None
        self._is_playing: bool = False
        self._current_frame: int = 0
        self._total_frames: int = 0
        self._fps: float = 25.0

        self.setStyleSheet("background: #111; color: #ddd;")
        self.setMinimumHeight(180)

        sm_font = QFont()
        sm_font.setPointSize(9)

        # ── frame display ─────────────────────────────────────────────────────
        self._frame_label = QLabel(self)
        self._frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._frame_label.setStyleSheet("background: #000;")
        self._frame_label.setMinimumHeight(120)

        # ── seek slider ───────────────────────────────────────────────────────
        self._seek_slider = QSlider(Qt.Orientation.Horizontal, self)
        self._seek_slider.setRange(0, 1000)
        self._seek_slider.setValue(0)
        self._seek_slider.setStyleSheet("""
            QSlider::groove:horizontal { height: 4px; background: #3a3a3a; border-radius: 2px; }
            QSlider::sub-page:horizontal { background: #3a6fc4; border-radius: 2px; }
            QSlider::handle:horizontal {
                width: 12px; height: 12px; margin: -4px 0;
                background: #7ab8e8; border-radius: 6px; }
        """)
        self._seek_slider.sliderPressed.connect(self._on_seek_pressed)
        self._seek_slider.sliderReleased.connect(self._on_seek_released)
        self._seek_slider_dragging = False

        # ── control row ───────────────────────────────────────────────────────
        ctrl = QHBoxLayout()
        ctrl.setSpacing(4)

        self._btn_back = QPushButton("◀◀ -10s", self)
        self._btn_back.setFont(sm_font)
        self._btn_back.clicked.connect(lambda: self._seek_relative(-10))
        self._btn_back.setStyleSheet(
            "QPushButton { background: #2e2e2e; border: 1px solid #3a3a3a;"
            "              border-radius: 3px; padding: 3px 8px; color: #ddd; }"
            "QPushButton:hover { background: #3a3a3a; }"
        )

        self._btn_play = QPushButton("▶", self)
        self._btn_play.setFont(sm_font)
        self._btn_play.setFixedWidth(40)
        self._btn_play.clicked.connect(self._on_play_pause)
        self._btn_play.setStyleSheet(
            "QPushButton { background: #3a6fc4; border: none;"
            "              border-radius: 3px; padding: 3px 8px; color: #fff; }"
            "QPushButton:hover { background: #4a7fd4; }"
        )

        self._btn_fwd = QPushButton("+10s ▶▶", self)
        self._btn_fwd.setFont(sm_font)
        self._btn_fwd.clicked.connect(lambda: self._seek_relative(10))
        self._btn_fwd.setStyleSheet(self._btn_back.styleSheet())

        self._name_label = QLabel("No file", self)
        self._name_label.setFont(sm_font)
        self._name_label.setStyleSheet("color: #aaa; padding-left: 6px;")
        self._name_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        self._time_label = QLabel("0:00 / 0:00", self)
        self._time_label.setFont(sm_font)
        self._time_label.setStyleSheet("color: #888; padding-right: 6px;")
        self._time_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        self._vol_label = QLabel("🔇 No audio", self)
        self._vol_label.setFont(sm_font)
        self._vol_label.setStyleSheet("color: #555;")
        self._vol_label.setToolTip("Audio playback is not supported by OpenCV")

        ctrl.addWidget(self._btn_back)
        ctrl.addWidget(self._btn_play)
        ctrl.addWidget(self._btn_fwd)
        ctrl.addWidget(self._name_label, 1)
        ctrl.addWidget(self._time_label)
        ctrl.addWidget(self._vol_label)

        # ── main layout ───────────────────────────────────────────────────────
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        layout.addWidget(self._frame_label, 1)
        layout.addWidget(self._seek_slider)
        layout.addLayout(ctrl)

        # ── position update timer ─────────────────────────────────────────────
        self._pos_timer = QTimer(self)
        self._pos_timer.setInterval(50)
        self._pos_timer.timeout.connect(self._update_position)

    # ── public API ─────────────────────────────────────────────────────────────
    def play(self, video_path: str, start_sec: float = 0.0):
        """Load and start playing a video."""
        self.stop()
        if not os.path.exists(video_path):
            return

        self._video_path = video_path
        self._current_frame = 0

        # Probe video metadata (hw-accel where supported)
        from video_capture_helper import open_capture
        cap = open_capture(video_path, hw_accel=True)
        if cap is None:
            return
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            self._fps = fps if 0 < fps < 300 else 25.0
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self._total_frames = max(0, total)
            self._duration = self._total_frames / self._fps if self._fps > 0 else 0
        finally:
            cap.release()

        self._seek_slider.setRange(0, max(1, self._total_frames))
        basename = os.path.basename(video_path)
        self._name_label.setText(basename)
        self.setWindowTitle(f"Video Player – {basename}")

        lbl = self._frame_label
        # Native FPS for player panel (real-time playback).  Prefer PyAV
        # when available — its FFmpeg has working D3D11VA / VAAPI /
        # VideoToolbox HW decode that OpenCV's bundled FFmpeg lacks on
        # standard pip installs.
        if _PyAVPlayThread is not None and _pyav_available():
            log.info("Player: using PyAV backend")
            thread = _PyAVPlayThread(
                video_path, start_sec, lbl.width(), lbl.height(),
                hw_accel=True,
            )
        else:
            log.info("Player: using OpenCV backend (install `av` for HW decode)")
            thread = _VideoPlayThread(
                video_path, start_sec, lbl.width(), lbl.height(),
                hw_accel=True,
            )
        # Hold a strong reference in the global set BEFORE start() — same
        # crash-prevention pattern as VideoThumbnailWidget._start_playback.
        # Without this, stop() drops self._play_thread to None and the QThread
        # Python wrapper can be GC'd while cv2.read() is still running →
        # "QThread: Destroyed while thread is still running".
        _running_play_threads.add(thread)
        # Registry removal on the GUI thread (QUEUED), not via a Direct lambda
        # inside finish(); C++ deletion handled by qthread_registry.install()
        # in __init__. See video_thumbnail_widget._PlayThreadReaper and
        # qthread_registry for the GIL ⊗ QThread-mutex deadlock this avoids.
        thread.finished.connect(
            _play_thread_reaper.reap, Qt.ConnectionType.UniqueConnection)
        thread.frame_ready.connect(self._on_frame)
        self._play_thread = thread
        self._is_playing = True
        self._current_frame = int(start_sec * self._fps)
        thread.start()

        self._btn_play.setText("⏸")
        self._pos_timer.start()

    def stop(self):
        """Stop playback and clean up thread."""
        self._pos_timer.stop()
        if self._play_thread is not None:
            thread = self._play_thread
            self._play_thread = None
            try:
                thread.frame_ready.disconnect(self._on_frame)
            except RuntimeError:
                pass
            thread.stop()
        self._is_playing = False
        self._btn_play.setText("▶")

    # ── internal slots ─────────────────────────────────────────────────────────
    @pyqtSlot(object)
    def _on_frame(self, qimage):
        if isinstance(qimage, QImage) and not qimage.isNull():
            self._frame_label.setPixmap(QPixmap.fromImage(qimage))
        # Increment frame counter
        self._current_frame += 1
        if self._total_frames > 0 and self._current_frame >= self._total_frames:
            self._current_frame = 0  # looped

    def _update_position(self):
        """Update the seek slider and time label based on frame counter."""
        if self._seek_slider_dragging:
            return
        self._seek_slider.blockSignals(True)
        self._seek_slider.setValue(self._current_frame)
        self._seek_slider.blockSignals(False)
        self._update_time_label()

    def _update_time_label(self):
        cur_sec = self._current_frame / self._fps if self._fps > 0 else 0
        self._time_label.setText(f"{self._fmt(cur_sec)} / {self._fmt(self._duration)}")

    @staticmethod
    def _fmt(secs: float) -> str:
        s = int(max(0, secs))
        h, rem = divmod(s, 3600)
        m, sec = divmod(rem, 60)
        return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"

    def _on_play_pause(self):
        if self._is_playing:
            # Pause
            self.stop()
            self._btn_play.setText("▶")
        else:
            # Resume from current position
            if self._video_path:
                seek = self._current_frame / self._fps if self._fps > 0 else 0
                self.play(self._video_path, seek)

    def _on_seek_pressed(self):
        self._seek_slider_dragging = True

    def _on_seek_released(self):
        self._seek_slider_dragging = False
        frame = self._seek_slider.value()
        self._current_frame = frame
        sec = frame / self._fps if self._fps > 0 else 0
        if self._play_thread is not None:
            self._play_thread.request_seek(sec)
        self._update_time_label()

    def _seek_relative(self, delta_sec: float):
        """Seek forward or backward by delta_sec seconds."""
        if not self._video_path:
            return
        new_frame = int(self._current_frame + delta_sec * self._fps)
        new_frame = max(0, min(self._total_frames - 1 if self._total_frames > 0 else 0, new_frame))
        self._current_frame = new_frame
        sec = new_frame / self._fps if self._fps > 0 else 0
        if self._play_thread is not None:
            self._play_thread.request_seek(sec)
        else:
            # Not playing: start from new position
            if self._video_path:
                self.play(self._video_path, sec)
        self._update_time_label()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Update thread display size on resize
        if self._play_thread is not None:
            lbl = self._frame_label
            self._play_thread.set_display_size(lbl.width(), lbl.height())

    def closeEvent(self, event):
        """Stop playback and notify the main window so it can un-check the action."""
        self.stop()
        self.closed.emit()
        super().closeEvent(event)
