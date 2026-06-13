"""
In-app "full screen" video player.

Double-clicking a thumbnail expands the video to fill the whole app window and
starts playback here (instead of launching an external player).

Backend: Qt Multimedia's QMediaPlayer + QVideoWidget. On Qt 6.5+ this is the
bundled FFmpeg backend, which HW-decodes (D3D11VA on Windows) and has HEVC/H.265
built in — so 8K VR HEVC plays smoothly without the Windows "HEVC Video
Extensions" store codec, and there is no Python-side frame loop to drop frames.
QVideoWidget composites on the GPU. Audio + A/V sync + seeking are native.

Controls: play/pause, a scrub seek bar, and skip back 30s / back 10s /
forward 10s / forward 30s. Esc (or the ✕ / double-click) exits; Space toggles
play/pause; ←/→ skip 10s.
"""

import os
import threading

from PyQt6.QtWidgets import (
    QWidget, QLabel, QSlider, QPushButton, QHBoxLayout, QVBoxLayout, QSizePolicy,
    QGraphicsScene, QGraphicsView, QGraphicsRectItem, QGraphicsItem, QFrame,
    QStackedLayout,
)
from PyQt6.QtCore import Qt, QUrl, QRect, QRectF, QSizeF, QObject, QTimer, pyqtSignal
from PyQt6.QtGui import QFont, QPainter, QColor, QPen, QTransform
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QGraphicsVideoItem

import vr_unwarp as vu

from app_logger import get_logger

log = get_logger(__name__)


# ── VR 2D view: crop modes ────────────────────────────────────────────────────
# A VR file packs the two eyes either side-by-side (left|right) or top-bottom
# (over/under). "Playing in 2D" = showing ONE eye, so the double image is gone.
# (Un-warping the fisheye/equirectangular projection to a true flat image is a
# separate, shader-based step — a planned follow-up; this is the eye-crop.)
CROP_OFF, CROP_LEFT, CROP_RIGHT, CROP_TOP, CROP_BOTTOM = (
    'off', 'left', 'right', 'top', 'bottom')
_CROP_CYCLE = [CROP_OFF, CROP_LEFT, CROP_RIGHT, CROP_TOP, CROP_BOTTOM]
_CROP_LABEL = {
    CROP_OFF: 'Full', CROP_LEFT: '2D: L', CROP_RIGHT: '2D: R',
    CROP_TOP: '2D: T', CROP_BOTTOM: '2D: B',
}


def detect_vr_layout(path: str) -> str:
    """Guess the stereo packing of a VR file from its name (the same trick
    DeoVR / HereSphere use), returning a CROP_* mode:
      • side-by-side  → CROP_LEFT  (show the left eye)
      • top-bottom    → CROP_TOP
      • mono / not VR → CROP_OFF   (no crop)
    A VR projection tag with no explicit stereo layout defaults to SBS (the
    overwhelming norm). The user can always cycle the eye manually if a file is
    tagged oddly or untagged."""
    name = os.path.basename(path).lower()

    mono_tags = ('_mono', 'monoscopic', '_2d_', ' 2d ')
    if any(t in name for t in mono_tags):
        return CROP_OFF

    # Top-bottom / over-under
    tb_tags = ('_tb', '_ou', 'over-under', 'overunder', 'top-bottom',
               'topbottom', '_3dv', 'htb', 'fulltb', '_180_tb', '_360_tb')
    if any(t in name for t in tb_tags):
        return CROP_TOP

    # Side-by-side (explicit) + common fisheye-lens tags (always SBS)
    sbs_tags = ('_lr', '_sbs', 'side-by-side', 'sidebyside', '_3dh',
                'half-sbs', 'halfsbs', '180x180_3dh', '_mkx', 'mkx200',
                'mkx220', '_vrca', 'vrca220', 'fisheye', '_f180', '_180_sbs')
    if any(t in name for t in sbs_tags):
        return CROP_LEFT

    # A VR projection is present but no explicit layout → assume SBS.
    vr_tags = ('vr180', 'vr360', '_180_', '_360_', '180.', '360.',
               'equirect', '_eac', 'panorama')
    if any(t in name for t in vr_tags):
        return CROP_LEFT

    return CROP_OFF


class _VideoView(QGraphicsView):
    """QGraphicsView hosting the QGraphicsVideoItem. Subclassed only so a resize
    re-fits the current crop rectangle to the new viewport size."""

    def __init__(self, scene, on_resize, parent=None):
        super().__init__(scene, parent)
        self._on_resize = on_resize
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setStyleSheet("background: #000; border: none;")
        self.setBackgroundBrush(QColor(0, 0, 0))
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._on_resize()

_SKIP_BACK_30  = -30_000
_SKIP_BACK_10  = -10_000
_SKIP_FWD_10   =  10_000
_SKIP_FWD_30   =  30_000

_BTN_STYLE = (
    "QPushButton { background: #2e2e2e; border: 1px solid #3a3a3a;"
    "              border-radius: 4px; padding: 6px 10px; color: #ddd; font-size: 13px; }"
    "QPushButton:hover  { background: #3a3a3a; }"
    "QPushButton:pressed{ background: #444; }"
)
_PLAY_STYLE = (
    "QPushButton { background: #3a6fc4; border: none; border-radius: 4px;"
    "              padding: 6px 14px; color: #fff; font-size: 15px; }"
    "QPushButton:hover { background: #4a7fd4; }"
)
_SLIDER_STYLE = """
    QSlider::groove:horizontal { height: 8px; background: #3a3a3a; border-radius: 4px; }
    QSlider::sub-page:horizontal { background: #3a6fc4; border-radius: 4px; }
    QSlider::handle:horizontal { width: 16px; height: 16px; margin: -5px 0;
        background: #7ab8e8; border-radius: 8px; }
"""


class _ReadAheadCache:
    """Background OS page-cache warmer for the file being played.

    Sequentially reads the bytes ~AHEAD_SECONDS ahead of the current playback
    position into the OS file cache (the bytes read are DISCARDED — only the
    cache is warmed, so the app's own RAM use stays ~one CHUNK). A forward skip
    then lands on already-cached bytes, so QMediaPlayer's seek needs no physical
    disk seek — the dominant cost on an HDD. It reads sequentially just ahead of
    playback, so the disk head barely moves relative to the playback reads;
    idles once it is AHEAD_SECONDS ahead and tops up as playback advances.

    The byte offset for a time is estimated as position/duration × file_size
    (≈ uniform bitrate). It's approximate, but the AHEAD window absorbs the
    error — we only need to warm the right *region*, not an exact byte.

    Read-only, so it runs as a daemon thread (a half-finished read at exit is
    harmless) AND is explicitly stopped + joined by the player for cleanliness.
    """
    AHEAD_SECONDS   = 120                      # read-ahead window (user-chosen 2 min)
    MAX_AHEAD_BYTES = 4 * 1024 * 1024 * 1024   # 4 GB cap (OS cache; reclaimable) — lets 120s
                                               # fully apply even for high-bitrate VR files
    CHUNK           = 4 * 1024 * 1024          # 4 MB reads

    def __init__(self, path: str):
        self._path = path
        try:
            self._size = os.path.getsize(path)
        except OSError:
            self._size = 0
        self._pos_ms = 0
        self._dur_ms = 0
        self._read_pos = 0      # how far the read-ahead has reached (bytes)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="vorg-readahead", daemon=True)

    def start(self):
        if self._size > 0:
            self._thread.start()

    def update(self, pos_ms: int, dur_ms: int):
        """Push the latest playback position/duration (called from the GUI
        thread on positionChanged/durationChanged)."""
        with self._lock:
            self._pos_ms = max(0, int(pos_ms))
            if dur_ms and dur_ms > 0:
                self._dur_ms = int(dur_ms)

    def read_ahead_ms(self) -> int:
        """How far ahead (in ms of playback time) the read-ahead has cached —
        for the on-screen buffered bar. Byte→time via the same uniform-bitrate
        estimate; approximate, which is fine for a visual indicator."""
        with self._lock:
            rp, dur = self._read_pos, self._dur_ms
        if dur <= 0 or self._size <= 0:
            return 0
        return int(rp * dur / self._size)

    def stop(self):
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    @staticmethod
    def _target_window(pos_ms, dur_ms, size, ahead_seconds, max_ahead_bytes):
        """(cur_byte, ahead_bytes) for the current position, or None if the
        duration/size aren't known yet."""
        if dur_ms <= 0 or size <= 0:
            return None
        bpms = size / dur_ms
        cur = min(size, int(pos_ms * bpms))
        ahead = min(int(ahead_seconds * 1000 * bpms), max_ahead_bytes)
        return cur, ahead

    def _run(self):
        try:
            f = open(self._path, 'rb', buffering=0)
        except OSError:
            return
        read_pos = 0
        last_cur = -1
        try:
            while not self._stop.is_set():
                with self._lock:
                    pos, dur = self._pos_ms, self._dur_ms
                win = self._target_window(pos, dur, self._size,
                                          self.AHEAD_SECONDS, self.MAX_AHEAD_BYTES)
                if win is None:
                    self._stop.wait(0.2)            # duration not known yet
                    continue
                cur, ahead = win
                # A big jump in position = a seek → restart read-ahead there.
                if last_cur < 0 or abs(cur - last_cur) > ahead:
                    read_pos = cur
                last_cur = cur
                if read_pos < cur:
                    read_pos = cur                  # never lag behind playback
                with self._lock:
                    self._read_pos = read_pos       # publish for the buffered bar
                if read_pos >= cur + ahead or read_pos >= self._size:
                    self._stop.wait(0.15)           # far enough ahead → idle
                    continue
                try:
                    f.seek(read_pos)
                    data = f.read(self.CHUNK)
                except OSError:
                    break
                if not data:
                    read_pos = self._size
                    continue
                read_pos += len(data)
                self._stop.wait(0.005)              # yield: stay stop-responsive
        finally:
            try:
                f.close()
            except Exception:
                pass


class _SeekSlider(QSlider):
    """Seek slider that also paints a lighter-blue 'read-ahead buffered' bar
    from the playhead out to how far the read-ahead has cached — so the user can
    watch the buffer fill and see it lag/grow on a slow HDD source."""

    _GROOVE_H = 8
    _HANDLE_W = 16

    def __init__(self, orientation, parent=None):
        super().__init__(orientation, parent)
        self._buffered_ms = 0

    def set_buffered(self, value_ms: int):
        v = max(0, min(int(value_ms), self.maximum()))
        if v != self._buffered_ms:
            self._buffered_ms = v
            self.update()

    def paintEvent(self, event):
        super().paintEvent(event)          # groove + played sub-page + handle
        mx = self.maximum()
        if mx <= 0 or self._buffered_ms <= self.value():
            return
        track_w = self.width() - self._HANDLE_W
        if track_w <= 0:
            return
        x0 = self._HANDLE_W // 2
        x_play = x0 + int(track_w * self.value() / mx)
        x_buf  = x0 + int(track_w * self._buffered_ms / mx)
        cy = self.height() // 2
        gh = self._GROOVE_H
        rect = QRect(x_play, cy - gh // 2, max(0, x_buf - x_play), gh)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(122, 184, 232, 150))   # translucent light blue
        p.drawRoundedRect(rect, gh // 2, gh // 2)
        p.end()


class FullscreenVideoPlayer(QWidget):
    """A full-window video player overlay. Shown over the whole main window;
    emits ``closed`` when the user exits so the host can hide + refocus."""

    closed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background: #000;")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._duration_ms = 0
        self._seeking = False
        self._readahead = None
        self._crop_mode = CROP_OFF
        self._native = QSizeF()

        # VR → 2D un-warp (Phase 2) state. The un-warp renders on a separate
        # Qt Quick 3D surface (built lazily in _setup_unwarp_surface); if that
        # isn't available the player still works and the "2D" button falls back
        # to the plain eye-crop on the graphics surface.
        self._unwarp_available = False
        self._unwarp_on = False
        self._u_proj = vu.PROJ_EQUIRECT_180
        self._u_eye = 'left'
        # Defaults the user settled on: a 220° view in the STEREOGRAPHIC
        # projection — the full immersive hemisphere flattened, with no
        # rectilinear edge-stretch. Tunable live with [ ] (FOV) and O (view).
        self._u_hfov = 220.0
        self._u_lens = 200.0
        self._u_flip = True          # Qt Quick 3D texture origin is bottom-left
        self._u_outproj = vu.OUT_STEREOGRAPHIC
        # Auto-enable bookkeeping: only auto-flatten once per file, and never
        # fight a manual toggle.
        self._auto_unwarp_done = False
        self._user_toggled_2d = False

        # ── media pipeline (HW-accelerated, GPU-composited) ──────────────────
        self._player = QMediaPlayer(self)
        self._audio  = QAudioOutput(self)
        self._audio.setVolume(0.85)
        self._player.setAudioOutput(self._audio)

        # The video renders into a QGraphicsVideoItem inside a QGraphicsView,
        # rather than a plain QVideoWidget, so we can show just ONE eye of a VR
        # video (crop to a sub-rectangle) while keeping QMediaPlayer's HW decode.
        # A QOpenGLWidget viewport keeps the (up to 8K) frame compositing on the
        # GPU — same no-dropped-frames goal as the old QVideoWidget path.
        self._scene = QGraphicsScene(self)
        # The video item is a CHILD of a clip item. The clip item is set to the
        # chosen eye's rectangle and clips its children to that shape — so the
        # OTHER eye can never bleed into the letterbox margins. The clip item
        # also carries the scale/translate that fits the eye to the viewport
        # (centred), so we never rely on QGraphicsView.fitInView (which proved
        # unreliable here, and wouldn't clip the second eye anyway).
        self._clip = QGraphicsRectItem()
        self._clip.setPen(QPen(Qt.PenStyle.NoPen))
        self._clip.setFlag(QGraphicsItem.GraphicsItemFlag.ItemClipsChildrenToShape, True)
        # Start wide-open (a zero rect would clip the video to nothing → black
        # until the first crop is applied). _apply_crop narrows it to one eye.
        self._clip.setRect(-1e7, -1e7, 2e7, 2e7)
        self._scene.addItem(self._clip)
        self._video_item = QGraphicsVideoItem(self._clip)
        self._video_item.setAspectRatioMode(Qt.AspectRatioMode.IgnoreAspectRatio)
        self._video = _VideoView(self._scene, self._apply_crop, self)
        # NOTE: no QOpenGLWidget viewport here. This graphics surface now only
        # shows NON-VR video (VR goes to the Qt Quick 3D un-warp surface), which
        # is rarely 8K, and overlapping two GL surfaces (this + the QQuickWidget,
        # kept alive by StackAll) is exactly the combination Qt warns about. A
        # raster viewport keeps a single GL surface in the window.
        self._player.setVideoOutput(self._video_item)
        self._video_item.nativeSizeChanged.connect(self._refresh_native_size)

        # ── control bar ──────────────────────────────────────────────────────
        font = QFont(); font.setPointSize(10)

        self._slider = _SeekSlider(Qt.Orientation.Horizontal, self)
        self._slider.setRange(0, 0)
        self._slider.setStyleSheet(_SLIDER_STYLE)
        self._slider.sliderPressed.connect(self._on_seek_pressed)
        self._slider.sliderReleased.connect(self._on_seek_released)
        self._slider.sliderMoved.connect(self._on_seek_moved)

        self._time = QLabel("0:00 / 0:00", self)
        self._time.setStyleSheet("color: #ccc;")
        self._time.setFont(font)
        self._time.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        self._name = QLabel("", self)
        self._name.setStyleSheet("color: #aaa;")
        self._name.setFont(font)

        self._btn_b30 = self._make_btn("⏮ 30s", lambda: self.skip(_SKIP_BACK_30), _BTN_STYLE)
        self._btn_b10 = self._make_btn("◀◀ 10s", lambda: self.skip(_SKIP_BACK_10), _BTN_STYLE)
        self._btn_play = self._make_btn("⏸", self.toggle_play, _PLAY_STYLE)
        self._btn_play.setFixedWidth(54)
        self._btn_f10 = self._make_btn("10s ▶▶", lambda: self.skip(_SKIP_FWD_10), _BTN_STYLE)
        self._btn_f30 = self._make_btn("30s ⏭", lambda: self.skip(_SKIP_FWD_30), _BTN_STYLE)

        # VR → 2D button. With the un-warp surface available it toggles the flat
        # (crop + un-warp) view; otherwise it falls back to cycling the eye-crop.
        self._btn_2d = self._make_btn("2D", self._on_2d_button, _BTN_STYLE)
        self._btn_2d.setFixedWidth(76)
        self._btn_2d.setToolTip(
            "VR → 2D: flatten a VR video to one un-warped eye.\n"
            "Click to toggle. Keys: V eye · P projection · [ ] zoom · F flip.")

        self._vol = QSlider(Qt.Orientation.Horizontal, self)
        self._vol.setRange(0, 100)
        self._vol.setValue(85)
        self._vol.setFixedWidth(110)
        self._vol.setStyleSheet(_SLIDER_STYLE)
        self._vol.valueChanged.connect(lambda v: self._audio.setVolume(v / 100))

        self._btn_close = self._make_btn("✕  Close", self.close_player, _BTN_STYLE)

        # seek row: name (left) … time (right) above the scrub bar
        toprow = QHBoxLayout()
        toprow.addWidget(self._name, 1)
        toprow.addWidget(self._time, 0)

        btnrow = QHBoxLayout()
        btnrow.setSpacing(8)
        for b in (self._btn_b30, self._btn_b10, self._btn_play, self._btn_f10, self._btn_f30):
            btnrow.addWidget(b)
        btnrow.addStretch(1)
        btnrow.addWidget(self._btn_2d)
        btnrow.addSpacing(12)
        vol_lbl = QLabel("🔊", self); vol_lbl.setFont(font)
        btnrow.addWidget(vol_lbl)
        btnrow.addWidget(self._vol)
        btnrow.addSpacing(12)
        btnrow.addWidget(self._btn_close)

        controls = QWidget(self)
        controls.setStyleSheet("background: rgba(20,20,20,235);")
        cl = QVBoxLayout(controls)
        cl.setContentsMargins(14, 8, 14, 10)
        cl.setSpacing(6)
        cl.addLayout(toprow)
        cl.addWidget(self._slider)
        cl.addLayout(btnrow)

        # ── video surfaces, overlapped: normal graphics surface + the VR un-warp
        #    Qt Quick 3D surface. StackAll keeps BOTH visible (rendering) and just
        #    raises the active one — a HIDDEN QQuickWidget releases its RHI/scene
        #    graph and returns BLACK when re-shown, which made the 2D toggle go
        #    black. set_unwarp() raises the right surface + routes the video. ────
        self._video.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._video_area = QWidget(self)
        self._stack = QStackedLayout(self._video_area)
        self._stack.setStackingMode(QStackedLayout.StackingMode.StackAll)
        self._stack.addWidget(self._video)
        self._setup_unwarp_surface()                 # adds the un-warp surface
        self._stack.setCurrentWidget(self._video)    # graphics on top by default

        # On-screen readout shown in un-warp mode (so the user can see + tune the
        # projection / eye / FOV live). A free overlay, raised above the video.
        self._u_hud = QLabel(self)
        self._u_hud.setStyleSheet(
            "QLabel { color:#eaeaea; background: rgba(0,0,0,165); padding:8px 12px;"
            " border-radius:6px; font-family:Consolas,monospace; font-size:12px; }")
        self._u_hud.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._u_hud.hide()

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._video_area, 1)
        root.addWidget(controls, 0)

        # ── signals ───────────────────────────────────────────────────────────
        self._player.positionChanged.connect(self._on_position_changed)
        self._player.durationChanged.connect(self._on_duration_changed)
        self._player.playbackStateChanged.connect(self._on_state_changed)
        self._player.errorOccurred.connect(self._on_error)

        # Poll the read-ahead extent (~3 Hz) to grow the buffered bar smoothly.
        self._buffer_timer = QTimer(self)
        self._buffer_timer.setInterval(300)
        self._buffer_timer.timeout.connect(self._update_buffer_indicator)

    def _update_buffer_indicator(self):
        # Also poll the frame size here: some backends never emit
        # nativeSizeChanged for a QGraphicsVideoItem, so the signal alone can
        # leave the crop un-applied. Polling at ~3 Hz applies it within a frame
        # or two of the first decoded frame.
        self._refresh_native_size()
        if self._readahead is not None:
            self._slider.set_buffered(self._readahead.read_ahead_ms())

    # ── public API ───────────────────────────────────────────────────────────
    def play(self, video_path: str):
        """Load and start playing `video_path` from the beginning."""
        self._stop_readahead()
        self._name.setText(os.path.basename(video_path))
        self._slider.setRange(0, 0)
        self._slider.set_buffered(0)
        self._duration_ms = 0
        # Reset the known frame size and auto-pick the eye-crop from the filename
        # (waits for the new video's nativeSizeChanged before actually cropping).
        self._native = QSizeF()
        layout = detect_vr_layout(video_path)        # 'off'/'left'/'right'/'top'/'bottom'
        self.set_crop_mode(layout)                   # graphics-surface fallback crop
        # Auto-detect the VR un-warp parameters and flatten VR files on open.
        self._u_proj = vu.detect_projection(video_path)
        self._u_lens = vu.detect_lens_fov(video_path)
        self._u_eye = 'left' if layout == CROP_OFF else layout
        self._user_toggled_2d = False
        if self._unwarp_available:
            filename_is_vr = (layout != CROP_OFF)
            self.set_unwarp(filename_is_vr)          # VR → flat; normal video → graphics
            # If the filename already decided, don't second-guess by frame shape;
            # otherwise leave the door open for aspect-based auto-detection once
            # the first frame's size is known (_maybe_auto_unwarp_by_aspect).
            self._auto_unwarp_done = filename_is_vr
        self._player.setSource(QUrl.fromLocalFile(video_path))
        self._audio.setVolume(self._vol.value() / 100)
        self._player.play()
        # Warm the OS cache ahead of playback so forward skips avoid a disk seek.
        self._readahead = _ReadAheadCache(video_path)
        self._readahead.start()
        self._buffer_timer.start()

    def stop(self):
        """Stop playback and release the media source."""
        self._buffer_timer.stop()
        self._slider.set_buffered(0)
        self._stop_readahead()
        try:
            self._player.stop()
            self._player.setSource(QUrl())
        except Exception as e:
            log.warning("fullscreen player stop failed: %s", e)

    def _stop_readahead(self):
        if self._readahead is not None:
            try:
                self._readahead.stop()
            except Exception:
                pass
            self._readahead = None

    def close_player(self):
        """Exit full-screen playback."""
        self.stop()
        self.closed.emit()

    def toggle_play(self):
        if self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self._player.pause()
        else:
            self._player.play()

    def skip(self, delta_ms: int):
        """Skip forward/backward by delta_ms, clamped to [0, duration]."""
        target = self._clamp_target(self._player.position(), delta_ms, self._duration_ms)
        self._player.setPosition(target)

    @staticmethod
    def _clamp_target(cur_ms: int, delta_ms: int, dur_ms: int) -> int:
        t = cur_ms + delta_ms
        if t < 0:
            t = 0
        if dur_ms > 0 and t > dur_ms:
            t = dur_ms
        return t

    # ── VR → 2D eye-crop ───────────────────────────────────────────────────────
    def _refresh_native_size(self, *_):
        """Get the decoded frame's pixel size — from the nativeSizeChanged signal
        OR polled by the buffer timer (some backends don't emit the signal for a
        QGraphicsVideoItem). On first/changed size, size the video item 1:1 to
        source pixels (so crop rects are in pixel units) and apply the crop."""
        try:
            size = self._video_item.nativeSize()
        except (RuntimeError, AttributeError):
            return
        if size.isValid() and not size.isEmpty() and size != self._native:
            self._native = QSizeF(size)
            self._video_item.setSize(self._native)
            log.info("VR player: frame %.0f×%.0f, crop=%s",
                     self._native.width(), self._native.height(), self._crop_mode)
            self._apply_crop()
            # Now the frame shape is known — auto-flatten an untagged stereo file.
            self._maybe_auto_unwarp_by_aspect()

    @staticmethod
    def _crop_rect_for(mode: str, w: float, h: float):
        """(x, y, w, h) sub-rectangle of a w×h frame for `mode`, or None if the
        size isn't known yet. Static + plain tuples so it's unit-testable
        without Qt or a running player."""
        if w <= 0 or h <= 0:
            return None
        if mode == CROP_LEFT:   return (0.0,     0.0,     w / 2.0, h)
        if mode == CROP_RIGHT:  return (w / 2.0, 0.0,     w / 2.0, h)
        if mode == CROP_TOP:    return (0.0,     0.0,     w,       h / 2.0)
        if mode == CROP_BOTTOM: return (0.0,     h / 2.0, w,       h / 2.0)
        return (0.0, 0.0, w, h)                       # CROP_OFF → full frame

    def _apply_crop(self):
        """Clip the video to the chosen eye's rectangle and scale/centre it to
        fill the viewport (letterboxed), entirely via the clip item's transform
        — no fitInView, and the other eye is clipped away (no margin bleed)."""
        rect = self._crop_rect_for(self._crop_mode,
                                   self._native.width(), self._native.height())
        if rect is None:
            return   # native size not known yet — applied once it arrives
        cx, cy, cw, ch = rect
        # Clip children (the full-frame video item) to just this eye.
        self._clip.setRect(cx, cy, cw, ch)
        vp = self._video.viewport()
        vw, vh = vp.width(), vp.height()
        if vw <= 0 or vh <= 0 or cw <= 0 or ch <= 0:
            return
        s = min(vw / cw, vh / ch)                 # fit the eye, keep aspect
        ox = (vw - cw * s) / 2.0                   # centre horizontally
        oy = (vh - ch * s) / 2.0                   # centre vertically
        # Map eye-rect top-left (cx,cy) → (ox,oy), scaled by s.
        t = QTransform()
        t.translate(ox - cx * s, oy - cy * s)
        t.scale(s, s)
        self._clip.setTransform(t)
        self._video.resetTransform()
        self._video.setSceneRect(0.0, 0.0, float(vw), float(vh))

    def set_crop_mode(self, mode: str):
        """Set the eye-crop mode (one of CROP_*) on the graphics surface and
        re-fit the view. The 2D button only shows the crop label in the FALLBACK
        case (no un-warp surface); otherwise the button reflects the un-warp
        toggle ('2D' / '2D ✓')."""
        if mode not in _CROP_LABEL:
            mode = CROP_OFF
        self._crop_mode = mode
        if not self._unwarp_available:
            self._btn_2d.setText(_CROP_LABEL[mode])
        self._apply_crop()

    def cycle_crop(self):
        """Cycle Full → Left → Right → Top → Bottom (eye-crop on the graphics
        surface — the fallback when the un-warp surface isn't available)."""
        i = _CROP_CYCLE.index(self._crop_mode) if self._crop_mode in _CROP_CYCLE else 0
        self.set_crop_mode(_CROP_CYCLE[(i + 1) % len(_CROP_CYCLE)])

    # ── VR → 2D un-warp (Phase 2) ───────────────────────────────────────────────
    def _setup_unwarp_surface(self):
        """Build the Qt Quick 3D un-warp surface and add it to the stack. On any
        failure (Quick3D unavailable, QML error) the player keeps working with
        the graphics surface only — _unwarp_available stays False."""
        try:
            from PyQt6.QtQuickWidgets import QQuickWidget
            from vr_unwarp_mesh import UnwarpGeometry
            qml = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'vr_unwarp_view.qml')
            self._qml = QQuickWidget(self)
            self._qml.setResizeMode(QQuickWidget.ResizeMode.SizeRootObjectToView)
            self._qml.setSource(QUrl.fromLocalFile(qml))
            if self._qml.status() == QQuickWidget.Status.Error:
                raise RuntimeError("; ".join(e.toString() for e in self._qml.errors()))
            root = self._qml.rootObject()
            self._u_model = root.findChild(QObject, 'unwarpModel')
            self._u_vout = root.findChild(QObject, 'videoOut')
            if root is None or self._u_model is None or self._u_vout is None:
                raise RuntimeError("un-warp QML scene incomplete")
            self._u_geo = UnwarpGeometry()
            self._u_model.setProperty('geometry', self._u_geo)
            self._u_root = root
            self._stack.addWidget(self._qml)             # index 1
            self._unwarp_available = True
        except Exception as e:
            log.warning("VR un-warp surface unavailable (2D button falls back to "
                        "eye-crop): %s", e)
            self._unwarp_available = False

    def _on_2d_button(self):
        """2D control button: toggle the un-warp view, or — if the un-warp
        surface failed to build — fall back to cycling the plain eye-crop."""
        self._user_toggled_2d = True          # respect manual control thereafter
        if self._unwarp_available:
            self.set_unwarp(not self._unwarp_on)
        else:
            self.cycle_crop()

    @staticmethod
    def _vr_eye_for_aspect(w, h):
        """Guess a stereo eye-crop from the FRAME SHAPE (shared with thumbnails /
        hover preview via vr_unwarp.detect_stereo_eye): a ~2:1 high-res frame is
        side-by-side → left eye; None otherwise so normal videos are untouched."""
        return vu.detect_stereo_eye(w, h)

    def _maybe_auto_unwarp_by_aspect(self):
        """Once the first frame's size is known, auto-flatten an untagged VR file
        if its frame shape says it's stereo. Runs at most once per file and never
        overrides a manual 2D toggle."""
        if (not self._unwarp_available or self._auto_unwarp_done
                or self._user_toggled_2d or self._unwarp_on):
            return
        eye = self._vr_eye_for_aspect(self._native.width(), self._native.height())
        if eye is not None:
            self._auto_unwarp_done = True
            self._u_eye = eye
            self.set_unwarp(True)

    def set_unwarp(self, on: bool):
        """Switch between the normal graphics surface and the VR un-warp surface
        and route the player's video to the active one."""
        if on and not self._unwarp_available:
            return
        self._unwarp_on = bool(on)
        if self._unwarp_on:
            self._rebuild_mesh()
            self._player.setVideoOutput(self._u_vout)
            self._stack.setCurrentWidget(self._qml)
        else:
            self._player.setVideoOutput(self._video_item)
            self._stack.setCurrentWidget(self._video)
        self._btn_2d.setText("2D ✓" if self._unwarp_on else "2D")
        self._update_unwarp_hud()

    def _rebuild_mesh(self):
        """Recompute the un-warp mesh for the current format/eye/FOV and the
        live viewport aspect, and push the vertical-flip flag to the texture."""
        if not self._unwarp_available:
            return
        w = max(1, self._qml.width())
        h = max(1, self._qml.height())
        self._u_geo.rebuild(projection=self._u_proj, hfov_deg=self._u_hfov,
                            out_aspect=w / h, lens_fov_deg=self._u_lens,
                            eye=self._u_eye, out_proj=self._u_outproj)
        self._u_root.setProperty('flipV', self._u_flip)
        self._update_unwarp_hud()

    def _update_unwarp_hud(self):
        """Show/refresh the on-screen un-warp readout (only in un-warp mode)."""
        if not (self._unwarp_available and self._unwarp_on):
            self._u_hud.hide()
            return
        self._u_hud.setText(
            f"VR → 2D   projection = {self._u_proj}   eye = {self._u_eye}   "
            f"FOV = {self._u_hfov:.0f}°   view = {self._u_outproj}\n"
            "P projection · O view(stretch↔natural) · V eye · [ ] zoom · F flip · 2 = off")
        self._u_hud.adjustSize()
        self._u_hud.move(16, 16)
        self._u_hud.show()
        self._u_hud.raise_()

    def cycle_unwarp_eye(self):
        if not self._unwarp_on:
            return
        order = ['left', 'right', 'top', 'bottom', 'mono']
        i = order.index(self._u_eye) if self._u_eye in order else 0
        self._u_eye = order[(i + 1) % len(order)]
        self._rebuild_mesh()

    def cycle_unwarp_projection(self):
        if not self._unwarp_on:
            return
        order = [vu.PROJ_EQUIRECT_180, vu.PROJ_EQUIRECT_360, vu.PROJ_FISHEYE]
        i = order.index(self._u_proj) if self._u_proj in order else 0
        self._u_proj = order[(i + 1) % len(order)]
        self._rebuild_mesh()

    def _max_unwarp_fov(self):
        # Rectilinear blows up toward 180° (tan → ∞), so cap it well below. The
        # stereographic projection has NO singularity until 360°, so it can show
        # a full 180° comfortably (and beyond, for 200–220° fisheye lenses).
        return 150.0 if self._u_outproj == vu.OUT_RECTILINEAR else 220.0

    def adjust_unwarp_fov(self, delta):
        if not self._unwarp_on:
            return
        self._u_hfov = max(30.0, min(self._max_unwarp_fov(), self._u_hfov + delta))
        self._rebuild_mesh()

    def cycle_unwarp_output(self):
        """Toggle the OUTPUT projection: rectilinear (straight lines, edges
        stretch) ↔ stereographic (no edge stretch, lines bow slightly)."""
        if not self._unwarp_on:
            return
        order = [vu.OUT_RECTILINEAR, vu.OUT_STEREOGRAPHIC]
        i = order.index(self._u_outproj) if self._u_outproj in order else 0
        self._u_outproj = order[(i + 1) % len(order)]
        # Switching to rectilinear may require pulling a wide FOV back under cap.
        self._u_hfov = min(self._u_hfov, self._max_unwarp_fov())
        self._rebuild_mesh()

    def toggle_unwarp_flip(self):
        if not self._unwarp_on:
            return
        self._u_flip = not self._u_flip
        self._rebuild_mesh()

    # ── helpers ────────────────────────────────────────────────────────────────
    def _make_btn(self, text, slot, style) -> QPushButton:
        b = QPushButton(text, self)
        b.setStyleSheet(style)
        b.setCursor(Qt.CursorShape.PointingHandCursor)
        b.setFocusPolicy(Qt.FocusPolicy.NoFocus)   # keep key events on the player
        b.clicked.connect(slot)
        return b

    @staticmethod
    def _fmt(ms: int) -> str:
        s = int(max(0, ms) // 1000)
        h, rem = divmod(s, 3600)
        m, sec = divmod(rem, 60)
        return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"

    def _update_time(self):
        self._time.setText(f"{self._fmt(self._player.position())} / {self._fmt(self._duration_ms)}")

    # ── media signals ────────────────────────────────────────────────────────
    def _on_position_changed(self, pos: int):
        if not self._seeking:
            self._slider.blockSignals(True)
            self._slider.setValue(pos)
            self._slider.blockSignals(False)
        self._update_time()
        if self._readahead is not None:
            self._readahead.update(pos, self._duration_ms)

    def _on_duration_changed(self, dur: int):
        self._duration_ms = dur
        self._slider.setRange(0, max(0, dur))
        self._update_time()
        if self._readahead is not None:
            self._readahead.update(self._player.position(), dur)

    def _on_state_changed(self, state):
        playing = state == QMediaPlayer.PlaybackState.PlayingState
        self._btn_play.setText("⏸" if playing else "▶")

    def _on_error(self, error, error_string: str):
        log.error("fullscreen player error: %s (%s)", error_string, error)
        self._name.setText(f"⚠ Cannot play this file: {error_string}")

    # ── seek slider interaction ────────────────────────────────────────────────
    def _on_seek_pressed(self):
        self._seeking = True

    def _on_seek_moved(self, value: int):
        # live scrub: update the time readout while dragging
        self._time.setText(f"{self._fmt(value)} / {self._fmt(self._duration_ms)}")

    def _on_seek_released(self):
        self._player.setPosition(self._slider.value())
        self._seeking = False

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # The un-warp mesh fits the viewport aspect → recompute it on resize.
        if self._unwarp_on:
            self._rebuild_mesh()

    # ── input ──────────────────────────────────────────────────────────────────
    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key.Key_Escape:
            self.close_player()
        elif key == Qt.Key.Key_Space:
            self.toggle_play()
        elif key == Qt.Key.Key_Left:
            self.skip(_SKIP_BACK_10)
        elif key == Qt.Key.Key_Right:
            self.skip(_SKIP_FWD_10)
        elif key == Qt.Key.Key_2:
            self._on_2d_button()                 # toggle 2D / cycle crop
        elif self._unwarp_on and key == Qt.Key.Key_V:
            self.cycle_unwarp_eye()
        elif self._unwarp_on and key == Qt.Key.Key_P:
            self.cycle_unwarp_projection()
        elif self._unwarp_on and key == Qt.Key.Key_O:
            self.cycle_unwarp_output()
        elif self._unwarp_on and key == Qt.Key.Key_F:
            self.toggle_unwarp_flip()
        elif self._unwarp_on and key == Qt.Key.Key_BracketLeft:
            self.adjust_unwarp_fov(-5)
        elif self._unwarp_on and key == Qt.Key.Key_BracketRight:
            self.adjust_unwarp_fov(+5)
        elif key == Qt.Key.Key_V:
            self.cycle_crop()                    # fallback: eye-crop on graphics
        else:
            super().keyPressEvent(event)

    def mouseDoubleClickEvent(self, event):
        # Double-click to exit, symmetric with double-click-to-enter.
        self.close_player()
        super().mouseDoubleClickEvent(event)
