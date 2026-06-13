r"""
Standalone preview for the VR → flat un-warp (Phase 2 walking skeleton).

Run it on ONE file to validate the un-warp rendering pipeline before it's folded
into the full-screen player:

    python vr_unwarp_preview.py "D:\Videos\somefile.mp4"

An on-screen overlay (top-left) shows the current settings + live status. Tune
live with the keyboard:

    V : cycle eye   (mono → left → right → top → bottom)
    P : cycle projection (equirect180 → equirect360 → fisheye)
    [ / ] : FOV  -5° / +5°       - / = : lens FOV (fisheye) -5° / +5°
    F : flip vertical (if the image is upside-down)
    Space : play / pause          Esc : quit

Tell me which settings give a correct flat image and I'll bake them in as the
auto-detected defaults in the player.
"""
import os
import sys

os.environ.setdefault('QT_MEDIA_BACKEND', 'ffmpeg')   # HW decode + HEVC

from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel
from PyQt6.QtQuickWidgets import QQuickWidget
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtCore import Qt, QUrl, QObject, QTimer, QEvent

import vr_unwarp as vu
from vr_unwarp_mesh import UnwarpGeometry

_HERE = os.path.dirname(os.path.abspath(__file__))
_EYES = ['mono', 'left', 'right', 'top', 'bottom']
_PROJS = [vu.PROJ_EQUIRECT_180, vu.PROJ_EQUIRECT_360, vu.PROJ_FISHEYE]

_HUD_STYLE = (
    "QLabel { color:#eaeaea; background: rgba(0,0,0,180); padding:10px 14px;"
    " border-radius:6px; font-family:Consolas,monospace; font-size:13px; }"
)


def _detect_eye(path):
    try:
        from fullscreen_player import detect_vr_layout
        m = detect_vr_layout(path)            # 'off'/'left'/'right'/'top'/'bottom'
        return 'mono' if m == 'off' else m
    except Exception:
        return 'left'


def _fmt(ms):
    s = max(0, int(ms)) // 1000
    return f"{s // 60}:{s % 60:02d}"


class PreviewWindow(QWidget):
    def __init__(self, path):
        super().__init__()
        self._path = path
        self._ok = False
        self.setWindowTitle("VR → 2D preview")
        self.setStyleSheet("background:#000;")
        self.resize(1280, 720)

        self._hud = QLabel(self)
        self._hud.setStyleSheet(_HUD_STYLE)
        self._hud.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._hud.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        self._hud.move(16, 16)
        self._hud.setText("Loading…")
        self._hud.raise_()

        self._qw = QQuickWidget(self)
        self._qw.setResizeMode(QQuickWidget.ResizeMode.SizeRootObjectToView)
        self._qw.lower()
        self._qw.setSource(QUrl.fromLocalFile(os.path.join(_HERE, 'vr_unwarp_view.qml')))
        if self._qw.status() == QQuickWidget.Status.Error:
            return self._fail("QML failed to load:\n" +
                              "\n".join(e.toString() for e in self._qw.errors()))
        self._root = self._qw.rootObject()
        if self._root is None:
            return self._fail("QML root is null — Qt Quick 3D may be unavailable on this GPU.")

        self._geo = UnwarpGeometry()
        model = self._root.findChild(QObject, 'unwarpModel')
        vout = self._root.findChild(QObject, 'videoOut')
        if model is None or vout is None:
            return self._fail("Could not find the un-warp Model / VideoOutput in the QML scene.")
        model.setProperty('geometry', self._geo)

        self._proj = vu.detect_projection(path)
        self._eye = _detect_eye(path)
        self._lens = vu.detect_lens_fov(path)
        self._hfov = 90.0
        self._flipv = True       # Qt Quick 3D texture origin is bottom-left → flip

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._qw)

        self._player = QMediaPlayer(self)
        self._audio = QAudioOutput(self)
        self._audio.setVolume(0.8)
        self._player.setAudioOutput(self._audio)
        self._player.setVideoOutput(vout)
        self._player.errorOccurred.connect(lambda _e, s: self._set_status(f"PLAYER ERROR: {s}"))
        self._player.setSource(QUrl.fromLocalFile(path))
        self._ok = True
        self._apply()
        self._player.play()

        # Route keys via an app-wide filter so they work even when the 3D widget
        # has focus; refresh the status line ~3×/s.
        QApplication.instance().installEventFilter(self)
        self._timer = QTimer(self)
        self._timer.setInterval(300)
        self._timer.timeout.connect(self._refresh)
        self._timer.start()

    # ── overlay ────────────────────────────────────────────────────────────────
    def _apply(self):
        if not self._ok:
            return
        out_aspect = self.width() / max(1, self.height())
        self._geo.rebuild(projection=self._proj, hfov_deg=self._hfov,
                          out_aspect=out_aspect, lens_fov_deg=self._lens, eye=self._eye)
        self._root.setProperty('flipV', self._flipv)
        self._refresh()

    def _refresh(self):
        if not self._ok:
            return
        p = self._player
        st = {QMediaPlayer.PlaybackState.PlayingState: "Playing",
              QMediaPlayer.PlaybackState.PausedState: "Paused",
              QMediaPlayer.PlaybackState.StoppedState: "Stopped"}.get(p.playbackState(), "?")
        self._hud.setText(
            "VR → 2D preview — " + os.path.basename(self._path) + "\n"
            f"projection = {self._proj}\n"
            f"eye        = {self._eye}\n"
            f"FOV        = {self._hfov:.0f}°      lens FOV = {self._lens:.0f}°\n"
            f"flip V     = {self._flipv}\n"
            f"status     = {st}   {_fmt(p.position())} / {_fmt(p.duration())}\n"
            "──────────────────────────────\n"
            "V eye · P projection · [ ] FOV · - = lens · F flip · Space · Esc")
        self._hud.adjustSize()
        self._hud.raise_()

    def _set_status(self, msg):
        self._hud.setText(msg)
        self._hud.adjustSize()
        self._hud.raise_()

    def _fail(self, msg):
        sys.stderr.write("\n*** " + msg + "\n")
        self._hud.setStyleSheet(_HUD_STYLE.replace("#eaeaea", "#ff9090"))
        self._hud.setText("PREVIEW FAILED\n" + msg)
        self._hud.adjustSize()
        self._hud.raise_()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._hud.move(16, 16)
        if getattr(self, '_geo', None) is not None:
            self._apply()

    # ── keys (via app event filter) ─────────────────────────────────────────────
    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.KeyPress and self.isActiveWindow():
            if self._handle_key(event.key()):
                return True
        return super().eventFilter(obj, event)

    def _handle_key(self, k):
        if not self._ok and k != Qt.Key.Key_Escape:
            return False
        if k == Qt.Key.Key_Escape:
            self.close()
        elif k == Qt.Key.Key_Space:
            p = self._player
            (p.pause if p.playbackState() == QMediaPlayer.PlaybackState.PlayingState
             else p.play)()
        elif k == Qt.Key.Key_V:
            self._eye = _EYES[(_EYES.index(self._eye) + 1) % len(_EYES)]; self._apply()
        elif k == Qt.Key.Key_P:
            self._proj = _PROJS[(_PROJS.index(self._proj) + 1) % len(_PROJS)]; self._apply()
        elif k == Qt.Key.Key_BracketLeft:
            self._hfov = max(30.0, self._hfov - 5); self._apply()
        elif k == Qt.Key.Key_BracketRight:
            self._hfov = min(150.0, self._hfov + 5); self._apply()
        elif k == Qt.Key.Key_Minus:
            self._lens = max(100.0, self._lens - 5); self._apply()
        elif k == Qt.Key.Key_Equal:
            self._lens = min(260.0, self._lens + 5); self._apply()
        elif k == Qt.Key.Key_F:
            self._flipv = not self._flipv; self._apply()
        else:
            return False
        return True

    def closeEvent(self, e):
        try:
            self._player.stop()
        except Exception:
            pass
        super().closeEvent(e)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("ERROR: pass a video file path.")
        sys.exit(2)
    path = sys.argv[1]
    if not os.path.isfile(path):
        print(f"ERROR: not a file: {path}")
        sys.exit(2)
    app = QApplication(sys.argv)
    w = PreviewWindow(path)
    w.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
