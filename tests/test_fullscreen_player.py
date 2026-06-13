"""
Tests for the in-app full-screen video player (double-click → play in-app).

Logic is tested without real decoding: skip-clamp math, time formatting, that
the skip buttons carry the right ±10/±30s offsets, that play() sets the media
source, and that Esc / close emit `closed`. Real HW playback is exercised by
hand (it needs a display + a real file).
"""
import os
import sys

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('QT_MEDIA_BACKEND', 'ffmpeg')
os.environ.setdefault('VIDEO_ORGANIZER_QSETTINGS_ORG', 'VOrgTest')
os.environ.setdefault('VIDEO_ORGANIZER_QSETTINGS_APP', 'FullscreenPlayerTest')

import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


@pytest.fixture(scope='module')
def qapp():
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication(sys.argv)


@pytest.fixture
def player(qapp):
    from fullscreen_player import FullscreenVideoPlayer
    p = FullscreenVideoPlayer()
    yield p
    try:
        p.stop()
    except Exception:
        pass
    p.deleteLater()
    qapp.processEvents()


# ── pure logic (no Qt object needed) ─────────────────────────────────────────
def test_skip_clamp_math():
    from fullscreen_player import FullscreenVideoPlayer as F
    assert F._clamp_target(5_000, 10_000, 60_000) == 15_000     # normal forward
    assert F._clamp_target(5_000, -10_000, 60_000) == 0         # clamp at start
    assert F._clamp_target(55_000, 30_000, 60_000) == 60_000    # clamp at end
    assert F._clamp_target(0, -30_000, 60_000) == 0             # already at start
    assert F._clamp_target(5_000, 10_000, 0) == 15_000          # unknown duration → no high clamp


def test_time_formatting():
    from fullscreen_player import FullscreenVideoPlayer as F
    assert F._fmt(0) == "0:00"
    assert F._fmt(65_000) == "1:05"
    assert F._fmt(3_661_000) == "1:01:01"
    assert F._fmt(-500) == "0:00"


# ── widget behaviour ─────────────────────────────────────────────────────────
def test_controls_exist(player):
    for attr in ('_btn_b30', '_btn_b10', '_btn_play', '_btn_f10', '_btn_f30',
                 '_btn_close', '_slider', '_video', '_player', '_audio'):
        assert getattr(player, attr) is not None, f"missing {attr}"


def test_skip_buttons_use_correct_offsets(player, monkeypatch):
    """The 4 skip buttons map to back-30, back-10, fwd-10, fwd-30 seconds."""
    calls = []
    monkeypatch.setattr(player, 'skip', lambda ms: calls.append(ms))
    player._btn_b30.click()
    player._btn_b10.click()
    player._btn_f10.click()
    player._btn_f30.click()
    assert calls == [-30_000, -10_000, 10_000, 30_000]


def test_play_sets_source_and_stop_clears_it(player, tmp_path):
    f = tmp_path / "clip.mp4"
    f.write_bytes(b"\x00" * 64)
    player.play(str(f))
    assert player._player.source().toLocalFile().endswith("clip.mp4")
    assert player._name.text() == "clip.mp4"
    player.stop()
    assert player._player.source().isEmpty()


def test_close_emits_closed(player):
    fired = []
    player.closed.connect(lambda: fired.append(True))
    player.close_player()
    assert fired == [True]


def test_escape_key_closes(player):
    from PyQt6.QtGui import QKeyEvent
    from PyQt6.QtCore import QEvent, Qt
    fired = []
    player.closed.connect(lambda: fired.append(True))
    ev = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier)
    player.keyPressEvent(ev)
    assert fired == [True]


def test_volume_slider_drives_audio(player):
    player._vol.setValue(50)
    assert abs(player._audio.volume() - 0.5) < 0.01


def test_readahead_target_window_math():
    """Byte-offset estimate for read-ahead: position/duration × size, with the
    ahead window capped, and None when size/duration are unknown."""
    from fullscreen_player import _ReadAheadCache as R
    # 1,000,000 bytes / 100,000 ms = 10 bytes/ms.
    assert R._target_window(0, 100_000, 1_000_000, 45, 10**12) == (0, 450_000)
    assert R._target_window(10_000, 100_000, 1_000_000, 45, 10**12) == (100_000, 450_000)
    assert R._target_window(200_000, 100_000, 1_000_000, 45, 10**12)[0] == 1_000_000  # cur clamps to size
    assert R._target_window(0, 100_000, 1_000_000, 45, 100_000) == (0, 100_000)        # ahead capped
    assert R._target_window(0, 0, 1_000_000, 45, 10**12) is None                       # duration unknown
    assert R._target_window(0, 100_000, 0, 45, 10**12) is None                         # size unknown


def test_readahead_lifecycle_starts_and_stops_cleanly(tmp_path):
    import time
    from fullscreen_player import _ReadAheadCache
    f = tmp_path / "big.bin"
    f.write_bytes(b"\x00" * (3 * 1024 * 1024))
    ra = _ReadAheadCache(str(f))
    ra.update(0, 10_000)            # known duration → it will read
    ra.start()
    time.sleep(0.05)                # let it do at least one read
    ra.stop()                       # signals + joins (timeout 2s)
    assert not ra._thread.is_alive(), "read-ahead thread must stop + join cleanly"


def test_play_starts_readahead_and_stop_clears_it(player, tmp_path):
    f = tmp_path / "clip.mp4"
    f.write_bytes(b"\x00" * (2 * 1024 * 1024))
    player.play(str(f))
    assert player._readahead is not None, "play() should start read-ahead"
    player.stop()
    assert player._readahead is None, "stop() should tear down read-ahead"


def test_readahead_window_is_120s():
    from fullscreen_player import _ReadAheadCache
    assert _ReadAheadCache.AHEAD_SECONDS == 120


def test_readahead_ms_reports_buffered_extent(tmp_path):
    """read_ahead_ms() converts the worker's byte read-position back to ms via
    the uniform-bitrate estimate (for the on-screen buffered bar)."""
    from fullscreen_player import _ReadAheadCache
    f = tmp_path / "v.bin"
    f.write_bytes(b"\x00" * 1_000_000)       # exactly 1,000,000 bytes
    ra = _ReadAheadCache(str(f))             # not start()ed → no worker thread, no race
    assert ra.read_ahead_ms() == 0           # duration unknown → 0
    ra.update(0, 100_000)                    # 1,000,000 B / 100,000 ms = 10 B/ms
    with ra._lock:
        ra._read_pos = 500_000               # worker has read to byte 500k
    assert ra.read_ahead_ms() == 50_000      # → 50,000 ms buffered


def test_seek_slider_set_buffered_clamps(qapp):
    from fullscreen_player import _SeekSlider
    from PyQt6.QtCore import Qt
    s = _SeekSlider(Qt.Orientation.Horizontal)
    s.setRange(0, 1000)
    s.set_buffered(500);   assert s._buffered_ms == 500
    s.set_buffered(99_999); assert s._buffered_ms == 1000     # clamped to max
    s.set_buffered(-5);    assert s._buffered_ms == 0         # clamped to 0


def test_buffer_indicator_updates_slider(player):
    """The poll timer's handler pushes the read-ahead extent onto the slider's
    buffered bar."""
    player._slider.setRange(0, 100_000)

    class _Stub:
        def read_ahead_ms(self):
            return 60_000

    player._readahead = _Stub()
    player._update_buffer_indicator()
    assert player._slider._buffered_ms == 60_000


def test_double_click_opens_inapp_player_over_whole_window(qapp, tmp_path):
    """End-to-end: the grid's open_requested handler now expands an in-app
    player over the entire window instead of launching an external one."""
    from main_window import MainWindow
    f = tmp_path / "movie.mp4"
    f.write_bytes(b"\x00" * 64)
    w = MainWindow()
    try:
        w.resize(1000, 700)
        w._on_open_video(str(f))                       # the double-click result
        fs = getattr(w, "_fs_player", None)
        # isVisibleTo(parent): would be visible once the (unshown) window is —
        # i.e. show() was called — without needing to show the heavy window.
        assert fs is not None and fs.isVisibleTo(w), "in-app player should appear"
        assert fs.size() == w.size(), "player should cover the whole window"
        assert fs._player.source().toLocalFile().endswith("movie.mp4")
        w._on_fullscreen_closed()
        assert not fs.isVisibleTo(w), "closing returns to the grid"
    finally:
        w.close()
        qapp.processEvents()


# ── VR → 2D eye-crop ──────────────────────────────────────────────────────────
def test_detect_vr_layout():
    from fullscreen_player import detect_vr_layout, CROP_OFF, CROP_LEFT, CROP_TOP
    # side-by-side / fisheye-lens tags → show the LEFT eye
    assert detect_vr_layout("Studio_Scene_180_LR.mp4")      == CROP_LEFT
    assert detect_vr_layout("clip_MKX200_alpha.mp4")        == CROP_LEFT
    assert detect_vr_layout("foo_FISHEYE190.mp4")           == CROP_LEFT
    assert detect_vr_layout("bar_sbs.mp4")                  == CROP_LEFT
    assert detect_vr_layout(r"X:\VR\thing_180x180_3dh.mp4") == CROP_LEFT
    # top-bottom / over-under
    assert detect_vr_layout("movie_360_TB.mp4")             == CROP_TOP
    assert detect_vr_layout("movie_ou.mkv")                 == CROP_TOP
    # a VR projection with no explicit layout → assume SBS
    assert detect_vr_layout("scene_vr180.mp4")              == CROP_LEFT
    # mono / ordinary videos → no crop
    assert detect_vr_layout("something_mono.mp4")           == CROP_OFF
    assert detect_vr_layout("Family Vacation 2024.mp4")     == CROP_OFF


def test_crop_rect_for_math():
    from fullscreen_player import (FullscreenVideoPlayer as F,
                                   CROP_OFF, CROP_LEFT, CROP_RIGHT, CROP_TOP, CROP_BOTTOM)
    W, H = 8000.0, 4000.0
    assert F._crop_rect_for(CROP_OFF,    W, H) == (0.0,    0.0,    8000.0, 4000.0)
    assert F._crop_rect_for(CROP_LEFT,   W, H) == (0.0,    0.0,    4000.0, 4000.0)
    assert F._crop_rect_for(CROP_RIGHT,  W, H) == (4000.0, 0.0,    4000.0, 4000.0)
    assert F._crop_rect_for(CROP_TOP,    W, H) == (0.0,    0.0,    8000.0, 2000.0)
    assert F._crop_rect_for(CROP_BOTTOM, W, H) == (0.0,    2000.0, 8000.0, 2000.0)
    assert F._crop_rect_for(CROP_LEFT,   0,  0) is None        # size unknown


def test_cycle_crop_updates_mode(player):
    """cycle_crop drives the graphics-surface eye-crop (the fallback when the
    un-warp surface is unavailable). It always updates _crop_mode; the button
    label only mirrors it in that fallback case."""
    from fullscreen_player import (CROP_OFF, CROP_LEFT, CROP_RIGHT, CROP_TOP, CROP_BOTTOM)
    assert player._crop_mode == CROP_OFF
    for expected in (CROP_LEFT, CROP_RIGHT, CROP_TOP, CROP_BOTTOM, CROP_OFF):
        player.cycle_crop()
        assert player._crop_mode == expected
    if not player._unwarp_available:                 # fallback: button shows crop label
        player.set_crop_mode(CROP_LEFT)
        assert player._btn_2d.text() == '2D: L'


# ── VR → 2D un-warp integration (Phase 2) ─────────────────────────────────────
def _need_unwarp(player):
    import pytest
    if not player._unwarp_available:
        pytest.skip("Qt Quick 3D un-warp surface not available in this environment")


def test_unwarp_toggle_swaps_surface(player, qapp):
    _need_unwarp(player)
    player.resize(1280, 720); player.show(); qapp.processEvents()
    assert player._stack.count() == 2
    assert player._unwarp_on is False
    player.set_unwarp(True)
    assert player._unwarp_on and player._stack.currentWidget() is player._qml
    assert player._u_geo.vertex_count() > 0
    player.set_unwarp(False)
    assert (not player._unwarp_on) and player._stack.currentWidget() is player._video
    player.hide()


def test_unwarp_cycle_controls(player):
    _need_unwarp(player)
    player.set_unwarp(True)
    player._u_eye, player._u_proj = 'left', 'equirect180'
    player.cycle_unwarp_eye();        assert player._u_eye == 'right'
    player.cycle_unwarp_projection(); assert player._u_proj == 'equirect360'
    f0 = player._u_hfov
    player.adjust_unwarp_fov(-5);     assert player._u_hfov == f0 - 5
    flip0 = player._u_flip
    player.toggle_unwarp_flip();      assert player._u_flip is (not flip0)


def test_play_vr_file_auto_enables_unwarp(player, qapp, tmp_path):
    _need_unwarp(player)
    f = str(tmp_path / "scene_180_MKX200_LR.mp4")
    open(f, 'wb').write(b'\x00' * 2048)
    player.play(f); qapp.processEvents()
    assert player._unwarp_on is True
    assert player._u_proj == 'fisheye' and player._u_eye == 'left' and player._u_lens == 200.0
    player.stop()


def test_play_normal_file_keeps_unwarp_off(player, qapp, tmp_path):
    _need_unwarp(player)
    f = str(tmp_path / "Family_Movie_2024.mp4")
    open(f, 'wb').write(b'\x00' * 2048)
    player.play(f); qapp.processEvents()
    assert player._unwarp_on is False
    player.stop()


def test_unwarp_defaults(player):
    # The look the user settled on: stereographic view, 160° FOV, flipped
    # (Quick3D texture is upside-down by default).
    import vr_unwarp as vu
    assert player._u_flip is True
    assert player._u_outproj == vu.OUT_STEREOGRAPHIC
    assert player._u_hfov == 220.0


def test_cycle_unwarp_output_toggles(player):
    _need_unwarp(player)
    import vr_unwarp as vu
    player.set_unwarp(True)
    start = player._u_outproj
    player.cycle_unwarp_output()
    assert player._u_outproj != start
    assert player._u_outproj in (vu.OUT_RECTILINEAR, vu.OUT_STEREOGRAPHIC)


def test_fov_cap_is_projection_aware(player):
    _need_unwarp(player)
    import vr_unwarp as vu
    player.set_unwarp(True)
    # stereographic has no 180° singularity → it can go past 180 (up to 220)
    player._u_outproj = vu.OUT_STEREOGRAPHIC
    player._u_hfov = 175.0
    player.adjust_unwarp_fov(10)
    assert player._u_hfov == 185.0
    # rectilinear caps well below 180 (it blows up there)
    player._u_outproj = vu.OUT_RECTILINEAR
    player._u_hfov = 145.0
    player.adjust_unwarp_fov(20)
    assert player._u_hfov == 150.0
    # switching stereographic→rectilinear pulls a too-wide FOV back under the cap
    player._u_outproj = vu.OUT_STEREOGRAPHIC
    player._u_hfov = 200.0
    player.cycle_unwarp_output()
    assert player._u_outproj == vu.OUT_RECTILINEAR and player._u_hfov <= 150.0


def test_vr_eye_for_aspect():
    """Frame-shape VR detection: ~2:1 high-res = side-by-side; everything else
    (incl. 21:9 / 2.39 cinema) is NOT VR, so normal videos are never flattened."""
    from fullscreen_player import FullscreenVideoPlayer as F
    assert F._vr_eye_for_aspect(7680, 3840) == 'left'    # 2:1 8K SBS
    assert F._vr_eye_for_aspect(3840, 1920) == 'left'    # 2:1 4K SBS
    assert F._vr_eye_for_aspect(1920, 1080) is None      # 16:9 normal
    assert F._vr_eye_for_aspect(3840, 2160) is None      # 16:9 4K
    assert F._vr_eye_for_aspect(4096, 1716) is None      # 2.39 cinema
    assert F._vr_eye_for_aspect(2560, 1080) is None      # 21:9 ultrawide
    assert F._vr_eye_for_aspect(3000, 1500) is None      # 2:1 but too low-res
    assert F._vr_eye_for_aspect(0, 0) is None


def test_auto_unwarp_by_aspect(player):
    _need_unwarp(player)
    from PyQt6.QtCore import QSizeF
    player._auto_unwarp_done = False
    player._user_toggled_2d = False
    player._unwarp_on = False
    player._native = QSizeF(7680, 3840)                  # 2:1 8K → SBS VR
    player._maybe_auto_unwarp_by_aspect()
    assert player._unwarp_on and player._u_eye == 'left'


def test_auto_unwarp_respects_manual_toggle(player):
    _need_unwarp(player)
    from PyQt6.QtCore import QSizeF
    player._auto_unwarp_done = False
    player._user_toggled_2d = True                       # user took manual control
    player._unwarp_on = False
    player._native = QSizeF(7680, 3840)
    player._maybe_auto_unwarp_by_aspect()
    assert player._unwarp_on is False                    # must NOT override the user


def test_auto_unwarp_skips_normal_video(player):
    _need_unwarp(player)
    from PyQt6.QtCore import QSizeF
    player._auto_unwarp_done = False
    player._user_toggled_2d = False
    player._unwarp_on = False
    player._native = QSizeF(1920, 1080)                  # normal 16:9
    player._maybe_auto_unwarp_by_aspect()
    assert player._unwarp_on is False


def test_play_auto_selects_eye_crop_from_filename(player, tmp_path):
    """play() picks the eye-crop from the filename BEFORE decoding starts, so a
    VR file opens already flattened to one eye."""
    from fullscreen_player import CROP_LEFT
    f = tmp_path / "scene_180_LR.mp4"
    f.write_bytes(b"\x00" * 1024)          # not a real video; crop is set pre-decode
    player.play(str(f))
    assert player._crop_mode == CROP_LEFT
    player.stop()


def test_apply_crop_clips_and_centres_the_eye(player, qapp):
    """Deterministic check that _apply_crop actually configures the scene: the
    clip item is narrowed to the chosen eye, and the transform fits that eye
    into the viewport, centred + aspect-preserved. (Verifies the geometry the
    real bug was in, without needing to render pixels.)"""
    from fullscreen_player import CROP_LEFT, CROP_OFF
    from PyQt6.QtCore import QPointF, QSizeF

    player.resize(1920, 1080)
    player.show()
    qapp.processEvents()
    vp = player._video.viewport()
    vw, vh = vp.width(), vp.height()
    assert vw > 0 and vh > 0, "viewport must have a real size for the geometry test"

    # Pretend an 8000×4000 side-by-side frame arrived (each eye 4000×4000).
    player._native = QSizeF(8000.0, 4000.0)
    player.set_crop_mode(CROP_LEFT)        # → _apply_crop

    # 1) the clip is narrowed to exactly the LEFT eye …
    r = player._clip.rect()
    assert (r.x(), r.y(), r.width(), r.height()) == (0.0, 0.0, 4000.0, 4000.0)

    # 2) … and the transform fits that eye into the viewport, centred.
    cw = ch = 4000.0
    s = min(vw / cw, vh / ch)
    ox, oy = (vw - cw * s) / 2.0, (vh - ch * s) / 2.0
    t = player._clip.transform()
    p0, p1 = t.map(QPointF(0.0, 0.0)), t.map(QPointF(cw, ch))
    assert abs(p0.x() - ox) < 1.0 and abs(p0.y() - oy) < 1.0
    assert abs(p1.x() - (ox + cw * s)) < 1.0 and abs(p1.y() - (oy + ch * s)) < 1.0
    # the eye fills the limiting viewport dimension (no scene gap beyond letterbox)
    assert abs((p1.x() - p0.x()) - cw * s) < 1.0
    assert min(p1.x() - p0.x(), p1.y() - p0.y()) >= min(vw, vh) - 2.0

    # CROP_OFF opens the clip back up to the whole frame.
    player.set_crop_mode(CROP_OFF)
    r = player._clip.rect()
    assert (r.width(), r.height()) == (8000.0, 4000.0)
    player.hide()

