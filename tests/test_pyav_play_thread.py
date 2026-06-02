"""
Tests for pyav_play_thread — the PyAV-backed player thread that gives us
working HW decode on Windows where OpenCV's bundled FFmpeg doesn't.

These tests run with `av` mocked so they don't require PyAV to actually
be installed at test time.  When PyAV IS installed, a final integration
test exercises the import path.
"""
import sys
import platform
from unittest.mock import MagicMock, patch

import pytest
import numpy as np


# ── HW method ordering by platform ──────────────────────────────────────────
class TestHwMethodOrdering:
    def test_windows_prefers_d3d11va(self):
        import pyav_play_thread as p
        with patch.object(platform, 'system', return_value='Windows'):
            methods = p._hw_methods()
        # d3d11va FIRST — it's the most reliable on Windows pip av
        assert methods[0] == 'd3d11va'
        assert 'cuda' in methods           # NVDEC for NVIDIA users
        assert 'dxva2' in methods           # legacy fallback

    def test_linux_uses_vaapi_first(self):
        import pyav_play_thread as p
        with patch.object(platform, 'system', return_value='Linux'):
            methods = p._hw_methods()
        assert methods[0] == 'vaapi'

    def test_macos_uses_videotoolbox(self):
        import pyav_play_thread as p
        with patch.object(platform, 'system', return_value='Darwin'):
            methods = p._hw_methods()
        assert methods == ['videotoolbox']

    def test_unknown_platform_returns_empty(self):
        import pyav_play_thread as p
        with patch.object(platform, 'system', return_value='Haiku'):
            assert p._hw_methods() == []


# ── availability gate ───────────────────────────────────────────────────────
class TestAvailability:
    def test_is_available_returns_pyav_import_status(self):
        import pyav_play_thread as p
        # Whatever HAS_PYAV is, is_available() should match
        assert p.is_available() == p.HAS_PYAV

    def test_import_error_string_present_when_unavailable(self):
        """If HAS_PYAV is False, import_error() returns the captured msg."""
        import pyav_play_thread as p
        if not p.HAS_PYAV:
            assert p.import_error() != ""
        else:
            assert p.import_error() == ""


# ── constructor refuses without PyAV ─────────────────────────────────────────
def test_constructor_raises_when_pyav_missing():
    """Caller must be able to detect via is_available() rather than catching
    a confusing later error.  The constructor's runtime check is a safety
    net for code that forgets."""
    import pyav_play_thread as p

    with patch.object(p, 'HAS_PYAV', False):
        with pytest.raises(RuntimeError, match="PyAV is not installed"):
            p._PyAVPlayThread("x.mp4", 0.0, 640, 360)


# ── HW open + fallback ──────────────────────────────────────────────────────
# These are pure-Python tests of the open logic.  We mock `av` since we
# can't easily fabricate a real video file for a unit test.

@pytest.fixture
def pyav_module_present():
    """Inject a stub `av` module into pyav_play_thread for the duration
    of the test, and pretend HAS_PYAV is True."""
    import pyav_play_thread as p

    fake_av = MagicMock()
    # Fabricate the namespaces the code dereferences
    fake_av.error.FFmpegError = type('FFmpegError', (Exception,), {})
    fake_av.codec.hwaccel.HWAccel = MagicMock()

    with patch.object(p, 'av', fake_av), \
         patch.object(p, 'HAS_PYAV', True):
        yield fake_av, p


def _make_fake_stream(time_base=1/90000, rate=30):
    """Build a fake av.video stream object."""
    s = MagicMock()
    s.time_base = time_base
    s.average_rate = rate
    s.codec_context = MagicMock()
    s.codec_context.thread_count = 0
    s.codec_context.thread_type = 'AUTO'
    return s


def _make_fake_container(stream):
    c = MagicMock()
    c.streams.video = [stream]
    return c


def test_hw_open_uses_pyav_17_hwaccel_kwarg(pyav_module_present):
    """We pass `hwaccel=HWAccel(...)` to av.open() — that's the PyAV 17+
    API.  Earlier approaches that set codec_context.hwaccel manually
    don't get allow_software_fallback semantics."""
    fake_av, p = pyav_module_present

    stream = _make_fake_stream()
    container = _make_fake_container(stream)

    packet = MagicMock()
    packet.decode.return_value = [MagicMock()]
    container.demux.return_value = iter([packet])

    fake_av.open.return_value = container
    fake_av.codec.hwaccel.HWAccel.return_value = "the-hw-handle"

    t = p._PyAVPlayThread.__new__(p._PyAVPlayThread)
    t._path = "v.mkv"
    t._stop_flag = False

    cont, strm = t._open_with_hw('d3d11va')

    # Verify HWAccel was constructed with allow_software_fallback=True
    fake_av.codec.hwaccel.HWAccel.assert_called_once_with(
        device_type='d3d11va',
        allow_software_fallback=True,
    )
    # Verify av.open() was called with hwaccel=HWAccel AND the spin-up-
    # tolerant (open, read) timeout tuple (native path open).
    fake_av.open.assert_called_with(
        'v.mkv', hwaccel='the-hw-handle',
        timeout=(p._PYAV_OPEN_TIMEOUT_S, p._PYAV_READ_TIMEOUT_S))
    # Container rewound after probe
    container.seek.assert_called_with(0, stream=stream, backward=True)


def test_hw_open_raises_when_probe_yields_no_frames(pyav_module_present):
    """If HW init fails, PyAV typically raises during decode.  We simulate
    by yielding no decodable frames — caller will move to the next HW
    method or fall back to SW."""
    fake_av, p = pyav_module_present

    stream = _make_fake_stream()
    container = _make_fake_container(stream)
    container.demux.return_value = iter([])      # no packets

    fake_av.open.return_value = container

    t = p._PyAVPlayThread.__new__(p._PyAVPlayThread)
    t._path = "v.mkv"
    t._stop_flag = False

    with pytest.raises(Exception):
        t._open_with_hw('d3d11va')
    # Container was closed on failure
    container.close.assert_called()


def test_open_container_falls_through_hw_attempts_then_sw(pyav_module_present):
    """The full open() pipeline: HW attempt fails → try next HW → finally SW."""
    fake_av, p = pyav_module_present

    sw_stream = _make_fake_stream()
    sw_container = _make_fake_container(sw_stream)
    fake_av.open.return_value = sw_container

    t = p._PyAVPlayThread.__new__(p._PyAVPlayThread)
    t._path = "v.mkv"
    t._stop_flag = False
    t._hw_accel = True
    t.active_hwaccel = None
    # _configure_codec_sw reads these; set them so attribute lookup doesn't
    # hit sip's "super __init__ not called" RuntimeError on this __new__'d
    # instance (the method swallows that, leaving thread_count unset → 0).
    t._thread_type = 'SLICE'
    t._skip_frame = None

    # Force every HW attempt to fail
    with patch.object(t, '_open_with_hw', side_effect=RuntimeError("nope")):
        container, stream = t._open_container()

    assert container is sw_container
    assert t.active_hwaccel is None  # SW path
    # SW path configured threads to 75% of cores (or user override).
    # On a 1-core box this is 1; on a 32-core box it's 24.  Either way
    # it should be > 0. With no explicit _thread_type on this bare instance,
    # _configure_codec_sw defaults to 'SLICE' (the seek-friendly mode).
    assert sw_stream.codec_context.thread_count >= 1
    assert sw_stream.codec_context.thread_type == 'SLICE'


def test_open_container_with_hw_accel_disabled_skips_hw(pyav_module_present):
    fake_av, p = pyav_module_present

    stream = _make_fake_stream()
    container = _make_fake_container(stream)
    fake_av.open.return_value = container

    t = p._PyAVPlayThread.__new__(p._PyAVPlayThread)
    t._path = "v.mkv"
    t._stop_flag = False
    t._hw_accel = False
    t.active_hwaccel = None

    # _open_with_hw should NEVER be called when hw_accel=False
    with patch.object(t, '_open_with_hw') as hw:
        t._open_container()
        hw.assert_not_called()


# ── frame conversion ─────────────────────────────────────────────────────────
def test_frame_to_qimage_uses_reformat_for_scaling(pyav_module_present):
    fake_av, p = pyav_module_present

    # PyAV frame whose reformat yields a 400×225 RGB array
    arr = np.zeros((225, 400, 3), dtype=np.uint8)
    reformatted = MagicMock()
    reformatted.to_ndarray.return_value = arr

    frame = MagicMock()
    frame.width  = 7680
    frame.height = 4320
    frame.reformat.return_value = reformatted

    t = p._PyAVPlayThread.__new__(p._PyAVPlayThread)
    t._disp_w = 400
    t._disp_h = 225

    qimg = t._frame_to_qimage(frame)
    assert qimg is not None
    assert qimg.width()  == 400
    assert qimg.height() == 225
    # reformat called with target size (aspect-preserved)
    frame.reformat.assert_called_once()
    kw = frame.reformat.call_args.kwargs
    assert kw['width']  == 400
    # 7680:4320 = 16:9, so 400 wide → 225 tall
    assert kw['height'] == 225
    assert kw['format'] == 'rgb24'


def test_frame_to_qimage_no_upscale(pyav_module_present):
    """Small frame on a big display: shouldn't be upscaled at decode time."""
    fake_av, p = pyav_module_present

    arr = np.zeros((100, 200, 3), dtype=np.uint8)
    reformatted = MagicMock()
    reformatted.to_ndarray.return_value = arr

    frame = MagicMock()
    frame.width = 200
    frame.height = 100
    frame.reformat.return_value = reformatted

    t = p._PyAVPlayThread.__new__(p._PyAVPlayThread)
    t._disp_w = 2000
    t._disp_h = 2000

    qimg = t._frame_to_qimage(frame)
    assert qimg is not None
    kw = frame.reformat.call_args.kwargs
    assert kw['width']  == 200    # original size — no upscale
    assert kw['height'] == 100


def test_frame_to_qimage_returns_none_on_zero_display_size(pyav_module_present):
    fake_av, p = pyav_module_present
    frame = MagicMock(); frame.width = 100; frame.height = 100
    t = p._PyAVPlayThread.__new__(p._PyAVPlayThread)
    t._disp_w = 0
    t._disp_h = 0
    assert t._frame_to_qimage(frame) is None


# ── seek timestamp math ──────────────────────────────────────────────────────
def test_seek_uses_pts_in_stream_timebase(pyav_module_present):
    fake_av, p = pyav_module_present

    container = MagicMock()
    stream = _make_fake_stream(time_base=1/90000)

    t = p._PyAVPlayThread.__new__(p._PyAVPlayThread)
    t._seek_container(container, stream, 5.5)

    # 5.5s @ 1/90000 = 495000 PTS
    container.seek.assert_called_once_with(495000, stream=stream, backward=True)


def test_seek_fallback_to_microseconds_when_no_timebase(pyav_module_present):
    fake_av, p = pyav_module_present
    container = MagicMock()
    stream = _make_fake_stream(time_base=None)

    t = p._PyAVPlayThread.__new__(p._PyAVPlayThread)
    t._seek_container(container, stream, 2.0)
    container.seek.assert_called_once_with(2_000_000, stream=stream, backward=True)


def test_seek_handles_exception_quietly(pyav_module_present):
    """A failed seek must not propagate — playback continues from current position."""
    fake_av, p = pyav_module_present
    container = MagicMock()
    container.seek.side_effect = RuntimeError("bad seek")
    stream = _make_fake_stream()

    t = p._PyAVPlayThread.__new__(p._PyAVPlayThread)
    t._seek_container(container, stream, 3.0)   # must not raise


# ── integration: when PyAV is actually installed ─────────────────────────────
@pytest.mark.skipif(
    'av' not in sys.modules and not __import__('importlib.util').util.find_spec('av'),
    reason="PyAV not installed in this test environment",
)
def test_real_pyav_module_imports_cleanly():
    """If `av` is in the environment, our module must import without error
    and is_available() must return True."""
    import pyav_play_thread as p
    assert p.HAS_PYAV is True
    assert p.is_available() is True


# ── integration: video_player_widget falls back gracefully ──────────────────
def test_video_player_widget_imports_with_or_without_pyav():
    """The player widget must not crash on import whether PyAV is
    available or not — the fallback path always exists."""
    import importlib
    import video_player_widget
    importlib.reload(video_player_widget)
    # Either _PyAVPlayThread is a real class or None; both are valid
    assert (video_player_widget._PyAVPlayThread is None
            or callable(video_player_widget._PyAVPlayThread))
    # _pyav_available is always callable
    assert callable(video_player_widget._pyav_available)
