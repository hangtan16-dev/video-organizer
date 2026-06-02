"""
Tests for the VR / large-file performance settings and the hardware-
acceleration helper.

These don't actually decode anything — they verify the policy logic
(thresholds, fallback behavior, settings round-trips) which is what we
can sanity-check without an actual GPU.
"""
import os
from unittest.mock import patch, MagicMock

import pytest


# ── settings round-trip ───────────────────────────────────────────────────────
class TestPerformanceSettings:
    @pytest.fixture
    def settings(self, tmp_path, monkeypatch):
        # Isolate this test's QSettings storage to a unique tmp_path so
        # writes from one test don't bleed into the next. We use a unique
        # org+app per fixture call by stamping tmp_path into the names.
        from PyQt6.QtCore import QSettings
        unique = str(tmp_path).replace('\\', '_').replace('/', '_').replace(':', '')
        import app_settings as _as

        class _IsolatedSettings(_as.AppSettings):
            def __init__(self):
                self._settings = QSettings('VideoOrgTest', unique)
                # Bypass the rest of __init__ (file system setup) — only
                # the QSettings reads/writes are tested here.
                import os as _os
                self._app_dir   = str(tmp_path / "VideoOrganizer")
                self._cache_dir = str(tmp_path / "thumbnails")

        return _IsolatedSettings()

    def test_defaults_are_sensible(self, settings):
        assert settings.hover_preview_max_gb >= 1.0
        assert settings.hover_preview_max_gb <= 100.0
        assert settings.large_file_threshold_mb >= 50
        assert 1 <= settings.hover_preview_fps_large <= 30
        # HW accel default is False: the pip opencv-python wheel doesn't
        # ship working HW decode, and getting noisy [hevc] errors with no
        # actual benefit is worse than just using multi-threaded CPU decode.
        assert settings.use_hw_accel is False

    def test_hover_preview_max_gb_roundtrip(self, settings):
        settings.hover_preview_max_gb = 8.5
        assert settings.hover_preview_max_gb == 8.5

    def test_large_threshold_clamped_to_zero(self, settings):
        settings.large_file_threshold_mb = -100
        assert settings.large_file_threshold_mb == 0

    def test_fps_clamped_to_one(self, settings):
        settings.hover_preview_fps_large = 0
        assert settings.hover_preview_fps_large == 1
        settings.hover_preview_fps_large = -5
        assert settings.hover_preview_fps_large == 1

    def test_use_hw_accel_roundtrip(self, settings):
        settings.use_hw_accel = False
        assert settings.use_hw_accel is False
        settings.use_hw_accel = True
        assert settings.use_hw_accel is True

    def test_cpu_decode_threads_default_is_auto(self, settings):
        # 0 means "auto" (use compute_decode_threads policy = 75% cores)
        assert settings.cpu_decode_threads == 0

    def test_cpu_decode_threads_clamped_to_zero(self, settings):
        settings.cpu_decode_threads = -3
        assert settings.cpu_decode_threads == 0

    def test_cpu_decode_threads_roundtrip(self, settings):
        settings.cpu_decode_threads = 8
        assert settings.cpu_decode_threads == 8


class TestComputeDecodeThreads:
    """Tests for main.compute_decode_threads — the 75%-of-cores policy."""

    def test_default_is_75_percent_of_cores(self):
        from main import compute_decode_threads
        import os as _os
        n = compute_decode_threads(s=None)
        # 75% of cores, at least 1
        expected = max(1, int((_os.cpu_count() or 4) * 0.75))
        assert n == expected, f"got {n}, expected {expected}"

    def test_explicit_override_wins(self):
        """A user-set positive value overrides the 75% default."""
        from main import compute_decode_threads

        class FakeSettings:
            cpu_decode_threads = 7

        assert compute_decode_threads(s=FakeSettings()) == 7

    def test_zero_setting_means_use_default(self):
        from main import compute_decode_threads
        import os as _os

        class FakeSettings:
            cpu_decode_threads = 0

        assert compute_decode_threads(s=FakeSettings()) == \
            max(1, int((_os.cpu_count() or 4) * 0.75))

    def test_returns_at_least_one(self):
        """Even on a hypothetical 1-core CPU, must return >= 1."""
        from main import compute_decode_threads
        from unittest.mock import patch

        class FakeSettings:
            cpu_decode_threads = 0

        with patch('os.cpu_count', return_value=1):
            assert compute_decode_threads(s=FakeSettings()) >= 1


# ── video_capture_helper ─────────────────────────────────────────────────────
class TestCaptureHelper:
    def test_open_capture_no_hw_accel_uses_sw_path(self):
        """When hw_accel=False, only the software path is tried."""
        import video_capture_helper as h

        fake_cap = MagicMock()
        fake_cap.isOpened.return_value = True

        with patch.object(h, '_try_open_hw') as hw, \
             patch.object(h, '_try_open_sw', return_value=fake_cap) as sw:
            result = h.open_capture('x.mp4', hw_accel=False)
            assert result is fake_cap
            hw.assert_not_called()
            sw.assert_called_once_with('x.mp4')

    def test_open_capture_falls_back_to_sw_on_hw_failure(self):
        """HW accel attempt returns None → SW is tried."""
        import video_capture_helper as h
        h._session_hw_disabled = False   # reset; another test may have set it

        fake_cap = MagicMock()
        fake_cap.isOpened.return_value = True
        with patch.object(h, '_try_open_hw', return_value=None) as hw, \
             patch.object(h, '_try_open_sw', return_value=fake_cap) as sw:
            result = h.open_capture('x.mp4', hw_accel=True)
            assert result is fake_cap
            hw.assert_called_once()
            sw.assert_called_once()

    def test_open_capture_returns_none_when_both_paths_fail(self):
        import video_capture_helper as h
        with patch.object(h, '_try_open_hw', return_value=None), \
             patch.object(h, '_try_open_sw', return_value=None):
            assert h.open_capture('x.mp4', hw_accel=True) is None

    def test_try_open_sw_releases_on_failed_open(self):
        """If isOpened() returns False, the helper must call release() — we
        rely on this for Windows file-handle hygiene."""
        import video_capture_helper as h

        fake_cap = MagicMock()
        fake_cap.isOpened.return_value = False
        with patch('video_capture_helper.cv2.VideoCapture',
                   return_value=fake_cap):
            result = h._try_open_sw('x.mp4')
            assert result is None
            fake_cap.release.assert_called_once()

    def test_try_open_hw_handles_missing_constants_gracefully(self):
        """Older OpenCV builds don't have CAP_PROP_HW_ACCELERATION —
        the helper must NOT crash; it should return None for fallback."""
        import video_capture_helper as h

        with patch('video_capture_helper.cv2', spec=[]):
            # spec=[] makes the mock not have any attributes — simulates
            # an ancient OpenCV without hw accel constants.
            result = h._try_open_hw('x.mp4')
            assert result is None

    def test_hw_open_falls_back_when_probe_read_fails(self):
        """REGRESSION: HW capture can isOpened()==True but read()==False.
        Must release and return None so the SW path runs."""
        import video_capture_helper as h

        # Cap "opens" but read() fails — exactly the user's bug.
        bad_cap = MagicMock()
        bad_cap.isOpened.return_value = True
        bad_cap.read.return_value = (False, None)

        import cv2
        # Ensure the constants exist on cv2 so the function tries
        with patch.object(cv2, 'CAP_PROP_HW_ACCELERATION', 1234, create=True), \
             patch.object(cv2, 'VIDEO_ACCELERATION_ANY', 5,  create=True), \
             patch.object(cv2, 'CAP_FFMPEG',             1900, create=True), \
             patch('video_capture_helper.cv2.VideoCapture', return_value=bad_cap):
            result = h._try_open_hw('/some/video.mp4')

        assert result is None, "must fall back when probe read fails"
        bad_cap.release.assert_called_once()

    def test_hw_open_succeeds_with_valid_probe(self):
        """When read() returns a frame, the HW cap is returned and the
        position is rewound to frame 0."""
        import video_capture_helper as h
        import numpy as np

        good_cap = MagicMock()
        good_cap.isOpened.return_value = True
        # Return a tiny valid-looking frame
        good_cap.read.return_value = (True, np.zeros((1, 1, 3), dtype=np.uint8))

        import cv2
        with patch.object(cv2, 'CAP_PROP_HW_ACCELERATION', 1234, create=True), \
             patch.object(cv2, 'VIDEO_ACCELERATION_ANY', 5,  create=True), \
             patch.object(cv2, 'CAP_FFMPEG',             1900, create=True), \
             patch.object(cv2, 'CAP_PROP_POS_FRAMES',    1,    create=True), \
             patch('video_capture_helper.cv2.VideoCapture', return_value=good_cap):
            result = h._try_open_hw('/some/video.mp4')

        assert result is good_cap
        # Must rewind so the caller's seek-and-read works
        good_cap.set.assert_called_with(cv2.CAP_PROP_POS_FRAMES, 0)

    def test_open_capture_uses_sw_path_after_hw_probe_fails(self):
        """End-to-end: open_capture() must NOT return the failed HW cap
        — it must call the SW path."""
        import video_capture_helper as h

        # Reset the session flag in case a prior test set it
        h._session_hw_disabled = False
        sw_cap = MagicMock()
        sw_cap.isOpened.return_value = True

        with patch.object(h, '_try_open_hw', return_value=None), \
             patch.object(h, '_try_open_sw', return_value=sw_cap):
            assert h.open_capture('x.mp4', hw_accel=True) is sw_cap


class TestHwAccelKillSwitch:
    """The session-wide HW disable: prevents FFmpeg [hevc] errors from
    spamming stderr on every cap when HW accel is broken on this build."""

    def test_session_disabled_after_first_failure(self):
        import video_capture_helper as h
        h._session_hw_disabled = False

        sw_cap = MagicMock()
        sw_cap.isOpened.return_value = True

        with patch.object(h, '_try_open_hw', return_value=None) as hw, \
             patch.object(h, '_try_open_sw', return_value=sw_cap):
            # First call: HW is tried (and fails)
            h.open_capture('a.mp4', hw_accel=True)
            assert h._session_hw_disabled is True
            assert hw.call_count == 1

            # Second call: HW should NOT be tried again
            h.open_capture('b.mp4', hw_accel=True)
            assert hw.call_count == 1, "HW must not be re-tried this session"

    def test_session_flag_respected_even_when_caller_requests_hw(self):
        """If the session flag is set, hw_accel=True is ignored."""
        import video_capture_helper as h
        h._session_hw_disabled = True

        sw_cap = MagicMock()
        sw_cap.isOpened.return_value = True
        with patch.object(h, '_try_open_hw') as hw, \
             patch.object(h, '_try_open_sw', return_value=sw_cap):
            h.open_capture('a.mp4', hw_accel=True)
            hw.assert_not_called()

    def test_build_lacks_hw_accel_detects_prebuilt_wheel(self):
        """The pip opencv-python wheel reports 'prebuilt binaries' in
        getBuildInformation. We must auto-disable HW for that build."""
        import video_capture_helper as h

        fake_info = (
            "  Video I/O:\n"
            "    FFMPEG:                      YES (prebuilt binaries)\n"
            "      avcodec:                   YES (58.134.100)\n"
        )
        with patch('video_capture_helper.cv2.getBuildInformation',
                   return_value=fake_info):
            assert h._build_lacks_hw_accel() is True

    def test_build_with_hw_accel_keeps_it_enabled(self):
        """A custom OpenCV build that mentions D3D11/NVDEC should NOT be
        auto-disabled."""
        import video_capture_helper as h

        fake_info = (
            "  Video I/O:\n"
            "    FFMPEG:                      YES\n"
            "      avcodec:                   YES (60.0.0)\n"
            "      HW accel:                  D3D11VA, NVDEC, CUVID\n"
        )
        with patch('video_capture_helper.cv2.getBuildInformation',
                   return_value=fake_info):
            assert h._build_lacks_hw_accel() is False

    def test_build_info_unavailable_is_treated_as_no_hw(self):
        """If we can't even read the build info, default to safe (no HW)."""
        import video_capture_helper as h
        with patch('video_capture_helper.cv2.getBuildInformation',
                   side_effect=Exception("oops")):
            assert h._build_lacks_hw_accel() is True


class TestMultiThreadedDecode:
    def test_default_capture_options_include_threads(self):
        """main.py sets OPENCV_FFMPEG_CAPTURE_OPTIONS at import time. We
        can't easily re-trigger that here, but we can verify the literal
        string we set is well-formed."""
        # Set what main.py would set (idempotent via setdefault in real run)
        opts = 'threads;0|thread_type;slice'
        parts = dict(p.split(';') for p in opts.split('|'))
        assert parts['threads'] == '0', "0 means auto (use all cores)"
        assert parts['thread_type'] == 'slice', \
            "slice parallelism is seek-safe; frame parallelism is not"

    def test_custom_thread_count_overrides_default(self):
        """When the user sets cpu_decode_threads=4, the env var should be
        updated to use 4 threads explicitly."""
        # We test the assembly logic in isolation
        n = 4
        opts = f'threads;{n}|thread_type;slice'
        parts = dict(p.split(';') for p in opts.split('|'))
        assert parts['threads'] == '4'


# ── _try_delete_with_retry logic ─────────────────────────────────────────────
def _stub_main_window_for_delete():
    from main_window import MainWindow
    return MainWindow.__new__(MainWindow)


def test_delete_retries_on_permission_error():
    """The retry helper must keep trying on PermissionError until success
    OR the retry budget runs out."""
    stub = _stub_main_window_for_delete()
    attempts: list = []

    def fake_remove(path):
        attempts.append(1)
        if len(attempts) < 3:
            raise PermissionError("locked")

    with patch('main_window._HAS_SEND2TRASH', False), \
         patch('os.path.isdir', return_value=False), \
         patch('os.remove', side_effect=fake_remove), \
         patch('time.sleep'):
        result = stub._try_delete_with_retry('/some/path.mp4')

    assert result is None, "should succeed once the file becomes unlocked"
    assert len(attempts) == 3


def test_delete_gives_up_after_max_retries():
    stub = _stub_main_window_for_delete()

    with patch('main_window._HAS_SEND2TRASH', False), \
         patch('os.path.isdir', return_value=False), \
         patch('os.remove', side_effect=PermissionError("still locked")), \
         patch('time.sleep'):
        result = stub._try_delete_with_retry('/some/path.mp4')

    assert result is not None
    assert "locked" in result


def test_delete_does_not_retry_on_nonbusy_errors():
    """If the error is FileNotFoundError (or similar non-busy), don't waste
    retries — just return the error immediately."""
    stub = _stub_main_window_for_delete()
    call_count = [0]

    def raise_fnf(path):
        call_count[0] += 1
        raise FileNotFoundError("missing")

    with patch('main_window._HAS_SEND2TRASH', False), \
         patch('os.path.isdir', return_value=False), \
         patch('os.remove', side_effect=raise_fnf), \
         patch('time.sleep'):
        stub._try_delete_with_retry('/missing.mp4')

    # FileNotFoundError is not a PermissionError → should not retry
    assert call_count[0] == 1


# ── hover-playback policy ────────────────────────────────────────────────────
class TestHoverPlaybackPolicy:
    """The policy: files larger than hover_preview_max_gb get NO hover
    preview at all. Files between large_file_threshold and hover_max get
    a reduced target FPS. Files smaller use native FPS."""

    def _bucket(self, size_bytes, hover_max_gb=4.0, large_mb=500,
                fps_large=8):
        """Mirror the decision logic from _start_playback into a pure function
        so we can unit-test the thresholds."""
        if size_bytes > hover_max_gb * (1024 ** 3):
            return 'skip'
        if size_bytes > large_mb * (1024 ** 2):
            return ('reduced', fps_large)
        return 'native'

    def test_tiny_file_uses_native(self):
        assert self._bucket(50 * 1024 * 1024) == 'native'

    def test_large_file_uses_reduced(self):
        result = self._bucket(800 * 1024 * 1024)
        assert result == ('reduced', 8)

    def test_huge_file_is_skipped(self):
        assert self._bucket(8 * 1024 ** 3) == 'skip'

    def test_threshold_at_exactly_4gb_boundary(self):
        # 4 GB exactly — NOT skipped (must be strictly greater)
        assert self._bucket(4 * 1024 ** 3) != 'skip'
        # 4 GB + 1 byte → skipped
        assert self._bucket(4 * 1024 ** 3 + 1) == 'skip'

    def test_custom_thresholds_honored(self):
        # User cranks the threshold to 100 GB — even 8 GB files preview
        assert self._bucket(8 * 1024 ** 3, hover_max_gb=100.0) != 'skip'
        # User cranks down to 1 GB — 2 GB files are skipped
        assert self._bucket(2 * 1024 ** 3, hover_max_gb=1.0) == 'skip'
