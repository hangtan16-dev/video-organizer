"""
Tests for _MoveWorker error classification — the part that decides whether
an OSError is "transient busy" (retry rename), "cross-drive" (fallthrough
to copy+delete), or "real error" (give up).

We avoid creating a QThread instance and test the @classmethod helpers
directly. They're pure logic and don't need a QApplication.
"""
import errno

import pytest

# Pull just the class — does NOT need QApplication for classmethod calls
from main_window import _MoveWorker


def _make_oserror(errno_val=None, winerror=None):
    """Build an OSError with the given errno or winerror set."""
    exc = OSError()
    if errno_val is not None:
        exc.errno = errno_val
    if winerror is not None:
        exc.winerror = winerror
    return exc


# ── busy / sharing-violation classification ──────────────────────────────────
class TestIsBusyError:
    def test_permission_error_is_busy(self):
        assert _MoveWorker._is_busy_error(PermissionError())

    def test_winerror_5_access_denied_is_busy(self):
        assert _MoveWorker._is_busy_error(_make_oserror(winerror=5))

    def test_winerror_32_sharing_violation_is_busy(self):
        assert _MoveWorker._is_busy_error(_make_oserror(winerror=32))

    def test_cross_device_is_not_busy(self):
        assert not _MoveWorker._is_busy_error(_make_oserror(winerror=17))
        assert not _MoveWorker._is_busy_error(_make_oserror(errno_val=errno.EXDEV))

    def test_unrelated_error_is_not_busy(self):
        assert not _MoveWorker._is_busy_error(_make_oserror(winerror=2))    # not found
        assert not _MoveWorker._is_busy_error(_make_oserror(winerror=87))   # parameter incorrect
        assert not _MoveWorker._is_busy_error(_make_oserror(errno_val=errno.ENOSPC))

    def test_filenotfound_is_not_busy(self):
        # FileNotFoundError is a subclass of OSError but not PermissionError
        assert not _MoveWorker._is_busy_error(FileNotFoundError())


# ── cross-device classification ──────────────────────────────────────────────
class TestIsCrossDeviceError:
    def test_winerror_17_is_cross_device(self):
        assert _MoveWorker._is_cross_device_error(_make_oserror(winerror=17))

    def test_errno_exdev_is_cross_device(self):
        assert _MoveWorker._is_cross_device_error(_make_oserror(errno_val=errno.EXDEV))

    def test_permission_denied_is_not_cross_device(self):
        assert not _MoveWorker._is_cross_device_error(PermissionError())
        assert not _MoveWorker._is_cross_device_error(_make_oserror(winerror=5))
        assert not _MoveWorker._is_cross_device_error(_make_oserror(winerror=32))

    def test_other_oserror_is_not_cross_device(self):
        assert not _MoveWorker._is_cross_device_error(_make_oserror(winerror=2))
        assert not _MoveWorker._is_cross_device_error(_make_oserror(errno_val=errno.ENOSPC))


# ── retry constants are sensible ──────────────────────────────────────────────
def test_retry_constants_are_positive():
    assert _MoveWorker._RENAME_RETRIES >= 3
    assert _MoveWorker._RENAME_RETRY_DELAY > 0
    assert _MoveWorker._DELETE_RETRIES >= 3
    assert _MoveWorker._DELETE_RETRY_DELAY > 0


def test_total_rename_retry_window_is_meaningful():
    """The total time spent retrying a busy rename should be at least 2s
    so it covers the worst case of cv2 + antivirus releasing the handle."""
    total_ms = (_MoveWorker._RENAME_RETRIES - 1) * _MoveWorker._RENAME_RETRY_DELAY * 1000
    assert total_ms >= 2000
