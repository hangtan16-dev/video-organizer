"""
Request high-resolution OS scheduler timer.

Without this, Windows' default scheduler granularity is 15.6 ms.  Any
QThread.msleep(N) where N isn't a multiple of ~15 gets rounded UP.  For
24 fps playback that means sleep(26) → actual ~31ms, pushing the loop
cycle past native frame time and capping observable FPS at ~80-90% of
native.

After timeBeginPeriod(1), msleep is accurate to ~1ms — enough to hit
real-time playback for any reasonable codec on a multi-core CPU.

The setting is global per-process.  Slight power-usage cost (the CPU
wakes more often).  For a video app this is the right trade.

On non-Windows systems this is a no-op.
"""
import platform

from app_logger import get_logger
log = get_logger(__name__)

_applied = False
_winmm = None


def request_high_resolution_timer() -> bool:
    """Set Windows timer resolution to 1 ms.  Idempotent.

    Returns True if applied (or already applied), False if not applicable
    or failed.  Call at app startup, BEFORE any playback / hover loop runs.
    """
    global _applied, _winmm
    if _applied:
        return True
    if platform.system() != 'Windows':
        return False
    try:
        import ctypes
        _winmm = ctypes.windll.winmm
        # timeBeginPeriod returns 0 (TIMERR_NOERROR) on success
        rc = _winmm.timeBeginPeriod(1)
        if rc == 0:
            _applied = True
            log.info("Windows timer resolution set to 1 ms (timeBeginPeriod)")
            return True
        log.warning("timeBeginPeriod(1) returned %s (non-zero = TIMERR_*)", rc)
        return False
    except Exception as e:
        log.warning("Could not request high-resolution timer: %s", e)
        return False


def release_high_resolution_timer() -> None:
    """Restore the system default timer resolution.  Should be paired
    with the request, ideally at app shutdown.  Safe to call even if
    request_high_resolution_timer wasn't called."""
    global _applied, _winmm
    if not _applied or _winmm is None:
        return
    try:
        _winmm.timeEndPeriod(1)
        _applied = False
    except Exception:
        pass


def is_active() -> bool:
    return _applied
