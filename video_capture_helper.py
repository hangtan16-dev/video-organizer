"""
Centralised cv2.VideoCapture factory with optional hardware acceleration.

For large VR videos encoded in H.265 / HEVC / VP9 the software decoder eats
50–200 ms per frame on CPU.  When the user's system has a GPU decoder
(NVDEC on NVIDIA, D3D11 on Windows, VAAPI on Linux) FFmpeg can offload the
decode, dropping per-frame cost to ~5 ms.

OpenCV exposes this through CAP_PROP_HW_ACCELERATION (added in 4.5+).  Not
all builds support the constants, so we use getattr() with fallbacks.

This module also normalises the "open + fail" cleanup: callers don't need
to remember to release a half-opened capture.

⚠ HW accel + the pip `opencv-python` wheel
──────────────────────────────────────────
The standard pip-installed opencv-python ships a prebuilt FFmpeg that does
NOT have working D3D11VA / DXVA2 / NVDEC implementations linked in for
HEVC on Windows, even though it accepts the CAP_PROP_HW_ACCELERATION hint.
FFmpeg prints noisy [hevc] errors and SW-fallback can be unreliable.

We detect this build at import time and disable HW accel preemptively
(setting the session flag).  Custom OpenCV builds with real HW accel
still work — they advertise it in getBuildInformation().
"""
from typing import Optional

import cv2

from app_logger import get_logger
log = get_logger(__name__)


# ── Session-wide HW-accel kill switch ────────────────────────────────────────
# Once any HW open fails (or we detect the build doesn't support it), we
# stop attempting HW accel for the rest of the session.  This prevents the
# noisy "[hevc] Failed setup for format d3d11" stderr spam that prints
# on every cap.read() attempt with a broken HW pipeline.
_session_hw_disabled = False


def _build_lacks_hw_accel() -> bool:
    """Heuristically detect OpenCV builds that don't actually support GPU
    decode.  The pip `opencv-python` wheel reports FFmpeg as `(prebuilt
    binaries)` and does NOT mention any HW backend in getBuildInformation.
    Returns True when HW accel should be skipped preemptively."""
    try:
        info = cv2.getBuildInformation()
    except Exception:
        return True   # can't tell → safer to skip
    # If FFmpeg is the pip prebuilt wheel, HW accel is almost certainly broken.
    if "prebuilt binaries" in info:
        return True
    # Custom builds typically mention one of these if HW is compiled in.
    hw_markers = ("D3D11", "DXVA2", "NVDEC", "CUVID", "VAAPI", "VideoToolbox")
    return not any(m in info for m in hw_markers)


# Apply the heuristic at import time
if _build_lacks_hw_accel():
    _session_hw_disabled = True
    log.info("HW accel pre-disabled: this OpenCV build doesn't ship with "
             "GPU decode support. Multi-threaded CPU decode (AVX2) is "
             "active instead.")


def session_hw_disabled() -> bool:
    """Public accessor for tests + diagnostics."""
    return _session_hw_disabled


def open_capture(path: str, *, hw_accel: bool = True) -> Optional['cv2.VideoCapture']:
    """Open a VideoCapture, returning None (and releasing internally) on failure.

    If hw_accel is True AND this OpenCV build supports it, the FFmpeg backend
    is requested with CAP_PROP_HW_ACCELERATION = VIDEO_ACCELERATION_ANY.  If
    that fails (e.g. unsupported codec on the GPU, no driver), we fall back
    to a plain software-decode open transparently.

    Returns the opened VideoCapture, or None if neither attempt could open
    the file.  Caller is responsible for cap.release() when done.
    """
    global _session_hw_disabled
    if hw_accel and not _session_hw_disabled:
        cap = _try_open_hw(path)
        if cap is not None:
            return cap
        # First-time HW failure → disable for the rest of this session so
        # we don't spam stderr with the same FFmpeg errors on every cap.
        _session_hw_disabled = True
        log.warning("HW accel disabled for this session (probe failed). "
                    "Falling back to multi-threaded CPU decode.")
    return _try_open_sw(path)


# ── implementation ────────────────────────────────────────────────────────────
def _try_open_hw(path: str) -> Optional['cv2.VideoCapture']:
    """Attempt hardware-accelerated open. Returns None if the build / file
    doesn't actually support GPU decode (caller will fall back to SW).

    The probe-read step is CRITICAL: on standard pip opencv-python wheels
    for Windows, `cv2.VideoCapture(path, CAP_FFMPEG, [HW_ACCEL, ANY])`
    happily returns a cap with isOpened()==True even when the underlying
    HW codec init silently failed.  Subsequent read() calls then return
    False with no error.  We were silently dropping playback for users
    whose OpenCV build doesn't include working HW accel.

    By reading one frame upfront we definitively verify the pipeline
    works.  If it doesn't, we release and return None so the SW path
    runs instead.
    """
    accel_prop  = getattr(cv2, 'CAP_PROP_HW_ACCELERATION', None)
    accel_any   = getattr(cv2, 'VIDEO_ACCELERATION_ANY', None)
    cap_ffmpeg  = getattr(cv2, 'CAP_FFMPEG', None)
    if accel_prop is None or accel_any is None or cap_ffmpeg is None:
        return None

    try:
        cap = cv2.VideoCapture(path, cap_ffmpeg, [int(accel_prop), int(accel_any)])
    except (cv2.error, TypeError, Exception):
        return None

    if cap is None or not cap.isOpened():
        if cap is not None:
            cap.release()
        return None

    # ── PROBE: read one frame to verify HW decode actually works ──────────
    try:
        ret, frame = cap.read()
    except Exception:
        ret, frame = False, None
    if not ret or frame is None:
        cap.release()
        log.info("HW-accel cap opened but probe read failed; falling back to CPU decode")
        return None

    # Rewind so the caller's subsequent seek-and-read lands where expected.
    # Some streams don't seek cleanly to frame 0 from frame 1 — ignore the
    # error, the caller usually seeks to a specific position anyway.
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    except Exception:
        pass
    return cap


def _try_open_sw(path: str) -> Optional['cv2.VideoCapture']:
    """Plain CPU-decode open. Returns None on failure."""
    try:
        cap = cv2.VideoCapture(path)
    except Exception as e:
        log.warning("cv2.VideoCapture(%s) raised: %s", path, e)
        return None
    if not cap.isOpened():
        cap.release()
        return None
    return cap
