"""
Benchmark hover preview decode rate on an 8K 60 fps HEVC file.

Measures the current pipeline AND a few decode-side tricks to see which
actually help:

  1. baseline             — current cv2 path (threads;24 slice)
  2. baseline_pyav        — current PyAV path (threads;24 slice)
  3. pyav_frame_threads   — thread_type='FRAME' (lookahead parallelism)
  4. pyav_skip_nonref     — codec_context.skip_frame = NONREF (drop B frames)
  5. pyav_skip_bidir      — drop only bidirectional frames
  6. pyav_skip_nonkey     — keyframes only (slideshow mode)
  7. pyav_frame_and_skip  — combine FRAME threading + NONREF

Run as a script:
    python tests/integration/bench_8k_hover.py
"""
import os
import sys
import time
from pathlib import Path

# Project paths + Qt setup
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                     os.pardir, os.pardir))
sys.path.insert(0, _ROOT)

os.environ['OPENCV_FFMPEG_LOGLEVEL'] = '-8'
os.environ['OPENCV_LOG_LEVEL']       = 'OFF'
os.environ['AV_LOG_LEVEL']           = 'quiet'
_NCPU = os.cpu_count() or 4
_THREADS = max(1, int(_NCPU * 0.75))
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = (
    f'threads;{_THREADS}|thread_type;slice'
)

# Force the high-res timer ASAP
try:
    from hires_timer import request_high_resolution_timer
    request_high_resolution_timer()
except Exception:
    pass


VR_FILE = Path(r'<VIDEO_DIR>\sample_8k_video.mp4')
DISP_W, DISP_H = 800, 450
DURATION = 8.0   # seconds of playback to measure


def bench_cv2_baseline():
    """Current cv2 path: open + read + resize + cvtColor + tobytes + msleep."""
    import cv2
    cap = cv2.VideoCapture(str(VR_FILE))
    if not cap.isOpened():
        print("cv2: failed to open"); return
    native = cap.get(cv2.CAP_PROP_FPS)
    # Warm up — decode first frame so the threadpool spins up
    cap.read()
    t0 = time.perf_counter()
    count = 0
    while time.perf_counter() - t0 < DURATION:
        ret, frame = cap.read()
        if not ret: break
        fh, fw = frame.shape[:2]
        s = min(DISP_W / fw, DISP_H / fh)
        nw, nh = max(1, int(fw * s)), max(1, int(fh * s))
        frame_s = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(frame_s, cv2.COLOR_BGR2RGB)
        _ = rgb.tobytes()
        count += 1
    cap.release()
    obs = count / DURATION
    print(f"  {obs:6.2f} fps  ({count} frames in {DURATION:.1f}s, {obs/native*100:.0f}% of {native:.1f} native)")


def bench_pyav(label, *, thread_type='SLICE', skip_frame=None):
    """Run PyAV decode with the given options."""
    import av
    container = av.open(str(VR_FILE))
    try:
        stream = container.streams.video[0]
        ctx = stream.codec_context
        try:
            ctx.thread_count = _THREADS
            ctx.thread_type = thread_type
        except Exception:
            pass
        if skip_frame is not None:
            try:
                ctx.skip_frame = skip_frame
            except Exception as e:
                print(f"  (skip_frame={skip_frame} not supported: {e})")
                return

        native = float(stream.average_rate) if stream.average_rate else 60.0

        # Warm up — pull a few frames so codec init is amortized
        warm_iter = container.decode(stream)
        for i, _ in enumerate(warm_iter):
            if i >= 5: break

        t0 = time.perf_counter()
        count = 0
        for frame in container.decode(stream):
            # reformat to RGB at display size — matches the play thread
            fw, fh = frame.width, frame.height
            s = min(DISP_W / fw, DISP_H / fh)
            nw, nh = max(1, int(fw * s)), max(1, int(fh * s))
            ref = frame.reformat(width=nw, height=nh, format='rgb24')
            _ = ref.to_ndarray().tobytes()
            count += 1
            if time.perf_counter() - t0 >= DURATION:
                break
        obs = count / DURATION
        print(f"  {obs:6.2f} fps  ({count} frames in {DURATION:.1f}s, "
              f"{obs/native*100:.0f}% of {native:.1f} native)  [{label}]")
    finally:
        container.close()


def main():
    if not VR_FILE.is_file():
        print(f"FILE NOT FOUND: {VR_FILE}")
        return 1

    size_gb = VR_FILE.stat().st_size / (1024**3)
    print(f"File:    {VR_FILE.name}")
    print(f"Size:    {size_gb:.1f} GB")

    # Probe codec info
    import av
    with av.open(str(VR_FILE)) as c:
        v = c.streams.video[0]
        print(f"Codec:   {v.codec_context.name} {v.width}x{v.height}")
        print(f"FPS:     {float(v.average_rate):.2f}")
        print(f"Threads: {_THREADS} (75% of {_NCPU} cores)")
    print(f"Output:  {DISP_W}x{DISP_H} RGB24")
    print()

    print("[1] cv2 baseline (threads;24, slice)")
    bench_cv2_baseline()
    print()

    print("[2] PyAV baseline (threads=24, slice)")
    bench_pyav("baseline", thread_type='SLICE')
    print()

    print("[3] PyAV with FRAME threading (lookahead parallelism)")
    bench_pyav("frame_threads", thread_type='FRAME')
    print()

    # PyAV's skip_frame accepts string enum values
    print("[4] PyAV + skip_frame=NONREF (drop non-reference frames)")
    bench_pyav("skip_nonref", thread_type='SLICE', skip_frame='NONREF')
    print()

    print("[5] PyAV + skip_frame=BIDIR (drop bidirectional frames)")
    bench_pyav("skip_bidir", thread_type='SLICE', skip_frame='BIDIR')
    print()

    print("[6] PyAV + skip_frame=NONKEY (keyframes only - slideshow)")
    bench_pyav("skip_nonkey", thread_type='SLICE', skip_frame='NONKEY')
    print()

    print("[7] PyAV + FRAME threading + NONREF skip (combined)")
    bench_pyav("frame+nonref", thread_type='FRAME', skip_frame='NONREF')
    print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
