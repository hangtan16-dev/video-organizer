"""Decisive probe: does PyAV decode/seek release the GIL?

A worker thread opens a huge file, deep-seeks, and decodes frames (exactly
what the hover preview does). The MAIN thread ticks a counter every ~5 ms.
If PyAV releases the GIL during its libav calls, the main-thread counter
keeps climbing smoothly. If PyAV HOLDS the GIL during decode/seek, the
main-thread counter FREEZES for long stretches — which is exactly what would
wedge the GUI event loop.

We log the largest gap between consecutive main-thread ticks. A gap of
seconds == the GIL was held that long by the worker.
"""
import os, sys, time, threading, faulthandler
os.environ['AV_LOG_LEVEL'] = 'quiet'
faulthandler.dump_traceback_later(8, repeat=True, file=sys.stderr)
import av
av.logging.set_level(av.logging.FATAL)

PATH = sys.argv[1] if len(sys.argv) > 1 else r"<VIDEO_DIR>\sample_8k_video.mp4"
SEEK_S = float(sys.argv[2]) if len(sys.argv) > 2 else 1066.0

def log(m): sys.stderr.write(m + "\n"); sys.stderr.flush()

worker_done = threading.Event()
worker_phase = ["init"]

def worker():
    try:
        worker_phase[0] = "av.open"
        t = time.perf_counter()
        c = av.open(PATH, timeout=(8.0, 4.0))
        log(f"  [worker] av.open took {time.perf_counter()-t:.2f}s")
        s = c.streams.video[0]
        s.codec_context.thread_count = 4
        s.codec_context.thread_type = 'FRAME'
        worker_phase[0] = "seek"
        t = time.perf_counter()
        tb = s.time_base
        target = int(SEEK_S / tb) if tb else int(SEEK_S * 1_000_000)
        c.seek(target, stream=s, backward=True)
        log(f"  [worker] seek took {time.perf_counter()-t:.2f}s")
        worker_phase[0] = "decode"
        t = time.perf_counter()
        n = 0
        for frame in c.decode(s):
            n += 1
            if n >= 30:
                break
        log(f"  [worker] decoded {n} frames in {time.perf_counter()-t:.2f}s")
        worker_phase[0] = "close"
        t = time.perf_counter()
        c.close()
        log(f"  [worker] close took {time.perf_counter()-t:.2f}s")
    except Exception as e:
        log(f"  [worker] ERROR: {e}")
    finally:
        worker_done.set()

log(f"=== GIL PROBE on {os.path.basename(PATH)} seek={SEEK_S}s ===")
th = threading.Thread(target=worker, daemon=True)
th.start()

# Main thread ticks; measure the largest stall (= longest GIL hold by worker)
last = time.perf_counter()
max_gap = 0.0
max_gap_phase = ""
ticks = 0
while not worker_done.is_set():
    now = time.perf_counter()
    gap = now - last
    if gap > max_gap:
        max_gap = gap
        max_gap_phase = worker_phase[0]
    last = now
    ticks += 1
    time.sleep(0.005)   # main thread wants to run every 5 ms

log(f"=== RESULT: main-thread ran {ticks} ticks; "
    f"MAX STALL = {max_gap*1000:.0f} ms during '{max_gap_phase}' ===")
if max_gap > 0.5:
    log(f"  >>> GIL WAS HELD ~{max_gap:.1f}s by PyAV during '{max_gap_phase}' "
        f"— this is what wedges the GUI. <<<")
else:
    log("  >>> PyAV released the GIL fine (max stall < 0.5s). GUI wedge is "
        "elsewhere. <<<")
