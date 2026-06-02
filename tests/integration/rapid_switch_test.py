"""
Reproduce the user's exact "freeze on rapid widget switch" bug.

The pattern:
    1. Hover widget A (preview starts)
    2. Seek widget A (slider drag)
    3. BEFORE A's playback fully starts: hover widget B (cursor moves)
    4. Seek widget B
    5. BEFORE B's playback fully starts: hover widget C
    ... etc

Previously, each switch left the OLD widget's play_thread still running
(stuck inside av.open() for a 30 GB MKV — 8s timeout). With multiple
play_threads × 24 FRAME-threading internal decoder workers each, the CPU
gets pegged and the GUI thread starves → Windows force-closes the app
with an "Application Hang" event.

This test:
    - Picks 10 different widgets
    - Hovers each one for only ~200 ms (less than av.open() takes on HDD)
    - Tracks: thread count, RAM, count of LIVE play_threads
    - FAILS if at any point >2 play_threads are alive (cross-widget accum)
    - FAILS if thread count grows unbounded

Run:
    python tests/integration/rapid_switch_test.py --source <VIDEO_DIR> --iters 25
"""
import os
import sys
import time
import random
import argparse
import datetime
import threading
import faulthandler
from pathlib import Path

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

os.environ['OPENCV_FFMPEG_LOGLEVEL'] = '-8'
os.environ['OPENCV_LOG_LEVEL']       = 'OFF'
os.environ['AV_LOG_LEVEL']           = 'quiet'
os.environ.setdefault('VIDEO_ORGANIZER_QSETTINGS_ORG', 'VOrgRapidSwitch')
os.environ.setdefault('VIDEO_ORGANIZER_QSETTINGS_APP', 'RapidSwitch')

try:
    from hires_timer import request_high_resolution_timer
    request_high_resolution_timer()
except Exception:
    pass

faulthandler.enable(file=sys.stderr)


def log(msg, indent=0):
    ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    sys.stderr.write(f"  {ts} {' ' * indent}{msg}\n")
    sys.stderr.flush()


def thread_count():
    try:
        import psutil
        return psutil.Process().num_threads()
    except ImportError:
        return threading.active_count()


def ram_mb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        return -1


def pump(app, ms):
    from PyQt6.QtCore import QEventLoop
    deadline = time.perf_counter() + ms / 1000.0
    while time.perf_counter() < deadline:
        app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 20)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='<VIDEO_DIR>')
    p.add_argument('--iters', type=int, default=25)
    p.add_argument('--hover_ms', type=int, default=200,
                   help='How long to "hover" each widget — should be LESS '
                        'than av.open() time so we exercise the race')
    p.add_argument('--max_threads', type=int, default=250,
                   help='Bail if thread count exceeds this')
    p.add_argument('--max_concurrent_play_threads', type=int, default=2,
                   help='Bail if more than this many play_threads are alive')
    args = p.parse_args()

    log(f"=== RAPID SWITCH TEST: {args.iters} switches, "
        f"hover {args.hover_ms} ms each ===")
    log(f"source: {args.source}")
    log(f"baseline: threads={thread_count()}, ram={ram_mb():.0f} MB")

    from PyQt6.QtWidgets import QApplication
    from cache_manager import CacheManager
    from app_settings import AppSettings
    from thumbnail_generator import ThumbnailGeneratorService
    from thumbnail_grid_widget import ThumbnailGridWidget
    # Import the module so we can peek at _running_play_threads
    import video_thumbnail_widget

    app = QApplication.instance() or QApplication(sys.argv)

    import tempfile
    cache_dir = Path(tempfile.mkdtemp(prefix='vorg_rapid_'))
    cache    = CacheManager(str(cache_dir / 'cache.db'), str(cache_dir / 'thumbs'))
    settings = AppSettings()
    settings.hover_preview_max_gb     = 100.0
    settings.large_file_threshold_mb  = 500
    settings.use_hw_accel             = True

    gen  = ThumbnailGeneratorService(cache)
    grid = ThumbnailGridWidget(gen, settings, cache)
    grid.resize(1600, 1000)

    log(f"loading {args.source}")
    grid.load_folder(args.source)
    deadline = time.perf_counter() + 30
    while time.perf_counter() < deadline:
        from PyQt6.QtCore import QEventLoop
        app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 50)
        if grid._total_count > 0 and grid._batch_idx >= grid._total_count:
            break
    log(f"after scan: threads={thread_count()}, ram={ram_mb():.0f} MB, "
        f"items={grid._total_count}")
    pump(app, 500)
    grid._full_relayout()
    pump(app, 500)

    actives = [(idx, w) for idx, w in sorted(grid._active.items())
               if hasattr(w, 'video_path') and hasattr(w, '_start_playback')]
    if len(actives) < 5:
        log(f"FAIL: need ≥5 active widgets, got {len(actives)}")
        sys.exit(1)
    log(f"have {len(actives)} active widgets for rapid switching")

    log("")
    log("=== RAPID SWITCH START ===")
    log(f"{'iter':>4} | {'widget':>20} | {'hover_ms':>8} | "
        f"{'threads':>7} | {'play_threads_alive':>18} | {'ram_mb':>6}")
    rng = random.Random(42)
    baseline_threads = thread_count()
    peak_play_threads = 0
    failures = []

    for i in range(1, args.iters + 1):
        # Pick a random widget; sometimes seek slightly before hover
        idx, w = rng.choice(actives)
        name = os.path.basename(w.video_path)[:20]

        # Optionally seek to a random position FIRST (slider drag pattern)
        if rng.random() < 0.5:
            dur = w._duration or 60.0
            w._seek_time = dur * rng.uniform(0.1, 0.9)
            try:
                w._apply_seek()
            except Exception as e:
                log(f"  iter {i} apply_seek error: {e}", indent=2)

        # Hover (start playback) — this is where the leak would happen
        t0 = time.perf_counter()
        w._hovering = True
        w._start_playback()
        pump(app, args.hover_ms)
        # NOTE: deliberately do NOT call _stop_playback. The next iteration
        # picks a DIFFERENT widget and the global guard should stop us.

        hover_ms = (time.perf_counter() - t0) * 1000
        threads = thread_count()
        ram = ram_mb()
        # Count live play_threads (from the global registry).
        # Some may have already exited; we count instances that are still
        # in the registry AND still isRunning().
        running = video_thumbnail_widget._running_play_threads
        n_play_alive = 0
        for t in list(running):
            try:
                if hasattr(t, 'isRunning') and t.isRunning():
                    n_play_alive += 1
            except RuntimeError:
                pass

        peak_play_threads = max(peak_play_threads, n_play_alive)
        log(f"{i:>4} | {name:>20} | {hover_ms:>8.0f} | "
            f"{threads:>7} | {n_play_alive:>18} | {ram:>6.0f}")

        # Hard limits — fail fast
        if threads > args.max_threads:
            failures.append(f"iter {i}: thread count {threads} > {args.max_threads}")
            log("  ⚠️ THREAD LIMIT EXCEEDED — dumping stacks", indent=2)
            faulthandler.dump_traceback(file=sys.stderr)
            break
        if n_play_alive > args.max_concurrent_play_threads:
            failures.append(
                f"iter {i}: {n_play_alive} concurrent play_threads > "
                f"{args.max_concurrent_play_threads} — old play_threads not "
                f"getting stopped on widget switch")

    log("")
    log("=== DONE ===")
    log(f"final: threads={thread_count()}, ram={ram_mb():.0f} MB")
    log(f"peak concurrent play_threads: {peak_play_threads}")
    log(f"leak above baseline: {thread_count() - baseline_threads} threads")

    # Stop all hover state so cleanup doesn't keep things alive
    for _, w in actives:
        w._hovering = False
        try:
            w._stop_playback()
        except Exception:
            pass
    pump(app, 1000)

    log("cleanup: shutdown_all_widgets")
    t0 = time.perf_counter()
    try:
        grid.shutdown_all_widgets()
    except Exception:
        import traceback
        traceback.print_exc(file=sys.stderr)
    log(f"shutdown took {(time.perf_counter()-t0)*1000:.0f} ms")
    log(f"after cleanup: threads={thread_count()}, ram={ram_mb():.0f} MB")

    if failures:
        log("")
        log("FAILURES:")
        for f in failures:
            log(f"  {f}")
        sys.exit(1)
    log("SUCCESS")


if __name__ == '__main__':
    main()
