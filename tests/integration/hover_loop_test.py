"""
Minimal repro: hover→stop on a single widget in a tight loop, tracking
OS thread count + RAM after each iteration. Finds the EXACT iteration
where things go sideways.

Usage:
    python tests/integration/hover_loop_test.py --source <VIDEO_DIR> --iters 50
"""
import os
import sys
import time
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
os.environ.setdefault('VIDEO_ORGANIZER_QSETTINGS_ORG', 'VOrgHoverLoop')
os.environ.setdefault('VIDEO_ORGANIZER_QSETTINGS_APP', 'HoverLoopRun')

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
    """Number of OS threads in THIS process (cross-platform)."""
    try:
        # Windows: use threading.active_count plus a check via Win32
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
    p.add_argument('--iters', type=int, default=50)
    p.add_argument('--hover_ms', type=int, default=500)
    p.add_argument('--max_threads', type=int, default=200,
                   help='Bail if thread count exceeds this')
    args = p.parse_args()

    log(f"=== HOVER LOOP TEST: {args.iters} iters on {args.source} ===")
    log(f"baseline: threads={thread_count()}, ram={ram_mb():.0f} MB")

    from PyQt6.QtWidgets import QApplication
    from cache_manager import CacheManager
    from app_settings import AppSettings
    from thumbnail_generator import ThumbnailGeneratorService
    from thumbnail_grid_widget import ThumbnailGridWidget

    app = QApplication.instance() or QApplication(sys.argv)

    import tempfile
    cache_dir = Path(tempfile.mkdtemp(prefix='vorg_hover_'))
    cache    = CacheManager(str(cache_dir / 'cache.db'), str(cache_dir / 'thumbs'))
    settings = AppSettings()
    settings.hover_preview_max_gb     = 100.0
    settings.large_file_threshold_mb  = 500
    settings.use_hw_accel             = True

    gen  = ThumbnailGeneratorService(cache)
    grid = ThumbnailGridWidget(gen, settings, cache)
    grid.resize(1600, 1000)

    log(f"after Qt setup: threads={thread_count()}, ram={ram_mb():.0f} MB")

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
    if not actives:
        log("FAIL: no active video widgets")
        sys.exit(1)
    _, w = actives[0]
    log(f"using widget: {os.path.basename(w.video_path)} "
        f"({w._file_size/(1024**3):.2f} GB)")
    log(f"after pick widget: threads={thread_count()}, ram={ram_mb():.0f} MB")

    log("")
    log("=== HOVER LOOP START ===")
    log(f"{'iter':>4} | {'hover_ms':>9} | {'stop_ms':>8} | {'threads':>7} | "
        f"{'delta':>5} | {'ram_mb':>6} | {'play_thread':>11}")
    prev_threads = thread_count()
    baseline_threads = prev_threads

    for i in range(1, args.iters + 1):
        # Hover
        t0 = time.perf_counter()
        w._hovering = True
        w._start_playback()
        pump(app, args.hover_ms)
        hover_ms = (time.perf_counter() - t0) * 1000

        # Stop
        t0 = time.perf_counter()
        w._hovering = False
        w._stop_playback()
        # Give the play_thread a moment to actually die
        pump(app, 300)
        stop_ms = (time.perf_counter() - t0) * 1000

        # Measure
        threads = thread_count()
        ram = ram_mb()
        delta = threads - prev_threads
        prev_threads = threads
        pt = "yes" if w._play_thread is not None else "no"

        log(f"{i:>4} | {hover_ms:>9.0f} | {stop_ms:>8.0f} | "
            f"{threads:>7} | {delta:>+5} | {ram:>6.0f} | {pt:>11}")

        # Bail on leak
        if threads > args.max_threads:
            log(f"⚠️ THREAD LEAK: {threads} threads (baseline {baseline_threads}, "
                f"max {args.max_threads}) — dumping stacks")
            faulthandler.dump_traceback(file=sys.stderr)
            sys.exit(2)

    log("")
    log("=== DONE ===")
    log(f"final: threads={thread_count()}, ram={ram_mb():.0f} MB")
    log(f"leak: {thread_count() - baseline_threads} threads above baseline")

    log("cleanup: shutdown_all_widgets")
    t0 = time.perf_counter()
    try:
        grid.shutdown_all_widgets()
    except Exception:
        import traceback
        traceback.print_exc(file=sys.stderr)
    log(f"shutdown_all_widgets took {(time.perf_counter()-t0)*1000:.0f} ms")

    log("cleanup: gen.cancel_all")
    t0 = time.perf_counter()
    try:
        gen.cancel_all()
    except Exception:
        import traceback
        traceback.print_exc(file=sys.stderr)
    log(f"gen.cancel_all took {(time.perf_counter()-t0)*1000:.0f} ms")

    log(f"after cleanup: threads={thread_count()}, ram={ram_mb():.0f} MB")
    log("exiting cleanly")


if __name__ == '__main__':
    main()
