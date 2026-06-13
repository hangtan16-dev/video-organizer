"""
Randomized soak test — tries to reproduce the flaky mid-session
"QThread: Destroyed while thread is still running" crash by hammering the exact
interaction mix the user reported: hover-preview, live-seek, hover-stop,
checkbox-select, scroll, and (the prime suspect) drag-of-selected-items, in a
random order across 100+ videos, for a sustained duration.

It is READ-ONLY on the source library:
  • previews/seeks only decode frames (no writes),
  • checkbox toggles are in-memory,
  • the drag path is exercised via a FAKE QDrag whose exec() runs a brief NESTED
    event loop (the real re-entrancy hazard) but performs NO OS drag/drop — so
    no file can ever be moved/copied/deleted.

Crash capture: a Qt message handler flags "QThread: Destroyed …", and
faulthandler dumps the native all-thread stack to stderr on any hard crash.

Run (point --source at a folder with 100+ videos; recursive by default):
    python tests/integration/random_soak_test.py --source <VIDEO_DIR> --duration 120
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
os.environ.setdefault('VIDEO_ORGANIZER_QSETTINGS_ORG', 'VOrgSoak')
os.environ.setdefault('VIDEO_ORGANIZER_QSETTINGS_APP', 'Soak')

try:
    from hires_timer import request_high_resolution_timer
    request_high_resolution_timer()
except Exception:
    pass

faulthandler.enable(file=sys.stderr)
faulthandler.dump_traceback_later(15, repeat=True, file=sys.stderr)

_log_lock = threading.Lock()


def log(msg, indent=0):
    ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    with _log_lock:
        sys.stderr.write(f"  {ts} {' ' * indent}{msg}\n")
        sys.stderr.flush()


# ── crash capture: route Qt messages here; flag the teardown abort warning ──
_CRASH_MSG = None


def _qt_msg_handler(mode, ctx, message):
    global _CRASH_MSG
    with _log_lock:
        sys.stderr.write(f"[Qt] {message}\n")
        sys.stderr.flush()
    if 'Destroyed while thread' in message or 'still running' in message:
        _CRASH_MSG = message
        faulthandler.dump_traceback(file=sys.stderr)
        sys.stderr.flush()


def pump(app, ms):
    from PyQt6.QtCore import QEventLoop
    deadline = time.perf_counter() + ms / 1000.0
    while time.perf_counter() < deadline:
        app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 15)


def live_play_threads():
    import video_thumbnail_widget
    n = 0
    for t in list(video_thumbnail_widget._running_play_threads):
        try:
            if t.isRunning():
                n += 1
        except RuntimeError:
            pass
    return n


def _make_fake_drag(app):
    """QDrag stand-in: NO real OS drag/drop (so nothing can be moved), but exec()
    runs a brief nested event loop — reproducing the re-entrancy of the real
    QDrag.exec() (offscreen the real one is a no-op, so we must simulate it)."""
    from PyQt6.QtCore import Qt, QEventLoop

    class _FakeDrag:
        def __init__(self, parent=None):
            pass

        def setMimeData(self, _m):
            pass

        def setPixmap(self, _p):
            pass

        def setHotSpot(self, _p):
            pass

        def exec(self, *_a, **_k):
            end = time.perf_counter() + 0.10
            while time.perf_counter() < end:
                app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 10)
            return Qt.DropAction.IgnoreAction

        exec_ = exec

    return _FakeDrag


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='<VIDEO_DIR>')
    p.add_argument('--duration', type=float, default=120.0, help='soak seconds')
    p.add_argument('--min-videos', type=int, default=100)
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--recursive', type=int, default=1)
    args = p.parse_args()

    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import qInstallMessageHandler
    from cache_manager import CacheManager
    from app_settings import AppSettings
    from thumbnail_generator import ThumbnailGeneratorService
    from thumbnail_grid_widget import ThumbnailGridWidget
    import thumbnail_grid_widget as tgw

    qInstallMessageHandler(_qt_msg_handler)
    app = QApplication.instance() or QApplication(sys.argv)

    # FAKE drag → exercises the nested-loop re-entrancy WITHOUT moving files.
    tgw.QDrag = _make_fake_drag(app)

    import tempfile
    cache_dir = Path(tempfile.mkdtemp(prefix='vorg_soak_'))
    cache    = CacheManager(str(cache_dir / 'cache.db'), str(cache_dir / 'thumbs'))
    settings = AppSettings()
    settings.hover_preview_max_gb    = 100.0
    settings.large_file_threshold_mb = 500
    settings.use_hw_accel            = True
    settings.recursive_view          = bool(args.recursive)

    gen  = ThumbnailGeneratorService(cache)
    grid = ThumbnailGridWidget(gen, settings, cache)
    grid.set_recursive(bool(args.recursive))
    grid.resize(1600, 1000)
    grid.show()

    log(f"=== RANDOM SOAK: source={args.source} duration={args.duration:.0f}s "
        f"recursive={bool(args.recursive)} seed={args.seed} ===")
    grid.load_folder(args.source)
    deadline = time.perf_counter() + 60
    from PyQt6.QtCore import QEventLoop
    while time.perf_counter() < deadline:
        app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 50)
        if grid._total_count > 0 and grid._batch_idx >= grid._total_count:
            break
    pump(app, 500)
    grid._full_relayout()
    pump(app, 500)

    n_videos = sum(1 for it in grid._items if not it.is_folder)
    log(f"loaded: {grid._total_count} items, {n_videos} videos, "
        f"scrollmax={grid.verticalScrollBar().maximum()}")
    if n_videos < args.min_videos:
        log(f"WARNING: only {n_videos} videos (<{args.min_videos}); soak continues "
            f"but coverage is reduced. Point --source at a bigger/recursive folder.")
    if n_videos == 0:
        log("FAIL: no videos found at source"); sys.exit(1)

    rng = random.Random(args.seed)
    bar = grid.verticalScrollBar()
    counts = {'scroll': 0, 'hover': 0, 'seek': 0, 'stop': 0, 'check': 0, 'drag': 0}
    actions = ['scroll', 'hover', 'hover', 'seek', 'seek', 'stop',
               'check', 'check', 'drag']

    def video_widgets():
        from video_thumbnail_widget import VideoThumbnailWidget
        return [w for w in grid._active.values()
                if isinstance(w, VideoThumbnailWidget)]

    log("=== SOAK START ===")
    t_end = time.perf_counter() + args.duration
    iters = 0
    last_report = time.perf_counter()
    while time.perf_counter() < t_end:
        if _CRASH_MSG is not None:
            break
        iters += 1
        act = rng.choice(actions)
        try:
            if act == 'scroll':
                mx = bar.maximum()
                if mx > 0:
                    bar.setValue(rng.randint(0, mx))
                counts['scroll'] += 1
            else:
                ws = video_widgets()
                if ws:
                    w = rng.choice(ws)
                    if act == 'hover':
                        w._hovering = True
                        w._start_playback()
                        counts['hover'] += 1
                    elif act == 'seek':
                        dur = getattr(w, '_duration', 0) or 60.0
                        w._seek_time = dur * rng.uniform(0.05, 0.95)
                        if getattr(w, '_play_thread', None) is not None:
                            try:
                                w._play_thread.request_seek(w._seek_time)
                            except RuntimeError:
                                pass
                        try:
                            w._apply_seek()
                        except Exception:
                            pass
                        counts['seek'] += 1
                    elif act == 'stop':
                        w._hovering = False
                        w._stop_playback()
                        counts['stop'] += 1
                    elif act == 'check':
                        w._checkbox.setChecked(not w._checkbox.isChecked())
                        counts['check'] += 1
                if act == 'drag':
                    if grid.get_checked_paths():
                        grid._start_drag()   # FakeDrag → nested loop, no real DnD
                        counts['drag'] += 1
        except RuntimeError:
            pass   # a widget was recycled mid-action — exactly the race we hunt
        except Exception as e:
            log(f"action {act} raised {type(e).__name__}: {e}")

        # frequent, short pumps keep timers/signals/finished firing (and let the
        # FakeDrag nested loop interleave with preview teardown)
        pump(app, rng.randint(3, 25))

        now = time.perf_counter()
        if now - last_report > 5.0:
            log(f"  t+{args.duration - (t_end - now):.0f}s iters={iters} "
                f"{counts} active={len(video_widgets())} "
                f"live_play={live_play_threads()} "
                f"checked={len(grid.get_checked_paths())}")
            last_report = now

    log("=== SOAK LOOP ENDED ===")
    log(f"total iters={iters} actions={counts}")

    crashed = _CRASH_MSG is not None
    if crashed:
        log(f"!!! CRASH SIGNAL: {_CRASH_MSG}")
        log("Native stack was dumped above. live_play=%d" % live_play_threads())

    # cleanup
    try:
        grid._stop_all_previews()
        pump(app, 300)
        grid.shutdown_all_widgets()
        gen.cancel_all()
        cache.close()
    except Exception:
        import traceback
        traceback.print_exc(file=sys.stderr)

    if crashed:
        log("FAIL: reproduced the QThread teardown crash")
        sys.exit(3)
    log("SUCCESS: no crash during soak")


if __name__ == '__main__':
    main()
