"""
Real stress test that runs for MINUTES, not seconds.

Reproduces the user-reported hang:
    1. Open folder with videos
    2. Scroll around up/down a bit
    3. Select a video, hover to preview, wait 1 second
    4. Seek to new location, hover to preview again
    5. Seek to another new location, hover but DO NOT wait
    6. Move to a different video and seek before previous playback starts
    7. Repeat with variety

Source: <VIDEO_DIR> (real user data). NEVER modifies it.

Run:
    python tests/integration/stress_workflow.py
    python tests/integration/stress_workflow.py --iterations 50
    python tests/integration/stress_workflow.py --source <TEST_DIR>  # smaller dataset
"""
import os
import sys
import time
import random
import argparse
import datetime
import threading
import traceback
import faulthandler
from pathlib import Path

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

os.environ['OPENCV_FFMPEG_LOGLEVEL'] = '-8'
os.environ['OPENCV_LOG_LEVEL']       = 'OFF'
os.environ['AV_LOG_LEVEL']           = 'quiet'
_DECODE_THREADS = max(1, int((os.cpu_count() or 4) * 0.75))
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = (
    f'threads;{_DECODE_THREADS}|thread_type;slice'
)
os.environ.setdefault('VIDEO_ORGANIZER_QSETTINGS_ORG', 'VideoOrganizerStress')
os.environ.setdefault('VIDEO_ORGANIZER_QSETTINGS_APP', 'StressRun')

try:
    from hires_timer import request_high_resolution_timer
    request_high_resolution_timer()
except Exception:
    pass

faulthandler.enable(file=sys.stderr)

_log_lock = threading.Lock()
_t0_overall = time.perf_counter()

def log(msg, indent=0, level='INFO'):
    ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    elapsed = time.perf_counter() - _t0_overall
    line = f"  {ts} [+{elapsed:6.1f}s] {' ' * indent}[{level}] {msg}"
    with _log_lock:
        sys.stderr.write(line + '\n')
        sys.stderr.flush()


def fail(msg, dump_threads=True):
    log(msg, level='FAIL')
    if dump_threads:
        log("=== ALL THREAD STACKS ===", level='FAIL')
        faulthandler.dump_traceback(file=sys.stderr)
    sys.stderr.flush()
    sys.exit(1)


# ─── Per-step watchdog ──────────────────────────────────────────────────────
class StepWatchdog:
    """Fires `dump_traceback` if a single step takes longer than `budget_s`.
    Useful for catching the SPECIFIC step that hangs in a long workflow."""
    def __init__(self, label, budget_s):
        self.label = label
        self.budget_s = budget_s
        self._timer = threading.Timer(budget_s, self._dump)
        self._timer.daemon = True
        self._t0 = None

    def __enter__(self):
        self._t0 = time.perf_counter()
        self._timer.start()
        return self

    def __exit__(self, *exc):
        self._timer.cancel()
        dt = time.perf_counter() - self._t0
        if dt > self.budget_s * 0.5:
            log(f"⚠️  step '{self.label}' took {dt:.2f}s "
                f"(budget {self.budget_s}s) — close to limit",
                indent=4, level='WARN')

    def _dump(self):
        dt = time.perf_counter() - self._t0
        log(f"⚠️  WATCHDOG FIRED: step '{self.label}' has been running "
            f"{dt:.1f}s (budget {self.budget_s}s). Dumping thread stacks:",
            level='WATCHDOG')
        faulthandler.dump_traceback(file=sys.stderr)
        sys.stderr.flush()


# ─── Qt event pump ──────────────────────────────────────────────────────────
def pump(app, ms):
    """Pump Qt events for ms milliseconds."""
    from PyQt6.QtCore import QEventLoop
    deadline = time.perf_counter() + ms / 1000.0
    while time.perf_counter() < deadline:
        app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 20)


def pump_until(app, predicate, max_ms, label):
    """Pump until predicate True or timeout. Returns (ok, elapsed_ms)."""
    from PyQt6.QtCore import QEventLoop
    t0 = time.perf_counter()
    deadline = t0 + max_ms / 1000.0
    while time.perf_counter() < deadline:
        if predicate():
            return True, (time.perf_counter() - t0) * 1000
        app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 20)
    return False, max_ms


# ─── Test helpers ───────────────────────────────────────────────────────────
def get_active_video_widgets(grid):
    """Return active widgets that are videos (not folder cards)."""
    return [(idx, w) for idx, w in sorted(grid._active.items())
            if hasattr(w, 'video_path') and hasattr(w, '_start_playback')]


def stop_playback_safe(widget, app, max_ms=500):
    """Stop a widget's hover playback and wait briefly for the thread to die.
    Does NOT block long — we want to verify stop() is interruptible, not
    sit forever."""
    widget._hovering = False
    widget._stop_playback()
    pump(app, max_ms)


# ─── The stress test itself ─────────────────────────────────────────────────
def stress_run(args):
    log("=" * 70, indent=0)
    log("STRESS TEST — runs for minutes, not seconds", indent=0)
    log("=" * 70, indent=0)
    log(f"source:     {args.source}", indent=0)
    log(f"iterations: {args.iterations}", indent=0)
    log(f"variety:    scroll, hover, seek, hover, seek, switch-mid-playback", indent=0)
    log("", indent=0)

    src = Path(args.source)
    if not src.is_dir():
        fail(f"{src} is not a directory")

    log("setting up Qt app + grid + generator", indent=0)
    from PyQt6.QtWidgets import QApplication
    from cache_manager import CacheManager
    from app_settings import AppSettings
    from thumbnail_generator import ThumbnailGeneratorService
    from thumbnail_grid_widget import ThumbnailGridWidget

    app = QApplication.instance() or QApplication(sys.argv)

    import tempfile
    cache_dir = Path(tempfile.mkdtemp(prefix='vorg_stress_'))
    cache    = CacheManager(str(cache_dir / 'cache.db'), str(cache_dir / 'thumbs'))
    settings = AppSettings()
    settings.hover_preview_max_gb     = 100.0
    settings.large_file_threshold_mb  = 500
    settings.use_hw_accel             = True
    settings.recursive_view           = False

    gen  = ThumbnailGeneratorService(cache)
    grid = ThumbnailGridWidget(gen, settings, cache)
    grid.resize(1600, 1000)

    # Load the folder
    log(f"loading {src}", indent=0)
    with StepWatchdog("load_folder", 5):
        t0 = time.perf_counter()
        grid.load_folder(str(src))
        load_ms = (time.perf_counter() - t0) * 1000
    log(f"load_folder() returned in {load_ms:.1f} ms", indent=2)

    # Wait for scan to populate items
    log("waiting for scan to populate items", indent=0)
    with StepWatchdog("scan", 30):
        ok, scan_ms = pump_until(app,
            lambda: grid._total_count > 0 and grid._batch_idx >= grid._total_count,
            max_ms=30000, label="scan")
    if not ok:
        fail(f"scan did not complete in 30s; have {len(grid.get_all_items())} items")
    n_videos = len([i for i in grid.get_all_items() if not i.is_folder])
    log(f"scan complete: {n_videos} videos in {scan_ms:.0f} ms", indent=2)

    # Need active widgets to interact with — pump a bit so virtual scroll
    # materializes them.
    pump(app, 500)
    grid._full_relayout()
    pump(app, 500)
    actives = get_active_video_widgets(grid)
    if len(actives) < 2:
        fail(f"need ≥2 active widgets, got {len(actives)}")
    log(f"have {len(actives)} active widgets to play with", indent=2)

    # ── Iteration loop ─────────────────────────────────────────────────────
    log("", indent=0)
    log(f"=== starting {args.iterations} stress iterations ===", indent=0)
    log("", indent=0)

    rng = random.Random(args.seed)
    total_hovers = 0
    total_seeks  = 0
    total_switches = 0
    failed_steps = []

    iter_t0 = time.perf_counter()
    for i in range(1, args.iterations + 1):
        # Re-fetch active widgets every iteration — they change with scroll
        actives = get_active_video_widgets(grid)
        if len(actives) < 2:
            log(f"iter {i}: lost active widgets, scrolling to refresh", indent=0)
            grid.verticalScrollBar().setValue(0)
            pump(app, 500)
            actives = get_active_video_widgets(grid)
            if len(actives) < 2:
                log(f"iter {i}: SKIP, not enough widgets", indent=2, level='WARN')
                continue

        # Pick two random widgets for this iteration
        w1_idx, w1 = rng.choice(actives)
        actives2 = [(i, w) for i, w in actives if i != w1_idx]
        w2_idx, w2 = rng.choice(actives2)

        name1 = os.path.basename(w1.video_path)[:40]
        name2 = os.path.basename(w2.video_path)[:40]
        log(f"── iter {i}/{args.iterations}: w1={name1} → w2={name2} ──",
            indent=0)

        try:
            # ── Step A: Scroll ──────────────────────────────────────────────
            scroll_to = rng.randint(0, max(0, grid.verticalScrollBar().maximum()))
            with StepWatchdog(f"i{i} scroll", 3):
                grid.verticalScrollBar().setValue(scroll_to)
                pump(app, 200)   # let virtual scroll catch up

            # ── Step B: Hover w1, wait 1s ───────────────────────────────────
            # In production this is: mouse enters widget → hover_delay timer
            # fires → _start_playback. Here we trigger directly.
            with StepWatchdog(f"i{i} hover w1", 5):
                w1._hovering = True
                w1._start_playback()
                pump(app, 1000)
                total_hovers += 1

            # ── Step C: Move slider (request_seek on existing play_thread) ──
            # Production: dragging the slider while hovering sends request_seek
            # to the live thread. It does NOT create a new play_thread. The
            # thread seeks the container in-place.
            dur1 = w1._duration or 60.0
            seek1 = dur1 * rng.uniform(0.1, 0.9)
            w1._seek_time = seek1
            with StepWatchdog(f"i{i} request_seek w1", 3):
                if w1._play_thread is not None:
                    w1._play_thread.request_seek(seek1)
                pump(app, 1000)   # let the thread seek + decode a few frames
                total_seeks += 1

            # ── Step D: Seek again, this time via slider-release path ───────
            seek2 = dur1 * rng.uniform(0.1, 0.9)
            w1._seek_time = seek2
            with StepWatchdog(f"i{i} apply_seek w1 (slider release)", 3):
                # _apply_seek triggers a static thumbnail regen — does NOT
                # stop the play_thread. The play_thread keeps streaming.
                w1._apply_seek()
                pump(app, 100)
                total_seeks += 1

            # ── Step E: BEFORE w1 playback fully stabilizes, switch to w2 ──
            # Real user repro: cursor moves to widget B. _stop_playback on
            # w1 is triggered by mouseLeave. Then mouseEnter on w2 starts
            # its hover delay → _start_playback on w2.
            with StepWatchdog(f"i{i} stop w1 (mouseLeave)", 3):
                stop_playback_safe(w1, app, max_ms=300)

            with StepWatchdog(f"i{i} hover w2 + seek", 5):
                w2._hovering = True
                w2._start_playback()
                pump(app, 200)   # let it begin opening
                dur2 = w2._duration or 60.0
                seek3 = dur2 * rng.uniform(0.1, 0.9)
                w2._seek_time = seek3
                if w2._play_thread is not None:
                    w2._play_thread.request_seek(seek3)
                pump(app, 800)
                total_switches += 1
                total_seeks += 1

            # ── Step F: stop w2 (mouseLeave) ────────────────────────────────
            with StepWatchdog(f"i{i} stop w2", 3):
                stop_playback_safe(w2, app, max_ms=300)

        except SystemExit:
            raise
        except Exception as e:
            log(f"iter {i} EXCEPTION: {e}", indent=2, level='ERROR')
            traceback.print_exc(file=sys.stderr)
            failed_steps.append((i, str(e)))

        # Heartbeat
        if i % 5 == 0:
            elapsed = time.perf_counter() - iter_t0
            rate = i / elapsed
            log(f"  [heartbeat] {i} iters in {elapsed:.1f}s ({rate:.2f}/s), "
                f"{total_hovers} hovers, {total_seeks} seeks, "
                f"{total_switches} switches, {len(failed_steps)} fails",
                indent=2)

    elapsed = time.perf_counter() - iter_t0
    log("", indent=0)
    log("=" * 70, indent=0)
    log(f"DONE — {args.iterations} iters in {elapsed:.1f}s "
        f"({args.iterations/elapsed:.2f}/s)", indent=0)
    log(f"  hovers: {total_hovers}", indent=2)
    log(f"  seeks: {total_seeks}", indent=2)
    log(f"  switches: {total_switches}", indent=2)
    log(f"  failures: {len(failed_steps)}", indent=2)
    if failed_steps:
        log("FAILED STEPS:", level='FAIL')
        for i, err in failed_steps:
            log(f"  iter {i}: {err}", indent=2, level='FAIL')
        sys.exit(1)
    log("=" * 70, indent=0)

    # Cleanup
    log("cleanup", indent=0)
    try:
        grid.shutdown_all_widgets()
    except Exception:
        log("shutdown_all_widgets failed (continuing)", indent=2, level='WARN')
    try:
        gen.cancel_all()
    except Exception:
        pass


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='<VIDEO_DIR>', help='Folder to load')
    p.add_argument('--iterations', type=int, default=25,
                   help='Number of stress iterations (each ~3s)')
    p.add_argument('--seed', type=int, default=42, help='RNG seed')
    args = p.parse_args()

    try:
        stress_run(args)
    except SystemExit:
        raise
    except Exception:
        log("UNCAUGHT:", level='FATAL')
        traceback.print_exc(file=sys.stderr)
        sys.exit(2)
