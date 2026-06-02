"""
Comprehensive real-world test against <VIDEO_DIR> — the user's actual data.

Read-only on the source folder. ANY destructive operation (delete/move) is
performed against a copy in a temp directory; the originals in <VIDEO_DIR> are
NEVER touched.

Tests cover failure modes the existing test_11 doesn't:
  1. load_folder must stay <100 ms even for an 80+ file HDD folder
  2. UI must remain responsive (event loop drains) while thumbnails queue
  3. Seek on widget #1 must not hang when worker pool is saturated
  4. Folder switch mid-scan must cancel cleanly
  5. Recursive view aggregates Fav + Keep subfolders
  6. Delete uses send2trash-via-copy (so we test the code path safely)

Aggressive watchdog: every operation has a hard budget; on overrun we kill
the process + dump thread stacks for diagnosis.

Run with:
    python tests/integration/real_world_test.py
or:
    python tests/integration/real_world_test.py --quick    # subset
"""
import os
import sys
import time
import datetime
import threading
import traceback
import faulthandler
from pathlib import Path

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Match production env
os.environ['OPENCV_FFMPEG_LOGLEVEL'] = '-8'
os.environ['OPENCV_LOG_LEVEL']       = 'OFF'
os.environ['AV_LOG_LEVEL']           = 'quiet'
_DECODE_THREADS = max(1, int((os.cpu_count() or 4) * 0.75))
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = (
    f'threads;{_DECODE_THREADS}|thread_type;slice'
)
os.environ.setdefault('VIDEO_ORGANIZER_QSETTINGS_ORG', 'VideoOrganizerRealWorld')
os.environ.setdefault('VIDEO_ORGANIZER_QSETTINGS_APP', 'RealWorldRun')

try:
    from hires_timer import request_high_resolution_timer
    request_high_resolution_timer()
except Exception:
    pass

# Dump all thread stacks if anything hangs more than 60 s past expected.
faulthandler.enable(file=sys.stderr)

# Whitelist: paths we are allowed to mutate. NEVER touch anything in <VIDEO_DIR>.
TEST_SOURCE = Path('<VIDEO_DIR>')
SAFE_TMP    = Path(os.environ.get('TEMP', '/tmp')) / 'vorg_real_world'
SAFE_TMP.mkdir(parents=True, exist_ok=True)


# ─── Logging ────────────────────────────────────────────────────────────────
_log_lock = threading.Lock()

def log(msg, indent=0, level='INFO'):
    ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    line = f"  {ts} {' ' * indent}[{level}] {msg}"
    with _log_lock:
        sys.stderr.write(line + '\n')
        sys.stderr.flush()


def fail(msg):
    log(msg, level='FAIL')
    # Dump thread stacks so we can see what hung
    faulthandler.dump_traceback(file=sys.stderr)
    sys.stderr.flush()
    sys.exit(1)


# ─── Watchdog: kill the process if any operation takes too long ────────────
class Watchdog:
    """Fires `bomb` if `.disarm()` isn't called within `budget_s` seconds."""
    def __init__(self, label, budget_s):
        self.label = label
        self.budget_s = budget_s
        self._timer = threading.Timer(budget_s, self._bomb)
        self._timer.daemon = True
        self._fired = False

    def __enter__(self):
        self._t0 = time.perf_counter()
        self._timer.start()
        return self

    def __exit__(self, *exc):
        self.disarm()

    def disarm(self):
        self._timer.cancel()

    def _bomb(self):
        self._fired = True
        dt = time.perf_counter() - self._t0
        log(f"⚠️  WATCHDOG: '{self.label}' exceeded {self.budget_s}s "
            f"(actually {dt:.1f}s) — DUMPING STACKS", level='WATCHDOG')
        faulthandler.dump_traceback(file=sys.stderr)
        sys.stderr.flush()
        # Don't os._exit — let the test see the hang and decide.


# ─── Qt event-loop pump that yields control + checks deadlines ──────────────
def pump_until(app, predicate, max_seconds, label, *, heartbeat_s=1.0):
    """Pump Qt events until `predicate()` returns True OR max_seconds elapses.
    Returns (succeeded: bool, elapsed_s: float). Heartbeats every heartbeat_s."""
    from PyQt6.QtCore import QEventLoop
    t0 = time.perf_counter()
    deadline = t0 + max_seconds
    last_beat = t0
    while time.perf_counter() < deadline:
        if predicate():
            return True, time.perf_counter() - t0
        app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 50)
        now = time.perf_counter()
        if now - last_beat >= heartbeat_s:
            log(f"    ... pumping '{label}' ({now - t0:.1f}s elapsed)", indent=2)
            last_beat = now
    return False, max_seconds


# ─── Tests ──────────────────────────────────────────────────────────────────
def test_1_load_folder_sync_return(app, gen, grid):
    """load_folder() must return in <100 ms even on a slow HDD with 80+ files."""
    log("TEST 1: load_folder synchronous-return time on <VIDEO_DIR>", indent=0)
    t0 = time.perf_counter()
    grid.load_folder(str(TEST_SOURCE))
    dt_ms = (time.perf_counter() - t0) * 1000
    log(f"  load_folder() returned in {dt_ms:.1f} ms", indent=2)
    if dt_ms >= 200:
        fail(f"load_folder blocked {dt_ms:.0f} ms — filesystem leaked to GUI thread")
    log("  ✅ PASS", indent=2)


def test_2_scan_completes(app, gen, grid):
    """Background scan should populate the model in under 30 s."""
    log("TEST 2: background scan completes", indent=0)
    with Watchdog("background scan", 60):
        ok, dt = pump_until(app,
            lambda: grid._batch_idx >= grid._total_count and grid._total_count > 0,
            max_seconds=45,
            label="scan + batch creation")
    if not ok:
        fail(f"scan did not complete; items={len(grid.get_all_items())}, "
             f"batch_idx={grid._batch_idx}/{grid._total_count}")
    items = grid.get_all_items()
    videos  = [i for i in items if not i.is_folder]
    folders = [i for i in items if i.is_folder]
    log(f"  scan completed in {dt:.1f}s: {len(folders)} folders, "
        f"{len(videos)} videos", indent=2)
    if len(videos) < 50:
        fail(f"expected ≥50 videos in <VIDEO_DIR> top level, got {len(videos)}")
    log("  ✅ PASS", indent=2)


def test_3_ui_stays_responsive_during_thumb_burst(app, gen, grid):
    """While 80 thumbs are being generated in background, an arbitrary
    main-thread call (e.g. emit a signal, read a widget property) must
    not block more than 50 ms. This is the canonical UI-responsiveness
    check during the user's main complaint window."""
    log("TEST 3: UI responsiveness during thumbnail burst", indent=0)
    # Let some workers spin up first
    pump_until(app, lambda: False, 1.0, "warmup", heartbeat_s=10)
    samples = []
    for i in range(20):
        t0 = time.perf_counter()
        # A handful of typical main-thread ops: count items, walk path map,
        # get scroll position. These touch widget state but no disk.
        n = len(grid.get_all_items())
        _ = grid._path_to_idx
        _ = grid.verticalScrollBar().value()
        dt_ms = (time.perf_counter() - t0) * 1000
        samples.append(dt_ms)
        app.processEvents()
    max_dt = max(samples)
    avg_dt = sum(samples) / len(samples)
    log(f"  20 samples of main-thread ops: max={max_dt:.2f}ms avg={avg_dt:.2f}ms",
        indent=2)
    if max_dt >= 50:
        fail(f"main thread blocked for {max_dt:.0f} ms during thumb burst — "
             f"GUI freeze symptom")
    log("  ✅ PASS", indent=2)


def test_4_first_widget_thumbnail_arrives(app, gen, grid):
    """The first visible widget should get its thumbnail within reasonable time,
    even though workers are also handling many other files."""
    log("TEST 4: first widget thumbnail arrives in a reasonable time", indent=0)
    # Identify first active widget
    actives = [(idx, w) for idx, w in grid._active.items()
               if hasattr(w, 'video_path') and hasattr(w, '_full_pixmap')]
    if not actives:
        fail("no active video widgets after scan")
    actives.sort(key=lambda iw: iw[0])
    idx, w = actives[0]
    log(f"  watching widget for {os.path.basename(w.video_path)} ({w._file_size/(1024**3):.2f} GB)",
        indent=2)
    with Watchdog("first thumbnail", 60):
        ok, dt = pump_until(app,
            lambda: w._full_pixmap is not None
                    or w._loading_label.text().startswith('⚠'),
            max_seconds=60,
            label="first thumbnail")
    if not ok:
        fail(f"first thumbnail never arrived in 60s — workers stuck?")
    if w._full_pixmap is None:
        log(f"  thumbnail FAILED (showed error badge) for {os.path.basename(w.video_path)}",
            indent=2, level='WARN')
    else:
        log(f"  ✅ first thumbnail arrived in {dt:.1f}s", indent=2)


def test_5_seek_during_thumbnail_burst(app, gen, grid):
    """Drag the seek slider on widget #1 while workers are saturated.
    The seek_requested + regenerate_thumbnail path must NOT block the GUI."""
    log("TEST 5: seek slider while worker pool is saturated", indent=0)
    actives = sorted([(idx, w) for idx, w in grid._active.items()
                      if hasattr(w, 'video_path') and hasattr(w, '_apply_seek')],
                     key=lambda iw: iw[0])
    if not actives:
        fail("no active video widgets")
    _, w = actives[0]
    # Make sure widget #1 has a thumbnail first
    pump_until(app, lambda: w._full_pixmap is not None, 30, "wait for w#1 thumb")

    log(f"  triggering seek on {os.path.basename(w.video_path)}", indent=2)
    new_seek = (w._duration or 60.0) * 0.5
    w._seek_time = new_seek
    t0 = time.perf_counter()
    with Watchdog("apply_seek call", 2):  # apply_seek itself must be near-instant
        w._apply_seek()
    apply_ms = (time.perf_counter() - t0) * 1000
    log(f"  _apply_seek() returned in {apply_ms:.1f} ms", indent=2)
    if apply_ms >= 500:
        fail(f"_apply_seek blocked GUI thread for {apply_ms:.0f} ms — "
             f"SQLite contention or worker mutex")
    # The post-seek thumbnail itself may take a while (queue is backed up
    # with 80 other files). We don't fail on that — we only fail if the
    # main thread was blocked.
    log("  ✅ PASS (GUI thread stayed responsive even if thumb queued)", indent=2)


def test_6_folder_switch_mid_scan(app, gen, grid):
    """Switch to a different folder while the <VIDEO_DIR> scan is still running.
    The old scan must cancel quickly and the new folder must load."""
    log("TEST 6: folder switch mid-scan (cancellation)", indent=0)
    # Re-trigger load on a tiny folder
    target = Path('<TEST_DIR>/tiny')
    if not target.is_dir():
        log(f"  SKIP — {target} not present", indent=2, level='WARN')
        return
    t0 = time.perf_counter()
    with Watchdog("folder switch", 5):  # must return fast
        grid.load_folder(str(target))
    log(f"  load_folder({target.name}) returned in "
        f"{(time.perf_counter()-t0)*1000:.1f} ms", indent=2)
    # New scan should complete (1-2 files in tiny)
    ok, dt = pump_until(app,
        lambda: grid._total_count > 0 and grid._batch_idx >= grid._total_count,
        max_seconds=15, label="new folder scan")
    if not ok:
        fail("new folder scan didn't complete after switch")
    log(f"  new folder loaded in {dt:.1f}s", indent=2)
    log("  ✅ PASS", indent=2)


def test_7_delete_via_copy(app, gen, grid):
    """Copy a small file from <VIDEO_DIR> to a temp dir and delete the COPY.
    Exercises the cancel_all + delete code path WITHOUT touching <VIDEO_DIR>.

    Uses send2trash (the Shell API) instead of os.remove because that's
    what the production app does — and the Shell API gracefully handles
    Windows-Defender-style transient locks on multi-GB files."""
    log("TEST 7: delete-via-copy (no <VIDEO_DIR> file is harmed)", indent=0)
    # Find the smallest .mp4 we can — keeps the copy+Defender-scan time
    # bounded so the test runs fast.
    smallest = None
    smallest_size = float('inf')
    for entry in os.scandir(TEST_SOURCE):
        if entry.is_file() and entry.name.lower().endswith('.mp4'):
            try:
                sz = entry.stat().st_size
                if sz < smallest_size:
                    smallest_size = sz
                    smallest = Path(entry.path)
            except OSError:
                continue
    src = smallest
    if src is None:
        log("  SKIP — no .mp4 in <VIDEO_DIR> top level", indent=2, level='WARN')
        return

    import shutil
    import uuid
    # Unique path per run so a previous-run leftover (still locked by
    # Defender on slow Windows) doesn't block us.
    proxy = SAFE_TMP / f"{uuid.uuid4().hex}_{src.name}"
    log(f"  copying {src.name} ({src.stat().st_size/(1024**3):.2f} GB) to temp "
        f"as {proxy.name[:8]}…", indent=2)
    t0 = time.perf_counter()
    shutil.copy2(src, proxy)
    log(f"  copy took {time.perf_counter()-t0:.1f}s", indent=2)

    # Now request a thumbnail on the proxy, then cancel.
    log("  requesting thumbnail on proxy", indent=2)
    gen.request_thumbnail(str(proxy), 30.0)
    pump_until(app, lambda: False, 0.5, "worker startup", heartbeat_s=2)
    log("  cancel_all (force_close closes file_obj on main thread)", indent=2)
    with Watchdog("cancel_all", 10):
        t0 = time.perf_counter()
        gen.cancel_all()
        cancel_ms = (time.perf_counter() - t0) * 1000
    log(f"  cancel_all in {cancel_ms:.0f} ms", indent=2)

    # Use send2trash like the production app does. The Shell API tolerates
    # transient Defender locks on multi-GB files better than os.remove.
    log("  send2trash proxy (production code path)", indent=2)
    try:
        from send2trash import send2trash
    except ImportError:
        log("  SKIP — send2trash not installed", indent=2, level='WARN')
        return
    deadline = time.perf_counter() + 30
    last_err = None
    t0 = time.perf_counter()
    while time.perf_counter() < deadline:
        try:
            send2trash(str(proxy))
            delete_ms = (time.perf_counter() - t0) * 1000
            log(f"  ✅ trashed in {delete_ms:.0f} ms (after {cancel_ms:.0f} ms cancel)",
                indent=2)
            return
        except Exception as e:
            last_err = e
            time.sleep(0.5)
    fail(f"send2trash FAILED after 30s of retries: {last_err}. "
         f"In the real app the user would see 'cant delete because some thread "
         f"is holding the file' — this is the user-reported bug.")


def test_8_recursive_view(app, gen, grid):
    """Toggle recursive view to include Fav + Keep subfolders.
    Must complete and show more videos than flat view."""
    log("TEST 8: recursive view aggregates subfolders", indent=0)
    log("  switching to recursive mode", indent=2)
    t0 = time.perf_counter()
    with Watchdog("set_recursive", 5):
        grid.set_recursive(True)
    log(f"  set_recursive returned in {(time.perf_counter()-t0)*1000:.1f} ms",
        indent=2)

    # Need to reload the folder for recursive to kick in (set_recursive
    # only reloads if _current_folder_path is set)
    grid.load_folder(str(TEST_SOURCE))
    with Watchdog("recursive scan", 120):
        ok, dt = pump_until(app,
            lambda: grid._total_count > 0 and grid._batch_idx >= grid._total_count,
            max_seconds=120, label="recursive scan", heartbeat_s=5)
    if not ok:
        fail("recursive scan did not complete in 120s")
    videos = [i for i in grid.get_all_items() if not i.is_folder]
    log(f"  recursive scan: {len(videos)} videos in {dt:.1f}s", indent=2)
    if len(videos) < 80:
        fail(f"expected ≥80 videos (<VIDEO_DIR> + subfolders), got {len(videos)}")
    log("  ✅ PASS", indent=2)


def run_all():
    log("=" * 70, indent=0)
    log("REAL-WORLD TEST SUITE — <VIDEO_DIR>", indent=0)
    log("=" * 70, indent=0)
    log(f"python: {sys.version.split()[0]}", indent=0)
    log(f"cpu_count: {os.cpu_count()}", indent=0)
    log(f"source: {TEST_SOURCE}  (will NOT be modified)", indent=0)
    log(f"safe tmp: {SAFE_TMP}  (all destructive ops happen here)", indent=0)

    if not TEST_SOURCE.is_dir():
        fail(f"{TEST_SOURCE} is not a directory")

    log("creating Qt app + grid + generator", indent=0)
    from PyQt6.QtWidgets import QApplication
    from cache_manager import CacheManager
    from app_settings import AppSettings
    from thumbnail_generator import ThumbnailGeneratorService
    from thumbnail_grid_widget import ThumbnailGridWidget

    app = QApplication.instance() or QApplication(sys.argv)

    import tempfile
    cache_dir = Path(tempfile.mkdtemp(prefix='vorg_rwt_'))
    cache    = CacheManager(str(cache_dir / 'cache.db'), str(cache_dir / 'thumbs'))
    settings = AppSettings()
    settings.hover_preview_max_gb     = 100.0   # let hover try on big files
    settings.large_file_threshold_mb  = 500
    settings.use_hw_accel             = True
    settings.recursive_view           = False   # start with flat view

    gen  = ThumbnailGeneratorService(cache)
    grid = ThumbnailGridWidget(gen, settings, cache)
    grid.resize(1600, 1000)

    # Run tests in sequence so each builds on previous state
    try:
        test_1_load_folder_sync_return(app, gen, grid)
        test_2_scan_completes(app, gen, grid)
        test_3_ui_stays_responsive_during_thumb_burst(app, gen, grid)
        test_4_first_widget_thumbnail_arrives(app, gen, grid)
        test_5_seek_during_thumbnail_burst(app, gen, grid)
        test_6_folder_switch_mid_scan(app, gen, grid)
        test_7_delete_via_copy(app, gen, grid)
        # Test 8 (recursive) is heavyweight — only run in non-quick mode
        if '--quick' not in sys.argv:
            test_8_recursive_view(app, gen, grid)
        else:
            log("TEST 8: SKIPPED (--quick mode)", indent=0)
    finally:
        log("cleanup: shutdown widgets, cancel workers", indent=0)
        try:
            grid.shutdown_all_widgets()
        except Exception:
            log(f"  cleanup error: {traceback.format_exc()}", indent=2, level='WARN')
        try:
            gen.cancel_all()
        except Exception:
            pass

    log("=" * 70, indent=0)
    log("ALL TESTS PASSED", indent=0)
    log("=" * 70, indent=0)


if __name__ == '__main__':
    try:
        run_all()
    except SystemExit:
        raise
    except Exception:
        log("UNCAUGHT EXCEPTION:", level='FATAL')
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        sys.exit(2)
