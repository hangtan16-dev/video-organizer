"""
Reproduce the user's EXACT real-world browsing pattern that hangs at ~video 5.

Per-video pattern:
    1. Preview (hover starts playback, watch for ~3 seconds — frames emit)
    2. Seek (slider drag to different position)
    3. Preview again at new position (~3 seconds — frames emit)
    4. Move to next video (mouseLeave → mouseEnter pattern)

This is MUCH longer than the rapid_switch_test (which only does 200 ms
hovers and never gets to actual frame emission). The user reports the
hang at video 5, so a real reproduction must exercise actual playback,
not just the av.open phase.

Tracks per-video:
    - thread count
    - RAM
    - active play_threads
    - frame count actually emitted (proves playback was real)
    - timing of each step

Run:
    python tests/integration/realistic_browse_test.py --source <VIDEO_DIR> --videos 8
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
os.environ.setdefault('VIDEO_ORGANIZER_QSETTINGS_ORG', 'VOrgRealistic')
os.environ.setdefault('VIDEO_ORGANIZER_QSETTINGS_APP', 'Realistic')

try:
    from hires_timer import request_high_resolution_timer
    request_high_resolution_timer()
except Exception:
    pass

faulthandler.enable(file=sys.stderr)

# Force a thread-stack dump every 5 seconds NO MATTER WHAT. This runs at C
# level so it survives even when the Python interpreter is stuck inside a
# blocking C call (e.g. inside Qt's processEvents waiting on an OS lock).
# Without this we get no visibility into hangs at all.
faulthandler.dump_traceback_later(5, repeat=True, file=sys.stderr)


_log_lock = threading.Lock()

def log(msg, indent=0):
    ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    with _log_lock:
        sys.stderr.write(f"  {ts} {' ' * indent}{msg}\n")
        sys.stderr.flush()


# Cache a single psutil.Process — constructing a fresh one per call re-opens
# the OS process handle every time, and these diagnostics are called 4×/video.
_PSUTIL_PROC = None


def _proc():
    global _PSUTIL_PROC
    if _PSUTIL_PROC is None:
        import psutil
        _PSUTIL_PROC = psutil.Process()
    return _PSUTIL_PROC


def thread_count():
    try:
        return _proc().num_threads()
    except Exception:
        return threading.active_count()


def ram_mb():
    try:
        return _proc().memory_info().rss / (1024 * 1024)
    except Exception:
        return -1


def live_play_threads():
    """Count play_threads that are actually still running."""
    import video_thumbnail_widget
    n = 0
    for t in list(video_thumbnail_widget._running_play_threads):
        try:
            if hasattr(t, 'isRunning') and t.isRunning():
                n += 1
        except RuntimeError:
            pass
    return n


def pump(app, ms, *, frame_callback=None):
    from PyQt6.QtCore import QEventLoop
    deadline = time.perf_counter() + ms / 1000.0
    while time.perf_counter() < deadline:
        app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 20)


def activate_index(grid, idx, app):
    """Return the live widget for item `idx`, or None.

    Offscreen Qt does NOT drive the QScrollArea's scrollbar range, so we can't
    scroll specific items into view the way a real user would (verified: only
    the first viewport-worth ever activates). Instead the harness widens the
    grid's virtualization buffer (see main) so every test-set widget stays
    alive at once. Thumbnail generation + preview behave identically — only
    the (here-irrelevant) create/destroy-on-scroll differs."""
    return grid._active.get(idx)


class FrameTracker:
    """Count emitted frames by listening to the widget's play thread's
    frame_ready signal alongside the widget's own slot. Does NOT monkey-
    patch _on_play_frame — that confused PyQt6's signal-slot bookkeeping
    and caused TypeError on disconnect."""
    def __init__(self, widget):
        self.widget = widget
        self.count = 0
        self._connected_thread = None

    def attach(self):
        """Call AFTER widget._start_playback() so play_thread exists."""
        t = self.widget._play_thread
        if t is not None and hasattr(t, 'frame_ready'):
            try:
                t.frame_ready.connect(self._on_frame)
                self._connected_thread = t
            except (RuntimeError, TypeError):
                pass

    def _on_frame(self, qimg):
        self.count += 1

    def restore(self):
        if self._connected_thread is not None:
            try:
                self._connected_thread.frame_ready.disconnect(self._on_frame)
            except (RuntimeError, TypeError):
                pass
        self._connected_thread = None


# ─── Watchdog: detect GUI thread freezes ───────────────────────────────────
class HangDetector:
    """Fires from a thread if the GUI thread doesn't tick fast enough.

    The main thread ticks `last_seen` every time pump() runs. The watcher
    polls last_seen; if it falls behind by `hang_threshold_s`, the GUI is
    starved — exactly the user's "Not Responding" condition before Windows
    force-closes the app."""
    def __init__(self, hang_threshold_s=3.0):
        self.hang_threshold_s = hang_threshold_s
        self.last_seen = time.perf_counter()
        self._stop = threading.Event()
        self.hang_detected_at = None
        self._t = threading.Thread(target=self._watch, daemon=True)
        self._t.start()

    def tick(self):
        self.last_seen = time.perf_counter()

    def stop(self):
        self._stop.set()

    def _watch(self):
        while not self._stop.is_set():
            now = time.perf_counter()
            gap = now - self.last_seen
            if gap > self.hang_threshold_s and self.hang_detected_at is None:
                self.hang_detected_at = gap
                log(f"⚠️  HANG DETECTED: GUI hasn't ticked in {gap:.1f}s",
                    indent=0)
                log("=== PYTHON THREAD STACKS ===", indent=0)
                faulthandler.dump_traceback(file=sys.stderr)
                sys.stderr.flush()
                # Capture NATIVE (C++) stacks via py-spy on our own PID, so we
                # can see exactly which C call the GUI thread is wedged in
                # (Qt paint? QPixmap? libav? a kernel read?). Run as a child
                # so it can attach to (its parent) us.
                try:
                    import subprocess, os as _os
                    log("=== NATIVE (py-spy) STACKS ===", indent=0)
                    out = subprocess.run(
                        ["py-spy", "dump", "--pid", str(_os.getpid()),
                         "--native"],
                        capture_output=True, text=True, timeout=30)
                    sys.stderr.write(out.stdout + "\n" + out.stderr + "\n")
                    sys.stderr.flush()
                except Exception as e:
                    log(f"py-spy native dump failed: {e}", indent=2)
            time.sleep(0.5)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='<VIDEO_DIR>')
    p.add_argument('--videos', type=int, default=8,
                   help='Number of videos to browse')
    p.add_argument('--preview_ms', type=int, default=3000,
                   help='How long each preview plays')
    p.add_argument('--hang_threshold_s', type=float, default=4.0,
                   help='GUI tick gap that signals a hang')
    p.add_argument('--thumb_wait_s', type=float, default=120.0,
                   help='After browsing, how long to wait for every browsed '
                        'video to finish generating its thumbnail')
    args = p.parse_args()

    log(f"=== REALISTIC BROWSE: {args.videos} videos, "
        f"{args.preview_ms} ms preview each ===")
    log(f"source: {args.source}")
    log(f"baseline: threads={thread_count()}, ram={ram_mb():.0f} MB")

    from PyQt6.QtWidgets import QApplication
    from cache_manager import CacheManager
    from app_settings import AppSettings
    from thumbnail_generator import ThumbnailGeneratorService
    from thumbnail_grid_widget import ThumbnailGridWidget

    app = QApplication.instance() or QApplication(sys.argv)

    import tempfile
    cache_dir = Path(tempfile.mkdtemp(prefix='vorg_realistic_'))
    cache    = CacheManager(str(cache_dir / 'cache.db'), str(cache_dir / 'thumbs'))
    settings = AppSettings()
    settings.hover_preview_max_gb     = 100.0
    settings.large_file_threshold_mb  = 500
    settings.use_hw_accel             = True

    gen  = ThumbnailGeneratorService(cache)
    grid = ThumbnailGridWidget(gen, settings, cache)
    grid.resize(1600, 1000)

    # ── Track thumbnail generation. Connect BEFORE load_folder so we catch
    #    every emission. generated[path] = {seek_times that succeeded};
    #    failed[path] = [reasons]. The grid requests a thumbnail for every
    #    item at load, so a browsed video is "generated" once it emits one. ─
    generated: dict = {}
    failed: dict = {}
    def _on_thumb_ready(path, seek, _qimg, _dur):
        generated.setdefault(path, set()).add(round(float(seek), 2))
    def _on_thumb_failed(path, reason):
        failed.setdefault(path, []).append(reason)
    gen.thumbnail_ready.connect(_on_thumb_ready)
    gen.thumbnail_failed.connect(_on_thumb_failed)

    # ── Cold-cache check: each run uses a brand-new temp cache dir, so NO
    #    thumbnail exists yet — assert it BEFORE load_folder starts generation,
    #    so the test provably begins from a wiped cache (the user's "start by
    #    wiping all thumbnail cache"). ────────────────────────────────────
    thumbs_dir = cache_dir / 'thumbs'
    thumbs_on_disk = (len([f for f in os.listdir(thumbs_dir) if f.endswith('.jpg')])
                      if thumbs_dir.exists() else 0)
    log(f"cold-cache check: {thumbs_on_disk} thumbnails on disk, "
        f"{len(generated)} generated so far (expect 0 / 0)")
    if thumbs_on_disk or generated:
        log("FAIL: thumbnail cache was not empty at start")
        sys.exit(1)

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

    # ── Test set: the first N video items, in display order. We browse each
    #    ONE exactly once (scrolling it into view), so generation stays fresh
    #    to override throughout — not the old "cycle the same visible few". ─
    video_indices = [i for i, it in enumerate(grid._items) if not it.is_folder]
    video_indices = video_indices[:args.videos]
    if not video_indices:
        log("FAIL: no video items in folder")
        sys.exit(1)
    test_paths = [grid._items[i].path for i in video_indices]
    log(f"test set: {len(test_paths)} videos (of {grid._total_count} items) — "
        f"previewing each WITHOUT waiting for its thumbnail")

    # Offscreen Qt won't drive the QScrollArea scrollbar, so we can't scroll
    # items into view. Widen the virtualization buffer enough to keep every
    # test-set widget alive at once, then preview each in place. Generation is
    # unaffected (every item's thumbnail was already requested at load).
    import thumbnail_grid_widget as _tgw
    needed = grid.viewport().height()
    for i in video_indices:
        if i < len(grid._layout_cache) and grid._layout_cache[i] != _tgw._FILTERED_SENTINEL:
            g = grid._layout_cache[i]
            needed = max(needed, g[1] + g[3])
    _tgw._BUFFER_PX = int(needed) + 400
    grid._full_relayout()
    pump(app, 800)
    active_n = sum(1 for i in video_indices if i in grid._active)
    log(f"activated {active_n}/{len(video_indices)} test widgets "
        f"(buffer={_tgw._BUFFER_PX}px, ram={ram_mb():.0f} MB)")

    # Start hang detector
    hangd = HangDetector(hang_threshold_s=args.hang_threshold_s)

    log("")
    log("=== BROWSE START (cold cache: previews must override thumbnail gen) ===")
    log("")
    rng = random.Random(42)
    preview_results = []   # (n, path, name, phase1_frames, phase3_frames)

    for n, idx in enumerate(video_indices, 1):
        path = test_paths[n - 1]
        name = os.path.basename(path)
        w = activate_index(grid, idx, app)
        if w is None or not (hasattr(w, 'video_path') and hasattr(w, '_start_playback')):
            log(f"── VIDEO {n}/{len(video_indices)}: {name[:50]} — "
                f"COULD NOT ACTIVATE (idx {idx}) ──")
            preview_results.append((n, path, name, -1, -1))
            continue
        sz_gb = w._file_size / (1024 ** 3) if getattr(w, '_file_size', 0) > 0 else 0
        log(f"── VIDEO {n}/{len(video_indices)}: {name[:50]} ({sz_gb:.2f} GB) ──")
        log(f"   pre-video: threads={thread_count()}, ram={ram_mb():.0f} MB, "
            f"live_play={live_play_threads()}", indent=2)
        hangd.tick()

        # ── Phase 1: Preview (hover a card that is likely still Generating…) ─
        log("phase 1: preview (start hover, wait for frames)", indent=2)
        tracker = FrameTracker(w)
        t0 = time.perf_counter()
        w._hovering = True
        w._start_playback()
        tracker.attach()
        while time.perf_counter() - t0 < args.preview_ms / 1000.0:
            pump(app, 250)
            hangd.tick()
            if hangd.hang_detected_at is not None:
                log(f"BAILING — hang at video {n}", indent=2)
                tracker.restore()
                sys.exit(2)
        phase1_frames = tracker.count
        tracker.restore()
        log(f"   phase 1 done: {phase1_frames} frames emitted in "
            f"{(time.perf_counter()-t0)*1000:.0f} ms", indent=2)

        # ── Phase 2: live seek (must ALSO override thumbnail gen) ──────────
        dur = w._duration or 60.0
        new_seek = dur * rng.uniform(0.1, 0.9)
        log(f"phase 2: live seek to {new_seek:.1f}s (dur={dur:.0f}s)", indent=2)
        t0 = time.perf_counter()
        w._seek_time = new_seek
        if w._play_thread is not None:
            try:
                w._play_thread.request_seek(new_seek)
            except RuntimeError:
                pass
        try:
            w._apply_seek()
        except Exception as e:
            log(f"   apply_seek error: {e}", indent=2)
        log(f"   seek dispatched in {(time.perf_counter()-t0)*1000:.0f} ms",
            indent=2)
        hangd.tick()

        # ── Phase 3: Preview again at new position ────────────────────────
        log("phase 3: preview at new seek position", indent=2)
        tracker = FrameTracker(w)
        t0 = time.perf_counter()
        if not w._hovering:
            w._hovering = True
        w._start_playback()
        tracker.attach()   # AFTER _start_playback so _play_thread exists
        while time.perf_counter() - t0 < args.preview_ms / 1000.0:
            pump(app, 250)
            hangd.tick()
            if hangd.hang_detected_at is not None:
                log(f"BAILING — hang at video {n} phase 3", indent=2)
                tracker.restore()
                sys.exit(2)
        phase3_frames = tracker.count
        tracker.restore()
        log(f"   phase 3 done: {phase3_frames} frames emitted", indent=2)

        # ── Phase 4: stop & move on ───────────────────────────────────────
        log("phase 4: stop hover (simulate mouseLeave)", indent=2)
        w._hovering = False
        w._stop_playback()
        pump(app, 200)
        hangd.tick()
        preview_results.append((n, path, name, phase1_frames, phase3_frames))
        log(f"   post-video: threads={thread_count()}, ram={ram_mb():.0f} MB, "
            f"live_play={live_play_threads()}", indent=2)
        log("")

    hangd.stop()
    log("=== BROWSE DONE ===")
    log(f"final: threads={thread_count()}, ram={ram_mb():.0f} MB, "
        f"live_play={live_play_threads()}")

    # ── Verdict 1: every preview emitted frames (foreground overrode bg) ──
    # Require frames in BOTH phases: hover (phase 1) AND post-seek (phase 3)
    # must each override background generation and deliver frames.
    preview_fails = [(n, name, f1, f3)
                     for (n, _p, name, f1, f3) in preview_results
                     if f1 < 1 or f3 < 1]
    log("")
    log("=== PREVIEW RESULTS ===")
    log(f"   {len(preview_results) - len(preview_fails)}/{len(preview_results)} "
        f"videos previewed OK (≥1 frame in BOTH hover and post-seek)")
    for (n, name, f1, f3) in preview_fails:
        log(f"   FAIL video {n} {name[:50]}: phase1={f1} phase3={f3}", indent=1)

    # ── Wait for ALL browsed thumbnails to finish generating ──────────────
    # The user asked: after browsing, give generation time and confirm every
    # browsed video ends up with a thumbnail (none stuck "Generating…").
    log("")
    log(f"=== WAITING UP TO {args.thumb_wait_s:.0f}s FOR THUMBNAILS ===")
    try:
        grid._stop_all_previews()   # idle the foreground → background resumes
    except Exception:
        pass
    pump(app, 500)
    twait0 = time.perf_counter()
    last_done = -1
    while time.perf_counter() - twait0 < args.thumb_wait_s:
        pump(app, 500)
        done = sum(1 for p in test_paths if p in generated)
        if done != last_done:
            log(f"   {done}/{len(test_paths)} thumbnails generated "
                f"({len(failed)} failures) @ {time.perf_counter()-twait0:.0f}s",
                indent=1)
            last_done = done
        if done >= len(test_paths):
            break

    missing_paths = [p for p in test_paths if p not in generated]
    log("=== THUMBNAIL RESULTS ===")
    log(f"   {len(test_paths) - len(missing_paths)}/{len(test_paths)} videos "
        f"have a generated thumbnail after {time.perf_counter()-twait0:.0f}s")
    if missing_paths:
        # Diagnostics: distinguish a _pending LEAK (path not queued → never
        # generated) from STARVATION (worker stuck parked) from a cached FAIL.
        try:
            pend = dict(gen._pending)
        except Exception:
            pend = {}
        log(f"   generator state: _pending size={len(pend)}, "
            f"pool active={gen._pool.activeThreadCount()}", indent=1)
    for p in missing_paths:
        reasons = failed.get(p)
        detail = ('FAILED: ' + '; '.join(reasons)) if reasons else 'still pending (stuck?)'
        w = gen._pending.get(p)
        if w is not None:
            wstate = (f"IN _pending [cancelled={getattr(w, '_cancelled', '?')} "
                      f"started_at={getattr(w, '_started_at', '?')} "
                      f"requeue={getattr(w, '_requeue_on_cancel', '?')}]")
        else:
            wstate = "NOT in _pending (LEAKED — never re-queued)"
        try:
            failed_cache = gen._cache.is_thumbnail_failed(p)
        except Exception:
            failed_cache = '?'
        log(f"   MISSING {os.path.basename(p)[:50]}: {detail}", indent=1)
        log(f"            {wstate}; is_failed_cache={failed_cache}", indent=1)

    # ── cleanup ───────────────────────────────────────────────────────────
    log("")
    log("cleanup")
    try:
        grid.shutdown_all_widgets()
    except Exception:
        import traceback
        traceback.print_exc(file=sys.stderr)
    try:
        gen.cancel_all()   # force-close any still-queued workers so we can exit
    except Exception:
        import traceback
        traceback.print_exc(file=sys.stderr)
    try:
        cache.close()   # stop the background DB writer thread
    except Exception:
        pass
    log(f"after cleanup: threads={thread_count()}, ram={ram_mb():.0f} MB")

    # ── Final verdict ─────────────────────────────────────────────────────
    if preview_fails:
        log(f"FAIL: {len(preview_fails)} video(s) did not preview — foreground "
            f"did NOT override thumbnail generation")
        sys.exit(1)
    if missing_paths:
        log(f"FAIL: {len(missing_paths)} thumbnail(s) never generated within "
            f"{args.thumb_wait_s:.0f}s — stuck 'Generating…' regression")
        sys.exit(1)
    log("SUCCESS")


if __name__ == '__main__':
    main()
