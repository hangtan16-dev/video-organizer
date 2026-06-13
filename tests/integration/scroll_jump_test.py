"""
Reproduce the "scroll down, view jumps back up on its own" glitch and measure it.

Mechanism under test: each thumbnail that finishes generating overwrites its
item's 16:9 PLACEHOLDER aspect ratio with the REAL one and triggers a full
relayout. _full_relayout is supposed to PIN the item under the viewport top
(scroll anchoring) so the content the user is looking at doesn't move. This
soak scrolls around a cold-cache 100+ video folder for N seconds while real
thumbnails stream in, and after every settle checks whether the ANCHOR ITEM's
offset under the viewport drifted — i.e. the content jumped without the user
scrolling.

Detection: drift = |(scroll_value - anchor_item_top) - recorded_offset|. If the
anchor holds, drift ≈ 0; a non-zero drift IS the visible jump. We log the worst
offenders with enough state (scrollbar max before/after, canvas height) to tell
a range-clamp from any other cause.

Run:
    python tests/integration/scroll_jump_test.py --source <VIDEO_DIR> --duration 30
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
os.environ.setdefault('VIDEO_ORGANIZER_QSETTINGS_ORG', 'VOrgScrollJump')
os.environ.setdefault('VIDEO_ORGANIZER_QSETTINGS_APP', 'ScrollJump')

faulthandler.enable(file=sys.stderr)
faulthandler.dump_traceback_later(20, repeat=True, file=sys.stderr)

_log_lock = threading.Lock()


def log(msg, indent=0):
    ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    with _log_lock:
        sys.stderr.write(f"  {ts} {' ' * indent}{msg}\n")
        sys.stderr.flush()


def pump(app, ms):
    from PyQt6.QtCore import QEventLoop
    deadline = time.perf_counter() + ms / 1000.0
    while time.perf_counter() < deadline:
        app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 15)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='<VIDEO_DIR>')
    p.add_argument('--duration', type=float, default=30.0)
    p.add_argument('--min-videos', type=int, default=100)
    p.add_argument('--seed', type=int, default=99)
    p.add_argument('--jump-threshold', type=int, default=8,
                   help='px of anchor drift counted as a jump')
    args = p.parse_args()

    from PyQt6.QtWidgets import QApplication
    from cache_manager import CacheManager
    from app_settings import AppSettings
    from thumbnail_generator import ThumbnailGeneratorService
    from thumbnail_grid_widget import ThumbnailGridWidget
    import thumbnail_grid_widget as tgw

    app = QApplication.instance() or QApplication(sys.argv)

    import tempfile
    cache_dir = Path(tempfile.mkdtemp(prefix='vorg_scrolljump_'))   # COLD cache
    cache    = CacheManager(str(cache_dir / 'cache.db'), str(cache_dir / 'thumbs'))
    settings = AppSettings()
    settings.recursive_view = True

    gen  = ThumbnailGeneratorService(cache)
    grid = ThumbnailGridWidget(gen, settings, cache)
    grid.set_recursive(True)
    grid.resize(1400, 900)
    grid.show()

    log(f"=== SCROLL-JUMP SOAK: source={args.source} duration={args.duration:.0f}s "
        f"seed={args.seed} ===")
    grid.load_folder(args.source)
    from PyQt6.QtCore import QEventLoop
    deadline = time.perf_counter() + 60
    while time.perf_counter() < deadline:
        app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 50)
        if grid._total_count > 0 and grid._batch_idx >= grid._total_count:
            break
    pump(app, 500)
    grid._full_relayout()
    pump(app, 500)

    n_videos = sum(1 for it in grid._items if not it.is_folder)
    bar = grid.verticalScrollBar()
    log(f"loaded: {grid._total_count} items, {n_videos} videos, scrollmax={bar.maximum()}")
    if bar.maximum() <= 0:
        log("FAIL: nothing to scroll (canvas not taller than viewport)"); sys.exit(1)

    rng = random.Random(args.seed)
    jumps = []          # (drift, sv0, sv1, anchor_idx, off0, off1, max0, max1, ch)
    checks = 0

    def scroll_to(v):
        v = max(0, min(int(v), bar.maximum()))
        bar.setValue(v)

    from video_thumbnail_widget import VideoThumbnailWidget

    def active_vids():
        return [w for w in grid._active.values()
                if isinstance(w, VideoThumbnailWidget)]

    log("=== SCROLL SOAK START ===")
    t_end = time.perf_counter() + args.duration
    last_report = time.perf_counter()
    from PyQt6.QtCore import QEventLoop
    while time.perf_counter() < t_end:
        # Mimic the user CLICK-SELECTING: check + FOCUS a checkbox on a visible
        # card. When we then scroll it out of view it gets recycled WHILE
        # focused; Qt reassigns focus, which can make QScrollArea auto-scroll to
        # the new focus widget — the suspected "jumps back up".
        vids = active_vids()
        if vids and rng.random() < 0.8:
            w = vids[rng.randrange(len(vids))]
            try:
                w._checkbox.setChecked(not w._checkbox.isChecked())
                w._checkbox.setFocus()
            except RuntimeError:
                pass

        mx = bar.maximum()
        mode = rng.random()
        if mode < 0.5:
            scroll_to(bar.value() + rng.randint(300, 1600))   # fast down
        elif mode < 0.8:
            scroll_to(rng.randint(int(mx * 0.5), mx))          # deep / near-bottom
        else:
            scroll_to(rng.randint(0, mx))                      # anywhere

        # Record the content the user is "looking at" RIGHT after the scroll.
        sv0 = bar.value()
        max0 = bar.maximum()
        a_idx, off0 = grid._capture_scroll_anchor(sv0)
        if a_idx is None:
            pump(app, 80)
            continue

        # Pump WITHOUT scrolling, sampling the anchor offset CONTINUOUSLY (catch
        # transient jumps). The focused off-screen card recycles here, focus
        # reassigns, and real thumbnails arrive → relayouts — all the moments the
        # view could jump out from under us. Drift = the view moved on its own.
        max_drift, worst = 0, None
        end = time.perf_counter() + rng.uniform(0.30, 0.65)
        while time.perf_counter() < end:
            app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 15)
            if a_idx < len(grid._layout_cache):
                rect = grid._layout_cache[a_idx]
                if rect != tgw._FILTERED_SENTINEL:
                    d = abs((bar.value() - rect[1]) - off0)
                    if d > max_drift:
                        max_drift, worst = d, (bar.value(), rect[1])
        checks += 1
        if max_drift > args.jump_threshold:
            sv1, ay1 = worst if worst else (bar.value(), 0)
            jumps.append((max_drift, sv0, sv1, a_idx, off0, sv1 - ay1,
                          max0, bar.maximum(), grid._container.height()))

        now = time.perf_counter()
        if now - last_report > 5.0:
            worst_d = max((j[0] for j in jumps), default=0)
            log(f"  t+{args.duration - (t_end - now):.0f}s checks={checks} "
                f"jumps={len(jumps)} worst_drift={worst_d}px scrollmax={bar.maximum()}")
            last_report = now

    log("=== SCROLL SOAK ENDED ===")
    log(f"total checks={checks}, jumps(>{args.jump_threshold}px)={len(jumps)}")
    jumps.sort(reverse=True)
    for (drift, sv0, sv1, a_idx, off0, off1, m0, m1, ch) in jumps[:15]:
        log(f"  JUMP drift={drift}px  sv {sv0}->{sv1}  off {off0}->{off1}  "
            f"anchor_item={a_idx}  scrollmax {m0}->{m1}  canvas_h={ch}", indent=1)

    try:
        grid._stop_all_previews(); pump(app, 200)
        grid.shutdown_all_widgets(); gen.cancel_all(); cache.close()
    except Exception:
        import traceback; traceback.print_exc(file=sys.stderr)

    if jumps:
        log(f"FAIL: reproduced {len(jumps)} scroll jump(s); worst={jumps[0][0]}px")
        sys.exit(3)
    log("SUCCESS: no scroll jump detected")


if __name__ == '__main__':
    main()
