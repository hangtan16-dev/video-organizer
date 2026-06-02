"""
Scroll-responsiveness stress test.

Reproduces the user-reported hang: "the app hangs while the user starts
scrolling while thumbnail generation is happening and/or while preview is
running."

The metric that matters: GUI-thread responsiveness. We measure the wall-clock
gap between consecutive GUI-thread ticks while we drive scrolling. If disk
work (thumbnail generation + preview) starves the GUI thread, those gaps
spike into the seconds and a real user sees a frozen window. With the disk
coordinator's note_ui_activity() suspending disk work on scroll, the gaps
should stay small (tens of ms).

Sequence:
  1. Load <VIDEO_DIR>  → kicks off thumbnail generation for ~86 items.
  2. Immediately (thumbnails still generating) start a hover preview on one
     widget — now we have background + foreground disk load.
  3. Drive a scroll gesture: many small scrollbar steps, pumping events
     between each and timing how long each pump takes.
  4. Assert the worst GUI tick gap stays under a responsiveness threshold.

Run:
    python tests/integration/scroll_stress_test.py --source <VIDEO_DIR>
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
os.environ.setdefault('VIDEO_ORGANIZER_QSETTINGS_ORG', 'VOrgScroll')
os.environ.setdefault('VIDEO_ORGANIZER_QSETTINGS_APP', 'ScrollStress')

try:
    from hires_timer import request_high_resolution_timer
    request_high_resolution_timer()
except Exception:
    pass

faulthandler.enable(file=sys.stderr)
faulthandler.dump_traceback_later(8, repeat=True, file=sys.stderr)


def log(msg, indent=0):
    ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    sys.stderr.write(f"  {ts} {' ' * indent}{msg}\n")
    sys.stderr.flush()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='<VIDEO_DIR>')
    p.add_argument('--scroll_steps', type=int, default=120)
    p.add_argument('--max_gui_gap_ms', type=float, default=400.0,
                   help='Fail if any single GUI tick takes longer than this')
    args = p.parse_args()

    log(f"=== SCROLL STRESS: {args.scroll_steps} scroll steps on {args.source} ===")

    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import QEventLoop
    from cache_manager import CacheManager
    from app_settings import AppSettings
    from thumbnail_generator import ThumbnailGeneratorService
    from thumbnail_grid_widget import ThumbnailGridWidget

    app = QApplication.instance() or QApplication(sys.argv)

    import tempfile
    cache_dir = Path(tempfile.mkdtemp(prefix='vorg_scroll_'))
    cache    = CacheManager(str(cache_dir / 'cache.db'), str(cache_dir / 'thumbs'))
    settings = AppSettings()
    settings.hover_preview_max_gb     = 100.0
    settings.use_hw_accel             = True

    gen  = ThumbnailGeneratorService(cache)
    grid = ThumbnailGridWidget(gen, settings, cache)
    grid.resize(1600, 1000)
    grid.show()

    def pump(ms):
        deadline = time.perf_counter() + ms / 1000.0
        while time.perf_counter() < deadline:
            app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 10)

    log(f"loading {args.source} (kicks off thumbnail generation)")
    grid.load_folder(args.source)
    # Let the scan complete + first batch start generating thumbnails.
    deadline = time.perf_counter() + 30
    while time.perf_counter() < deadline:
        app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 50)
        if grid._total_count > 0 and grid._batch_idx >= grid._total_count:
            break
    n_items = grid._total_count
    log(f"loaded {n_items} items; thumbnails now generating in background")
    pump(300)
    grid._full_relayout()
    pump(300)

    # Start a hover preview to add foreground disk load on top of background.
    actives = [(i, w) for i, w in sorted(grid._active.items())
               if hasattr(w, 'video_path') and hasattr(w, '_start_playback')]
    if actives:
        _, w = actives[0]
        log(f"starting hover preview on {os.path.basename(w.video_path)[:40]}")
        w._hovering = True
        w._start_playback()
        pump(500)   # let it begin

    # ── Drive the scroll gesture, timing each GUI tick ───────────────────
    log(f"driving {args.scroll_steps} scroll steps while disk is busy")
    sb = grid.verticalScrollBar()
    vmax = sb.maximum()
    if vmax <= 0:
        # Not enough content to scroll — still measure ticks
        log(f"  WARNING: scrollbar max={vmax}, limited scroll range", indent=2)

    gaps_ms = []
    worst_gap = 0.0
    worst_at = -1
    last = time.perf_counter()
    for step in range(args.scroll_steps):
        # Oscillate scroll position across the full range
        frac = (step % 40) / 40.0
        target = int(frac * vmax) if vmax > 0 else 0
        sb.setValue(target)
        # One processEvents "tick" — this is where a frozen GUI would stall.
        app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 50)
        now = time.perf_counter()
        gap = (now - last) * 1000
        gaps_ms.append(gap)
        if gap > worst_gap:
            worst_gap = gap
            worst_at = step
        last = now
        # Brief inter-step delay simulating a human scroll cadence.
        time.sleep(0.02)
        # Heartbeat every 30 steps
        if (step + 1) % 30 == 0:
            recent_worst = max(gaps_ms[-30:])
            log(f"  step {step+1}/{args.scroll_steps}: recent worst tick "
                f"{recent_worst:.0f} ms", indent=2)

    avg = sum(gaps_ms) / len(gaps_ms)
    # p95
    s = sorted(gaps_ms)
    p95 = s[int(len(s) * 0.95)]
    log("")
    log(f"GUI tick gaps: avg={avg:.0f} ms  p95={p95:.0f} ms  "
        f"worst={worst_gap:.0f} ms (at step {worst_at})")

    # Cleanup
    log("cleanup")
    try:
        grid._stop_all_previews()
        grid.shutdown_all_widgets()
    except Exception:
        import traceback
        traceback.print_exc(file=sys.stderr)
    gen.cancel_all()

    if worst_gap > args.max_gui_gap_ms:
        log(f"FAIL: worst GUI tick {worst_gap:.0f} ms exceeds "
            f"{args.max_gui_gap_ms:.0f} ms — scroll would feel frozen")
        sys.exit(1)
    log(f"PASS: GUI stayed responsive (worst tick {worst_gap:.0f} ms "
        f"< {args.max_gui_gap_ms:.0f} ms) throughout scroll under disk load")


if __name__ == '__main__':
    main()
