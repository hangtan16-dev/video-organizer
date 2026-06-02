"""Standalone version of test_11 — runs the user's browse-review-delete
workflow directly so we can see what happens without pytest buffering.

Output goes to stderr with explicit flush so partial progress survives
even a Windows fatal exception (access violation, etc.).
"""
import os
import sys
import time
import datetime
from pathlib import Path

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Match pytest env
os.environ['OPENCV_FFMPEG_LOGLEVEL'] = '-8'
os.environ['OPENCV_LOG_LEVEL']       = 'OFF'
os.environ['AV_LOG_LEVEL']           = 'quiet'
_DECODE_THREADS = max(1, int((os.cpu_count() or 4) * 0.75))
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = (
    f'threads;{_DECODE_THREADS}|thread_type;slice'
)
os.environ.setdefault('VIDEO_ORGANIZER_QSETTINGS_ORG', 'VideoOrganizerStandalone')
os.environ.setdefault('VIDEO_ORGANIZER_QSETTINGS_APP', 'StandaloneRun')

try:
    from hires_timer import request_high_resolution_timer
    request_high_resolution_timer()
except Exception:
    pass


def say(msg, indent=2):
    """Log to stderr with timestamp and immediate flush."""
    ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    sys.stderr.write(f"  {ts} {' ' * indent}{msg}\n")
    sys.stderr.flush()


def main():
    say("=== STANDALONE WORKFLOW ===", indent=0)
    say(f"python: {sys.version}", indent=0)
    say(f"cpu_count: {os.cpu_count()}", indent=0)

    say("importing PyQt6")
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import QEventLoop, QTimer, Qt
    say("importing project modules")
    from cache_manager import CacheManager
    from app_settings import AppSettings
    from thumbnail_generator import ThumbnailGeneratorService
    from thumbnail_grid_widget import ThumbnailGridWidget

    app = QApplication.instance() or QApplication(sys.argv)

    say("creating cache + settings")
    import tempfile
    cache_dir = Path(tempfile.mkdtemp())
    cache = CacheManager(str(cache_dir / 'cache.db'), str(cache_dir / 'thumbs'))
    settings = AppSettings()
    settings.hover_preview_max_gb     = 4.0    # default — disables hover on >4GB
    settings.large_file_threshold_mb  = 500
    settings.use_hw_accel             = True

    # IMPORTANT: turn hover preview cap up so the big stress files can be
    # hover-previewed. The user reports hangs at hover so we need to
    # exercise that code path.
    settings.hover_preview_max_gb = 100.0    # allow even the 50GB files

    say("creating grid widget")
    gen  = ThumbnailGeneratorService(cache)
    grid = ThumbnailGridWidget(gen, settings, cache)
    grid.resize(1400, 900)

    say("set_recursive(True)")
    grid.set_recursive(True)
    say("load_folder(<TEST_DIR>)")
    t0 = time.perf_counter()
    grid.load_folder('<TEST_DIR>')

    # Wait for the scan + batch loader.
    say("waiting up to 30s for items to populate")
    deadline = time.perf_counter() + 30
    while time.perf_counter() < deadline:
        app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 100)
        items = grid.get_all_items()
        videos = [i for i in items if not i.is_folder]
        if videos and grid._batch_idx >= grid._total_count:
            break
    grid._full_relayout()
    # Pump events to materialize visible widgets
    loop_end = time.perf_counter() + 2.0
    while time.perf_counter() < loop_end:
        app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 100)
    load_ms = (time.perf_counter() - t0) * 1000

    items = grid.get_all_items()
    videos = [i for i in items if not i.is_folder]
    say(f"loaded {len(videos)} videos in {load_ms:.0f} ms (batch_idx={grid._batch_idx}/{grid._total_count})")

    # Get visible widgets
    say("scanning for active video widgets")
    active = []
    for idx, w in grid._active.items():
        if hasattr(w, 'video_path') and hasattr(w, '_start_playback'):
            active.append(w)
    say(f"got {len(active)} active video widgets")

    if not active:
        say("NO ACTIVE WIDGETS — bailing", indent=0)
        return

    # Per-widget: thumbnail wait, hover, seek, hover, mark every 3rd for delete
    for i, w in enumerate(active):
        name = os.path.basename(w.video_path)
        size_gb = w._file_size / (1024 ** 3) if w._file_size > 0 else 0
        say(f"── [{i+1}/{len(active)}] {name[:50]} ({size_gb:.2f} GB) ──", indent=0)

        # ── Step 1: wait for thumbnail (15s budget)
        say(f"  waiting for thumbnail", indent=2)
        thumb_deadline = time.perf_counter() + 15
        got_thumb = False
        while time.perf_counter() < thumb_deadline:
            if w._full_pixmap is not None:
                got_thumb = True
                break
            if w._loading_label.text().startswith('⚠'):
                break
            app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 100)
        thumb_ms = (time.perf_counter() - (thumb_deadline - 15)) * 1000
        say(f"  thumbnail: {thumb_ms:.0f} ms, got={got_thumb}", indent=2)
        if not got_thumb:
            say(f"  ❌ NO THUMBNAIL for {name} — would hang in UI", indent=2)
            continue

        # ── Step 2: hover #1 (2 seconds at native FPS)
        say(f"  starting hover #1 (2s)", indent=2)
        frame_count = [0]
        first_frame_time = [None]
        original_on_frame = w._on_play_frame
        def tap(qimg, _w=w, _orig=original_on_frame, _fc=frame_count, _ff=first_frame_time):
            if _ff[0] is None:
                _ff[0] = time.perf_counter()
            _fc[0] += 1
            _orig(qimg)
        w._on_play_frame = tap

        hover_start = time.perf_counter()
        w._hovering = True
        w._start_playback()
        # Wait 2s while pumping events
        hover_end = hover_start + 2.0
        while time.perf_counter() < hover_end:
            app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 50)
        w._hovering = False
        say(f"  stopping play thread (was {w._play_thread is not None})", indent=2)
        # Use the widget's own _stop_playback (which clears _play_thread = None)
        # rather than directly calling .stop() on the thread.
        w._stop_playback()
        # Pump events for a moment so finished signal can fire
        for _ in range(20):
            app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 50)
        w._on_play_frame = original_on_frame
        ff_ms = ((first_frame_time[0] - hover_start) * 1000) if first_frame_time[0] else None
        say(f"  hover #1: {frame_count[0]} frames, first @ "
            f"{f'{ff_ms:.0f} ms' if ff_ms else 'NONE'}", indent=2)
        if frame_count[0] == 0:
            say(f"  ❌ NO HOVER FRAMES for {name}", indent=2)

        # ── Step 3: seek to 50%
        dur = w._duration or 60.0
        new_seek = dur * 0.5
        say(f"  seek to {new_seek:.1f}s (dur={dur:.0f}s)", indent=2)
        w._seek_time = new_seek
        t0 = time.perf_counter()
        w._apply_seek()
        # Wait for new thumbnail
        seek_deadline = t0 + 15
        got_seek_thumb = False
        while time.perf_counter() < seek_deadline:
            if w._full_pixmap is not None:
                got_seek_thumb = True
                break
            if w._loading_label.text().startswith('⚠'):
                break
            app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 100)
        seek_thumb_ms = (time.perf_counter() - t0) * 1000
        say(f"  post-seek thumbnail: {seek_thumb_ms:.0f} ms, got={got_seek_thumb}",
            indent=2)
        if not got_seek_thumb:
            say(f"  ❌ NO POST-SEEK THUMBNAIL for {name}", indent=2)

        # ── Step 4: hover #2 (2 seconds)
        say(f"  starting hover #2 (2s)", indent=2)
        frame_count2 = [0]
        original_on_frame = w._on_play_frame
        def tap2(qimg, _w=w, _orig=original_on_frame, _fc=frame_count2):
            _fc[0] += 1
            _orig(qimg)
        w._on_play_frame = tap2

        hover2_start = time.perf_counter()
        w._hovering = True
        w._start_playback()
        hover2_end = hover2_start + 2.0
        while time.perf_counter() < hover2_end:
            app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 50)
        w._hovering = False
        say(f"  stopping play thread", indent=2)
        w._stop_playback()
        for _ in range(20):
            app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 50)
        w._on_play_frame = original_on_frame
        say(f"  hover #2: {frame_count2[0]} frames", indent=2)
        if frame_count2[0] == 0:
            say(f"  ❌ NO HOVER #2 FRAMES for {name}", indent=2)

        # ── Step 5: every 3rd file gets delete check (just verify checkbox)
        if i % 3 == 0:
            w.set_checked(True)
            say(f"  [check] marked {name} for delete", indent=2)

        say(f"  ✅ widget {i+1} complete", indent=2)

    say("=== ALL DONE ===", indent=0)
    say("cleanup: shutting down widgets")
    grid.shutdown_all_widgets()
    gen.cancel_all()
    say("cleanup complete, exiting")


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        sys.exit(1)
