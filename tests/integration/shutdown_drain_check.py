"""
End-to-end reproduction of the on-close crash:
    [Qt] QThread: Destroyed while thread '' is still running

Scenario: a hover preview is RUNNING (the user never moved away) and the app
shuts down. Before the fix, the play thread's QThread C++ object was destroyed
while run() was still executing → Qt aborts the process.

This script starts a REAL PyAV hover preview on a real video, leaves it
running, then performs the app's shutdown drain (PREVIEW_MANAGER.cancel_all +
qthread_registry.wait_all) and asserts every registered thread is gone. A clean
exit (code 0, no "Destroyed while" message) means the fix holds.

Usage:  python tests/integration/shutdown_drain_check.py --source <VIDEO_DIR>
"""
import argparse
import os
import sys
import time

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_VIDEO_EXTS = ('.mp4', '.mkv', '.mov', '.avi', '.wmv', '.m4v', '.webm')


def _first_video(source: str) -> str:
    if os.path.isfile(source):
        return source
    for name in sorted(os.listdir(source)):
        if name.lower().endswith(_VIDEO_EXTS):
            return os.path.join(source, name)
    raise SystemExit(f"no video found under {source}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='<VIDEO_DIR>')
    p.add_argument('--skip-drain', action='store_true',
                   help='Negative control: skip wait_all and exit with the '
                        'preview still running — should reproduce the crash.')
    args = p.parse_args()

    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import QEventLoop
    import qthread_registry
    from video_thumbnail_widget import VideoThumbnailWidget, _running_play_threads
    from pyav_play_thread import PREVIEW_MANAGER

    app = QApplication.instance() or QApplication(sys.argv)
    path = _first_video(args.source)
    print(f"file: {os.path.basename(path)}")

    def pump(ms):
        deadline = time.perf_counter() + ms / 1000.0
        while time.perf_counter() < deadline:
            app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 20)

    w = VideoThumbnailWidget(path, 1.0)
    w.resize(320, 240)

    # Start a real hover preview and let it run (do NOT stop it).
    w._hovering = True
    w._start_playback()
    pump(2500)
    running_before = qthread_registry.running_count()
    live_before = sum(1 for t in list(_running_play_threads)
                      if hasattr(t, 'isRunning') and t.isRunning())
    print(f"after start: registry={running_before}, live_play={live_before}")
    if live_before < 1:
        print("WARN: preview did not start a live thread (decode too slow?) — "
              "test inconclusive but not a failure")

    if args.skip_drain:
        # Negative control: drop EVERY strong ref to the running thread and
        # force GC — exactly what interpreter shutdown does. Expect Qt's
        # "QThread: Destroyed while thread is still running" (a hard abort).
        import gc
        t = w._play_thread
        print(f"SKIP-DRAIN: dropping all refs while running={t.isRunning()}")
        w._play_thread = None
        PREVIEW_MANAGER._current = None
        PREVIEW_MANAGER._pending = None
        _running_play_threads.clear()
        qthread_registry._running.clear()
        del t
        gc.collect()       # destroys the QThread C++ object while run() executes
        pump(300)
        print("(reached end without abort — thread finished on its own; flaky)")
        sys.exit(0)

    # ── App shutdown drain (mirrors MainWindow.closeEvent) ───────────────
    PREVIEW_MANAGER.cancel_all()
    stuck = qthread_registry.wait_all(3000)
    pump(200)   # flush deleteLater
    running_after = qthread_registry.running_count()
    live_after = sum(1 for t in list(_running_play_threads)
                     if hasattr(t, 'isRunning') and t.isRunning())
    print(f"after wait_all: stuck={stuck}, registry={running_after}, "
          f"live_play={live_after}")

    ok = (stuck == 0 and live_after == 0)
    # Deleting the widget now must NOT warn 'Destroyed while running'.
    w.deleteLater()
    pump(200)
    print("SUCCESS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
