"""Minimal repro of test_08 to find what keeps the process alive at exit."""
import os, sys, time, threading, faulthandler
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.environ['OPENCV_FFMPEG_LOGLEVEL'] = '-8'; os.environ['OPENCV_LOG_LEVEL'] = 'OFF'; os.environ['AV_LOG_LEVEL'] = 'quiet'

def log(m): sys.stderr.write(f"  {time.strftime('%H:%M:%S')} {m}\n"); sys.stderr.flush()

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QEventLoop
import pyav_play_thread as ppt

# find a medium test file
from pathlib import Path
TEST = Path('<TEST_DIR>/medium')
f = next((p for p in TEST.glob('*') if p.suffix.lower() in ('.mkv','.mp4')), None)
log(f"file: {f}")
app = QApplication.instance() or QApplication(sys.argv)

th = ppt._PyAVPlayThread(str(f), 0.0, 1280, 720, hw_accel=False, target_fps=0)
frames = []
th.frame_ready.connect(lambda q: frames.append(1))
th.start()
def pump(ms):
    dl = time.perf_counter()+ms/1000
    while time.perf_counter() < dl:
        app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 20)
pump(1000)
th.request_seek(60.0)
pump(2000)
log(f"frames: {len(frames)}")
log("stopping thread")
th.stop()
ok = th.wait(2000)
log(f"thread.wait returned {ok}, isRunning={th.isRunning()}")

pump(500)
log(f"alive python threads: {[t.name for t in threading.enumerate()]}")
log("dumping all thread stacks:")
faulthandler.dump_traceback(file=sys.stderr)
log("=== reached end of script; if process hangs now, a non-daemon/QThread is alive ===")
# Schedule a dump in 5s in case exit hangs
faulthandler.dump_traceback_later(5, file=sys.stderr)
