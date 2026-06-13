r"""
End-to-end move / delete / undo through a real MainWindow.

The unit tests in test_move_delete.py prove the file-operation logic
(_MoveWorker._move_one, _try_delete_with_retry). THIS file drives the real
user-facing handlers — _on_move (async QThread worker), _on_delete (synchronous),
_on_undo — so the whole chain is exercised: confirm dialog → handle release →
worker thread → signal back on the GUI thread → undo stack → cache invalidate.

Everything runs on temp files; the real library is never touched. The grid's
get_checked_paths / remove_paths are stubbed so we test MainWindow orchestration
without depending on grid rendering (which has its own tests).
"""
import os
import sys
import time

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('VIDEO_ORGANIZER_QSETTINGS_ORG', 'VOrgTest')
os.environ.setdefault('VIDEO_ORGANIZER_QSETTINGS_APP', 'MoveDeleteE2E')

import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


@pytest.fixture(scope='module')
def qapp():
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication(sys.argv)


@pytest.fixture
def window(qapp):
    from main_window import MainWindow
    w = MainWindow()
    yield w
    try:
        w.close()
    except Exception:
        pass
    qapp.processEvents()


def _pump_until(qapp, cond, timeout=10.0):
    end = time.perf_counter() + timeout
    while time.perf_counter() < end:
        qapp.processEvents()
        if cond():
            return True
        time.sleep(0.005)
    qapp.processEvents()
    return cond()


def _eq_paths(a, b):
    return {os.path.normcase(os.path.normpath(p)) for p in a} == \
           {os.path.normcase(os.path.normpath(p)) for p in b}


def _stub_grid(monkeypatch, window, checked):
    removed = []
    monkeypatch.setattr(window._grid, 'get_checked_paths', lambda: list(checked))
    monkeypatch.setattr(window._grid, 'remove_paths', lambda paths: removed.extend(paths))
    return removed


def _silence_dialogs(monkeypatch):
    import main_window as mw
    monkeypatch.setattr(mw.QMessageBox, 'warning',
                        staticmethod(lambda *a, **k: None), raising=False)
    monkeypatch.setattr(mw.QMessageBox, 'information',
                        staticmethod(lambda *a, **k: None), raising=False)


# ── MOVE + UNDO ───────────────────────────────────────────────────────────────
def test_move_then_undo_through_mainwindow(window, qapp, tmp_path, monkeypatch):
    import main_window as mw
    src = tmp_path / "src"; src.mkdir()
    dst = tmp_path / "dst"; dst.mkdir()
    f1 = str(src / "a.mp4"); open(f1, 'wb').write(b"one")
    f2 = str(src / "b.mp4"); open(f2, 'wb').write(b"two")

    _silence_dialogs(monkeypatch)
    monkeypatch.setattr(mw.QFileDialog, 'getExistingDirectory',
                        staticmethod(lambda *a, **k: str(dst)))
    removed = _stub_grid(monkeypatch, window, [f1, f2])

    window._on_move()
    assert _pump_until(qapp, lambda: window._move_worker is None), "move worker never finished"

    assert not os.path.exists(f1) and not os.path.exists(f2), "sources should be gone"
    assert (dst / "a.mp4").read_bytes() == b"one"
    assert (dst / "b.mp4").read_bytes() == b"two"
    assert _eq_paths(removed, [f1, f2]), "moved items must be removed from the grid"

    op = window._undo_stack[-1]
    assert op.kind == 'move'
    assert _eq_paths(op.dest_paths, [str(dst / "a.mp4"), str(dst / "b.mp4")])

    # Undo → files come back to the source folder.
    window._on_undo()
    assert _pump_until(qapp, lambda: window._move_worker is None), "undo worker never finished"
    assert os.path.exists(f1) and os.path.exists(f2), "undo should restore the files"
    assert not (dst / "a.mp4").exists() and not (dst / "b.mp4").exists()


def test_move_collision_e2e_undo_restores_our_file_not_the_collider(window, qapp, tmp_path, monkeypatch):
    """The subtle correctness win: when a collision forces a unique name, Undo
    must bring OUR file back, leaving the pre-existing same-named file alone."""
    import main_window as mw
    src = tmp_path / "src"; src.mkdir()
    dst = tmp_path / "dst"; dst.mkdir()
    f = str(src / "clip.mp4"); open(f, 'wb').write(b"MINE")
    open(dst / "clip.mp4", 'wb').write(b"PREEXISTING")        # collision

    _silence_dialogs(monkeypatch)
    monkeypatch.setattr(mw.QFileDialog, 'getExistingDirectory',
                        staticmethod(lambda *a, **k: str(dst)))
    _stub_grid(monkeypatch, window, [f])

    window._on_move()
    assert _pump_until(qapp, lambda: window._move_worker is None)
    assert not os.path.exists(f)
    assert (dst / "clip.mp4").read_bytes() == b"PREEXISTING", "must NOT clobber existing file"
    renamed = [p for p in dst.iterdir() if p.name != "clip.mp4"]
    assert len(renamed) == 1 and renamed[0].read_bytes() == b"MINE"

    window._on_undo()
    assert _pump_until(qapp, lambda: window._move_worker is None)
    # The pre-existing collider stays put…
    assert (dst / "clip.mp4").read_bytes() == b"PREEXISTING"
    # …and exactly our file is back in the source folder.
    back = list(src.iterdir())
    assert len(back) == 1 and back[0].read_bytes() == b"MINE"


# ── DELETE ────────────────────────────────────────────────────────────────────
def test_delete_e2e_is_permanent(window, qapp, tmp_path, monkeypatch):
    import main_window as mw
    src = tmp_path / "s"; src.mkdir()
    f1 = str(src / "a.mp4"); open(f1, 'wb').write(b"x")
    f2 = str(src / "b.mp4"); open(f2, 'wb').write(b"y")

    _silence_dialogs(monkeypatch)
    monkeypatch.setattr(mw.QMessageBox, 'question',
                        staticmethod(lambda *a, **k: mw.QMessageBox.StandardButton.Yes))
    recorder = []
    monkeypatch.setattr(mw, '_HAS_SEND2TRASH', True, raising=False)
    monkeypatch.setattr(mw, 'send2trash',
                        type('S', (), {'send2trash': staticmethod(lambda p: recorder.append(p))}),
                        raising=False)
    removed = _stub_grid(monkeypatch, window, [f1, f2])

    window._on_delete()     # synchronous

    assert not os.path.exists(f1) and not os.path.exists(f2), "files must be permanently gone"
    assert recorder == [], "delete must NOT route through the Recycle Bin"
    assert _eq_paths(removed, [f1, f2])


def test_delete_e2e_cancel_keeps_files(window, qapp, tmp_path, monkeypatch):
    import main_window as mw
    f = str(tmp_path / "a.mp4"); open(f, 'wb').write(b"x")
    _silence_dialogs(monkeypatch)
    monkeypatch.setattr(mw.QMessageBox, 'question',
                        staticmethod(lambda *a, **k: mw.QMessageBox.StandardButton.Cancel))
    _stub_grid(monkeypatch, window, [f])

    window._on_delete()
    assert os.path.exists(f), "Cancel on the confirm dialog must keep the file"
