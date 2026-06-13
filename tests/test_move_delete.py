r"""
Move + Delete correctness tests.

User reports:
  • DELETE is "broken" — it sends files to the Recycle Bin instead of actually
    deleting them. Desired: a permanent delete (with the existing confirm
    dialog as the safety gate).
  • MOVE is "unreliable and fails often". Analysis of _MoveWorker._move_one
    found the deterministic culprits:
      1. Destination name COLLISION. os.rename on Windows raises
         FileExistsError (WinError 183) when the destination already holds a
         file of that name. That code is classified as neither "busy" nor
         "cross-device", so the move just returns "rename failed" — i.e. it
         fails whenever you move a file whose name already exists at the
         target (very common when consolidating, or in recursive view where
         two subfolders hold same-named clips). Same hazard on the
         cross-device copytree path.
      2. No path normalization — a Qt forward-slash path ("D:/Videos/x.mp4").
      3. Moving an item into the folder it already lives in should be a
         no-op, not an attempted self-move.

These tests assert the DESIRED behavior and run entirely on temp files (the
real library is never touched — per the standing data-safety rule).
"""
import os
import sys
import time

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('VIDEO_ORGANIZER_QSETTINGS_ORG', 'VOrgTest')
os.environ.setdefault('VIDEO_ORGANIZER_QSETTINGS_APP', 'MoveDeleteTest')

import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


@pytest.fixture(scope='module')
def qapp():
    # main_window builds module-level QObjects on import (reapers); a
    # QApplication must exist first. _MoveWorker is a QThread, so it also needs
    # the app — we construct workers but never start() them (we call _move_one
    # synchronously).
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication(sys.argv)


# ── helpers ──────────────────────────────────────────────────────────────────
def _worker(dest, copy_only=False):
    import main_window as mw
    w = mw._MoveWorker([], str(dest), copy_only=copy_only)
    # shrink retry backoff so the busy-retry tests run fast
    w._RENAME_RETRY_DELAY = 0.01
    w._DELETE_RETRY_DELAY = 0.01
    return w


def _delete(path):
    """Call MainWindow._try_delete_with_retry without building a window."""
    import main_window as mw
    return mw.MainWindow._try_delete_with_retry(object(), path)


def _mk(p, data=b"data"):
    p = str(p)
    with open(p, 'wb') as fh:
        fh.write(data)
    return p


# ── MOVE ───────────────────────────────────────────────────────────────────
def test_move_simple_same_drive(qapp, tmp_path):
    src_dir = tmp_path / "a"; src_dir.mkdir()
    dst_dir = tmp_path / "b"; dst_dir.mkdir()
    src = _mk(src_dir / "clip.mp4", b"hello")

    err = _worker(dst_dir)._move_one(src)
    assert err == '', f"unexpected error: {err}"
    assert not os.path.exists(src), "source should be gone after a move"
    moved = dst_dir / "clip.mp4"
    assert moved.exists() and moved.read_bytes() == b"hello"


def test_move_collision_keeps_both_with_unique_name(qapp, tmp_path):
    """The bug: dst already has a same-named (DIFFERENT) file. The move must
    SUCCEED without clobbering — the moved file gets a unique name."""
    src_dir = tmp_path / "a"; src_dir.mkdir()
    dst_dir = tmp_path / "b"; dst_dir.mkdir()
    src = _mk(src_dir / "clip.mp4", b"NEW")
    _mk(dst_dir / "clip.mp4", b"EXISTING")          # collision

    err = _worker(dst_dir)._move_one(src)
    assert err == '', f"collision should not fail the move: {err}"
    assert not os.path.exists(src), "source should be gone"
    # The pre-existing file is untouched…
    assert (dst_dir / "clip.mp4").read_bytes() == b"EXISTING"
    # …and the moved file is present under a unique name with our bytes.
    others = [p for p in dst_dir.iterdir() if p.name != "clip.mp4"]
    assert len(others) == 1, f"expected one renamed file, got {[p.name for p in dst_dir.iterdir()]}"
    assert others[0].read_bytes() == b"NEW"
    assert others[0].suffix == ".mp4"


def test_move_into_same_folder_is_noop(qapp, tmp_path):
    """Moving an item into the folder it already lives in must not fail and
    must not duplicate it."""
    d = tmp_path / "a"; d.mkdir()
    src = _mk(d / "clip.mp4", b"x")
    err = _worker(d)._move_one(src)
    assert err == '', f"self-folder move should be a no-op success: {err}"
    assert os.path.exists(src)
    assert [p.name for p in d.iterdir()] == ["clip.mp4"], "must not duplicate"


def test_move_directory(qapp, tmp_path):
    src_parent = tmp_path / "a"; src_parent.mkdir()
    dst_dir    = tmp_path / "b"; dst_dir.mkdir()
    folder = src_parent / "season1"; folder.mkdir()
    _mk(folder / "e1.mp4", b"one")

    err = _worker(dst_dir)._move_one(str(folder))
    assert err == '', f"folder move failed: {err}"
    assert not folder.exists()
    assert (dst_dir / "season1" / "e1.mp4").read_bytes() == b"one"


def test_move_forward_slash_path(qapp, tmp_path):
    src_dir = tmp_path / "a"; src_dir.mkdir()
    dst_dir = tmp_path / "b"; dst_dir.mkdir()
    src = _mk(src_dir / "clip.mp4", b"y")
    forward = src.replace("\\", "/")

    err = _worker(dst_dir)._move_one(forward)
    assert err == '', f"forward-slash path failed: {err}"
    assert (dst_dir / "clip.mp4").exists()


def test_move_busy_source_retries_then_succeeds(qapp, tmp_path, monkeypatch):
    """A sharing violation (file just released by a preview) is transient —
    the move must retry and succeed, not fail on the first attempt."""
    import main_window as mw
    src_dir = tmp_path / "a"; src_dir.mkdir()
    dst_dir = tmp_path / "b"; dst_dir.mkdir()
    src = _mk(src_dir / "clip.mp4", b"z")

    real_rename = os.rename
    calls = {"n": 0}

    def flaky_rename(a, b):
        calls["n"] += 1
        if calls["n"] <= 2:               # first two attempts: "in use"
            e = OSError("sharing violation")
            e.winerror = 32
            raise e
        return real_rename(a, b)

    monkeypatch.setattr(mw.os, 'rename', flaky_rename)
    err = _worker(dst_dir)._move_one(src)
    assert err == '', f"busy retry should have succeeded: {err}"
    assert calls["n"] == 3
    assert (dst_dir / "clip.mp4").exists()


def test_move_cross_device_falls_back_to_copy_delete(qapp, tmp_path, monkeypatch):
    """A cross-drive move (WinError 17 / EXDEV) must copy then delete the
    source — verified by content + source removal."""
    import main_window as mw
    src_dir = tmp_path / "a"; src_dir.mkdir()
    dst_dir = tmp_path / "b"; dst_dir.mkdir()
    src = _mk(src_dir / "clip.mp4", b"crossdev")

    def xdev_rename(a, b):
        e = OSError("cross device")
        e.winerror = 17
        import errno
        e.errno = errno.EXDEV
        raise e

    monkeypatch.setattr(mw.os, 'rename', xdev_rename)
    err = _worker(dst_dir)._move_one(src)
    assert err == '', f"cross-device move failed: {err}"
    assert not os.path.exists(src), "source must be deleted after copy"
    assert (dst_dir / "clip.mp4").read_bytes() == b"crossdev"


def test_copy_only_collision_unique_and_keeps_source(qapp, tmp_path):
    src_dir = tmp_path / "a"; src_dir.mkdir()
    dst_dir = tmp_path / "b"; dst_dir.mkdir()
    src = _mk(src_dir / "clip.mp4", b"COPYME")
    _mk(dst_dir / "clip.mp4", b"EXISTING")

    err = _worker(dst_dir, copy_only=True)._move_one(src)
    assert err == '', f"copy collision should not fail: {err}"
    assert os.path.exists(src), "copy must preserve the source"
    assert (dst_dir / "clip.mp4").read_bytes() == b"EXISTING"
    others = [p for p in dst_dir.iterdir() if p.name != "clip.mp4"]
    assert len(others) == 1 and others[0].read_bytes() == b"COPYME"


# ── DELETE ───────────────────────────────────────────────────────────────────
def test_delete_file_is_permanent_not_recycled(qapp, tmp_path, monkeypatch):
    """The headline fix: delete must PERMANENTLY remove the file, never route
    through send2trash (the Recycle Bin)."""
    import main_window as mw
    recorder = []
    monkeypatch.setattr(mw, '_HAS_SEND2TRASH', True, raising=False)
    monkeypatch.setattr(mw, 'send2trash',
                        type('S', (), {'send2trash': staticmethod(lambda p: recorder.append(p))}),
                        raising=False)

    f = _mk(tmp_path / "v.mp4", b"bye")
    assert _delete(f) is None, "delete should succeed"
    assert not os.path.exists(f), "file must be gone"
    assert recorder == [], "send2trash (Recycle Bin) must NOT be used"


def test_delete_folder_permanent(qapp, tmp_path):
    d = tmp_path / "season"; d.mkdir()
    _mk(d / "e1.mp4", b"a")
    _mk(d / "e2.mp4", b"b")
    assert _delete(str(d)) is None
    assert not d.exists()


def test_delete_busy_then_succeeds(qapp, tmp_path, monkeypatch):
    import main_window as mw
    monkeypatch.setattr(mw, '_HAS_SEND2TRASH', False, raising=False)
    monkeypatch.setattr(time, 'sleep', lambda *_a, **_k: None)

    f = _mk(tmp_path / "v.mp4", b"x")
    real_remove = os.remove
    calls = {"n": 0}

    def flaky_remove(p):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise PermissionError("locked")
        return real_remove(p)

    monkeypatch.setattr(mw.os, 'remove', flaky_remove)
    assert _delete(f) is None
    assert calls["n"] == 3
    assert not os.path.exists(f)


def test_delete_relative_path_refused(qapp):
    res = _delete("some/relative/clip.mp4")
    assert res is not None, "a relative path must be refused, never deleted"


def test_delete_forward_slash_normalized(qapp, tmp_path, monkeypatch):
    import main_window as mw
    monkeypatch.setattr(mw, '_HAS_SEND2TRASH', False, raising=False)
    f = _mk(tmp_path / "v.mp4", b"x")
    assert _delete(f.replace("\\", "/")) is None
    assert not os.path.exists(f)


def test_delete_missing_file_reports_error(qapp, tmp_path, monkeypatch):
    import main_window as mw
    monkeypatch.setattr(mw, '_HAS_SEND2TRASH', False, raising=False)
    res = _delete(str(tmp_path / "nope.mp4"))
    assert res is not None, "deleting a nonexistent file should report, not crash"
