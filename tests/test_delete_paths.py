r"""
Regression tests for delete PATH handling (separator normalization, UNC, and
the relative-path safety net).

Delete is now a PERMANENT delete everywhere — a real delete, NOT a move to the
Recycle Bin (the user's explicit requirement). send2trash is no longer used for
ANY path. These tests pin the path-handling that must survive that change:

  • A Qt-supplied path uses forward slashes ("D:/Videos/x" or "//server/share/…")
    which, joined with os.path.join, become MIXED ("//server/share\\sub") that
    the Windows file APIs can't resolve → the delete silently did nothing.
    MainWindow._try_delete_with_retry normalizes (os.path.normpath) first.
  • A relative path must be REFUSED (never resolved against the CWD).
  • send2trash (Recycle Bin) must NOT be called for local OR network paths.

Behavioral correctness of permanent delete (file gone, folder gone, busy-retry,
read-only) lives in tests/test_move_delete.py.
"""
import os
import sys

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('VIDEO_ORGANIZER_QSETTINGS_ORG', 'VOrgTest')
os.environ.setdefault('VIDEO_ORGANIZER_QSETTINGS_APP', 'DeletePathTest')

import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


@pytest.fixture(scope='module')
def qapp():
    # main_window creates module-level QObjects on import (reapers) — ensure a
    # QApplication exists first.
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication(sys.argv)


class _FakeSend2Trash:
    """Records any (incorrect) attempt to route a delete through the Recycle
    Bin. After the permanent-delete change it must NEVER be called."""
    def __init__(self):
        self.calls = []
    def send2trash(self, p):
        self.calls.append(p)


def _delete(path):
    """Call the (self-less) MainWindow._try_delete_with_retry unbound."""
    import main_window as mw
    return mw.MainWindow._try_delete_with_retry(object(), path)


def test_forward_slash_local_path_is_normalized_and_permanently_deleted(qapp, monkeypatch, tmp_path):
    import main_window as mw
    fake = _FakeSend2Trash()
    monkeypatch.setattr(mw, '_HAS_SEND2TRASH', True, raising=False)
    monkeypatch.setattr(mw, 'send2trash', fake, raising=False)

    f = tmp_path / "v.mp4"; f.write_bytes(b"x")
    forward = str(f).replace("\\", "/")          # forward-slash form
    assert _delete(forward) is None              # success
    assert not f.exists(), "file must be permanently deleted"
    assert fake.calls == [], "send2trash (Recycle Bin) must NOT be used"


@pytest.mark.skipif(os.name != 'nt', reason="UNC backslash semantics are Windows-only")
def test_unc_network_path_deletes_permanently_not_via_send2trash(qapp, monkeypatch):
    import main_window as mw
    fake = _FakeSend2Trash()
    monkeypatch.setattr(mw, '_HAS_SEND2TRASH', True, raising=False)
    monkeypatch.setattr(mw, 'send2trash', fake, raising=False)
    removed = []
    monkeypatch.setattr(mw.shutil, 'rmtree', lambda p, **kw: removed.append(p))
    monkeypatch.setattr(mw.os, 'remove', lambda p: removed.append(p))
    # A bare UNC root has no real dir on disk; make isdir report False so the
    # code takes the os.remove branch deterministically.
    monkeypatch.setattr(mw.os.path, 'isdir', lambda p: False)

    assert _delete("//nas/media/oldfile.mp4") is None    # success
    # send2trash is NOT used (no Recycle Bin on a network share)…
    assert fake.calls == []
    # …it's permanently deleted, with normalized UNC backslashes.
    assert removed == [os.path.normpath("//nas/media/oldfile.mp4")]
    assert removed[0].startswith("\\\\")


def test_local_delete_does_not_use_recycle_bin(qapp, monkeypatch, tmp_path):
    """The headline behavior change: a LOCAL delete is permanent — it must use
    os.remove, never send2trash."""
    import main_window as mw
    fake = _FakeSend2Trash()
    monkeypatch.setattr(mw, '_HAS_SEND2TRASH', True, raising=False)
    monkeypatch.setattr(mw, 'send2trash', fake, raising=False)

    f = tmp_path / "v.mp4"; f.write_bytes(b"x")
    assert _delete(str(f)) is None
    assert not f.exists()
    assert fake.calls == []


def test_relative_path_is_refused(qapp, monkeypatch):
    import main_window as mw
    fake = _FakeSend2Trash()
    monkeypatch.setattr(mw, '_HAS_SEND2TRASH', True, raising=False)
    monkeypatch.setattr(mw, 'send2trash', fake, raising=False)
    res = _delete("some/relative/folder")
    assert res is not None                      # refused, not deleted
    assert fake.calls == []
