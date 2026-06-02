r"""
Regression tests for deleting NETWORK / UNC folders.

User report: deleting network folders silently did nothing. Cause: a network
root reached through the Qt folder tree uses forward slashes
("//server/share/…") and, joined with os.path.join, becomes a MIXED-separator
path ("//server/share\\sub"). The Windows shell API that send2trash uses
(SHCreateItemFromParsingName) can't resolve forward-slash / UNC paths, so the
delete failed. And network shares have NO Recycle Bin, so send2trash can't
recycle them at all.

Fix (MainWindow._try_delete_with_retry): normalize separators first
(os.path.normpath → backslash / proper "\\server\share" UNC), refuse relative
paths, and for UNC paths delete PERMANENTLY (the only option on a network
location — what Explorer does too) instead of calling send2trash.
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
    def __init__(self):
        self.calls = []
    def send2trash(self, p):
        self.calls.append(p)


def _delete(path):
    """Call the (self-less) MainWindow._try_delete_with_retry unbound."""
    import main_window as mw
    return mw.MainWindow._try_delete_with_retry(object(), path)


def test_forward_slash_local_path_is_normalized_for_send2trash(qapp, monkeypatch, tmp_path):
    import main_window as mw
    fake = _FakeSend2Trash()
    monkeypatch.setattr(mw, '_HAS_SEND2TRASH', True)
    monkeypatch.setattr(mw, 'send2trash', fake)

    f = tmp_path / "v.mp4"; f.write_bytes(b"x")
    forward = str(f).replace("\\", "/")          # forward-slash form
    assert _delete(forward) is None              # success
    # send2trash received the NORMALIZED (backslash on Windows) path.
    assert fake.calls == [os.path.normpath(forward)]
    if os.name == 'nt':
        assert "/" not in fake.calls[0]


@pytest.mark.skipif(os.name != 'nt', reason="UNC backslash semantics are Windows-only")
def test_unc_network_path_deletes_permanently_not_via_send2trash(qapp, monkeypatch):
    import main_window as mw
    fake = _FakeSend2Trash()
    monkeypatch.setattr(mw, '_HAS_SEND2TRASH', True)
    monkeypatch.setattr(mw, 'send2trash', fake)
    removed = []
    monkeypatch.setattr(mw.shutil, 'rmtree', lambda p: removed.append(p))
    monkeypatch.setattr(mw.os, 'remove', lambda p: removed.append(p))

    assert _delete("//nas/media/oldfolder") is None     # success
    # send2trash is NOT used (no Recycle Bin on a network share)…
    assert fake.calls == []
    # …it's permanently deleted, with normalized UNC backslashes.
    assert removed == [os.path.normpath("//nas/media/oldfolder")]
    assert removed[0].startswith("\\\\")


def test_local_send2trash_failure_is_reported_not_hard_deleted(qapp, monkeypatch, tmp_path):
    """Safety: a LOCAL send2trash failure must be surfaced, never silently
    escalated to a permanent delete (that would defeat the Recycle Bin)."""
    import main_window as mw
    class _Boom:
        def send2trash(self, p):
            raise RuntimeError("boom")
    monkeypatch.setattr(mw, '_HAS_SEND2TRASH', True)
    monkeypatch.setattr(mw, 'send2trash', _Boom())
    removed = []
    monkeypatch.setattr(mw.shutil, 'rmtree', lambda p: removed.append(p))
    monkeypatch.setattr(mw.os, 'remove', lambda p: removed.append(p))

    f = tmp_path / "v.mp4"; f.write_bytes(b"x")
    res = _delete(str(f))
    assert res is not None and res != ""        # error reported
    assert removed == []                        # NOT hard-deleted


def test_relative_path_is_refused(qapp, monkeypatch):
    import main_window as mw
    fake = _FakeSend2Trash()
    monkeypatch.setattr(mw, '_HAS_SEND2TRASH', True)
    monkeypatch.setattr(mw, 'send2trash', fake)
    res = _delete("some/relative/folder")
    assert res is not None                      # refused, not deleted
    assert fake.calls == []
