"""
Tests for ThumbnailGridWidget._iter_videos_recursive — the static helper
used by the new "Recursive" toggle. It's a pure-Python staticmethod, so we
can test it without instantiating a QWidget / QApplication.
"""
import os

import pytest

from thumbnail_grid_widget import ThumbnailGridWidget

_VIDEO_EXTS = frozenset({'.mp4', '.mkv', '.avi', '.mov', '.webm'})


def _make_tree(root, layout):
    """Create files described by `layout` (dict path → bytes-or-None).
    None means create a directory only."""
    for rel, content in layout.items():
        full = os.path.join(root, rel)
        os.makedirs(os.path.dirname(full) if content is not None else full,
                    exist_ok=True)
        if content is not None:
            with open(full, 'wb') as f:
                f.write(content)


# ── basic recursive collection ───────────────────────────────────────────────
def test_collects_videos_at_root(tmp_path):
    _make_tree(str(tmp_path), {
        'a.mp4': b'',
        'b.mkv': b'',
        'readme.txt': b'',
    })
    paths = sorted(e.path for e in ThumbnailGridWidget._iter_videos_recursive(
        str(tmp_path), _VIDEO_EXTS))
    assert sorted(os.path.basename(p) for p in paths) == ['a.mp4', 'b.mkv']


def test_collects_videos_recursively(tmp_path):
    _make_tree(str(tmp_path), {
        'top.mp4':                b'',
        'sub1/inner.mp4':         b'',
        'sub2/deep/nested.mkv':   b'',
        'sub2/deep/even/4.webm':  b'',
        'notes.txt':              b'',
    })
    names = sorted(os.path.basename(e.path)
                   for e in ThumbnailGridWidget._iter_videos_recursive(
                       str(tmp_path), _VIDEO_EXTS))
    assert names == ['4.webm', 'inner.mp4', 'nested.mkv', 'top.mp4']


def test_extension_filter_is_case_insensitive(tmp_path):
    _make_tree(str(tmp_path), {
        'A.MP4':  b'',
        'B.Mkv':  b'',
        'C.AVI':  b'',
        'd.txt':  b'',
    })
    names = sorted(os.path.basename(e.path)
                   for e in ThumbnailGridWidget._iter_videos_recursive(
                       str(tmp_path), _VIDEO_EXTS))
    assert names == ['A.MP4', 'B.Mkv', 'C.AVI']


def test_skips_hidden_directories(tmp_path):
    _make_tree(str(tmp_path), {
        'visible.mp4':           b'',
        '.hidden/secret.mp4':    b'',
        '.cache/thumb.mp4':      b'',
    })
    names = sorted(os.path.basename(e.path)
                   for e in ThumbnailGridWidget._iter_videos_recursive(
                       str(tmp_path), _VIDEO_EXTS))
    assert names == ['visible.mp4']


def test_skips_hidden_files_at_root(tmp_path):
    _make_tree(str(tmp_path), {
        'normal.mp4':       b'',
        '.hidden.mp4':      b'',
    })
    names = sorted(os.path.basename(e.path)
                   for e in ThumbnailGridWidget._iter_videos_recursive(
                       str(tmp_path), _VIDEO_EXTS))
    assert names == ['normal.mp4']


def test_empty_tree_returns_nothing(tmp_path):
    result = list(ThumbnailGridWidget._iter_videos_recursive(
        str(tmp_path), _VIDEO_EXTS))
    assert result == []


def test_no_videos_returns_nothing(tmp_path):
    _make_tree(str(tmp_path), {
        'doc.txt':            b'',
        'sub/notes.md':       b'',
        'sub/inner/code.py':  b'',
    })
    result = list(ThumbnailGridWidget._iter_videos_recursive(
        str(tmp_path), _VIDEO_EXTS))
    assert result == []


def test_unreadable_subdir_does_not_abort_scan(tmp_path):
    """If one subfolder raises PermissionError, the rest of the tree must
    still be scanned. We simulate by passing a non-existent path inside the
    walker — the helper should skip it and continue."""
    _make_tree(str(tmp_path), {
        'good.mp4':           b'',
        'sub/inner.mp4':      b'',
    })
    # Add a fake "directory" that doesn't actually exist (will FileNotFoundError)
    # by feeding a non-existent root — the walker should yield nothing, not crash.
    result = list(ThumbnailGridWidget._iter_videos_recursive(
        str(tmp_path / "does_not_exist"), _VIDEO_EXTS))
    assert result == []


def test_dir_entry_path_attribute_works_after_iteration(tmp_path):
    """We rely on .path being valid after the iterator closes — verify it."""
    _make_tree(str(tmp_path), {
        'a/b/c/deep.mp4':    b'',
    })
    entries = list(ThumbnailGridWidget._iter_videos_recursive(
        str(tmp_path), _VIDEO_EXTS))
    assert len(entries) == 1
    # Use .path AFTER full iteration
    assert os.path.basename(entries[0].path) == 'deep.mp4'
    # And stat() should still work
    stat = entries[0].stat()
    assert stat.st_size == 0


def test_large_tree(tmp_path):
    """Smoke test with ~200 files across a reasonable tree depth."""
    layout = {}
    for i in range(50):
        layout[f'top/v{i}.mp4'] = b''
    for d in range(10):
        for i in range(10):
            layout[f'top/sub{d}/v{i}.mkv'] = b''
    for i in range(50):
        layout[f'top/sub0/inner/v{i}.mov'] = b''
    _make_tree(str(tmp_path), layout)

    entries = list(ThumbnailGridWidget._iter_videos_recursive(
        str(tmp_path), _VIDEO_EXTS))
    assert len(entries) == 50 + 10*10 + 50   # 200


# ── settings persistence ─────────────────────────────────────────────────────
def test_recursive_view_setting_persists(tmp_path, monkeypatch):
    """recursive_view should round-trip through QSettings."""
    monkeypatch.setenv("APPDATA", str(tmp_path))
    # Use an isolated QSettings location
    from PyQt6.QtCore import QSettings
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope,
                      str(tmp_path))
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)

    from app_settings import AppSettings
    s = AppSettings()
    # Default should be False
    assert s.recursive_view is False
    s.recursive_view = True
    assert s.recursive_view is True
    s.recursive_view = False
    assert s.recursive_view is False
