"""
Regression tests for transient thumbnail-failure RETRY.

User-reported bug: changing thumbnails-per-row (or any mass regen) under
scroll left some videos blank with "Cannot read frame" forever. Root cause:
a single contended/timed-out frame read was cached as a PERMANENT failure and
never retried. Fix: transient failures ("Could not read frame", "Timeout")
are auto-retried with backoff and never permanently cached until several
retries fail; hard failures ("Cannot open video") stay failed.

These exercise ThumbnailGridWidget._on_thumbnail_failed / _retry_thumbnail with
a mock generator and a captured (not real) retry timer, so they're fast and
deterministic.
"""
import os
import sys

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
# Isolate QSettings so AppSettings() doesn't touch the real user config.
os.environ.setdefault('VIDEO_ORGANIZER_QSETTINGS_ORG', 'VOrgTest')
os.environ.setdefault('VIDEO_ORGANIZER_QSETTINGS_APP', 'ThumbRetryTest')

import pytest
from unittest.mock import MagicMock

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


@pytest.fixture(scope='module')
def qapp():
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication(sys.argv)


def _make_grid(tmp_path):
    from cache_manager import CacheManager
    from app_settings import AppSettings
    import thumbnail_grid_widget as tgw
    cache = CacheManager(str(tmp_path / 'cache.db'), str(tmp_path / 'thumbs'))
    settings = AppSettings()
    gen = MagicMock()                       # signals' .connect() are no-ops
    grid = tgw.ThumbnailGridWidget(gen, settings, cache)
    return grid, gen, cache, tgw


def _add_item(grid, tgw, path):
    item = tgw._Item(path=path, is_folder=False)
    idx = len(grid._items)
    grid._items.append(item)
    grid._path_to_idx[path] = idx
    return item


def test_transient_failure_schedules_retry_not_permanent(qapp, tmp_path, monkeypatch):
    from PyQt6 import QtCore
    grid, gen, cache, tgw = _make_grid(tmp_path)
    f = tmp_path / 'v.mp4'; f.write_bytes(b'x')
    item = _add_item(grid, tgw, str(f))

    scheduled = []
    monkeypatch.setattr(QtCore.QTimer, 'singleShot',
                        lambda ms, cb: scheduled.append(cb))

    grid._on_thumbnail_failed(str(f), "Could not read frame")
    # NOT marked permanently failed; a retry is scheduled.
    assert item.thumbnail_failed is False
    assert grid._thumb_retry[str(f)] == 1
    assert len(scheduled) == 1

    # Firing the retry re-requests generation for that path.
    gen.request_thumbnail.reset_mock()
    scheduled[-1]()
    assert gen.request_thumbnail.called
    assert gen.request_thumbnail.call_args[0][0] == str(f)


def test_success_clears_failed_state_and_counter(qapp, tmp_path, monkeypatch):
    from PyQt6 import QtCore
    from PyQt6.QtGui import QImage
    grid, gen, cache, tgw = _make_grid(tmp_path)
    f = tmp_path / 'v.mp4'; f.write_bytes(b'x')
    item = _add_item(grid, tgw, str(f))
    monkeypatch.setattr(QtCore.QTimer, 'singleShot', lambda ms, cb: None)

    grid._on_thumbnail_failed(str(f), "Could not read frame")
    assert grid._thumb_retry.get(str(f)) == 1

    img = QImage(8, 8, QImage.Format.Format_RGB888); img.fill(0)
    grid._on_thumbnail_ready(str(f), item.seek_time, img, 1.0)
    assert item.thumbnail_failed is False
    assert str(f) not in grid._thumb_retry


def test_retries_exhausted_becomes_permanent(qapp, tmp_path, monkeypatch):
    from PyQt6 import QtCore
    grid, gen, cache, tgw = _make_grid(tmp_path)
    f = tmp_path / 'v.mp4'; f.write_bytes(b'x')
    item = _add_item(grid, tgw, str(f))
    monkeypatch.setattr(QtCore.QTimer, 'singleShot', lambda ms, cb: None)

    for _ in range(grid._MAX_THUMB_RETRIES + 1):
        grid._on_thumbnail_failed(str(f), "Could not read frame")

    assert item.thumbnail_failed is True
    # Cached under a PERMANENT reason so it's not retried forever.
    assert cache.is_thumbnail_failed(str(f)) is True


def test_hard_failure_is_permanent_immediately(qapp, tmp_path, monkeypatch):
    from PyQt6 import QtCore
    grid, gen, cache, tgw = _make_grid(tmp_path)
    f = tmp_path / 'broken.mp4'; f.write_bytes(b'x')
    item = _add_item(grid, tgw, str(f))

    scheduled = []
    monkeypatch.setattr(QtCore.QTimer, 'singleShot',
                        lambda ms, cb: scheduled.append(cb))

    grid._on_thumbnail_failed(str(f), "Cannot open video")
    assert item.thumbnail_failed is True       # no retry for hard failures
    assert len(scheduled) == 0


def test_evicted_pixmap_reloads_from_disk_not_generating(qapp, tmp_path):
    """Regression: a generated thumbnail must NOT revert to 'Generating…' after
    the user scrolls away and back. Scrolling past many videos evicts older
    pixmaps from the bounded RAM LRU; on return the widget is recreated and
    must reload the thumbnail straight from the on-disk JPEG cache (instant),
    not show a loading state or re-decode. _load_disk_thumbnail is that reload."""
    import numpy as np
    grid, gen, cache, tgw = _make_grid(tmp_path)
    f = tmp_path / "v.mp4"; f.write_bytes(b"\x00" * 64)
    # The thumbnail was generated earlier and lives in the on-disk cache…
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8); frame[:] = (40, 80, 120)
    assert cache.store_thumbnail(str(f), 5.0, frame, 321.0) is not None
    item = tgw._Item(path=str(f), is_folder=False, seek_time=5.0)

    # …but its RAM pixmap was evicted (in-memory cache empty). Recreation must
    # reload it from disk, scaled to display width — never "Generating…".
    res = grid._load_disk_thumbnail(item)
    assert res is not None, "evicted-but-on-disk thumbnail failed to reload"
    pix, dur = res
    assert not pix.isNull()
    assert 0 < pix.width() <= tgw._THUMB_DISPLAY_MAX_W
    assert dur == 321.0

    # A video with NO on-disk thumbnail returns None (→ falls back to load/retry).
    g = tmp_path / "nope.mp4"; g.write_bytes(b"\x00" * 64)
    item2 = tgw._Item(path=str(g), is_folder=False, seek_time=5.0)
    assert grid._load_disk_thumbnail(item2) is None


# ── service level: a cancelled worker must not leak _pending ────────────────
# (Regression: previewing preempts background thumbnail workers via
# force_close; the cancelled worker emitted nothing, so its path stayed in
# _pending forever and request_thumbnail() deduped every retry → stuck
# "Generating…". A cancelled-for-preview worker must clear _pending AND
# re-queue; a cancel_all'd worker must clear and NOT re-queue.)

def _make_service(tmp_path, monkeypatch):
    from thumbnail_generator import ThumbnailGeneratorService
    from cache_manager import CacheManager
    cache = CacheManager(str(tmp_path / 'c.db'), str(tmp_path / 't'))
    gen = ThumbnailGeneratorService(cache)
    # Don't actually run workers on the pool — we drive run() deterministically.
    # (*a swallows the optional QThreadPool priority arg request_thumbnail passes.)
    monkeypatch.setattr(gen._pool, 'start', lambda w, *a: None)
    return gen


def test_preempted_worker_clears_pending_and_requeues(qapp, tmp_path, monkeypatch):
    gen = _make_service(tmp_path, monkeypatch)
    f = tmp_path / 'v.mp4'; f.write_bytes(b'\x00' * 64)
    gen.request_thumbnail(str(f), 1.0)
    w1 = gen._pending[str(f)]

    # A hover preview preempts it: force_close(requeue=True), then the worker
    # exits run() cancelled (emits neither ready nor error — only finished).
    w1.force_close(requeue=True)
    w1.run()

    assert str(f) in gen._pending, (
        "preempted worker left its path NOT pending — that's the stuck "
        "'Generating…' bug (it would be re-requested-and-deduped forever)"
    )
    assert gen._pending[str(f)] is not w1, "expected a fresh re-queued worker"
    gen.cancel_all()


def test_cancel_all_worker_does_not_requeue(qapp, tmp_path, monkeypatch):
    gen = _make_service(tmp_path, monkeypatch)
    f = tmp_path / 'v.mp4'; f.write_bytes(b'\x00' * 64)
    gen.request_thumbnail(str(f), 1.0)
    w1 = gen._pending[str(f)]

    gen.cancel_all()    # clears _pending + force_close(requeue=False)
    w1.run()            # cancelled exit → finished → must NOT re-queue

    assert str(f) not in gen._pending, (
        "cancel_all'd worker must not regenerate (folder teardown / delete)"
    )
