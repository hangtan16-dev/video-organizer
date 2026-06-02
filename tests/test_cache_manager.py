"""
Tests for CacheManager — covers schema migration, persistence, and the new
bulk loaders that eliminate N+1 queries during folder load.

These tests are pure-Python (no Qt event loop needed) and run fast.
"""
import os
import sqlite3
import tempfile
import shutil

import pytest

from cache_manager import CacheManager


# ── fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture
def cache(tmp_path):
    """Fresh CacheManager backed by a temp file. Cleaned up automatically."""
    db_file = tmp_path / "cache.db"
    cache_dir = tmp_path / "thumbnails"
    cm = CacheManager(str(db_file), str(cache_dir))
    yield cm


# ── ratings ───────────────────────────────────────────────────────────────────
def test_set_and_get_rating(cache):
    cache.set_rating("/videos/a.mp4", 4)
    assert cache.get_rating("/videos/a.mp4") == 4


def test_get_rating_missing_returns_zero(cache):
    assert cache.get_rating("/nonexistent.mp4") == 0


def test_overwrite_rating(cache):
    cache.set_rating("/videos/a.mp4", 3)
    cache.set_rating("/videos/a.mp4", 5)
    assert cache.get_rating("/videos/a.mp4") == 5


# ── tags ──────────────────────────────────────────────────────────────────────
def test_set_tags_replaces_all(cache):
    cache.set_tags("/v.mp4", ["action", "drama"])
    assert sorted(cache.get_tags("/v.mp4")) == ["action", "drama"]
    cache.set_tags("/v.mp4", ["comedy"])
    assert cache.get_tags("/v.mp4") == ["comedy"]


def test_set_tags_normalizes_case_and_whitespace(cache):
    cache.set_tags("/v.mp4", ["  Action  ", "DRAMA", ""])
    tags = cache.get_tags("/v.mp4")
    assert sorted(tags) == ["action", "drama"]


def test_set_tags_dedupes(cache):
    cache.set_tags("/v.mp4", ["action", "Action", "ACTION"])
    assert cache.get_tags("/v.mp4") == ["action"]


def test_add_remove_tag(cache):
    cache.set_tags("/v.mp4", ["action"])
    cache.add_tag("/v.mp4", "drama")
    assert sorted(cache.get_tags("/v.mp4")) == ["action", "drama"]
    cache.remove_tag("/v.mp4", "action")
    assert cache.get_tags("/v.mp4") == ["drama"]


# ── watched ───────────────────────────────────────────────────────────────────
def test_set_watched_persists(cache):
    cache.set_watched("/v.mp4", True)
    assert cache.is_watched("/v.mp4") is True
    cache.set_watched("/v.mp4", False)
    assert cache.is_watched("/v.mp4") is False


def test_set_watched_persists_across_instances(cache):
    """Critical: writes must commit, not stay in an unflushed transaction."""
    cache.set_watched("/v.mp4", True)
    cache.set_rating("/v.mp4", 3)
    cache.set_tags("/v.mp4", ["action"])

    # Re-open the same DB from a fresh CacheManager
    fresh = CacheManager(cache._db_path, cache._cache_dir)
    assert fresh.is_watched("/v.mp4") is True
    assert fresh.get_rating("/v.mp4") == 3
    assert fresh.get_tags("/v.mp4") == ["action"]


# ── seek overrides ────────────────────────────────────────────────────────────
def test_seek_override(cache):
    cache.set_seek_override("/v.mp4", 12.5)
    assert cache.get_seek_override("/v.mp4") == 12.5


# ── bulk loaders ──────────────────────────────────────────────────────────────
def test_get_ratings_bulk_returns_only_set_paths(cache):
    cache.set_rating("/a.mp4", 5)
    cache.set_rating("/b.mp4", 3)
    result = cache.get_ratings_bulk(["/a.mp4", "/b.mp4", "/c.mp4"])
    assert result == {"/a.mp4": 5, "/b.mp4": 3}


def test_get_ratings_bulk_empty_input(cache):
    assert cache.get_ratings_bulk([]) == {}


def test_get_watched_bulk_returns_only_watched(cache):
    cache.set_watched("/a.mp4", True)
    cache.set_watched("/b.mp4", False)
    cache.set_watched("/c.mp4", True)
    result = cache.get_watched_bulk(["/a.mp4", "/b.mp4", "/c.mp4", "/d.mp4"])
    assert result == {"/a.mp4", "/c.mp4"}


def test_get_tags_bulk_groups_correctly(cache):
    cache.set_tags("/a.mp4", ["action", "drama"])
    cache.set_tags("/b.mp4", ["comedy"])
    result = cache.get_tags_bulk(["/a.mp4", "/b.mp4", "/c.mp4"])
    assert sorted(result["/a.mp4"]) == ["action", "drama"]
    assert result["/b.mp4"] == ["comedy"]
    assert "/c.mp4" not in result


def test_get_seek_overrides_bulk(cache):
    cache.set_seek_override("/a.mp4", 5.0)
    cache.set_seek_override("/b.mp4", 10.0)
    result = cache.get_seek_overrides_bulk(["/a.mp4", "/b.mp4", "/c.mp4"])
    assert result == {"/a.mp4": 5.0, "/b.mp4": 10.0}


def test_bulk_loaders_handle_large_input(cache):
    """SQLite has a default parameter limit (~999); ensure we don't hit it
    with a typical folder size. 500 is well within limits and a realistic
    upper bound for a single folder."""
    paths = [f"/v{i}.mp4" for i in range(500)]
    for i, p in enumerate(paths):
        cache.set_rating(p, (i % 5) + 1)
    result = cache.get_ratings_bulk(paths)
    assert len(result) == 500
    assert result["/v0.mp4"] == 1
    assert result["/v499.mp4"] == 5


# ── schema migration / backup ────────────────────────────────────────────────
def test_backup_created_on_existing_db(tmp_path):
    """When CacheManager opens an existing DB, it should snapshot it before
    running ALTER TABLE migrations."""
    db = tmp_path / "cache.db"
    cd = tmp_path / "thumbs"
    # First run — no backup expected (DB doesn't exist yet)
    CacheManager(str(db), str(cd))
    backup_path = str(db) + f".v{CacheManager.SCHEMA_VERSION}.bak"
    assert not os.path.exists(backup_path), "no backup should exist on first init"

    # Second run — DB exists; backup should be created
    CacheManager(str(db), str(cd))
    assert os.path.exists(backup_path), "backup should exist after second init"


def test_backup_not_overwritten(tmp_path):
    """Backup is one-shot per schema version: re-opening must not overwrite
    an existing backup with a freshly-migrated DB."""
    db = tmp_path / "cache.db"
    cd = tmp_path / "thumbs"
    CacheManager(str(db), str(cd))
    # Force a backup to exist
    cm = CacheManager(str(db), str(cd))
    cm.set_rating("/v.mp4", 5)
    backup_path = str(db) + f".v{CacheManager.SCHEMA_VERSION}.bak"
    assert os.path.exists(backup_path)

    # Capture backup mtime
    mtime_before = os.path.getmtime(backup_path)

    # Re-open — backup should NOT be touched
    CacheManager(str(db), str(cd))
    assert os.path.getmtime(backup_path) == mtime_before


# ── thumbnail-failure cache ──────────────────────────────────────────────────
class TestThumbnailFailureCache:
    def test_unknown_path_is_not_failed(self, cache, tmp_path):
        f = tmp_path / "v.mp4"
        f.write_bytes(b"")
        assert cache.is_thumbnail_failed(str(f)) is False

    def test_marked_path_is_failed(self, cache, tmp_path):
        f = tmp_path / "broken.mp4"
        f.write_bytes(b"junk")
        cache.mark_thumbnail_failed(str(f), "Cannot decode")
        assert cache.is_thumbnail_failed(str(f)) is True

    def test_failure_invalidated_when_file_changes(self, cache, tmp_path):
        """If the user re-encodes / replaces the file (mtime changes), the
        cached failure must NOT block a fresh attempt."""
        import time
        f = tmp_path / "broken.mp4"
        f.write_bytes(b"junk")
        cache.mark_thumbnail_failed(str(f), "test")
        assert cache.is_thumbnail_failed(str(f)) is True

        # Modify the file → mtime changes
        time.sleep(2.1)   # exceed the 2.0s tolerance in is_thumbnail_failed
        f.write_bytes(b"new content")
        os.utime(str(f), None)   # bump mtime to now

        assert cache.is_thumbnail_failed(str(f)) is False, (
            "Failure cache should expire when the file's mtime changes — "
            "otherwise re-encoded files never get a fresh thumbnail attempt."
        )

    def test_clear_thumbnail_failure(self, cache, tmp_path):
        f = tmp_path / "v.mp4"
        f.write_bytes(b"x")
        cache.mark_thumbnail_failed(str(f))
        assert cache.is_thumbnail_failed(str(f))
        cache.clear_thumbnail_failure(str(f))
        assert not cache.is_thumbnail_failed(str(f))

    def test_failure_survives_cache_reopen(self, cache, tmp_path):
        """The whole point of persisting failures is to skip re-decode on
        every app launch, not just every scroll."""
        f = tmp_path / "broken.mp4"
        f.write_bytes(b"x")
        cache.mark_thumbnail_failed(str(f), "test")

        fresh = CacheManager(cache._db_path, cache._cache_dir)
        assert fresh.is_thumbnail_failed(str(f))

    def test_get_failed_paths_bulk(self, cache, tmp_path):
        a = tmp_path / "a.mp4"; a.write_bytes(b"")
        b = tmp_path / "b.mp4"; b.write_bytes(b"")
        c = tmp_path / "c.mp4"; c.write_bytes(b"")
        cache.mark_thumbnail_failed(str(a), "1")
        cache.mark_thumbnail_failed(str(c), "2")
        result = cache.get_failed_paths_bulk([str(a), str(b), str(c)])
        assert result == {str(a), str(c)}

    def test_get_failed_paths_bulk_empty(self, cache):
        assert cache.get_failed_paths_bulk([]) == set()

    def test_transient_failures_are_not_permanent(self, cache, tmp_path):
        """Transient thumbnail failures (disk contention / read-timeout during
        a mass regen) must NOT be treated as permanent — the grid auto-retries
        them. Only hard failures stick. Regression for: changing
        thumbnails-per-row left videos blank with 'Cannot read frame' forever."""
        f = tmp_path / "v.mp4"; f.write_bytes(b"x")

        # Transient reasons → ignored (so the item gets re-requested/retried).
        for transient in ("Could not read frame", "Timeout"):
            cache.clear_thumbnail_failure(str(f))
            cache.mark_thumbnail_failed(str(f), transient)
            assert cache.is_thumbnail_failed(str(f)) is False, transient
            assert cache.get_failed_paths_bulk([str(f)]) == set(), transient

        # Hard / permanent reasons (incl. the grid's retries-exhausted marker)
        # → still treated as failed so we don't hammer an unreadable file.
        for permanent in ("Cannot open video", "Repeated read failures"):
            cache.clear_thumbnail_failure(str(f))
            cache.mark_thumbnail_failed(str(f), permanent)
            assert cache.is_thumbnail_failed(str(f)) is True, permanent
            assert cache.get_failed_paths_bulk([str(f)]) == {str(f)}, permanent

    def test_clear_failures_by_reason(self, cache, tmp_path):
        """Startup recovery: clear ONLY the given reason(s) so slow-I/O-churn
        false-positives ('Repeated read failures') are retried, while genuine
        hard failures ('Cannot open video') are left intact."""
        bad   = tmp_path / "bad.mp4";   bad.write_bytes(b"x")
        slow  = tmp_path / "slow.mp4";  slow.write_bytes(b"x")
        cache.mark_thumbnail_failed(str(bad), "Cannot open video")
        cache.mark_thumbnail_failed(str(slow), "Repeated read failures")
        assert cache.is_thumbnail_failed(str(bad)) is True
        assert cache.is_thumbnail_failed(str(slow)) is True

        n = cache.clear_failures_by_reason("Repeated read failures")
        assert n == 1
        # The slow-but-good file is now retryable; the corrupt one still skips.
        assert cache.is_thumbnail_failed(str(slow)) is False
        assert cache.is_thumbnail_failed(str(bad)) is True
        # No-op when nothing matches / no reasons given.
        assert cache.clear_failures_by_reason("Repeated read failures") == 0
        assert cache.clear_failures_by_reason() == 0

    def test_mark_for_missing_file_is_safe(self, cache, tmp_path):
        """If the file is gone by the time we try to mark it failed,
        the call should silently no-op rather than raise."""
        cache.mark_thumbnail_failed(str(tmp_path / "does_not_exist.mp4"), "x")
        # No assertion needed — just verify no exception escaped


# ── concurrency smoke test ────────────────────────────────────────────────────
def test_concurrent_writes_dont_corrupt_db(cache):
    """The CacheManager opens a new connection per call. WAL mode allows
    concurrent reads and serialises writes. Hammer it from threads to
    verify no corruption."""
    import threading
    paths = [f"/conc/{i}.mp4" for i in range(50)]

    def writer(start):
        for i in range(start, start + 10):
            cache.set_rating(paths[i], (i % 5) + 1)
            cache.set_tags(paths[i], [f"tag{i}", "common"])

    threads = [threading.Thread(target=writer, args=(s,)) for s in (0, 10, 20, 30, 40)]
    for t in threads: t.start()
    for t in threads: t.join()

    result = cache.get_ratings_bulk(paths)
    assert len(result) == 50
    # All ratings must be 1..5
    assert all(1 <= v <= 5 for v in result.values())
