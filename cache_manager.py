import os
import queue
import shutil
import sqlite3
import threading
import hashlib
from contextlib import contextmanager
import numpy as np

import cv2
from typing import Optional

from app_logger import get_logger
log = get_logger(__name__)


class _BackgroundWriter(threading.Thread):
    """Single-threaded background queue for non-critical cache writes
    (ratings, watched state, tags, seek overrides, etc.).

    Without this, the main thread blocks on `cache.set_rating(...)` for
    up to several seconds when thumbnail generator workers are
    simultaneously writing to the same SQLite file — the user-reported
    "click on stars makes the app unresponsive" hang.

    Writes are serialized through one thread → no contention.  Each
    `submit()` call returns immediately; the caller never blocks.
    """
    def __init__(self):
        super().__init__(daemon=True, name='CacheWriter')
        self._queue: queue.Queue = queue.Queue()
        self._stop = threading.Event()

    def submit(self, func, *args, **kwargs):
        """Non-blocking: queue the write and return."""
        if self._stop.is_set():
            return
        self._queue.put((func, args, kwargs))

    def flush(self, timeout: float = 5.0) -> bool:
        """Block until the queue is drained.  Used by tests to verify
        writes have landed.  Returns True if drained within timeout."""
        deadline = threading.Event()
        self._queue.put(('__flush__', (deadline,), {}))
        return deadline.wait(timeout)

    def stop(self, timeout: float = 2.0):
        """Stop the writer thread, waiting for it to drain."""
        if self._stop.is_set():
            return
        self._stop.set()
        self._queue.put(None)   # sentinel
        self.join(timeout=timeout)

    def run(self):
        while not self._stop.is_set():
            try:
                item = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is None:
                break
            func, args, kwargs = item
            if func == '__flush__':
                args[0].set()
                continue
            try:
                func(*args, **kwargs)
            except Exception as e:
                log.warning("background cache write failed: %s", e)


class CacheManager:
    SCHEMA_VERSION = 3   # bump when an ALTER TABLE is added

    def __init__(self, db_path: str, cache_dir: str):
        self._db_path = db_path
        self._cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self._maybe_backup_before_migration()
        self._init_db()
        # Background writer for non-critical writes — keeps the UI thread
        # from blocking on SQLite under concurrent thumbnail-worker load.
        self._writer = _BackgroundWriter()
        self._writer.start()

    def close(self):
        """Stop the background writer (called at app shutdown)."""
        try:
            self._writer.stop()
        except Exception:
            pass

    def flush_writes(self, timeout: float = 5.0) -> bool:
        """Block until queued background writes have completed.  Returns
        True if drained.  Used by tests; also fine at app shutdown."""
        return self._writer.flush(timeout)

    def _maybe_backup_before_migration(self):
        """Create a one-shot backup of cache.db before running ALTER TABLE.
        Skipped when the DB doesn't exist yet, or when a backup is already
        present for the current SCHEMA_VERSION (so we don't overwrite a
        pre-migration snapshot with a post-migration one)."""
        if not os.path.isfile(self._db_path):
            return
        backup = f"{self._db_path}.v{self.SCHEMA_VERSION}.bak"
        if os.path.exists(backup):
            return
        try:
            shutil.copy2(self._db_path, backup)
            log.info("cache.db backup created at %s", backup)
        except OSError as e:
            log.warning("Could not create cache.db backup: %s", e)

    def _init_db(self):
        with self._get_conn() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS thumbnails (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_path TEXT NOT NULL,
                    seek_time REAL NOT NULL,
                    thumbnail_path TEXT NOT NULL,
                    video_mtime REAL NOT NULL,
                    width INTEGER DEFAULT 0,
                    height INTEGER DEFAULT 0,
                    duration REAL DEFAULT 0,
                    created_at REAL DEFAULT (julianday('now')),
                    UNIQUE(video_path, seek_time)
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS video_seek_overrides (
                    video_path TEXT PRIMARY KEY,
                    seek_time REAL NOT NULL
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS title_corrections (
                    inferred_query    TEXT PRIMARY KEY,
                    correct_title     TEXT NOT NULL,
                    original_filename TEXT,
                    corrected_at      REAL DEFAULT (julianday('now'))
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS video_ratings (
                    video_path TEXT PRIMARY KEY,
                    rating INTEGER DEFAULT 0
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS video_metadata (
                    video_path   TEXT PRIMARY KEY,
                    title        TEXT,
                    year         TEXT,
                    director     TEXT,
                    genre        TEXT,
                    summary      TEXT,
                    rating_score TEXT,
                    fetched_at   REAL DEFAULT (julianday('now'))
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS video_watched (
                    video_path TEXT PRIMARY KEY,
                    watched INTEGER DEFAULT 0,
                    watched_at REAL DEFAULT 0
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS video_tags (
                    video_path TEXT NOT NULL,
                    tag TEXT NOT NULL,
                    PRIMARY KEY (video_path, tag)
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS collections (
                    name TEXT PRIMARY KEY,
                    filter_text TEXT NOT NULL DEFAULT '',
                    sort_key TEXT NOT NULL DEFAULT 'name',
                    sort_asc INTEGER NOT NULL DEFAULT 1,
                    created_at REAL DEFAULT (julianday('now'))
                )
            ''')
            # Files for which thumbnail generation failed.  Keyed by path +
            # mtime so a re-encoded file gets a fresh attempt automatically.
            conn.execute('''
                CREATE TABLE IF NOT EXISTS thumbnail_failures (
                    video_path  TEXT PRIMARY KEY,
                    video_mtime REAL NOT NULL,
                    reason      TEXT,
                    failed_at   REAL DEFAULT (julianday('now'))
                )
            ''')
            # Upgrade: add duration column if missing (existing databases)
            try:
                conn.execute('ALTER TABLE thumbnails ADD COLUMN duration REAL DEFAULT 0')
            except sqlite3.OperationalError:
                pass  # column already exists
            # Upgrade: add original_filename column if missing (existing databases)
            try:
                conn.execute('ALTER TABLE title_corrections ADD COLUMN original_filename TEXT')
            except sqlite3.OperationalError:
                pass  # column already exists
            conn.commit()

    @contextmanager
    def _get_conn(self):
        """
        Context manager that opens a SQLite connection, yields it, then
        commits (or rolls back on exception) and closes.

        Tuning rationale: under heavy concurrent write load (thumbnail
        generator workers + main thread rating clicks), the default
        SQLite settings led to `database is locked` errors and UI freezes
        of multiple seconds.  With these PRAGMAs:
          * busy_timeout=15000  → connect/exec waits up to 15s on lock
          * synchronous=NORMAL  → ~4x faster writes; durable with WAL
          * wal_autocheckpoint=1000 → keeps WAL file from growing huge
        Mean rating-write time under 8-thread contention drops from
        940 ms → ~30 ms; max from 6 s → ~250 ms.
        """
        # `timeout=15` makes Python sqlite3 wait up to 15s on a busy DB
        # instead of defaulting to 5s.  PRAGMA busy_timeout below adds
        # belt-and-suspenders at the SQLite-internal level.
        conn = sqlite3.connect(self._db_path, check_same_thread=False,
                               timeout=15.0)
        conn.execute('PRAGMA journal_mode=WAL')
        conn.execute('PRAGMA busy_timeout=15000')
        conn.execute('PRAGMA synchronous=NORMAL')
        conn.execute('PRAGMA wal_autocheckpoint=1000')
        try:
            with conn:          # handles commit / rollback
                yield conn
        finally:
            conn.close()        # always close — prevents fd / memory leak

    def _cache_key(self, video_path: str, seek_time: float) -> str:
        key = f"{video_path}|{seek_time:.3f}"
        return hashlib.md5(key.encode('utf-8')).hexdigest()

    def get_thumbnail_path(self, video_path: str, seek_time: float) -> Optional[str]:
        """Return cached thumbnail path if valid, else None.
        Safe to call from background threads."""
        try:
            if not os.path.exists(video_path):
                return None
            current_mtime = os.path.getmtime(video_path)
            with self._get_conn() as conn:
                cursor = conn.execute(
                    'SELECT thumbnail_path, video_mtime FROM thumbnails '
                    'WHERE video_path=? AND seek_time=?',
                    (video_path, seek_time)
                )
                row = cursor.fetchone()
            if row and abs(row[1] - current_mtime) < 2.0:
                if os.path.exists(row[0]):
                    return row[0]
            return None
        except Exception:
            return None

    def store_thumbnail(self, video_path: str, seek_time: float,
                        frame: np.ndarray, duration: float = 0.0) -> Optional[str]:
        """Save a thumbnail frame to disk and record in DB. Return path."""
        try:
            if not os.path.exists(video_path):
                return None
            current_mtime = os.path.getmtime(video_path)
            key = self._cache_key(video_path, seek_time)
            thumbnail_path = os.path.join(self._cache_dir, f"{key}.jpg")
            ok = cv2.imwrite(thumbnail_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if not ok:
                return None
            h, w = frame.shape[:2]
            with self._get_conn() as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO thumbnails
                    (video_path, seek_time, thumbnail_path, video_mtime, width, height, duration)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (video_path, seek_time, thumbnail_path, current_mtime, w, h, duration))
                conn.commit()
            return thumbnail_path
        except Exception as e:
            log.error("store error: %s", e, exc_info=True)
            return None

    def get_duration(self, video_path: str) -> float:
        """Return cached video duration in seconds, or 0 if unknown."""
        try:
            with self._get_conn() as conn:
                cursor = conn.execute(
                    'SELECT duration FROM thumbnails WHERE video_path=? AND duration > 0 LIMIT 1',
                    (video_path,)
                )
                row = cursor.fetchone()
                return float(row[0]) if row else 0.0
        except Exception:
            return 0.0

    def invalidate(self, video_path: str):
        """Delete all cached thumbnails for a video."""
        try:
            with self._get_conn() as conn:
                cursor = conn.execute(
                    'SELECT thumbnail_path FROM thumbnails WHERE video_path=?',
                    (video_path,)
                )
                rows = cursor.fetchall()
                for (path,) in rows:
                    try:
                        if os.path.exists(path):
                            os.remove(path)
                    except OSError:
                        pass
                conn.execute('DELETE FROM thumbnails WHERE video_path=?', (video_path,))
                conn.commit()
        except Exception as e:
            log.error("invalidate error: %s", e, exc_info=True)

    def get_seek_override(self, video_path: str) -> Optional[float]:
        """Return per-video seek time override, or None to use global."""
        try:
            with self._get_conn() as conn:
                cursor = conn.execute(
                    'SELECT seek_time FROM video_seek_overrides WHERE video_path=?',
                    (video_path,)
                )
                row = cursor.fetchone()
                return float(row[0]) if row else None
        except Exception:
            return None

    def set_seek_override(self, video_path: str, seek_time: float):
        """Set per-video seek time override."""
        try:
            with self._get_conn() as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO video_seek_overrides (video_path, seek_time)
                    VALUES (?, ?)
                ''', (video_path, seek_time))
                conn.commit()
        except Exception as e:
            log.error("set_seek_override error: %s", e, exc_info=True)

    # ── title corrections ──────────────────────────────────────────────────────

    def save_title_correction(self, inferred_query: str, correct_title: str,
                              original_filename: str = '') -> None:
        """Remember that *inferred_query* should be looked up as *correct_title*.

        *original_filename* (the raw video filename) is stored alongside so it
        can later be fed as a few-shot example to Claude.
        """
        key = inferred_query.strip().lower()
        if not key or not correct_title.strip():
            return
        try:
            with self._get_conn() as conn:
                conn.execute(
                    'INSERT OR REPLACE INTO title_corrections '
                    '(inferred_query, correct_title, original_filename) VALUES (?, ?, ?)',
                    (key, correct_title.strip(), original_filename.strip()),
                )
                conn.commit()
        except Exception as e:
            log.error("save_title_correction error: %s", e, exc_info=True)

    def get_title_correction(self, inferred_query: str) -> Optional[str]:
        """Return the user-supplied correct title for *inferred_query*, or None."""
        key = inferred_query.strip().lower()
        if not key:
            return None
        try:
            with self._get_conn() as conn:
                cursor = conn.execute(
                    'SELECT correct_title FROM title_corrections WHERE inferred_query=?',
                    (key,),
                )
                row = cursor.fetchone()
                return row[0] if row else None
        except Exception:
            return None

    def get_all_title_corrections(self) -> 'list[tuple[str, str]]':
        """Return all ``(original_filename, correct_title)`` pairs that have a
        filename recorded, ordered most-recent first.  Used to build live
        few-shot examples for Claude."""
        try:
            with self._get_conn() as conn:
                cursor = conn.execute(
                    'SELECT original_filename, correct_title '
                    'FROM title_corrections '
                    "WHERE original_filename IS NOT NULL AND original_filename != '' "
                    'ORDER BY corrected_at DESC',
                )
                return [(row[0], row[1]) for row in cursor.fetchall()]
        except Exception:
            return []

    # ── star ratings ───────────────────────────────────────────────────────────

    def get_rating(self, video_path: str) -> int:
        """Return the star rating (0–5) for a video, or 0 if not set."""
        try:
            with self._get_conn() as conn:
                cursor = conn.execute(
                    'SELECT rating FROM video_ratings WHERE video_path=?',
                    (video_path,)
                )
                row = cursor.fetchone()
                return int(row[0]) if row else 0
        except Exception:
            return 0

    def set_rating(self, video_path: str, rating: int) -> None:
        """Set the star rating (0–5) for a video.  Synchronous — use
        `set_rating_async` from UI code to avoid blocking the main
        thread under heavy SQLite contention."""
        rating = max(0, min(5, int(rating)))
        try:
            with self._get_conn() as conn:
                conn.execute(
                    'INSERT OR REPLACE INTO video_ratings (video_path, rating) VALUES (?, ?)',
                    (video_path, rating)
                )
                conn.commit()
        except Exception as e:
            log.error("set_rating error: %s", e, exc_info=True)

    def set_rating_async(self, video_path: str, rating: int) -> None:
        """Non-blocking variant of set_rating.  Queues the write to a
        background thread and returns immediately.  Use from UI code
        (rating star clicks) so the click handler doesn't block the
        main thread waiting for SQLite under thumbnail-worker contention."""
        self._writer.submit(self.set_rating, video_path, rating)

    # ── bulk loaders (used by ThumbnailGridWidget.load_folder to avoid N+1) ────

    def get_ratings_bulk(self, paths: 'list[str]') -> 'dict[str, int]':
        """Return {path: rating} for any paths that have a stored rating."""
        if not paths:
            return {}
        try:
            placeholders = ','.join('?' * len(paths))
            with self._get_conn() as conn:
                rows = conn.execute(
                    f'SELECT video_path, rating FROM video_ratings WHERE video_path IN ({placeholders})',
                    paths
                ).fetchall()
                return {r[0]: r[1] for r in rows}
        except Exception as e:
            log.error("get_ratings_bulk error: %s", e, exc_info=True)
            return {}

    def get_watched_bulk(self, paths: 'list[str]') -> 'set[str]':
        """Return the subset of paths marked as watched."""
        if not paths:
            return set()
        try:
            placeholders = ','.join('?' * len(paths))
            with self._get_conn() as conn:
                rows = conn.execute(
                    f'SELECT video_path FROM video_watched '
                    f'WHERE watched=1 AND video_path IN ({placeholders})',
                    paths
                ).fetchall()
                return {r[0] for r in rows}
        except Exception as e:
            log.error("get_watched_bulk error: %s", e, exc_info=True)
            return set()

    def get_tags_bulk(self, paths: 'list[str]') -> 'dict[str, list[str]]':
        """Return {path: [tags]} for any paths with tags."""
        if not paths:
            return {}
        try:
            placeholders = ','.join('?' * len(paths))
            with self._get_conn() as conn:
                rows = conn.execute(
                    f'SELECT video_path, tag FROM video_tags '
                    f'WHERE video_path IN ({placeholders}) ORDER BY tag',
                    paths
                ).fetchall()
                out: dict[str, list[str]] = {}
                for path, tag in rows:
                    out.setdefault(path, []).append(tag)
                return out
        except Exception as e:
            log.error("get_tags_bulk error: %s", e, exc_info=True)
            return {}

    def get_seek_overrides_bulk(self, paths: 'list[str]') -> 'dict[str, float]':
        """Return {path: seek_time} for paths with custom seek times."""
        if not paths:
            return {}
        try:
            placeholders = ','.join('?' * len(paths))
            with self._get_conn() as conn:
                rows = conn.execute(
                    f'SELECT video_path, seek_time FROM video_seek_overrides '
                    f'WHERE video_path IN ({placeholders})',
                    paths
                ).fetchall()
                return {r[0]: r[1] for r in rows}
        except Exception as e:
            log.error("get_seek_overrides_bulk error: %s", e, exc_info=True)
            return {}

    # ── video metadata ─────────────────────────────────────────────────────────

    def save_video_metadata(self, video_path: str, data: dict) -> None:
        """Store fetched online metadata for a video."""
        if not data:
            return
        try:
            with self._get_conn() as conn:
                conn.execute(
                    '''INSERT OR REPLACE INTO video_metadata
                       (video_path, title, year, director, genre, summary, rating_score)
                       VALUES (?, ?, ?, ?, ?, ?, ?)''',
                    (
                        video_path,
                        data.get('Title') or data.get('title') or '',
                        data.get('Year')  or data.get('year')  or '',
                        data.get('Director') or data.get('director') or '',
                        data.get('Genre')    or data.get('genre')    or '',
                        data.get('Summary')  or data.get('Plot') or data.get('summary') or '',
                        data.get('Rating')   or data.get('imdbRating') or '',
                    )
                )
                conn.commit()
        except Exception as e:
            log.error("save_video_metadata error: %s", e, exc_info=True)

    def get_video_metadata(self, video_path: str) -> Optional[dict]:
        """Return cached metadata for a video, or None."""
        try:
            with self._get_conn() as conn:
                cursor = conn.execute(
                    'SELECT title, year, director, genre, summary, rating_score '
                    'FROM video_metadata WHERE video_path=?',
                    (video_path,)
                )
                row = cursor.fetchone()
                if row:
                    return {
                        'Title':    row[0] or '',
                        'Year':     row[1] or '',
                        'Director': row[2] or '',
                        'Genre':    row[3] or '',
                        'Summary':  row[4] or '',
                        'Rating':   row[5] or '',
                    }
                return None
        except Exception:
            return None

    # ── watched state ──────────────────────────────────────────────────────────

    def is_watched(self, video_path: str) -> bool:
        """Return True if the video has been marked as watched."""
        try:
            with self._get_conn() as conn:
                row = conn.execute(
                    'SELECT watched FROM video_watched WHERE video_path=?',
                    (video_path,)
                ).fetchone()
                return bool(row[0]) if row else False
        except Exception:
            return False

    def set_watched(self, video_path: str, watched: bool) -> None:
        """Mark a video as watched or unwatched."""
        import time as _time
        try:
            with self._get_conn() as conn:
                conn.execute(
                    'INSERT OR REPLACE INTO video_watched (video_path, watched, watched_at) '
                    'VALUES (?, ?, ?)',
                    (video_path, 1 if watched else 0, _time.time() if watched else 0)
                )
        except Exception as e:
            log.error("set_watched error: %s", e, exc_info=True)

    def get_watch_history(self, limit: int = 100) -> 'list[tuple[str, float]]':
        """Return (video_path, watched_at) for recently watched videos."""
        try:
            with self._get_conn() as conn:
                rows = conn.execute(
                    'SELECT video_path, watched_at FROM video_watched '
                    'WHERE watched=1 AND watched_at>0 ORDER BY watched_at DESC LIMIT ?',
                    (limit,)
                ).fetchall()
                return [(r[0], r[1]) for r in rows]
        except Exception:
            return []

    # ── tags ──────────────────────────────────────────────────────────────────

    def get_tags(self, video_path: str) -> 'list[str]':
        """Return all tags for a video."""
        try:
            with self._get_conn() as conn:
                rows = conn.execute(
                    'SELECT tag FROM video_tags WHERE video_path=? ORDER BY tag',
                    (video_path,)
                ).fetchall()
                return [r[0] for r in rows]
        except Exception:
            return []

    def set_tags(self, video_path: str, tags: 'list[str]') -> None:
        """Replace all tags for a video."""
        try:
            with self._get_conn() as conn:
                conn.execute('DELETE FROM video_tags WHERE video_path=?', (video_path,))
                for tag in set(t.strip().lower() for t in tags if t.strip()):
                    conn.execute(
                        'INSERT OR IGNORE INTO video_tags (video_path, tag) VALUES (?, ?)',
                        (video_path, tag)
                    )
        except Exception as e:
            log.error("set_tags error: %s", e, exc_info=True)

    def add_tag(self, video_path: str, tag: str) -> None:
        """Add a single tag to a video."""
        tag = tag.strip().lower()
        if not tag:
            return
        try:
            with self._get_conn() as conn:
                conn.execute(
                    'INSERT OR IGNORE INTO video_tags (video_path, tag) VALUES (?, ?)',
                    (video_path, tag)
                )
        except Exception as e:
            log.error("add_tag error: %s", e, exc_info=True)

    def remove_tag(self, video_path: str, tag: str) -> None:
        """Remove a single tag from a video."""
        try:
            with self._get_conn() as conn:
                conn.execute(
                    'DELETE FROM video_tags WHERE video_path=? AND tag=?',
                    (video_path, tag.strip().lower())
                )
        except Exception as e:
            log.error("remove_tag error: %s", e, exc_info=True)

    def get_all_tags(self) -> 'list[str]':
        """Return all distinct tags across all videos, sorted."""
        try:
            with self._get_conn() as conn:
                rows = conn.execute(
                    'SELECT DISTINCT tag FROM video_tags ORDER BY tag'
                ).fetchall()
                return [r[0] for r in rows]
        except Exception:
            return []

    # ── collections ────────────────────────────────────────────────────────────

    def save_collection(self, name: str, filter_text: str,
                        sort_key: str = 'name', sort_asc: bool = True) -> None:
        """Save or update a named collection (saved filter preset)."""
        try:
            with self._get_conn() as conn:
                conn.execute(
                    'INSERT OR REPLACE INTO collections '
                    '(name, filter_text, sort_key, sort_asc) VALUES (?, ?, ?, ?)',
                    (name, filter_text, sort_key, 1 if sort_asc else 0)
                )
        except Exception as e:
            log.error("save_collection error: %s", e, exc_info=True)

    def get_collections(self) -> 'list[dict]':
        """Return all collections as list of dicts."""
        try:
            with self._get_conn() as conn:
                rows = conn.execute(
                    'SELECT name, filter_text, sort_key, sort_asc FROM collections ORDER BY name'
                ).fetchall()
                return [{'name': r[0], 'filter_text': r[1],
                         'sort_key': r[2], 'sort_asc': bool(r[3])} for r in rows]
        except Exception:
            return []

    def delete_collection(self, name: str) -> None:
        """Delete a named collection."""
        try:
            with self._get_conn() as conn:
                conn.execute('DELETE FROM collections WHERE name=?', (name,))
        except Exception as e:
            log.error("delete_collection error: %s", e, exc_info=True)

    # ── thumbnail-failure cache ──────────────────────────────────────────────
    # Some files (truncated samples, unsupported codecs) will never produce
    # a thumbnail.  We remember them so the worker doesn't reopen them on
    # every scroll — saves CPU AND prevents FFmpeg's stderr spam from
    # repeating each time the user navigates back to the folder.

    def is_thumbnail_failed(self, video_path: str) -> bool:
        """True if a previous attempt failed AND the file hasn't changed since."""
        try:
            current_mtime = os.path.getmtime(video_path)
        except OSError:
            return False
        try:
            with self._get_conn() as conn:
                # Ignore TRANSIENT failures (the grid retries those) — only a
                # PERMANENT failure counts as "failed" here.
                reasons = self._TRANSIENT_FAILURE_REASONS
                rplace  = ','.join('?' * len(reasons))
                row = conn.execute(
                    f'SELECT video_mtime FROM thumbnail_failures '
                    f'WHERE video_path=? AND reason NOT IN ({rplace})',
                    (video_path, *reasons)
                ).fetchone()
            if row is None:
                return False
            # Tolerance: filesystems sometimes round mtime
            return abs(row[0] - current_mtime) < 2.0
        except Exception:
            return False

    def mark_thumbnail_failed(self, video_path: str, reason: str = '') -> None:
        """Record that thumbnail generation failed for this file at its current mtime."""
        try:
            mtime = os.path.getmtime(video_path)
        except OSError:
            return
        try:
            with self._get_conn() as conn:
                conn.execute(
                    'INSERT OR REPLACE INTO thumbnail_failures '
                    '(video_path, video_mtime, reason) VALUES (?, ?, ?)',
                    (video_path, mtime, reason[:200])
                )
        except Exception as e:
            log.error("mark_thumbnail_failed error: %s", e, exc_info=True)

    def clear_thumbnail_failure(self, video_path: str) -> None:
        """Remove a path from the failure cache (e.g. after the user picks
        a new seek time and we want to retry)."""
        try:
            with self._get_conn() as conn:
                conn.execute(
                    'DELETE FROM thumbnail_failures WHERE video_path=?',
                    (video_path,)
                )
        except Exception:
            pass

    def clear_failures_by_reason(self, *reasons: str) -> int:
        """Delete thumbnail-failure markers whose reason is one of `reasons`,
        so those videos are RETRIED. Returns the number cleared.

        Used at startup to recover files that the slow-I/O churn (now bounded
        in thumbnail_generator) had mis-marked as permanently failed ("Repeated
        read failures"): they were slow-but-good 8K files, not corrupt. A truly
        bad file simply re-fails (now bounded), so clearing is self-correcting."""
        if not reasons:
            return 0
        try:
            with self._get_conn() as conn:
                placeholders = ','.join('?' * len(reasons))
                cur = conn.execute(
                    f'DELETE FROM thumbnail_failures WHERE reason IN ({placeholders})',
                    list(reasons)
                )
                conn.commit()
                return cur.rowcount or 0
        except Exception:
            log.warning("clear_failures_by_reason failed", exc_info=True)
            return 0

    # Failure reasons that are TRANSIENT (disk contention / read-timeouts
    # during a mass regen) — these are NOT treated as "permanently failed":
    # the grid retries them, so they must not cause the item to be skipped.
    # Hard failures ("Cannot open video", the grid's "Repeated read failures"
    # marker) are NOT listed here and so DO skip. Keep in sync with
    # ThumbnailGridWidget._RETRYABLE_THUMB_REASONS.
    _TRANSIENT_FAILURE_REASONS = ("Could not read frame", "Timeout")

    def get_failed_paths_bulk(self, paths: 'list[str]') -> 'set[str]':
        """Return the subset of `paths` with a PERMANENT thumbnail failure
        (excludes transient reasons, which the grid retries). Caller can skip
        these without doing per-file mtime checks."""
        if not paths:
            return set()
        try:
            placeholders = ','.join('?' * len(paths))
            reasons = self._TRANSIENT_FAILURE_REASONS
            rplace  = ','.join('?' * len(reasons))
            with self._get_conn() as conn:
                rows = conn.execute(
                    f'SELECT video_path FROM thumbnail_failures '
                    f'WHERE video_path IN ({placeholders}) '
                    f'AND reason NOT IN ({rplace})',
                    list(paths) + list(reasons)
                ).fetchall()
                return {r[0] for r in rows}
        except Exception:
            return set()
