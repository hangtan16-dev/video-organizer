import os
import sqlite3
import hashlib
from contextlib import contextmanager
import numpy as np
import cv2
from typing import Optional


class CacheManager:
    def __init__(self, db_path: str, cache_dir: str):
        self._db_path = db_path
        self._cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self._init_db()

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
        commits (or rolls back on exception) and closes.  Always call as
        ``with self._get_conn() as conn:``.
        """
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.execute('PRAGMA journal_mode=WAL')
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
            print(f"[Cache] store error: {e}")
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
            print(f"[Cache] invalidate error: {e}")

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
            print(f"[Cache] set_seek_override error: {e}")

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
            print(f"[Cache] save_title_correction error: {e}")

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
        """Set the star rating (0–5) for a video."""
        rating = max(0, min(5, int(rating)))
        try:
            with self._get_conn() as conn:
                conn.execute(
                    'INSERT OR REPLACE INTO video_ratings (video_path, rating) VALUES (?, ?)',
                    (video_path, rating)
                )
                conn.commit()
        except Exception as e:
            print(f"[Cache] set_rating error: {e}")

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
            print(f"[Cache] save_video_metadata error: {e}")

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
            print(f"[Cache] set_watched error: {e}")

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
            print(f"[Cache] set_tags error: {e}")

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
            print(f"[Cache] add_tag error: {e}")

    def remove_tag(self, video_path: str, tag: str) -> None:
        """Remove a single tag from a video."""
        try:
            with self._get_conn() as conn:
                conn.execute(
                    'DELETE FROM video_tags WHERE video_path=? AND tag=?',
                    (video_path, tag.strip().lower())
                )
        except Exception as e:
            print(f"[Cache] remove_tag error: {e}")

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
            print(f"[Cache] save_collection error: {e}")

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
            print(f"[Cache] delete_collection error: {e}")
