"""
Video metadata dialog — local file info + anonymous online lookup.

Local metadata (background thread)
────────────────────────────────────
• Tries ffprobe (subprocess) first for rich data: codec, bitrate, container,
  audio streams, color space, exact frame rate.
• Falls back to cv2.VideoCapture for basic resolution / FPS / four-cc codec.
• Always computed from the file on disk — no network, no cache.

Online lookup (background thread)
───────────────────────────────────
Backends are tried in priority order until one returns a result:

  1. TMDB  — The Movie Database (https://www.themoviedb.org)
             Set the TMDB_API_KEY environment variable to enable.
             Richest data: director, full cast, genres, studio, box office.
             Free API key at: https://www.themoviedb.org/settings/api

  2. OMDb  — Open Movie Database (https://www.omdbapi.com)
             Set the OMDB_API_KEY environment variable to enable.
             Good structured data: director, actors, plot, awards, ratings.
             Free key (1 000 req/day) at: https://www.omdbapi.com/apikey.aspx

  3. Wikipedia — always available, no key required.
             Returns a short description + summary paragraph.
             Less structured than TMDB/OMDb but reliably finds most films.

  4. DuckDuckGo Instant Answers — last resort.
             Only fires when DDG has a direct knowledge-box for the title;
             unreliable for less-famous films.

All backends
• Use an opener built WITHOUT HTTPCookieProcessor → no cookies stored/sent
• Send DNT: 1
• Use a plain non-browser User-Agent (no fingerprinting)
• Degrade gracefully when the network is unavailable or has no match.
"""

import html as _html
import json
import os
import re
import subprocess
import sys
import urllib.parse
import urllib.request

import cv2

from PyQt6.QtCore import QObject, QRunnable, QThreadPool, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QDialog, QDialogButtonBox, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QTextBrowser, QVBoxLayout, QWidget,
)


# ── module-level singletons (set once at startup) ─────────────────────────────
_cm = None   # type: ignore[assignment]   # CacheManager | None
_custom_urls: 'list[str]' = []


def set_cache_manager(cache_manager) -> None:
    """Call once from main_window after creating CacheManager."""
    global _cm
    _cm = cache_manager


def set_custom_search_urls(urls: 'list[str]') -> None:
    """Set (or clear) the user-configured custom search URL bases."""
    global _custom_urls
    _custom_urls = [u.strip() for u in urls if u.strip()]


# ══════════════════════════════════════════════════════════════════════════════
# Formatters
# ══════════════════════════════════════════════════════════════════════════════

def _fmt_size(n: int) -> str:
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if n < 1024:
            return f"{n} {unit}" if unit == 'B' else f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PB"


def _fmt_duration(secs: float) -> str:
    s = int(secs)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def _kv_html(pairs: dict) -> str:
    """Render a dict as an HTML table for QTextBrowser (dark-theme colours)."""
    rows = ''
    for k, v in pairs.items():
        if k.startswith('_'):
            continue
        v_cell = _html.escape(str(v)).replace('\n', '<br>')
        rows += (
            '<tr>'
            f'<td style="color:#888;padding:2px 14px 2px 0;'
            f'white-space:nowrap;vertical-align:top">'
            f'{_html.escape(str(k))}</td>'
            f'<td style="color:#ddd;padding:2px 0;word-break:break-word">'
            f'{v_cell}</td>'
            '</tr>'
        )
    return (
        '<table style="border-collapse:collapse;font-size:10pt;'
        f'width:100%">{rows}</table>'
    )


# ══════════════════════════════════════════════════════════════════════════════
# Filename → search query
# ══════════════════════════════════════════════════════════════════════════════

_REMOVE_TAGS = re.compile(
    r'\b(?:'
    # Resolution
    r'1080[ip]|720[ip]|480[ip]|2160p|4[Kk]|8[Kk]|UHD|'
    # Source / disc / streaming
    r'BluRay|BDRip|BRRip|WEBRip|WEB-?DL|WEB|HDRip|HDTV|DVDRip|DVDScr|CAM|TS|'
    r'iT|AMZN|NF|DSNP|HMAX|PCOK|MA|Hybrid|'
    # Container formats (appear as bare tags in some scene releases)
    r'MP4|MKV|AVI|MOV|WEBM|M4V|'
    # Codec / encode
    r'x264|x265|H\.?264|H\.?265|HEVC|AVC|VP9|AV1|X265|X264|'
    r'\d+[Bb]it|'                                    # bit depth: 10bit 12bit
    # HDR / colour — handles HDR  HDR10  HDR10+  HDR10Plus
    r'HDR\d*(?:\+|[Pp]lus)?|SDR|DoVi|DV|Dolby\.?Vision|PLUS|'
    # Audio codec
    r'AAC|AC3|DTS[-\w]*|MP3|FLAC|DDP?5\.?1|DD5\.?1|TrueHD|EAC3|Atmos|'
    # Audio channel layout
    r'\d\.\d|'                                       # 7.1  5.1  2.0  1.0  etc.
    # Release flags
    r'INTERNAL|PROPER|REPACK|EXTENDED|UNRATED|DC|THEATRICAL|REMUX|'
    # Language / subtitle flags
    r'MULTi|SUBS?|DUBBED|SUBBED|'
    r'ENG|FRENCH|HINDI|LATINO|SPANISH|GERMAN|ITALIAN|PORTUGUESE|'
    r'ARABIC|RUSSIAN|CHINESE|JAPANESE|KOREAN|TURKISH|POLISH|VOSTFR|'
    # Season / episode codes
    r'S\d{1,2}E\d{1,3}|S\d{1,2}|E\d{1,3}'
    r')\b',
    re.IGNORECASE,
)
_YEAR_RE = re.compile(r'\b(19[5-9]\d|20[0-4]\d)\b')


def _filename_to_queries(path: str) -> 'list[str]':
    """
    Return up to 3 deduplicated search-query candidates from a video filename.

    Primary strategy — year-split (covers ~99 % of scene filenames):
        Scene releases follow the convention ``Title.Year.Tags-Group``.
        The title is ALWAYS everything before the year; all technical tags
        (source, codec, HDR, audio, language, release group …) live after
        the year, so they are ignored automatically without any tag vocabulary.

    Fallback — tag-removal regex:
        Used when no year is present, or when the year appears as the very
        first token (numeric titles such as "2001" or "1917" where the film
        title is itself a year-like number).  ``_REMOVE_TAGS`` strips known
        tags from the whole string in that case.
    """
    name = os.path.splitext(os.path.basename(path))[0]
    name = urllib.parse.unquote(name)

    # Strip leading [tag] / (tag) blocks common in fansub / scene prefixes
    name = re.sub(r'^\[.*?\]\s*', '', name)
    name = re.sub(r'^\(.*?\)\s*', '', name)

    year_m = _YEAR_RE.search(name)
    year   = year_m.group(0) if year_m else ''

    if year_m and year_m.start() > 0:
        # ── PRIMARY PATH ──────────────────────────────────────────────────────
        # Title is everything BEFORE the year token.  No tag vocabulary needed:
        # codec, HDR, audio, language, release-group are all after the year.
        raw_title = name[:year_m.start()]

    else:
        # ── FALLBACK PATH ─────────────────────────────────────────────────────
        # No year found, or the year IS the first token (e.g. "2001.A.Space…",
        # "1917.2019…" — note: 1917 itself doesn't match _YEAR_RE).
        # Apply tag-removal regex to the whole string.
        raw_title = name
        # Release-group suffix: handles -CMRG, -BEN.THE.MEN, -JustWatch, etc.
        raw_title = re.sub(
            r'-[A-Z0-9]+(?:\.[A-Z0-9]+)*(?:\[.*?\])?$',
            '',
            raw_title,
            flags=re.IGNORECASE,
        )
        raw_title = re.sub(r'\[.*?\]|\(.*?\)', '', raw_title)
        # CRITICAL: tag removal BEFORE separator replacement so "WEB-DL" is
        # still one hyphenated token when the regex runs.
        raw_title = _REMOVE_TAGS.sub(' ', raw_title)
        raw_title = _YEAR_RE.sub('', raw_title)   # remove year from body

    # Normalise remaining separators (dots, hyphens, underscores) to spaces
    title = ' '.join(re.sub(r'[._\-]+', ' ', raw_title).split()).strip()

    # Build candidate list ─────────────────────────────────────────────────────
    seen:       set[str]  = set()
    candidates: list[str] = []

    def _add(q: str) -> None:
        q = q.strip()
        if q and q.lower() not in seen:
            seen.add(q.lower())
            candidates.append(q)

    if title and year:
        _add(f"{title} ({year})")   # most specific: "Top Gun Maverick (2022)"
    if title:
        _add(title)                  # title-only fallback

    # Shorter variant: first 4 words + year (helps for very long titles with
    # residual noise in the fallback path)
    words = title.split()
    if len(words) > 4:
        short = ' '.join(words[:4])
        _add(f"{short} ({year})" if year else short)

    return candidates


def _filename_to_query(path: str) -> str:
    """Single-result wrapper kept for backward compatibility."""
    qs = _filename_to_queries(path)
    return qs[0] if qs else ''


def _claude_infer_titles(
    filename: str,
    learned_examples: 'list[tuple[str, str]] | None' = None,
) -> 'tuple[list[str], str]':
    """
    Ask Claude for up to 3 ranked candidate titles inferred from a video filename.

    Uses few-shot examples in a system prompt so Claude sees the exact
    input → output transformation rather than just a list of rules.

    *learned_examples* is a list of ``(original_filename, correct_title)`` pairs
    collected from the user's own past corrections.  They are injected into the
    system prompt so Claude generalises from the user's specific library and
    naming conventions — including abbreviations, language tags, or release-group
    styles that the built-in rules don't cover.

    Returns ``(titles, error_message)``.
    • On success:  titles is a non-empty list, error_message is ``''``.
    • On failure:  titles is ``[]``,  error_message describes what went wrong.
    """
    # Build the learned-examples section (capped at 100 most-recent entries so
    # the prompt stays within a reasonable size).
    learned_section = ''
    if learned_examples:
        lines = [
            f'{fn} → {title}'
            for fn, title in learned_examples[:100]
            if fn and title
        ]
        if lines:
            learned_section = (
                'The following examples come from corrections this user has '
                'already made for their own video library.  Study them carefully '
                '— they reveal the specific abbreviations, language tags, and '
                'release-group naming conventions used in this collection:\n'
                + '\n'.join(lines)
                + '\n\n'
            )

    system_prompt = (
        # ── task description ──────────────────────────────────────────
        'You are a video filename parser.  Given a filename, your job '
        'is to extract the clean movie or TV show title by removing all '
        'technical metadata tags.\n\n'
        # ── what to strip ─────────────────────────────────────────────
        'Always strip:\n'
        '• Resolution / format: 1080p 720p 2160p 4K UHD 8K\n'
        '• Source / encode: BluRay BDRip WEB-DL WEBRip HDTV DVDRip '
        'REMUX Hybrid MA x264 x265 HEVC AVC H264 H265 H.265 VP9 AV1\n'
        '• Bit depth: 10bit 12bit\n'
        '• HDR: HDR HDR10 HDR10+ HDR10Plus DV DoVi SDR PLUS\n'
        '• Audio codec: AAC AC3 DTS DTS-HD MP3 FLAC DD5.1 DDP5.1 '
        'Atmos TrueHD EAC3\n'
        '• Audio channels: 7.1 5.1 2.0 etc.\n'
        '• Release flags: INTERNAL PROPER REPACK EXTENDED THEATRICAL\n'
        '• Language / sub: MULTi ENG HINDI LATINO FRENCH SPANISH '
        'DUBBED SUBBED SUB SUBS VOSTFR and similar\n'
        '• Container tags: MP4 MKV AVI MOV (when appearing as a bare '
        'word, not a file extension)\n'
        '• Release-group names: everything after the last hyphen '
        '(e.g. -CMRG, -BEN.THE.MEN, -JustWatch) and bracket blocks '
        'like [Ben The Men]\n'
        '• Season / episode codes: S01E03 E05 S2 etc.\n\n'
        # ── built-in few-shot examples ────────────────────────────────
        'Examples (filename → correct title):\n'
        'Napoleon.2023.2160p.Dolby.Vision.HDR10.PLUS.ENG.HINDI.LATINO.'
        'Multi.Sub.DDP5.1.Atmos.DV.x265.MP4-BEN.THE.MEN.mp4'
        ' → Napoleon (2023)\n'
        'The.Creator.2023.REPACK.2160p.MA.WEB-DL.DDP5.1.Atmos.DV.HDR.'
        'H.265-FLUX.mkv'
        ' → The Creator (2023)\n'
        '1917.2019.2160p.UHD.BluRay.x265.10bit.HDR.DTS-HD.MA.TrueHD.'
        '7.1.Atmos-SWTYBLZ.mkv'
        ' → 1917 (2019)\n'
        'Alien.1979.INTERNAL.Theatrical.HDR10Plus.2160p.UHD.BluRay.'
        'X265-IAMABLE.mkv'
        ' → Alien (1979)\n'
        'Alien.Covenant.2017.2160p.BluRay.x265.10bit.HDR.TrueHD.7.1.'
        'Atmos-TERMiNAL.mkv'
        ' → Alien Covenant (2017)\n'
        'Alita.Battle.Angel.2019.iNTERNAL.HDR10Plus.2160p.UHD.BluRay.'
        'x265-JustWatch.mkv'
        ' → Alita Battle Angel (2019)\n'
        'Oppenheimer.2023.IMAX.4K.HDR.DV.2160p.BDRip Ita Eng x265-NAHOM.mkv'
        ' → Oppenheimer (2023)\n'
        'Predator.Badlands.2025.2160p.iT.WEB-DL.DV.HDR10+.MULTi[Ben The Men].mp4'
        ' → Predator Badlands (2025)\n'
        'Prey.2022.DUBBED.COMANCHE.2160p.DSNP.WEB-DL.x265.10bit.HDR.DDP5.1.Atmos-NOGRP.mkv'
        ' → Prey (2022)\n'
        'Tron.Ares.2025.2160p.iT.WEB-DL.DV.HDR10+.MULTi[Ben The Men].mp4'
        ' → Tron Ares (2025)\n\n'
        # ── user-learned examples (injected at runtime) ───────────────
        + learned_section
        # ── output format ─────────────────────────────────────────────
        + 'For the filename the user gives you, return up to 3 candidate '
        'titles ranked from most to least likely, one per line.  '
        'Include the year in parentheses when it appears in the filename.  '
        'Output ONLY the bare titles — no numbers, no bullets, no '
        'explanations, no arrows.'
    )

    api_key = os.environ.get('ANTHROPIC_API_KEY', '').strip()
    if not api_key:
        err = (
            'ANTHROPIC_API_KEY environment variable is not set.  '
            'Get a free key at https://console.anthropic.com and set it with:\n'
            '  set ANTHROPIC_API_KEY=sk-ant-...'
        )
        return [], err, system_prompt, err

    try:
        import anthropic                                 # lazy import
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model='claude-haiku-4-5',
            max_tokens=150,
            system=system_prompt,
            messages=[{
                'role':    'user',
                'content': filename,   # bare filename, matching the example format
            }],
        )
        if not msg.content:
            return [], 'Claude returned an empty response', system_prompt, ''
        raw_response = msg.content[0].text.strip()
        raw_lines    = raw_response.splitlines()
        titles: list[str] = []
        for ln in raw_lines:
            t = ln.strip()
            if not t:
                continue
            # Strip any prefix Claude adds despite instructions
            t = re.sub(r'^[\d]+[.\)]\s*', '', t)    # "1. " / "1) "
            t = re.sub(r'^[-*•→]\s*',     '', t)    # "- " / "* " / "• " / "→ "
            t = t.strip().strip('"\'')
            if t:
                titles.append(t)
        if titles:
            return titles[:3], '', system_prompt, raw_response
        return [], 'Claude returned an empty response', system_prompt, raw_response
    except ImportError:
        err = 'anthropic package not installed (run: pip install anthropic)'
        return [], err, system_prompt, err
    except Exception as exc:
        err = str(exc)
        return [], err, system_prompt, err


# ══════════════════════════════════════════════════════════════════════════════
# Local metadata gathering
# ══════════════════════════════════════════════════════════════════════════════

_NO_WINDOW: dict = (
    {'creationflags': 0x08000000} if sys.platform == 'win32' else {}
)


def _strip_html(text: str) -> str:
    """Remove HTML tags and decode entities (for cleaning DDG infobox values)."""
    return _html.unescape(re.sub(r'<[^>]+>', ' ', text)).strip()


def _try_ffprobe(path: str) -> 'dict | None':
    """
    Gather rich video metadata via ffprobe.
    Returns None if ffprobe is not in PATH or fails for any reason.
    """
    try:
        proc = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-print_format', 'json',
             '-show_streams', '-show_format', '--', path],
            capture_output=True, text=True, timeout=15, **_NO_WINDOW,
        )
        if proc.returncode != 0:
            return None
        data = json.loads(proc.stdout)
    except (FileNotFoundError, OSError,
            subprocess.TimeoutExpired, json.JSONDecodeError):
        return None

    meta: dict = {}
    fmt = data.get('format', {})

    if 'duration' in fmt:
        meta['Duration'] = _fmt_duration(float(fmt['duration']))

    if 'bit_rate' in fmt:
        br = int(fmt['bit_rate'])
        meta['Bit rate'] = (
            f"{br / 1_000_000:.2f} Mbps" if br >= 1_000_000
            else f"{br // 1000} kbps"
        )

    if 'format_long_name' in fmt:
        meta['Container'] = fmt['format_long_name']

    audio_tracks: list[str] = []
    for stream in data.get('streams', []):
        ctype = stream.get('codec_type', '')

        if ctype == 'video' and 'Resolution' not in meta:
            w = stream.get('width', 0)
            h = stream.get('height', 0)
            if w and h:
                meta['Resolution'] = f"{w} × {h}"

            codec = (stream.get('codec_long_name')
                     or stream.get('codec_name', ''))
            if codec:
                meta['Video codec'] = codec

            fps_str = stream.get('avg_frame_rate', '')
            if fps_str and '/' in fps_str:
                try:
                    num, den = (int(x) for x in fps_str.split('/'))
                    if den:
                        meta['Frame rate'] = f"{num / den:.3f} fps"
                except (ValueError, ZeroDivisionError):
                    pass

            # Per-stream video bitrate (fallback if format has no bit_rate)
            vbr = stream.get('bit_rate', '')
            if vbr and 'Bit rate' not in meta:
                br = int(vbr)
                meta['Video bitrate'] = (
                    f"{br / 1_000_000:.2f} Mbps" if br >= 1_000_000
                    else f"{br // 1000} kbps"
                )

            cs = stream.get('color_space', '')
            if cs:
                meta['Color space'] = cs

        elif ctype == 'audio':
            codec = (stream.get('codec_long_name')
                     or stream.get('codec_name', ''))
            ch = stream.get('channels', 0)
            sr = stream.get('sample_rate', '')
            parts = [p for p in (
                codec,
                f"{ch}ch" if ch else '',
                f"{int(sr) // 1000}kHz" if sr else '',
            ) if p]
            if parts:
                audio_tracks.append(', '.join(parts))

    if audio_tracks:
        meta['Audio'] = ' / '.join(audio_tracks)

    return meta or None


def _cv2_metadata(path: str, file_size: int) -> dict:
    """Basic metadata via cv2.VideoCapture (fallback when ffprobe unavailable)."""
    meta: dict = {}
    cap = None
    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return meta

        w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec  = ''.join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4))
        codec  = codec.strip('\x00').strip()

        if w and h:
            meta['Resolution'] = f"{w} × {h}"
        if fps > 0:
            meta['Frame rate'] = f"{fps:.3f} fps"
        if codec and codec.isprintable():
            meta['Video codec'] = codec
        if fps > 0 and frames > 0:
            dur = frames / fps
            meta['Duration'] = _fmt_duration(dur)
            if file_size > 0:
                br = int(file_size * 8 / dur)
                meta['Bit rate'] = (
                    f"{br / 1_000_000:.2f} Mbps" if br >= 1_000_000
                    else f"{br // 1000} kbps"
                )
    except Exception:
        pass
    finally:
        if cap is not None:
            cap.release()
    return meta


# ══════════════════════════════════════════════════════════════════════════════
# Online search backends
# ══════════════════════════════════════════════════════════════════════════════

def _make_opener() -> urllib.request.OpenerDirector:
    """
    Build a privacy-preserving urllib opener.

    • No HTTPCookieProcessor  → no cookies stored or sent
    • DNT: 1 header           → Do-Not-Track request
    • Non-browser User-Agent  → no fingerprint correlation
    """
    opener = urllib.request.build_opener()
    opener.addheaders = [
        ('User-Agent',      'Mozilla/5.0 VideoOrganizer/1.0 (anonymous-lookup)'),
        ('Accept',          'application/json, */*'),
        ('Accept-Language', 'en-US,en;q=0.9'),
        ('DNT',             '1'),
        ('Cache-Control',   'no-cache'),
    ]
    return opener


# ── TMDB ──────────────────────────────────────────────────────────────────────

def _search_tmdb(title: str, year: str, api_key: str) -> dict:
    """
    Query The Movie Database (TMDB) — richest structured data.

    Two requests: search → top result ID → detail + credits.
    Requires a free TMDB_API_KEY env var.
    """
    opener = _make_opener()

    # 1. Find the best-matching movie
    search_params = urllib.parse.urlencode({
        'api_key':        api_key,
        'query':          title,
        'year':           year,
        'language':       'en-US',
        'include_adult':  'false',
    })
    resp  = opener.open(
        f'https://api.themoviedb.org/3/search/movie?{search_params}', timeout=10
    )
    hits  = json.loads(resp.read().decode('utf-8', errors='replace')).get('results', [])
    if not hits:
        return {}

    movie_id = hits[0]['id']

    # 2. Fetch full details including cast/crew
    detail_params = urllib.parse.urlencode({
        'api_key':              api_key,
        'append_to_response':   'credits',
        'language':             'en-US',
    })
    resp2 = opener.open(
        f'https://api.themoviedb.org/3/movie/{movie_id}?{detail_params}', timeout=10
    )
    d = json.loads(resp2.read().decode('utf-8', errors='replace'))

    result: dict = {}

    if d.get('title'):
        result['Title'] = d['title']
    if d.get('release_date'):
        result['Released'] = d['release_date'][:4]
    if d.get('runtime'):
        result['Runtime'] = f"{d['runtime']} min"

    genres = [g['name'] for g in d.get('genres', [])]
    if genres:
        result['Genre'] = ', '.join(genres)

    overview = d.get('overview', '')
    if overview:
        result['Summary'] = overview[:500] + ('…' if len(overview) > 500 else '')

    crew      = d.get('credits', {}).get('crew', [])
    directors = [c['name'] for c in crew if c.get('job') == 'Director']
    if directors:
        result['Director'] = ', '.join(directors)

    cast = [c['name'] for c in d.get('credits', {}).get('cast', [])[:5]]
    if cast:
        result['Starring'] = ', '.join(cast)

    studios = [c['name'] for c in d.get('production_companies', [])[:2]]
    if studios:
        result['Studio'] = ', '.join(studios)

    countries = [c['name'] for c in d.get('production_countries', [])[:2]]
    if countries:
        result['Country'] = ', '.join(countries)

    if d.get('budget', 0) > 0:
        result['Budget'] = f"${d['budget']:,}"
    if d.get('revenue', 0) > 0:
        result['Box office'] = f"${d['revenue']:,}"

    result['_source']     = 'TMDB'
    result['_source_url'] = f'https://www.themoviedb.org/movie/{movie_id}'
    return result


# ── OMDb ──────────────────────────────────────────────────────────────────────

_OMDB_MAP: list[tuple[str, str]] = [
    ('Title',    'Title'),
    ('Year',     'Released'),
    ('Runtime',  'Runtime'),
    ('Genre',    'Genre'),
    ('Director', 'Director'),
    ('Actors',   'Starring'),
    ('Plot',     'Summary'),
    ('Country',  'Country'),
    ('Awards',   'Awards'),
    ('Language', 'Language'),
    ('Rated',    'Rated'),
]


def _search_omdb(title: str, year: str, api_key: str) -> dict:
    """
    Query the Open Movie Database (OMDb) — good structured data.

    One request; returns director, actors, plot, ratings, etc.
    Requires a free OMDB_API_KEY env var (1 000 requests/day).
    """
    params = urllib.parse.urlencode({
        'apikey': api_key,
        't':      title,
        'y':      year,
        'type':   'movie',
        'plot':   'short',
        'r':      'json',
    })
    opener = _make_opener()
    resp   = opener.open(f'https://www.omdbapi.com/?{params}', timeout=10)
    data   = json.loads(resp.read().decode('utf-8', errors='replace'))

    if data.get('Response') != 'True' or not data.get('Title'):
        return {}

    result: dict = {}
    for omdb_key, our_key in _OMDB_MAP:
        val = data.get(omdb_key, 'N/A')
        if val and val != 'N/A':
            result[our_key] = val

    imdb_id = data.get('imdbID', '')
    if imdb_id:
        result['_source']     = 'IMDb / OMDb'
        result['_source_url'] = f'https://www.imdb.com/title/{imdb_id}/'

    return result


# ── Wikipedia ─────────────────────────────────────────────────────────────────

_FILM_KEYWORDS = frozenset(
    ('film', 'movie', 'documentary', 'television', 'series',
     'miniseries', 'animated', 'special', 'picture')
)


def _search_wikipedia(query: str) -> dict:
    """
    Search Wikipedia for movie info — no API key required, always available.

    Two requests: text search → top-result page summary via REST API.
    Only accepts pages whose description contains a film-related keyword
    (to avoid matching a person/company with the same name).
    """
    opener = _make_opener()

    # Step 1: text search
    search_params = urllib.parse.urlencode({
        'action':   'query',
        'list':     'search',
        'srsearch': f'{query} film',
        'srlimit':  '5',
        'format':   'json',
        'utf8':     '1',
    })
    resp        = opener.open(
        f'https://en.wikipedia.org/w/api.php?{search_params}', timeout=10
    )
    search_data = json.loads(resp.read().decode('utf-8', errors='replace'))
    hits        = search_data.get('query', {}).get('search', [])
    if not hits:
        return {}

    # Step 2: check each hit's REST summary until we find a film page
    for hit in hits[:3]:
        page_title  = hit['title']
        summary_url = (
            'https://en.wikipedia.org/api/rest_v1/page/summary/'
            + urllib.parse.quote(page_title, safe='')
        )
        try:
            resp2 = opener.open(summary_url, timeout=10)
            s     = json.loads(resp2.read().decode('utf-8', errors='replace'))
        except Exception:
            continue

        description = s.get('description', '').lower()
        if not any(kw in description for kw in _FILM_KEYWORDS):
            continue                                 # skip non-film pages

        result: dict = {}
        if s.get('title'):
            result['Title'] = s['title']
        if s.get('description'):
            result['Description'] = s['description'].capitalize()
        extract = s.get('extract', '')
        if extract:
            result['Summary'] = extract[:500] + ('…' if len(extract) > 500 else '')
        page_url = s.get('content_urls', {}).get('desktop', {}).get('page', '')
        if page_url:
            result['_source']     = 'Wikipedia'
            result['_source_url'] = page_url
        return result                                # first matching film page

    return {}


# ── DuckDuckGo (last resort) ──────────────────────────────────────────────────

# DDG infobox label → friendly display name mapping
_DDG_FIELDS: dict[str, str] = {
    'Directed by':  'Director',
    'Written by':   'Writers',
    'Produced by':  'Producers',
    'Starring':     'Starring',
    'Release date': 'Released',
    'Running time': 'Runtime',
    'Country':      'Country',
    'Language':     'Language',
    'Budget':       'Budget',
    'Box office':   'Box office',
    'Distributor':  'Distributor',
    'Studio':       'Studio',
    'Network':      'Network',
    'Genre':        'Genre',
    'Based on':     'Based on',
    'Publisher':    'Publisher',
    'Developer':    'Developer',
}


def _ddg_search(query: str) -> dict:
    """
    Query DuckDuckGo Instant Answers API (last-resort fallback).

    Only fires when DDG has a direct knowledge-box for the title.
    Returns a flat dict; internal keys are prefixed with '_'.
    """
    params = urllib.parse.urlencode({
        'q':             f"{query} film",
        'format':        'json',
        'no_html':       '1',
        'skip_disambig': '1',
        'kp':            '-2',
        'kl':            'us-en',
    })
    opener = _make_opener()
    resp   = opener.open(f'https://api.duckduckgo.com/?{params}', timeout=15)
    data   = json.loads(resp.read().decode('utf-8', errors='replace'))

    result: dict = {}

    heading = data.get('Heading', '').strip()
    if heading:
        result['Title'] = heading

    abstract = _strip_html(data.get('AbstractText', ''))
    if abstract:
        result['Summary'] = abstract[:500] + ('…' if len(abstract) > 500 else '')

    src     = data.get('AbstractSource', '')
    src_url = data.get('AbstractURL', '')
    if src:
        result['_source']     = src
        result['_source_url'] = src_url

    infobox = data.get('Infobox', {})
    if isinstance(infobox, dict):
        for item in infobox.get('content', []):
            lbl = item.get('label', '')
            val = _strip_html(str(item.get('value', '')))
            if lbl in _DDG_FIELDS and val:
                result.setdefault(_DDG_FIELDS[lbl], val)

    return result


# ── Multi-backend orchestrator ────────────────────────────────────────────────

_YEAR_IN_PARENS = re.compile(r'\((\d{4})\)\s*$')


def _search_custom_url(candidate: str, base_url: str) -> dict:
    """
    Fetch ``base_url + url_encoded(candidate)`` and try to extract metadata.

    Accepts two response formats:
    • JSON object — maps common keys (title/Title, year/Year, plot/Plot, …)
      to the standardised result dict.
    • Any other content — stored verbatim in 'Summary' so it is displayed
      in the Online Lookup panel.

    Always sets ``_source`` / ``_source_url`` so a clickable link appears.
    """
    encoded  = urllib.parse.quote(candidate)
    full_url = base_url + encoded
    domain   = urllib.parse.urlparse(full_url).netloc or base_url

    opener = _make_opener()
    try:
        with opener.open(full_url, timeout=10) as resp:
            content_type = resp.headers.get('Content-Type', '')
            raw = resp.read(65_536).decode('utf-8', errors='replace')
    except Exception as exc:
        return {}

    result: dict = {
        '_source':     domain,
        '_source_url': full_url,
    }

    # ── try JSON ──────────────────────────────────────────────────────────────
    if 'json' in content_type or raw.lstrip().startswith(('{', '[')):
        try:
            data = json.loads(raw)
            # Unwrap common list wrappers: {"results": [...]} or [...]
            if isinstance(data, list) and data:
                data = data[0]
            elif isinstance(data, dict):
                for key in ('results', 'data', 'items', 'movies'):
                    if isinstance(data.get(key), list) and data[key]:
                        data = data[key][0]
                        break

            if isinstance(data, dict):
                _STR_MAPS = [
                    ('Title',    ('title', 'Title', 'name', 'Name', 'original_title')),
                    ('Year',     ('year', 'Year', 'release_year', 'releaseYear')),
                    ('Released', ('released', 'Released', 'release_date', 'releaseDate')),
                    ('Genre',    ('genre', 'Genre', 'genres')),
                    ('Director', ('director', 'Director')),
                    ('Starring', ('starring', 'Starring', 'cast', 'actors', 'Actors')),
                    ('Runtime',  ('runtime', 'Runtime', 'duration')),
                    ('Summary',  ('plot', 'Plot', 'overview', 'description',
                                  'synopsis', 'summary', 'Summary')),
                    ('Rating',   ('rating', 'Rating', 'imdbRating', 'vote_average')),
                ]
                for out_key, candidates in _STR_MAPS:
                    for k in candidates:
                        v = data.get(k)
                        if v:
                            result[out_key] = (
                                ', '.join(str(x) for x in v)
                                if isinstance(v, list) else str(v)
                            )
                            break
                if result.get('Title') or result.get('Summary'):
                    return result
        except (json.JSONDecodeError, Exception):
            pass

    # ── fallback: plain text / HTML — store truncated content as Summary ──────
    text = _strip_html(raw)[:1000].strip()
    if text:
        result['Summary'] = text
    return result


def _try_all_backends(candidate: str) -> dict:
    """
    Try every configured search backend for one candidate title string.

    Returns the first non-empty result, or {} if all backends miss.

    Priority:
      1. TMDB      (if TMDB_API_KEY env var is set)
      2. OMDb      (if OMDB_API_KEY env var is set)
      3. Wikipedia (always available — no key required)
      4. DDG       (last resort)
    """
    # Split "Title (YYYY)" into components for year-aware APIs
    m          = _YEAR_IN_PARENS.search(candidate)
    year       = m.group(1)              if m else ''
    title_only = candidate[:m.start()].strip() if m else candidate.strip()

    # 0. Custom URLs (user-configured — tried first, in order)
    for base_url in _custom_urls:
        try:
            r = _search_custom_url(candidate, base_url)
            if r.get('Title') or r.get('Summary'):
                return r
        except Exception:
            pass

    # 1. TMDB
    tmdb_key = os.environ.get('TMDB_API_KEY', '').strip()
    if tmdb_key:
        try:
            r = _search_tmdb(title_only, year, tmdb_key)
            if r.get('Title'):
                return r
        except Exception:
            pass

    # 2. OMDb
    omdb_key = os.environ.get('OMDB_API_KEY', '').strip()
    if omdb_key:
        try:
            r = _search_omdb(title_only, year, omdb_key)
            if r.get('Title'):
                return r
        except Exception:
            pass

    # 3. Wikipedia
    try:
        r = _search_wikipedia(candidate)
        if r.get('Title') or r.get('Summary'):
            return r
    except Exception:
        pass

    # 4. DuckDuckGo
    try:
        return _ddg_search(candidate)
    except Exception:
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# Background workers
# ══════════════════════════════════════════════════════════════════════════════

class _LocalSignals(QObject):
    done  = pyqtSignal(dict)
    error = pyqtSignal(str)


class _OnlineSignals(QObject):
    claude_status    = pyqtSignal(str)         # '' = success, else human-readable error
    claude_debug     = pyqtSignal(str, str)    # (system_prompt, raw_response_or_error)
    title_identified = pyqtSignal(str)         # emitted once per DDG candidate, before the request
    done             = pyqtSignal(dict, str)   # (results, search_query)
    error            = pyqtSignal(str)


class _LocalMetaWorker(QRunnable):
    """Gathers file metadata on a background thread."""

    def __init__(self, path: str):
        super().__init__()
        self.setAutoDelete(False)   # Python GC manages lifetime; see MetadataDialog
        self._path   = path
        self.signals = _LocalSignals()

    def run(self):
        try:
            path = self._path
            stat = os.stat(path)
            ext  = os.path.splitext(path)[1].upper().lstrip('.')

            meta: dict = {
                'File name': os.path.basename(path),
                'Location':  os.path.dirname(path),
                'File size': _fmt_size(stat.st_size),
                'Format':    ext or '—',
            }

            # ffprobe first; cv2 fallback
            video_meta = _try_ffprobe(path) or _cv2_metadata(path, stat.st_size)
            meta.update(video_meta)

            self.signals.done.emit(meta)
        except Exception as exc:
            self.signals.error.emit(str(exc))


class _OnlineSearchWorker(QRunnable):
    """Performs anonymous online lookup on a background thread."""

    def __init__(self, path: str, override_candidates: 'list[str] | None' = None):
        super().__init__()
        self.setAutoDelete(False)   # Python GC manages lifetime; see MetadataDialog
        self._path               = path
        self._override_candidates = override_candidates
        self.signals             = _OnlineSignals()

    def run(self):
        try:
            if self._override_candidates is not None:
                # User manually specified the search query — skip Claude / regex.
                ordered = [c.strip() for c in self._override_candidates if c.strip()]
                self.signals.claude_status.emit('')
            else:
                filename = os.path.basename(self._path)

                # 1. Load all past user corrections to feed to Claude as
                #    live few-shot examples.
                learned = _cm.get_all_title_corrections() if _cm is not None else []

                # 2. Ask Claude for up to 3 ranked candidate titles.
                #    _claude_infer_titles returns (titles, error, prompt, response).
                claude_titles, claude_err, claude_prompt, claude_raw = _claude_infer_titles(
                    filename, learned_examples=learned or None,
                )
                self.signals.claude_status.emit(claude_err)              # '' on success
                self.signals.claude_debug.emit(claude_prompt, claude_raw)

                # 2. Regex-based fallback generates up to 3 variants itself.
                regex_candidates = _filename_to_queries(self._path)

                # Merge: Claude candidates first, then regex fallback.
                # Deduplicate (case-insensitive) while preserving order.
                seen: set[str] = set()
                ordered: list[str] = []
                for t in claude_titles + regex_candidates:
                    key = t.strip().lower()
                    if key and key not in seen:
                        seen.add(key)
                        ordered.append(t.strip())

                # 3. Prepend any user corrections stored for these candidates.
                if _cm is not None:
                    seen_keys = {c.lower() for c in ordered}
                    prepend: list[str] = []
                    for c in ordered:
                        correction = _cm.get_title_correction(c)
                        if correction:
                            ck = correction.strip().lower()
                            if ck and ck not in seen_keys:
                                seen_keys.add(ck)
                                prepend.append(correction.strip())
                    ordered = prepend + ordered

            if not ordered:
                self.signals.error.emit(
                    "Cannot determine search query from filename"
                )
                return

            # Try each candidate through the full backend chain.
            # Stop at the first that returns a title or summary.
            last_results: dict = {}
            last_query:   str  = ordered[0]

            for candidate in ordered:
                self.signals.title_identified.emit(candidate)
                results = _try_all_backends(candidate)

                if results.get('Title') or results.get('Summary'):
                    self.signals.done.emit(results, candidate)
                    return

                last_results = results
                last_query   = candidate

            # All candidates exhausted — emit whatever came back last.
            self.signals.done.emit(last_results, last_query)

        except Exception as exc:
            self.signals.error.emit(str(exc))


# ══════════════════════════════════════════════════════════════════════════════
# Dialog
# ══════════════════════════════════════════════════════════════════════════════

_STYLE = """
QDialog {
    background: #1e1e1e;
    color: #ccc;
}
QGroupBox {
    color: #aaa;
    font-weight: bold;
    border: 1px solid #3a3a3a;
    border-radius: 4px;
    margin-top: 6px;
    padding-top: 4px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 6px;
    left: 8px;
}
QTextBrowser {
    background: #111;
    color: #ccc;
    border: none;
    selection-background-color: #3a6a9a;
}
QPushButton {
    background: #2a2a2a;
    color: #ccc;
    border: 1px solid #3a3a3a;
    padding: 4px 16px;
    border-radius: 3px;
    min-width: 64px;
}
QPushButton:hover  { background: #3a3a3a; }
QPushButton:pressed{ background: #222;    }
QScrollBar:vertical {
    background: #1e1e1e;
    width: 10px;
    border: none;
}
QScrollBar::handle:vertical {
    background: #3a3a3a;
    border-radius: 5px;
    min-height: 20px;
}
QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical { height: 0; }
"""


class MetadataDialog(QDialog):
    """
    Right-click → Properties dialog for a video file.

    Opens immediately; both sections populate asynchronously:
    • File Metadata  — local file stat + ffprobe/cv2 video info
    • Online Lookup  — DuckDuckGo Instant Answers (anonymous, no cookies)
    """

    def __init__(self, video_path: str, parent=None):
        super().__init__(parent)
        self._video_path       = video_path
        self._auto_filled_query = ''   # what was auto-filled; used to detect user edits
        self.setWindowTitle(f"Properties — {os.path.basename(video_path)}")
        self.setMinimumSize(500, 500)
        self.resize(560, 640)
        self.setStyleSheet(_STYLE)

        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(12, 12, 12, 8)

        # ── header ─────────────────────────────────────────────────────────
        hdr = QLabel(
            f"<b>{_html.escape(os.path.basename(video_path))}</b>"
        )
        hdr.setStyleSheet("font-size: 11pt; color: #eee; padding-bottom: 2px;")
        hdr.setWordWrap(True)
        layout.addWidget(hdr)

        # ── local metadata section ─────────────────────────────────────────
        local_box = QGroupBox("File Metadata")
        local_vbox = QVBoxLayout(local_box)
        local_vbox.setContentsMargins(6, 14, 6, 6)

        self._local_tb = QTextBrowser()
        self._local_tb.setOpenLinks(False)
        self._local_tb.setMinimumHeight(130)
        self._local_tb.setMaximumHeight(230)
        self._local_tb.setHtml(
            "<p style='color:#555;font-size:10pt;margin:4px'>Loading…</p>"
        )
        local_vbox.addWidget(self._local_tb)
        layout.addWidget(local_box)

        # ── online lookup section ──────────────────────────────────────────
        online_box = QGroupBox("Online Lookup")
        online_vbox = QVBoxLayout(online_box)
        online_vbox.setContentsMargins(6, 14, 6, 6)

        # Label updates: "Asking Claude…" → "Searching for: <title>" → done
        self._query_lbl = QLabel(
            "<span style='color:#555;font-size:9pt'>"
            "Asking Claude to identify title…</span>"
        )
        self._query_lbl.setWordWrap(True)
        online_vbox.addWidget(self._query_lbl)

        self._online_tb = QTextBrowser()
        self._online_tb.setOpenLinks(False)    # handle clicks manually
        self._online_tb.anchorClicked.connect(self._open_url)
        self._online_tb.setMinimumHeight(160)
        self._online_tb.setHtml(
            "<p style='color:#555;font-size:10pt;margin:4px'>Searching…</p>"
        )
        online_vbox.addWidget(self._online_tb)
        layout.addWidget(online_box)

        # ── Claude debug section (collapsed by default) ───────────────────
        self._debug_toggle = QPushButton("▶  Claude Debug")
        self._debug_toggle.setCheckable(True)
        self._debug_toggle.setChecked(False)
        self._debug_toggle.setStyleSheet(
            "QPushButton { background:transparent; color:#555; border:none;"
            " text-align:left; font-size:9pt; padding:2px 0; }"
            "QPushButton:hover  { color:#888; }"
            "QPushButton:checked{ color:#aaa; }"
        )
        layout.addWidget(self._debug_toggle)

        self._debug_widget = QWidget()
        self._debug_widget.setVisible(False)
        debug_vbox = QVBoxLayout(self._debug_widget)
        debug_vbox.setContentsMargins(0, 0, 0, 0)
        debug_vbox.setSpacing(4)

        prompt_lbl = QLabel("System prompt sent to Claude:")
        prompt_lbl.setStyleSheet("color:#666; font-size:8pt;")
        debug_vbox.addWidget(prompt_lbl)

        self._debug_prompt_tb = QTextBrowser()
        self._debug_prompt_tb.setOpenLinks(False)
        self._debug_prompt_tb.setMinimumHeight(120)
        self._debug_prompt_tb.setMaximumHeight(200)
        self._debug_prompt_tb.setStyleSheet(
            "QTextBrowser { background:#1a1a1a; color:#999; font-size:8pt;"
            " font-family: Consolas, monospace; border:1px solid #333; }"
        )
        self._debug_prompt_tb.setPlainText("(waiting for Claude…)")
        debug_vbox.addWidget(self._debug_prompt_tb)

        response_lbl = QLabel("Claude's raw response:")
        response_lbl.setStyleSheet("color:#666; font-size:8pt;")
        debug_vbox.addWidget(response_lbl)

        self._debug_response_tb = QTextBrowser()
        self._debug_response_tb.setOpenLinks(False)
        self._debug_response_tb.setMinimumHeight(50)
        self._debug_response_tb.setMaximumHeight(90)
        self._debug_response_tb.setStyleSheet(
            "QTextBrowser { background:#1a1a1a; color:#bdb; font-size:8pt;"
            " font-family: Consolas, monospace; border:1px solid #333; }"
        )
        self._debug_response_tb.setPlainText("(waiting for Claude…)")
        debug_vbox.addWidget(self._debug_response_tb)

        layout.addWidget(self._debug_widget)

        self._debug_toggle.toggled.connect(self._on_debug_toggle)

        # ── correct-title row ──────────────────────────────────────────────
        correct_row = QWidget()
        correct_row.setStyleSheet("background: transparent;")
        correct_hl = QHBoxLayout(correct_row)
        correct_hl.setContentsMargins(0, 0, 0, 0)
        correct_hl.setSpacing(6)

        correct_lbl = QLabel("Correct title:")
        correct_lbl.setStyleSheet("color: #777; font-size: 9pt;")
        correct_hl.addWidget(correct_lbl)

        self._correct_edit = QLineEdit()
        self._correct_edit.setPlaceholderText("Enter the correct movie / show title…")
        self._correct_edit.setStyleSheet(
            "QLineEdit { background:#2a2a2a; color:#ddd; border:1px solid #444;"
            " border-radius:3px; padding:2px 5px; font-size:9pt; }"
            "QLineEdit:focus { border-color:#5a7fc0; }"
        )
        self._correct_edit.returnPressed.connect(self._on_retrigger_search)
        correct_hl.addWidget(self._correct_edit, 1)

        self._correct_btn = QPushButton("Search →")
        self._correct_btn.setFixedWidth(72)
        self._correct_btn.setStyleSheet(
            "QPushButton { background:#2e4a7a; color:#ddd; border:none;"
            " border-radius:3px; padding:3px 8px; font-size:9pt; }"
            "QPushButton:hover  { background:#3a5a8a; }"
            "QPushButton:pressed{ background:#1e3a6a; }"
        )
        self._correct_btn.clicked.connect(self._on_retrigger_search)
        correct_hl.addWidget(self._correct_btn)

        layout.addWidget(correct_row)
        layout.addStretch(1)

        # ── close button ───────────────────────────────────────────────────
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        btns.rejected.connect(self.accept)
        layout.addWidget(btns)

        # ── start workers in a private pool (don't compete with thumbnail gen) ──
        # IMPORTANT: keep strong Python references (_lw / _ow) to both workers
        # so their .signals QObjects are not garbage-collected while threads run.
        # setAutoDelete(False) on both workers; we release the refs in the
        # done/error handlers once the signal has been delivered.
        self._tried_titles: list[str] = []   # every DDG query string, in order tried
        self._pool = QThreadPool()
        self._pool.setMaxThreadCount(2)

        self._lw = _LocalMetaWorker(video_path)
        self._lw.signals.done.connect(self._on_local_done)
        self._lw.signals.error.connect(self._on_local_error)
        self._pool.start(self._lw)

        self._ow = _OnlineSearchWorker(video_path)
        self._ow.signals.claude_status.connect(self._on_claude_status)
        self._ow.signals.claude_debug.connect(self._on_claude_debug)
        self._ow.signals.title_identified.connect(self._on_title_identified)
        self._ow.signals.done.connect(self._on_online_done)
        self._ow.signals.error.connect(self._on_online_error)
        self._pool.start(self._ow)

    # ── worker result handlers ─────────────────────────────────────────────────

    def _build_query_label_html(self, current: 'str | None' = None,
                                matched: 'str | None' = None) -> str:
        """
        Render the query-status label showing every tried candidate in order.

        current  — candidate currently being searched (shows "…" suffix, lighter)
        matched  — candidate that returned results (shows ✓, blue)
        All other entries in _tried_titles are shown as dim bullets (failed).
        """
        if not self._tried_titles:
            return ("<span style='color:#555;font-size:9pt'>"
                    "Asking Claude to identify title…</span>")

        rows = []
        for t in self._tried_titles:
            esc = _html.escape(t)
            if t == matched:
                rows.append(
                    f"<span style='color:#7ab8e8'>&#10003; <i>{esc}</i></span>"
                )
            elif t == current:
                rows.append(
                    f"<span style='color:#aaa'>&rarr; <i>{esc}</i>&#8230;</span>"
                )
            else:
                rows.append(
                    f"<span style='color:#555'>&bull; <i>{esc}</i></span>"
                )

        if matched:
            header = "Queries tried:"
        elif current:
            header = "Searching:"
        else:
            header = "Queries tried (no match):"

        body = '<br>'.join(rows)
        return (f"<span style='color:#666;font-size:9pt'>{header}"
                f"<br>{body}</span>")

    def _queries_tb_html(self, matched: 'str | None' = None) -> str:
        """
        Compact HTML strip showing every tried query, for embedding at the
        TOP of _online_tb.  Because it lives inside the text browser it is
        never overwritten — it stays visible regardless of what the results
        section below it says.

        matched — query that returned results; rendered with a ✓ in blue.
        All other tried queries are rendered as dim italics.
        If no queries were tried yet, returns an empty string.
        """
        if not self._tried_titles:
            return ''

        parts = []
        for t in self._tried_titles:
            esc = _html.escape(t)
            if t == matched:
                parts.append(
                    f"<span style='color:#7ab8e8'>&#10003;&nbsp;<i>{esc}</i></span>"
                )
            else:
                parts.append(f"<i style='color:#666'>{esc}</i>")

        label = "Searched for:" if matched else "Queries tried:"
        body  = ",&nbsp; ".join(parts)
        return (
            f"<p style='color:#555;font-size:9pt;margin:4px 4px 6px'>"
            f"{label}&nbsp; {body}</p>"
            "<hr style='border:none;border-top:1px solid #2a2a2a;"
            "margin:0 0 6px'>"
        )

    def _on_debug_toggle(self, checked: bool) -> None:
        self._debug_toggle.setText(
            ('▼' if checked else '▶') + '  Claude Debug'
        )
        self._debug_widget.setVisible(checked)
        if checked:
            self.adjustSize()

    def _on_claude_debug(self, prompt: str, response: str) -> None:
        """Populate the Claude debug section with the actual prompt and response."""
        self._debug_prompt_tb.setPlainText(prompt)
        self._debug_response_tb.setPlainText(response)

    def _on_claude_status(self, err: str):
        """
        Called once the Claude inference attempt completes.
        ``err`` is an empty string on success; a human-readable message on failure.
        Shows a subtle warning so the user knows Claude was unavailable.
        """
        if err:
            self._query_lbl.setText(
                f"<span style='color:#b8860b;font-size:9pt'>"
                f"&#9888;&nbsp;Claude unavailable: {_html.escape(err[:160])}"
                f"&nbsp;—&nbsp;using filename parser&hellip;</span>"
            )

    def _on_title_identified(self, title: str):
        """
        Called once per DDG candidate, before the network request fires.
        Appends the candidate to the tried list and refreshes the status label.
        """
        self._tried_titles.append(title)
        self._query_lbl.setText(self._build_query_label_html(current=title))

    def _on_local_done(self, meta: dict):
        self._lw = None                          # release worker ref (run() done)
        self._local_tb.setHtml(_kv_html(meta))

    def _on_local_error(self, msg: str):
        self._lw = None
        self._local_tb.setHtml(
            f"<p style='color:#c66;font-size:10pt;margin:4px'>"
            f"Error: {_html.escape(msg)}</p>"
        )

    def _on_online_done(self, meta: dict, query: str):
        self._ow = None                          # release worker ref (run() done)
        # Pre-fill the correction field with whatever was actually searched for,
        # so the user can edit it if the result is wrong.
        if not self._correct_edit.text():
            self._correct_edit.setText(query)
            self._auto_filled_query = query   # remember so we can detect user edits
        has_results = bool(meta)
        self._query_lbl.setText(
            self._build_query_label_html(matched=query if has_results else None)
        )

        # _queries_tb_html() is always prepended so the search strings are
        # permanently visible inside the text browser, regardless of whether
        # results were found or the content below them changes.
        queries_strip = self._queries_tb_html(matched=query if has_results else None)

        visible = {k: v for k, v in meta.items() if not k.startswith('_')}
        if not visible:
            self._online_tb.setHtml(
                queries_strip +
                "<p style='color:#666;font-size:10pt;margin:4px'>"
                "No results found. The filename may not match a known title.<br>"
                "<span style='color:#555;font-size:9pt'>"
                "Tip: rename the file to include the movie/show title.</span></p>"
            )
            return

        html_body = _kv_html(visible)

        # Append a sourced link if available
        source     = meta.get('_source', '')
        source_url = meta.get('_source_url', '')
        if source:
            if source_url:
                link = (
                    f'<a href="{_html.escape(source_url)}" '
                    f'style="color:#5a9fd4;text-decoration:none">'
                    f'{_html.escape(source)}</a>'
                )
            else:
                link = _html.escape(source)
            html_body += (
                f"<p style='color:#555;font-size:9pt;margin-top:8px'>"
                f"Source: {link}</p>"
            )

        self._online_tb.setHtml(queries_strip + html_body)

    def _on_online_error(self, msg: str):
        self._ow = None
        self._query_lbl.setText(self._build_query_label_html())  # all failed
        self._online_tb.setHtml(
            self._queries_tb_html() +                             # keep queries visible
            f"<p style='color:#666;font-size:10pt;margin:4px'>"
            f"Could not retrieve online information: {_html.escape(msg)}</p>"
        )

    def _on_retrigger_search(self):
        """User edited the correction field and pressed Search → or Enter."""
        new_title = self._correct_edit.text().strip()
        if not new_title:
            return

        # Save the correction so future lookups benefit in two ways:
        #   1. Exact-match fast path: inferred_query → correct_title
        #      (fires immediately for files that produce the same candidate)
        #   2. Claude few-shot generalisation: original_filename → correct_title
        #      (feeds Claude on every future lookup so it learns the user's
        #       specific abbreviations, language tags, and naming style)
        if _cm is not None and self._tried_titles:
            orig_filename = os.path.basename(self._video_path)
            first = self._tried_titles[0]
            _cm.save_title_correction(first, new_title,
                                      original_filename=orig_filename)
            # Also store without the trailing (YYYY) so year variants match too.
            title_only = _YEAR_IN_PARENS.sub('', first).strip()
            if title_only and title_only.lower() != first.strip().lower():
                _cm.save_title_correction(title_only, new_title)

        # Disconnect the still-running (or already-finished) old worker so
        # its late signals don't overwrite the new search results.
        if self._ow is not None:
            try:
                self._ow.signals.claude_status.disconnect()
                self._ow.signals.claude_debug.disconnect()
                self._ow.signals.title_identified.disconnect()
                self._ow.signals.done.disconnect()
                self._ow.signals.error.disconnect()
            except Exception:
                pass
            self._ow = None

        # Reset UI state.
        self._tried_titles.clear()
        self._query_lbl.setText(
            "<span style='color:#555;font-size:9pt'>Searching…</span>"
        )
        self._online_tb.setHtml(
            "<p style='color:#555;font-size:10pt;margin:4px'>Searching…</p>"
        )
        self._correct_btn.setEnabled(False)
        self._correct_edit.setEnabled(False)

        # Start a fresh search with the user-supplied title.
        self._ow = _OnlineSearchWorker(
            self._video_path,
            override_candidates=[new_title],
        )
        self._ow.signals.claude_status.connect(self._on_claude_status)
        self._ow.signals.claude_debug.connect(self._on_claude_debug)
        self._ow.signals.title_identified.connect(self._on_title_identified)
        self._ow.signals.done.connect(self._on_retrigger_done)
        self._ow.signals.error.connect(self._on_retrigger_error)
        self._pool.start(self._ow)

    def _on_retrigger_done(self, meta: dict, query: str):
        """Like _on_online_done but re-enables the correction controls."""
        self._correct_btn.setEnabled(True)
        self._correct_edit.setEnabled(True)
        self._on_online_done(meta, query)
        # Allow the user to further refine if the result is still wrong.
        self._correct_edit.setText(query)

    def _on_retrigger_error(self, msg: str):
        """Like _on_online_error but re-enables the correction controls."""
        self._correct_btn.setEnabled(True)
        self._correct_edit.setEnabled(True)
        self._on_online_error(msg)

    def _save_pending_correction(self) -> None:
        """Save the correction field if the user edited it but never hit Search →.

        This ensures the correction is persisted even when the online lookup
        returned no results (or wrong results) and the user just typed the
        correct title and closed the dialog.
        """
        if _cm is None or not self._tried_titles:
            return
        typed = self._correct_edit.text().strip()
        if not typed:
            return
        # Only save if the user actually changed the auto-filled value.
        if typed.lower() == self._auto_filled_query.strip().lower():
            return
        orig_filename = os.path.basename(self._video_path)
        first = self._tried_titles[0]
        _cm.save_title_correction(first, typed, original_filename=orig_filename)
        title_only = _YEAR_IN_PARENS.sub('', first).strip()
        if title_only and title_only.lower() != first.strip().lower():
            _cm.save_title_correction(title_only, typed)

    def closeEvent(self, event):
        """
        Persist any unsaved correction, then cancel/disconnect running workers
        so they don't signal back to this half-destroyed dialog and cannot
        prevent garbage collection via circular signal references.
        """
        self._save_pending_correction()

        # ── local metadata worker ─────────────────────────────────────────────
        if self._lw is not None:
            try:
                self._lw.signals.done.disconnect()
                self._lw.signals.error.disconnect()
            except RuntimeError:
                pass
            self._lw = None

        # ── online search worker ──────────────────────────────────────────────
        if self._ow is not None:
            try:
                self._ow.signals.claude_status.disconnect()
                self._ow.signals.claude_debug.disconnect()
                self._ow.signals.title_identified.disconnect()
                self._ow.signals.done.disconnect()
                self._ow.signals.error.disconnect()
            except RuntimeError:
                pass
            self._ow = None

        # Remove any queued-but-not-yet-started workers from the pool.
        # Workers already running will finish on their own; their signals are
        # now disconnected so they won't touch the dialog.
        self._pool.clear()

        super().closeEvent(event)

    def _open_url(self, url: QUrl):
        """Open a clicked hyperlink in the system browser."""
        if url.isValid():
            QDesktopServices.openUrl(url)
