"""
Prepare a local test folder with a representative subset of video files copied
from your media library, for the full-functionality test rig.

Set these env vars before running:
    VORG_SOURCE_LIBRARY  — folder to copy source clips from
    VORG_TEST_ROOT       — destination folder for the prepared test files

We need a variety of file sizes and codecs to exercise every code path:

  tier        size      strategy            tests
  ──────────  ────────  ───────────────────  ─────────────────────────────
  tiny        ~50 MB    ffmpeg stream copy   hover at native FPS, fast thumb
  small       360 MB    direct copy (Sample) hover at native FPS
  medium      1-2 GB    ffmpeg stream copy   hover at reduced FPS (~8)
  large       5+ GB     direct copy          hover DISABLED (over 4 GB)
  huge        ~12 GB    ffmpeg stream copy   stress test, seek latency

Each file is HEVC / HDR / 10-bit so we exercise the most demanding decode
path the app encounters in real use.

Usage:
    python tests/integration/prepare_test_data.py

It's idempotent: skips files that already exist with the expected size.
"""
import os
import subprocess
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

SOURCE_ROOT = Path(os.environ.get('VORG_SOURCE_LIBRARY', ''))
TEST_ROOT   = Path(os.environ.get('VORG_TEST_ROOT', ''))

# ffmpeg path discovery (relies on ffmpeg being on PATH)
import shutil as _sh
FFMPEG = _sh.which('ffmpeg') or 'ffmpeg'


@dataclass
class TestFile:
    """One file to prepare in the local test folder."""
    tier:       str        # tiny / small / medium / large / huge
    dest_name:  str
    source:     Path
    strategy:   str        # 'copy' or 'trim'
    trim_start: str = ''   # ffmpeg -ss value (for trim)
    trim_dur:   str = ''   # ffmpeg -t value  (for trim)

    @property
    def dest_path(self) -> Path:
        return TEST_ROOT / self.tier / self.dest_name


def _find_source_files() -> 'dict[str, Path]':
    """Find specific source files we'll use for trimming."""
    sources: dict[str, Path] = {}
    # Direct files (no subdirectory)
    for name, key in [
        ('Alien.Romulus.2024.2160p.WEB-DL.DDP5.1.Atmos.DV.HDR.H.265-FLUX.mkv', 'alien_romulus'),
        ('Aliens.1986.Special.Edition.UHD.BluRay.2160p.TrueHD.Atmos.7.1.DV.HEVC.REMUX-FraMeSToR.mkv', 'aliens_1986'),
    ]:
        p = SOURCE_ROOT / name
        if p.is_file():
            sources[key] = p

    # Files inside subdirectories
    for subdir_name, subfile, key in [
        ('Nosferatu.2024.Hybrid.2160p.WEB-DL.DV.HDR.DDP5.1.Atmos.H265-AOC',
         'Sample.mkv', 'nosferatu_sample'),
        ('The.Gorge.2025.2160p.Hybrid.MULTI.WEB-DL.DV.HDR.H265-AOC',
         'Sample.mkv', 'gorge_sample'),
        ('Jurassic.World.Rebirth.2025.2160p.HDR10Plus.DV.WEBRip.6CH.x265.HEVC-PSA',
         'Jurassic.World.Rebirth.2025.2160p.HDR10Plus.DV.WEBRip.6CH.x265.HEVC-PSA.mkv',
         'jurassic_world'),
        ('Arrival.2016.2160p.BluRay.x265.10bit.HDR.DTS-HD.MA.7.1-TERMiNAL',
         'Arrival.2016.2160p.UHD.BluRay.x265-TERMiNAL.mkv',
         'arrival'),
    ]:
        p = SOURCE_ROOT / subdir_name / subfile
        if p.is_file():
            sources[key] = p

    return sources


def _build_plan(sources: 'dict[str, Path]') -> 'list[TestFile]':
    """Decide what test files to create based on available sources."""
    plan: list[TestFile] = []

    # ── tiny: ~50 MB trimmed snippet (hover at native FPS, fast thumb gen) ─
    if 'nosferatu_sample' in sources:
        plan.append(TestFile(
            tier='tiny',
            dest_name='tiny_hdr_30s.mkv',
            source=sources['nosferatu_sample'],
            strategy='trim',
            trim_start='00:00:10',
            trim_dur='00:00:30',     # 30-second slice
        ))

    # ── small: 360 MB Sample.mkv direct copy ───────────────────────────────
    if 'gorge_sample' in sources:
        plan.append(TestFile(
            tier='small',
            dest_name='small_hdr_sample.mkv',
            source=sources['gorge_sample'],
            strategy='copy',
        ))

    # ── medium: ~1-2 GB trim (hover at reduced FPS) ─────────────────────────
    if 'arrival' in sources:
        plan.append(TestFile(
            tier='medium',
            dest_name='medium_hdr_5min.mkv',
            source=sources['arrival'],
            strategy='trim',
            trim_start='00:20:00',
            trim_dur='00:05:00',     # 5-minute slice
        ))

    # ── large: ~5 GB direct copy (hover DISABLED — over 4 GB threshold) ─────
    if 'jurassic_world' in sources:
        plan.append(TestFile(
            tier='large',
            dest_name='large_hdr_jurassic.mkv',
            source=sources['jurassic_world'],
            strategy='copy',
        ))

    # ── huge: ~10 GB trim from a 50+ GB remux (seek-latency stress) ─────────
    if 'aliens_1986' in sources:
        plan.append(TestFile(
            tier='huge',
            dest_name='huge_hdr_remux_15min.mkv',
            source=sources['aliens_1986'],
            strategy='trim',
            trim_start='00:30:00',
            trim_dur='00:15:00',     # 15-minute slice of REMUX
        ))

    return plan


def _ensure_dest_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


def _file_ok(p: Path, min_bytes: int = 100_000) -> bool:
    """A previously-prepared file is OK if it exists with non-trivial size."""
    try:
        return p.is_file() and p.stat().st_size >= min_bytes
    except OSError:
        return False


def _do_copy(f: TestFile) -> bool:
    print(f"  [copy] {f.source.name} -> {f.dest_path}")
    t0 = time.perf_counter()
    try:
        shutil.copy2(f.source, f.dest_path)
    except Exception as e:
        print(f"  FAILED: {e}")
        return False
    dt = time.perf_counter() - t0
    sz = f.dest_path.stat().st_size / (1024**3)
    print(f"  OK ({sz:.2f} GB in {dt:.1f}s, {sz*1024/dt:.0f} MB/s)")
    return True


def _do_trim(f: TestFile) -> bool:
    """Stream-copy a time range with ffmpeg — preserves original codec
    (HEVC / HDR metadata / 10-bit) without re-encoding."""
    print(f"  [trim {f.trim_start} +{f.trim_dur}] {f.source.name} -> {f.dest_path}")
    cmd = [
        FFMPEG,
        '-hide_banner', '-loglevel', 'error',
        '-y',                            # overwrite if exists
        '-ss', f.trim_start,             # before -i: fast seek to keyframe
        '-i', str(f.source),
        '-t',  f.trim_dur,
        '-c',  'copy',                   # no re-encode
        '-map', '0',                     # all streams (video, audio, subs)
        '-avoid_negative_ts', 'make_zero',
        str(f.dest_path),
    ]
    t0 = time.perf_counter()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        print(f"  FAILED: ffmpeg timeout")
        return False
    dt = time.perf_counter() - t0
    if r.returncode != 0:
        print(f"  FAILED (exit {r.returncode}): {r.stderr[:500]}")
        # Clean up partial output
        try:
            f.dest_path.unlink(missing_ok=True)
        except Exception:
            pass
        return False
    if not f.dest_path.is_file():
        print(f"  FAILED: ffmpeg returned 0 but no output file")
        return False
    sz = f.dest_path.stat().st_size / (1024**3)
    print(f"  OK ({sz:.2f} GB in {dt:.1f}s)")
    return True


def main() -> int:
    print("=" * 70)
    print(f"Preparing test data in {TEST_ROOT}")
    print("=" * 70)

    if not SOURCE_ROOT.is_dir():
        print(f"ERROR: Source folder not found: {SOURCE_ROOT}")
        return 1
    if not os.path.exists(FFMPEG):
        print(f"ERROR: ffmpeg not found at {FFMPEG}")
        return 1

    TEST_ROOT.mkdir(parents=True, exist_ok=True)

    sources = _find_source_files()
    print(f"\nFound {len(sources)} source files:")
    for k, v in sources.items():
        try:
            sz_gb = v.stat().st_size / (1024**3)
            print(f"  {k:25s} {sz_gb:6.2f} GB  {v.name}")
        except OSError:
            print(f"  {k:25s}    ?    {v.name}")

    plan = _build_plan(sources)
    print(f"\nPlan: {len(plan)} test files to prepare")
    for f in plan:
        print(f"  [{f.tier:6s}] {f.strategy:5s}  -> {f.dest_name}")
    print()

    n_ok, n_skip, n_fail = 0, 0, 0
    for f in plan:
        _ensure_dest_dir(f.dest_path)
        if _file_ok(f.dest_path):
            sz_gb = f.dest_path.stat().st_size / (1024**3)
            print(f"[{f.tier:6s}] already exists ({sz_gb:.2f} GB) — skipping")
            n_skip += 1
            continue

        print(f"[{f.tier:6s}] preparing {f.dest_name}")
        ok = _do_copy(f) if f.strategy == 'copy' else _do_trim(f)
        if ok:
            n_ok += 1
        else:
            n_fail += 1

    print()
    print("=" * 70)
    print(f"Summary: {n_ok} prepared, {n_skip} already present, {n_fail} failed")
    print()
    print(f"Test files in {TEST_ROOT}:")
    total_bytes = 0
    for tier in ('tiny', 'small', 'medium', 'large', 'huge'):
        d = TEST_ROOT / tier
        if not d.is_dir():
            continue
        for f in sorted(d.iterdir()):
            sz = f.stat().st_size
            total_bytes += sz
            print(f"  {tier:6s}/  {sz/(1024**3):6.2f} GB  {f.name}")
    print(f"  TOTAL: {total_bytes/(1024**3):.2f} GB")
    print("=" * 70)
    return 0 if n_fail == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
