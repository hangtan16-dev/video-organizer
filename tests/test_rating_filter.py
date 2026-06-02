"""
Tests for the star-rating filter (ThumbnailGridWidget.set_rating_filter).

User request: "view only videos with a certain number of stars. e.g. if the
user selects 3 stars, then only videos that have been rated 3 stars show. if
the user selects 0 stars then videos that have zero stars or have not been
rated are shown."

Semantics verified here:
  - EXACT match (selecting 3 shows only 3-star videos, NOT 4/5-star).
  - 0 = unrated / zero-star.
  - None ("All") = filter off.
  - Folders are EXEMPT (stay visible so the user can navigate while filtering).
  - ANDs with the existing text filter.

Tests the predicate (_is_item_filtered) + public setter directly with a mock
generator, so they're fast and deterministic (no real files / decoding).
"""
import os
import sys

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('VIDEO_ORGANIZER_QSETTINGS_ORG', 'VOrgTest')
os.environ.setdefault('VIDEO_ORGANIZER_QSETTINGS_APP', 'RatingFilterTest')

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
    return grid, tgw


def _add(grid, tgw, name, *, rating=0, is_folder=False):
    item = tgw._Item(path=name, is_folder=is_folder, rating=rating)
    grid._items.append(item)
    grid._path_to_idx[name] = len(grid._items) - 1
    return item


def _shown(grid):
    """Set of item paths NOT filtered out (i.e. visible)."""
    return {it.path for it in grid._items if not grid._is_item_filtered(it)}


def test_default_no_rating_filter_shows_everything(qapp, tmp_path):
    grid, tgw = _make_grid(tmp_path)
    _add(grid, tgw, 'a.mp4', rating=0)
    _add(grid, tgw, 'b.mp4', rating=3)
    assert grid._rating_filter is None
    assert _shown(grid) == {'a.mp4', 'b.mp4'}


def test_exact_rating_match_only(qapp, tmp_path):
    grid, tgw = _make_grid(tmp_path)
    _add(grid, tgw, 'r0.mp4',  rating=0)
    _add(grid, tgw, 'r3a.mp4', rating=3)
    _add(grid, tgw, 'r3b.mp4', rating=3)
    _add(grid, tgw, 'r5.mp4',  rating=5)
    grid.set_rating_filter(3)
    assert _shown(grid) == {'r3a.mp4', 'r3b.mp4'}, "only exactly-3-star videos show"


def test_zero_stars_shows_unrated(qapp, tmp_path):
    grid, tgw = _make_grid(tmp_path)
    _add(grid, tgw, 'unrated.mp4', rating=0)
    _add(grid, tgw, 'rated.mp4',   rating=2)
    grid.set_rating_filter(0)
    assert _shown(grid) == {'unrated.mp4'}, "0 stars = unrated / zero-star only"


def test_rating_filter_is_exact_not_gte(qapp, tmp_path):
    """Selecting 3 must NOT show 4- or 5-star videos (exact match, not >=)."""
    grid, tgw = _make_grid(tmp_path)
    _add(grid, tgw, 'r3.mp4', rating=3)
    _add(grid, tgw, 'r4.mp4', rating=4)
    _add(grid, tgw, 'r5.mp4', rating=5)
    grid.set_rating_filter(3)
    assert _shown(grid) == {'r3.mp4'}


def test_folders_stay_visible_under_rating_filter(qapp, tmp_path):
    """Folders are exempt from the rating filter so navigation still works,
    even for a non-zero rating that no folder (rating 0) could match."""
    grid, tgw = _make_grid(tmp_path)
    _add(grid, tgw, 'sub',    rating=0, is_folder=True)
    _add(grid, tgw, 'r3.mp4', rating=3)
    _add(grid, tgw, 'r1.mp4', rating=1)
    grid.set_rating_filter(3)
    assert _shown(grid) == {'sub', 'r3.mp4'}, "folder stays; only 3-star video shows"


def test_none_clears_rating_filter(qapp, tmp_path):
    grid, tgw = _make_grid(tmp_path)
    _add(grid, tgw, 'r1.mp4', rating=1)
    _add(grid, tgw, 'r2.mp4', rating=2)
    grid.set_rating_filter(2)
    assert _shown(grid) == {'r2.mp4'}
    grid.set_rating_filter(None)
    assert _shown(grid) == {'r1.mp4', 'r2.mp4'}, "'All' (None) clears the filter"


def test_rating_filter_ands_with_text_filter(qapp, tmp_path):
    """The rating filter combines (AND) with the filename text filter."""
    grid, tgw = _make_grid(tmp_path)
    _add(grid, tgw, 'holiday_r3.mp4', rating=3)
    _add(grid, tgw, 'work_r3.mp4',    rating=3)
    _add(grid, tgw, 'holiday_r1.mp4', rating=1)
    grid.set_rating_filter(3)
    grid.set_filter('holiday')
    assert _shown(grid) == {'holiday_r3.mp4'}, "must match BOTH rating AND text"


def test_rating_filter_clamped_to_0_5(qapp, tmp_path):
    grid, tgw = _make_grid(tmp_path)
    grid.set_rating_filter(9)
    assert grid._rating_filter == 5, "out-of-range high rating clamps to 5"
    grid.set_rating_filter(-3)
    assert grid._rating_filter == 0, "negative rating clamps to 0"
