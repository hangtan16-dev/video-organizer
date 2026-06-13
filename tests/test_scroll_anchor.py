"""
Regression test for the scroll glitch: "scrolling down quickly, the screen
suddenly jumps back up on its own."

Root cause: each item's row height depends on its aspect ratio, which starts at
a 16:9 placeholder and is overwritten with the REAL ratio when the thumbnail
finishes generating (_on_thumbnail_ready). That fires a full relayout. When the
item whose ratio just changed is ABOVE the viewport, re-flowing shifts every
row below it — so the content under the user's view jumps (upward for VR
thumbnails, which are taller than 16:9).

Fix: _full_relayout captures a scroll ANCHOR (the item under the viewport top)
before re-flowing and re-pins it afterwards, so relayouts never move the content
the user is looking at.
"""
import os
import sys

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('VIDEO_ORGANIZER_QSETTINGS_ORG', 'VOrgTest')
os.environ.setdefault('VIDEO_ORGANIZER_QSETTINGS_APP', 'ScrollAnchorTest')

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
    gen = MagicMock()
    grid = tgw.ThumbnailGridWidget(gen, settings, cache)
    return grid, tgw


def _build(grid, tgw, qapp, n=60):
    """Add n folder items (lightweight widgets), size + lay out the grid."""
    grid._update_active_widgets = lambda: None   # isolate the layout math
    for i in range(n):
        it = tgw._Item(path=f"f{i:03d}", is_folder=True)   # aspect_ratio defaults 16:9
        grid._items.append(it)
        grid._path_to_idx[it.path] = i
    grid.resize(800, 600)
    grid.show()
    qapp.processEvents()
    grid._full_relayout()
    qapp.processEvents()


def test_relayout_preserves_scroll_anchor(qapp, tmp_path):
    grid, tgw = _make_grid(tmp_path)
    try:
        _build(grid, tgw, qapp)
        bar = grid.verticalScrollBar()
        assert bar.maximum() > 0, "canvas must be taller than the viewport"

        bar.setValue(bar.maximum() // 2)        # scroll into the middle
        sv0 = bar.value()
        assert sv0 > 0

        # Anchor = last item whose top is at/above the viewport top.
        anchor_idx, anchor_y = None, 0
        for idx, rect in enumerate(grid._layout_cache):
            if rect[1] <= sv0:
                anchor_idx, anchor_y = idx, rect[1]
            else:
                break
        assert anchor_idx and anchor_idx > 0
        offset0 = sv0 - anchor_y

        # Thumbnails arrive for every item ABOVE the anchor → their rows grow
        # taller (square is taller than 16:9), exactly the VR case.
        for i in range(anchor_idx):
            grid._items[i].aspect_ratio = 1.0
        grid._full_relayout()
        qapp.processEvents()

        new_y  = grid._layout_cache[anchor_idx][1]
        new_sv = grid.verticalScrollBar().value()
        assert new_y > anchor_y, "rows above should have grown (test sanity)"
        # The anchor item stays at the SAME offset under the viewport top:
        # the content did not jump. (Without the fix, sv stays sv0 while new_y
        # grows, so the offset shrinks → visible upward jump.)
        assert abs((new_sv - new_y) - offset0) <= 2, (
            f"content jumped: anchor offset is now {new_sv - new_y}, "
            f"was {offset0}"
        )
    finally:
        grid.close()
        qapp.processEvents()


def test_relayout_at_top_stays_at_top(qapp, tmp_path):
    """At the very top, growing rows must not push the view off 0."""
    grid, tgw = _make_grid(tmp_path)
    try:
        _build(grid, tgw, qapp)
        grid.verticalScrollBar().setValue(0)
        for it in grid._items:
            it.aspect_ratio = 1.0
        grid._full_relayout()
        qapp.processEvents()
        assert grid.verticalScrollBar().value() == 0
    finally:
        grid.close()
        qapp.processEvents()


def test_folder_restore_skipped_when_user_scrolled(qapp, tmp_path):
    """Revisiting a folder restores the remembered scroll position — but NOT if
    the user has already started scrolling during the load (that override was a
    second source of the 'jumps on its own' glitch)."""
    grid, tgw = _make_grid(tmp_path)
    try:
        _build(grid, tgw, qapp)
        bar = grid.verticalScrollBar()
        bar.setValue(0)
        saved = max(1, bar.maximum() // 2)
        grid._user_scrolled_since_load = True          # user took over
        grid._restore_folder_scroll(saved)
        assert bar.value() == 0, "must NOT yank the user back to the saved position"
    finally:
        grid.close()
        qapp.processEvents()


def test_folder_restore_applies_when_user_idle(qapp, tmp_path):
    """If the user hasn't touched the scroll, the remembered position IS
    restored (the feature still works)."""
    grid, tgw = _make_grid(tmp_path)
    try:
        _build(grid, tgw, qapp)
        bar = grid.verticalScrollBar()
        bar.setValue(0)
        saved = max(1, bar.maximum() // 2)
        grid._user_scrolled_since_load = False         # user idle
        grid._restore_folder_scroll(saved)
        assert bar.value() == saved, "should restore the remembered position"
    finally:
        grid.close()
        qapp.processEvents()


# ── second source of the "jumps back up" glitch: focus-driven auto-scroll ──────
# A QScrollArea auto-scrolls to keep its FOCUSED child visible. So when a card's
# checkbox gets focus (the user clicks it to select), then the user scrolls that
# card off-screen, virtualization recycles it; Qt reassigns focus to another
# card's checkbox elsewhere; the scroll area then teleports the view to that
# newly-focused widget — the screen "jumps back up". The 30s soak
# (tests/integration/scroll_jump_test.py) reproduced this as 35 jumps up to
# ~38000 px. Fix: NO card child may accept keyboard focus — clicks still toggle
# the checkbox, but nothing focusable lives inside the scroll area, so the
# ensure-focused-visible auto-scroll can never fire. The grid itself keeps focus
# for keyboard navigation (it draws its own focus indicator).

def test_video_card_children_are_not_focusable(qapp):
    """Every child of a VideoThumbnailWidget must be NoFocus (the checkbox, the
    seek slider and the stars rating in particular). A focusable child is what
    lets QScrollArea yank the view back to it on scroll."""
    from PyQt6.QtWidgets import QWidget
    from PyQt6.QtCore import Qt
    from video_thumbnail_widget import VideoThumbnailWidget

    w = VideoThumbnailWidget("fake_video.mp4", 1.0)
    try:
        # The interactive controls that DEFAULT to focusable must be NoFocus.
        for name in ("_checkbox", "_seek_slider", "_stars_widget"):
            child = getattr(w, name)
            assert child.focusPolicy() == Qt.FocusPolicy.NoFocus, (
                f"{name} is focusable ({child.focusPolicy()}) — a focused card "
                f"child makes QScrollArea auto-scroll to it (scroll jumps back)"
            )
        # Future-proof: NO descendant widget anywhere may accept focus.
        focusable = [c for c in w.findChildren(QWidget)
                     if c.focusPolicy() != Qt.FocusPolicy.NoFocus]
        assert not focusable, (
            "card descendants accept focus: "
            + ", ".join(f"{type(c).__name__}({c.focusPolicy()})" for c in focusable)
        )
    finally:
        w.deleteLater()
        qapp.processEvents()


def test_folder_card_children_are_not_focusable(qapp):
    """Same invariant for the folder card (its checkbox must be NoFocus)."""
    from PyQt6.QtWidgets import QWidget
    from PyQt6.QtCore import Qt
    from folder_thumbnail_widget import FolderThumbnailWidget

    w = FolderThumbnailWidget("C:/fake/folder", child_count=3, video_count=2)
    try:
        assert w._checkbox.focusPolicy() == Qt.FocusPolicy.NoFocus
        focusable = [c for c in w.findChildren(QWidget)
                     if c.focusPolicy() != Qt.FocusPolicy.NoFocus]
        assert not focusable, (
            "folder card descendants accept focus: "
            + ", ".join(f"{type(c).__name__}({c.focusPolicy()})" for c in focusable)
        )
    finally:
        w.deleteLater()
        qapp.processEvents()
