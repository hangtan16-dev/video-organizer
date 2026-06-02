"""
Regression tests for the two-row top toolbar.

Row 1 (title "Main"): file operations → folder operations → layout format →
tools & settings, all as text buttons sharing one MINIMUM width (so the narrow
ones grow to line up; long labels still stretch — a hard same-width would
force every button to the widest label and overflow the window).

Row 2 (title "Filter & Sort"): filter-files box (expands) · rating filter ·
sort dropdown + asc/desc.

Dropped per the user: the "▶ Player" button and the breadcrumb 📌 pin.

Equal-MIN-width invariants (row 1):
  - Every text button shares the same minimum width.
  - Narrow buttons grow to exactly that floor; it's a floor, not a fixed size.
  - The growth it adds is small/bounded (won't blow past the window).
  - Uniform toolbar item spacing.
"""
import os
import sys

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('VIDEO_ORGANIZER_QSETTINGS_ORG', 'VOrgTest')
os.environ.setdefault('VIDEO_ORGANIZER_QSETTINGS_APP', 'ToolbarLayoutTest')

import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


@pytest.fixture(scope='module')
def qapp():
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication(sys.argv)


@pytest.fixture
def window(qapp):
    from main_window import MainWindow
    w = MainWindow()
    yield w
    try:
        w.close()
    except Exception:
        pass
    qapp.processEvents()


def _bars(w):
    from PyQt6.QtWidgets import QToolBar
    return {tb.windowTitle(): tb for tb in w.findChildren(QToolBar)}


def _action_buttons(tb):
    """(label, QToolButton) for every text button in a toolbar."""
    from PyQt6.QtWidgets import QToolButton
    out = []
    for act in tb.actions():
        if act.isSeparator():
            continue
        bw = tb.widgetForAction(act)
        if isinstance(bw, QToolButton):
            out.append((act.text() or bw.text(), bw))
    return out


def _ordered_labels(tb):
    """Ordered non-empty action labels (skips separators + bare widgets)."""
    return [act.text() for act in tb.actions() if act.text()]


def _widgets(tb):
    """Every widget hosted in a toolbar (for addWidget items like the filter)."""
    out = []
    for act in tb.actions():
        if act.isSeparator():
            continue
        bw = tb.widgetForAction(act)
        if bw is not None:
            out.append(bw)
    return out


# ── structure ────────────────────────────────────────────────────────────────
def test_two_toolbar_rows_exist(window):
    bars = _bars(window)
    assert "Main" in bars, "row-1 action toolbar missing"
    assert "Filter & Sort" in bars, "row-2 filter/sort toolbar missing"


def test_row1_groups_in_order(window):
    """File ops → folder ops → tools/settings, left to right."""
    labels = _ordered_labels(_bars(window)["Main"])
    idx = {t: labels.index(t) for t in labels}
    # File operations, in the user's order (Move kept between Copy and Delete).
    assert idx["Select All"] < idx["Deselect All"] < idx["Copy Selected…"]
    assert idx["Copy Selected…"] < idx["Move Selected…"] < idx["Delete Selected"]
    assert idx["Delete Selected"] < idx["Rename…"] < idx["Undo"]
    # Folder ops come after file ops…
    assert idx["Undo"] < idx["Open Folder…"] < idx["⬆ Up"]
    # …and tools/settings come after folder ops.
    assert idx["⬆ Up"] < idx["Settings"]


def test_filtering_and_sorting_live_on_row_two(window):
    bars = _bars(window)
    row2 = set(_widgets(bars["Filter & Sort"]))
    # The grid-filter, rating-filter and sort controls are on row 2…
    assert window._filter_edit in row2
    assert window._rating_filter_combo in row2
    # sort combo sits inside the row-2 "Sort:" group widget
    assert window._sort_combo.window() is window
    assert bars["Filter & Sort"].isAncestorOf(window._sort_combo)
    # …and NOT on row 1.
    row1 = set(_widgets(bars["Main"]))
    assert window._filter_edit not in row1
    assert window._rating_filter_combo not in row1


def test_folder_path_on_row_two(window):
    """The folder-path breadcrumb now lives on row 2 (same row as the filter),
    not in the panel above the grid."""
    row2 = set(_widgets(_bars(window)["Filter & Sort"]))
    assert window._breadcrumb in row2, "folder path (breadcrumb) should be on row 2"


def test_row2_path_expands_filter_is_fixed(window):
    """The path takes the row's stretch; the filter box is a fixed width so it
    doesn't fight the path for space."""
    from PyQt6.QtWidgets import QSizePolicy
    assert window._breadcrumb.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Expanding
    assert window._filter_edit.minimumWidth() == window._filter_edit.maximumWidth(), \
        "filter box should be a fixed width"
    assert 0 < window._filter_edit.maximumWidth() <= 360


def test_columns_slider_is_snug_not_stretched(window):
    """The Columns group is width-capped so the toolbar can't stretch it — that
    stretch ballooned the inner labels and pushed the slider away from the
    'Columns:' text. A compact, fixed-width slider sits right next to the label."""
    grp = window._cols_slider.parent()
    assert grp.maximumWidth() < 16777215, "Columns group must have a maximum-width cap"
    assert grp.maximumWidth() > 50, "the cap must be a real (natural) width, not collapsed"
    assert window._cols_slider.minimumWidth() == window._cols_slider.maximumWidth(), \
        "slider should be a fixed width"
    assert window._cols_slider.maximumWidth() <= 100, "slider should be compact, not stretched"


# ── removals ─────────────────────────────────────────────────────────────────
def test_player_button_removed(window):
    bars = _bars(window)
    all_labels = _ordered_labels(bars["Main"]) + _ordered_labels(bars["Filter & Sort"])
    assert not any("Player" in t for t in all_labels), "Player button should be gone"
    assert not hasattr(window, "_act_player"), "_act_player attribute should be removed"


def test_ai_search_removed(window):
    bars = _bars(window)
    all_labels = _ordered_labels(bars["Main"]) + _ordered_labels(bars["Filter & Sort"])
    assert not any("AI Search" in t or "Clear AI" in t for t in all_labels), \
        "AI Search / Clear AI Filter buttons should be gone"
    assert not hasattr(window, "_act_ai_search"), "_act_ai_search should be removed"
    assert not hasattr(window, "_act_clear_ai"), "_act_clear_ai should be removed"
    # The grid's AI-filter machinery is gone too.
    assert not hasattr(window._grid, "set_ai_filter")
    assert not hasattr(window._grid, "_ai_filter_paths")


def test_breadcrumb_pin_removed(window):
    assert not hasattr(window._breadcrumb, "_pin_btn"), "the 📌 breadcrumb pin should be gone"


# ── equal-MIN-width (row 1) ──────────────────────────────────────────────────
def test_all_buttons_share_one_minimum_width(window):
    btns = _action_buttons(_bars(window)["Main"])
    assert len(btns) >= 6, "expected the full set of row-1 buttons"
    mins = {bw.minimumWidth() for _, bw in btns}
    assert len(mins) == 1, f"buttons must share ONE minimum width, got {mins}"
    assert mins.pop() > 0, "the shared minimum width must be a real value"


def test_narrow_buttons_grow_to_the_floor(window):
    btns = _action_buttons(_bars(window)["Main"])
    floor = btns[0][1].minimumWidth()
    narrow = [(lbl, bw) for lbl, bw in btns if bw.sizeHint().width() < floor]
    assert narrow, "expected at least one naturally-narrow button (e.g. Undo/Up)"
    for lbl, bw in narrow:
        eff = max(bw.sizeHint().width(), bw.minimumWidth())
        assert eff == floor, f"{lbl!r} did not grow to the shared floor"


def test_floor_is_a_minimum_not_a_fixed_size(window):
    btns = _action_buttons(_bars(window)["Main"])
    floor = btns[0][1].minimumWidth()
    widest = max(bw.sizeHint().width() for _, bw in btns)
    assert widest > floor, "the floor must be below the widest button (a MIN, not a fixed size)"


def test_button_growth_is_bounded(window):
    btns = _action_buttons(_bars(window)["Main"])
    nat = sum(bw.sizeHint().width() for _, bw in btns)
    eff = sum(max(bw.sizeHint().width(), bw.minimumWidth()) for _, bw in btns)
    growth = (eff - nat) / nat
    assert growth <= 0.15, f"toolbar widened by {growth:.0%} — too much, risks overflow"


def test_toolbar_spacing_is_uniform(window):
    qss = window.styleSheet()
    assert "spacing: 6px" in qss, "toolbar should declare a uniform 6px item spacing"
