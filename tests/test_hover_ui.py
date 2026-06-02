"""
UI regression tests for the hover-preview interaction:

1. Hover preview must arm ONLY when the cursor is over the thumbnail IMAGE,
   not over the seek bar / name / rating below it.
2. The seek slider must IGNORE the mouse wheel (seek via click/drag only); the
   wheel should scroll the grid instead of nudging the slider value.
"""
import os
import sys

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('VIDEO_ORGANIZER_QSETTINGS_ORG', 'VOrgTest')
os.environ.setdefault('VIDEO_ORGANIZER_QSETTINGS_APP', 'HoverUITest')

import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


@pytest.fixture(scope='module')
def qapp():
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication(sys.argv)


@pytest.fixture
def widget(qapp, tmp_path):
    """A VideoThumbnailWidget that is fully torn down after the test so its
    armed hover/shimmer timers can't fire (→ start a preview on the fake file)
    into a later test."""
    from video_thumbnail_widget import VideoThumbnailWidget
    f = tmp_path / 'clip.mp4'; f.write_bytes(b'\x00' * 32)
    w = VideoThumbnailWidget(str(f), 0.0)
    w.resize(320, 260)
    yield w
    for timer in ('_hover_delay', '_shimmer_timer', '_shimmer_max_timer'):
        try:
            getattr(w, timer).stop()
        except Exception:
            pass
    try:
        w._hovering = False
        w._stop_playback()
    except Exception:
        pass
    w.deleteLater()
    qapp.processEvents()


def test_seek_slider_ignores_mouse_wheel(qapp):
    from PyQt6.QtCore import Qt, QPoint, QPointF
    from PyQt6.QtGui import QWheelEvent
    from video_thumbnail_widget import _NoWheelSlider

    s = _NoWheelSlider(Qt.Orientation.Horizontal)
    s.setRange(0, 100)
    s.setValue(50)

    ev = QWheelEvent(
        QPointF(5, 5), QPointF(5, 5),       # pos, globalPos
        QPoint(0, 0), QPoint(0, 120),       # pixelDelta, angleDelta (one notch up)
        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase, False,
    )
    s.wheelEvent(ev)
    assert s.value() == 50, "scroll wheel must NOT change the seek slider value"
    # And the slider declined the event so it can scroll the grid instead.
    assert not ev.isAccepted()


def test_hover_arms_only_over_the_image(widget, monkeypatch):
    from PyQt6.QtCore import QEvent
    w = widget
    assert w._hovering is False

    # Entering the IMAGE LABEL arms the hover-preview delay.
    w.eventFilter(w._image_label, QEvent(QEvent.Type.Enter))
    assert w._hovering is True
    assert w._hover_delay.isActive()

    # Leaving the image (cursor no longer over it) ends hover.
    monkeypatch.setattr(w, '_cursor_over_image', lambda: False)
    w.eventFilter(w._image_label, QEvent(QEvent.Type.Leave))
    assert w._hovering is False
    assert not w._hover_delay.isActive()


def test_hover_does_not_arm_over_seek_bar(widget):
    from PyQt6.QtCore import QEvent
    w = widget

    # Enter over the seek bar (or any non-image child) → NO hover preview.
    w.eventFilter(w._seek_slider, QEvent(QEvent.Type.Enter))
    assert w._hovering is False
    assert not w._hover_delay.isActive()

    # Entering the image afterwards DOES arm it.
    w.eventFilter(w._image_label, QEvent(QEvent.Type.Enter))
    assert w._hovering is True


def test_cursor_over_image_overlay_keeps_hover(widget, monkeypatch):
    """Moving onto an overlay (checkbox/badge) that sits ON the image fires
    Leave on the label, but the cursor is still geometrically over the
    thumbnail — hover must persist."""
    from PyQt6.QtCore import QEvent
    w = widget
    w.eventFilter(w._image_label, QEvent(QEvent.Type.Enter))
    assert w._hovering is True

    # Cursor still over the image rect (e.g. on the checkbox overlay).
    monkeypatch.setattr(w, '_cursor_over_image', lambda: True)
    w.eventFilter(w._image_label, QEvent(QEvent.Type.Leave))
    assert w._hovering is True, "hover should persist while cursor is over an overlay on the image"


def test_generating_overlay_is_mouse_transparent(widget):
    """The 'Generating…' overlay and decorative badges sit on TOP of the image
    label; if they ate mouse events the image label's Enter would never fire →
    hover preview & live-seek dead on every not-yet-generated card. They must
    be mouse-transparent. The checkbox must stay clickable."""
    from PyQt6.QtCore import Qt
    attr = Qt.WidgetAttribute.WA_TransparentForMouseEvents
    assert widget._loading_label.testAttribute(attr)
    assert widget._watched_badge.testAttribute(attr)
    assert widget._sub_badge.testAttribute(attr)
    assert widget._nfo_badge.testAttribute(attr)
    assert not widget._checkbox.testAttribute(attr), "checkbox must remain clickable"


def test_incidental_hover_does_not_regenerate_thumbnail(widget):
    """A hover with NO seek (e.g. the cursor brushing a card while the user
    scrolls) must NOT emit seek_requested on hover-end — that signal makes the
    grid pop the pixmap + regenerate ("Generating…"). Regression for the
    scroll-triggered regeneration bug: _seek_time_at_last_thumb must start equal
    to seek_time so the FIRST hover-end doesn't fire on a None mismatch."""
    w = widget
    emitted = []
    w.seek_requested.connect(lambda path, t: emitted.append((path, t)))
    w._begin_thumb_hover()
    w._end_thumb_hover()          # no seek happened during the hover
    assert emitted == [], "an incidental hover regenerated the thumbnail"


def test_hover_with_real_seek_persists_new_position(widget):
    """If the user DID seek during the hover, hover-end persists it (emits
    seek_requested → grid regenerates at the new position). The fix must not
    break this real behaviour."""
    w = widget
    emitted = []
    w.seek_requested.connect(lambda path, t: emitted.append((path, t)))
    w._begin_thumb_hover()
    w._seek_time = w._seek_time + 30.0    # user dragged the slider mid-hover
    w._end_thumb_hover()
    assert len(emitted) == 1, "a real hover-seek must persist + regenerate"
    assert emitted[0][1] == w._seek_time


def test_clicking_thumbnail_does_not_select_or_border(widget):
    """Clicking anywhere in the thumbnail must NOT check the box and must NOT
    draw a blue border around the card — selection is checkbox-only now."""
    from PyQt6.QtCore import Qt, QPointF, QEvent
    from PyQt6.QtGui import QMouseEvent
    w = widget
    assert not w._checkbox.isChecked()
    ev = QMouseEvent(QEvent.Type.MouseButtonPress, QPointF(20, 20), QPointF(20, 20),
                     Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
                     Qt.KeyboardModifier.NoModifier)
    w.mousePressEvent(ev)
    assert not w._checkbox.isChecked(), "clicking the thumbnail must NOT check the box"
    assert "#3a6fc4" not in w.styleSheet(), "clicking must NOT add a blue card border"


def test_checkbox_is_only_selector_with_blue_border(widget):
    """The checkbox carries a blue border (the only visible selector) and still
    toggles directly; checking it does NOT put a blue ring on the whole card."""
    w = widget
    cb_style = w._checkbox.styleSheet()
    assert "#5a9fd4" in cb_style and "border" in cb_style, \
        "checkbox must have a blue border to be visible"
    # Direct checkbox interaction still works.
    w._checkbox.setChecked(True)
    assert w._checkbox.isChecked()
    # …but the card itself stays borderless (no blue selection ring).
    w._update_frame_style()
    assert "#3a6fc4" not in w.styleSheet()
    assert "#5a9fd4" not in w.styleSheet()


def test_seek_bar_is_thick_for_easy_clicking(widget):
    """Ease-of-use: the seek bar groove was thickened (4→8px) and its row made
    taller so it's a bigger, easier click target."""
    import video_thumbnail_widget as vtw
    assert vtw._SLIDER_H >= 30, "seek-slider row should be taller"
    assert vtw.BOTTOM_H == vtw._SLIDER_H + vtw._NAME_H + vtw._RATING_H
    assert "height: 8px" in widget._seek_slider.styleSheet(), "groove should be ~2x thicker"


def test_folder_checkbox_only_selection(qapp, tmp_path):
    """Folders get the SAME checkbox-only selection as videos: clicking the
    folder must not check it or draw a blue card border; the checkbox is the
    sole selector and carries a blue border."""
    from PyQt6.QtCore import Qt, QPointF, QEvent
    from PyQt6.QtGui import QMouseEvent
    from folder_thumbnail_widget import FolderThumbnailWidget
    d = tmp_path / "sub"; d.mkdir()
    fw = FolderThumbnailWidget(str(d), 3, 5, None)
    try:
        assert not fw.is_checked()
        ev = QMouseEvent(QEvent.Type.MouseButtonPress, QPointF(20, 20), QPointF(20, 20),
                         Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
                         Qt.KeyboardModifier.NoModifier)
        fw.mousePressEvent(ev)
        assert not fw.is_checked(), "clicking a folder must NOT check it"
        assert "#3a6fc4" not in fw.styleSheet(), "no blue card border on click"
        cb = fw._checkbox.styleSheet()
        assert "#5a9fd4" in cb and "border" in cb, "folder checkbox needs a blue border"
        # Direct checkbox interaction still selects; card stays borderless.
        fw.set_checked(True)
        assert fw.is_checked()
        fw._update_frame_style()
        assert "#3a6fc4" not in fw.styleSheet()
    finally:
        fw.deleteLater()
        qapp.processEvents()
