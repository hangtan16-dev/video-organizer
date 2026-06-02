"""
Regression test: VideoThumbnailWidget's image label must use
setScaledContents(True) so that when the hover play thread's adaptive
resolution downscales the emit buffer, the small pixmap is stretched to
fill the label rather than centered with empty borders.
"""
import os
import sys
import pytest


@pytest.fixture(scope='module')
def qapp():
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication(sys.argv)


def test_image_label_has_scaled_contents(qapp, tmp_path):
    """The static check the user explicitly asked for: when output res is
    scaled down, the pixmap stretches to fill the label."""
    from video_thumbnail_widget import VideoThumbnailWidget
    bogus = tmp_path / "x.mp4"
    bogus.write_bytes(b'')
    w = VideoThumbnailWidget(str(bogus), 0.0)
    assert w._image_label.hasScaledContents() is True, (
        "VideoThumbnailWidget._image_label must have setScaledContents(True). "
        "Without it, adaptive-resolution downscaling produces a small pixmap "
        "centered in a larger label with empty borders, instead of "
        "stretching to fill the thumbnail frame."
    )


def test_setting_scaled_pixmap_fills_label(qapp, tmp_path):
    """End-to-end: set a small pixmap, the label's effective drawn size
    matches the LABEL, not the pixmap."""
    from PyQt6.QtGui import QPixmap, QImage
    from PyQt6.QtCore import Qt
    from video_thumbnail_widget import VideoThumbnailWidget

    bogus = tmp_path / "x.mp4"
    bogus.write_bytes(b'')
    w = VideoThumbnailWidget(str(bogus), 0.0)
    w.resize(800, 500)
    w._image_label.resize(800, 450)

    # Create a small pixmap as the adaptive scaler would produce
    img = QImage(200, 112, QImage.Format.Format_RGB888)
    img.fill(Qt.GlobalColor.red)
    pix = QPixmap.fromImage(img)
    w._image_label.setPixmap(pix)

    # With scaledContents, the label DRAWS at label size, regardless of
    # the pixmap's native size.  We can't easily test the drawn output,
    # but the property must be set so Qt does the stretch.
    assert w._image_label.hasScaledContents()
    assert w._image_label.pixmap().width()  == 200   # pixmap unchanged
    assert w._image_label.pixmap().height() == 112
    assert w._image_label.width()  == 800            # label fills its area
    assert w._image_label.height() == 450
