"""Entry point for the Video Organizer application."""

import sys
import os

# Ensure the script's directory is on sys.path so all local modules resolve.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPalette, QColor

from main_window import MainWindow


def main():
    # High-DPI support
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("Video Organizer")
    app.setOrganizationName("VideoOrganizer")
    app.setApplicationVersion("1.0.0")

    # Dark palette fallback (in case stylesheet misses something)
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(30, 30, 30))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(221, 221, 221))
    palette.setColor(QPalette.ColorRole.Base, QColor(37, 37, 37))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(45, 45, 45))
    palette.setColor(QPalette.ColorRole.Text, QColor(221, 221, 221))
    palette.setColor(QPalette.ColorRole.Button, QColor(46, 46, 46))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(221, 221, 221))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(58, 111, 196))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
