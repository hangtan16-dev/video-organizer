"""Settings dialog for the Video Organizer."""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QDoubleSpinBox,
    QSpinBox, QDialogButtonBox, QGroupBox, QLabel, QPlainTextEdit
)
from PyQt6.QtCore import pyqtSignal

from app_settings import AppSettings


class SettingsDialog(QDialog):
    settings_changed = pyqtSignal()

    def __init__(self, settings: AppSettings, parent=None):
        super().__init__(parent)
        self._settings = settings
        self.setWindowTitle("Settings")
        self.setMinimumWidth(320)

        layout = QVBoxLayout(self)

        # --- Thumbnail group ---
        group = QGroupBox("Thumbnail")
        form = QFormLayout(group)

        self._seek_spin = QDoubleSpinBox()
        self._seek_spin.setRange(0.0, 99999.0)
        self._seek_spin.setDecimals(2)
        self._seek_spin.setSingleStep(1.0)
        self._seek_spin.setSuffix(" s")
        self._seek_spin.setValue(settings.seek_time)
        form.addRow("Default seek time:", self._seek_spin)

        self._cols_spin = QSpinBox()
        self._cols_spin.setRange(1, 20)
        self._cols_spin.setValue(settings.thumbnails_per_row)
        form.addRow("Thumbnails per row:", self._cols_spin)

        layout.addWidget(group)

        # --- Online lookup group ---
        lookup_group = QGroupBox("Online Lookup")
        lookup_form = QFormLayout(lookup_group)

        self._url_edit = QPlainTextEdit()
        self._url_edit.setPlaceholderText(
            "https://www.imdb.com/find?q=\n"
            "https://letterboxd.com/search/\n"
            "http://myserver/api/movies?title="
        )
        self._url_edit.setPlainText('\n'.join(settings.custom_search_urls))
        self._url_edit.setFixedHeight(90)
        lookup_form.addRow("Custom search URLs:", self._url_edit)

        url_hint = QLabel(
            "One URL per line. The movie name is appended to each.\n"
            "These are tried first, then the built-in backends (TMDB, OMDb, Wikipedia, DDG)."
        )
        url_hint.setWordWrap(True)
        url_hint.setStyleSheet("color: #888; font-size: 10px;")
        lookup_form.addRow(url_hint)

        layout.addWidget(lookup_group)

        # --- Info ---
        info = QLabel(
            "Thumbnails are cached at 1080p height.\n"
            "Changes apply to new folders opened after saving."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(info)

        # --- Buttons ---
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self._save)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _save(self):
        self._settings.seek_time = self._seek_spin.value()
        self._settings.thumbnails_per_row = self._cols_spin.value()
        self._settings.custom_search_urls = self._url_edit.toPlainText().splitlines()
        self.settings_changed.emit()
        self.accept()
