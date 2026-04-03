"""
Export library dialog.

Generates a report of all videos in the current view as:
- HTML file (styled, with thumbnail paths as img src)
- CSV file (plain data, Excel-compatible)
- JSON file (machine-readable)

Includes: filename, title (from metadata cache), duration, size,
resolution, rating, tags, watched status, date modified, folder path.
"""

import os
import csv
import json
import datetime

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QRadioButton,
    QPushButton, QFileDialog, QDialogButtonBox, QProgressBar,
    QGroupBox, QCheckBox, QLineEdit, QMessageBox
)
from PyQt6.QtCore import Qt


class ExportDialog(QDialog):
    """
    Export library dialog.

    Parameters
    ----------
    items : list
        List of item objects/dicts from the grid.  Each must expose:
        .path, .size, .duration, .mtime, .rating, .is_watched, .tags,
        .is_folder (bool).  Also accepts plain dicts with the same keys.
    cache : CacheManager
        Used for get_video_metadata().
    """

    def __init__(self, items: list, cache, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export Library")
        self.setMinimumWidth(420)

        # Support both object-style and dict-style items
        self._items = [i for i in items if not _get(i, 'is_folder', False)]
        self._cache = cache

        self.setStyleSheet("""
            QDialog { background: #1e1e1e; color: #ddd; }
            QLabel { color: #ddd; }
            QGroupBox { color: #ccc; border: 1px solid #3a3a3a; border-radius: 4px;
                        margin-top: 8px; padding-top: 8px; }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; color: #888; }
            QRadioButton { color: #ddd; }
            QCheckBox { color: #ddd; }
            QLineEdit { background: #2e2e2e; color: #ddd; border: 1px solid #3a3a3a;
                        border-radius: 3px; padding: 4px; }
            QPushButton { background: #2e2e2e; color: #ddd; border: 1px solid #3a3a3a;
                          border-radius: 3px; padding: 4px 12px; }
            QPushButton:hover { background: #3a3a3a; }
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # Summary
        layout.addWidget(QLabel(f"Export {len(self._items)} video(s)"))

        # ── Format selection ──────────────────────────────────────────────
        fmt_group = QGroupBox("Format")
        fmt_layout = QVBoxLayout(fmt_group)
        self._rb_html = QRadioButton("HTML (styled report)")
        self._rb_csv  = QRadioButton("CSV (spreadsheet)")
        self._rb_json = QRadioButton("JSON (data file)")
        self._rb_html.setChecked(True)
        fmt_layout.addWidget(self._rb_html)
        fmt_layout.addWidget(self._rb_csv)
        fmt_layout.addWidget(self._rb_json)
        layout.addWidget(fmt_group)

        # ── Fields selection ──────────────────────────────────────────────
        fields_group = QGroupBox("Include fields")
        fields_layout = QVBoxLayout(fields_group)
        self._cb_metadata  = QCheckBox("Online metadata (title, director, genre)")
        self._cb_tags      = QCheckBox("Tags")
        self._cb_watched   = QCheckBox("Watched status")
        self._cb_rating    = QCheckBox("Star rating")
        self._cb_technical = QCheckBox("Technical info (resolution, codec)")
        for cb in [self._cb_metadata, self._cb_tags, self._cb_watched,
                   self._cb_rating, self._cb_technical]:
            cb.setChecked(True)
            fields_layout.addWidget(cb)
        layout.addWidget(fields_group)

        # ── Buttons ───────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        export_btn = QPushButton("Export…")
        export_btn.setStyleSheet(
            "QPushButton { background: #3a6fc4; color: white; border: none; "
            "border-radius: 3px; padding: 6px 16px; }"
            "QPushButton:hover { background: #4a7fd4; }"
        )
        export_btn.clicked.connect(self._do_export)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(export_btn)
        btn_row.addWidget(cancel_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

    # ── Export orchestration ──────────────────────────────────────────────

    def _do_export(self):
        if self._rb_html.isChecked():
            ext, fmt = "HTML Files (*.html)", "html"
        elif self._rb_csv.isChecked():
            ext, fmt = "CSV Files (*.csv)", "csv"
        else:
            ext, fmt = "JSON Files (*.json)", "json"

        path, _ = QFileDialog.getSaveFileName(self, "Export to…", "", ext)
        if not path:
            return

        rows = self._build_rows()
        try:
            if fmt == 'html':
                self._write_html(path, rows)
            elif fmt == 'csv':
                self._write_csv(path, rows)
            else:
                self._write_json(path, rows)
            self.accept()
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))

    # ── Row building ──────────────────────────────────────────────────────

    def _build_rows(self):
        rows = []
        for item in self._items:
            mtime = _get(item, 'mtime', None)
            row = {
                'Filename': os.path.basename(_get(item, 'path', '')),
                'Folder':   os.path.dirname(_get(item, 'path', '')),
                'Size':     self._human_size(_get(item, 'size', 0)),
                'Duration': self._fmt_dur(_get(item, 'duration', 0)),
                'Modified': (
                    datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M')
                    if mtime else ''
                ),
            }
            if self._cb_rating.isChecked():
                rating = _get(item, 'rating', 0) or 0
                row['Rating'] = '★' * rating + '☆' * (5 - rating)
            if self._cb_watched.isChecked():
                row['Watched'] = 'Yes' if _get(item, 'is_watched', False) else 'No'
            if self._cb_tags.isChecked():
                tags = _get(item, 'tags', []) or []
                row['Tags'] = ', '.join(tags)
            if self._cb_metadata.isChecked():
                meta = {}
                if hasattr(self._cache, 'get_video_metadata'):
                    try:
                        meta = self._cache.get_video_metadata(_get(item, 'path', '')) or {}
                    except Exception:
                        pass
                row['Title']    = meta.get('title', '')
                row['Director'] = meta.get('director', '')
                row['Genre']    = meta.get('genre', '')
                row['Year']     = meta.get('year', '')
            if self._cb_technical.isChecked():
                row['Resolution'] = _get(item, 'resolution', '')
                row['Codec']      = _get(item, 'codec', '')
            rows.append(row)
        return rows

    # ── Writers ───────────────────────────────────────────────────────────

    def _write_html(self, path: str, rows: list):
        cols = list(rows[0].keys()) if rows else []
        now  = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
        lines = [
            '<!DOCTYPE html><html><head><meta charset="utf-8">',
            '<title>Video Library Export</title>',
            '<style>',
            'body { background:#1e1e1e; color:#ddd; font-family:sans-serif; margin:24px; }',
            'h2   { color:#fff; }',
            'p    { color:#888; }',
            'table { border-collapse:collapse; width:100%; }',
            'th { background:#252525; padding:8px 10px; text-align:left; '
            '     border:1px solid #3a3a3a; color:#ccc; }',
            'td { padding:6px 10px; border:1px solid #2a2a2a; }',
            'tr:nth-child(even) { background:#1a1a1a; }',
            'tr:hover { background:#2a2a2a; }',
            '</style></head><body>',
            f'<h2>Video Library — {len(rows)} item{"s" if len(rows) != 1 else ""}</h2>',
            f'<p>Exported {now}</p>',
            '<table>',
            '<tr>' + ''.join(f'<th>{c}</th>' for c in cols) + '</tr>',
        ]
        for row in rows:
            lines.append(
                '<tr>' + ''.join(f'<td>{_esc(row.get(c, ""))}</td>' for c in cols) + '</tr>'
            )
        lines.append('</table></body></html>')
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

    def _write_csv(self, path: str, rows: list):
        cols = list(rows[0].keys()) if rows else []
        with open(path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            writer.writerows(rows)

    def _write_json(self, path: str, rows: list):
        payload = {
            'exported': datetime.datetime.now().isoformat(),
            'count':    len(rows),
            'items':    rows,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    # ── Static helpers ────────────────────────────────────────────────────

    @staticmethod
    def _human_size(n):
        if not n:
            return ''
        for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
            if n < 1024:
                return f"{n:.1f} {unit}" if unit != 'B' else f"{n} B"
            n /= 1024
        return f"{n:.1f} PB"

    @staticmethod
    def _fmt_dur(secs):
        if not secs:
            return ''
        s = int(secs)
        h, r = divmod(s, 3600)
        m, sec = divmod(r, 60)
        return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


# ── Module-level helpers ──────────────────────────────────────────────────────

def _get(obj, key, default=None):
    """Get a value from either a dict or an object attribute."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _esc(text: str) -> str:
    """Minimal HTML escaping."""
    return (
        str(text)
        .replace('&', '&amp;')
        .replace('<', '&lt;')
        .replace('>', '&gt;')
    )
