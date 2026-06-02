"""Pytest configuration:
- Make the project root importable as a package root.
- Redirect app_logger output into a per-session tempdir so tests don't write
  into the production log at %APPDATA%\\VideoOrganizer\\app.log.
- Redirect AppSettings to a unique QSettings org/app name so tests do NOT
  pollute the user's real VideoOrganizer settings.

Why the env-var trick: QSettings on Windows uses NativeFormat (HKCU
registry) by default, and `monkeypatch.setenv('APPDATA', tmp)` does NOT
redirect the registry.  setDefaultFormat(IniFormat) also doesn't apply
to QSettings(org, app) constructions — that overload is hardcoded to
NativeFormat.  Earlier tests wrote directly to HKCU\\Software\\VideoOrganizer
and broke the user's hover preview (set large_file_threshold_mb=0).

The fix: AppSettings now reads its org/app from env vars
(VIDEO_ORGANIZER_QSETTINGS_{ORG,APP}).  Conftest sets these to unique
test values BEFORE any test imports the module, so the registry path
for tests is HKCU\\Software\\VideoOrganizerTest\\<uuid> — completely
separate from the real HKCU\\Software\\VideoOrganizer\\VideoOrganizer.
"""
import logging
import os
import sys
import tempfile
import uuid

import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ── QSettings isolation: must happen BEFORE any test imports app_settings ───
# We use a unique per-session UUID so re-running tests doesn't see stale
# values from a prior pytest invocation either.
os.environ['VIDEO_ORGANIZER_QSETTINGS_ORG'] = 'VideoOrganizerTest'
os.environ['VIDEO_ORGANIZER_QSETTINGS_APP'] = (
    f'TestSession_{uuid.uuid4().hex[:12]}'
)


@pytest.fixture(autouse=True)
def _isolate_app_logger(tmp_path, monkeypatch):
    """Auto-applied to every test. Redirects app_logger writes to tmp_path
    instead of the user's real %APPDATA%\\VideoOrganizer\\app.log.

    Strategy: redirect APPDATA + the cached module paths, then strip any
    existing root-logger handlers (they point at a previous test's tmp_path
    which is about to be deleted). The next get_logger() call inside the
    test will create fresh handlers pointing at this tmp_path.
    """
    monkeypatch.setenv("APPDATA", str(tmp_path))

    import app_logger
    new_dir  = os.path.join(str(tmp_path), "VideoOrganizer")
    new_file = os.path.join(new_dir, "app.log")
    monkeypatch.setattr(app_logger, "_LOG_DIR",  new_dir)
    monkeypatch.setattr(app_logger, "_LOG_FILE", new_file)
    monkeypatch.setattr(app_logger, "_initialised", False)

    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass

    yield

    # Tear down test handlers so the next test starts clean.
    for h in list(root.handlers):
        root.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass
