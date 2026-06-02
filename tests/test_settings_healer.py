"""
Regression tests for `_heal_bad_settings()` in main.py.

This was added after my own VR-performance tests polluted the user's real
QSettings (Windows registry) — setting `large_file_threshold_mb=0` made
the hover-preview policy treat EVERY file as "large", capping at 1 FPS,
which felt like hover was broken.

The healer resets any persisted value outside a sane range to the default.
"""
import pytest
from PyQt6.QtCore import QSettings

from main import _heal_bad_settings


@pytest.fixture
def settings():
    """Fresh isolated QSettings for this test (conftest already forces
    IniFormat into a tempdir, so this won't touch the real registry)."""
    # Use a unique org/app name per test to avoid cross-test contamination
    import uuid
    s = QSettings('VideoOrganizerHealerTest', uuid.uuid4().hex)
    s.clear()
    return s


def test_healer_clears_out_of_range_threshold(settings, monkeypatch):
    """large_file_threshold_mb=0 (a real bug that made every file 'large')
    must be cleared so the default takes over."""
    settings.setValue('large_file_threshold_mb', 0)
    settings.sync()

    # Patch QSettings construction in the healer to use OUR test instance
    def fake_qsettings(*args, **kwargs):
        return settings
    monkeypatch.setattr('main.QSettings', fake_qsettings, raising=False)
    # The healer imports QSettings inside the function — patch the module
    import main as _m
    monkeypatch.setattr('PyQt6.QtCore.QSettings', fake_qsettings)

    _heal_bad_settings()
    # Confirm bad value is gone (caller's AppSettings property will return default)
    settings.sync()
    assert settings.value('large_file_threshold_mb') is None


def test_healer_keeps_valid_values(settings, monkeypatch):
    """A user who has set large_file_threshold_mb=1000 explicitly should
    NOT have their setting wiped."""
    settings.setValue('large_file_threshold_mb', 1000)
    settings.setValue('hover_preview_max_gb', 8.0)
    settings.setValue('hover_preview_fps_large', 12)
    settings.sync()

    def fake_qsettings(*args, **kwargs):
        return settings
    monkeypatch.setattr('PyQt6.QtCore.QSettings', fake_qsettings)

    _heal_bad_settings()
    settings.sync()
    assert int(settings.value('large_file_threshold_mb')) == 1000
    assert float(settings.value('hover_preview_max_gb')) == 8.0
    assert int(settings.value('hover_preview_fps_large')) == 12


def test_healer_clears_unparseable_string(settings, monkeypatch):
    """If somehow a string got stored where a number should be, drop it."""
    settings.setValue('hover_preview_max_gb', 'not-a-number')
    settings.sync()

    def fake_qsettings(*args, **kwargs):
        return settings
    monkeypatch.setattr('PyQt6.QtCore.QSettings', fake_qsettings)

    _heal_bad_settings()
    settings.sync()
    assert settings.value('hover_preview_max_gb') is None


def test_healer_clears_fps_below_minimum(settings, monkeypatch):
    """The test that originally polluted: hover_preview_fps_large=1 means
    1 frame per second, which feels like 'no preview'. Must heal to default."""
    settings.setValue('hover_preview_fps_large', 1)
    settings.sync()

    def fake_qsettings(*args, **kwargs):
        return settings
    monkeypatch.setattr('PyQt6.QtCore.QSettings', fake_qsettings)

    _heal_bad_settings()
    settings.sync()
    assert settings.value('hover_preview_fps_large') is None


def test_healer_no_op_on_empty_settings(settings, monkeypatch):
    """If nothing is set, the healer should run cleanly and leave the
    storage empty (defaults will be used by AppSettings on read)."""
    def fake_qsettings(*args, **kwargs):
        return settings
    monkeypatch.setattr('PyQt6.QtCore.QSettings', fake_qsettings)

    _heal_bad_settings()
    settings.sync()
    # No keys set, no keys removed
    assert len(settings.allKeys()) == 0


def test_test_qsettings_org_is_isolated_from_production():
    """Verify conftest.py's env-var isolation actually works.  AppSettings
    in tests must NOT use 'VideoOrganizer' as the org — otherwise it would
    write to HKCU\\Software\\VideoOrganizer\\VideoOrganizer which is the
    user's REAL production settings.  If this regresses, tests will
    pollute the user's hover-preview / sort / window-geometry settings."""
    import os
    org = os.environ.get('VIDEO_ORGANIZER_QSETTINGS_ORG', '')
    app = os.environ.get('VIDEO_ORGANIZER_QSETTINGS_APP', '')
    assert org and org != 'VideoOrganizer', (
        f"VIDEO_ORGANIZER_QSETTINGS_ORG={org!r} — conftest.py must set "
        f"this to a non-production value before tests run."
    )
    assert app and app != 'VideoOrganizer', (
        f"VIDEO_ORGANIZER_QSETTINGS_APP={app!r} — same reason."
    )

    # And verify AppSettings honors the env var
    from app_settings import AppSettings
    s = AppSettings()
    org_name, app_name = s._settings.organizationName(), s._settings.applicationName()
    assert org_name != 'VideoOrganizer' or app_name != 'VideoOrganizer', (
        "AppSettings is writing to the production registry path. "
        "Did AppSettings.__init__ get reverted to hard-coded names?"
    )
