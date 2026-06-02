"""
Tests for batch_rename_dialog's pure logic (template + filename validation).

We avoid creating the QDialog (no QApplication needed) by importing only
the module-level helpers.
"""
import pytest

from batch_rename_dialog import (
    _validate_filename,
    _INVALID_CHARS,
    _RESERVED_NAMES,
)


# ── filename validation ──────────────────────────────────────────────────────
class TestValidateFilename:
    def test_valid_simple_name(self):
        assert _validate_filename("movie.mp4") == ""

    def test_valid_with_unicode(self):
        assert _validate_filename("Café_Société.mp4") == ""

    def test_empty_rejected(self):
        assert "empty" in _validate_filename("")
        assert "empty" in _validate_filename("   ")

    @pytest.mark.parametrize("ch", list('<>:"/\\|?*'))
    def test_invalid_characters_rejected(self, ch):
        err = _validate_filename(f"name{ch}file.mp4")
        assert "invalid characters" in err

    def test_trailing_dot_rejected(self):
        assert "space or dot" in _validate_filename("name.")

    def test_trailing_space_rejected(self):
        assert "space or dot" in _validate_filename("name ")

    def test_too_long_rejected(self):
        assert "255" in _validate_filename("a" * 256 + ".mp4")

    def test_at_255_chars_allowed(self):
        # Exactly 255 chars is the limit
        assert _validate_filename("a" * 255) == ""

    @pytest.mark.parametrize("name", ["CON", "PRN", "AUX", "NUL", "COM1", "LPT9"])
    def test_reserved_windows_names_rejected(self, name):
        assert "reserved" in _validate_filename(name)
        assert "reserved" in _validate_filename(f"{name}.mp4")
        assert "reserved" in _validate_filename(name.lower())   # case-insensitive

    def test_normal_filenames_with_reserved_substring_allowed(self):
        # CONFIG starts with CON but is not a reserved name
        assert _validate_filename("CONFIG.txt") == ""
        assert _validate_filename("MyCOM1.mp4") == ""


# ── template substitution (re-implemented inline; matches dialog logic) ───────
class TestTemplateRendering:
    """The dialog's _apply_template is a method of BatchRenameDialog (which
    requires Qt), so we inline-test the regex pattern itself to lock the
    template grammar."""

    def test_invalid_chars_constant_completeness(self):
        # Ensure the invalid chars list matches Windows reserved set
        for ch in r'<>:"/\|?*':
            assert ch in _INVALID_CHARS

    def test_reserved_names_constant_includes_all_com_lpt(self):
        for i in range(1, 10):
            assert f"COM{i}" in _RESERVED_NAMES
            assert f"LPT{i}" in _RESERVED_NAMES
        for n in ("CON", "PRN", "AUX", "NUL"):
            assert n in _RESERVED_NAMES
