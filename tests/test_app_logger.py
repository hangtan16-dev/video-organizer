"""
Tests for app_logger — verifies the logger initialises only once and
writes to the expected location.
"""
import logging
import os
import tempfile

import pytest


@pytest.fixture
def isolated_logger(tmp_path, monkeypatch):
    """Point %APPDATA% at a temp dir so we don't pollute the real log file,
    and reload app_logger so it picks up the patched env var."""
    monkeypatch.setenv("APPDATA", str(tmp_path))
    import app_logger
    # Reset the singleton flag
    app_logger._initialised = False
    # Recompute paths since they were captured at import time
    app_logger._LOG_DIR  = os.path.join(str(tmp_path), "VideoOrganizer")
    app_logger._LOG_FILE = os.path.join(app_logger._LOG_DIR, "app.log")
    # Strip any handlers from the root logger left over from a prior test
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    yield app_logger


def test_log_file_created(isolated_logger):
    log = isolated_logger.get_logger("test")
    log.info("hello")
    for h in logging.getLogger().handlers:
        h.flush()
    assert os.path.exists(isolated_logger.log_path())


def test_log_writes_message(isolated_logger):
    log = isolated_logger.get_logger("test_module")
    log.info("a unique message 12345")
    for h in logging.getLogger().handlers:
        h.flush()
    with open(isolated_logger.log_path(), encoding="utf-8") as f:
        content = f.read()
    assert "a unique message 12345" in content
    assert "test_module" in content


def test_get_logger_returns_distinct_loggers(isolated_logger):
    a = isolated_logger.get_logger("module.a")
    b = isolated_logger.get_logger("module.b")
    assert a.name == "module.a"
    assert b.name == "module.b"


def test_setup_idempotent(isolated_logger):
    """Calling get_logger many times should not stack handlers — i.e. the
    count after N calls must equal the count after the first call."""
    isolated_logger.get_logger("test")
    after_one = len(logging.getLogger().handlers)
    for _ in range(10):
        isolated_logger.get_logger("test")
    after_many = len(logging.getLogger().handlers)
    assert after_many == after_one, (
        f"handlers stacked: {after_one} → {after_many}"
    )
