"""
Centralised logging for Video Organizer.

Log file: %APPDATA%\\VideoOrganizer\\app.log  (rotating, 5 MB × 3 backups)
Console : WARNING and above (stderr)

Usage in any module:
    from app_logger import get_logger
    log = get_logger(__name__)
    log.debug("...")
    log.info("...")
    log.warning("...")
    log.error("...", exc_info=True)
"""

import logging
import logging.handlers
import os
import sys

_LOG_DIR = os.path.join(os.environ.get("APPDATA", os.path.expanduser("~")),
                        "VideoOrganizer")
_LOG_FILE = os.path.join(_LOG_DIR, "app.log")

_FMT = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
_DATE = "%Y-%m-%d %H:%M:%S"

_initialised = False


def _setup():
    global _initialised
    if _initialised:
        return

    os.makedirs(_LOG_DIR, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # ── rotating file handler ─────────────────────────────────────────────────
    fh = logging.handlers.RotatingFileHandler(
        _LOG_FILE,
        maxBytes=5 * 1024 * 1024,   # 5 MB
        backupCount=3,
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(_FMT, datefmt=_DATE))
    root.addHandler(fh)

    # ── stderr handler (WARNING+) ─────────────────────────────────────────────
    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(logging.WARNING)
    sh.setFormatter(logging.Formatter(_FMT, datefmt=_DATE))
    root.addHandler(sh)

    _initialised = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger, initialising logging on first call."""
    _setup()
    return logging.getLogger(name)


def log_path() -> str:
    """Return the path to the current log file."""
    return _LOG_FILE
