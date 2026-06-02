"""
Regression tests for play-thread QThread lifecycle.

Two invariants protect against two distinct, previously-observed failures:

1. STRONG REF (prevents "QThread: Destroyed while thread is still running"):
   Both the hover-thumbnail playback (VideoThumbnailWidget._start_playback)
   and the bottom player panel (VideoPlayerWidget.play) must keep a Python
   reference to every _VideoPlayThread alive until AFTER Qt's `finished`
   fires. They do so by (a) `qthread_registry.install(self)` in the thread's
   __init__ and (b) adding it to the module-level `_running_play_threads` set.

2. NO DIRECT-CONNECTED finished SLOT (prevents the GIL ⊗ QThread-mutex
   deadlock that froze the app during heavy big-file browsing):
   `finished` must NOT be connected to a bare lambda/closure. A bare callable
   is a DirectConnection — it runs synchronously inside the worker's
   QThreadPrivate::finish() (holding the QThread mutex, needing the GIL) and
   deadlocks against the GUI thread's deleteLater→~QThread::wait(). Cleanup
   must instead route through a GUI-thread QObject slot (queued):
   `_play_thread_reaper.reap`.

These tests grep the source so they stay fast, deterministic, and Qt-free.
"""
import os
import re

import pytest

_SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
_PLAY_FILES = ("video_thumbnail_widget.py", "video_player_widget.py")


def _read(name: str) -> str:
    with open(os.path.join(_SRC_ROOT, name), encoding="utf-8") as f:
        return f.read()


@pytest.mark.parametrize("fname", _PLAY_FILES)
def test_play_thread_added_to_strong_ref_set(fname):
    """Each creator must add its thread to the module-level set BEFORE
    launching it (so it can never be the only — droppable — reference)."""
    src = _read(fname)
    assert "_running_play_threads.add(thread)" in src, (
        f"{fname} must register its play thread in _running_play_threads."
    )


@pytest.mark.parametrize("fname", _PLAY_FILES)
def test_play_thread_cleanup_is_queued_not_direct(fname):
    """Removal from the set must go through the GUI-thread reaper (queued),
    never a bare lambda on `finished` (which would be a DirectConnection
    running inside the worker's finish() → GIL ⊗ mutex deadlock)."""
    src = _read(fname)
    # The safe, queued cleanup hook must be present.
    assert "_play_thread_reaper.reap" in src, (
        f"{fname} must connect `finished` to _play_thread_reaper.reap "
        f"(a GUI-thread QObject slot ⇒ queued connection)."
    )
    # The dangerous pattern must be ABSENT: a bare lambda connected to
    # finished that touches the registry directly.
    bad = re.search(r"finished\.connect\(\s*lambda", src)
    assert bad is None, (
        f"{fname}: `finished.connect(lambda ...)` is a DirectConnection that "
        f"runs inside the worker's finish() and can deadlock the GUI thread. "
        f"Route cleanup through _play_thread_reaper.reap instead."
    )


@pytest.mark.parametrize("fname", _PLAY_FILES)
def test_no_per_creator_deletelater_on_finished(fname):
    """C++ deletion is owned centrally by qthread_registry.install() (called
    in the thread __init__). Creators must NOT also connect
    `finished → deleteLater`: a second deletion racing the dying thread is
    part of the teardown hazard, and the duplicate is unnecessary."""
    src = _read(fname)
    assert "thread.finished.connect(thread.deleteLater)" not in src, (
        f"{fname}: do not connect finished→deleteLater here; "
        f"qthread_registry.install() handles C++ cleanup safely."
    )


def test_running_play_threads_set_is_module_level():
    """The set must be a module-global, not a class attribute, so all
    creators (thumbnail + player) share the same registry."""
    from video_thumbnail_widget import _running_play_threads
    assert isinstance(_running_play_threads, set)


def test_player_imports_strong_ref_set_from_thumbnail_module():
    """Both files must use the SAME set object."""
    from video_thumbnail_widget import _running_play_threads as a
    from video_player_widget import _running_play_threads as b
    assert a is b


def test_reaper_is_shared_qobject():
    """The reaper must be a single shared QObject instance (so it has a fixed
    GUI-thread affinity that makes the finished connection queued)."""
    from PyQt6.QtCore import QObject
    from video_thumbnail_widget import _play_thread_reaper as a
    from video_player_widget import _play_thread_reaper as b
    assert a is b
    assert isinstance(a, QObject)


@pytest.mark.parametrize("fname", _PLAY_FILES)
def test_register_happens_before_launch(fname):
    """The .add() call must appear before the thread is launched — whether
    launched directly via thread.start() (player) or handed to the
    PREVIEW_MANAGER (thumbnail hover)."""
    instantiation_re = re.compile(r"^\s*\w+\s*=\s*_VideoPlayThread\(", re.MULTILINE)
    src = _read(fname)
    matches = list(instantiation_re.finditer(src))
    assert matches, f"no _VideoPlayThread instantiation found in {fname}"

    for m in matches:
        tail = src[m.start():]
        window = "\n".join(tail.splitlines()[:35])
        add_pos    = window.find("_running_play_threads.add")
        # A launch is either a direct start() or a hand-off to the manager.
        start_pos  = window.find("thread.start()")
        submit_pos = window.find("PREVIEW_MANAGER.submit")
        launch_pos = min([p for p in (start_pos, submit_pos) if p != -1],
                         default=-1)
        assert add_pos != -1, f"missing .add in {fname} near {m.start()}"
        assert launch_pos != -1, (
            f"missing thread launch (start/submit) in {fname} near {m.start()}"
        )
        assert add_pos < launch_pos, (
            f"{fname}: _running_play_threads.add must come BEFORE the thread "
            f"is launched — saw add@{add_pos} launch@{launch_pos}"
        )
