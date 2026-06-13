"""
Tests for qthread_registry — and a static-source audit that every QThread
subclass in the project uses it.

This is the canonical defence against "QThread: Destroyed while thread is
still running". If a new QThread is added without registering itself, this
test fails loudly with the file/class name.
"""
import os
import re

import pytest

import qthread_registry

_SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


# ── basic registry ───────────────────────────────────────────────────────────
def test_initial_state_empty():
    # Force a clean slate (other tests may have left residue)
    qthread_registry._running.clear()
    assert qthread_registry.running_count() == 0


def test_register_and_unregister_round_trip():
    qthread_registry._running.clear()

    class FakeThread:
        pass

    t = FakeThread()
    qthread_registry.register(t)
    assert qthread_registry.is_registered(t)
    assert qthread_registry.running_count() == 1

    qthread_registry.unregister(t)
    assert not qthread_registry.is_registered(t)
    assert qthread_registry.running_count() == 0


def test_unregister_unknown_is_safe():
    """unregister() should never raise on an unknown thread (idempotent)."""
    qthread_registry._running.clear()

    class T: pass
    qthread_registry.unregister(T())   # should not raise
    assert qthread_registry.running_count() == 0


def test_register_is_idempotent():
    """Registering the same thread twice should not double-count."""
    qthread_registry._running.clear()

    class T: pass
    t = T()
    qthread_registry.register(t)
    qthread_registry.register(t)
    assert qthread_registry.running_count() == 1


# ── static audit: every QThread.run() in the project must register/unregister ─
# (Skip files that legitimately mention QThread but don't subclass it.)
_FILES_WITH_QTHREAD_SUBCLASS = [
    "bulk_meta_worker.py",
    "duplicate_finder_dialog.py",
    "folder_size_scanner.py",
    "main_window.py",          # _MoveWorker
    "video_thumbnail_widget.py",  # _VideoPlayThread (uses its own _running_play_threads)
]


def _read(name: str) -> str:
    with open(os.path.join(_SRC_ROOT, name), encoding="utf-8") as f:
        return f.read()


def test_every_qthread_file_uses_some_strong_ref_pattern():
    """Every file that defines a QThread subclass must use the central
    qthread_registry (via install()) or a module-private equivalent.

    The key invariant: a strong Python reference must be held until
    AFTER Qt's `finished` signal fires, NOT just during run()."""
    for fname in _FILES_WITH_QTHREAD_SUBCLASS:
        src = _read(fname)
        if not re.search(r"class\s+\w+\s*\(\s*QThread\s*\)", src):
            continue
        uses_registry  = "from qthread_registry import" in src or \
                         "qthread_registry.install" in src
        uses_local_set = "_running_play_threads" in src
        assert uses_registry or uses_local_set, (
            f"{fname} defines a QThread subclass but has no strong-ref "
            f"protection. Call `from qthread_registry import install; "
            f"install(self)` in __init__."
        )


# ── verify each known QThread subclass uses install() in __init__ ───────────
@pytest.mark.parametrize("fname,classname", [
    ("bulk_meta_worker.py",           "BulkMetaWorker"),
    ("duplicate_finder_dialog.py",    "_ScanWorker"),
    ("folder_size_scanner.py",        "FolderSizeScanner"),
    ("main_window.py",                "_MoveWorker"),
    ("video_thumbnail_widget.py",     "_VideoPlayThread"),
])
def test_class_calls_install_in_init(fname, classname):
    """For each QThread subclass, verify __init__ calls install(self).

    Calling install() at construction time (not inside run()) is the only
    way to keep a strong Python ref ALIVE PAST Qt's `finished` emission —
    closing the GC-during-post-run-cleanup race."""
    src = _read(fname)
    cls_match = re.search(
        rf"class\s+{re.escape(classname)}\s*\(\s*QThread\s*\)\s*:",
        src,
    )
    assert cls_match, f"class {classname} not found in {fname}"
    body = src[cls_match.start():]

    # Match `def __init__(...)` allowing multi-line signatures
    init_match = re.search(
        r"^    def\s+__init__\s*\(.*?\)\s*:",
        body, re.MULTILINE | re.DOTALL,
    )
    assert init_match, f"{classname}.__init__ not found"

    init_body = body[init_match.end():]
    # Cap at next method at same indent level
    next_def = re.search(r"\n    def\s", init_body)
    if next_def:
        init_body = init_body[:next_def.start()]

    assert "install(self)" in init_body, (
        f"{classname}.__init__ in {fname} must call `install(self)` to "
        f"register a strong reference + auto-cleanup hook before start()."
    )


def test_install_function_exists_and_signature():
    """install() must exist and have the right effect: thread is added to the
    set and a finished slot is connected for auto-removal.

    Cleanup is now routed through a GUI-thread reaper (a QObject slot) so the
    connection is QUEUED — it must NOT be a bare/Direct lambda that would run
    inside the worker's QThreadPrivate::finish() and deadlock GIL ⊗ d->mutex
    (see qthread_registry module docstring)."""
    import qthread_registry
    assert hasattr(qthread_registry, "install")
    qthread_registry._running.clear()

    # Stand-in exposing a `finished` signal whose connect() accepts the
    # optional Qt connection-type argument (install passes UniqueConnection).
    class FakeSignal:
        def __init__(self):
            self.connections: list = []
        def connect(self, slot, *type_args):
            self.connections.append(slot)

    class FakeThread:
        def __init__(self):
            self.finished = FakeSignal()
            self._running = True
            self._finished = False
        def isRunning(self):
            return self._running
        def isFinished(self):
            return self._finished
        def deleteLater(self):
            pass

    t = FakeThread()
    qthread_registry.install(t)
    assert t in qthread_registry._running
    # Exactly one cleanup slot is connected, and it is the reaper's bound
    # method (a QObject slot ⇒ queued), NOT a bare lambda/closure.
    assert len(t.finished.connections) == 1
    assert t.finished.connections[0] == qthread_registry._reaper.reap

    # While the thread reports running, the reaper must NOT drop it (dropping
    # a still-running thread is the GC-during-finish crash we guard against).
    qthread_registry._reaper.reap()
    assert t in qthread_registry._running

    # Once the thread has fully finished, the reaper removes it.
    t._running = False
    t._finished = True
    qthread_registry._reaper.reap()
    assert t not in qthread_registry._running


def test_reaper_keeps_not_started_thread():
    """REGRESSION (0xC0000409 crash): a thread that is registered but NEVER
    STARTED — isRunning()==False AND isFinished()==False, exactly a preview
    deferred by PREVIEW_MANAGER — must NOT be reaped. The old `not isRunning()`
    test deleteLater'd it, then PREVIEW_MANAGER._launch() start()ed the deleted
    QThread → 'QThread: Destroyed while thread is still running' abort. The
    reaper must only reap FINISHED threads."""
    qthread_registry._running.clear()

    class _Pending:
        def __init__(self):
            self.finished = type('S', (), {'connect': lambda *a, **k: None})()
        def isRunning(self):
            return False     # never started
        def isFinished(self):
            return False     # …and never finished → will be launched later
        def deleteLater(self):
            raise AssertionError("a not-yet-started thread must NOT be deleted")

    t = _Pending()
    qthread_registry.install(t)
    qthread_registry._reaper.reap()
    assert t in qthread_registry._running, "pending thread was wrongly reaped"
    qthread_registry._running.discard(t)


# ── runtime: real QThreads reaped without the teardown deadlock ──────────────
def test_install_reaps_real_threads_without_deadlock():
    """The regression test for the GIL ⊗ QThread-mutex teardown deadlock.

    Spin up many real QThreads via install() and wait for each to FINISH. With
    the old bare-lambda-on-finished + deleteLater pattern, finish() ran a
    Direct-connected Python slot needing the GIL while the GUI thread held the
    GIL in deleteLater→~QThread::wait() — so a finishing thread would WEDGE and
    `wait()` here would hang (failing via the assert below / overall timeout).
    With the fix (cleanup routed through a queued GUI-thread reaper, no Direct
    slot in finish()), every thread finishes cleanly. We then drain the
    registry and confirm it empties.

    (We don't rely on a processEvents pump to dispatch the *queued* reap — that
    is timing-sensitive under a pre-existing QApplication — we wait on the
    threads directly, which is what actually detects the teardown deadlock,
    then reap synchronously via reap_now().)"""
    pytest.importorskip("PyQt6")
    from PyQt6.QtCore import QThread, QCoreApplication, QEventLoop

    app = QCoreApplication.instance() or QCoreApplication([])
    qthread_registry._running.clear()

    class _Quick(QThread):
        def run(self):
            self.msleep(5)   # brief work, then return → finish() runs

    threads = [_Quick() for _ in range(24)]
    for t in threads:
        qthread_registry.install(t)
        t.start()

    # If finish() deadlocked (the bug), wait() would never return.
    for t in threads:
        assert t.wait(5000), (
            "a QThread did not finish within 5 s — finish() likely deadlocked "
            "(GIL ⊗ QThread mutex). The teardown-deadlock fix has regressed."
        )

    # Threads are done; drain the registry (queued reaper + synchronous sweep).
    app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 50)
    qthread_registry.reap_now()
    assert qthread_registry.running_count() == 0, (
        f"{qthread_registry.running_count()} finished thread(s) not reaped"
    )
    app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 50)  # flush deleteLater
    # Let any pending deleteLater events flush so we don't leak C++ objects.
    app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 50)


def test_wait_all_stops_and_drains_running_threads():
    """wait_all() must stop + wait EVERY registered thread so none is left
    running when the process exits — that is the 'QThread: Destroyed while
    thread is still running' crash the app hit on close while a hover preview
    was still draining. Threads here loop until stop() flips their flag, the
    same contract every real worker honours."""
    pytest.importorskip("PyQt6")
    from PyQt6.QtCore import QThread, QCoreApplication, QEventLoop

    app = QCoreApplication.instance() or QCoreApplication([])
    qthread_registry._running.clear()

    class _Looper(QThread):
        def __init__(self):
            super().__init__()
            self._stop = False
        def stop(self):
            self._stop = True
        def run(self):
            while not self._stop:
                self.msleep(10)

    threads = [_Looper() for _ in range(8)]
    for t in threads:
        qthread_registry.install(t)
        t.start()
    # Make sure every thread is genuinely inside run() before we drain, so the
    # wait() phase is exercised (not skipped by an isRunning()==False check).
    for t in threads:
        for _ in range(200):
            if t.isRunning():
                break
            QThread.msleep(2)
        assert t.isRunning()

    stuck = qthread_registry.wait_all(3000)
    assert stuck == 0, f"{stuck} thread(s) did not stop within the timeout"
    for t in threads:
        assert not t.isRunning(), "wait_all returned with a thread still running"
    assert qthread_registry.running_count() == 0, "registry not drained"
    app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 50)  # flush deleteLater


def test_wait_all_reports_unstoppable_thread_as_stuck():
    """A worker that can't be cooperatively stopped in time — e.g. a hover
    preview parked inside a native, non-interruptible av.open() — must be
    REPORTED as stuck by wait_all (return > 0). That's the signal the app uses
    to force-exit (os._exit) instead of letting Qt destroy the running QThread,
    which aborts with 'QThread: Destroyed while thread is still running'."""
    pytest.importorskip("PyQt6")
    from PyQt6.QtCore import QThread, QCoreApplication

    app = QCoreApplication.instance() or QCoreApplication([])
    qthread_registry._running.clear()

    class _Unstoppable(QThread):
        def __init__(self):
            super().__init__()
            self._force = False
        def stop(self):
            pass            # ignore the cooperative stop (stuck native call)
        def run(self):
            while not self._force:
                self.msleep(10)

    t = _Unstoppable()
    qthread_registry.install(t)
    t.start()
    for _ in range(200):
        if t.isRunning():
            break
        QThread.msleep(2)
    assert t.isRunning()

    stuck = qthread_registry.wait_all(300)      # short — it won't cooperate
    assert stuck >= 1, "an unstoppable running thread must be reported stuck"

    # Cleanup: really stop it and drain so it doesn't leak into other tests.
    t._force = True
    assert t.wait(3000)
    qthread_registry._running.discard(t)
