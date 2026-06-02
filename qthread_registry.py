"""
Central registry of running QThreads — prevents both the
"QThread: Destroyed while thread is still running" crash AND a
GIL ⊗ QThread-mutex deadlock during teardown (see below).

The crash this prevents
───────────────────────
1. A QThread Python wrapper has only one or two references (`self._worker`).
2. The owning object drops those references (`self._worker = None`).
3. Python GC destroys the QThread wrapper while the OS thread is still
   inside its post-run cleanup (e.g. before Qt emits `finished`).
4. The Python destructor calls the C++ ~QThread, which sees isRunning()==true.

The DEADLOCK this prevents (the 10-minute "torture test" freeze)
────────────────────────────────────────────────────────────────
The naive idiom is:

    thread.finished.connect(lambda t=thread: _running.discard(t))   # (A)
    thread.finished.connect(thread.deleteLater)                     # (B)

(A) connects `finished` to a BARE Python callable with no QObject context.
PyQt makes that a **DirectConnection** — it runs SYNCHRONOUSLY, on the dying
worker's own OS thread, *inside* `QThreadPrivate::finish()`, which holds the
QThread's internal `d->mutex`. The lambda body touches Python objects, so it
must take the **GIL**.

Meanwhile the GUI event loop, processing this thread's queued `deleteLater`
(B), calls `~QThread`, which (if the OS thread is still finishing) calls
`QThread::wait()` — needing that same `d->mutex` while the GUI thread holds
the **GIL**.

  • worker:  holds d->mutex  → wants GIL   (to run the Direct lambda)
  • GUI:     holds GIL       → wants d->mutex (in ~QThread::wait)

That is a textbook two-lock deadlock. It is timing-dependent (the GUI must
process deleteLater exactly while the worker is in finish()), which is why it
showed up as a *flaky* freeze, worst on a busy/cold-disk system.

The fix
───────
Never connect a bare/Direct callable to `finished`, and never let the GUI run
`~QThread::wait()` on a thread that is still finishing. We do BOTH by routing
all cleanup through a single `_Reaper` QObject that lives in the thread which
imports this module (the GUI/main thread). Because the reaper is a QObject
with GUI-thread affinity, connecting a worker's `finished` to its slot is an
AutoConnection that resolves to a **QueuedConnection** — the slot runs LATER,
on the GUI thread, AFTER `finish()` has fully completed and released
`d->mutex`. At that point the thread is truly finished, so dropping our strong
ref and calling `deleteLater()` is safe and `~QThread` never waits.

Usage at the CREATION site (NOT inside run):

    from qthread_registry import install
    thread = MyWorker(...)
    install(thread)        # holds a strong ref; auto-released on finished
    thread.start()
"""
from PyQt6.QtCore import QThread, QObject, Qt, pyqtSlot

# Module-level strong refs. Threads removed automatically (on the GUI thread)
# once they have fully finished — see _Reaper.
_running: 'set[QThread]' = set()


class _Reaper(QObject):
    """Reaps finished QThreads on the GUI thread, OFF the finish() critical
    section. See the module docstring for why this must not run inside the
    worker's QThreadPrivate::finish()."""

    @pyqtSlot()
    def reap(self) -> None:
        # Runs on the GUI thread via a QUEUED connection, i.e. only after the
        # emitting thread's finish() has returned and released d->mutex. So
        # every thread we touch here is genuinely done: isRunning() is False,
        # deleteLater() won't trigger ~QThread::wait(), and dropping the strong
        # ref can't race the C++ teardown.
        for t in list(_running):
            try:
                still_running = t.isRunning()
            except RuntimeError:
                still_running = False   # underlying C++ object already gone
            if not still_running:
                _running.discard(t)
                try:
                    t.deleteLater()
                except (RuntimeError, AttributeError):
                    pass


# Created on the importing (GUI/main) thread, so its slot always runs there.
_reaper = _Reaper()


def install(thread: QThread) -> None:
    """Add `thread` to the strong-ref set and arrange for it to be reaped
    safely (on the GUI thread) once it has fully finished. Call this BEFORE
    thread.start().

    The connection target is a GUI-thread QObject slot, so it is a QUEUED
    connection — cleanup runs after the worker's finish() completes, never
    synchronously inside it. This is what avoids the GIL ⊗ d->mutex deadlock
    described in the module docstring.
    """
    _running.add(thread)
    try:
        # AutoConnection: emitter (worker OS thread) != receiver (_reaper's GUI
        # thread) ⇒ QueuedConnection. UniqueConnection makes a repeat install()
        # a no-op instead of stacking duplicate reap() calls.
        thread.finished.connect(
            _reaper.reap, Qt.ConnectionType.UniqueConnection)
    except (TypeError, RuntimeError):
        # Already connected (UniqueConnection rejected) or a non-Qt stand-in
        # used in unit tests — fall back to a plain connect.
        try:
            thread.finished.connect(_reaper.reap)
        except (TypeError, RuntimeError):
            pass


def running_count() -> int:
    """Number of threads currently registered. Useful for tests / debugging."""
    return len(_running)


def is_registered(thread: QThread) -> bool:
    return thread in _running


def reap_now() -> None:
    """Synchronously reap finished threads. Safe to call from the GUI thread
    (e.g. during shutdown) — same finished-only logic as the queued path."""
    _reaper.reap()


def wait_all(timeout_ms: int = 3000) -> int:
    """Stop (if the thread exposes stop()) and WAIT for every registered thread
    to finish. Call this on app shutdown, BEFORE the window/widgets are
    destroyed.

    Why it's needed even though `_running` holds strong refs: those refs keep
    the QThread wrappers alive past `finished` during normal operation, but at
    shutdown/interpreter-exit the set itself is torn down. Any wrapper whose
    run() is STILL executing is then destroyed mid-flight → the C++ ~QThread
    sees isRunning()==true → 'QThread: Destroyed while thread is still running',
    which aborts the process. A hover preview that was stopped but is still
    draining is no longer referenced by its (cleared) widget, so per-widget
    shutdown can't see it — but it IS here. Waiting drains it first.

    Two-phase (signal every stop, THEN wait each) so N threads cost ≈ the
    slowest one, not the sum. Returns the count that did NOT exit in time.
    """
    threads = list(_running)
    # Phase 1: signal stop on everything that supports it (non-blocking).
    for t in threads:
        try:
            stop = getattr(t, "stop", None)
            if callable(stop):
                stop()
        except (RuntimeError, AttributeError):
            pass
    # Phase 2: wait for each. Stops are already in flight, so a thread that
    # honours its flag returns almost immediately.
    stuck = 0
    for t in threads:
        try:
            if t.isRunning() and not t.wait(timeout_ms):
                stuck += 1
        except (RuntimeError, AttributeError):
            pass   # C++ object already gone, or a non-QThread test stand-in
    reap_now()
    return stuck


# ── Legacy helpers kept for backwards compatibility ──────────────────────────
# Older code paths called these from inside run().  That pattern has the race
# described above, but the helpers themselves are still safe primitives —
# they just shouldn't be the ONLY mechanism in a run() body.

def register(thread: QThread) -> None:
    """Low-level add to the set.  Prefer install() at the creation site."""
    _running.add(thread)


def unregister(thread: QThread) -> None:
    """Low-level remove from the set.  Idempotent."""
    _running.discard(thread)
