"""
Disk-access coordinator — the "master" that arbitrates between background
thumbnail generation and foreground preview/seek so they NEVER touch the
disk at the same time.

The problem this solves
───────────────────────
On an HDD, two threads each reading a different multi-GB video file force
the drive head to thrash between them. Each read that would take 1 s alone
takes 10 s+ when fighting another. With enough concurrent readers (a few
thumbnail workers + a hover-preview thread, all opening 10–30 GB MKVs) every
read stalls, the GUI event loop starves, and Windows force-closes the app
("Application Hang"). This was the user-reported "freeze after a few videos".

The fix: exactly two modes of disk access, mutually exclusive.
  • BACKGROUND — thumbnail generation. The expensive open+decode of a source
    file is SERIALIZED to one worker at a time (MAX_BG_CONCURRENT_DECODES);
    two workers each reading a different multi-GB file thrash the HDD head
    just like a thumbnail + a preview do, and on a cold disk that thrash can
    keep them from draining when a preview arrives (→ degraded overlap → a
    decoder thread dies). Cache hits do NOT enter a background section, so
    warmed thumbnails still load fully in parallel.
  • FOREGROUND — user-initiated preview or seek. Exclusive: while a
    foreground op runs, NO background work touches the disk.

Foreground has priority: the moment a foreground request starts waiting,
new background sections block at entry, and in-flight background sections
are allowed to finish ("complete their current thumbnail and stop"), then
the foreground op proceeds. When it ends, background resumes.

How threads use it
──────────────────
Background thumbnail worker (runs on the QThreadPool):

    with COORDINATOR.background_section(thread_label, video_path):
        ... open file, decode one frame, write JPEG ...

Foreground preview / seek thread:

    if COORDINATOR.begin_foreground(label, timeout_s=8):
        try:
            ... open file, stream frames ...
        finally:
            COORDINATOR.end_foreground(label)

Both `background_section` entry and `begin_foreground` block the CALLING
worker thread — never the GUI thread. The GUI only ever starts/stops
threads and emits signals.

Why a shared gate instead of a literal master thread
─────────────────────────────────────────────────────
A dedicated coordinator thread would itself be a component that can wedge,
deadlock, or be the thing we have to debug next. A lock + condition variable
is the standard, race-free realization of the same coordinator ROLE: every
thread consults the shared state, and the OS scheduler + condition variable
do the hand-off. We still keep a worker→video registry (see `snapshot()`)
so the coordinator's state is fully observable for logging/diagnostics —
satisfying "the master can see which thread is working on which video".
"""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager

from app_logger import get_logger

log = get_logger(__name__)


class DiskAccessCoordinator:
    """Arbitrates BACKGROUND (thumbnail) vs FOREGROUND (preview/seek) disk
    access. Thread-safe; all public methods may be called from any worker
    thread. Do NOT call the blocking methods (`begin_foreground`,
    `background_section` entry) from the GUI thread."""

    # Seconds background thumbnail work stays parked AFTER the last
    # foreground op releases. While the user is actively browsing (a hover
    # or seek every couple of seconds), this keeps thumbnails suspended so
    # each preview gets the drive head to itself. Thumbnails only resume
    # once the user has genuinely paused for this long. Realizes the user's
    # "allow the thumbnail threads to start back up again [when done]".
    BACKGROUND_COOLDOWN_S = 2.0

    # Max background DECODE sections (open + decode of a source file) allowed
    # to run at once. This is 1 — i.e. background decodes are SERIALIZED —
    # for two reasons learned from a cold-disk crash:
    #   1. Two workers each reading a different multi-GB file thrash the HDD
    #      head exactly the way a thumbnail + a preview do. Serial decode is
    #      actually FASTER per file on an HDD (no head thrash) and surfaces
    #      the FIRST thumbnail sooner; the total wall-clock for a batch is the
    #      same or better.
    #   2. force_close() on a native av.open(path) worker can only set a flag
    #      (there is no Python file object to close) — it CANNOT interrupt an
    #      in-flight decode. So N concurrent workers stuck in slow cold-disk
    #      reads can't drain inside begin_foreground's timeout → the preview
    #      proceeds DEGRADED and overlaps them → N+1 readers thrash a cold HDD
    #      until a decoder thread dies (native exit 127). Serializing to 1
    #      means begin_foreground drains in a SINGLE decode, so a preview
    #      almost never goes degraded, and the worst-case overlap is 1 bg + 1
    #      fg instead of 3 bg + 1 fg.
    # NOTE: this gates only the expensive open+decode section. Cache HITS are
    # loaded before the background_section (see thumbnail_generator._do_work)
    # so warmed thumbnails still load fully in parallel across the pool.
    MAX_BG_CONCURRENT_DECODES = 1

    def __init__(self) -> None:
        self._cond = threading.Condition(threading.Lock())
        # Number of foreground requests currently WAITING to start. While
        # this is > 0, background sections block at entry (priority).
        self._fg_waiting = 0
        # True while a foreground op is actively using the disk (exclusive).
        self._fg_active = False
        # True while ALL background work is paused indefinitely (e.g. the
        # full-screen player owns the disk for the whole duration of playback).
        # Distinct from _fg_active, which is a transient preview/seek burst.
        self._bg_paused = False
        # Label of the current foreground holder (None when idle). end_
        # foreground() only releases if the caller's label matches this, so a
        # stopped-but-still-unwinding preview can't clear a newer preview's
        # hold, and a release is safely idempotent.
        self._fg_holder: 'str | None' = None
        # Count of background sections currently holding disk access.
        self._bg_active = 0
        # Monotonic time of the last foreground release (for the cooldown).
        self._last_fg_release = 0.0
        # Observability: which worker/thread is doing what right now.
        #   key   = caller-supplied label (e.g. "thumb-1", "preview")
        #   value = (mode, video_path, started_monotonic)
        self._registry: dict[str, tuple[str, str, float]] = {}
        # Yield callbacks for in-flight background sections. When a
        # foreground request starts waiting, we invoke these so a worker
        # stuck in a long av.open() aborts (closes its file) instead of
        # making foreground wait out the full open timeout.
        #   key = label, value = callable()  (e.g. worker.force_close)
        self._yield_callbacks: dict[str, callable] = {}
        # FIFO fairness for the serialized background-decode slot. Each worker
        # takes a monotonic ticket on entry; only the OLDEST waiting ticket may
        # claim the slot. WITHOUT this, a freshly-pooled worker "barges" — it
        # checks the slot, finds it free the instant a decode releases it, and
        # claims it BEFORE a worker that has been parked waiting re-acquires the
        # lock. With an endless queue of fresh workers, a parked one can be
        # barged forever → its thumbnail never generates (observed: a browsed
        # video stuck "Generating…" while 50+ others completed). `_bg_waiting`
        # holds the tickets of workers currently inside background_section
        # (bounded by the pool size, so min() is cheap).
        self._bg_ticket_next = 0
        self._bg_waiting: set[int] = set()

    # ── BACKGROUND (thumbnail workers) ───────────────────────────────────
    @contextmanager
    def background_section(self, label: str, video_path: str,
                           on_yield=None):
        """Context manager wrapping one thumbnail's disk work.

        Blocks at ENTRY until: (a) no foreground op is active or waiting,
        AND (b) the post-foreground cooldown has elapsed. So a worker
        between files parks the moment the user interacts and stays parked
        until the user has been idle for BACKGROUND_COOLDOWN_S.

        `on_yield`, if given, is a callable the coordinator may invoke from
        a FOREGROUND thread to abort this section's in-flight disk op (e.g.
        the worker's force_close, which shuts its file so a stuck av.open
        returns). The worker must tolerate being aborted (return cleanly).
        """
        # Take a FIFO ticket BEFORE waiting, so the serialized slot is granted
        # in strict arrival order (no barging — see _bg_waiting in __init__).
        with self._cond:
            ticket = self._bg_ticket_next
            self._bg_ticket_next += 1
            self._bg_waiting.add(ticket)
        acquired = False
        try:
            with self._cond:
                while True:
                    if self._fg_active or self._fg_waiting > 0 or self._bg_paused:
                        self._cond.wait()
                        continue
                    since = time.monotonic() - self._last_fg_release
                    if since < self.BACKGROUND_COOLDOWN_S:
                        # Idle wait until cooldown expires (or a new fg arrives,
                        # which notify_all wakes us to re-check).
                        self._cond.wait(timeout=self.BACKGROUND_COOLDOWN_S - since)
                        continue
                    if self._bg_active >= self.MAX_BG_CONCURRENT_DECODES:
                        # Serialize background decodes (see MAX_BG_CONCURRENT_DECODES):
                        # another worker is already reading a multi-GB file; waiting
                        # avoids HDD head-thrash and keeps begin_foreground's drain to
                        # a single decode. The finishing section's notify_all wakes us.
                        self._cond.wait()
                        continue
                    if ticket != min(self._bg_waiting):
                        # An OLDER ticket is still waiting — let it go first.
                        # Serving strictly oldest-first makes the serialized
                        # slot starvation-free (a parked worker can't be barged
                        # forever by an endless stream of freshly-pooled ones).
                        self._cond.wait()
                        continue
                    break
                self._bg_active += 1
                acquired = True
                self._registry[label] = ("background", video_path, time.monotonic())
                if on_yield is not None:
                    self._yield_callbacks[label] = on_yield
            yield
        finally:
            with self._cond:
                # Always drop our ticket so the next-oldest waiter advances —
                # even if we abandoned before acquiring (e.g. wait() raised),
                # which would otherwise wedge the FIFO permanently.
                self._bg_waiting.discard(ticket)
                if acquired:
                    # Clamp at 0: reset() (folder switch / shutdown) may have
                    # zeroed the counter while this section was still draining.
                    # Going negative would make `_bg_active > 0` perpetually
                    # False and let foreground skip the drain wait.
                    self._bg_active = max(0, self._bg_active - 1)
                    self._registry.pop(label, None)
                    self._yield_callbacks.pop(label, None)
                # Wake the next-oldest waiter / any foreground waiter.
                self._cond.notify_all()

    # ── FOREGROUND (preview / seek) ──────────────────────────────────────
    def begin_foreground(self, label: str, video_path: str = "",
                         timeout_s: float = 8.0) -> bool:
        """Acquire exclusive foreground disk access.

        Returns True once acquired (no background sections in flight and no
        other foreground op active). Returns False if `timeout_s` elapses
        first — the caller may then proceed in a DEGRADED mode (it logs a
        warning) rather than block forever, so a single wedged background
        worker can't permanently deny preview.

        Blocks the CALLING thread (a preview/seek QThread), never the GUI.
        """
        deadline = time.monotonic() + timeout_s
        # Phase 1: register as waiting (blocks NEW background) and snapshot
        # the in-flight yield callbacks — all under the lock, but briefly.
        with self._cond:
            self._fg_waiting += 1
            self._cond.notify_all()
            yielders = list(self._yield_callbacks.values())
        # Phase 2: abort in-flight background sections OUTSIDE the lock. A
        # yield callback (force_close) may BLOCK if a read is in flight, and
        # holding the coordinator lock across a blocking close would freeze
        # every other thread that touches the coordinator (this was a real
        # GUI-freeze bug). We run on the preview thread, so blocking here is
        # harmless. force_close is idempotent.
        for cb in yielders:
            try:
                cb()
            except Exception:
                log.debug("background yield callback raised", exc_info=True)
        # Phase 3: wait for drain + acquire.
        with self._cond:
            timed_out = False
            try:
                # Wait until no background is in flight and no other
                # foreground holds the gate.
                while self._fg_active or self._bg_active > 0:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        log.warning(
                            "Foreground '%s' timed out after %.1fs waiting "
                            "for disk (fg_active=%s bg_active=%d) — proceeding "
                            "degraded (claims gate to block NEW background). "
                            "In-flight: %s",
                            label, timeout_s, self._fg_active, self._bg_active,
                            self._describe_locked())
                        timed_out = True
                        break
                    self._cond.wait(timeout=remaining)
                # Claim the gate whether we drained cleanly or timed out.
                # Either way, NEW background sections now block until we
                # end_foreground(). A degraded (timed_out) caller may still
                # overlap one wedged in-flight bg op, but never new ones.
                self._fg_active = True
                self._fg_holder = label
                self._registry[label] = ("foreground", video_path,
                                         time.monotonic())
                return not timed_out
            finally:
                self._fg_waiting -= 1
                self._cond.notify_all()

    def end_foreground(self, label: str) -> None:
        """Release foreground disk access — but ONLY if `label` is the current
        holder. This is the crux of decoupling gate-release from thread
        teardown: a preview's stop() calls this from the GUI thread to free
        the gate INSTANTLY, even though the preview's own worker thread may
        still be grinding inside a slow av.open() on a 27 GB file. The wedged
        thread keeps unwinding on its own (its file was async-closed); when it
        finally returns, its _play finally calls end_foreground() again, but
        by then a newer preview may hold the gate, so the label won't match
        and this is a harmless no-op.

        Background does NOT resume immediately — it waits out
        BACKGROUND_COOLDOWN_S so a user moving between videos every second
        keeps thumbnails parked (each preview gets clean disk)."""
        with self._cond:
            self._registry.pop(label, None)
            if self._fg_holder == label:
                self._fg_active = False
                self._fg_holder = None
                self._last_fg_release = time.monotonic()
            self._cond.notify_all()

    # ── UI activity (scroll / resize) ────────────────────────────────────
    def note_ui_activity(self) -> None:
        """Tell the coordinator the GUI is busy (scrolling, resizing, etc.).

        NON-BLOCKING — safe to call from the GUI thread on every scroll
        tick. It (1) aborts any in-flight background thumbnail work so the
        disk frees up immediately, and (2) bumps the cooldown so background
        stays parked until the user has been idle for BACKGROUND_COOLDOWN_S.

        Why: even though scrolling does no disk I/O itself, 3 thumbnail
        workers + a preview all reading 20 GB files saturate the HDD and
        make the whole process (GUI paint/layout included) unresponsive.
        Suspending disk work the instant the user scrolls keeps the UI
        smooth. This realizes the user's "before responding to scrolling,
        abort all those tasks" requirement without ever blocking the GUI
        thread."""
        # GUI-thread safe: ONLY bump the cooldown + wake parked workers.
        # We deliberately do NOT invoke yield callbacks here — force_close
        # does file.close(), which can block for seconds on Windows when a
        # read is in flight, and this runs on the GUI thread. Blocking the
        # GUI thread is the very freeze we're preventing. In-flight
        # background work instead winds down on its own within its bounded
        # av.open timeout, then parks (cooldown keeps it parked while the
        # user keeps scrolling). New background work blocks immediately.
        with self._cond:
            self._last_fg_release = time.monotonic()
            self._cond.notify_all()

    def set_background_paused(self, paused: bool) -> None:
        """Pause/resume ALL background thumbnail work for an extended period —
        e.g. while the in-app full-screen player owns the disk. Unlike a
        foreground preview (a transient burst gated by begin/end_foreground),
        playback can run for minutes, so this is a simple held flag.

        GUI-thread SAFE and non-blocking: it only sets the flag, wakes parked
        workers, and (on resume) applies the normal post-foreground cooldown so
        a quick open→close doesn't immediately re-thrash the disk. Like
        note_ui_activity(), it does NOT force-close in-flight work from the GUI
        (a blocking file.close() would freeze the GUI); the one bg decode that
        may already be running winds down on its own, after which background
        stays parked until resumed. New background sections block immediately.
        """
        with self._cond:
            self._bg_paused = bool(paused)
            if not paused:
                self._last_fg_release = time.monotonic()
            self._cond.notify_all()

    # ── observability / diagnostics ──────────────────────────────────────
    def snapshot(self) -> dict:
        """Return a copy of current coordinator state for logging/tests."""
        with self._cond:
            return {
                "fg_waiting": self._fg_waiting,
                "fg_active": self._fg_active,
                "bg_active": self._bg_active,
                "bg_paused": self._bg_paused,
                "registry": dict(self._registry),
            }

    def _describe_locked(self) -> str:
        # Caller must hold self._cond.
        now = time.monotonic()
        parts = [f"{lbl}:{mode}:{video_path[-30:]}({now - t0:.1f}s)"
                 for lbl, (mode, video_path, t0) in self._registry.items()]
        return "; ".join(parts) if parts else "(idle)"

    def reset(self) -> None:
        """Force state back to idle. Use only when tearing down (folder
        switch / shutdown) and you have already stopped all workers."""
        with self._cond:
            self._fg_waiting = 0
            self._fg_active = False
            self._fg_holder = None
            self._bg_active = 0
            self._bg_paused = False
            self._bg_waiting.clear()
            self._bg_ticket_next = 0
            self._registry.clear()
            self._cond.notify_all()


# Process-wide singleton. Every thumbnail worker and every preview/seek
# thread shares this one instance.
COORDINATOR = DiskAccessCoordinator()
