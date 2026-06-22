"""
Tests for disk_coordinator.DiskAccessCoordinator.

The coordinator's whole job is to ensure the HDD head is never thrashed by
two threads reading different multi-GB files at once. These tests pin the
invariants that prevent the cold-disk crash:

  • BACKGROUND DECODES ARE SERIALIZED (MAX_BG_CONCURRENT_DECODES == 1). Two
    thumbnail workers each reading a different file thrashed a cold HDD until a
    decoder thread died (native exit 127); the fix serializes the open+decode
    section so only one runs at a time and begin_foreground drains in a single
    decode (so previews almost never go "degraded" and overlap a live decode).
  • FOREGROUND has priority: while it holds the gate, NEW background sections
    block at entry.
  • begin_foreground proceeds DEGRADED (returns False but still claims the
    gate) if an un-abortable in-flight background section won't drain in time.
  • end_foreground is LABEL-AWARE: a stale release can't clear a newer holder.
"""
import threading
import time

import pytest

from disk_coordinator import DiskAccessCoordinator


def test_serialization_constant_is_one():
    assert DiskAccessCoordinator.MAX_BG_CONCURRENT_DECODES == 1


def test_background_decodes_are_serialized():
    """Three workers race into background_section; never more than
    MAX_BG_CONCURRENT_DECODES may be inside at once."""
    c = DiskAccessCoordinator()  # fresh: _last_fg_release=0 → no cooldown wait

    lock = threading.Lock()
    inside = 0
    max_inside = 0
    first_in = threading.Event()
    release = threading.Event()

    def worker(i):
        nonlocal inside, max_inside
        with c.background_section(f"w{i}", f"/vid/{i}.mp4"):
            with lock:
                inside += 1
                max_inside = max(max_inside, inside)
            first_in.set()
            # Hold the section so the others would (wrongly) pile in if the
            # serialization gate were broken.
            release.wait(3.0)
            with lock:
                inside -= 1

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
    for t in threads:
        t.start()

    assert first_in.wait(2.0), "no worker ever entered the background section"
    time.sleep(0.25)  # ample time for the other two to enter if not serialized
    with lock:
        assert max_inside == 1, (
            f"background decodes not serialized: {max_inside} were inside at "
            f"once (MAX_BG_CONCURRENT_DECODES={c.MAX_BG_CONCURRENT_DECODES})"
        )

    release.set()
    for t in threads:
        t.join(timeout=3.0)
        assert not t.is_alive()
    with lock:
        assert max_inside == 1  # stayed serialized through the whole drain


def test_background_slot_is_fifo_no_starvation():
    """The serialized bg slot MUST be granted oldest-waiter-first. Without
    FIFO, a worker that arrives just as the slot frees can "barge" ahead of a
    worker already parked; with an endless stream of arrivals the parked one
    starves forever — the reproduced 'stuck Generating…' bug. This pins the
    fix: a late "barger" (D) cannot jump an earlier waiter (C)."""
    c = DiskAccessCoordinator()
    acquired = []
    lock = threading.Lock()
    rel = {n: threading.Event() for n in "ABCD"}
    inside = {n: threading.Event() for n in "ABCD"}
    threads = {}

    def worker(name):
        with c.background_section(name, f"/{name}.mp4"):
            with lock:
                acquired.append(name)
            inside[name].set()
            rel[name].wait(5.0)

    def launch(name):
        t = threading.Thread(target=worker, args=(name,), daemon=True)
        threads[name] = t
        t.start()

    # A enters first and holds the (only) slot.
    launch("A")
    assert inside["A"].wait(2.0)
    # B, then C park behind A (staggered so tickets order A < B < C).
    launch("B"); time.sleep(0.15)
    launch("C"); time.sleep(0.15)
    with lock:
        assert acquired == ["A"]
    # Release A → B (oldest waiter) is served next.
    rel["A"].set()
    assert inside["B"].wait(2.0)
    # A late "barger" D arrives while B holds the slot.
    launch("D"); time.sleep(0.15)
    with lock:
        assert acquired == ["A", "B"]
    # Release B → C must go next; the barger D must NOT jump the queue.
    rel["B"].set()
    assert inside["C"].wait(2.0)
    with lock:
        assert acquired == ["A", "B", "C"], f"barger jumped the FIFO queue: {acquired}"
    rel["C"].set()
    assert inside["D"].wait(2.0)
    rel["D"].set()
    for t in threads.values():
        t.join(timeout=3.0)
    with lock:
        assert acquired == ["A", "B", "C", "D"]


def test_abandoned_ticket_does_not_wedge_fifo():
    """If a worker takes a ticket but its section body raises (abandons before
    completing), the FIFO must still advance — its ticket is dropped so the
    next waiter isn't blocked forever."""
    c = DiskAccessCoordinator()

    # A worker whose body raises right after acquiring.
    try:
        with c.background_section("boom", "/boom.mp4"):
            raise RuntimeError("decode blew up")
    except RuntimeError:
        pass

    # The slot/FIFO must be clean: a fresh worker acquires immediately.
    got = []
    with c.background_section("after", "/after.mp4"):
        got.append("in")
    assert got == ["in"]
    assert c.snapshot()["bg_active"] == 0


def test_active_foreground_blocks_new_background():
    c = DiskAccessCoordinator()
    # No background in flight → foreground acquires cleanly.
    assert c.begin_foreground("fg", "/preview.mp4", timeout_s=1.0) is True

    entered = threading.Event()

    def bg():
        with c.background_section("bg", "/thumb.mp4"):
            entered.set()

    t = threading.Thread(target=bg)
    t.start()
    try:
        # While foreground holds the gate, background must NOT enter.
        assert not entered.wait(0.4), "background ran while foreground held the gate"
        c.end_foreground("fg")
        # After release + the post-fg cooldown, background may proceed.
        assert entered.wait(c.BACKGROUND_COOLDOWN_S + 1.5)
    finally:
        t.join(timeout=2.0)


def test_begin_foreground_degraded_when_bg_wont_drain():
    """An in-flight background section with NO yield callback can't be aborted;
    a foreground op must still return (degraded=False) after its timeout and
    claim the gate so NEW background work is blocked."""
    c = DiskAccessCoordinator()
    holding = threading.Event()
    release = threading.Event()

    def stuck_bg():
        # No on_yield → begin_foreground cannot force it to drain.
        with c.background_section("bg", "/huge.mkv"):
            holding.set()
            release.wait(3.0)

    t = threading.Thread(target=stuck_bg)
    t.start()
    try:
        assert holding.wait(2.0)
        got = c.begin_foreground("fg", "/preview.mp4", timeout_s=0.3)
        assert got is False, "expected a DEGRADED (timed-out) foreground grant"
        snap = c.snapshot()
        assert snap["fg_active"] is True, "degraded fg must still claim the gate"
        c.end_foreground("fg")
    finally:
        release.set()
        t.join(timeout=3.0)


def test_end_foreground_is_label_aware():
    c = DiskAccessCoordinator()
    assert c.begin_foreground("A", "/a.mp4", timeout_s=1.0) is True
    # A stale release from a stopped-but-unwinding preview must NOT clear the
    # current holder.
    c.end_foreground("STALE")
    assert c.snapshot()["fg_active"] is True
    # The real holder's release clears it.
    c.end_foreground("A")
    assert c.snapshot()["fg_active"] is False


def test_begin_foreground_clean_acquire_returns_true():
    c = DiskAccessCoordinator()
    assert c.begin_foreground("fg", "/p.mp4", timeout_s=1.0) is True
    assert c.snapshot()["fg_active"] is True
    c.end_foreground("fg")
    assert c.snapshot()["fg_active"] is False


def test_set_background_paused_blocks_until_resumed():
    """While paused (the in-app full-screen player owns the disk), a background
    section must NOT enter; once resumed it proceeds. This is the fix for
    thumbnail generation stalling/thrashing during full-screen playback."""
    c = DiskAccessCoordinator()
    c.BACKGROUND_COOLDOWN_S = 0.0          # isolate pause logic from the cooldown
    c.set_background_paused(True)
    assert c.snapshot()["bg_paused"] is True

    entered = threading.Event()

    def worker():
        with c.background_section("w", "/vid/x.mp4"):
            entered.set()
            time.sleep(0.05)

    t = threading.Thread(target=worker)
    t.start()
    # Must NOT enter while paused.
    assert not entered.wait(0.4), "background entered the disk while paused"
    # Resume → it proceeds.
    c.set_background_paused(False)
    assert entered.wait(2.0), "background did not resume after unpause"
    t.join(timeout=3.0)
    assert not t.is_alive()
    assert c.snapshot()["bg_paused"] is False


def test_reset_clears_background_paused():
    c = DiskAccessCoordinator()
    c.set_background_paused(True)
    c.reset()
    assert c.snapshot()["bg_paused"] is False
