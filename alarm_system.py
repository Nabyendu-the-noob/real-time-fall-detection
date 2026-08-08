"""
Alarm system for fall detection — two-phase escalation.

Phase 1 ALERT   : 2 short beeps at 700 Hz  → fall just confirmed
Phase 2 CRITICAL: continuous fast beeps at 1400 Hz → person still on ground after 5s

Called from fall_detector.py:
    self.alarm.trigger()    — on transition.confirmed_fall
    self.alarm.escalate()   — when fall_duration >= 5.0 seconds
    self.alarm.reset()      — when fall clears (person got up)
    self.alarm.close()      — on shutdown (finally block)
"""

import sys
import threading
import time

# ── Backend ───────────────────────────────────────────────────────────────────
_BACKEND = "silent"

if sys.platform == "win32":
    try:
        import winsound as _winsound
        _BACKEND = "winsound"
    except ImportError:
        pass

if _BACKEND == "silent":
    try:
        import pygame as _pygame
        _pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
        _BACKEND = "pygame"
    except ImportError:
        pass

if _BACKEND == "silent":
    print("[AlarmSystem] WARNING: No sound backend found. "
          "Install pygame (pip install pygame) for audio.")

# ── Sound patterns ────────────────────────────────────────────────────────────
# (frequency_hz, duration_ms, gap_after_ms)

# Phase 1 — soft double-beep: "fall detected"
_ALERT = {
    "pattern": [(700, 250, 120), (700, 250, 0)],
    "repeats": 1,
}

# Phase 2 — continuous rapid high-pitch: "person still on ground"
_CRITICAL = {
    "pattern": [(1400, 100, 60)],
    "repeats": 999,          # loops until reset() is called
}


# ── Helpers ───────────────────────────────────────────────────────────────────


def _beep(freq_hz, duration_ms, stop_event):
    if _BACKEND == "winsound":
        chunk_ms = 50
        elapsed = 0
        while elapsed < duration_ms:
            if stop_event.is_set():
                return
            step = min(chunk_ms, duration_ms - elapsed)
            _winsound.Beep(freq_hz, step)
            elapsed += step
    else:
        deadline = time.time() + duration_ms / 1000
        while time.time() < deadline:
            if stop_event.is_set():
                return
            time.sleep(0.02)


def _gap(ms, stop_event):
    deadline = time.time() + ms / 1000
    while time.time() < deadline:
        if stop_event.is_set():
            return
        time.sleep(0.02)


def _play(profile, stop_event):
    for _ in range(profile["repeats"]):
        if stop_event.is_set():
            return
        for freq, dur, gap in profile["pattern"]:
            if stop_event.is_set():
                return
            _beep(freq, dur, stop_event)
            if gap > 0:
                _gap(gap, stop_event)


# ── AlarmSystem ───────────────────────────────────────────────────────────────

class AlarmSystem:
    def __init__(self):
        self._mode       = None     # None | "alert" | "critical"
        self._stop_event = threading.Event()
        self._thread     = None
        self._lock       = threading.Lock()

    # ── Public API ────────────────────────────────────────────────────────────

    def trigger(self):
        """
        Phase 1: play 2 short low beeps when a fall is first confirmed.
        Does nothing if already in alert or critical mode.
        """
        with self._lock:
            if self._mode is not None:
                return
            self._mode = "alert"
            self._start(_ALERT)
        print(f"[AlarmSystem] ALERT — fall detected | backend={_BACKEND}")

    def escalate(self):
        """
        Phase 2: switch to continuous high-pitch loop when person is still
        on the ground after 5 seconds.  Safe to call every frame — only
        acts once when transitioning from alert → critical.
        """
        with self._lock:
            if self._mode == "critical":
                return                   # already in critical, do nothing
            # Stop the alert beeps (or start fresh if somehow called first)
            self._stop_current()
            self._mode = "critical"
            self._start(_CRITICAL)
        print("[AlarmSystem] CRITICAL — person still on ground!")

    def reset(self):
        """
        Stop all sound.  Call when the fall clears (person got up).
        """
        with self._lock:
            if self._mode is None:
                return
            self._stop_current()
            self._mode = None
        print("[AlarmSystem] Reset — fall cleared.")

    def close(self):
        """Stop everything on shutdown."""
        with self._lock:
            self._stop_current()
            self._mode = None

    @property
    def is_critical(self):
        return self._mode == "critical"

    # ── Internal ──────────────────────────────────────────────────────────────

    def _start(self, profile):
        """Must be called with self._lock held."""
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=_play,
            args=(profile, self._stop_event),
            daemon=True,
            name=f"alarm-{self._mode}",
        )
        self._thread.start()

    def _stop_current(self):
        """Signal current thread to stop and release it. Lock must be held."""
        self._stop_event.set()
        self._thread = None