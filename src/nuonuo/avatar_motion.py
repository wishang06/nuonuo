"""Blink timing and audio-to-mouth mapping. No pygame, so the avatar loop can stay free."""

import math
import random

BLINK_CLOSE_S = 0.12
BLINK_OPEN_S = 0.12
DOUBLE_GAP_S = 0.08
DOUBLE_CHANCE = 0.22
BLINK_GAP_MIN = 2.0
BLINK_GAP_MAX = 6.0
MOUTH_TAU = 0.09
LOOK_TAU = 0.35


def eyelid_amount(elapsed: float, close_s: float = BLINK_CLOSE_S, open_s: float = BLINK_OPEN_S) -> float:
    """0 open, 1 shut. The lid closes in close_s and opens in open_s."""
    if elapsed < 0:
        return 0.0
    if elapsed < close_s:
        return elapsed / close_s
    if elapsed < close_s + open_s:
        return 1.0 - (elapsed - close_s) / open_s
    return 0.0


class BlinkScheduler:
    """Fires a blink every 2-6s, and sometimes a second blink just after."""

    def __init__(self, rng: random.Random | None = None, now: float = 0.0):
        self.rng = rng or random.Random()
        self.last_gap = 0.0
        self._double_at: float | None = None
        self._next = 0.0
        self._arm(now)

    def _arm(self, now: float) -> None:
        self.last_gap = self.rng.uniform(BLINK_GAP_MIN, BLINK_GAP_MAX)
        self._next = now + self.last_gap

    def poll(self, now: float) -> list[str]:
        events: list[str] = []
        if self._double_at is not None and now >= self._double_at:
            self._double_at = None
            events.append("blink")
        if now >= self._next:
            events.append("blink")
            if self.rng.random() < DOUBLE_CHANCE:
                self._double_at = now + BLINK_CLOSE_S + BLINK_OPEN_S + DOUBLE_GAP_S
                self._arm(self._double_at)
            else:
                self._double_at = None
                self._arm(now)
        return events


def smooth_toward(current: float, target: float, dt: float, tau: float) -> float:
    if dt <= 0:
        return current
    a = 1.0 - math.exp(-dt / max(tau, 1e-3))
    return current + a * (target - current)


def smooth_mouth(prev: float, level: float, dt: float, tau: float = MOUTH_TAU) -> float:
    return smooth_toward(prev, level, dt, tau)


def mouth_shape(level: float, speaking: bool) -> str:
    """closed, small, open, or wide. Silent audio stays closed."""
    if not speaking:
        return "closed"
    if level < 0.12:
        return "closed"
    if level < 0.35:
        return "small"
    if level < 0.62:
        return "open"
    return "wide"
