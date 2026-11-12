"""When a voice session may open, when it must close, and how much voice time is left today.

The gate is plain logic with no I/O apart from the small usage file, so it can be tested
without a camera, a mic, or a socket. The voice client feeds it readings and asks it what to do.

    asleep  -> awake   a face has been in view for wake_seconds (with no camera: speech near the mic)
    awake   -> asleep  idle_seconds with no face and no speech ("idle"), or the session cap ("session_cap")
    any     -> capped  today's voice time is used up ("daily_cap"); back to asleep on the next day
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path

log = logging.getLogger("nuonuo.gate")

ASLEEP = "asleep"
AWAKE = "awake"
CAPPED = "capped"

KEEP_DAYS = 14


def _env_float(env: Mapping[str, str], name: str, default: float) -> float:
    raw = env.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError:
        log.warning("%s=%r is not a number; using %s", name, raw, default)
        return default
    return max(0.0, value)


@dataclass(frozen=True)
class GateConfig:
    """Times are in seconds. A cap or timeout of 0 turns it off."""

    wake_seconds: float = 1.5
    idle_seconds: float = 3 * 60
    session_cap_seconds: float = 40 * 60
    daily_cap_seconds: float = 120 * 60
    voice_wake_seconds: float = 0.5
    checkpoint_seconds: float = 60.0
    # The client observes the gate every fraction of a second, and at most ~30 s apart during a
    # reconnect backoff. A longer gap means the machine slept (Windows Modern Standby suspends the
    # app, and time.monotonic() keeps counting). That time is not voice time and not idle time.
    suspend_gap_seconds: float = 60.0

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> GateConfig:
        env = os.environ if env is None else env
        return cls(
            wake_seconds=_env_float(env, "WAKE_SECONDS", 1.5),
            idle_seconds=60 * _env_float(env, "IDLE_TIMEOUT_MIN", 3),
            session_cap_seconds=60 * _env_float(env, "SESSION_CAP_MIN", 40),
            daily_cap_seconds=60 * _env_float(env, "DAILY_CAP_MIN", 120),
        )

    def describe(self) -> str:
        def minutes(seconds: float) -> str:
            return "off" if seconds <= 0 else f"{seconds / 60:g} min"

        return (
            f"wake after {self.wake_seconds:g}s of face, idle close {minutes(self.idle_seconds)}, "
            f"session cap {minutes(self.session_cap_seconds)}, daily cap {minutes(self.daily_cap_seconds)}"
        )


class DailyUsage:
    """Voice seconds used per local day, saved as JSON so a restart keeps today's total."""

    def __init__(self, path: Path | str | None = None, today: Callable[[], date] = date.today):
        self.path = Path(path) if path else None
        self.today = today
        self._lock = threading.Lock()
        self._days = self._load()

    def _load(self) -> dict[str, float]:
        if self.path is None or not self.path.is_file():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            log.warning("could not read %s; starting today's voice usage at 0", self.path)
            return {}
        if not isinstance(data, dict):
            return {}
        return {
            str(day): float(seconds)
            for day, seconds in data.items()
            if isinstance(seconds, (int, float)) and not isinstance(seconds, bool)
        }

    def used(self, day: date | None = None) -> float:
        key = (day or self.today()).isoformat()
        with self._lock:
            return self._days.get(key, 0.0)

    def add(self, seconds: float, day: date | None = None) -> float:
        """Add seconds to the day's total, save, and return the new total."""
        key = (day or self.today()).isoformat()
        with self._lock:
            if seconds > 0:
                self._days[key] = self._days.get(key, 0.0) + seconds
                for old in sorted(self._days)[:-KEEP_DAYS]:
                    del self._days[old]
                self._save()
            return self._days.get(key, 0.0)

    def _save(self) -> None:
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._days, indent=1), encoding="utf-8")
            tmp.replace(self.path)
        except OSError:
            log.warning("could not save voice usage to %s", self.path, exc_info=True)


class VoiceGate:
    """Wake gate, idle timeout, per-session cap and daily cap for the voice session."""

    def __init__(self, cfg: GateConfig | None = None, usage: DailyUsage | None = None, now: float = 0.0):
        self.cfg = cfg or GateConfig()
        self.usage = usage or DailyUsage()
        self.state = ASLEEP
        self.reason = ""
        self.present_since: float | None = None
        self.last_activity = now
        self.session_start: float | None = None
        self.camera = True
        self.resumed_at: float | None = None  # when the last sleep/suspend gap ended
        self._booked = 0.0
        self._checkpoint_at = 0.0
        self._last_seen: float | None = None
        self._lock = threading.RLock()

    def observe(self, now: float, face: bool | None, speech: bool = False, heard: bool = False) -> str:
        """Feed the latest readings and return the gate state.

        face is True/False from the camera, or None when there is no camera reading. Then
        `heard` (speech picked up by the local mic) stands in for presence. `speech` means
        someone is talking in the live session, either the user or Nuonuo.
        """
        with self._lock:
            self._skip_suspend(now)
            self.camera = face is not None
            present = bool(face) if face is not None else heard
            if present:
                if self.present_since is None:
                    self.present_since = now
            else:
                self.present_since = None
            if face or speech:
                self.last_activity = now
            self._step(now)
            return self.state

    def daily_left(self, now: float) -> float:
        """Seconds of voice time left today, counting the open session. Infinite with no cap."""
        cap = self.cfg.daily_cap_seconds
        if cap <= 0:
            return float("inf")
        with self._lock:
            unbooked = 0.0
            if self.session_start is not None:
                unbooked = max(0.0, now - self.session_start - self._booked)
            return cap - self.usage.used() - unbooked

    def session_opened(self, now: float) -> None:
        with self._lock:
            self.session_start = now
            self._booked = 0.0
            self._checkpoint_at = now
            self.last_activity = now

    def session_closed(self, now: float) -> float:
        """Book the session's remaining time and return its total length in seconds."""
        with self._lock:
            self._skip_suspend(now)
            if self.session_start is None:
                return 0.0
            length = max(0.0, now - self.session_start)
            self.usage.add(length - self._booked)
            self.session_start = None
            self._booked = 0.0
            return length

    def status(self) -> str:
        if self.state == CAPPED:
            return "resting: daily cap"
        if self.state == ASLEEP:
            return "sleeping" if self.camera else "sleeping (talk to wake)"
        return "awake"

    def _step(self, now: float) -> None:
        if self.session_start is not None and now - self._checkpoint_at >= self.cfg.checkpoint_seconds:
            self._checkpoint(now)
        out_of_time = self.daily_left(now) <= 0
        if self.state == CAPPED:
            if out_of_time:
                return
            self.state, self.reason = ASLEEP, ""
        if out_of_time:
            self._fall_asleep(CAPPED, "daily_cap")
            return
        if self.state == ASLEEP:
            need = self.cfg.wake_seconds if self.camera else self.cfg.voice_wake_seconds
            if self.session_start is None and self.present_since is not None and now - self.present_since >= need:
                self.state, self.reason = AWAKE, ""
                self.last_activity = now
            return
        idle = self.cfg.idle_seconds
        cap = self.cfg.session_cap_seconds
        if idle > 0 and now - self.last_activity >= idle:
            self._fall_asleep(ASLEEP, "idle")
        elif cap > 0 and self.session_start is not None and now - self.session_start >= cap:
            self._fall_asleep(ASLEEP, "session_cap")

    def _skip_suspend(self, now: float) -> None:
        """Leave a sleep/suspend gap out of the session length, the daily total, and the idle timer."""
        last, self._last_seen = self._last_seen, now
        if last is None:
            return
        gap = now - last
        if gap < self.cfg.suspend_gap_seconds:
            return
        log.info("no readings for %.0f s (the machine slept); not counting that as voice or idle time", gap)
        self.resumed_at = now
        self.present_since = None
        self.last_activity += gap
        if self.session_start is not None:
            self.session_start += gap
            self._checkpoint_at += gap

    def _fall_asleep(self, state: str, reason: str) -> None:
        self.state, self.reason = state, reason
        self.present_since = None

    def _checkpoint(self, now: float) -> None:
        """Book the open session's time so far, so a crash loses at most one checkpoint."""
        assert self.session_start is not None
        length = now - self.session_start
        self.usage.add(length - self._booked)
        self._booked = length
        self._checkpoint_at = now
