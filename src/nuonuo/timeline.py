"""Behaviour spans for one session, with older time folded into aggregates."""

from __future__ import annotations

import threading
from dataclasses import dataclass

RECENT_MS = 60_000
HISTORY_MS = 10 * 60_000
MINUTE_MS = 60_000

_EXPRESSIONS = (
    ("frowning", "frowning"),
    ("smiling", "smiling"),
    ("surprised", "surprised"),
    ("tired", "tired"),
)


@dataclass
class Span:
    state: str
    start_ms: int
    end_ms: int


def _flags(state: str) -> set[str]:
    if state in ("", "present", "neutral", "unknown"):
        return set()
    return {part for part in state.split("+") if part and part not in ("present", "neutral", "unknown")}


def _has(state: str, flag: str) -> bool:
    return flag in _flags(state)


class BehaviourTimeline:
    def __init__(self, recent_ms: int = RECENT_MS, history_ms: int = HISTORY_MS):
        self.recent_ms = recent_ms
        self.history_ms = history_ms
        self.spans: list[Span] = []
        self.minutes: dict[int, dict[str, int]] = {}
        self.away_opens: dict[int, int] = {}
        self.down_opens: dict[int, int] = {}
        self.minute_end_state: dict[int, str] = {}
        self._tail_away = False
        self._tail_down = False
        self._lock = threading.Lock()

    def observe(self, t_ms: int, state: str) -> None:
        if not state or state == "unknown":
            return
        t_ms = max(0, int(t_ms))
        with self._lock:
            if self.spans and self.spans[-1].state == state:
                if t_ms >= self.spans[-1].end_ms:
                    self.spans[-1].end_ms = t_ms
            elif not self.spans or t_ms >= self.spans[-1].end_ms:
                self.spans.append(Span(state, t_ms, t_ms))
            self._compact(t_ms)

    def recent_spans(self) -> list[Span]:
        with self._lock:
            return [Span(s.state, s.start_ms, s.end_ms) for s in self.spans]

    def minute_count(self) -> int:
        with self._lock:
            return len(self.minutes)

    def current_note(self, now_ms: int | None = None) -> str:
        del now_ms
        spans = self.recent_spans()
        if not spans:
            return ""
        state = spans[-1].state
        flags = _flags(state)
        if state == "absent" or "absent" in flags:
            return "Not in frame"
        parts: list[str] = []
        if "looking_down" in flags:
            parts.append("Looking down at phone")
        elif "looking_away" in flags:
            parts.append("Looking away")
        else:
            parts.append("Looking at you")
        labels = {
            "frowning": "Frowning",
            "smiling": "Smiling",
            "surprised": "Surprised",
            "tired": "Tired",
        }
        for key, label in labels.items():
            if key in flags:
                parts.append(label)
        return ". ".join(parts)

    def aggregate_text(self, now_ms: int) -> str:
        durations, observed, away, down = self._stats(now_ms)
        if observed <= 0:
            return "No camera readings yet."
        mins = min(10, max(1, round(observed / MINUTE_MS)))
        parts = []
        for flag, label in _EXPRESSIONS:
            pct = round(100 * durations.get(flag, 0) / observed)
            if pct > 0:
                parts.append(f"{label} {pct}%")
        if "looking_down" in durations and durations["looking_down"] > 0:
            pct = round(100 * durations["looking_down"] / observed)
            if pct > 0:
                parts.append(f"looking down {pct}%")
        head = (", ".join(parts) if parts else "no strong expression") + f" of last {mins} min"
        bits = [head]
        if away:
            bits.append(_count_phrase("away", away))
        if down:
            bits.append(_count_phrase("down", down))
        return ", ".join(bits)

    def aggregate_record(self, now_ms: int) -> dict:
        durations, observed, away, down = self._stats(now_ms)
        smile = round(100 * durations.get("smiling", 0) / observed) if observed else 0
        return {
            "text": self.aggregate_text(now_ms),
            "smile_pct": smile,
            "away_count": away,
            "down_count": down,
            "observed_ms": observed,
            "window_ms": min(self.history_ms, now_ms),
        }

    def _stats(self, now_ms: int) -> tuple[dict[str, int], int, int, int]:
        with self._lock:
            durations: dict[str, int] = {}
            observed = 0
            for bucket in self.minutes.values():
                observed += bucket.get("__ms", 0)
                for key, value in bucket.items():
                    if key == "__ms":
                        continue
                    durations[key] = durations.get(key, 0) + value
            for span in self.spans:
                dur = max(0, span.end_ms - span.start_ms)
                observed += dur
                for flag in _flags(span.state):
                    durations[flag] = durations.get(flag, 0) + dur
            away = sum(self.away_opens.values())
            down = sum(self.down_opens.values())
            prev_away = self._tail_away
            prev_down = self._tail_down
            for span in self.spans:
                away_now = _has(span.state, "looking_away")
                down_now = _has(span.state, "looking_down")
                if away_now and not prev_away:
                    away += 1
                if down_now and not prev_down:
                    down += 1
                prev_away = away_now
                prev_down = down_now
            return durations, observed, away, down

    def _compact(self, now: int) -> None:
        recent_start = now - self.recent_ms
        kept: list[Span] = []
        for span in self.spans:
            if span.end_ms <= span.start_ms:
                if span.start_ms >= recent_start:
                    kept.append(span)
                continue
            if span.end_ms <= recent_start:
                self._fold(span)
            elif span.start_ms < recent_start:
                self._fold(Span(span.state, span.start_ms, recent_start))
                kept.append(Span(span.state, recent_start, span.end_ms))
            else:
                kept.append(span)
        self.spans = kept
        self._expire(now)

    def _fold(self, span: Span) -> None:
        if span.end_ms <= span.start_ms:
            return
        away = _has(span.state, "looking_away")
        down = _has(span.state, "looking_down")
        minute = (span.start_ms // MINUTE_MS) * MINUTE_MS
        if away and not self._tail_away:
            self.away_opens[minute] = self.away_opens.get(minute, 0) + 1
        if down and not self._tail_down:
            self.down_opens[minute] = self.down_opens.get(minute, 0) + 1
        self._tail_away = away
        self._tail_down = down
        flags = _flags(span.state)
        cursor = span.start_ms
        while cursor < span.end_ms:
            bucket_start = (cursor // MINUTE_MS) * MINUTE_MS
            end = min(bucket_start + MINUTE_MS, span.end_ms)
            bucket = self.minutes.setdefault(bucket_start, {})
            dur = end - cursor
            bucket["__ms"] = bucket.get("__ms", 0) + dur
            for flag in flags:
                bucket[flag] = bucket.get(flag, 0) + dur
            self.minute_end_state[bucket_start] = span.state
            cursor = end

    def _expire(self, now: int) -> None:
        cutoff = now - self.history_ms

        def keep(minute: int) -> bool:
            return minute + MINUTE_MS > cutoff

        self.minutes = {m: b for m, b in self.minutes.items() if keep(m)}
        self.away_opens = {m: n for m, n in self.away_opens.items() if keep(m)}
        self.down_opens = {m: n for m, n in self.down_opens.items() if keep(m)}
        self.minute_end_state = {m: s for m, s in self.minute_end_state.items() if keep(m)}
        if self.minute_end_state:
            latest = max(self.minute_end_state)
            state = self.minute_end_state[latest]
            self._tail_away = _has(state, "looking_away")
            self._tail_down = _has(state, "looking_down")
        else:
            self._tail_away = False
            self._tail_down = False


def _count_phrase(label: str, count: int) -> str:
    if count == 1:
        return f"{label} once"
    if count == 2:
        return f"{label} twice"
    return f"{label} {count} times"


def read_camera(timeline: BehaviourTimeline, now_ms: int) -> dict:
    """Current state plus a compact timeline summary. No landmarks."""
    current = timeline.current_note(now_ms) or "No face yet"
    return {"current": current, "summary": timeline.aggregate_text(now_ms)}
