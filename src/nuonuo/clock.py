"""One session clock. Every t_ms is milliseconds since this clock started."""

import time


def format_clock(elapsed_ms: int) -> str:
    total = max(0, int(elapsed_ms) // 1000)
    hours, rem = divmod(total, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"


class SessionClock:
    def __init__(self, start: float | None = None):
        self.start = time.monotonic() if start is None else float(start)

    def t_ms(self, now: float | None = None) -> int:
        moment = time.monotonic() if now is None else now
        return max(0, int((moment - self.start) * 1000))

    def format(self, now: float | None = None) -> str:
        return format_clock(self.t_ms(now))
