import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

RATES = {
    "mini": {"audio_in": 10.0, "audio_out": 20.0, "cached_audio_in": 0.30, "text_in": 0.60, "text_out": 2.40, "cached_text_in": 0.60},
    "full": {"audio_in": 32.0, "audio_out": 64.0, "cached_audio_in": 0.40, "text_in": 4.00, "text_out": 24.00, "cached_text_in": 0.40},
}


def rates_for(model: str) -> dict:
    return RATES["mini" if "mini" in model else "full"]


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


class Usage:
    def __init__(self):
        self.text_in = self.audio_in = self.cached_text_in = self.cached_audio_in = 0
        self.text_out = self.audio_out = 0
        self.responses = 0

    def add(self, usage: dict) -> None:
        if not usage:
            return
        self.responses += 1
        inp = usage.get("input_token_details") or {}
        out = usage.get("output_token_details") or {}
        cached = inp.get("cached_tokens_details") or {}
        c_text, c_audio = cached.get("text_tokens", 0), cached.get("audio_tokens", 0)
        if not cached and inp.get("cached_tokens"):
            c_text = inp["cached_tokens"]
        self.cached_text_in += c_text
        self.cached_audio_in += c_audio
        self.text_in += max(inp.get("text_tokens", 0) - c_text, 0)
        self.audio_in += max(inp.get("audio_tokens", 0) - c_audio, 0)
        self.text_out += out.get("text_tokens", 0)
        self.audio_out += out.get("audio_tokens", 0)

    def cost(self, model: str) -> float:
        r = rates_for(model)
        return (
            self.audio_in * r["audio_in"]
            + self.audio_out * r["audio_out"]
            + self.cached_audio_in * r["cached_audio_in"]
            + self.text_in * r["text_in"]
            + self.cached_text_in * r["cached_text_in"]
            + self.text_out * r["text_out"]
        ) / 1_000_000

    def as_dict(self) -> dict:
        return {k: v for k, v in vars(self).items()}


class SessionLogger:
    def __init__(self, log_dir: Path, model: str):
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.reaction_path = log_dir / f"reactions-{stamp}.jsonl"
        self.session_path = log_dir / f"session-{stamp}.json"
        self.model = model
        self.usage = Usage()
        self.start_wall = _now()
        self.start = time.monotonic()
        self.events: dict[str, dict] = {}
        self.flushed = 0
        self.lock = threading.Lock()

    def _flush(self, force: bool = False) -> None:
        cutoff = time.time() - 8
        done = [k for k, e in self.events.items()
                if force or e["response_followed"] or e["_t"] < cutoff]
        if not done:
            return
        with self.reaction_path.open("a", encoding="utf-8") as f:
            for k in done:
                e = self.events.pop(k)
                e.pop("_t", None)
                f.write(json.dumps(e) + "\n")
        self.flushed += len(done)

    def vision_event(self, event_id: str, text: str, salient: bool, requested: bool) -> None:
        with self.lock:
            self._flush()
            self.events[event_id] = {"_t": time.time(),
                "ts": _now(), "id": event_id, "text": text, "salient": salient,
                "response_requested": requested, "response_followed": False,
            }

    def response_started(self) -> None:
        with self.lock:
            cutoff = time.time() - 8
            for e in self.events.values():
                if e["_t"] >= cutoff:
                    e["response_followed"] = True
            self._flush()

    def add_usage(self, usage: dict) -> None:
        with self.lock:
            self.usage.add(usage)

    def close(self) -> dict:
        with self.lock:
            self._flush(force=True)
            summary = {
                "model": self.model,
                "start": self.start_wall,
                "end": _now(),
                "duration_seconds": round(time.monotonic() - self.start, 1),
                "usage": self.usage.as_dict(),
                "estimated_cost_usd": round(self.usage.cost(self.model), 5),
                "rates_per_million": rates_for(self.model),
                "vision_events": self.flushed,
            }
            self.session_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            return summary
