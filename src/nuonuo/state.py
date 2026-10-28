import threading
import time
from dataclasses import dataclass, field


@dataclass
class SharedState:
    mood: str = "neutral"
    speaking: bool = False
    mouth_level: float = 0.0
    vision_enabled: bool = True
    preview: bool = False
    mode: str = "chatty"
    status: str = "starting"
    last_event: str = ""
    face_x: float | None = None
    face_y: float | None = None
    user_speaking: bool = False
    stop: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)
    last_audio_ts: float = 0.0

    def set(self, **kw):
        with self.lock:
            for k, v in kw.items():
                setattr(self, k, v)

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "mood": self.mood,
                "speaking": self.speaking or (time.monotonic() - self.last_audio_ts < 0.15),
                "mouth_level": self.mouth_level,
                "vision_enabled": self.vision_enabled,
                "preview": self.preview,
                "mode": self.mode,
                "status": self.status,
                "last_event": self.last_event,
                "face_x": self.face_x,
                "face_y": self.face_y,
                "user_speaking": self.user_speaking,
            }
