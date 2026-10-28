import math
from collections import deque
from dataclasses import dataclass, field

CONFIG = {
    "ema_seconds": 1.0,
    "hold_seconds": 0.4,
    "smile": 0.45,
    "frown": 0.35,
    "surprise_jaw": 0.35,
    "surprise_brow": 0.45,
    "tired_eyes": 0.45,
    "blink_closed": 0.6,
    "slow_blink_seconds": 0.4,
    "slow_blink_count": 2,
    "slow_blink_window": 10.0,
    "look_away_yaw_deg": 25.0,
    "look_away_eye": 0.55,
    "look_down_pitch_deg": -15.0,
    "pose_ema_seconds": 0.25,
    "absent_seconds": 2.0,
    "gesture_hold_seconds": 0.3,
    "gesture_cooldown_seconds": 3.0,
    "wave_window_seconds": 1.2,
    "wave_min_amplitude": 0.03,
    "wave_min_reversals": 2,
    "wave_cooldown_seconds": 4.0,
}

GESTURE_PHRASES = {
    "Thumb_Up": "gave a thumbs up",
    "Thumb_Down": "gave a thumbs down",
    "Victory": "made a peace sign",
    "Open_Palm": "held up an open palm",
    "Closed_Fist": "made a fist",
    "Pointing_Up": "pointed up",
    "ILoveYou": "made the I-love-you sign",
}

STATE_PHRASES = {
    "smiling": ("smiled", None),
    "frowning": ("frowned", None),
    "surprised": ("looks surprised", None),
    "tired": ("looks tired", None),
    "looking_away": ("looked away", "looked back"),
    "looking_down": ("looked down at their phone", "looked up"),
    "absent": ("left", "came back"),
}

SALIENT = {"wave", "absent_end", "ILoveYou", "Thumb_Up", "Victory"}


@dataclass
class PerceptionEvent:
    text: str
    kinds: list[str]
    salient: bool


@dataclass
class _Debounced:
    value: bool = False
    candidate: bool = False
    since: float = 0.0


@dataclass
class Perception:
    user_name: str = "Andy"
    cfg: dict = field(default_factory=lambda: dict(CONFIG))

    def __post_init__(self):
        self.smooth: dict[str, float] = {}
        self.last_t: float | None = None
        self.states = {k: _Debounced() for k in STATE_PHRASES}
        self.last_face_t: float | None = None
        self.blink_start: float | None = None
        self.slow_blinks: deque[float] = deque()
        self.gesture_candidate: str | None = None
        self.gesture_since = 0.0
        self.gesture_last_emit: dict[str, float] = {}
        self.wrist_hist: deque[tuple[float, float]] = deque()
        self.last_wave = -1e9
        self.mood = "neutral"
        self.yaw_smooth: float | None = None
        self.pitch_smooth: float | None = None

    def _ema(self, blend: dict[str, float], dt: float) -> None:
        a = 1.0 - math.exp(-dt / max(self.cfg["ema_seconds"], 1e-3)) if self.smooth else 1.0
        for k, v in blend.items():
            self.smooth[k] = self.smooth.get(k, v) + a * (v - self.smooth.get(k, v))

    def s(self, name: str) -> float:
        return self.smooth.get(name, 0.0)

    def _pair(self, base: str) -> float:
        return (self.s(base + "Left") + self.s(base + "Right")) / 2

    def _track_blinks(self, t: float, blend: dict[str, float]) -> None:
        raw = (blend.get("eyeBlinkLeft", 0.0) + blend.get("eyeBlinkRight", 0.0)) / 2
        if raw >= self.cfg["blink_closed"]:
            if self.blink_start is None:
                self.blink_start = t
        elif self.blink_start is not None:
            if t - self.blink_start >= self.cfg["slow_blink_seconds"]:
                self.slow_blinks.append(t)
            self.blink_start = None
        while self.slow_blinks and t - self.slow_blinks[0] > self.cfg["slow_blink_window"]:
            self.slow_blinks.popleft()

    def _smooth_pose(self, yaw: float | None, pitch: float | None, dt: float) -> tuple[float | None, float | None]:
        tau = max(float(self.cfg["pose_ema_seconds"]), 1e-3)
        a = 1.0 - math.exp(-dt / tau)

        def step(prev: float | None, value: float | None) -> float | None:
            if value is None:
                return prev
            if prev is None:
                return float(value)
            return prev + a * (float(value) - prev)

        self.yaw_smooth = step(self.yaw_smooth, yaw)
        self.pitch_smooth = step(self.pitch_smooth, pitch)
        return self.yaw_smooth, self.pitch_smooth

    def _raw_states(self, t: float, face: bool, yaw: float | None, pitch: float | None) -> dict[str, bool]:
        c = self.cfg
        absent = (not face) and self.last_face_t is not None and t - self.last_face_t >= c["absent_seconds"]
        if not face:
            return {"absent": absent} | {k: False for k in STATE_PHRASES if k != "absent"}
        eye_out = max(
            (self.s("eyeLookOutLeft") + self.s("eyeLookInRight")) / 2,
            (self.s("eyeLookInLeft") + self.s("eyeLookOutRight")) / 2,
            self._pair("eyeLookUp"),
            self._pair("eyeLookDown"),
        )
        smile = self._pair("mouthSmile")
        surprised = self.s("jawOpen") > c["surprise_jaw"] and (
            self.s("browInnerUp") > c["surprise_brow"] or self._pair("eyeWide") > c["surprise_brow"]
        )
        looking_down = pitch is not None and pitch <= c["look_down_pitch_deg"]
        looking_away = (not looking_down) and (
            (yaw is not None and abs(yaw) > c["look_away_yaw_deg"]) or eye_out > c["look_away_eye"]
        )
        return {
            "absent": False,
            "smiling": smile > c["smile"],
            "frowning": smile < c["smile"] * 0.5
            and (self._pair("mouthFrown") > c["frown"] or self._pair("browDown") > c["frown"] + 0.15),
            "surprised": surprised,
            "tired": (c["tired_eyes"] < self._pair("eyeBlink") < c["blink_closed"] + 0.25)
            or len(self.slow_blinks) >= c["slow_blink_count"],
            "looking_away": looking_away,
            "looking_down": looking_down,
        }

    def state_label(self) -> str:
        if self.last_face_t is None:
            return "unknown"
        if self.states["absent"].value:
            return "absent"
        active = sorted(name for name in STATE_PHRASES if name != "absent" and self.states[name].value)
        return "+".join(active) if active else "present"

    def _debounce(self, t: float, raw: dict[str, bool]) -> list[str]:
        changes = []
        for k, v in raw.items():
            d = self.states[k]
            if v != d.candidate:
                d.candidate, d.since = v, t
            hold = 0.0 if k == "absent" else self.cfg["hold_seconds"]
            if d.candidate != d.value and t - d.since >= hold:
                d.value = d.candidate
                changes.append(k if d.value else k + "_end")
        return changes

    def _gestures(self, t: float, gesture: str | None, wrist_x: float | None) -> list[str]:
        c = self.cfg
        out = []
        if gesture == "Open_Palm" and wrist_x is not None:
            self.wrist_hist.append((t, wrist_x))
        else:
            self.wrist_hist.clear()
        while self.wrist_hist and t - self.wrist_hist[0][0] > c["wave_window_seconds"]:
            self.wrist_hist.popleft()
        if self._is_wave() and t - self.last_wave > c["wave_cooldown_seconds"]:
            self.last_wave = t
            self.gesture_last_emit["Open_Palm"] = t
            self.wrist_hist.clear()
            out.append("wave")

        g = gesture if gesture in GESTURE_PHRASES else None
        if g != self.gesture_candidate:
            self.gesture_candidate, self.gesture_since = g, t
        elif g and t - self.gesture_since >= c["gesture_hold_seconds"]:
            if t - self.gesture_last_emit.get(g, -1e9) > c["gesture_cooldown_seconds"]:
                if not (g == "Open_Palm" and len(self.wrist_hist) > 2 and self._x_range() > c["wave_min_amplitude"]):
                    self.gesture_last_emit[g] = t
                    out.append(g)
        return out

    def _x_range(self) -> float:
        xs = [x for _, x in self.wrist_hist]
        return max(xs) - min(xs) if xs else 0.0

    def _is_wave(self) -> bool:
        if len(self.wrist_hist) < 5 or self._x_range() < self.cfg["wave_min_amplitude"]:
            return False
        amp = self.cfg["wave_min_amplitude"] / 2
        reversals, direction, anchor = 0, 0, self.wrist_hist[0][1]
        for _, x in self.wrist_hist:
            d = x - anchor
            if abs(d) >= amp:
                nd = 1 if d > 0 else -1
                if direction and nd != direction:
                    reversals += 1
                direction, anchor = nd, x
            elif direction and (x - anchor) * direction > 0:
                anchor = x
        return reversals >= self.cfg["wave_min_reversals"]

    def _phrase(self, kinds: list[str]) -> str:
        parts = []
        for k in kinds:
            if k == "wave":
                parts.append("waved")
            elif k in GESTURE_PHRASES:
                parts.append(GESTURE_PHRASES[k])
            elif k.endswith("_end"):
                end = STATE_PHRASES[k[:-4]][1]
                if end:
                    parts.append(end)
            else:
                parts.append(STATE_PHRASES[k][0])
        if not parts:
            return ""
        body = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]
        verb = "just " if not body.startswith("looks") else ""
        return f"[{self.user_name} {verb}{body}]"

    def _update_mood(self) -> None:
        st = {k: d.value for k, d in self.states.items()}
        if st["surprised"]:
            self.mood = "surprised"
        elif st["smiling"]:
            self.mood = "happy"
        elif st["frowning"] or st["tired"] or st["absent"]:
            self.mood = "sad"
        else:
            self.mood = "neutral"

    def update(
        self,
        t: float,
        blend: dict[str, float] | None,
        yaw: float | None = None,
        gesture: str | None = None,
        wrist_x: float | None = None,
        pitch: float | None = None,
    ) -> PerceptionEvent | None:
        dt = 0.033 if self.last_t is None else max(t - self.last_t, 1e-3)
        self.last_t = t
        face = blend is not None
        if face:
            self.last_face_t = t
            self._ema(blend, dt)
            self._track_blinks(t, blend)
            yaw, pitch = self._smooth_pose(yaw, pitch, dt)
        else:
            self.yaw_smooth = None
            self.pitch_smooth = None
            yaw = pitch = None
        kinds = self._debounce(t, self._raw_states(t, face, yaw, pitch))
        kinds += self._gestures(t, gesture, wrist_x)
        self._update_mood()
        kinds = [k for k in kinds if not (k.endswith("_end") and STATE_PHRASES[k[:-4]][1] is None)]
        if not kinds:
            return None
        text = self._phrase(kinds)
        if not text:
            return None
        return PerceptionEvent(text=text, kinds=kinds, salient=any(k in SALIENT for k in kinds))
