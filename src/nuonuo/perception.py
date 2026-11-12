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
    # Detection cadence (vision.py). Face detection runs every detect_min..detect_max seconds
    # instead of every frame: back to the minimum whenever something changes, and stretched by
    # detect_slowdown after each calm detection. Readings are held between detections.
    "detect_min_seconds": 0.15,
    "detect_max_seconds": 0.5,
    "detect_slowdown": 1.25,
    "detect_motion": 0.05,  # face centre moving this far (in -1..1 frame units) counts as a change
    "gesture_every": 4,  # run the gesture recognizer on every Nth detection
    "gesture_boost_with_hand": True,  # ...and on every detection while a hand is in view, so waves still register
    "face_hold_seconds": 0.75,  # keep the last face position through a missed detection
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
        elif st["frowning"] or st["absent"]:
            self.mood = "sad"
        elif st["tired"]:
            self.mood = "sleepy"
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
        hand_fresh: bool = True,
    ) -> PerceptionEvent | None:
        """hand_fresh=False means the gesture recognizer did not run on this frame.
        Gesture tracking then keeps its last state instead of treating the hand as gone."""
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
        if hand_fresh:
            kinds += self._gestures(t, gesture, wrist_x)
        self._update_mood()
        kinds = [k for k in kinds if not (k.endswith("_end") and STATE_PHRASES[k[:-4]][1] is None)]
        if not kinds:
            return None
        text = self._phrase(kinds)
        if not text:
            return None
        return PerceptionEvent(text=text, kinds=kinds, salient=any(k in SALIENT for k in kinds))


class DetectionScheduler:
    """When vision.py runs face detection, and when it runs the gesture recognizer too.

    Detection runs every detect_min..detect_max seconds instead of every frame. A detection
    where something changed (the face appeared or left, moved, changed state, or raised an
    event) sets the next interval to the minimum. A calm one stretches it by detect_slowdown,
    up to the maximum. Gestures run on every gesture_every-th detection, and on every detection
    while a hand is in view so a wave still gets enough wrist samples.
    """

    def __init__(self, cfg: dict | None = None):
        self.cfg = dict(CONFIG) if cfg is None else cfg
        self.interval = float(self.cfg["detect_min_seconds"])
        self.next_at = float("-inf")
        self.count = 0
        self.hand = False
        self._face: tuple[float, float] | None = None
        self._label: str | None = None

    def due(self, now: float) -> bool:
        return now >= self.next_at

    def gestures_due(self) -> bool:
        if self.hand and self.cfg["gesture_boost_with_hand"]:
            return True
        return self.count % max(1, int(self.cfg["gesture_every"])) == 0

    def done(
        self,
        now: float,
        face: tuple[float, float] | None,
        label: str,
        event: bool = False,
        hand: bool | None = None,
    ) -> float:
        """Record a finished detection and schedule the next. hand is None if gestures did not run."""
        c = self.cfg
        active = event or label != self._label or (face is None) != (self._face is None)
        if face is not None and self._face is not None:
            active = active or math.dist(face, self._face) > c["detect_motion"]
        self._face, self._label = face, label
        if hand is not None:
            self.hand = hand
        lo, hi = float(c["detect_min_seconds"]), float(c["detect_max_seconds"])
        if active or (self.hand and c["gesture_boost_with_hand"]):
            self.interval = lo
        else:
            self.interval = min(hi, max(lo, self.interval * float(c["detect_slowdown"])))
        self.count += 1
        self.next_at = now + self.interval
        return self.interval


class ReadingHold:
    """Hold the last face position through a short gap so the avatar's gaze does not jump."""

    def __init__(self, hold_seconds: float):
        self.hold_seconds = hold_seconds
        self.value: tuple[float, float] | None = None
        self.at = float("-inf")

    def update(self, now: float, value: tuple[float, float] | None) -> tuple[float, float] | None:
        if value is not None:
            self.value, self.at = value, now
        elif self.value is not None and now - self.at > self.hold_seconds:
            self.value = None
        return self.value

    def clear(self) -> None:
        self.value = None
        self.at = float("-inf")
