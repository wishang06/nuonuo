"""Clip files and the avatar state machine. Frame decoding stays out of the tests."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

TALK_RELEASE_S = 0.15
IDLE_VARIANT_MIN = 5.0
IDLE_VARIANT_MAX = 15.0
CROSSFADE_FRAMES = 4
MOUTH_BUCKETS = 5
NOISE_FLOOR = 0.04
ATTACK_S = 0.030
RELEASE_S = 0.080
WINDOW_S = 0.040


@dataclass
class ClipSpec:
    name: str
    file: Path
    loop: bool
    state: str


@dataclass
class ClipConfig:
    mouth_box: tuple[int, int, int, int]
    buckets: int
    clips: list[ClipSpec]

    def by_state(self) -> dict[str, list[str]]:
        grouped: dict[str, list[str]] = {}
        for clip in self.clips:
            grouped.setdefault(clip.state, []).append(clip.name)
        return grouped

    def named(self, name: str) -> ClipSpec:
        for clip in self.clips:
            if clip.name == name:
                return clip
        raise KeyError(name)


def load_clips_config(path: Path) -> ClipConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "clips" not in data:
        raise ValueError("clips.yaml needs a clips list")
    mouth = data.get("mouth") or {}
    box = mouth.get("box") or [0, 0, 1, 1]
    if len(box) != 4:
        raise ValueError("mouth.box must be x, y, w, h")
    clips = []
    root = Path(path).parent
    for item in data["clips"]:
        clips.append(
            ClipSpec(
                name=str(item["name"]),
                file=(root / str(item["file"])).resolve(),
                loop=bool(item.get("loop", False)),
                state=str(item["state"]),
            )
        )
    if not clips:
        raise ValueError("clips.yaml has no clips")
    return ClipConfig(
        mouth_box=tuple(int(v) for v in box),
        buckets=int(mouth.get("buckets", MOUTH_BUCKETS)),
        clips=clips,
    )


@dataclass
class ClipFrames:
    name: str
    state: str
    loop: bool
    durations: list[float]
    frames: list[np.ndarray] = field(repr=False)

    @property
    def total(self) -> float:
        return float(sum(self.durations))

    def index_at(self, elapsed: float) -> int:
        count = len(self.durations)
        if count == 0:
            return 0
        if elapsed <= 0:
            return 0
        if not self.loop and elapsed >= self.total:
            return count - 1
        cursor = elapsed % self.total if self.loop else elapsed
        acc = 0.0
        for index, duration in enumerate(self.durations):
            acc += duration
            if cursor < acc:
                return index
        return count - 1


def load_frames(path: Path) -> tuple[list[np.ndarray], list[float]]:
    suffix = path.suffix.lower()
    if suffix == ".gif":
        return _load_gif(path)
    return _load_video(path)


def _load_gif(path: Path) -> tuple[list[np.ndarray], list[float]]:
    from PIL import Image

    image = Image.open(path)
    frames: list[np.ndarray] = []
    durations: list[float] = []
    index = 0
    while True:
        frame = image.convert("RGB")
        frames.append(np.asarray(frame))
        durations.append(max(0.01, float(image.info.get("duration", 100)) / 1000.0))
        index += 1
        try:
            image.seek(index)
        except EOFError:
            break
    return _same_size(frames), durations


def _load_video(path: Path) -> tuple[list[np.ndarray], list[float]]:
    import imageio.v3 as iio

    meta = iio.immeta(path)
    fps = float(meta.get("fps") or 24.0)
    duration = 1.0 / fps if fps else 1.0 / 24.0
    frames = [np.asarray(frame)[..., :3] for frame in iio.imiter(path)]
    return _same_size(frames), [duration] * len(frames)


def _same_size(frames: list[np.ndarray]) -> list[np.ndarray]:
    if not frames:
        return frames
    height, width = frames[0].shape[:2]
    fitted = []
    for frame in frames:
        if frame.shape[0] != height or frame.shape[1] != width:
            from PIL import Image

            image = Image.fromarray(frame).resize((width, height))
            frame = np.asarray(image)
        fitted.append(np.ascontiguousarray(frame))
    return fitted


def resize_frames(frames: list[np.ndarray], size: tuple[int, int]) -> list[np.ndarray]:
    width, height = size
    if not frames or (frames[0].shape[1] == width and frames[0].shape[0] == height):
        return frames
    from PIL import Image

    return [
        np.ascontiguousarray(Image.fromarray(frame).resize((width, height)))
        for frame in frames
    ]


class ClipStateMachine:
    """idle loops, talking follows the voice, expressions play once."""

    def __init__(
        self,
        by_state: dict[str, list[str]],
        durations: dict[str, float],
        rng: random.Random | None = None,
        now: float = 0.0,
    ):
        self.by_state = by_state
        self.durations = durations
        self.rng = rng or random.Random()
        self.requested = "idle"
        self._talk_until = -1.0
        self._latched: str | None = None
        self._shot: str | None = None
        self._shot_until = -1.0
        self._variant: str | None = None
        self._variant_until = -1.0
        self._next_variant = now + self.rng.uniform(IDLE_VARIANT_MIN, IDLE_VARIANT_MAX)

    def play_state(self, state: str, now: float) -> None:
        self.requested = state or "idle"
        if self.requested == "talking":
            self._talk_until = now + TALK_RELEASE_S

    def set_expression(self, name: str | None, now: float) -> None:
        if not name or name in ("neutral", "sad"):
            self._latched = None
            return
        if name == self._latched:
            return
        self._latched = name
        clip = self._first(name)
        if clip is None:
            return
        self._shot = clip
        self._shot_until = now + self.durations.get(clip, 0.0)

    def clip_at(self, now: float) -> str:
        if self._shot and now < self._shot_until:
            return self._shot
        if now < self._talk_until and self._first("talking"):
            return self._first("talking")
        if self.requested == "listening" and self._first("listening"):
            return self._first("listening")
        idles = self.by_state.get("idle") or []
        if not idles:
            return self._shot or "idle"
        if self._variant and now < self._variant_until:
            return self._variant
        if len(idles) > 1 and now >= self._next_variant:
            self._variant = self.rng.choice(idles[1:])
            self._variant_until = now + self.durations.get(self._variant, 0.0)
            self._next_variant = self._variant_until + self.rng.uniform(IDLE_VARIANT_MIN, IDLE_VARIANT_MAX)
            return self._variant
        return idles[0]

    def _first(self, state: str) -> str | None:
        names = self.by_state.get(state) or []
        return names[0] if names else None


def crossfade(elapsed: float, frame_dt: float, frames: int = CROSSFADE_FRAMES) -> float:
    window = max(frame_dt, 1e-3) * frames
    if elapsed <= 0:
        return 0.0
    if elapsed >= window:
        return 1.0
    return elapsed / window


def openness(rgb: np.ndarray, box: tuple[int, int, int, int]) -> float:
    x, y, width, height = box
    crop = rgb[y : y + height, x : x + width]
    if crop.size == 0:
        return 0.0
    red = crop[..., 0].astype(np.int16)
    green = crop[..., 1].astype(np.int16)
    blue = crop[..., 2].astype(np.int16)
    dark = (red < 110) & (green < 100) & (blue < 100)
    warm = (red > 140) & (red > green + 25) & (red > blue + 15)
    return float((dark | warm).mean())


def assign_buckets(scores: list[float], count: int = MOUTH_BUCKETS) -> list[list[int]]:
    buckets: list[list[int]] = [[] for _ in range(max(1, count))]
    order = sorted(range(len(scores)), key=lambda index: (scores[index], index))
    if not order:
        return buckets
    span = len(buckets)
    for rank, index in enumerate(order):
        slot = min(span - 1, rank * span // len(order))
        buckets[slot].append(index)
    return buckets


def bucket_for_level(level: float, count: int = MOUTH_BUCKETS, floor: float = NOISE_FLOOR) -> int:
    if level <= floor:
        return 0
    span = max(1e-6, 1.0 - floor)
    scaled = (level - floor) / span * count
    return min(count - 1, max(0, int(scaled)))


def step_level(previous: float, sample: float, dt: float, floor: float = NOISE_FLOOR) -> float:
    target = 0.0 if sample < floor else sample
    tau = ATTACK_S if target > previous else RELEASE_S
    if dt <= 0:
        return previous
    weight = 1.0 - np.exp(-dt / tau)
    return previous + weight * (target - previous)


class MouthEnvelope:
    def __init__(self, buckets: int = MOUTH_BUCKETS):
        self.buckets = buckets
        self.level = 0.0
        self._window: list[float] = []

    def push(self, sample: float, dt: float) -> float:
        cap = max(1, round(WINDOW_S / max(dt, 1e-3)))
        self._window.append(sample)
        if len(self._window) > cap:
            self._window = self._window[-cap:]
        averaged = sum(self._window) / len(self._window)
        self.level = step_level(self.level, averaged, dt)
        return self.level

    def bucket(self) -> int:
        return bucket_for_level(self.level, self.buckets)


def pick_bucket_frame(buckets: list[list[int]], slot: int, tick: int) -> int:
    chosen = buckets[slot] if slot < len(buckets) and buckets[slot] else []
    if not chosen:
        for group in buckets:
            if group:
                chosen = group
                break
    if not chosen:
        return 0
    return chosen[tick % len(chosen)]
