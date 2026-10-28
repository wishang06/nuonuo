"""Sprite avatar. Clips in assets/anim, mouth frames chosen from the voice level.

SpriteAvatar implements set_mouth_level, play_state, look_at, and set_expression.
A Live2D, Inochi2D, or LivePortrait renderer can replace it without changing the app.
"""

import argparse
import math
import os
import sys
import time
import wave
from pathlib import Path
from typing import Protocol

import numpy as np
import pygame

from nuonuo.clips import (
    ClipFrames,
    ClipStateMachine,
    MouthEnvelope,
    assign_buckets,
    crossfade,
    load_clips_config,
    load_frames,
    openness,
    pick_bucket_frame,
    resize_frames,
)
from nuonuo.state import SharedState

DISPLAY_WIDTH = 320
STATUS_H = 52
MAX_LOOK_PX = 6
TILT_DEG = 3.0
LOOK_TAU = 0.35


class AvatarRenderer(Protocol):
    def set_mouth_level(self, level: float) -> None: ...

    def play_state(self, state: str) -> None: ...

    def look_at(self, x: float | None, y: float | None) -> None: ...

    def set_expression(self, name: str) -> None: ...


def anim_dir() -> Path:
    candidates = [
        Path(__file__).resolve().parents[2] / "assets" / "anim",
        Path.cwd() / "assets" / "anim",
    ]
    for path in candidates:
        if (path / "clips.yaml").is_file():
            return path
    return candidates[0]


def _smooth(current: float, target: float, dt: float, tau: float) -> float:
    if dt <= 0:
        return current
    weight = 1.0 - math.exp(-dt / max(tau, 1e-3))
    return current + weight * (target - current)


class SpriteAvatar:
    def __init__(self, clips: dict[str, ClipFrames], surfaces: dict[str, list[pygame.Surface]], buckets: list[list[int]], talk_name: str | None):
        self.clips = clips
        self.surfaces = surfaces
        self.buckets = buckets
        self.talk_name = talk_name
        durations = {name: clip.total for name, clip in clips.items()}
        by_state: dict[str, list[str]] = {}
        for clip in clips.values():
            by_state.setdefault(clip.state, []).append(clip.name)
        self.machine = ClipStateMachine(by_state, durations, now=time.monotonic())
        self.mouth = MouthEnvelope(buckets=max(1, len(buckets)))
        self._sample = 0.0
        self._state = "idle"
        self._expression = "neutral"
        self._look = (0.0, 0.0)
        self._look_target = (0.0, 0.0)
        self._now = time.monotonic()
        self._last = self._now
        self._clip = ""
        self._clip_at = self._now
        self._fade_from: pygame.Surface | None = None
        self._fade_at = 0.0
        self.bucket = 0
        self.debug = ""
        sample = next(iter(surfaces.values()))
        self.size = sample[0].get_size() if sample else (DISPLAY_WIDTH, DISPLAY_WIDTH)

    def set_mouth_level(self, level: float) -> None:
        self._sample = max(0.0, min(1.0, float(level)))

    def play_state(self, state: str) -> None:
        self._state = state or "idle"

    def look_at(self, x: float | None, y: float | None) -> None:
        if x is None or y is None:
            self._look_target = (0.0, 0.0)
            return
        self._look_target = (max(-1.0, min(1.0, x)), max(-1.0, min(1.0, y)))

    def set_expression(self, name: str) -> None:
        self._expression = name or "neutral"

    def advance(self, now: float) -> tuple[pygame.Surface, pygame.Surface | None, float]:
        dt = min(0.1, max(0.0, now - self._last))
        self._last = now
        self._now = now
        self.mouth.push(self._sample, dt or 1 / 30)
        self.bucket = self.mouth.bucket()
        self._look = (
            _smooth(self._look[0], self._look_target[0], dt, LOOK_TAU),
            _smooth(self._look[1], self._look_target[1], dt, LOOK_TAU),
        )
        self.machine.play_state(self._state, now)
        self.machine.set_expression(self._expression, now)
        name = self.machine.clip_at(now)
        if name not in self.clips:
            name = next(iter(self.clips))
        if name != self._clip:
            if self._clip in self.surfaces:
                self._fade_from = self._current_surface()
                self._fade_at = now
            self._clip = name
            self._clip_at = now
        surface = self._current_surface()
        fade = 1.0
        previous = None
        if self._fade_from is not None:
            frame_dt = self.clips[name].durations[0] if self.clips[name].durations else 1 / 24
            fade = crossfade(now - self._fade_at, frame_dt)
            if fade < 1:
                previous = self._fade_from
            else:
                self._fade_from = None
        self.debug = f"{name}  rms {self.mouth.level:.2f}  bucket {self.bucket}  {self._state}"
        return surface, previous, fade

    def _current_surface(self) -> pygame.Surface:
        clip = self.clips[self._clip]
        frames = self.surfaces[self._clip]
        if self._clip == self.talk_name and self.buckets:
            tick = int((self._now - self._clip_at) / max(clip.durations[0], 1e-3))
            index = pick_bucket_frame(self.buckets, self.bucket, tick)
        else:
            index = clip.index_at(self._now - self._clip_at)
        return frames[min(index, len(frames) - 1)]

    def blit(self, dest: pygame.Surface, dest_rect: pygame.Rect, now: float) -> None:
        current, previous, fade = self.advance(now)
        angle = -self._look[0] * TILT_DEG
        shift = (int(round(self._look[0] * MAX_LOOK_PX)), int(round(self._look[1] * MAX_LOOK_PX * 0.65)))
        if previous is not None and fade < 1:
            _paint(dest, dest_rect, previous, shift, angle, int(255 * (1 - fade)))
        _paint(dest, dest_rect, current, shift, angle, int(255 * fade) if previous is not None else 255)


def _paint(dest, rect, surface, shift, angle, alpha) -> None:
    image = pygame.transform.rotozoom(surface, angle, 1.0) if abs(angle) > 0.2 else surface
    if alpha < 255:
        image = image.copy()
        image.set_alpha(alpha)
    target = image.get_rect(center=(rect.centerx + shift[0], rect.centery + shift[1]))
    dest.blit(image, target)


def load_sprite_avatar() -> SpriteAvatar:
    folder = anim_dir()
    config = load_clips_config(folder / "clips.yaml")
    loaded: dict[str, ClipFrames] = {}
    size = None
    for spec in config.clips:
        frames, durations = load_frames(spec.file)
        if size is None and frames:
            size = (frames[0].shape[1], frames[0].shape[0])
        if size:
            frames = resize_frames(frames, size)
        loaded[spec.name] = ClipFrames(spec.name, spec.state, spec.loop, durations, frames)
    view = (DISPLAY_WIDTH, int(DISPLAY_WIDTH * size[1] / size[0])) if size else (DISPLAY_WIDTH, DISPLAY_WIDTH)
    surfaces = {name: [_surface(frame, view) for frame in clip.frames] for name, clip in loaded.items()}
    talk_name = next((clip.name for clip in loaded.values() if clip.state == "talking"), None)
    buckets: list[list[int]] = []
    if talk_name:
        buckets = _talk_buckets(folder, loaded[talk_name], config.mouth_box, config.buckets)
    return SpriteAvatar(loaded, surfaces, buckets, talk_name)


def _talk_buckets(folder, clip, box, count: int) -> list[list[int]]:
    import json

    cache = folder / "talk_mouth.json"
    stamp = clip.frames[0].shape if clip.frames else (0, 0)
    key = {"box": list(box), "frames": len(clip.frames), "shape": list(stamp), "buckets": count}
    if cache.is_file():
        try:
            saved = json.loads(cache.read_text(encoding="utf-8"))
            if saved.get("key") == key and isinstance(saved.get("buckets"), list):
                return saved["buckets"]
        except (OSError, json.JSONDecodeError):
            pass
    scores = [openness(frame, box) for frame in clip.frames]
    buckets = assign_buckets(scores, count)
    cache.write_text(json.dumps({"key": key, "scores": scores, "buckets": buckets}), encoding="utf-8")
    return buckets


def _surface(frame: np.ndarray, size: tuple[int, int]) -> pygame.Surface:
    if frame.shape[1] != size[0] or frame.shape[0] != size[1]:
        frame = resize_frames([frame], size)[0]
    raw = np.ascontiguousarray(frame)
    image = pygame.image.frombytes(raw.tobytes(), (raw.shape[1], raw.shape[0]), "RGB")
    return image.convert()


def _drive(renderer: SpriteAvatar, state: SharedState, now: float) -> None:
    snap = state.snapshot()
    speaking = bool(snap["speaking"])
    renderer.set_mouth_level(snap["mouth_level"] if speaking else 0.0)
    if speaking:
        renderer.play_state("talking")
    elif snap.get("user_speaking"):
        renderer.play_state("listening")
    else:
        renderer.play_state("idle")
    mood = snap["mood"] if snap["vision_enabled"] else "neutral"
    renderer.set_expression(mood)
    if snap["vision_enabled"]:
        renderer.look_at(snap.get("face_x"), snap.get("face_y"))
    else:
        renderer.look_at(None, None)
    del now


def _always_on_top() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes

        hwnd = pygame.display.get_wm_info()["window"]
        ctypes.windll.user32.SetWindowPos(hwnd, -1, 0, 0, 0, 0, 0x0001 | 0x0002)
    except Exception:
        pass


def _open(caption: str, size: tuple[int, int]) -> pygame.Surface:
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption(caption)
    _always_on_top()
    return screen


def run_avatar(state: SharedState) -> None:
    os.environ.setdefault("SDL_VIDEO_WINDOW_POS", "40,40")
    pygame.init()
    pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_WIDTH))
    renderer = load_sprite_avatar()
    width, height = renderer.size
    screen = _open("Nuonuo", (width, height + STATUS_H))
    font = pygame.font.SysFont(None, 18)
    clock = pygame.time.Clock()
    picture = pygame.Rect(0, 0, width, height)
    try:
        while not state.stop.is_set():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    state.stop.set()
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        state.stop.set()
                    elif event.key == pygame.K_v:
                        state.set(vision_enabled=not state.vision_enabled)
                    elif event.key == pygame.K_p:
                        state.set(preview=not state.preview)
                    elif event.key == pygame.K_m:
                        current = state.snapshot()["mode"]
                        state.set(mode="quiet" if current == "chatty" else "chatty")
            now = time.monotonic()
            _drive(renderer, state, now)
            screen.fill((255, 236, 244))
            renderer.blit(screen, picture, now)
            snap = state.snapshot()
            mode = snap["mode"] if snap["mode"] in ("chatty", "quiet") else "chatty"
            screen.blit(font.render(f"{snap['status']}  {mode} (M)", True, (90, 60, 110)), (6, height + 4))
            info = f"vision {'on' if snap['vision_enabled'] else 'off'} (V)   P preview"
            screen.blit(font.render(info, True, (90, 60, 110)), (6, height + 24))
            pygame.display.flip()
            clock.tick(30)
    finally:
        pygame.quit()


def run_preview(wav_path: str | None = None) -> None:
    os.environ.setdefault("SDL_VIDEO_WINDOW_POS", "60,60")
    pygame.init()
    pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_WIDTH))
    renderer = load_sprite_avatar()
    width, height = renderer.size
    screen = _open("Nuonuo preview", (width, height + STATUS_H))
    font = pygame.font.SysFont(None, 18)
    clock = pygame.time.Clock()
    picture = pygame.Rect(0, 0, width, height)
    tone = False
    forced = "idle"
    expression = "neutral"
    wav = _wav_track(wav_path) if wav_path else None
    if wav is not None:
        _play_wav(wav)
    t0 = time.monotonic()
    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q)
                ):
                    return
                if event.type != pygame.KEYDOWN:
                    continue
                if event.key == pygame.K_1:
                    forced, expression, tone = "idle", "neutral", False
                    _play_tone(False)
                elif event.key == pygame.K_2:
                    forced, expression, tone = "talking", "neutral", True
                    _play_tone(True)
                elif event.key == pygame.K_3:
                    forced, expression, tone = "idle", "happy", False
                    _play_tone(False)
                elif event.key == pygame.K_4:
                    forced, expression, tone = "idle", "sleepy", False
                    _play_tone(False)
                elif event.key == pygame.K_5:
                    forced, expression, tone = "idle", "surprised", False
                    _play_tone(False)
                elif event.key == pygame.K_6:
                    forced, expression, tone = "listening", "neutral", False
                    _play_tone(False)
                elif event.key == pygame.K_t:
                    tone = not tone
                    if tone:
                        forced = "talking"
                    _play_tone(tone)
            now = time.monotonic()
            level = 0.0
            if wav is not None:
                level = wav.level_at(now - t0)
                forced = "talking" if level > 0.04 else "idle"
            elif tone:
                level = 0.5 + 0.5 * math.sin(now * 8.0)
            renderer.play_state(forced)
            renderer.set_expression(expression)
            renderer.set_mouth_level(level if forced == "talking" else 0.0)
            renderer.look_at(math.sin(now * 0.4) * 0.4, 0.0)
            screen.fill((255, 236, 244))
            renderer.blit(screen, picture, now)
            help_line = "1 idle  2 talk  3 happy  4 sleepy  5 surprised  6 listen  T tone"
            screen.blit(font.render(help_line, True, (90, 60, 110)), (6, height + 2))
            screen.blit(font.render(renderer.debug, True, (40, 40, 40)), (6, height + 24))
            pygame.display.flip()
            clock.tick(30)
    finally:
        _play_tone(False)
        pygame.quit()


class _WavTrack:
    def __init__(self, levels: list[tuple[float, float]], samples: np.ndarray, rate: int):
        self.levels = levels
        self.samples = samples
        self.rate = rate

    def level_at(self, elapsed: float) -> float:
        value = 0.0
        for stamp, level in self.levels:
            if stamp > elapsed:
                break
            value = level
        if self.levels and elapsed > self.levels[-1][0] + 0.08:
            return 0.0
        return value


def _wav_track(path: str) -> _WavTrack | None:
    try:
        with wave.open(path, "rb") as handle:
            rate = handle.getframerate()
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            raw = handle.readframes(handle.getnframes())
    except (OSError, wave.Error):
        return None
    if width != 2:
        return None
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    hop = max(1, int(rate * 0.04))
    levels = []
    for start in range(0, len(samples), hop):
        chunk = samples[start : start + hop]
        rms = float(np.sqrt(np.mean(chunk * chunk))) if len(chunk) else 0.0
        levels.append((start / rate, min(1.0, rms * 6.0)))
    return _WavTrack(levels, samples, rate)


def _play_wav(track: _WavTrack) -> None:
    try:
        import sounddevice as sd

        sd.play(track.samples, track.rate)
    except Exception:
        return


def _play_tone(active: bool) -> None:
    try:
        import sounddevice as sd

        if not active:
            sd.stop()
            return
        rate = 24000
        times = np.arange(rate) / rate
        wave_data = (0.08 * np.sin(2 * math.pi * 220 * times) * (0.5 + 0.5 * np.sin(2 * math.pi * 4 * times))).astype(np.float32)
        sd.play(wave_data, rate, loop=True)
    except Exception:
        return


def run_calibrate() -> None:
    """Show the mouth box on the first talk frame. Arrows move, +/- resize, C saves."""
    os.environ.setdefault("SDL_VIDEO_WINDOW_POS", "40,40")
    pygame.init()
    folder = anim_dir()
    config = load_clips_config(folder / "clips.yaml")
    talk = next(clip for clip in config.clips if clip.state == "talking")
    frames, _durations = load_frames(talk.file)
    frame = frames[len(frames) // 3]
    pygame.display.set_mode((frame.shape[1], frame.shape[0]))
    screen = pygame.display.set_mode((frame.shape[1], frame.shape[0]))
    base = pygame.image.frombytes(np.ascontiguousarray(frame).tobytes(), (frame.shape[1], frame.shape[0]), "RGB").convert()
    box = list(config.mouth_box)
    font = pygame.font.SysFont(None, 20)
    clock = pygame.time.Clock()
    try:
        while True:
            step = 1
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q)
                ):
                    return
                if event.type != pygame.KEYDOWN:
                    continue
                if event.mod & pygame.KMOD_SHIFT:
                    step = 5
                if event.key == pygame.K_LEFT:
                    box[0] -= step
                elif event.key == pygame.K_RIGHT:
                    box[0] += step
                elif event.key == pygame.K_UP:
                    box[1] -= step
                elif event.key == pygame.K_DOWN:
                    box[1] += step
                elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                    box[2] += step
                    box[3] += step
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    box[2] = max(4, box[2] - step)
                    box[3] = max(4, box[3] - step)
                elif event.key == pygame.K_c:
                    _write_mouth_box(folder / "clips.yaml", box)
                    print("mouth.box", tuple(box))
            screen.blit(base, (0, 0))
            pygame.draw.rect(screen, (220, 60, 140), pygame.Rect(box), 2)
            screen.blit(font.render(f"mouth {tuple(box)}   arrows  +/-  C saves  Esc", True, (255, 255, 255)), (8, 8))
            pygame.display.flip()
            clock.tick(30)
    finally:
        pygame.quit()


def _write_mouth_box(path: Path, box: list[int]) -> None:
    import re

    text = path.read_text(encoding="utf-8")
    updated = re.sub(r"box:\s*\[[^\]]+\]", "box: [" + ", ".join(str(v) for v in box) + "]", text, count=1)
    path.write_text(updated, encoding="utf-8")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="python -m nuonuo.avatar")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--calibrate", action="store_true", help="set the talk mouth box")
    parser.add_argument("--wav", default=None, help="drive talking from a WAV file")
    args = parser.parse_args(argv)
    if args.calibrate:
        run_calibrate()
    else:
        run_preview(args.wav)


if __name__ == "__main__":
    main()
