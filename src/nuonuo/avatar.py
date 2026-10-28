"""Painted avatar with eyelid, iris, and mouth overlays.

OverlayAvatar is the current renderer. A Live2D or Inochi2D class can replace
it by implementing set_mouth, blink, look_at, and set_expression.
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Protocol

import pygame

from nuonuo.avatar_motion import (
    BLINK_CLOSE_S,
    BLINK_OPEN_S,
    LOOK_TAU,
    BlinkScheduler,
    eyelid_amount,
    mouth_shape,
    smooth_mouth,
    smooth_toward,
)
from nuonuo.state import SharedState

# Source pixels on assets/nuonuo_base.png (448x446). Tune with --calibrate.
REGIONS = {
    "left_eye": (130, 186, 72, 40),
    "right_eye": (246, 184, 84, 42),
    "left_iris": (142, 198, 28, 18),
    "right_iris": (256, 192, 50, 20),
    "mouth": (198, 240, 40, 22),
}
SKIN_SAMPLE = (220, 236)
DISPLAY_WIDTH = 320
STATUS_H = 36
MAX_LOOK_PX = 4
TILT_DEG = 3.0
BOB_PX = 1.5
BOB_PERIOD = 4.0
SKIN_FALLBACK = (255, 236, 230)
OUTLINE = (72, 58, 64)
MOUTH_FILL = (176, 72, 86)

_REGION_ORDER = ("left_eye", "right_eye", "left_iris", "right_iris", "mouth")


def asset_path() -> Path:
    name = "nuonuo_base.png"
    candidates = [
        Path(__file__).resolve().parents[2] / "assets" / name,
        Path.cwd() / "assets" / name,
    ]
    for path in candidates:
        if path.is_file():
            return path
    return candidates[0]


class AvatarRenderer(Protocol):
    """Swap in a Live2D or Inochi2D renderer without changing the app loop."""

    def set_mouth(self, level: float) -> None: ...

    def blink(self) -> None: ...

    def look_at(self, x: float | None, y: float | None) -> None: ...

    def set_expression(self, name: str) -> None: ...


class OverlayAvatar(AvatarRenderer):
    def __init__(self, image: pygame.Surface):
        self.base = image.convert()
        self.expression = "neutral"
        self._mouth_target = 0.0
        self._mouth = 0.0
        self._look = (0.0, 0.0)
        self._look_target = (0.0, 0.0)
        self._blinks: list[float] = []
        self._now = time.monotonic()
        self._last = self._now
        sample = self.base.get_at(SKIN_SAMPLE)
        self.skin = (sample.r, sample.g, sample.b) if sample else SKIN_FALLBACK
        self.frame = pygame.Surface(self.base.get_size())

    def set_mouth(self, level: float) -> None:
        self._mouth_target = max(0.0, min(1.0, float(level)))

    def blink(self) -> None:
        self._blinks.append(time.monotonic())

    def look_at(self, x: float | None, y: float | None) -> None:
        if x is None or y is None:
            self._look_target = (0.0, 0.0)
            return
        self._look_target = (max(-1.0, min(1.0, x)), max(-1.0, min(1.0, y)))

    def set_expression(self, name: str) -> None:
        self.expression = name or "neutral"

    def advance(self, now: float) -> None:
        dt = min(0.1, max(0.0, now - self._last))
        self._last = now
        self._now = now
        self._mouth = smooth_mouth(self._mouth, self._mouth_target, dt)
        self._look = (
            smooth_toward(self._look[0], self._look_target[0], dt, LOOK_TAU),
            smooth_toward(self._look[1], self._look_target[1], dt, LOOK_TAU),
        )
        span = BLINK_CLOSE_S + BLINK_OPEN_S + 0.02
        self._blinks = [started for started in self._blinks if now - started < span]

    def _lid(self) -> float:
        amount = 0.0
        for started in self._blinks:
            amount = max(amount, eyelid_amount(self._now - started))
        if self.expression == "sleepy":
            amount = max(amount, 0.42)
        return amount

    def _compose(self) -> pygame.Surface:
        frame = self.frame
        frame.blit(self.base, (0, 0))
        dx = int(round(self._look[0] * MAX_LOOK_PX))
        dy = int(round(self._look[1] * MAX_LOOK_PX * 0.6))
        if dx or dy:
            _shift_iris(frame, REGIONS["left_iris"], REGIONS["left_eye"], dx, dy, self.skin)
            _shift_iris(frame, REGIONS["right_iris"], REGIONS["right_eye"], dx, dy, self.skin)
        if self.expression == "happy":
            _happy_eyes(frame, self.skin)
        elif self.expression == "surprised":
            _wider_eyes(frame)
        lid = self._lid()
        if lid > 0.02:
            _draw_lids(frame, lid, self.skin)
        shape = "o" if self.expression == "surprised" and self._mouth_target <= 0.02 else mouth_shape(
            self._mouth, self._mouth_target > 0.02
        )
        if shape != "closed":
            _draw_mouth(frame, shape, self.skin)
        elif self.expression == "sad":
            _draw_mouth(frame, "frown", self.skin)
        return frame

    def blit(self, dest: pygame.Surface, dest_rect: pygame.Rect, now: float) -> None:
        self.advance(now)
        composed = self._compose()
        angle = -self._look[0] * TILT_DEG
        bob = math.sin(now * (2 * math.pi / BOB_PERIOD)) * BOB_PX
        scale = dest_rect.w / composed.get_width()
        shown = pygame.transform.rotozoom(composed, angle, scale)
        rect = shown.get_rect(center=(dest_rect.centerx, dest_rect.centery + bob))
        dest.blit(shown, rect)


def _rect(name: str) -> pygame.Rect:
    x, y, w, h = REGIONS[name]
    return pygame.Rect(x, y, w, h)


def _shift_iris(frame, iris, eye, dx, dy, skin) -> None:
    iris_r = pygame.Rect(iris)
    eye_r = pygame.Rect(eye)
    patch = frame.subsurface(iris_r).copy()
    pygame.draw.rect(frame, skin, iris_r)
    clip = frame.get_clip()
    frame.set_clip(eye_r)
    frame.blit(patch, (iris_r.x + dx, iris_r.y + dy))
    frame.set_clip(clip)


def _happy_eyes(frame, skin) -> None:
    for name in ("left_eye", "right_eye"):
        rect = _rect(name)
        pygame.draw.ellipse(frame, skin, rect)
        arc = rect.inflate(0, -rect.h // 3)
        pygame.draw.arc(frame, OUTLINE, arc, math.radians(200), math.radians(340), 3)


def _wider_eyes(frame) -> None:
    for name in ("left_eye", "right_eye"):
        rect = _rect(name)
        patch = frame.subsurface(rect).copy()
        big = pygame.transform.smoothscale(patch, (max(1, int(rect.w * 1.12)), max(1, int(rect.h * 1.18))))
        frame.blit(big, big.get_rect(center=rect.center))


def _draw_lids(frame, amount: float, skin) -> None:
    for name in ("left_eye", "right_eye"):
        rect = _rect(name)
        cover = max(1, int(rect.h * amount))
        pygame.draw.ellipse(frame, skin, pygame.Rect(rect.x, rect.y - rect.h + cover, rect.w, rect.h))
        pygame.draw.line(frame, OUTLINE, (rect.x + 2, rect.y + cover), (rect.right - 2, rect.y + cover), 2)


def _draw_mouth(frame, shape: str, skin) -> None:
    rect = _rect("mouth")
    cx, cy = rect.center
    pygame.draw.ellipse(frame, skin, rect.inflate(4, 2))
    if shape == "small":
        pygame.draw.ellipse(frame, MOUTH_FILL, (cx - 6, cy - 2, 12, 5))
    elif shape == "open":
        pygame.draw.ellipse(frame, MOUTH_FILL, (cx - 8, cy - 5, 16, 11))
        pygame.draw.ellipse(frame, OUTLINE, (cx - 8, cy - 5, 16, 11), 1)
    elif shape == "wide":
        pygame.draw.ellipse(frame, MOUTH_FILL, (cx - 12, cy - 7, 24, 14))
        pygame.draw.ellipse(frame, OUTLINE, (cx - 12, cy - 7, 24, 14), 1)
    elif shape == "o":
        pygame.draw.circle(frame, MOUTH_FILL, (cx, cy), 6)
        pygame.draw.circle(frame, OUTLINE, (cx, cy), 6, 1)
    elif shape == "frown":
        pygame.draw.arc(frame, OUTLINE, (cx - 10, cy - 2, 20, 12), math.radians(20), math.radians(160), 2)


def _load_base() -> pygame.Surface:
    path = asset_path()
    if not path.is_file():
        raise FileNotFoundError(f"Missing avatar image: {path}")
    return pygame.image.load(str(path))


def _always_on_top() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes

        hwnd = pygame.display.get_wm_info()["window"]
        ctypes.windll.user32.SetWindowPos(hwnd, -1, 0, 0, 0, 0, 0x0001 | 0x0002)
    except Exception:
        pass


def _drive(renderer: OverlayAvatar, state: SharedState, scheduler: BlinkScheduler, now: float) -> None:
    snap = state.snapshot()
    mood = snap["mood"] if snap["vision_enabled"] else "neutral"
    renderer.set_expression(mood)
    renderer.set_mouth(snap["mouth_level"] if snap["speaking"] else 0.0)
    if snap["vision_enabled"]:
        renderer.look_at(snap.get("face_x"), snap.get("face_y"))
    else:
        renderer.look_at(None, None)
    for _event in scheduler.poll(now):
        renderer.blink()


def _view_height(image: pygame.Surface) -> int:
    return int(image.get_height() * (DISPLAY_WIDTH / image.get_width()))


def run_avatar(state: SharedState) -> None:
    os.environ.setdefault("SDL_VIDEO_WINDOW_POS", "40,40")
    pygame.init()
    raw = _load_base()
    view_h = _view_height(raw)
    screen = pygame.display.set_mode((DISPLAY_WIDTH, view_h + STATUS_H))
    pygame.display.set_caption("Nuonuo")
    _always_on_top()
    base = raw.convert()
    font = pygame.font.SysFont(None, 18)
    clock = pygame.time.Clock()
    renderer = OverlayAvatar(base)
    scheduler = BlinkScheduler(now=time.monotonic())
    picture = pygame.Rect(0, 0, DISPLAY_WIDTH, view_h)
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
            _drive(renderer, state, scheduler, now)
            screen.fill((255, 236, 244))
            renderer.blit(screen, picture, now)
            snap = state.snapshot()
            mode = snap["mode"] if snap["mode"] in ("chatty", "quiet") else "chatty"
            screen.blit(font.render(f"{snap['status']}  {mode} (M)", True, (90, 60, 110)), (6, view_h + 2))
            info = f"vision {'on' if snap['vision_enabled'] else 'off'} (V)   P preview"
            screen.blit(font.render(info, True, (90, 60, 110)), (6, view_h + 18))
            pygame.display.flip()
            clock.tick(30)
    finally:
        pygame.quit()


def run_preview() -> None:
    os.environ.setdefault("SDL_VIDEO_WINDOW_POS", "60,60")
    pygame.init()
    raw = _load_base()
    view_h = _view_height(raw)
    screen = pygame.display.set_mode((DISPLAY_WIDTH, view_h + STATUS_H))
    pygame.display.set_caption("Nuonuo preview")
    _always_on_top()
    base = raw.convert()
    font = pygame.font.SysFont(None, 18)
    clock = pygame.time.Clock()
    renderer = OverlayAvatar(base)
    picture = pygame.Rect(0, 0, DISPLAY_WIDTH, view_h)
    phases = ("blink", "look", "talk", "expressions")
    expressions = ("happy", "sleepy", "surprised", "neutral")
    t0 = time.monotonic()
    shown = ""
    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q)
                ):
                    return
            now = time.monotonic()
            elapsed = now - t0
            phase = phases[int(elapsed / 2.5) % len(phases)]
            local = elapsed % 2.5
            if phase != shown:
                shown = phase
                if phase == "blink":
                    renderer.blink()
            renderer.set_expression("neutral")
            renderer.set_mouth(0.0)
            renderer.look_at(0.0, 0.0)
            if phase == "look":
                renderer.look_at(math.sin(elapsed * 1.4), math.cos(elapsed * 0.7) * 0.5)
            elif phase == "talk":
                renderer.set_mouth(abs(math.sin(elapsed * 8.0)))
            elif phase == "expressions":
                renderer.set_expression(expressions[int(local / 0.625) % len(expressions)])
            screen.fill((255, 236, 244))
            renderer.blit(screen, picture, now)
            label = phase if phase != "expressions" else f"expressions:{renderer.expression}"
            screen.blit(font.render(f"preview  {label}   Esc quits", True, (90, 60, 110)), (6, view_h + 10))
            pygame.display.flip()
            clock.tick(30)
    finally:
        pygame.quit()


def run_calibrate() -> None:
    """Draw the hardcoded boxes. Arrows nudge, +/- resize, Tab selects, C prints."""
    os.environ.setdefault("SDL_VIDEO_WINDOW_POS", "40,40")
    pygame.init()
    raw = _load_base()
    screen = pygame.display.set_mode(raw.get_size())
    pygame.display.set_caption("Nuonuo calibrate")
    base = raw.convert()
    font = pygame.font.SysFont(None, 18)
    boxes = {name: list(REGIONS[name]) for name in _REGION_ORDER}
    colors = {
        "left_eye": (40, 180, 80),
        "right_eye": (40, 180, 180),
        "left_iris": (230, 190, 40),
        "right_iris": (230, 140, 40),
        "mouth": (220, 60, 140),
    }
    selected = 0
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
                if event.key == pygame.K_TAB:
                    selected = (selected + 1) % len(_REGION_ORDER)
                elif event.key == pygame.K_c:
                    _print_regions(boxes)
                else:
                    if event.mod & pygame.KMOD_SHIFT:
                        step = 5
                    box = boxes[_REGION_ORDER[selected]]
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
                        box[2] = max(2, box[2] - step)
                        box[3] = max(2, box[3] - step)
            screen.blit(base, (0, 0))
            for name, box in boxes.items():
                thick = 3 if name == _REGION_ORDER[selected] else 1
                pygame.draw.rect(screen, colors[name], pygame.Rect(box), thick)
            name = _REGION_ORDER[selected]
            box = boxes[name]
            screen.blit(
                font.render(f"{name} {tuple(box)}   Tab  arrows  +/-  C prints  Esc", True, (20, 20, 20)),
                (8, 8),
            )
            pygame.display.flip()
            clock.tick(30)
    finally:
        pygame.quit()


def _print_regions(boxes: dict) -> None:
    print("REGIONS = {")
    for name in _REGION_ORDER:
        print(f"    {name!r}: {tuple(boxes[name])},")
    print("}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="python -m nuonuo.avatar")
    parser.add_argument("--preview", action="store_true", help="cycle blink, look, talking, and expressions")
    parser.add_argument("--calibrate", action="store_true", help="show eye and mouth boxes on the base image")
    args = parser.parse_args(argv)
    if args.calibrate:
        run_calibrate()
    else:
        run_preview()


if __name__ == "__main__":
    main()
