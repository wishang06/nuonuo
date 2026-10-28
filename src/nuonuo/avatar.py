import math
import os
import sys
import time

import pygame

from nuonuo.state import SharedState

SIZE = (260, 300)
BG = (255, 236, 244)
SKIN = (255, 224, 210)
HAIR = (120, 70, 160)
EYE = (60, 40, 90)
BLUSH = (255, 160, 180)
MOUTH = (180, 60, 90)


def _always_on_top() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes

        hwnd = pygame.display.get_wm_info()["window"]
        ctypes.windll.user32.SetWindowPos(hwnd, -1, 0, 0, 0, 0, 0x0001 | 0x0002)
    except Exception:
        pass


def draw_face(surf: pygame.Surface, mood: str, speaking: bool, mouth_level: float, t: float) -> None:
    w, h = surf.get_size()
    cx, cy = w // 2, h // 2 + 10
    bob = math.sin(t * 3) * 4 if mood == "happy" else math.sin(t * 1.5) * 2
    cy += int(bob)
    surf.fill(BG)
    pygame.draw.circle(surf, HAIR, (cx, cy - 10), 105)
    pygame.draw.ellipse(surf, SKIN, (cx - 85, cy - 75, 170, 165))
    pygame.draw.polygon(surf, HAIR, [(cx - 90, cy - 40), (cx - 30, cy - 95), (cx + 20, cy - 50), (cx + 90, cy - 40), (cx + 60, cy - 110), (cx - 60, cy - 110)])
    pygame.draw.circle(surf, HAIR, (cx - 95, cy - 85), 22)
    pygame.draw.circle(surf, HAIR, (cx + 95, cy - 85), 22)

    blink = (t % 4.0) < 0.12
    ey = cy + 5
    for dx in (-35, 35):
        ex = cx + dx
        if blink or mood == "happy":
            pygame.draw.arc(surf, EYE, (ex - 16, ey - 10, 32, 24), math.radians(20), math.radians(160), 4)
        elif mood == "surprised":
            pygame.draw.circle(surf, EYE, (ex, ey), 17)
            pygame.draw.circle(surf, (255, 255, 255), (ex - 5, ey - 6), 6)
        elif mood == "sad":
            pygame.draw.ellipse(surf, EYE, (ex - 12, ey - 8, 24, 22))
            pygame.draw.line(surf, EYE, (ex - 16, ey - 20 + (6 if dx < 0 else -6) * -1), (ex + 16, ey - 20 + (6 if dx < 0 else -6)), 3)
        else:
            pygame.draw.ellipse(surf, EYE, (ex - 13, ey - 16, 26, 32))
            pygame.draw.circle(surf, (255, 255, 255), (ex - 4, ey - 7), 5)
    if mood in ("happy", "neutral"):
        for dx in (-55, 55):
            pygame.draw.ellipse(surf, BLUSH, (cx + dx - 14, cy + 28, 28, 12))

    my = cy + 48
    if speaking:
        open_h = 6 + int(22 * max(mouth_level, 0.15 + 0.15 * abs(math.sin(t * 18))))
        pygame.draw.ellipse(surf, MOUTH, (cx - 14, my - open_h // 2, 28, open_h))
    elif mood == "happy":
        pygame.draw.arc(surf, MOUTH, (cx - 20, my - 16, 40, 28), math.radians(200), math.radians(340), 4)
    elif mood == "surprised":
        pygame.draw.circle(surf, MOUTH, (cx, my), 10, 3)
    elif mood == "sad":
        pygame.draw.arc(surf, MOUTH, (cx - 16, my, 32, 20), math.radians(20), math.radians(160), 3)
    else:
        pygame.draw.line(surf, MOUTH, (cx - 10, my), (cx + 10, my), 3)


def run_avatar(state: SharedState) -> None:
    os.environ.setdefault("SDL_VIDEO_WINDOW_POS", "40,40")
    pygame.init()
    screen = pygame.display.set_mode(SIZE, pygame.NOFRAME if os.getenv("NUONUO_NOFRAME") else 0)
    pygame.display.set_caption("Nuonuo")
    _always_on_top()
    font = pygame.font.SysFont(None, 18)
    clock = pygame.time.Clock()
    t0 = time.monotonic()
    try:
        while not state.stop.is_set():
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    state.stop.set()
                elif e.type == pygame.KEYDOWN:
                    if e.key in (pygame.K_ESCAPE, pygame.K_q):
                        state.stop.set()
                    elif e.key == pygame.K_v:
                        state.set(vision_enabled=not state.vision_enabled)
                    elif e.key == pygame.K_p:
                        state.set(preview=not state.preview)
            s = state.snapshot()
            mood = s["mood"] if s["vision_enabled"] else "neutral"
            draw_face(screen, mood, s["speaking"], s["mouth_level"], time.monotonic() - t0)
            info = f"{s['status']} | vision {'on' if s['vision_enabled'] else 'off'} (V) | P preview"
            screen.blit(font.render(info, True, (90, 60, 110)), (6, SIZE[1] - 18))
            if s["last_event"]:
                screen.blit(font.render(s["last_event"][:40], True, (140, 100, 160)), (6, 4))
            pygame.display.flip()
            clock.tick(30)
    finally:
        pygame.quit()
