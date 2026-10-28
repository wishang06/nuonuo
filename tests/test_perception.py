import math

from nuonuo.perception import Perception

NEUTRAL = {
    "mouthSmileLeft": 0.05, "mouthSmileRight": 0.05,
    "mouthFrownLeft": 0.0, "mouthFrownRight": 0.0,
    "browDownLeft": 0.0, "browDownRight": 0.0,
    "browInnerUp": 0.05, "jawOpen": 0.02,
    "eyeBlinkLeft": 0.1, "eyeBlinkRight": 0.1,
    "eyeWideLeft": 0.0, "eyeWideRight": 0.0,
}
FPS = 30


def face(**kw):
    return NEUTRAL | kw


def run(p, seconds, start, blend=None, use_neutral=True, **kw):
    events = []
    n = int(seconds * FPS)
    for i in range(n):
        t = start + i / FPS
        b = blend if blend is not None else (NEUTRAL if use_neutral else None)
        k = {key: (v(t) if callable(v) else v) for key, v in kw.items()}
        ev = p.update(t, b, **k)
        if ev:
            events.append(ev)
    return events, start + n / FPS


def texts(evs):
    return [e.text for e in evs]


def test_neutral_emits_nothing():
    p = Perception()
    evs, _ = run(p, 3, 0)
    assert evs == []
    assert p.mood == "neutral"


def test_smile_once_on_change():
    p = Perception()
    _, t = run(p, 1, 0)
    evs, t = run(p, 3, t, blend=face(mouthSmileLeft=0.9, mouthSmileRight=0.9))
    assert texts(evs) == ["[Andy just smiled]"]
    assert p.mood == "happy"


def test_ema_ignores_brief_spike():
    p = Perception()
    _, t = run(p, 1, 0)
    evs, t = run(p, 0.1, t, blend=face(mouthSmileLeft=1.0, mouthSmileRight=1.0))
    evs2, _ = run(p, 2, t)
    assert evs == [] and evs2 == []


def test_frown_and_surprise():
    p = Perception()
    _, t = run(p, 1, 0)
    evs, t = run(p, 3, t, blend=face(mouthFrownLeft=0.7, mouthFrownRight=0.7))
    assert "[Andy just frowned]" in texts(evs)
    p2 = Perception()
    _, t = run(p2, 1, 0)
    evs, _ = run(p2, 3, t, blend=face(jawOpen=0.8, browInnerUp=0.9))
    assert any("surprised" in x for x in texts(evs))
    assert p2.mood == "surprised"


def test_tired_half_closed_eyes():
    p = Perception()
    _, t = run(p, 1, 0)
    evs, _ = run(p, 3, t, blend=face(eyeBlinkLeft=0.55, eyeBlinkRight=0.55))
    assert "[Andy looks tired]" in texts(evs)


def test_tired_slow_blinks():
    p = Perception()
    t = 0.0
    evs = []
    for _ in range(3):
        e, t = run(p, 1.5, t)
        evs += e
        e, t = run(p, 0.6, t, blend=face(eyeBlinkLeft=0.95, eyeBlinkRight=0.95))
        evs += e
    e, t = run(p, 1.0, t)
    evs += e
    assert any("tired" in x for x in texts(evs))


def test_looking_away_by_yaw_and_back():
    p = Perception()
    _, t = run(p, 1, 0, yaw=0.0)
    evs, t = run(p, 1, t, yaw=40.0)
    assert texts(evs) == ["[Andy just looked away]"]
    evs, _ = run(p, 1, t, yaw=0.0)
    assert texts(evs) == ["[Andy just looked back]"]


def test_looking_away_by_eyes():
    p = Perception()
    _, t = run(p, 1, 0)
    evs, _ = run(p, 3, t, blend=face(eyeLookOutLeft=0.9, eyeLookInRight=0.9))
    assert "[Andy just looked away]" in texts(evs)


def test_absent_and_return_is_salient():
    p = Perception()
    _, t = run(p, 1, 0)
    evs, t = run(p, 3, t, use_neutral=False)
    assert texts(evs) == ["[Andy just left]"]
    evs, _ = run(p, 1, t)
    assert texts(evs) == ["[Andy just came back]"]
    assert evs[0].salient


def test_gesture_debounced_once():
    p = Perception()
    _, t = run(p, 1, 0)
    evs, t = run(p, 2, t, gesture="Thumb_Up", wrist_x=0.5)
    assert texts(evs) == ["[Andy just gave a thumbs up]"]
    evs, _ = run(p, 0.1, t, gesture="Victory", wrist_x=0.5)
    assert evs == []


def test_gesture_flicker_ignored():
    p = Perception()
    _, t = run(p, 1, 0)
    evs = []
    for i in range(60):
        g = "Closed_Fist" if i % 2 else None
        e = p.update(t + i / FPS, NEUTRAL, gesture=g, wrist_x=0.5)
        if e:
            evs.append(e)
    assert evs == []


def test_wave_detected_not_open_palm():
    p = Perception()
    _, t = run(p, 1, 0)
    start = t
    evs, _ = run(p, 1.5, t, gesture="Open_Palm",
                 wrist_x=lambda tt: 0.5 + 0.08 * math.sin(2 * math.pi * 2.5 * (tt - start)))
    assert texts(evs) == ["[Andy just waved]"]
    assert evs[0].salient


def test_still_open_palm_is_not_wave():
    p = Perception()
    _, t = run(p, 1, 0)
    evs, _ = run(p, 1.5, t, gesture="Open_Palm", wrist_x=0.5)
    assert texts(evs) == ["[Andy just held up an open palm]"]


def test_combined_smile_and_wave():
    p = Perception(user_name="Andy", cfg=Perception().cfg | {"hold_seconds": 0.2})
    _, t = run(p, 1, 0)
    smile = face(mouthSmileLeft=1.0, mouthSmileRight=1.0)
    found = []
    for i in range(int(2 * FPS)):
        tt = t + i / FPS
        x = 0.5 + 0.08 * math.sin(2 * math.pi * 2.5 * (tt - t))
        e = p.update(tt, smile, gesture="Open_Palm", wrist_x=x)
        if e:
            found.append(e.text)
    joined = " ".join(found)
    assert "smiled" in joined and "waved" in joined
