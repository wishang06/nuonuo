"""Adaptive detection cadence: fewer MediaPipe runs, readings held between them, gestures still caught."""

import math

from nuonuo.perception import CONFIG, DetectionScheduler, Perception, ReadingHold
from tests.test_perception import NEUTRAL

FPS = 30


def simulate(seconds: float, frame):
    """Drive Perception the way vision.py does: a 30 fps frame clock, detections when the scheduler says so.

    frame(t) returns (blend, gesture, wrist_x) for the camera image at time t.
    """
    p = Perception()
    sched = DetectionScheduler(p.cfg)
    out = {"events": [], "detections": 0, "gesture_runs": 0, "intervals": []}
    for i in range(int(seconds * FPS)):
        t = i / FPS
        if not sched.due(t):
            continue
        blend, gesture, wrist_x = frame(t)
        run_gestures = sched.gestures_due()
        hand = None
        if run_gestures:
            out["gesture_runs"] += 1
            hand = wrist_x is not None
        else:
            gesture = wrist_x = None
        ev = p.update(t, blend, None, gesture, wrist_x, None, hand_fresh=run_gestures)
        if ev:
            out["events"].append(ev.text)
        face = (0.0, 0.0) if blend is not None else None
        out["intervals"].append(sched.done(t, face, p.state_label(), event=ev is not None, hand=hand))
        out["detections"] += 1
    return out


def test_intervals_live_in_perception_config():
    assert CONFIG["detect_min_seconds"] == 0.15
    assert CONFIG["detect_max_seconds"] == 0.5
    assert CONFIG["gesture_every"] == 4


def test_calm_scene_stretches_to_the_max_and_a_change_snaps_back():
    sched = DetectionScheduler()
    t, intervals = 0.0, []
    for _ in range(10):
        assert sched.due(t)
        intervals.append(sched.done(t, (0.0, 0.0), "present"))
        assert not sched.due(t + intervals[-1] - 0.01)
        t += intervals[-1]
    assert intervals[0] == 0.15
    assert intervals[-1] == 0.5
    assert intervals == sorted(intervals)
    assert sched.done(t, (0.3, 0.0), "present") == 0.15  # face moved
    assert sched.done(t + 0.15, (0.3, 0.0), "present") > 0.15
    assert sched.done(t + 0.4, (0.3, 0.0), "smiling") == 0.15  # state changed
    assert sched.done(t + 0.55, None, "smiling") == 0.15  # face left


def test_gestures_run_every_fourth_detection_and_every_detection_with_a_hand():
    sched = DetectionScheduler()
    runs = []
    for i in range(12):
        due = sched.gestures_due()
        runs.append(due)
        sched.done(i * 0.5, (0.0, 0.0), "present", hand=False if due else None)
    assert runs == [True, False, False, False] * 3
    sched.done(6.0, (0.0, 0.0), "present", hand=True)
    assert [sched.gestures_due() for _ in range(3)] == [True, True, True]
    assert sched.interval == 0.15


def test_a_still_user_costs_a_fraction_of_the_frames():
    out = simulate(60, lambda t: (NEUTRAL, None, None))
    frames = 60 * FPS
    assert out["detections"] < frames / 12  # about 2 per second instead of 30
    assert out["gesture_runs"] <= out["detections"] // 4 + 1
    assert out["events"] == []
    assert max(out["intervals"]) == 0.5


def test_an_empty_room_is_checked_at_the_slow_rate():
    out = simulate(30, lambda t: (None, None, None))
    assert out["detections"] < 30 / 0.5 + 10


def first_gesture_latency(start: float, kind: str) -> float | None:
    """Seconds from the gesture starting (and being held) to its event, in an otherwise calm scene."""

    def frame(t):
        if t < start:
            return NEUTRAL, None, None
        if kind == "thumb":
            return NEUTRAL, "Thumb_Up", 0.5
        return NEUTRAL, "Open_Palm", 0.5 + 0.08 * math.sin(2 * math.pi * 2.5 * (t - start))

    p = Perception()
    sched = DetectionScheduler(p.cfg)
    for i in range(int((start + 6) * FPS)):
        t = i / FPS
        if not sched.due(t):
            continue
        blend, gesture, wrist_x = frame(t)
        run = sched.gestures_due()
        ev = p.update(t, blend, None, gesture if run else None, wrist_x if run else None, None, hand_fresh=run)
        sched.done(t, (0.0, 0.0), p.state_label(), event=ev is not None, hand=(wrist_x is not None) if run else None)
        if ev and ("thumbs up" in ev.text or "waved" in ev.text):
            return t - start
    return None


def test_held_gestures_register_within_two_and_a_half_seconds_from_any_phase():
    # Calm scene: gestures run on every 4th detection, about every 2 s. Once a hand is seen they run on
    # every detection, so the wave gets enough wrist samples. Latency is bounded, nothing is missed.
    starts = [5.0 + k * 0.1 for k in range(25)]
    thumbs = [first_gesture_latency(s, "thumb") for s in starts]
    waves = [first_gesture_latency(s, "wave") for s in starts]
    assert None not in thumbs and max(thumbs) < 2.5
    assert None not in waves and max(waves) < 3.0


def test_thumbs_up_still_registers_once_at_the_reduced_rate():
    def frame(t):
        if 5.0 <= t < 8.0:
            return NEUTRAL, "Thumb_Up", 0.5
        return NEUTRAL, None, None

    out = simulate(10, frame)
    assert out["events"] == ["[Andy just gave a thumbs up]"]


def test_a_wave_is_a_wave_not_an_open_palm():
    def frame(t):
        if 5.0 <= t < 9.0:
            return NEUTRAL, "Open_Palm", 0.5 + 0.08 * math.sin(2 * math.pi * 2.5 * (t - 5.0))
        return NEUTRAL, None, None

    out = simulate(10, frame)
    assert "[Andy just waved]" in out["events"]
    assert "[Andy just held up an open palm]" not in out["events"]


def test_a_skipped_gesture_run_does_not_reset_a_held_gesture():
    p = Perception()
    p.update(0.0, NEUTRAL, gesture="Thumb_Up", wrist_x=0.5)
    assert p.update(0.15, NEUTRAL, hand_fresh=False) is None
    assert p.update(0.30, NEUTRAL, hand_fresh=False) is None
    ev = p.update(0.45, NEUTRAL, gesture="Thumb_Up", wrist_x=0.5)
    assert ev is not None and ev.text == "[Andy just gave a thumbs up]"


def test_reading_hold_keeps_the_face_through_a_missed_detection():
    hold = ReadingHold(0.75)
    assert hold.update(0.0, (0.2, -0.1)) == (0.2, -0.1)
    assert hold.update(0.5, None) == (0.2, -0.1)
    assert hold.update(0.75, None) == (0.2, -0.1)
    assert hold.update(0.9, None) is None
    assert hold.update(1.0, (0.1, 0.0)) == (0.1, 0.0)
