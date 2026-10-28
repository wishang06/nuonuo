"""Tests for blink timing and the RMS-to-mouth map. No window, no webcam."""

from nuonuo.avatar_motion import (
    BLINK_CLOSE_S,
    BLINK_GAP_MAX,
    BLINK_GAP_MIN,
    BLINK_OPEN_S,
    DOUBLE_GAP_S,
    BlinkScheduler,
    eyelid_amount,
    mouth_shape,
    smooth_mouth,
)


class _Scripted:
    def __init__(self, uniforms, randoms):
        self._uniforms = list(uniforms)
        self._randoms = list(randoms)

    def uniform(self, _a, _b):
        return self._uniforms.pop(0)

    def random(self):
        return self._randoms.pop(0)


def test_eyelid_closes_in_120ms_and_opens_again():
    assert eyelid_amount(0) == 0
    assert eyelid_amount(BLINK_CLOSE_S / 2) == 0.5
    assert eyelid_amount(BLINK_CLOSE_S) == 1
    assert eyelid_amount(BLINK_CLOSE_S + BLINK_OPEN_S / 2) == 0.5
    assert eyelid_amount(BLINK_CLOSE_S + BLINK_OPEN_S) == 0
    assert eyelid_amount(1.0) == 0


def test_blink_gaps_stay_between_2_and_6_seconds():
    sched = BlinkScheduler(_Scripted([3.0] * 12, [0.99] * 12), now=0.0)
    assert BLINK_GAP_MIN <= sched.last_gap <= BLINK_GAP_MAX
    now = sched.last_gap
    for _ in range(8):
        assert sched.poll(now - 0.01) == []
        assert sched.poll(now) == ["blink"]
        assert BLINK_GAP_MIN <= sched.last_gap <= BLINK_GAP_MAX
        now += sched.last_gap


def test_occasional_double_blink_is_scheduled_after_the_first():
    gap_after = 4.0
    sched = BlinkScheduler(_Scripted([3.0, 4.0, 5.0], [0.0, 0.9]), now=0.0)
    assert sched.poll(3.0) == ["blink"]
    second = 3.0 + BLINK_CLOSE_S + BLINK_OPEN_S + DOUBLE_GAP_S
    assert sched.poll(second - 0.01) == []
    assert sched.poll(second) == ["blink"]
    assert sched.poll(second + gap_after) == ["blink"]


def test_silent_audio_keeps_the_mouth_closed():
    for level in (0.0, 0.2, 0.5, 1.0):
        assert mouth_shape(level, speaking=False) == "closed"


def test_talking_level_picks_a_mouth_shape():
    assert mouth_shape(0.0, True) == "closed"
    assert mouth_shape(0.2, True) == "small"
    assert mouth_shape(0.5, True) == "open"
    assert mouth_shape(0.9, True) == "wide"


def test_mouth_smoothing_does_not_jump_on_one_frame():
    stepped = smooth_mouth(0.0, 1.0, dt=1 / 30)
    assert 0.05 < stepped < 0.4
    settled = 0.0
    for _ in range(30):
        settled = smooth_mouth(settled, 1.0, dt=1 / 30)
    assert settled > 0.9
    closed = smooth_mouth(settled, 0.0, dt=1 / 30)
    assert closed < settled
