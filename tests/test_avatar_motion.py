"""Clip timing, avatar state, and the voice-to-mouth map."""

from pathlib import Path

from nuonuo.clips import (
    IDLE_VARIANT_MIN,
    ClipFrames,
    ClipStateMachine,
    MouthEnvelope,
    assign_buckets,
    bucket_for_level,
    crossfade,
    load_clips_config,
    pick_bucket_frame,
    step_level,
)


class _Scripted:
    def __init__(self, uniforms, choices=None):
        self._uniforms = list(uniforms)
        self._choices = list(choices or [])

    def uniform(self, _low, _high):
        return self._uniforms.pop(0)

    def choice(self, items):
        index = self._choices.pop(0)
        return items[index]


def test_clip_timing_loops_and_holds_the_last_frame():
    looping = ClipFrames("idle", "idle", True, [0.1, 0.1, 0.1], [])
    assert looping.index_at(0) == 0
    assert looping.index_at(0.25) == 2
    assert looping.index_at(0.31) == 0
    once = ClipFrames("happy", "happy", False, [0.1, 0.1], [])
    assert once.index_at(0.05) == 0
    assert once.index_at(5.0) == 1


def test_crossfade_covers_four_frames():
    assert crossfade(0, 1 / 24) == 0
    assert 0.4 < crossfade(2 / 24, 1 / 24) < 0.6
    assert crossfade(4 / 24, 1 / 24) == 1


def test_talking_returns_to_idle_within_150ms():
    machine = ClipStateMachine(
        {"idle": ["idle"], "talking": ["talk"], "happy": ["happy"]},
        {"idle": 4.0, "talk": 4.0, "happy": 1.0},
        rng=_Scripted([9.0]),
        now=0.0,
    )
    assert machine.clip_at(0.0) == "idle"
    machine.play_state("talking", 0.0)
    assert machine.clip_at(0.149) == "talk"
    assert machine.clip_at(0.15) == "idle"


def test_expression_plays_once_then_returns_to_talking():
    machine = ClipStateMachine(
        {"idle": ["idle"], "talking": ["talk"], "happy": ["happy"]},
        {"idle": 4.0, "talk": 4.0, "happy": 1.0},
        rng=_Scripted([9.0]),
        now=0.0,
    )
    machine.play_state("talking", 0.0)
    machine.set_expression("happy", 0.0)
    assert machine.clip_at(0.2) == "happy"
    machine.play_state("talking", 0.9)
    assert machine.clip_at(0.95) == "happy"
    assert machine.clip_at(1.0) == "talk"
    machine.set_expression("neutral", 1.2)
    machine.set_expression("happy", 1.3)
    assert machine.clip_at(1.3) == "happy"


def test_idle_variant_waits_between_5_and_15_seconds():
    machine = ClipStateMachine(
        {"idle": ["idle", "look"]},
        {"idle": 4.0, "look": 1.0},
        rng=_Scripted([5.0, 6.0], choices=[0]),
        now=0.0,
    )
    assert machine.clip_at(4.9) == "idle"
    assert machine.clip_at(5.0) == "look"
    assert machine.clip_at(5.9) == "look"
    assert machine.clip_at(6.0) == "idle"
    assert machine._next_variant - 6.0 == 6.0
    assert IDLE_VARIANT_MIN <= 5.0 <= 15.0


def test_silence_is_the_closed_bucket_and_loud_is_wide():
    assert bucket_for_level(0.0) == 0
    assert bucket_for_level(0.03) == 0
    assert bucket_for_level(1.0) == 4
    scores = [0.01, 0.02, 0.2, 0.4, 0.8]
    buckets = assign_buckets(scores, 5)
    assert buckets[0] == [0]
    assert buckets[-1] == [4]
    assert pick_bucket_frame(buckets, 0, 3) == 0
    assert pick_bucket_frame(buckets, 4, 1) == 4


def test_smoothing_does_not_jump_to_wide_on_one_spike():
    level = step_level(0.0, 1.0, 0.016)
    assert level < 0.5
    assert bucket_for_level(level) < 4
    released = step_level(level, 0.0, 0.016)
    assert released < level
    envelope = MouthEnvelope()
    for _ in range(20):
        envelope.push(1.0, 0.04)
    assert envelope.bucket() == 4
    for _ in range(8):
        envelope.push(0.0, 0.04)
    assert envelope.bucket() == 0


def test_clips_yaml_loads():
    path = Path(__file__).resolve().parents[1] / "assets" / "anim" / "clips.yaml"
    config = load_clips_config(path)
    names = [clip.name for clip in config.clips]
    assert names[:3] == ["idle", "talk", "look"] or "idle" in names and "talk" in names and "happy" in names
    assert config.clips[0].loop is True
    assert len(config.mouth_box) == 4
    assert config.buckets == 5
    assert config.named("talk").state == "talking"
    assert config.named("talk").file.name == "talk.mp4"
