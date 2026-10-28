from nuonuo.notes import HEARTBEAT_MS, NoteGate, RemarkGate, format_camera_note
from tests.support import load_fixture


def _sent_notes(name: str) -> list[tuple[int, str, bool]]:
    from nuonuo.timeline import BehaviourTimeline

    gate = NoteGate()
    timeline = BehaviourTimeline()
    sent = []
    for step in load_fixture(name)["steps"]:
        timeline.observe(step["t_ms"], step["state"])
        decision = gate.consider(timeline.current_note(step["t_ms"]), step["t_ms"])
        if decision.send:
            sent.append((step["t_ms"], timeline.current_note(step["t_ms"]), decision.changed))
    return sent


def test_identical_states_wait_for_the_heartbeat():
    sent = _sent_notes("steady_smile.json")
    assert [item[0] for item in sent] == [0, HEARTBEAT_MS]
    assert sent[0][1] == sent[1][1]
    assert sent[0][2] is True
    assert sent[1][2] is False
    assert sent[0][1].startswith("Looking at you")
    assert format_camera_note(sent[0][1]).startswith("[camera note, do not read aloud]")


def test_a_real_change_sends_before_the_heartbeat():
    sent = _sent_notes("note_change.json")
    assert [item[0] for item in sent] == [0, 1500]
    assert "Smiling" in sent[0][1]
    assert "Looking away" in sent[1][1]
    assert sent[1][2] is True


def test_quiet_never_schedules_a_visual_remark():
    gate = RemarkGate()
    assert gate.allow("quiet", True, False, False, 0) is False
    assert gate.allow("quiet", True, False, False, 120_000) is False


def test_chatty_respects_the_minute_cap_and_does_not_talk_over_andy():
    gate = RemarkGate()
    assert gate.allow("chatty", True, False, False, 0) is True
    gate.mark(0)
    assert gate.allow("chatty", True, False, False, 30_000) is False
    assert gate.allow("chatty", True, False, False, 59_999) is False
    assert gate.allow("chatty", True, False, False, 60_000) is True
    assert gate.allow("chatty", True, True, False, 60_000) is False
    assert gate.allow("chatty", False, False, False, 60_000) is False
