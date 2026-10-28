import json

from nuonuo.notes import camera_call_ids, camera_tool_reply
from nuonuo.persona import build_instructions
from nuonuo.timeline import read_camera
from tests.support import load_fixture, replay


def test_read_camera_handler_output_and_reply_events():
    timeline = replay("long_session.json")
    result = read_camera(timeline, 600_000)
    assert set(result) == {"current", "summary"}
    assert result["current"] == "Looking at you. Smiling"
    assert "away twice" in result["summary"]

    event = load_fixture("read_camera_done.json")
    assert camera_call_ids(event) == ["call_test_camera"]
    reply = camera_tool_reply("call_test_camera", result)
    assert reply[0]["type"] == "conversation.item.create"
    assert reply[0]["item"]["type"] == "function_call_output"
    assert reply[0]["item"]["call_id"] == "call_test_camera"
    assert json.loads(reply[0]["item"]["output"]) == result
    assert reply[1] == {"type": "response.create"}


def test_persona_knows_she_can_see_and_remembers():
    text = build_instructions(
        "Andy",
        ["Last time Andy was tired and talked about his oral exam."],
        "quiet",
    )
    assert "read_camera" in text
    assert "[camera note, do not read aloud]" in text
    assert "Never say you can't see Andy." in text
    assert "Last time Andy was tired and talked about his oral exam." in text
    assert "Quiet mode" in text
    chatty = build_instructions("Andy", mode="chatty")
    assert "once a minute" in chatty
    assert "never while Andy is speaking" in chatty
