"""When a camera note is worth sending, and when Chatty mode may speak."""

import json
from dataclasses import dataclass

HEARTBEAT_MS = 8_000
REMARK_CAP_MS = 60_000
CAMERA_NOTE_PREFIX = "[camera note, do not read aloud]"

READ_CAMERA_TOOL = {
    "type": "function",
    "name": "read_camera",
    "description": (
        "Read what the camera sees right now and a short history of the user's "
        "face and attention. Use this when they ask what you see, how they look, "
        "or what they were doing earlier. The result is plain language, not landmarks."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    },
    "strict": True,
}


@dataclass
class NoteDecision:
    send: bool
    changed: bool


class NoteGate:
    """Send when the note text changes, and again on a fixed heartbeat."""

    def __init__(self, heartbeat_ms: int = HEARTBEAT_MS):
        self.heartbeat_ms = heartbeat_ms
        self.last_text = ""
        self.last_sent_ms = -1

    def consider(self, text: str, now_ms: int) -> NoteDecision:
        body = " ".join(text.split())
        if not body:
            return NoteDecision(False, False)
        changed = body != self.last_text
        due = self.last_sent_ms < 0 or changed or now_ms - self.last_sent_ms >= self.heartbeat_ms
        if not due:
            return NoteDecision(False, False)
        self.last_text = body
        self.last_sent_ms = now_ms
        return NoteDecision(True, changed)


class RemarkGate:
    """Chatty may comment on a visual change at most about once a minute."""

    def __init__(self, cap_ms: int = REMARK_CAP_MS):
        self.cap_ms = cap_ms
        self.last_ms = -1

    def allow(
        self,
        mode: str,
        changed: bool,
        user_speaking: bool,
        assistant_speaking: bool,
        now_ms: int,
    ) -> bool:
        if mode != "chatty" or not changed:
            return False
        if user_speaking or assistant_speaking:
            return False
        if self.last_ms >= 0 and now_ms - self.last_ms < self.cap_ms:
            return False
        return True

    def mark(self, now_ms: int) -> None:
        self.last_ms = now_ms


def format_camera_note(body: str) -> str:
    return f"{CAMERA_NOTE_PREFIX} {body.strip()}"


def unwrap_live_event(event: dict) -> dict:
    """Responses events arrive nested inside a response.event envelope."""
    if event.get("type") == "response.event" and isinstance(event.get("event"), dict):
        return event["event"]
    return event


def camera_call_ids(event: dict) -> list[str]:
    """Pull completed read_camera call ids from a GPT-Live server event."""
    inner = unwrap_live_event(event)
    if inner.get("type") != "response.output_item.done":
        return []
    item = inner.get("item") or {}
    if item.get("type") == "function_call" and item.get("name") == "read_camera" and item.get("call_id"):
        return [item["call_id"]]
    return []


def camera_tool_reply(call_id: str, result: dict) -> list[dict]:
    """Return the tool result, then continue the delegated Responses turn."""
    return [
        {
            "type": "response.item.create",
            "event_id": f"tool_{call_id}",
            "item": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": json.dumps(result),
            },
        },
        {"type": "response.create", "event_id": f"continue_{call_id}"},
    ]


def thinking_append(event_id: str, content: str) -> dict:
    return {
        "type": "session.thinking.append",
        "event_id": event_id,
        "delegation_id": None,
        "content": content,
    }


def commentary_append(event_id: str, content: str) -> dict:
    return {
        "type": "session.commentary.append",
        "event_id": event_id,
        "delegation_id": None,
        "content": content,
    }


def instructions_append(event_id: str, content: str) -> dict:
    return {
        "type": "session.instructions.append",
        "event_id": event_id,
        "delegation_id": None,
        "content": content,
    }
