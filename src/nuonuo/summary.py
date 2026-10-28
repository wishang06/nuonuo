"""End-of-session summary via the Responses API. Failure leaves the summary empty."""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request

log = logging.getLogger("nuonuo.summary")

DEFAULT_SUMMARY_MODEL = "gpt-6-luna"

SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "topics": {"type": "array", "items": {"type": "string"}},
        "mood_trend": {"type": "string"},
        "follow_ups": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "topics", "mood_trend", "follow_ups"],
    "additionalProperties": False,
}


def _usable_key(api_key: str | None) -> bool:
    if not api_key:
        return False
    return not api_key.startswith("sk-your")


def parse_summary_body(body: dict) -> dict | None:
    text = body.get("output_text")
    if not isinstance(text, str) or not text.strip():
        text = None
        for item in body.get("output") or []:
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            for part in item.get("content") or []:
                if isinstance(part, dict) and part.get("type") in ("output_text", "text") and part.get("text"):
                    text = part["text"]
                    break
            if text:
                break
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    if not all(key in parsed for key in ("summary", "topics", "mood_trend", "follow_ups")):
        return None
    if not isinstance(parsed["summary"], str) or not isinstance(parsed["mood_trend"], str):
        return None
    if not isinstance(parsed["topics"], list) or not isinstance(parsed["follow_ups"], list):
        return None
    return {
        "summary": parsed["summary"],
        "topics": [str(item) for item in parsed["topics"]],
        "mood_trend": parsed["mood_trend"],
        "follow_ups": [str(item) for item in parsed["follow_ups"]],
    }


def summary_request(model: str, user_name: str, record: dict) -> dict:
    transcript = record.get("transcript") or []
    aggregates = record.get("aggregates") or {}
    notes = record.get("notes") or []
    lines = []
    for turn in transcript:
        if isinstance(turn, dict) and turn.get("text"):
            lines.append(f"{turn.get('role', 'user')}: {turn['text']}")
    note_lines = []
    for note in notes:
        if isinstance(note, dict) and note.get("text"):
            note_lines.append(note["text"])
    camera = aggregates.get("text") if isinstance(aggregates, dict) else ""
    prompt = (
        f"Write a short memory of this session with {user_name} for a companion who will see them again.\n"
        "Use only the transcript, camera notes, and camera summary. Do not invent facts.\n"
        f"summary: one or two sentences, like \"Last time {user_name} was tired and talked about his oral exam.\"\n"
        "topics: short labels for what you talked about.\n"
        "mood_trend: how they seemed, hedged, from the conversation and the camera summary.\n"
        "follow_ups: up to 3 things worth asking next time.\n\n"
        f"Camera summary: {camera or 'none'}\n\n"
        "Camera notes:\n"
        + ("\n".join(note_lines) if note_lines else "(none)")
        + "\n\nTranscript:\n"
        + ("\n".join(lines) if lines else "(none)")
    )
    return {
        "model": model,
        "store": False,
        "reasoning": {"effort": "low"},
        "input": [{"role": "user", "content": prompt}],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "session_summary",
                "strict": True,
                "schema": SUMMARY_SCHEMA,
            }
        },
    }


def _post(api_key: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=data,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def summarize_session(
    api_key: str | None,
    model: str,
    user_name: str,
    record: dict,
    transport=None,
) -> dict | None:
    if not _usable_key(api_key):
        return None
    payload = summary_request(model or DEFAULT_SUMMARY_MODEL, user_name, record)
    try:
        body = transport(payload) if transport is not None else _post(api_key, payload)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError, ValueError) as exc:
        log.warning("session summary failed: %s", exc)
        return None
    except Exception as exc:
        log.warning("session summary failed: %s", exc)
        return None
    if not isinstance(body, dict):
        return None
    return parse_summary_body(body)
