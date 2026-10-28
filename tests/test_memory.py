from datetime import datetime, timedelta, timezone

import pytest

from nuonuo.memory import Memory, reject_stored_media
from nuonuo.summary import parse_summary_body, summarize_session
from tests.support import load_fixture


def _record(text: str = "smiling 50% of last 10 min, away twice") -> dict:
    return {
        "user_name": "Andy",
        "transcript": [{"role": "user", "text": "the oral exam is tomorrow", "t_ms": 1000}],
        "notes": [{"t_ms": 0, "text": "[camera note, do not read aloud] Looking at you. Tired"}],
        "spans": [{"state": "tired", "start_ms": 0, "end_ms": 1000}],
        "aggregates": {"text": text, "smile_pct": 0, "away_count": 0, "down_count": 0, "observed_ms": 1000},
        "summary": None,
    }


def test_memory_rejects_landmark_payloads(tmp_path):
    memory = Memory(tmp_path / "nuonuo.db")
    points = [[0.1, 0.2, 0.3] for _ in range(478)]
    with pytest.raises(ValueError):
        memory.save_session({**_record(), "landmarks": points})
    with pytest.raises(ValueError):
        reject_stored_media({"mesh": points})
    with pytest.raises(ValueError):
        memory.save_session({"frames": points, "transcript": []})
    with pytest.raises(ValueError):
        memory.save_session({"trace": points})
    assert memory.recent_sessions() == []


def test_memory_deletes_rows_older_than_30_days(tmp_path):
    memory = Memory(tmp_path / "nuonuo.db")
    now = datetime(2026, 10, 28, tzinfo=timezone.utc)
    memory.save_session(_record("old"), created_at=now - timedelta(days=31))
    memory.save_session(_record("kept"), created_at=now - timedelta(days=1))
    assert memory.purge_expired(now) == 1
    left = memory.recent_sessions()
    assert len(left) == 1
    assert left[0]["aggregates"]["text"] == "kept"


def test_forget_wipes_sessions(tmp_path, monkeypatch):
    monkeypatch.setenv("NUONUO_HOME", str(tmp_path))
    memory = Memory(tmp_path / "data" / "nuonuo.db")
    memory.save_session(_record())
    from nuonuo.app import main

    main(["--forget"])
    assert Memory(tmp_path / "data" / "nuonuo.db").recent_sessions() == []


def test_missing_key_saves_without_a_summary(tmp_path):
    memory = Memory(tmp_path / "nuonuo.db")
    record = _record()
    record["summary"] = summarize_session("", "gpt-6-luna", "Andy", record)
    assert record["summary"] is None
    memory.save_session(record)
    assert memory.recent_sessions()[0]["summary"] is None
    failed = summarize_session("sk-test", "gpt-6-luna", "Andy", record, transport=lambda _payload: (_ for _ in ()).throw(RuntimeError("down")))
    assert failed is None


def test_summary_parser_reads_the_responses_shape():
    parsed = parse_summary_body(load_fixture("summary_response.json"))
    assert parsed["summary"] == "Last time Andy was tired and talked about his oral exam."
    assert parsed["topics"] == ["oral exam"]
    assert parsed["mood_trend"] == "tired"
    assert parsed["follow_ups"] == ["ask how the exam went"]


def test_memory_lines_are_oldest_first(tmp_path):
    memory = Memory(tmp_path / "nuonuo.db")
    now = datetime(2026, 10, 28, tzinfo=timezone.utc)
    first = _record()
    first["summary"] = {"summary": "Last time Andy was tired and talked about his oral exam.", "topics": [], "mood_trend": "tired", "follow_ups": []}
    second = _record()
    second["summary"] = None
    second["aggregates"] = {"text": "smiling 10% of last 10 min"}
    memory.save_session(first, created_at=now - timedelta(days=2))
    memory.save_session(second, created_at=now - timedelta(days=1))
    assert memory.memory_lines() == [
        "Last time Andy was tired and talked about his oral exam.",
        "smiling 10% of last 10 min",
    ]
