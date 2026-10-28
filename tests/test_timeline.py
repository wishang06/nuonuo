from nuonuo.clock import SessionClock, format_clock
from nuonuo.timeline import RECENT_MS, read_camera
from tests.support import load_fixture, replay


def test_identical_states_merge_into_one_span():
    timeline = replay("steady_smile.json")
    spans = timeline.recent_spans()
    assert len(spans) == 1
    assert spans[0].state == "smiling"
    assert spans[0].start_ms == 0
    assert spans[0].end_ms == 8000


def test_long_run_keeps_recent_spans_and_folds_the_rest():
    timeline = replay("long_session.json")
    end = 600_000
    spans = timeline.recent_spans()
    assert spans
    assert all(span.end_ms > end - RECENT_MS for span in spans)
    assert len(spans) < len(load_fixture("long_session.json")["steps"])
    text = timeline.aggregate_text(end)
    assert "smiling 50% of last 10 min" in text
    assert "away twice" in text


def test_older_than_ten_minutes_drops_out_of_the_aggregate():
    timeline = replay("bounded_session.json")
    end = 1_200_000
    spans = timeline.recent_spans()
    assert len(spans) == 1
    assert spans[0].start_ms >= end - RECENT_MS
    assert timeline.minute_count() <= 10
    text = timeline.aggregate_text(end)
    assert "smiling 50% of last 10 min" in text
    assert "away 5 times" in text
    assert "away 10 times" not in text


def test_clock_is_the_only_source_of_t_ms():
    clock = SessionClock(start=1000.0)
    assert clock.t_ms(1000.0) == 0
    assert clock.t_ms(1001.5) == 1500
    assert format_clock(65_000) == "1:05"
    assert format_clock(3_661_000) == "1:01:01"


def test_read_camera_shape_from_fixture():
    timeline = replay("long_session.json")
    result = read_camera(timeline, 600_000)
    assert set(result) == {"current", "summary"}
    assert isinstance(result["current"], str) and result["current"]
    assert isinstance(result["summary"], str)
    assert "away twice" in result["summary"]
    assert "landmark" not in result["summary"].lower()
    assert not any(isinstance(value, list) for value in result.values())
