"""Shutdown: the final save, the session log, and voice minutes survive Ctrl+C and cancellation."""

import _thread
import asyncio
import json
import signal
import sqlite3
import threading
import time

import pytest

from nuonuo import app
from nuonuo.clock import SessionClock
from nuonuo.gate import DailyUsage
from nuonuo.logs import SessionLogger
from nuonuo.shutdown import run_to_completion
from nuonuo.timeline import BehaviourTimeline
from tests.live_fakes import FakeServer, Talk, make_client, wait_until
from tests.test_reconnect import TURNS


class StoppedVision:
    def join(self, timeout=None):
        pass


class RecordingMemory:
    def __init__(self, gate: threading.Event | None = None, started: threading.Event | None = None):
        self.saved: list[dict] = []
        self.gate = gate
        self.started = started

    def save_session(self, record: dict) -> str:
        if self.started is not None:
            self.started.set()
        if self.gate is not None:
            self.gate.wait(5)
        self.saved.append(record)
        return "row"


def _timeline() -> BehaviourTimeline:
    timeline = BehaviourTimeline()
    timeline.observe(0, "present")
    timeline.observe(5_000, "smiling")
    return timeline


def test_cancelling_the_voice_client_mid_session_books_minutes_and_closes_cleanly(tmp_path):
    server = FakeServer([Talk(TURNS)])
    usage = DailyUsage(tmp_path / "voice_usage.json")
    client = make_client(tmp_path, server, usage=usage)

    async def scenario():
        task = asyncio.create_task(client.run())
        await wait_until(client.started.is_set)
        await asyncio.sleep(0.2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    assert usage.used() >= 0.2
    assert DailyUsage(tmp_path / "voice_usage.json").used() == usage.used()
    assert client.logger.voice_sessions[-1]["reason"] == "cancelled"
    # The server is still told to close, so it does not hold the session against the next start.
    assert any(event["type"] == "session.close" for event in server.sockets[0].sent)
    assert client.mic.events == ["start", "stop", "close"]
    assert client.player.closed


def test_ctrl_c_during_the_final_save_still_saves_and_writes_the_log(tmp_path):
    release = threading.Event()
    saving = threading.Event()
    memory = RecordingMemory(gate=release, started=saving)
    slog = SessionLogger(tmp_path / "logs", "gpt-live-1")

    def press_ctrl_c():
        saving.wait(5)
        time.sleep(0.1)
        _thread.interrupt_main()  # what Ctrl+C does to the main thread
        time.sleep(0.6)
        release.set()

    previous = signal.signal(signal.SIGINT, signal.default_int_handler)
    presser = threading.Thread(target=press_ctrl_c)
    presser.start()
    interrupted = False
    try:
        try:
            run_to_completion(
                app._finish, StoppedVision(), None, None, memory, SessionClock(), _timeline(), "Andy", None, slog
            )
        except KeyboardInterrupt:
            interrupted = True
        # Checked the moment the interrupt surfaces, before anything below lets the save go on.
        saved_by_then = len(memory.saved)
        log_by_then = slog.session_path.exists()
    finally:
        signal.signal(signal.SIGINT, previous)
        release.set()
        presser.join()

    assert interrupted, "Ctrl+C should still be reported after the save"
    assert saved_by_then == 1
    assert memory.saved[0]["spans"]
    assert log_by_then


def test_the_session_log_is_written_even_when_the_save_fails(tmp_path):
    class LockedMemory:
        def save_session(self, record):
            raise sqlite3.OperationalError("database is locked")

    slog = SessionLogger(tmp_path / "logs", "gpt-live-1")
    summary = app._finish(StoppedVision(), None, None, LockedMemory(), SessionClock(), _timeline(), "Andy", None, slog)
    assert slog.session_path.exists()
    assert json.loads(slog.session_path.read_text(encoding="utf-8"))["end"] == summary["end"]


def test_shutdown_books_a_voice_session_the_voice_thread_did_not_close(tmp_path):
    usage = DailyUsage(tmp_path / "voice_usage.json")
    client = make_client(tmp_path, FakeServer(), usage=usage)
    opened = client._now() - 90
    client._started_at = opened
    client.gate.session_opened(opened)
    client.state.stop.set()
    memory = RecordingMemory()

    summary = app._finish(StoppedVision(), None, client, memory, SessionClock(), _timeline(), "Andy", None, client.logger)

    assert 90 <= usage.used() < 95
    assert summary["voice_sessions"][0]["reason"] == "stop"
    assert summary["voice_minutes"] == pytest.approx(1.5, abs=0.1)
    assert len(memory.saved) == 1
    # Calling it again, as the voice thread would when it finally exits, does not double-count.
    client.finish()
    assert usage.used() < 95


def test_run_to_completion_returns_the_result_and_passes_errors_through():
    assert run_to_completion(lambda a, b: a + b, 2, 3) == 5

    def boom():
        raise ValueError("save failed")

    with pytest.raises(ValueError):
        run_to_completion(boom)
