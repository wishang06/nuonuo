"""Wake gate, idle timeout, per-session cap, daily cap, and the usage file."""

import asyncio
from datetime import date
from pathlib import Path

import pytest

from nuonuo.gate import ASLEEP, AWAKE, CAPPED, DailyUsage, GateConfig, VoiceGate
from tests.live_fakes import FakeServer, Talk, make_client, run_until, wait_until

ROOT = Path(__file__).resolve().parents[1]


class Today:
    def __init__(self, day: date):
        self.day = day

    def __call__(self) -> date:
        return self.day


def feed(gate: VoiceGate, start: float, end: float, step: float = 0.2, **reading) -> str:
    """Observe the same reading every `step` seconds from start to end inclusive."""
    t = start
    state = gate.state
    while t <= end + 1e-9:
        state = gate.observe(t, **reading)
        t += step
    return state


def test_wakes_only_after_a_face_has_been_present_for_the_wake_time():
    gate = VoiceGate(GateConfig(wake_seconds=1.5))
    assert feed(gate, 0.0, 1.4, face=True) == ASLEEP
    assert gate.observe(1.5, face=True) == AWAKE


def test_a_gap_in_the_face_restarts_the_wake_timer():
    gate = VoiceGate(GateConfig(wake_seconds=1.5))
    feed(gate, 0.0, 1.0, face=True)
    gate.observe(1.2, face=False)
    assert feed(gate, 1.4, 2.8, face=True) == ASLEEP
    assert gate.observe(3.0, face=True) == AWAKE


def test_with_no_camera_speech_near_the_mic_wakes_her():
    gate = VoiceGate(GateConfig(voice_wake_seconds=0.5))
    assert feed(gate, 0.0, 10.0, face=None) == ASLEEP
    assert gate.status() == "sleeping (talk to wake)"
    assert feed(gate, 10.2, 10.6, face=None, heard=True) == ASLEEP
    assert gate.observe(10.8, face=None, heard=True) == AWAKE


def test_idle_close_needs_no_face_and_no_speech_for_the_whole_timeout():
    gate = VoiceGate(GateConfig(wake_seconds=0, idle_seconds=180))
    assert gate.observe(0.0, face=True) == AWAKE
    gate.session_opened(0.0)
    # Away from the camera but still talking: stays open.
    assert feed(gate, 10.0, 150.0, step=10.0, face=False, speech=True) == AWAKE
    assert feed(gate, 160.0, 329.0, step=1.0, face=False) == AWAKE
    assert gate.observe(330.0, face=False) == ASLEEP
    assert gate.reason == "idle"
    assert gate.session_closed(331.0) == 331.0
    # Coming back wakes her again through the normal wake gate.
    assert gate.observe(340.0, face=True) == AWAKE


def test_session_cap_closes_and_the_next_session_needs_a_fresh_wake():
    gate = VoiceGate(GateConfig(wake_seconds=1.5, session_cap_seconds=40 * 60, daily_cap_seconds=0))
    feed(gate, 0.0, 1.6, face=True)
    gate.session_opened(2.0)
    assert feed(gate, 2.0, 2401.0, step=1.0, face=True, speech=True) == AWAKE
    assert gate.observe(2402.0, face=True) == ASLEEP
    assert gate.reason == "session_cap"
    # Still open on the wire: the gate does not wake again until the old session is closed.
    assert feed(gate, 2402.2, 2405.0, face=True) == ASLEEP
    assert gate.session_closed(2405.0) == 2403.0
    assert gate.usage.used() == 2403.0
    # The face has been in view for the wake time since the cap, so the next session may open.
    assert gate.observe(2405.2, face=True) == AWAKE


def test_the_wake_timer_restarts_at_the_session_cap():
    gate = VoiceGate(GateConfig(wake_seconds=1.5, session_cap_seconds=60, daily_cap_seconds=0))
    feed(gate, 0.0, 2.0, face=True)
    gate.session_opened(2.0)
    assert feed(gate, 2.2, 61.8, face=True) == AWAKE
    assert gate.observe(62.0, face=True) == ASLEEP
    gate.session_closed(62.1)
    assert feed(gate, 62.2, 63.6, face=True) == ASLEEP
    assert gate.observe(64.0, face=True) == AWAKE


def test_daily_cap_counts_the_open_session_and_lifts_the_next_day():
    today = Today(date(2026, 11, 12))
    usage = DailyUsage(today=today)
    usage.add(115 * 60)
    gate = VoiceGate(GateConfig(wake_seconds=0, session_cap_seconds=0, daily_cap_seconds=120 * 60), usage)
    assert gate.observe(0.0, face=True) == AWAKE
    gate.session_opened(0.0)
    assert feed(gate, 10.0, 290.0, step=10.0, face=True, speech=True) == AWAKE
    assert gate.observe(299.0, face=True, speech=True) == AWAKE
    assert gate.observe(300.0, face=True, speech=True) == CAPPED
    assert gate.reason == "daily_cap"
    gate.session_closed(301.0)
    assert usage.used() == 120 * 60 + 1
    assert feed(gate, 302.0, 400.0, step=10.0, face=True) == CAPPED
    assert gate.status() == "resting: daily cap"
    today.day = date(2026, 11, 13)
    assert gate.observe(410.0, face=True) == AWAKE


def test_a_long_session_is_booked_as_it_goes(tmp_path):
    usage = DailyUsage(tmp_path / "voice_usage.json")
    gate = VoiceGate(GateConfig(wake_seconds=0, checkpoint_seconds=60), usage)
    gate.observe(0.0, face=True)
    gate.session_opened(0.0)
    feed(gate, 0.0, 125.0, step=5.0, face=True)
    assert usage.used() == 120.0
    # A crash here would lose at most one checkpoint. A restart reads the same total back.
    assert DailyUsage(tmp_path / "voice_usage.json").used() == 120.0
    assert gate.session_closed(130.0) == 130.0
    assert DailyUsage(tmp_path / "voice_usage.json").used() == 130.0


def test_caps_and_timeouts_come_from_env_and_zero_turns_one_off():
    cfg = GateConfig.from_env(
        {"WAKE_SECONDS": "2", "IDLE_TIMEOUT_MIN": "5", "SESSION_CAP_MIN": "0", "DAILY_CAP_MIN": "lots"}
    )
    assert cfg == GateConfig(wake_seconds=2, idle_seconds=300, session_cap_seconds=0, daily_cap_seconds=7200)
    assert GateConfig.from_env({}) == GateConfig()
    assert "session cap off" in cfg.describe()
    gate = VoiceGate(GateConfig(wake_seconds=0, idle_seconds=0, session_cap_seconds=0, daily_cap_seconds=0))
    gate.observe(0.0, face=True)
    gate.session_opened(0.0)
    assert feed(gate, 0.0, 10 * 3600.0, step=50.0, face=False) == AWAKE


def test_env_example_documents_every_gate_setting():
    keys = {
        line.split("=", 1)[0].strip()
        for line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
        if "=" in line and not line.lstrip().startswith("#")
    }
    assert {"WAKE_SECONDS", "IDLE_TIMEOUT_MIN", "SESSION_CAP_MIN", "DAILY_CAP_MIN"} <= keys


def test_client_waits_for_a_face_before_connecting(tmp_path):
    server = FakeServer([Talk()])
    client = make_client(tmp_path, server, cfg=GateConfig(wake_seconds=0.3))
    client.state.set(face_present=False)

    async def scenario():
        task = asyncio.create_task(client.run())
        await asyncio.sleep(0.6)
        assert server.connects == 0
        assert client.state.snapshot()["status"] == "sleeping"
        client.state.set(face_present=True)
        await wait_until(client.started.is_set)
        client.state.stop.set()
        await asyncio.wait_for(task, 5)

    asyncio.run(scenario())
    assert server.connects == 1


def test_client_closes_when_idle_and_greets_on_return(tmp_path):
    server = FakeServer([Talk(), Talk()])
    client = make_client(tmp_path, server, cfg=GateConfig(wake_seconds=0, idle_seconds=0.3))

    async def scenario():
        task = asyncio.create_task(client.run())
        await wait_until(client.started.is_set)
        client.state.set(face_present=False)
        await wait_until(lambda: client.logger.voice_sessions)
        await wait_until(lambda: client.state.snapshot()["status"] == "sleeping")
        assert server.connects == 1
        client.state.set(face_present=True)
        await wait_until(lambda: len(server.starts) == 2 and client.started.is_set())
        client.state.stop.set()
        await asyncio.wait_for(task, 5)

    asyncio.run(scenario())
    assert client.logger.voice_sessions[0]["reason"] == "idle"
    assert client.sleeps == []
    assert "stepped away" in server.instructions()[1]


def test_client_rotates_at_the_session_cap_without_a_greeting(tmp_path):
    server = FakeServer([Talk(), Talk()])
    client = make_client(tmp_path, server, cfg=GateConfig(wake_seconds=0, session_cap_seconds=0.3))

    asyncio.run(run_until(client, lambda: len(server.starts) == 2 and client.started.is_set()))

    first = client.logger.voice_sessions[0]
    assert first["reason"] == "session_cap"
    assert first["minutes"] > 0
    assert client.sleeps == []
    assert "Do not greet Andy" in server.instructions()[1]


def test_client_stops_for_the_day_at_the_daily_cap(tmp_path):
    usage = DailyUsage(tmp_path / "voice_usage.json")
    usage.add(120 * 60 - 0.3)
    server = FakeServer([Talk()])
    client = make_client(tmp_path, server, usage=usage)

    async def scenario():
        task = asyncio.create_task(client.run())
        await wait_until(lambda: client.state.snapshot()["status"] == "resting: daily cap")
        await asyncio.sleep(0.5)
        client.state.stop.set()
        await asyncio.wait_for(task, 5)

    asyncio.run(scenario())
    assert server.connects == 1
    assert client.logger.voice_sessions[0]["reason"] == "daily_cap"
    assert usage.used() >= 120 * 60


def test_laptop_sleep_is_not_booked_as_voice_time_or_idle_time():
    # time.monotonic() keeps counting through Windows Modern Standby, which suspends the app.
    gate = VoiceGate(GateConfig(wake_seconds=0, idle_seconds=180, session_cap_seconds=2400, daily_cap_seconds=7200))
    gate.observe(0.0, face=True)
    gate.session_opened(0.0)
    feed(gate, 0.0, 30.0, step=0.2, face=True)
    # Lid closed for 90 minutes with the session open. The first reading after waking:
    assert gate.observe(30.0 + 5400, face=False) == AWAKE
    assert gate.resumed_at == 30.0 + 5400
    assert gate.daily_left(5431.0) > 7200 - 60
    assert gate.session_closed(5432.0) == pytest.approx(32.0, abs=0.5)
    assert gate.usage.used() == pytest.approx(32.0, abs=0.5)


def test_a_stale_camera_reading_is_treated_as_no_camera(tmp_path):
    client = make_client(tmp_path, FakeServer())
    now = client._now()
    client.state.set(face_present=True, vision_ts=now)
    client._observe_gate()
    assert client.gate.camera is True
    client.state.set(vision_ts=now - 10)
    client._observe_gate()
    assert client.gate.camera is False


def test_client_reopens_the_session_after_the_laptop_wakes(tmp_path):
    server = FakeServer([Talk(), Talk()])
    usage = DailyUsage(tmp_path / "voice_usage.json")
    client = make_client(tmp_path, server, usage=usage)
    offset = {"s": 0.0}
    real_now = client._now
    client._now = lambda: real_now() + offset["s"]

    async def scenario():
        task = asyncio.create_task(client.run())
        await wait_until(client.started.is_set)
        await asyncio.sleep(0.3)
        offset["s"] = 3600.0  # an hour of standby passes between two readings
        await wait_until(lambda: len(server.starts) == 2 and client.started.is_set())
        client.state.stop.set()
        await asyncio.wait_for(task, 5)

    asyncio.run(scenario())
    first = client.logger.voice_sessions[0]
    assert first["reason"] == "resumed"
    assert first["minutes"] < 1
    assert usage.used() < 60
    assert client.sleeps == [1.0]
    assert "Do not greet Andy" in server.instructions()[1]
