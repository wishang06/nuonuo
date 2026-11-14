"""Auto-reconnect: backoff, recap seeding, and keeping the mic and clock across sessions."""

import asyncio

from nuonuo import realtime
from nuonuo.persona import ConversationRecap, build_instructions, resume_note
from nuonuo.realtime import Backoff
from tests.live_fakes import Deny, Fail, FakePlayer, FakeServer, Talk, make_client, run_until, wait_until

TURNS = [
    ("user", "I want to map my whole flat with the robot vacuum"),
    ("nuonuo", "Ooh, a little SLAM map of your home!"),
]


def test_backoff_doubles_to_a_30_second_cap_and_resets():
    backoff = Backoff()
    assert [backoff.next() for _ in range(8)] == [1, 2, 4, 8, 16, 30, 30, 30]
    backoff.reset()
    assert backoff.next() == 1


def test_drop_reopens_with_backoff_and_seeds_the_recap(tmp_path):
    server = FakeServer([Talk(TURNS, then="drop"), Fail(), Talk()])
    client = make_client(tmp_path, server)
    clock_start = client.clock.start
    FakePlayer.made = 0

    asyncio.run(run_until(client, lambda: len(server.starts) == 2 and client.started.is_set()))

    assert client.sleeps == [1.0, 2.0]
    first, second = server.instructions()
    assert "Picking up where you left off" not in first
    assert "Do not greet Andy" in second
    assert "Andy: I want to map my whole flat with the robot vacuum" in second
    assert "Nuonuo: Ooh, a little SLAM map of your home!" in second
    assert 'Current topic, in Andy\'s words: "I want to map my whole flat with the robot vacuum"' in second
    # One mic stream and one speaker for the whole run, closed only at the end. Same clock throughout.
    assert client.mic.opened == 1
    assert client.mic.events == ["start", "stop", "close"]
    assert FakePlayer.made == 1
    assert client.clock.start == clock_start
    # The transcript keeps everything said across sessions, for the end-of-day save.
    assert [turn["role"] for turn in client.transcript] == ["user", "nuonuo"]


def test_server_side_expiry_reconnects_after_one_second_without_a_greeting(tmp_path):
    server = FakeServer([Talk(TURNS, then="expire"), Talk()])
    client = make_client(tmp_path, server)

    asyncio.run(run_until(client, lambda: len(server.starts) == 2 and client.started.is_set()))

    assert client.sleeps == [1.0]
    assert "Do not greet Andy" in server.instructions()[1]
    reasons = [entry["reason"] for entry in client.logger.voice_sessions]
    assert reasons[0] == "server_closed"


def test_repeated_failures_back_off_to_the_cap_and_forbidden_is_retried(tmp_path):
    server = FakeServer([Fail()] * 6 + [Deny(), Talk()])
    client = make_client(tmp_path, server)

    asyncio.run(run_until(client, lambda: client.started.is_set()))

    assert client.sleeps == [1, 2, 4, 8, 16, 30, 30]
    # Nothing was ever said, so the session that finally opens is a normal first session.
    assert all("Picking up where you left off" not in text for text in server.instructions())


def test_a_session_that_stayed_up_resets_the_backoff(tmp_path, monkeypatch):
    monkeypatch.setattr(realtime, "BACKOFF_RESET_AFTER", 0.1)
    server = FakeServer([Fail(), Fail(), Talk(then="drop", after=0.3), Fail(), Talk()])
    client = make_client(tmp_path, server)

    asyncio.run(run_until(client, lambda: len(server.starts) == 2 and client.started.is_set()))

    assert client.sleeps == [1, 2, 1, 2]


def test_stop_closes_the_session_and_does_not_reconnect(tmp_path):
    server = FakeServer([Talk()])
    client = make_client(tmp_path, server)

    asyncio.run(run_until(client, lambda: client.started.is_set()))

    assert server.connects == 1
    assert client.sleeps == []
    assert any(event["type"] == "session.close" for event in server.sockets[0].sent)
    assert client.logger.voice_sessions[-1]["reason"] == "stop"


def test_recap_merges_deltas_keeps_recent_turns_and_a_real_topic():
    recap = ConversationRecap(max_turns=3)
    recap.add("user", "can we talk about", 0, 400)
    recap.add("user", " my oral exam tomorrow", 500, 1200)
    recap.add("nuonuo", "Of course! Nervous?", 2000, 3000)
    recap.add("user", "yeah", 4000, 4200)
    assert recap.lines("Andy") == ["Andy: can we talk about my oral exam tomorrow", "Nuonuo: Of course! Nervous?", "Andy: yeah"]
    assert recap.topic == "can we talk about my oral exam tomorrow"
    recap.add("nuonuo", "You'll do great.", 5000, 6000)
    assert len(recap.lines("Andy")) == 3
    assert recap.lines("Andy")[0].startswith("Nuonuo: Of course")


def test_resume_wording_for_reconnect_and_wake():
    recap = ConversationRecap()
    recap.add("user", "tell me about the moon landing", 0, 1000)
    reconnect = resume_note("Andy", recap, "reconnect")
    assert "Do not greet Andy" in reconnect and "do not mention the connection" in reconnect
    wake = resume_note("Andy", recap, "wake")
    assert "stepped away" in wake and "Do not greet" not in wake
    assert "Andy: tell me about the moon landing" in wake
    assert "Picking up where you left off" not in build_instructions("Andy", recap=recap)


def test_a_refused_voice_falls_back_to_the_default_right_away(tmp_path):
    # OpenAI answers "forbidden: Voice session access denied." for a voice name it does not accept.
    server = FakeServer(voices={"marin", "cedar"})
    client = make_client(tmp_path, server, voice="maple")

    asyncio.run(run_until(client, lambda: client.started.is_set()))

    voices = [start["session"]["audio"]["output"]["voice"] for start in server.starts]
    assert voices == ["maple", "marin"]
    assert client.sleeps == []  # no backoff: the fix is known, so the retry is immediate
    assert client.voice == "marin"


def test_the_default_voice_being_refused_still_backs_off(tmp_path):
    server = FakeServer(voices={"cedar"})
    client = make_client(tmp_path, server, voice="maple")

    async def scenario():
        task = asyncio.create_task(client.run())
        await wait_until(lambda: len(client.sleeps) >= 3)
        client.state.stop.set()
        await asyncio.wait_for(task, 5)

    asyncio.run(scenario())
    assert client.sleeps[:3] == [1, 2, 4]
    assert [start["session"]["audio"]["output"]["voice"] for start in server.starts][:3] == ["maple", "marin", "marin"]
