"""A scripted stand-in for the GPT-Live websocket, mic, and speaker. No network, no audio devices."""

import asyncio
import json

from websockets.exceptions import ConnectionClosedError

from nuonuo.gate import DailyUsage, GateConfig, VoiceGate
from nuonuo.logs import SessionLogger
from nuonuo.realtime import RealtimeClient
from nuonuo.state import SharedState


class Fail:
    """The connection attempt itself fails, as when the network is down."""


class Deny:
    """The server answers session.start with a forbidden error."""


class Talk:
    """The session starts, the scripted turns arrive, then `then` happens.

    then="hold": stay open until the client asks to close.
    then="drop": the socket dies without a close frame.
    then="expire": the server closes the session on its own.
    """

    def __init__(self, turns=(), then="hold", after=0.0):
        self.turns = list(turns)
        self.then = then
        self.after = after


class FakeSocket:
    def __init__(self, server, plan):
        self.server = server
        self.plan = plan
        self.sent: list[dict] = []
        self.inbox: asyncio.Queue = asyncio.Queue()
        self.open = True

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        self.open = False
        return False

    def push(self, item) -> None:
        self.inbox.put_nowait(item)

    async def send(self, raw: str) -> None:
        if not self.open:
            raise ConnectionClosedError(None, None)
        event = json.loads(raw)
        self.sent.append(event)
        kind = event["type"]
        if kind == "session.start":
            self.server.starts.append(event)
            self._on_start()
        elif kind == "session.close":
            self.push({"type": "session.closed", "reason": "close_requested", "usage": {"seconds": 2}})

    def _on_start(self) -> None:
        plan = self.plan
        voice = self.server.starts[-1]["session"]["audio"]["output"]["voice"]
        if isinstance(plan, Deny) or (self.server.voices is not None and voice not in self.server.voices):
            self.push({"type": "error", "error": {"code": "forbidden", "message": "Voice session access denied."}})
            return
        self.push({"type": "session.started", "session": {"id": f"live_test_{len(self.server.starts)}"}})
        t_ms = 1000
        for role, text in plan.turns:
            kind = "session.input_transcript.delta" if role == "user" else "session.output_transcript.delta"
            self.push({"type": kind, "delta": text, "start_ms": t_ms, "end_ms": t_ms + 1500})
            t_ms += 4000
        ending = None
        if plan.then == "drop":
            ending = ConnectionClosedError(None, None)
        elif plan.then == "expire":
            ending = {"type": "session.closed", "reason": "expired", "usage": {"seconds": 30}}
        if ending is not None:
            if plan.after:
                asyncio.get_running_loop().call_later(plan.after, self.push, ending)
            else:
                self.push(ending)

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        item = await self.inbox.get()
        if isinstance(item, BaseException):
            self.open = False
            raise item
        return json.dumps(item)


class FakeServer:
    """Hands out one scripted connection per connect(). Extra connects hold open."""

    def __init__(self, plans=(), voices=None):
        self.plans = list(plans)
        self.voices = voices  # voice names the server accepts; None accepts any, like the plans expect
        self.starts: list[dict] = []
        self.sockets: list[FakeSocket] = []
        self.connects = 0

    def connect(self, url, additional_headers=None, max_size=None):
        self.connects += 1
        plan = self.plans.pop(0) if self.plans else Talk()
        if isinstance(plan, Fail):
            raise OSError("network is unreachable")
        sock = FakeSocket(self, plan)
        self.sockets.append(sock)
        return sock

    def instructions(self) -> list[str]:
        return [start["session"]["instructions"] for start in self.starts]


class FakeMic:
    def __init__(self):
        self.events: list[str] = []
        self.opened = 0

    def factory(self, callback):
        self.opened += 1
        self.callback = callback
        return self

    def start(self):
        self.events.append("start")

    def stop(self):
        self.events.append("stop")

    def close(self):
        self.events.append("close")


class FakePlayer:
    made = 0

    def __init__(self, state):
        FakePlayer.made += 1
        self.closed = False

    def start(self):
        pass

    def push(self, pcm):
        pass

    def interrupt(self):
        pass

    def playing(self) -> bool:
        return False

    def close(self):
        self.closed = True


def make_client(
    tmp_path, server: FakeServer, *, cfg: GateConfig | None = None, usage: DailyUsage | None = None, voice: str = "marin"
):
    """A client wired to fakes. Backoff sleeps are recorded in client.sleeps instead of waited out."""
    state = SharedState()
    state.set(face_present=True)
    slog = SessionLogger(tmp_path / "logs", "gpt-live-1")
    usage = usage or DailyUsage(tmp_path / "voice_usage.json")
    gate = VoiceGate(cfg or GateConfig(wake_seconds=0), usage)
    mic = FakeMic()
    client = RealtimeClient(
        state,
        "sk-test",
        "gpt-live-1",
        voice,
        "Andy",
        slog,
        gate=gate,
        connect=server.connect,
        player_factory=FakePlayer,
        mic_factory=mic.factory,
    )
    client.sleeps = []

    async def record_sleep(seconds: float) -> None:
        client.sleeps.append(seconds)
        await asyncio.sleep(0)

    client._sleep = record_sleep
    client.mic = mic
    return client


async def wait_until(condition, timeout: float = 5.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not condition():
        if loop.time() > deadline:
            raise AssertionError("timed out waiting for the client")
        await asyncio.sleep(0.01)


async def run_until(client, condition, timeout: float = 5.0) -> None:
    """Run the client until condition() holds, then stop it the way the app does."""
    task = asyncio.create_task(client.run())
    try:
        await wait_until(lambda: condition() or task.done(), timeout)
        if task.done():
            task.result()
            raise AssertionError("client stopped early")
    finally:
        client.state.stop.set()
        await asyncio.wait_for(task, timeout)
