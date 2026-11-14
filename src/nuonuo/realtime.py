"""GPT-Live voice session over a primary WebSocket, reopened when it drops or expires.

The mic stream, speaker, and session clock live for the whole app run. Live sessions open and
close underneath them: the wake gate (gate.py) decides when one should be open, and a dropped or
expired session is reopened with exponential backoff. Each new session.start carries a short recap
(persona.ConversationRecap) so Nuonuo picks up the conversation instead of greeting again.
"""

import asyncio
import base64
import json
import logging
import threading
import time
import uuid

import numpy as np
import sounddevice as sd
import websockets

from nuonuo.clock import SessionClock
from nuonuo.gate import AWAKE, VoiceGate
from nuonuo.logs import SessionLogger
from nuonuo.notes import (
    READ_CAMERA_TOOL,
    NoteGate,
    RemarkGate,
    camera_call_ids,
    camera_tool_reply,
    commentary_append,
    format_camera_note,
    instructions_append,
    thinking_append,
)
from nuonuo.perception import PerceptionEvent
from nuonuo.persona import ConversationRecap, build_backend_instructions, build_instructions, mode_update
from nuonuo.state import SharedState
from nuonuo.timeline import BehaviourTimeline, read_camera

log = logging.getLogger("nuonuo.live")

LIVE_URL = "wss://api.openai.com/v1/live/sessions"
DEFAULT_VOICE = "marin"  # a voice name the server does not accept is refused with "forbidden"
RATE = 24000
CHUNK = 480  # 20 ms
SALIENT_RESPONSE_COOLDOWN = 6.0
USER_SPEECH_GAP = 0.8
NOTE_LIMIT = 1800

BACKOFF_START = 1.0
BACKOFF_MAX = 30.0
BACKOFF_RESET_AFTER = 30.0  # a session that stayed up this long starts the next backoff at 1 s again
START_TIMEOUT = 15.0  # seconds to wait for session.started before treating the attempt as failed
CLOSE_WAIT = 5.0  # seconds to wait for session.closed after asking to close
GATE_POLL = 0.2
MIC_WAKE_RMS = 0.03  # mic level that counts as speech for waking her when there is no camera
HEARD_GAP = 0.3
VISION_STALE = 3.0  # a face reading older than this is ignored (the camera stopped sending frames)

# Session end reasons that reopen with backoff. Gate reasons ("idle", "session_cap", "daily_cap")
# reopen when the gate wakes again, and "stop" ends the run.
# "resumed": the machine slept with a session open. The server has likely dropped it, so reopen now
# rather than wait ~40 s for the websocket keepalive to notice.
RETRY_REASONS = frozenset({"dropped", "server_closed", "denied", "start_timeout", "resumed"})
# Expected ways for a connection to fail. TimeoutError and socket errors are OSError subclasses.
NETWORK_ERRORS = (OSError, websockets.ConnectionClosed, websockets.InvalidHandshake, json.JSONDecodeError)


class Backoff:
    """Reconnect delays: 1 s, 2 s, 4 s, ... capped at 30 s."""

    def __init__(self, start: float = BACKOFF_START, cap: float = BACKOFF_MAX):
        self.start = start
        self.cap = cap
        self.attempt = 0

    def next(self) -> float:
        delay = min(self.cap, self.start * (2 ** self.attempt))
        self.attempt += 1
        return delay

    def reset(self) -> None:
        self.attempt = 0


class Player:
    def __init__(self, state: SharedState):
        self.state = state
        self.buf = bytearray()
        self.lock = threading.Lock()
        self.stream = sd.RawOutputStream(
            samplerate=RATE, channels=1, dtype="int16", blocksize=CHUNK, callback=self._cb
        )

    def _cb(self, outdata, frames, time_info, status):
        n = frames * 2
        with self.lock:
            chunk = bytes(self.buf[:n])
            del self.buf[:n]
            idle = not self.buf and not chunk
        if chunk:
            samples = np.frombuffer(chunk, dtype=np.int16)
            level = float(np.sqrt(np.mean((samples.astype(np.float32) / 32768) ** 2)))
            self.state.set(mouth_level=min(level * 6, 1.0), last_audio_ts=time.monotonic(), speaking=True)
        elif idle:
            self.state.set(mouth_level=0.0, speaking=False)
        outdata[: len(chunk)] = chunk
        if len(chunk) < n:
            outdata[len(chunk):] = b"\x00" * (n - len(chunk))

    def start(self):
        self.stream.start()

    def push(self, pcm: bytes):
        with self.lock:
            self.buf.extend(pcm)

    def interrupt(self) -> None:
        with self.lock:
            self.buf.clear()
        self.state.set(mouth_level=0.0, speaking=False, last_audio_ts=0.0)

    def playing(self) -> bool:
        with self.lock:
            pending = bool(self.buf)
        snap = self.state.snapshot()
        return pending or snap["speaking"]

    def close(self):
        self.stream.stop()
        self.stream.close()


def open_mic(callback):
    return sd.RawInputStream(samplerate=RATE, channels=1, dtype="int16", blocksize=CHUNK, callback=callback)


class RealtimeClient:
    def __init__(
        self,
        state: SharedState,
        api_key: str,
        model: str,
        voice: str,
        user_name: str,
        logger: SessionLogger,
        timeline: BehaviourTimeline | None = None,
        clock: SessionClock | None = None,
        memories: list[str] | None = None,
        backend_model: str = "gpt-6-luna",
        *,
        gate: VoiceGate | None = None,
        connect=None,
        player_factory=Player,
        mic_factory=open_mic,
        backoff: Backoff | None = None,
    ):
        self.state = state
        self.api_key = api_key
        self.model = model
        self.voice = voice
        self.user_name = user_name
        self.logger = logger
        self.timeline = timeline or BehaviourTimeline()
        self.clock = clock or SessionClock()
        self.memories = list(memories or [])
        self.backend_model = backend_model
        self.gate = gate or VoiceGate(now=time.monotonic())
        self.backoff = backoff or Backoff()
        self._connect = connect or websockets.connect
        self._player_factory = player_factory
        self._mic_factory = mic_factory
        self.notes = NoteGate()
        self.remarks = RemarkGate()
        self.recap = ConversationRecap()
        self.transcript: list[dict] = []
        self.note_log: list[dict] = []
        self.loop: asyncio.AbstractEventLoop | None = None
        self.out_q: asyncio.Queue | None = None
        self.mic_q: asyncio.Queue | None = None
        self.started = asyncio.Event()
        self.closed = False
        self.user_speaking = False
        self.last_user_transcript = 0.0
        self.last_salient_response = 0.0
        self.last_heard = float("-inf")
        self.answered_calls: set[str] = set()
        self._mode_sent: str | None = None
        self._tick_queued = False
        self.player: Player | None = None
        self._closing: str | None = None
        self._end_reason: str | None = None
        self._started_at: float | None = None
        self._last_uptime = 0.0
        self._end_lock = threading.Lock()

    def _now(self) -> float:
        return time.monotonic()

    def submit_vision(self, ev: PerceptionEvent) -> None:
        loop = self.loop
        if loop is not None and self.out_q is not None:
            try:
                loop.call_soon_threadsafe(self._put_vision, ev)
            except RuntimeError:
                pass  # loop already closed during shutdown

    def _put_vision(self, ev: PerceptionEvent) -> None:
        # Nothing to tell a session that isn't open. Stale events are dropped, not replayed later.
        if self.out_q is not None and self.started.is_set() and not self.closed:
            self.out_q.put_nowait(("vision", ev))

    def _mic_cb(self, indata, frames, time_info, status):
        blob = bytes(indata)
        if len(blob) >= 2:
            samples = np.frombuffer(blob[: len(blob) // 2 * 2], dtype=np.int16).astype(np.float32) / 32768
            if float(np.sqrt(np.mean(samples**2))) > MIC_WAKE_RMS:
                self.last_heard = self._now()
        loop = self.loop
        if loop is not None:
            try:
                loop.call_soon_threadsafe(self._put_mic, blob)
            except RuntimeError:
                pass

    def _put_mic(self, blob: bytes) -> None:
        if self.mic_q is not None and self.started.is_set() and not self.closed and not self.mic_q.full():
            self.mic_q.put_nowait(blob)

    async def run(self) -> None:
        """Keep a live session open whenever the gate wants one, until the app stops."""
        self.loop = asyncio.get_running_loop()
        self.out_q = asyncio.Queue()
        self.mic_q = asyncio.Queue(maxsize=200)
        self.player = self._player_factory(self.state)
        self.player.start()
        mic = self._mic_factory(self._mic_cb)
        mic.start()
        try:
            await self._supervise()
        finally:
            self.finish()
            try:
                mic.stop()
                mic.close()
            finally:
                self.player.close()

    async def _supervise(self) -> None:
        resume: str | None = None  # how the next session.start frames the recap; None for the first one
        while not self.state.stop.is_set():
            if not await self._wait_until_awake():
                return
            reason = await self._run_session(resume)
            if self.state.stop.is_set() or reason == "stop":
                return
            if reason == "voice_refused":
                continue  # the voice is now the default: try again right away, no backoff
            started = self._last_uptime > 0
            if reason in RETRY_REASONS:
                if self._last_uptime >= BACKOFF_RESET_AFTER:
                    self.backoff.reset()
                delay = self.backoff.next()
                if started:
                    resume = "reconnect"
                label = "voice denied, retry" if reason == "denied" else "reconnecting"
                log.warning("live session ended (%s); reopening in %.0fs", reason, delay)
                self.state.set(status=f"{label} in {delay:.0f}s", speaking=False)
                await self._sleep(delay)
            else:
                self.backoff.reset()
                if started:
                    resume = "reconnect" if reason == "session_cap" else "wake"

    async def _sleep(self, seconds: float) -> None:
        """Sleep, waking early if the app is stopping."""
        end = self._now() + seconds
        while not self.state.stop.is_set():
            left = end - self._now()
            if left <= 0:
                return
            await asyncio.sleep(min(left, 0.1))

    def _observe_gate(self) -> str:
        snap = self.state.snapshot()
        now = self._now()
        face = snap["face_present"] if snap["vision_enabled"] else None
        seen = snap["vision_ts"]
        if face is not None and seen is not None and now - seen > VISION_STALE:
            face = None  # don't keep her awake (or asleep) on a reading from a camera that went quiet
        speech = self.user_speaking or self._assistant_speaking()
        return self.gate.observe(now, face, speech, now - self.last_heard < HEARD_GAP)

    async def _wait_until_awake(self) -> bool:
        shown = None
        while not self.state.stop.is_set():
            if self._observe_gate() == AWAKE:
                return True
            status = self.gate.status()
            if status != shown:
                self.state.set(status=status)
                shown = status
            await asyncio.sleep(GATE_POLL)
        return False

    async def _run_session(self, resume: str | None) -> str:
        """Open one live session and serve it until it ends. Returns why it ended."""
        self.started.clear()
        self.closed = False
        self._closing = None
        self._end_reason = None
        self._last_uptime = 0.0
        self.state.set(status="connecting" if resume is None else "reconnecting")
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            async with self._connect(LIVE_URL, additional_headers=headers, max_size=None) as ws:
                await self._send(ws, self._session_start(resume))
                await self._serve(ws)
                # Book the minutes now. Leaving this block waits for the websocket close handshake,
                # which can take a couple of seconds and is not voice time.
                self._end_session()
        except asyncio.CancelledError:
            if self._end_reason is None:
                self._end_reason = "cancelled"
            raise
        except Exception as exc:
            if self._end_reason is None:
                self._end_reason = "dropped"
                if isinstance(exc, NETWORK_ERRORS):
                    log.warning("live connection lost: %s", str(exc) or type(exc).__name__)
                else:
                    # Still reconnect so the app stays up, but a bug should not hide behind "connection lost".
                    log.exception("live session failed unexpectedly; reconnecting")
        finally:
            self._end_session()
        return self._end_reason or "dropped"

    async def _serve(self, ws) -> None:
        tasks = [
            asyncio.create_task(self._mic_sender(ws)),
            asyncio.create_task(self._out_sender(ws)),
            asyncio.create_task(self._receiver(ws)),
            asyncio.create_task(self._watch(ws)),
            asyncio.create_task(self._tick()),
        ]
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        except asyncio.CancelledError:
            await self._close_quietly(ws)
            raise
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        for task in done:
            if task.exception() is not None:
                raise task.exception()

    async def _close_quietly(self, ws) -> None:
        """On cancel, still ask the server to close so it doesn't hold the session open."""
        self._closing = self._closing or "cancelled"
        if self._end_reason is None:
            self._end_reason = "cancelled"
        if not self.started.is_set() or self.closed:
            return
        try:
            close = {"type": "session.close", "event_id": self._event_id("close")}
            await asyncio.wait_for(asyncio.shield(self._send(ws, close)), 1.0)
        except BaseException:
            pass

    def _end_session(self) -> None:
        """Book the session's minutes and log them. Safe to call more than once, from any thread."""
        with self._end_lock:
            started_at, self._started_at = self._started_at, None
        if started_at is None:
            return
        now = self._now()
        seconds = self.gate.session_closed(now)
        self._last_uptime = max(now - started_at, 1e-6)
        reason = self._end_reason or ("stop" if self.state.stop.is_set() else "dropped")
        self.logger.voice_session_ended(seconds, reason)
        cap = self.gate.cfg.daily_cap_seconds
        log.info(
            "voice session ended (%s): %.1f min used, %.1f of %s min today",
            reason,
            seconds / 60,
            self.gate.usage.used() / 60,
            f"{cap / 60:g}" if cap > 0 else "unlimited",
        )
        self.user_speaking = False
        self.state.set(user_speaking=False)

    def finish(self) -> None:
        """Book an open session's minutes. Called at shutdown in case the voice thread is still closing."""
        self._end_session()

    def _mode(self) -> str:
        mode = self.state.snapshot()["mode"]
        return mode if mode in ("chatty", "quiet") else "chatty"

    def _session_start(self, resume: str | None = None) -> dict:
        self._mode_sent = self._mode()
        instructions = build_instructions(
            self.user_name, self.memories, self._mode_sent, resume=resume, recap=self.recap
        )
        return {
            "type": "session.start",
            "event_id": "event_start",
            "session": {
                "model": self.model,
                "instructions": instructions,
                "audio": {
                    "format": {"type": "audio/pcm", "rate": RATE},
                    "output": {"voice": self.voice},
                },
                "delegation": {
                    "type": "responses",
                    "responses": {
                        "model": self.backend_model,
                        "instructions": build_backend_instructions(self.user_name),
                        "tools": [READ_CAMERA_TOOL],
                        "tool_choice": "auto",
                        "reasoning": {"effort": "low"},
                    },
                },
            },
        }

    async def _send(self, ws, event: dict) -> None:
        await ws.send(json.dumps(event))

    def _event_id(self, prefix: str) -> str:
        return f"{prefix}_{uuid.uuid4().hex[:12]}"

    def _clip(self, text: str) -> str:
        text = text.strip()
        if len(text) <= NOTE_LIMIT:
            return text
        return text[: NOTE_LIMIT - 1].rstrip() + "…"

    async def _watch(self, ws) -> None:
        """Close the session when the app stops or the gate says so, and give up on a start that hangs."""
        opened = self._now()
        while True:
            if self.closed:
                return
            if self.state.stop.is_set():
                reason = "stop"
                break
            if self._observe_gate() != AWAKE:
                reason = self.gate.reason or "idle"
                break
            resumed = self.gate.resumed_at
            if resumed is not None and self._started_at is not None and resumed > self._started_at:
                log.info("woke from sleep with a live session open; reopening it")
                reason = "resumed"
                break
            if not self.started.is_set() and self._now() - opened > START_TIMEOUT:
                log.warning("live session did not start within %.0fs", START_TIMEOUT)
                self._end_reason = "start_timeout"
                return
            await asyncio.sleep(GATE_POLL)
        await self._request_close(ws, reason)

    async def _request_close(self, ws, reason: str) -> None:
        self._closing = reason
        if self._end_reason is None:
            self._end_reason = reason
        if not self.started.is_set() or self.closed:
            return
        log.info("closing live session (%s)", reason)
        await self._send(ws, {"type": "session.close", "event_id": self._event_id("close")})
        deadline = self._now() + CLOSE_WAIT
        while not self.closed and self._now() < deadline:
            await asyncio.sleep(0.1)

    async def _mic_sender(self, ws):
        await self.started.wait()
        while True:
            pcm = await self.mic_q.get()
            if not self.started.is_set() or self.closed:
                continue
            if len(pcm) % 2:
                pcm = pcm[:-1]
            if not pcm:
                continue
            await self._send(ws, {"type": "session.input_audio.append", "audio": base64.b64encode(pcm).decode()})

    async def _tick(self):
        while True:
            await asyncio.sleep(0.25)
            if self.user_speaking and time.monotonic() - self.last_user_transcript > USER_SPEECH_GAP:
                self.user_speaking = False
                self.state.set(user_speaking=False)
                if not (self.player and self.player.playing()):
                    self.state.set(status="listening")
            if self._tick_queued or self.out_q is None:
                continue
            self._tick_queued = True
            self.out_q.put_nowait(("tick", None))

    async def _out_sender(self, ws):
        await self.started.wait()
        while True:
            kind, payload = await self.out_q.get()
            if self.closed:
                continue
            if kind == "vision":
                await self._send_vision(ws, payload)
            elif kind == "tick":
                self._tick_queued = False
                await self._maybe_send_note(ws)
                await self._maybe_update_mode(ws)

    def _assistant_speaking(self) -> bool:
        return bool(self.player and self.player.playing())

    def _can_speak(self) -> bool:
        return not self.user_speaking and not self._assistant_speaking()

    async def _maybe_update_mode(self, ws):
        mode = self._mode()
        if mode == self._mode_sent:
            return
        self._mode_sent = mode
        await self._send(ws, instructions_append(self._event_id("mode"), mode_update(self.user_name, mode)))
        log.info("mode -> %s", mode)

    async def _maybe_send_note(self, ws):
        now_ms = self.clock.t_ms()
        snap = self.state.snapshot()
        body = "Camera off" if not snap["vision_enabled"] else self.timeline.current_note(now_ms)
        decision = self.notes.consider(body, now_ms)
        if not decision.send:
            return
        text = format_camera_note(body)
        remark = self.remarks.allow(
            snap["mode"], decision.changed, self.user_speaking, self._assistant_speaking(), now_ms
        )
        if remark:
            self.remarks.mark(now_ms)
            content = self._clip(
                f"{text} You may comment on this in one short spoken sentence. Do not read the note aloud."
            )
            await self._send(ws, commentary_append(self._event_id("remark"), content))
        else:
            await self._send(ws, thinking_append(self._event_id("camera"), self._clip(text)))
        self.note_log.append({"t_ms": now_ms, "text": text})
        eid = uuid.uuid4().hex[:12]
        self.logger.vision_event(eid, text, False, remark)
        log.info("camera note -> %s%s", text, " (remark)" if remark else "")

    async def _send_vision(self, ws, ev: PerceptionEvent):
        if not ev.salient:
            return
        now = time.monotonic()
        speak = self._can_speak() and now - self.last_salient_response > SALIENT_RESPONSE_COOLDOWN
        eid = uuid.uuid4().hex[:12]
        if speak:
            self.last_salient_response = now
            content = self._clip(f"{ev.text} React in one short spoken line. Do not read this note aloud.")
            await self._send(ws, commentary_append(self._event_id("salient"), content))
        else:
            await self._send(ws, thinking_append(self._event_id("salient"), self._clip(ev.text)))
        self.logger.vision_event(eid, ev.text, True, speak)
        log.info("vision -> %s%s", ev.text, " (remark)" if speak else "")

    async def _answer_camera(self, ws, event: dict):
        for call_id in camera_call_ids(event):
            if call_id in self.answered_calls:
                continue
            self.answered_calls.add(call_id)
            result = read_camera(self.timeline, self.clock.t_ms())
            log.info("read_camera -> %s", result.get("current", ""))
            for message in camera_tool_reply(call_id, result):
                await self._send(ws, message)

    def _remember_transcript(self, role: str, event: dict) -> None:
        delta = event.get("delta") or ""
        if not delta:
            return
        start_ms = int(event.get("start_ms") or self.clock.t_ms())
        end_ms = int(event.get("end_ms") or start_ms)
        if self.transcript and self.transcript[-1]["role"] == role and start_ms - int(self.transcript[-1].get("end_ms", start_ms)) < 1500:
            self.transcript[-1]["text"] += delta
            self.transcript[-1]["end_ms"] = end_ms
        else:
            self.transcript.append({"role": role, "text": delta, "t_ms": start_ms, "end_ms": end_ms})
        self.recap.add(role, delta, start_ms, end_ms)
        log.info("%s: %s", "You" if role == "user" else "Nuonuo", delta)

    def _mark_started(self) -> None:
        for queue in (self.mic_q, self.out_q):
            while queue is not None and not queue.empty():
                queue.get_nowait()
        self._tick_queued = False
        # Send the current camera note on the first tick, so the new session knows what she sees.
        self.notes.rearm()
        now = self._now()
        with self._end_lock:
            self._started_at = now
        self.gate.session_opened(now)
        self.started.set()
        self.state.set(status="listening")

    async def _receiver(self, ws):
        async for raw in ws:
            ev = json.loads(raw)
            kind = ev.get("type", "")
            if kind == "session.started":
                session = ev.get("session") or {}
                log.info("live session started (%s)", session.get("id", ""))
                self._mark_started()
            elif kind == "session.output_audio.delta":
                delta = ev.get("delta")
                if delta and self.player:
                    self.player.push(base64.b64decode(delta))
            elif kind == "session.input_transcript.delta":
                self.user_speaking = True
                self.last_user_transcript = time.monotonic()
                self.state.set(user_speaking=True, status="hearing you")
                if self.player:
                    self.player.interrupt()
                self._remember_transcript("user", ev)
            elif kind == "session.output_transcript.delta":
                self._remember_transcript("nuonuo", ev)
                self.logger.response_started()
            elif kind == "response.event":
                await self._answer_camera(ws, ev)
                inner = ev.get("event") or {}
                if inner.get("type") in ("response.completed", "response.done"):
                    usage = (inner.get("response") or {}).get("usage") or inner.get("usage")
                    if isinstance(usage, dict) and (
                        "input_token_details" in usage or "output_token_details" in usage
                    ):
                        self.logger.add_usage(usage)
            elif kind == "response.output_item.done":
                await self._answer_camera(ws, ev)
            elif kind == "session.usage.updated":
                seconds = (ev.get("usage") or {}).get("seconds")
                if isinstance(seconds, (int, float)):
                    self.logger.set_voice_seconds(float(seconds))
            elif kind == "session.closed":
                seconds = (ev.get("usage") or {}).get("seconds")
                if isinstance(seconds, (int, float)):
                    self.logger.set_voice_seconds(float(seconds))
                log.info("live session closed (%s)", ev.get("reason", ""))
                self.closed = True
                if self._end_reason is None:
                    self._end_reason = self._closing or "server_closed"
                self.state.set(speaking=False)
                return
            elif kind == "error":
                err = ev.get("error", ev)
                code = err.get("code") if isinstance(err, dict) else None
                message = err.get("message") if isinstance(err, dict) else str(err)
                log.warning("live error: %s", err)
                if code == "forbidden":
                    self.closed = True
                    if self.voice != DEFAULT_VOICE:
                        # The server answers "forbidden: Voice session access denied." for any voice
                        # name it does not accept, so try the default voice before giving up.
                        log.warning(
                            "OpenAI refused the voice session (%s) with voice %r. It answers this way for a "
                            "voice name it does not accept. Falling back to %r. Set NUONUO_VOICE=%s in .env "
                            "to skip this.",
                            message,
                            self.voice,
                            DEFAULT_VOICE,
                            DEFAULT_VOICE,
                        )
                        self.voice = DEFAULT_VOICE
                        self._end_reason = "voice_refused"
                    else:
                        log.error(
                            "OpenAI refused the voice session (%s) with the default voice. "
                            "A previous run may still be holding one, or this API key cannot open GPT-Live. "
                            "Retrying with backoff. The avatar keeps running.",
                            message,
                        )
                        self._end_reason = "denied"
                        self.state.set(status="voice denied")
                    return
        if self._end_reason is None:
            self._end_reason = self._closing or "dropped"


def run_realtime_thread(client: RealtimeClient) -> threading.Thread:
    def target():
        try:
            asyncio.run(client.run())
        except Exception as exc:
            log.exception("Live session ended with error: %s", exc)
            client.state.set(status="voice offline")

    thread = threading.Thread(target=target, name="live", daemon=True)
    thread.start()
    return thread
