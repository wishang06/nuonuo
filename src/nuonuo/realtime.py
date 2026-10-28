"""GPT-Live voice session over a primary WebSocket."""

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
from nuonuo.persona import build_backend_instructions, build_instructions, mode_update
from nuonuo.state import SharedState
from nuonuo.timeline import BehaviourTimeline, read_camera

log = logging.getLogger("nuonuo.live")

LIVE_URL = "wss://api.openai.com/v1/live/sessions"
RATE = 24000
CHUNK = 480  # 20 ms
SALIENT_RESPONSE_COOLDOWN = 6.0
USER_SPEECH_GAP = 0.8
NOTE_LIMIT = 1800


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
        self.notes = NoteGate()
        self.remarks = RemarkGate()
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
        self.answered_calls: set[str] = set()
        self._mode_sent: str | None = None
        self._tick_queued = False
        self.player: Player | None = None

    def submit_vision(self, ev: PerceptionEvent) -> None:
        if self.loop and self.out_q:
            self.loop.call_soon_threadsafe(self.out_q.put_nowait, ("vision", ev))

    async def run(self) -> None:
        self.loop = asyncio.get_running_loop()
        self.out_q = asyncio.Queue()
        self.mic_q = asyncio.Queue(maxsize=200)
        headers = {"Authorization": f"Bearer {self.api_key}"}
        self.state.set(status="connecting")
        async with websockets.connect(LIVE_URL, additional_headers=headers, max_size=None) as ws:
            await self._send(ws, self._session_start())
            self.player = Player(self.state)
            self.player.start()

            def mic_cb(indata, frames, time_info, status):
                self.loop.call_soon_threadsafe(_put, bytes(indata))

            def _put(blob: bytes):
                if self.mic_q is not None and not self.mic_q.full():
                    self.mic_q.put_nowait(blob)

            mic = sd.RawInputStream(samplerate=RATE, channels=1, dtype="int16", blocksize=CHUNK, callback=mic_cb)
            mic.start()
            tasks = [
                asyncio.create_task(self._mic_sender(ws)),
                asyncio.create_task(self._out_sender(ws)),
                asyncio.create_task(self._receiver(ws)),
                asyncio.create_task(self._stop_watch(ws)),
                asyncio.create_task(self._tick()),
            ]
            try:
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in pending:
                    task.cancel()
                for task in done:
                    if task.exception():
                        raise task.exception()
            finally:
                mic.stop()
                mic.close()
                self.player.close()
                if not self.started.is_set():
                    self.state.set(status="voice offline")

    def _mode(self) -> str:
        mode = self.state.snapshot()["mode"]
        return mode if mode in ("chatty", "quiet") else "chatty"

    def _session_start(self) -> dict:
        self._mode_sent = self._mode()
        return {
            "type": "session.start",
            "event_id": "event_start",
            "session": {
                "model": self.model,
                "instructions": build_instructions(self.user_name, self.memories, self._mode_sent),
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

    async def _stop_watch(self, ws):
        while not self.state.stop.is_set() and not self.closed:
            await asyncio.sleep(0.1)
        if self.started.is_set() and not self.closed:
            await self._send(ws, {"type": "session.close", "event_id": "event_close"})
            for _ in range(50):
                if self.closed:
                    return
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
        log.info("%s: %s", "You" if role == "user" else "Nuonuo", delta)

    def _mark_started(self) -> None:
        if self.mic_q is not None:
            while True:
                try:
                    self.mic_q.get_nowait()
                except asyncio.QueueEmpty:
                    break
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
                if self.player:
                    self.player.interrupt()
                self.state.set(status="hearing you")
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
                self.state.set(speaking=False, status="listening")
                return
            elif kind == "error":
                err = ev.get("error", ev)
                log.warning("live error: %s", err)


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
