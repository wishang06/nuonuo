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
    format_camera_note,
    text_item,
)
from nuonuo.perception import PerceptionEvent
from nuonuo.persona import build_instructions
from nuonuo.state import SharedState
from nuonuo.timeline import BehaviourTimeline, read_camera

log = logging.getLogger("nuonuo.realtime")

RATE = 24000
CHUNK = 480  # 20 ms
SALIENT_RESPONSE_COOLDOWN = 6.0


class Player:
    def __init__(self, state: SharedState):
        self.state = state
        self.buf = bytearray()
        self.lock = threading.Lock()
        self.item_id: str | None = None
        self.played_bytes = 0
        self.stream = sd.RawOutputStream(
            samplerate=RATE, channels=1, dtype="int16", blocksize=CHUNK, callback=self._cb
        )

    def _cb(self, outdata, frames, time_info, status):
        n = frames * 2
        with self.lock:
            chunk = bytes(self.buf[:n])
            del self.buf[:n]
            self.played_bytes += len(chunk)
        if chunk:
            samples = np.frombuffer(chunk, dtype=np.int16)
            level = float(np.sqrt(np.mean((samples.astype(np.float32) / 32768) ** 2)))
            self.state.set(mouth_level=min(level * 6, 1.0), last_audio_ts=time.monotonic())
        outdata[: len(chunk)] = chunk
        if len(chunk) < n:
            outdata[len(chunk):] = b"\x00" * (n - len(chunk))

    def start(self):
        self.stream.start()

    def push(self, item_id: str, pcm: bytes):
        with self.lock:
            if item_id != self.item_id:
                self.item_id, self.played_bytes = item_id, 0
            self.buf.extend(pcm)

    def interrupt(self) -> tuple[str | None, int]:
        with self.lock:
            had_audio = bool(self.buf)
            self.buf.clear()
            played_ms = self.played_bytes * 1000 // (RATE * 2)
            item = self.item_id if had_audio else None
        self.state.set(mouth_level=0.0, speaking=False)
        return item, played_ms

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
        self.notes = NoteGate()
        self.remarks = RemarkGate()
        self.transcript: list[dict] = []
        self.note_log: list[dict] = []
        self.loop: asyncio.AbstractEventLoop | None = None
        self.out_q: asyncio.Queue | None = None
        self.response_active = False
        self.response_pending = False
        self.user_speaking = False
        self.last_salient_response = 0.0
        self.answered_calls: set[str] = set()
        self._mode_sent: str | None = None
        self._tick_queued = False

    def submit_vision(self, ev: PerceptionEvent) -> None:
        if self.loop and self.out_q:
            self.loop.call_soon_threadsafe(self.out_q.put_nowait, ("vision", ev))

    async def run(self) -> None:
        self.loop = asyncio.get_running_loop()
        self.out_q = asyncio.Queue()
        url = f"wss://api.openai.com/v1/realtime?model={self.model}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        self.state.set(status="connecting")
        async with websockets.connect(url, additional_headers=headers, max_size=None) as ws:
            await self._send(ws, self._session_update())
            self.player = Player(self.state)
            self.player.start()
            mic_q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=200)

            def mic_cb(indata, frames, time_info, status):
                self.loop.call_soon_threadsafe(_put, bytes(indata))

            def _put(b):
                if not mic_q.full():
                    mic_q.put_nowait(b)

            mic = sd.RawInputStream(samplerate=RATE, channels=1, dtype="int16", blocksize=CHUNK, callback=mic_cb)
            mic.start()
            self.state.set(status="listening")
            tasks = [
                asyncio.create_task(self._mic_sender(ws, mic_q)),
                asyncio.create_task(self._out_sender(ws)),
                asyncio.create_task(self._receiver(ws)),
                asyncio.create_task(self._stop_watch()),
                asyncio.create_task(self._tick()),
            ]
            try:
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for t in pending:
                    t.cancel()
                for t in done:
                    if t.exception():
                        raise t.exception()
            finally:
                mic.stop()
                mic.close()
                self.player.close()

    def _instructions(self) -> str:
        mode = self.state.snapshot()["mode"]
        if mode not in ("chatty", "quiet"):
            mode = "chatty"
        return build_instructions(self.user_name, self.memories, mode)

    def _session_update(self) -> dict:
        mode = self.state.snapshot()["mode"]
        if mode not in ("chatty", "quiet"):
            mode = "chatty"
        self._mode_sent = mode
        return {
            "type": "session.update",
            "session": {
                "type": "realtime",
                "model": self.model,
                "instructions": build_instructions(self.user_name, self.memories, mode),
                "output_modalities": ["audio"],
                "tools": [READ_CAMERA_TOOL],
                "tool_choice": "auto",
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": RATE},
                        "transcription": {"model": "gpt-4o-mini-transcribe"},
                        "turn_detection": {
                            "type": "server_vad",
                            "threshold": 0.5,
                            "prefix_padding_ms": 300,
                            "silence_duration_ms": 500,
                            "create_response": True,
                            "interrupt_response": True,
                        },
                    },
                    "output": {"format": {"type": "audio/pcm", "rate": RATE}, "voice": self.voice},
                },
            },
        }

    async def _send(self, ws, event: dict) -> None:
        await ws.send(json.dumps(event))

    async def _stop_watch(self):
        while not self.state.stop.is_set():
            await asyncio.sleep(0.1)

    async def _mic_sender(self, ws, mic_q: asyncio.Queue):
        while True:
            pcm = await mic_q.get()
            await self._send(ws, {"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm).decode()})

    async def _tick(self):
        while True:
            await asyncio.sleep(0.25)
            if self._tick_queued or self.out_q is None:
                continue
            self._tick_queued = True
            self.out_q.put_nowait(("tick", None))

    async def _out_sender(self, ws):
        while True:
            kind, payload = await self.out_q.get()
            if kind == "vision":
                await self._send_vision(ws, payload)
            elif kind == "tick":
                self._tick_queued = False
                await self._maybe_send_note(ws)
                await self._maybe_update_mode(ws)

    def _can_respond(self) -> bool:
        return not self.response_active and not self.response_pending and not self.user_speaking

    async def _maybe_update_mode(self, ws):
        mode = self.state.snapshot()["mode"]
        if mode == self._mode_sent:
            return
        self._mode_sent = mode
        await self._send(ws, {
            "type": "session.update",
            "session": {"type": "realtime", "instructions": self._instructions()},
        })

    async def _maybe_send_note(self, ws):
        now_ms = self.clock.t_ms()
        snap = self.state.snapshot()
        body = "Camera off" if not snap["vision_enabled"] else self.timeline.current_note(now_ms)
        decision = self.notes.consider(body, now_ms)
        if not decision.send:
            return
        text = format_camera_note(body)
        await self._send(ws, text_item(text))
        self.note_log.append({"t_ms": now_ms, "text": text})
        assistant_speaking = self.response_active or self.response_pending
        remark = self.remarks.allow(
            snap["mode"], decision.changed, self.user_speaking, assistant_speaking, now_ms
        )
        if remark:
            self.remarks.mark(now_ms)
            self.response_pending = True
            await self._send(ws, {"type": "response.create"})
        eid = uuid.uuid4().hex[:12]
        self.logger.vision_event(eid, text, False, remark)
        log.info("camera note -> %s%s", text, " (remark)" if remark else "")

    async def _send_vision(self, ws, ev: PerceptionEvent):
        if not ev.salient:
            return
        now = time.monotonic()
        request = self._can_respond() and now - self.last_salient_response > SALIENT_RESPONSE_COOLDOWN
        eid = uuid.uuid4().hex[:12]
        await self._send(ws, text_item(ev.text))
        if request:
            self.last_salient_response = now
            self.response_pending = True
            await self._send(ws, {"type": "response.create"})
        self.logger.vision_event(eid, ev.text, ev.salient, request)
        log.info("vision -> %s%s", ev.text, " (response requested)" if request else "")

    async def _answer_camera(self, ws, event: dict):
        for call_id in camera_call_ids(event):
            if call_id in self.answered_calls:
                continue
            self.answered_calls.add(call_id)
            result = read_camera(self.timeline, self.clock.t_ms())
            log.info("read_camera -> %s", result.get("current", ""))
            for message in camera_tool_reply(call_id, result):
                if message["type"] == "response.create":
                    self.response_pending = True
                await self._send(ws, message)

    async def _receiver(self, ws):
        async for raw in ws:
            ev = json.loads(raw)
            t = ev.get("type", "")
            if t in ("response.output_audio.delta", "response.audio.delta"):
                self.player.push(ev.get("item_id", ""), base64.b64decode(ev["delta"]))
                self.state.set(speaking=True)
            elif t == "input_audio_buffer.speech_started":
                self.user_speaking = True
                item, played_ms = self.player.interrupt()
                if item:
                    await self._send(ws, {
                        "type": "conversation.item.truncate",
                        "item_id": item, "content_index": 0, "audio_end_ms": played_ms,
                    })
                self.state.set(status="hearing you")
            elif t == "input_audio_buffer.speech_stopped":
                self.user_speaking = False
                self.state.set(status="thinking")
            elif t == "response.created":
                self.response_pending = False
                self.response_active = True
                self.logger.response_started()
            elif t == "response.cancelled":
                self.response_active = False
                self.response_pending = False
                self.state.set(speaking=False, status="listening")
            elif t == "response.output_item.done":
                await self._answer_camera(ws, ev)
            elif t == "response.done":
                self.response_active = False
                calls = camera_call_ids(ev)
                resp = ev.get("response") or {}
                self.logger.add_usage(resp.get("usage") or {})
                await self._answer_camera(ws, ev)
                if not calls:
                    self.response_pending = False
                self.state.set(speaking=False, status="listening")
            elif t in ("response.output_audio_transcript.done", "response.audio_transcript.done"):
                text = ev.get("transcript") or ""
                log.info("Nuonuo: %s", text)
                if text:
                    self.transcript.append({"role": "nuonuo", "text": text, "t_ms": self.clock.t_ms()})
            elif t == "conversation.item.input_audio_transcription.completed":
                text = ev.get("transcript") or ""
                log.info("You: %s", text)
                if text:
                    self.transcript.append({"role": "user", "text": text, "t_ms": self.clock.t_ms()})
            elif t == "session.created":
                log.info("session created")
            elif t == "error":
                self.response_pending = False
                err = ev.get("error", ev)
                log.warning("realtime error: %s", err)
                if (err.get("code") if isinstance(err, dict) else None) == "response_cancel_not_active":
                    continue


def run_realtime_thread(client: RealtimeClient) -> threading.Thread:
    def target():
        try:
            asyncio.run(client.run())
        except Exception as e:
            log.exception("Realtime session ended with error: %s", e)
            client.state.set(status="voice offline")

    th = threading.Thread(target=target, name="realtime", daemon=True)
    th.start()
    return th
