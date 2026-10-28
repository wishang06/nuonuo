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

from nuonuo.logs import SessionLogger
from nuonuo.perception import PerceptionEvent
from nuonuo.persona import build_instructions
from nuonuo.state import SharedState

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
    def __init__(self, state: SharedState, api_key: str, model: str, voice: str, user_name: str, logger: SessionLogger):
        self.state = state
        self.api_key = api_key
        self.model = model
        self.voice = voice
        self.user_name = user_name
        self.logger = logger
        self.loop: asyncio.AbstractEventLoop | None = None
        self.out_q: asyncio.Queue | None = None
        self.response_active = False
        self.user_speaking = False
        self.last_salient_response = 0.0

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

    def _session_update(self) -> dict:
        return {
            "type": "session.update",
            "session": {
                "type": "realtime",
                "model": self.model,
                "instructions": build_instructions(self.user_name),
                "output_modalities": ["audio"],
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": RATE},
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

    async def _out_sender(self, ws):
        while True:
            kind, payload = await self.out_q.get()
            if kind == "vision":
                await self._send_vision(ws, payload)

    async def _send_vision(self, ws, ev: PerceptionEvent):
        now = time.monotonic()
        request = (
            ev.salient
            and not self.response_active
            and not self.user_speaking
            and now - self.last_salient_response > SALIENT_RESPONSE_COOLDOWN
        )
        eid = uuid.uuid4().hex[:12]
        await self._send(ws, {
            "type": "conversation.item.create",
            "item": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": ev.text}]},
        })
        if request:
            self.last_salient_response = now
            await self._send(ws, {"type": "response.create"})
        self.logger.vision_event(eid, ev.text, ev.salient, request)
        log.info("vision -> %s%s", ev.text, " (response requested)" if request else "")

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
                self.response_active = True
                self.logger.response_started()
            elif t == "response.done":
                self.response_active = False
                resp = ev.get("response") or {}
                self.logger.add_usage(resp.get("usage") or {})
                self.state.set(speaking=False, status="listening")
            elif t in ("response.output_audio_transcript.done", "response.audio_transcript.done"):
                log.info("Nuonuo: %s", ev.get("transcript", ""))
            elif t == "conversation.item.input_audio_transcription.completed":
                log.info("You: %s", ev.get("transcript", ""))
            elif t == "session.created":
                log.info("session created")
            elif t == "error":
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
