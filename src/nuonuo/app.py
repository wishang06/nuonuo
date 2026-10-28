import argparse
import logging
import os
import signal
import sys
from pathlib import Path

from dotenv import load_dotenv

from nuonuo.state import SharedState

log = logging.getLogger("nuonuo")


def project_root() -> Path:
    return Path(os.getenv("NUONUO_HOME", Path.cwd()))


def _truthy(v: str | None, default: bool = True) -> bool:
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="nuonuo", description="Nuonuo anime desktop companion")
    p.add_argument("--no-vision", action="store_true", help="disable webcam perception")
    p.add_argument("--no-voice", action="store_true", help="run avatar + vision only (no API calls)")
    p.add_argument("--preview", action="store_true", help="show webcam debug preview with landmarks")
    p.add_argument("--camera", type=int, default=0, help="webcam index")
    p.add_argument("--model", default=None, help="override NUONUO_MODEL")
    p.add_argument("--mode", choices=("chatty", "quiet"), default="chatty", help="chatty comments on what she sees; quiet does not")
    p.add_argument("--forget", action="store_true", help="delete saved sessions and exit")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def _persist(memory, clock, timeline, client, user: str, api_key: str | None) -> None:
    from nuonuo.summary import DEFAULT_SUMMARY_MODEL, summarize_session

    now_ms = clock.t_ms()
    record = {
        "user_name": user,
        "transcript": list(client.transcript) if client else [],
        "notes": list(client.note_log) if client else [],
        "spans": [
            {"state": span.state, "start_ms": span.start_ms, "end_ms": span.end_ms}
            for span in timeline.recent_spans()
        ],
        "aggregates": timeline.aggregate_record(now_ms),
        "summary": None,
    }
    has_story = bool(record["transcript"] or record["notes"] or record["spans"] or timeline.minute_count())
    if not has_story:
        return
    if client is not None:
        model = os.getenv("NUONUO_SUMMARY_MODEL", DEFAULT_SUMMARY_MODEL)
        record["summary"] = summarize_session(api_key, model, user, record)
    memory.save_session(record)


def main(argv=None) -> None:
    root = project_root()
    load_dotenv(root / ".env")
    args = parse_args(argv)
    logs_dir = root / "logs"
    logs_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(logs_dir / "nuonuo.log", encoding="utf-8")],
    )
    from nuonuo.memory import Memory

    memory = Memory(root / "data" / "nuonuo.db")
    if args.forget:
        removed = memory.forget_all()
        log.info("Forgot %s saved session(s).", removed)
        return

    expired = memory.purge_expired()
    if expired:
        log.info("Deleted %s session(s) older than 30 days.", expired)
    memories = memory.memory_lines()

    user = os.getenv("USER_NAME", "Andy")
    model = args.model or os.getenv("NUONUO_MODEL", "gpt-live-1")
    if model.startswith("gpt-realtime"):
        log.warning("NUONUO_MODEL=%s is a Realtime model; using gpt-live-1", model)
        model = "gpt-live-1"
    vision_on = _truthy(os.getenv("VISION_ENABLED"), True) and not args.no_vision
    state = SharedState(vision_enabled=vision_on, preview=args.preview, mode=args.mode)

    from nuonuo.clock import SessionClock
    from nuonuo.logs import SessionLogger
    from nuonuo.timeline import BehaviourTimeline

    clock = SessionClock()
    timeline = BehaviourTimeline()
    slog = SessionLogger(logs_dir, model)
    key = os.getenv("OPENAI_API_KEY")
    client = rt_thread = None
    if not args.no_voice:
        if not key or key.startswith("sk-your"):
            log.error("OPENAI_API_KEY missing; set it in .env or use --no-voice")
            sys.exit(1)
        from nuonuo.realtime import RealtimeClient, run_realtime_thread

        client = RealtimeClient(
            state,
            key,
            model,
            os.getenv("NUONUO_VOICE", "marin"),
            user,
            slog,
            timeline,
            clock,
            memories,
            os.getenv("NUONUO_BACKEND_MODEL", "gpt-6-luna"),
        )
        rt_thread = run_realtime_thread(client)
    else:
        state.set(status="voice off")

    def on_event(ev):
        if client:
            client.submit_vision(ev)
        else:
            log.info("vision: %s", ev.text)

    from nuonuo.vision import VisionWorker

    vision = VisionWorker(state, on_event, root / "models", user, args.camera, timeline, clock)
    vision.start()

    signal.signal(signal.SIGINT, lambda *_: state.stop.set())
    log.info("Nuonuo started (model=%s, vision=%s, mode=%s). Ctrl+C or Esc to quit.", model, vision_on, args.mode)
    try:
        from nuonuo.avatar import run_avatar

        run_avatar(state)
    except KeyboardInterrupt:
        pass
    finally:
        state.stop.set()
        vision.join(timeout=3)
        if rt_thread:
            rt_thread.join(timeout=3)
        try:
            _persist(memory, clock, timeline, client, user, key)
        except Exception:
            log.exception("Could not save session memory")
        summary = slog.close()
        log.info(
            "Session: %.0fs, est. cost $%.4f -> %s",
            summary["duration_seconds"],
            summary["estimated_cost_usd"],
            slog.session_path,
        )


if __name__ == "__main__":
    main()
