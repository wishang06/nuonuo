import argparse
import logging
import os
import signal
import sys
from pathlib import Path

from dotenv import load_dotenv

from nuonuo.state import SharedState

ROOT = Path(os.getenv("NUONUO_HOME", Path.cwd()))
log = logging.getLogger("nuonuo")


def _truthy(v: str | None, default: bool = True) -> bool:
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="nuonuo", description="Nuonuo anime desktop companion")
    p.add_argument("--no-vision", action="store_true", help="disable webcam perception")
    p.add_argument("--no-voice", action="store_true", help="run avatar + vision only (no API calls)")
    p.add_argument("--preview", action="store_true", help="show webcam debug preview with landmarks")
    p.add_argument("--camera", type=int, default=0, help="webcam index")
    p.add_argument("--model", default=None, help="override NUONUO_MODEL")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> None:
    load_dotenv(ROOT / ".env")
    args = parse_args(argv)
    logs_dir = ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(logs_dir / "nuonuo.log", encoding="utf-8")],
    )
    user = os.getenv("USER_NAME", "Andy")
    model = args.model or os.getenv("NUONUO_MODEL", "gpt-realtime-2.1-mini")
    vision_on = _truthy(os.getenv("VISION_ENABLED"), True) and not args.no_vision
    state = SharedState(vision_enabled=vision_on, preview=args.preview)

    from nuonuo.logs import SessionLogger

    slog = SessionLogger(logs_dir, model)
    client = rt_thread = None
    if not args.no_voice:
        key = os.getenv("OPENAI_API_KEY")
        if not key or key.startswith("sk-your"):
            log.error("OPENAI_API_KEY missing; set it in .env or use --no-voice")
            sys.exit(1)
        from nuonuo.realtime import RealtimeClient, run_realtime_thread

        client = RealtimeClient(state, key, model, os.getenv("NUONUO_VOICE", "marin"), user, slog)
        rt_thread = run_realtime_thread(client)
    else:
        state.set(status="voice off")

    def on_event(ev):
        if client:
            client.submit_vision(ev)
        else:
            log.info("vision: %s", ev.text)

    from nuonuo.vision import VisionWorker

    vision = VisionWorker(state, on_event, ROOT / "models", user, args.camera)
    vision.start()

    signal.signal(signal.SIGINT, lambda *_: state.stop.set())
    log.info("Nuonuo started (model=%s, vision=%s). Ctrl+C or Esc to quit.", model, vision_on)
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
        summary = slog.close()
        log.info("Session: %.0fs, est. cost $%.4f -> %s", summary["duration_seconds"], summary["estimated_cost_usd"], slog.session_path)


if __name__ == "__main__":
    main()
