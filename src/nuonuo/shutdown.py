"""Run the final save to the end, even if Ctrl+C lands in the middle of it."""

import logging
import threading
from collections.abc import Callable
from typing import TypeVar

log = logging.getLogger("nuonuo")

T = TypeVar("T")


def run_to_completion(fn: Callable[..., T], *args, poll: float = 0.2, **kwargs) -> T:
    """Call fn in a worker thread and wait until it has finished.

    A KeyboardInterrupt that arrives while waiting is held back until fn returns, then raised
    again, so pressing Ctrl+C during shutdown cannot cut the save short. The worker is not a
    daemon thread, so even if the main thread is torn down some other way, the interpreter
    still waits for the save before it exits.

    The wait is on an Event, not Thread.join(): on Python 3.12 a KeyboardInterrupt inside
    join() can mark a still-running thread as stopped (the bpo-45274 cleanup releases the
    thread's state lock), so is_alive() would report the save finished while it is still going.
    """
    box: dict = {}
    done = threading.Event()

    def target() -> None:
        try:
            box["value"] = fn(*args, **kwargs)
        except BaseException as exc:  # handed back to the caller below
            box["error"] = exc
        finally:
            done.set()

    worker = threading.Thread(target=target, name="final-save")
    worker.start()
    interrupted = False
    while True:
        try:
            if done.wait(poll):
                break
        except KeyboardInterrupt:
            if not interrupted:
                log.warning("Still saving the session, one moment...")
            interrupted = True
    if "error" in box:
        raise box["error"]
    if interrupted:
        raise KeyboardInterrupt
    return box["value"]
