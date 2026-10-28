import json
from pathlib import Path

from nuonuo.timeline import BehaviourTimeline

FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(name: str):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def replay(name: str, timeline: BehaviourTimeline | None = None) -> BehaviourTimeline:
    timeline = timeline or BehaviourTimeline()
    for step in load_fixture(name)["steps"]:
        timeline.observe(step["t_ms"], step["state"])
    return timeline
