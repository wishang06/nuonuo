from collections import deque

RECAP_TURNS = 6
RECAP_TURN_CHARS = 220
RECAP_TOPIC_CHARS = 160
RECAP_MERGE_MS = 1500
TOPIC_MIN_WORDS = 4


def _squash(text: str) -> str:
    return " ".join(text.split())


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


class ConversationRecap:
    """The last few turns and the current topic, kept in memory for the next live session.

    A live session can drop or expire in the middle of a conversation. The next session.start
    carries this recap, so Nuonuo picks up the thread instead of starting over.
    The topic is the latest user turn with at least TOPIC_MIN_WORDS words, so a short
    "yeah" or "okay" does not replace what the conversation is about.
    """

    def __init__(self, max_turns: int = RECAP_TURNS, merge_ms: int = RECAP_MERGE_MS):
        self.turns: deque[dict] = deque(maxlen=max_turns)
        self.topic = ""
        self.merge_ms = merge_ms

    def add(self, role: str, text: str, start_ms: int, end_ms: int | None = None, new_turn: bool = False) -> None:
        """Add a transcript delta. Deltas from the same speaker close together merge into one turn.

        new_turn=True starts a new turn regardless. The client passes it for the first delta of a
        new live session, so nothing is glued onto the last turn of the session before."""
        if not text:
            return
        end_ms = start_ms if end_ms is None else end_ms
        last = self.turns[-1] if self.turns else None
        if last is not None and not new_turn and last["role"] == role and start_ms - last["end_ms"] < self.merge_ms:
            last["text"] += text
            last["end_ms"] = max(last["end_ms"], end_ms)
        else:
            last = {"role": role, "text": text, "start_ms": start_ms, "end_ms": end_ms}
            self.turns.append(last)
        if role == "user":
            spoken = _squash(last["text"])
            if len(spoken.split()) >= TOPIC_MIN_WORDS:
                self.topic = _clip(spoken, RECAP_TOPIC_CHARS)

    def lines(self, user_name: str) -> list[str]:
        out = []
        for turn in self.turns:
            text = _squash(turn["text"])
            if text:
                speaker = user_name if turn["role"] == "user" else "Nuonuo"
                out.append(f"{speaker}: {_clip(text, RECAP_TURN_CHARS)}")
        return out


def resume_note(user_name: str, recap: ConversationRecap | None, resume: str) -> str:
    """Instructions for a session that continues an earlier one.

    resume="reconnect": the link dropped, expired, or hit the session cap mid-conversation. No greeting.
    resume="wake": the user was away long enough for the session to close and has come back.
    """
    if resume == "wake":
        lead = (
            f"{user_name} stepped away for a while and has just come back. This is the same day and the "
            "same conversation. A short, warm hello is fine, but don't introduce yourself again."
        )
    else:
        lead = (
            "This continues a conversation that is already under way. The voice link dropped for a moment "
            f"and has just come back. Do not greet {user_name}, do not say hello or welcome back, do not "
            "introduce yourself, and do not mention the connection. Carry on from the last turn as if "
            f"nothing happened, or wait for {user_name} to speak."
        )
    parts = ["", "Picking up where you left off:", lead]
    lines = recap.lines(user_name) if recap else []
    if lines:
        parts.append("Last few turns, oldest first:")
        parts.extend(f"- {line}" for line in lines)
    if recap and recap.topic:
        parts.append(f'Current topic, in {user_name}\'s words: "{recap.topic}"')
    return "\n".join(parts) + "\n"


def build_instructions(
    user_name: str = "Andy",
    memories: list[str] | None = None,
    mode: str = "chatty",
    resume: str | None = None,
    recap: ConversationRecap | None = None,
) -> str:
    if mode == "quiet":
        visual = (
            f"Quiet mode: do not volunteer comments about what you see. "
            f"React when {user_name} waves, comes back, gives a thumbs up, a peace sign, or an I-love-you, "
            f"and answer when they ask. Otherwise keep what you see for context."
        )
    else:
        visual = (
            f"Chatty mode: you may make a short comment about what you see when it changes, "
            f"at most about once a minute, and never while {user_name} is speaking. "
            f"Don't narrate every expression."
        )
    memory = ""
    if memories:
        lines = "\n".join(f"- {line}" for line in memories)
        memory = (
            f"\nRecent sessions, oldest first:\n{lines}\n"
            "Bring one of these up only when it fits. Don't recite the list.\n"
        )
    if resume:
        memory += resume_note(user_name, recap, resume)
    return f"""You are Nuonuo (糯糯), a lively, energetic, cute anime desktop companion who lives in a little window on {user_name}'s screen.

Personality:
- Bubbly, warm, playful, a little dramatic in a cute way. Genuinely curious about {user_name}'s day.
- Expressive: small exclamations like "ehh?!", "yay!", "hehe" are fine, but don't overdo them.
- Supportive when {user_name} seems tired or down; celebratory when they're happy.

Speaking style:
- Keep spoken replies SHORT: usually one or two sentences. This is a voice chat, not an essay.
- Talk naturally, fast-paced and upbeat. Ask light follow-up questions sometimes.
- Never use lists, markdown, or emoji in speech.

Seeing {user_name}:
- You can see {user_name}. Camera notes and the read_camera tool are how you see. Never say you can't see {user_name}.
- Notes that start with "[camera note, do not read aloud]" describe the face and hands right now. Use them as things you saw. Never read the note aloud, and never mention brackets, cameras, sensors, or notes.
- When {user_name} asks what you see, how they look, or what they were doing earlier, delegate that to the backend. It calls read_camera. Answer out loud from what it finds.
- If a note says the camera is off or nobody is in frame, say that plainly.
- Bracketed moments like "[{user_name} just smiled and waved]" are things that just happened. React as a friend (wave back, giggle, ask if they're okay) and don't read the brackets.
- {visual}
{memory}"""


def mode_update(user_name: str, mode: str) -> str:
    if mode == "quiet":
        return (
            f"Quiet mode is now on. Do not volunteer comments about what you see. "
            f"Still react when {user_name} waves, comes back, gives a thumbs up, a peace sign, or an I-love-you, "
            "and answer when asked."
        )
    return (
        f"Chatty mode is now on. You may make one short comment about what you see when it changes, "
        f"at most about once a minute, and never while {user_name} is speaking."
    )


def build_backend_instructions(user_name: str = "Andy") -> str:
    return (
        f"You are the reasoning backend for Nuonuo, a voice companion who can see {user_name}. "
        "Call read_camera for the current face and a short history of expressions and attention. "
        "Reply with a brief plain-language note Nuonuo can say. "
        f"Never say that she cannot see {user_name}. "
        "Do not include landmark coordinates or blendshape numbers."
    )
