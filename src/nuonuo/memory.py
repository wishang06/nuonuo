"""Session memory. Transcripts and behaviour aggregates only, for 30 days."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

RETENTION_DAYS = 30

_FORBIDDEN_KEYS = {
    "mesh",
    "landmarks",
    "landmark",
    "frames",
    "frame",
    "image",
    "images",
    "video",
    "videos",
    "image_url",
    "blendshapes",
}


def _is_point(item: object) -> bool:
    if isinstance(item, (list, tuple)) and len(item) >= 2 and all(
        isinstance(n, (int, float)) and not isinstance(n, bool) for n in item[:3]
    ):
        return True
    if isinstance(item, dict) and "x" in item and "y" in item:
        return True
    return False


def _is_number_list(value: list) -> bool:
    return len(value) >= 400 and all(isinstance(n, (int, float)) and not isinstance(n, bool) for n in value)


def reject_stored_media(value: object) -> None:
    """Raise if a row contains frames, a mesh, or a landmark-sized point list."""
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in _FORBIDDEN_KEYS:
                raise ValueError("video frames and mesh points are not stored")
            reject_stored_media(item)
        return
    if isinstance(value, list):
        landmark_sized = len(value) >= 400 and bool(value)
        point_rows = all(_is_point(item) or isinstance(item, list) for item in value)
        if landmark_sized and (_is_number_list(value) or point_rows):
            raise ValueError("video frames and mesh points are not stored")
        for item in value:
            reject_stored_media(item)


class Memory:
    def __init__(self, path: Path):
        self.path = Path(path)

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                payload TEXT NOT NULL
            )
            """
        )
        connection.execute("CREATE INDEX IF NOT EXISTS sessions_created_at ON sessions (created_at)")
        return connection

    def save_session(self, record: dict, *, created_at: datetime | None = None) -> str:
        reject_stored_media(record)
        session_id = str(uuid.uuid4())
        moment = created_at or datetime.now(timezone.utc)
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        stamp = moment.astimezone(timezone.utc).isoformat()
        connection = self._connect()
        try:
            connection.execute(
                "INSERT INTO sessions (id, created_at, payload) VALUES (?, ?, ?)",
                (session_id, stamp, json.dumps(record)),
            )
            connection.commit()
        finally:
            connection.close()
        return session_id

    def purge_expired(self, now: datetime | None = None) -> int:
        moment = now or datetime.now(timezone.utc)
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        cutoff = (moment.astimezone(timezone.utc) - timedelta(days=RETENTION_DAYS)).isoformat()
        connection = self._connect()
        try:
            cursor = connection.execute("DELETE FROM sessions WHERE created_at < ?", (cutoff,))
            connection.commit()
            return cursor.rowcount
        finally:
            connection.close()

    def forget_all(self) -> int:
        connection = self._connect()
        try:
            cursor = connection.execute("DELETE FROM sessions")
            connection.commit()
            return cursor.rowcount
        finally:
            connection.close()

    def recent_sessions(self, limit: int = 3) -> list[dict]:
        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT payload FROM sessions ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        finally:
            connection.close()
        out = []
        for row in rows:
            payload = json.loads(row["payload"])
            if isinstance(payload, dict):
                out.append(payload)
        return out

    def memory_lines(self, limit: int = 3) -> list[str]:
        lines = []
        for record in reversed(self.recent_sessions(limit)):
            summary = record.get("summary") or {}
            text = summary.get("summary") if isinstance(summary, dict) else None
            if not text:
                aggregates = record.get("aggregates") or {}
                text = aggregates.get("text") if isinstance(aggregates, dict) else None
            if text:
                lines.append(str(text))
        return lines
