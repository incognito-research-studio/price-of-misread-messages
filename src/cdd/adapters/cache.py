"""A persistent store of model replies, keyed by the full request.

Experiments call local models at temperature 0, and the same request can recur: the same
message and state in the main run and in an oracle reference, or in a rerun after a change
downstream of the reader. Storing each reply under a hash of the exact request body (and the
model's digest, so an updated model never reuses an old reply) makes a repeated request return
the first reply. Reruns then cost minutes instead of hours, and identical prompts get identical
answers even where the server is not deterministic. A study that measures decision consistency
across repeated samples must not use the cache.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def request_key(body: dict[str, Any], digest: str | None) -> str:
    payload = json.dumps({"digest": digest, "body": body}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


class ReplyCache:
    """One SQLite file; safe to share between runs in sequence."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path, timeout=60)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS replies (key TEXT PRIMARY KEY, record TEXT NOT NULL,"
            " created TEXT NOT NULL)"
        )
        self._db.commit()

    def get(self, key: str) -> dict[str, Any] | None:
        row = self._db.execute("SELECT record FROM replies WHERE key = ?", (key,)).fetchone()
        return None if row is None else dict(json.loads(row[0]))

    def put(self, key: str, record: dict[str, Any]) -> None:
        created = datetime.now(UTC).isoformat(timespec="seconds")
        self._db.execute(
            "INSERT OR IGNORE INTO replies (key, record, created) VALUES (?, ?, ?)",
            (key, json.dumps(record), created),
        )
        self._db.commit()

    def __len__(self) -> int:
        return int(self._db.execute("SELECT COUNT(*) FROM replies").fetchone()[0])
