"""Local swing history (SQLite). This is the tool's memory across sessions."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import config


@dataclass
class Swing:
    id: int
    created_at: str
    club: str | None
    feel: str
    video_path: str | None
    tags: list[str]
    analysis: dict


def _connect(path: Path | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(path or config.db_path())
    conn.row_factory = sqlite3.Row
    return conn


def init_db(path: Path | None = None) -> None:
    with _connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS swings (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at  TEXT NOT NULL,
                club        TEXT,
                feel        TEXT NOT NULL,
                video_path  TEXT,
                tags        TEXT NOT NULL DEFAULT '[]',
                analysis    TEXT NOT NULL DEFAULT '{}'
            )
            """
        )


def add_swing(
    feel: str,
    analysis: dict,
    club: str | None = None,
    video_path: str | None = None,
    tags: list[str] | None = None,
) -> int:
    init_db()
    created_at = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO swings (created_at, club, feel, video_path, tags, analysis)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (
                created_at,
                club,
                feel,
                video_path,
                json.dumps(tags or []),
                json.dumps(analysis),
            ),
        )
        return int(cur.lastrowid)


def _row_to_swing(row: sqlite3.Row) -> Swing:
    return Swing(
        id=row["id"],
        created_at=row["created_at"],
        club=row["club"],
        feel=row["feel"],
        video_path=row["video_path"],
        tags=json.loads(row["tags"]),
        analysis=json.loads(row["analysis"]),
    )


def list_swings(limit: int = 10, club: str | None = None) -> list[Swing]:
    init_db()
    query = "SELECT * FROM swings"
    params: list = []
    if club:
        query += " WHERE club = ?"
        params.append(club)
    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    with _connect() as conn:
        return [_row_to_swing(r) for r in conn.execute(query, params).fetchall()]


def get_swing(swing_id: int) -> Swing | None:
    init_db()
    with _connect() as conn:
        row = conn.execute("SELECT * FROM swings WHERE id = ?", (swing_id,)).fetchone()
        return _row_to_swing(row) if row else None
