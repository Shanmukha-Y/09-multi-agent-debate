"""SQLite transcript store: every debate (and baseline-arm run) is persisted
so `debate replay <id>` can re-render it and the bench can look back at what
happened. Schema is deliberately flat JSON-blob-per-round rather than a
normalized message table — transcripts are read whole for replay, never
queried by field, so normalization would add joins with no benefit.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from debate import config
from debate.debate import DebateTranscript, RoundData
from debate.messages import Critique, FinalVerdict, Proposal
from debate.voting import PersonaScore, VoteResult

_SCHEMA = """
CREATE TABLE IF NOT EXISTS debates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT NOT NULL,
    arm TEXT NOT NULL DEFAULT 'debate',
    created_at TEXT NOT NULL,
    rounds_used INTEGER NOT NULL,
    answer TEXT NOT NULL,
    confidence REAL NOT NULL,
    is_split INTEGER NOT NULL,
    total_tokens INTEGER NOT NULL,
    payload_json TEXT NOT NULL
);
"""


def _connect(db_path: str | None = None) -> sqlite3.Connection:
    path = db_path or config.DB_PATH
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(_SCHEMA)
    return conn


def _round_to_dict(r: RoundData) -> dict:
    return {
        "round_num": r.round_num,
        "proposals": {name: p.model_dump() for name, p in r.proposals.items()},
        "critiques": {name: c.model_dump() for name, c in r.critiques.items()},
        "persona_to_letter": r.persona_to_letter,
        "vote": {
            "scores": [asdict(s) for s in r.vote.scores],
            "is_clear_winner": r.vote.is_clear_winner,
            "lead_ratio": r.vote.lead_ratio,
            "winner": asdict(r.vote.winner),
            "runner_up": asdict(r.vote.runner_up) if r.vote.runner_up else None,
        },
    }


def transcript_to_dict(transcript: DebateTranscript, arm: str = "debate") -> dict:
    return {
        "question": transcript.question,
        "arm": arm,
        "rounds_used": transcript.rounds_used,
        "total_tokens": transcript.total_tokens,
        "rounds": [_round_to_dict(r) for r in transcript.rounds],
        "verdict": transcript.verdict.model_dump(),
    }


def save_debate(transcript: DebateTranscript, arm: str = "debate", db_path: str | None = None) -> int:
    payload = transcript_to_dict(transcript, arm=arm)
    with closing(_connect(db_path)) as conn:
        cur = conn.execute(
            "INSERT INTO debates (question, arm, created_at, rounds_used, answer, confidence, "
            "is_split, total_tokens, payload_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                transcript.question,
                arm,
                datetime.now(timezone.utc).isoformat(),
                transcript.rounds_used,
                transcript.verdict.answer,
                transcript.verdict.confidence,
                int(transcript.verdict.is_split),
                transcript.total_tokens,
                json.dumps(payload),
            ),
        )
        conn.commit()
        return cur.lastrowid


def get_debate(debate_id: int, db_path: str | None = None) -> dict | None:
    with closing(_connect(db_path)) as conn:
        row = conn.execute(
            "SELECT id, question, arm, created_at, payload_json FROM debates WHERE id = ?", (debate_id,)
        ).fetchone()
    if row is None:
        return None
    debate_id_, question, arm, created_at, payload_json = row
    payload = json.loads(payload_json)
    payload.update({"id": debate_id_, "created_at": created_at})
    return payload


def list_debates(db_path: str | None = None, limit: int = 50) -> list[dict]:
    with closing(_connect(db_path)) as conn:
        rows = conn.execute(
            "SELECT id, question, arm, created_at, answer, confidence, is_split, total_tokens "
            "FROM debates ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [
        {
            "id": r[0],
            "question": r[1],
            "arm": r[2],
            "created_at": r[3],
            "answer": r[4],
            "confidence": r[5],
            "is_split": bool(r[6]),
            "total_tokens": r[7],
        }
        for r in rows
    ]


def export_transcript_json(transcript: DebateTranscript, out_path: str, arm: str = "debate") -> None:
    """Write a standalone JSON snapshot for committing to transcripts/."""
    payload = transcript_to_dict(transcript, arm=arm)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(payload, indent=2))
