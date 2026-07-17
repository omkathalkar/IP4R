"""SQLite-backed job store with 30-day retention.

Schema:
  jobs(job_id TEXT PK, status TEXT, video_path TEXT, result_json TEXT,
       error TEXT, created_at REAL, expires_at REAL, inference_ms REAL)

Statuses: pending | processing | done | error
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

_RETENTION_DAYS = 30
_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id       TEXT    PRIMARY KEY,
    status       TEXT    NOT NULL DEFAULT 'pending',
    video_path   TEXT    NOT NULL,
    result_json  TEXT,
    error        TEXT,
    created_at   REAL    NOT NULL,
    expires_at   REAL    NOT NULL,
    inference_ms REAL
);
"""


class JobStore:
    def __init__(self, db_path: str | Path = "data/jobs/jobs.db"):
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(str(self._path), check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_db(self) -> None:
        self._conn().execute(_SCHEMA)
        self._conn().commit()

    # ── write ops ─────────────────────────────────────────────────────────────

    def create_job(self, job_id: str, video_path: str) -> None:
        now = time.time()
        self._conn().execute(
            "INSERT OR IGNORE INTO jobs (job_id, status, video_path, created_at, expires_at) "
            "VALUES (?, 'pending', ?, ?, ?)",
            (job_id, video_path, now, now + _RETENTION_DAYS * 86400),
        )
        self._conn().commit()

    def mark_processing(self, job_id: str) -> None:
        self._conn().execute(
            "UPDATE jobs SET status='processing' WHERE job_id=?", (job_id,)
        )
        self._conn().commit()

    def mark_done(self, job_id: str, result: dict, inference_ms: float) -> None:
        self._conn().execute(
            "UPDATE jobs SET status='done', result_json=?, inference_ms=? WHERE job_id=?",
            (json.dumps(result), inference_ms, job_id),
        )
        self._conn().commit()

    def mark_error(self, job_id: str, error: str) -> None:
        self._conn().execute(
            "UPDATE jobs SET status='error', error=? WHERE job_id=?",
            (error, job_id),
        )
        self._conn().commit()

    # ── read ops ──────────────────────────────────────────────────────────────

    def get_job(self, job_id: str) -> dict | None:
        row = self._conn().execute(
            "SELECT * FROM jobs WHERE job_id=?", (job_id,)
        ).fetchone()
        if row is None:
            return None
        d = dict(row)
        if d.get("result_json"):
            d["result"] = json.loads(d.pop("result_json"))
        else:
            d.pop("result_json", None)
        return d

    def get_status(self) -> dict:
        rows = self._conn().execute(
            "SELECT status, COUNT(*) as n FROM jobs GROUP BY status"
        ).fetchall()
        counts = {r["status"]: r["n"] for r in rows}
        pending    = counts.get("pending", 0)
        processing = counts.get("processing", 0)
        done       = counts.get("done", 0)
        error      = counts.get("error", 0)

        # avg inference time over last 100 completed jobs
        perf = self._conn().execute(
            "SELECT AVG(inference_ms) as avg_ms FROM "
            "(SELECT inference_ms FROM jobs WHERE status='done' AND inference_ms IS NOT NULL "
            " ORDER BY created_at DESC LIMIT 100)"
        ).fetchone()
        avg_ms = perf["avg_ms"] if perf and perf["avg_ms"] else None

        return {
            "pending": pending,
            "processing": processing,
            "done": done,
            "error": error,
            "avg_inference_ms": round(avg_ms, 1) if avg_ms else None,
        }

    def recent_jobs(self, limit: int = 50) -> list[dict]:
        rows = self._conn().execute(
            "SELECT job_id, status, inference_ms, created_at, error, result_json "
            "FROM jobs ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            if d.get("result_json"):
                d["result"] = json.loads(d.pop("result_json"))
            else:
                d.pop("result_json", None)
            out.append(d)
        return out

    # ── maintenance ───────────────────────────────────────────────────────────

    def purge_expired(self) -> int:
        """Delete jobs past their 30-day expiry. Returns count deleted."""
        now = time.time()
        cur = self._conn().execute(
            "DELETE FROM jobs WHERE expires_at < ?", (now,)
        )
        self._conn().commit()
        return cur.rowcount
