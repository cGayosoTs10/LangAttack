"""SQLite connection manager, WAL mode initialization, DDL schema, and CRUD functions."""

import json
import sqlite3
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from lang_attack.config import DEFAULT_DB_PATH


def get_db_path() -> Path:
    """Return the default database path."""
    return DEFAULT_DB_PATH


def get_connection(db_path: Path | str | None = None) -> sqlite3.Connection:
    """Open and configure a SQLite connection in WAL mode with busy timeout."""
    target_path = (
        get_db_path()
        if db_path is None
        else (db_path if isinstance(db_path, Path) else Path(db_path))
    )
    path_str = str(target_path)

    if path_str != ":memory:" and not path_str.startswith(":memory:"):
        target_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(path_str, timeout=30.0)
    conn.row_factory = sqlite3.Row

    # Pragmas
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA busy_timeout = 30000;")
    conn.execute("PRAGMA synchronous = NORMAL;")

    return conn


@contextmanager
def db_transaction(
    conn: sqlite3.Connection | None = None, write: bool = True
) -> Generator[sqlite3.Connection, None, None]:
    """Context manager for safe SQLite transactions with write lock serialization and guaranteed rollback."""
    should_close = False
    if conn is None:
        conn = get_connection()
        should_close = True

    try:
        if write and not conn.in_transaction:
            conn.execute("BEGIN IMMEDIATE;")
        yield conn
        conn.commit()
    except Exception:
        if conn.in_transaction:
            conn.rollback()
        raise
    finally:
        if should_close:
            conn.close()


def init_db(
    target_or_conn: sqlite3.Connection | Path | str | None = None,
    db_path: Path | str | None = None,
    conn: sqlite3.Connection | None = None,
) -> sqlite3.Connection:
    """Initialize SQLite database with all 4 core tables and indexes."""
    if isinstance(target_or_conn, sqlite3.Connection):
        active_conn = target_or_conn
    elif conn is not None:
        active_conn = conn
    elif isinstance(target_or_conn, (Path, str)):
        active_conn = get_connection(target_or_conn)
    elif db_path is not None:
        active_conn = get_connection(db_path)
    else:
        active_conn = get_connection()

    # Ensure pragmas are applied
    active_conn.execute("PRAGMA journal_mode = WAL;")
    active_conn.execute("PRAGMA foreign_keys = ON;")
    active_conn.execute("PRAGMA busy_timeout = 30000;")
    active_conn.execute("PRAGMA synchronous = NORMAL;")

    cursor = active_conn.cursor()
    cursor.executescript(
        """
        -- 1. study_sessions
        CREATE TABLE IF NOT EXISTS study_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            session_uuid TEXT,
            language TEXT NOT NULL,
            topic TEXT DEFAULT 'General',
            session_type TEXT DEFAULT 'practice',
            duration_seconds REAL DEFAULT 0.0,
            score REAL DEFAULT 0.0,
            notes TEXT DEFAULT '',
            started_at TEXT DEFAULT (datetime('now')),
            start_time TEXT DEFAULT (datetime('now')),
            created_at TEXT DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_study_sessions_lang ON study_sessions(language);
        CREATE INDEX IF NOT EXISTS idx_study_sessions_sess_id ON study_sessions(session_id);
        CREATE INDEX IF NOT EXISTS idx_study_sessions_sess_uuid ON study_sessions(session_uuid);

        -- 2. phonetic_history
        CREATE TABLE IF NOT EXISTS phonetic_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            record_id TEXT,
            session_id TEXT,
            language TEXT NOT NULL,
            target_text TEXT NOT NULL,
            recognized_text TEXT,
            recognized_transcript TEXT,
            overall_score REAL NOT NULL,
            fluency_score REAL DEFAULT 0.0,
            completeness_score REAL DEFAULT 0.0,
            words_json TEXT DEFAULT '[]',
            details_json TEXT DEFAULT '[]',
            recorded_audio_path TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_phonetic_lang ON phonetic_history(language);
        CREATE INDEX IF NOT EXISTS idx_phonetic_session ON phonetic_history(session_id);
        CREATE INDEX IF NOT EXISTS idx_phonetic_score ON phonetic_history(overall_score);

        -- 3. vocabulary_srs
        CREATE TABLE IF NOT EXISTS vocabulary_srs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            item_uuid TEXT,
            item_id TEXT,
            term TEXT NOT NULL,
            word TEXT,
            translation TEXT,
            language TEXT NOT NULL,
            srs_stage INTEGER DEFAULT 0,
            repetition INTEGER DEFAULT 0,
            repetition_count INTEGER DEFAULT 0,
            interval INTEGER DEFAULT 0,
            interval_days INTEGER DEFAULT 0,
            ease_factor REAL DEFAULT 2.5,
            next_review_at TEXT,
            next_review_date TEXT,
            due_date TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_vocab_term_lang ON vocabulary_srs(term, language);
        CREATE INDEX IF NOT EXISTS idx_vocab_lang ON vocabulary_srs(language);
        CREATE INDEX IF NOT EXISTS idx_vocab_due ON vocabulary_srs(next_review_at);

        -- 4. pipeline_metrics
        CREATE TABLE IF NOT EXISTS pipeline_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            metric_id TEXT,
            pipeline_name TEXT NOT NULL,
            duration_ms REAL NOT NULL,
            status TEXT NOT NULL,
            error_type TEXT DEFAULT 'NONE',
            details TEXT DEFAULT '',
            details_json TEXT DEFAULT '{}',
            metadata_json TEXT DEFAULT '{}',
            extra_data_json TEXT DEFAULT '{}',
            timestamp TEXT DEFAULT (datetime('now')),
            created_at TEXT DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_metrics_name ON pipeline_metrics(pipeline_name);
        CREATE INDEX IF NOT EXISTS idx_metrics_status ON pipeline_metrics(status);
        CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON pipeline_metrics(timestamp);
        """
    )
    active_conn.commit()
    return active_conn


def record_session(
    session_data: dict[str, Any] | Any = None,
    conn: sqlite3.Connection | None = None,
    **kwargs: Any,
) -> int:
    """Record a study session into study_sessions."""
    with db_transaction(conn, write=True) as active_conn:
        if hasattr(session_data, "model_dump"):
            data = session_data.model_dump()
        elif isinstance(session_data, dict):
            data = dict(session_data)
        elif kwargs:
            data = dict(kwargs)
        else:
            data = getattr(session_data, "__dict__", {}) if session_data else {}

        sess_id = data.get("session_id") or data.get("id") or str(uuid.uuid4())
        sess_uuid = data.get("session_uuid") or sess_id
        language = data.get("language", "en")
        topic = data.get("topic", "General")
        session_type = data.get("session_type", "practice")
        duration = float(data.get("duration_seconds", 0.0))
        score = float(data.get("score") if data.get("score") is not None else data.get("average_score", 0.0))
        notes = data.get("notes", "")
        now_iso = datetime.now(UTC).isoformat()
        started_at = data.get("started_at") or data.get("start_time") or now_iso

        cursor = active_conn.cursor()
        cursor.execute(
            """
            INSERT INTO study_sessions (
                session_id, session_uuid, language, topic, session_type,
                duration_seconds, score, notes, started_at, start_time, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                sess_id,
                sess_uuid,
                language,
                topic,
                session_type,
                duration,
                score,
                notes,
                started_at,
                started_at,
                now_iso,
            ),
        )
        return int(cursor.lastrowid) if cursor.lastrowid is not None else 0


def record_phonetic_score(
    result: Any = None,
    language: str = "en",
    conn: sqlite3.Connection | None = None,
    **kwargs: Any,
) -> int:
    """Record phonetic evaluation score into phonetic_history."""
    with db_transaction(conn, write=True) as active_conn:
        if result is None and kwargs:
            result = kwargs

        target_text = (
            getattr(result, "target_text", None)
            or (result.get("target_text") if isinstance(result, dict) else "")
            or getattr(result, "reference_text", "")
            or (result.get("word") if isinstance(result, dict) else "")
            or ""
        )
        rec_text = (
            getattr(result, "recognized_text", None)
            or (result.get("recognized_text") if isinstance(result, dict) else "")
            or getattr(result, "transcript", "")
            or target_text
        )
        score_val = (
            getattr(result, "overall_score", None)
            if hasattr(result, "overall_score")
            else (result.get("overall_score") if isinstance(result, dict) else None)
        )
        if score_val is None and isinstance(result, dict):
            score_val = result.get("gop_score", 0.0)
        score = float(score_val if score_val is not None else 0.0)

        fluency = float(
            getattr(result, "fluency_score", 0.0)
            or (result.get("fluency_score", 0.0) if isinstance(result, dict) else 0.0)
            or 0.0
        )
        completeness = float(
            getattr(result, "completeness_score", 0.0)
            or (result.get("completeness_score", 0.0) if isinstance(result, dict) else 0.0)
            or 0.0
        )
        audio_path = (
            getattr(result, "recorded_audio_path", None)
            or (result.get("recorded_audio_path") if isinstance(result, dict) else None)
        )

        words_data = (
            getattr(result, "words", None)
            or (result.get("words") if isinstance(result, dict) else [])
        )
        if hasattr(words_data, "model_dump"):
            words_json = json.dumps(words_data.model_dump())
        elif isinstance(words_data, (list, dict)):
            try:
                words_json = json.dumps(
                    [w.model_dump() if hasattr(w, "model_dump") else w for w in words_data]
                    if isinstance(words_data, list)
                    else words_data,
                    default=str,
                )
            except Exception:
                words_json = "[]"
        else:
            words_json = "[]"

        rec_id = f"rec-{uuid.uuid4().hex[:8]}"
        now_iso = datetime.now(UTC).isoformat()

        cursor = active_conn.cursor()
        cursor.execute(
            """
            INSERT INTO phonetic_history (
                record_id, session_id, language, target_text, recognized_text,
                recognized_transcript, overall_score, fluency_score, completeness_score,
                words_json, details_json, recorded_audio_path, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                rec_id,
                rec_id,
                language,
                target_text,
                rec_text,
                rec_text,
                score,
                fluency,
                completeness,
                words_json,
                words_json,
                str(audio_path) if audio_path else None,
                now_iso,
            ),
        )
        return int(cursor.lastrowid) if cursor.lastrowid is not None else 0


def update_vocabulary_srs(
    word: str | None = None,
    language: str = "en",
    grade: int = 3,
    conn: sqlite3.Connection | None = None,
    **kwargs: Any,
) -> None:
    """Update spaced repetition card using SuperMemo SM-2 algorithm."""
    if word is None:
        word = kwargs.get("term") or kwargs.get("word") or ""
    if not word:
        return

    with db_transaction(conn, write=True) as active_conn:
        cursor = active_conn.cursor()
        cursor.execute(
            """
            SELECT id, repetition, interval_days, ease_factor
            FROM vocabulary_srs
            WHERE (term = ? OR word = ?) AND language = ?;
            """,
            (word, word, language),
        )
        row = cursor.fetchone()

        if row is None:
            repetition = 0
            interval_days = 0
            ease_factor = 2.5
            is_new = True
            row_id = None
        else:
            row_id = row[0]
            repetition = int(row[1] or 0)
            interval_days = int(row[2] or 0)
            ease_factor = float(row[3] or 2.5)
            is_new = False

        # SM-2 calculation
        if grade < 3:
            repetition = 0
            interval_days = 1
        else:
            if repetition == 0:
                interval_days = 1
            elif repetition == 1:
                interval_days = 6
            else:
                interval_days = max(1, int(round(interval_days * ease_factor)))
            repetition += 1

        ease_factor = max(1.3, ease_factor + (0.1 - (5 - grade) * (0.08 + (5 - grade) * 0.02)))
        next_review = datetime.now(UTC) + timedelta(days=interval_days)
        review_str = next_review.isoformat()
        now_str = datetime.now(UTC).isoformat()

        if is_new:
            item_id = f"srs-{uuid.uuid4().hex[:8]}"
            cursor.execute(
                """
                INSERT INTO vocabulary_srs (
                    item_uuid, item_id, term, word, translation, language,
                    srs_stage, repetition, repetition_count, interval, interval_days,
                    ease_factor, next_review_at, next_review_date, due_date, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    item_id,
                    item_id,
                    word,
                    word,
                    kwargs.get("translation", ""),
                    language,
                    repetition,
                    repetition,
                    repetition,
                    interval_days,
                    interval_days,
                    ease_factor,
                    review_str,
                    review_str,
                    review_str,
                    now_str,
                ),
            )
        else:
            cursor.execute(
                """
                UPDATE vocabulary_srs
                SET srs_stage = ?, repetition = ?, repetition_count = ?,
                    interval = ?, interval_days = ?, ease_factor = ?,
                    next_review_at = ?, next_review_date = ?, due_date = ?
                WHERE id = ?;
                """,
                (
                    repetition,
                    repetition,
                    repetition,
                    interval_days,
                    interval_days,
                    ease_factor,
                    review_str,
                    review_str,
                    review_str,
                    row_id,
                ),
            )


def record_pipeline_metric(
    pipeline: str,
    duration_ms: float,
    status: str,
    details: str = "",
    error_type: str | None = None,
    conn: sqlite3.Connection | None = None,
) -> None:
    """Persist execution latency and status for a pipeline run."""
    with db_transaction(conn, write=True) as active_conn:
        metric_id = f"m-{uuid.uuid4().hex[:8]}"
        now_iso = datetime.now(UTC).isoformat()
        cursor = active_conn.cursor()
        cursor.execute(
            """
            INSERT INTO pipeline_metrics (
                metric_id, pipeline_name, duration_ms, status, error_type,
                details, details_json, metadata_json, extra_data_json,
                timestamp, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                metric_id,
                pipeline,
                duration_ms,
                status,
                error_type or "NONE",
                details,
                details,
                details,
                details,
                now_iso,
                now_iso,
            ),
        )


def get_session_history(
    limit: int = 50,
    language: str | None = None,
    conn: sqlite3.Connection | None = None,
) -> list[dict[str, Any]]:
    """Retrieve recent study sessions."""
    should_close = False
    if conn is None:
        conn = get_connection()
        should_close = True

    try:
        cursor = conn.cursor()
        if language is not None:
            cursor.execute(
                """
                SELECT id, session_id, session_uuid, language, topic, session_type,
                       duration_seconds, score, notes, started_at, created_at
                FROM study_sessions
                WHERE language = ?
                ORDER BY id DESC
                LIMIT ?;
                """,
                (language, limit),
            )
        else:
            cursor.execute(
                """
                SELECT id, session_id, session_uuid, language, topic, session_type,
                       duration_seconds, score, notes, started_at, created_at
                FROM study_sessions
                ORDER BY id DESC
                LIMIT ?;
                """,
                (limit,),
            )
        return [dict(row) for row in cursor.fetchall()]
    finally:
        if should_close:
            conn.close()


def get_due_srs_words(
    language: str | None = None,
    conn: sqlite3.Connection | None = None,
) -> list[dict[str, Any]]:
    """Retrieve vocabulary items due for review."""
    should_close = False
    if conn is None:
        conn = get_connection()
        should_close = True

    try:
        now_str = datetime.now(UTC).isoformat()
        cursor = conn.cursor()
        if language is not None:
            cursor.execute(
                """
                SELECT id, item_uuid, item_id, term, word, translation, language,
                       srs_stage, repetition, interval_days, ease_factor, next_review_at
                FROM vocabulary_srs
                WHERE language = ? AND (next_review_at <= ? OR next_review_at IS NULL OR next_review_date <= ? OR due_date <= ?)
                ORDER BY next_review_at ASC;
                """,
                (language, now_str, now_str, now_str),
            )
        else:
            cursor.execute(
                """
                SELECT id, item_uuid, item_id, term, word, translation, language,
                       srs_stage, repetition, interval_days, ease_factor, next_review_at
                FROM vocabulary_srs
                WHERE next_review_at <= ? OR next_review_at IS NULL OR next_review_date <= ? OR due_date <= ?
                ORDER BY next_review_at ASC;
                """,
                (now_str, now_str, now_str),
            )
        return [dict(row) for row in cursor.fetchall()]
    finally:
        if should_close:
            conn.close()


__all__ = [
    "db_transaction",
    "get_connection",
    "get_db_path",
    "get_due_srs_words",
    "get_session_history",
    "init_db",
    "record_phonetic_score",
    "record_pipeline_metric",
    "record_session",
    "update_vocabulary_srs",
]
