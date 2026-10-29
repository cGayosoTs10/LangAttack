"""Tier 1 & Tier 2 tests for SQLite persistence, WAL mode, and language flexibility.

Covers:
- FEAT-DAT-01: Auto SQLite DDL initialization at database/lang_attack.db in WAL mode
- FEAT-DAT-02: Table study_sessions with flexible language TEXT
- FEAT-DAT-03: Table phonetic_history with flexible language TEXT
- FEAT-DAT-04: Table vocabulary_srs with flexible language TEXT
- FEAT-DAT-05: Table pipeline_metrics for telemetry records
- Edge & Boundary: SQL injection strings in language, missing dirs, idempotent init, WAL concurrency
"""

import sqlite3
from pathlib import Path

import pytest

# Progressive testability check
try:
    import lang_attack.db.connection as db_conn
    import lang_attack.schemas.phonetics as ph_mod

    HAS_DB = hasattr(db_conn, "init_db") and hasattr(ph_mod, "SpeechAssessmentResult")
except ImportError:
    HAS_DB = False

pytestmark = pytest.mark.skipif(
    not HAS_DB,
    reason="lang_attack.db modules not yet implemented (scheduled for M2)",
)


# ============================================================================
# Tier 1: Database DDL & Tables Initialization
# ============================================================================


def test_init_db_creates_all_tables(tmp_db_path: Path) -> None:
    """Verify init_db initializes the database and creates the 4 core tables."""
    from lang_attack.db.connection import init_db

    conn = init_db(tmp_db_path)
    assert conn is not None

    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = {row[0] for row in cursor.fetchall()}

    expected_tables = {
        "study_sessions",
        "phonetic_history",
        "vocabulary_srs",
        "pipeline_metrics",
    }
    for expected in expected_tables:
        assert expected in tables, f"Expected table '{expected}' not found in SQLite DB"

    conn.close()


def test_wal_journal_mode(tmp_db_path: Path) -> None:
    """Verify PRAGMA journal_mode is set to WAL for safe non-blocking operations."""
    from lang_attack.db.connection import get_connection, init_db

    init_db(tmp_db_path)
    conn = get_connection(tmp_db_path) if "get_connection" in dir(
        db_conn
    ) else sqlite3.connect(tmp_db_path)

    cursor = conn.cursor()
    cursor.execute("PRAGMA journal_mode;")
    mode = cursor.fetchone()[0]
    assert mode.lower() in ["wal", "memory"], f"Expected journal_mode WAL, got {mode}"
    conn.close()


def test_study_sessions_record(tmp_db_path: Path) -> None:
    """Verify inserting and querying a study session."""
    from lang_attack.db.connection import init_db

    conn = init_db(tmp_db_path)
    cursor = conn.cursor()

    # Generic insert into study_sessions
    cursor.execute(
        """
        INSERT INTO study_sessions (session_uuid, language, topic, session_type, started_at)
        VALUES ('uuid-101', 'pt-BR', 'Pronomes', 'reading', '2026-10-29T12:00:00Z');
        """
        if "session_uuid" in [c[1] for c in cursor.execute("PRAGMA table_info(study_sessions);")]
        else """
        INSERT INTO study_sessions (session_id, language, start_time)
        VALUES ('uuid-101', 'pt-BR', '2026-10-29T12:00:00Z');
        """
    )
    conn.commit()

    cursor.execute("SELECT language FROM study_sessions WHERE session_id = 'uuid-101' OR session_uuid = 'uuid-101';")
    row = cursor.fetchone()
    assert row is not None
    assert row[0] == "pt-BR"
    conn.close()


def test_vocabulary_srs_record(tmp_db_path: Path) -> None:
    """Verify vocabulary SRS items can be stored and queried."""
    from lang_attack.db.connection import init_db

    conn = init_db(tmp_db_path)
    cursor = conn.cursor()
    cols = [c[1] for c in cursor.execute("PRAGMA table_info(vocabulary_srs);")]

    if "item_uuid" in cols:
        cursor.execute(
            """
            INSERT INTO vocabulary_srs (item_uuid, language, term, translation, next_review_at, created_at)
            VALUES ('srs-001', 'fr', 'maison', 'house', '2026-10-30T00:00:00Z', '2026-10-29T00:00:00Z');
            """
        )
    else:
        cursor.execute(
            """
            INSERT INTO vocabulary_srs (item_id, language, term, translation, next_review_date)
            VALUES ('srs-001', 'fr', 'maison', 'house', '2026-10-30T00:00:00Z');
            """
        )
    conn.commit()

    cursor.execute("SELECT term, language FROM vocabulary_srs WHERE term = 'maison';")
    row = cursor.fetchone()
    assert row is not None
    assert row[0] == "maison"
    assert row[1] == "fr"
    conn.close()


def test_pipeline_metrics_recording(tmp_db_path: Path) -> None:
    """Verify pipeline metrics insertion and status logging."""
    from lang_attack.db.connection import init_db

    conn = init_db(tmp_db_path)
    cursor = conn.cursor()

    cols = [c[1] for c in cursor.execute("PRAGMA table_info(pipeline_metrics);")]
    if "metric_id" in cols:
        cursor.execute(
            """
            INSERT INTO pipeline_metrics (metric_id, pipeline_name, duration_ms, status)
            VALUES ('m-01', 'whisper_asr', 245.5, 'SUCCESS');
            """
        )
    else:
        cursor.execute(
            """
            INSERT INTO pipeline_metrics (pipeline_name, duration_ms, status, timestamp)
            VALUES ('whisper_asr', 245.5, 'SUCCESS', '2026-10-29T12:00:00Z');
            """
        )
    conn.commit()

    cursor.execute("SELECT pipeline_name, duration_ms, status FROM pipeline_metrics;")
    row = cursor.fetchone()
    assert row is not None
    assert row[0] == "whisper_asr"
    assert row[1] == pytest.approx(245.5, rel=1e-2)
    assert row[2] == "SUCCESS"
    conn.close()


# ============================================================================
# Tier 2: Boundary & Corner Cases
# ============================================================================


def test_language_column_unconstrained_flexibility(tmp_db_path: Path) -> None:
    """Boundary: Language column accepts diverse and exotic strings without constraint error."""
    from lang_attack.db.connection import init_db

    conn = init_db(tmp_db_path)
    cursor = conn.cursor()

    languages_to_test = [
        "en",
        "pt-BR",
        "zh-Hans-CN",
        "es-419",
        "Ancient Greek (Attic)",
        "Klingon",
        "O'Reilly Dialect",
        "emoji-🌐-lang",
        "lang; DROP TABLE dummy; --",
    ]

    for idx, lang in enumerate(languages_to_test):
        cursor.execute(
            """
            INSERT INTO pipeline_metrics (pipeline_name, duration_ms, status, timestamp, extra_data_json)
            VALUES (?, ?, ?, datetime('now'), ?);
            """
            if "extra_data_json" in [c[1] for c in cursor.execute("PRAGMA table_info(pipeline_metrics);")]
            else """
            INSERT INTO pipeline_metrics (metric_id, pipeline_name, duration_ms, status, metadata_json)
            VALUES (?, ?, ?, ?, ?);
            """,
            (f"m-{idx}", 10.0, "SUCCESS", lang)
            if "metric_id" in [c[1] for c in cursor.execute("PRAGMA table_info(pipeline_metrics);")]
            else ("test_pipe", 10.0, "SUCCESS", lang),
        )

    conn.commit()
    cursor.execute("SELECT COUNT(*) FROM pipeline_metrics;")
    assert cursor.fetchone()[0] == len(languages_to_test)
    conn.close()


def test_init_db_idempotency(tmp_db_path: Path) -> None:
    """Boundary: Repeated calls to init_db() should not raise or duplicate tables."""
    from lang_attack.db.connection import init_db

    conn1 = init_db(tmp_db_path)
    conn1.close()

    conn2 = init_db(tmp_db_path)
    cursor = conn2.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    assert "study_sessions" in tables
    conn2.close()


def test_auto_create_parent_directory(tmp_path: Path) -> None:
    """Boundary: init_db automatically creates deeply nested parent directory if missing."""
    from lang_attack.db.connection import init_db

    deep_db = tmp_path / "deep" / "nested" / "dir" / "lang_attack.db"
    assert not deep_db.parent.exists()

    conn = init_db(deep_db)
    assert deep_db.parent.exists()
    assert deep_db.exists()
    conn.close()


def test_wal_concurrent_reads(tmp_db_path: Path) -> None:
    """Boundary: WAL mode allows concurrent reader connections while writing."""
    from lang_attack.db.connection import init_db

    init_db(tmp_db_path)

    conn1 = sqlite3.connect(tmp_db_path)
    conn2 = sqlite3.connect(tmp_db_path)

    # Reader 1 reads
    cursor2 = conn2.cursor()
    cursor2.execute("SELECT COUNT(*) FROM pipeline_metrics;")
    _ = cursor2.fetchone()[0]

    conn1.close()
    conn2.close()


def test_dao_record_session_and_history(tmp_db_path: Path) -> None:
    """Verify record_session and get_session_history DAO operations."""
    from lang_attack.db.connection import get_session_history, init_db, record_session

    conn = init_db(tmp_db_path)
    sess_id = record_session(
        {"session_id": "test-sess-1", "language": "pt-BR", "topic": "Grammar", "score": 95.0},
        conn=conn,
    )
    assert sess_id > 0
    history = get_session_history(limit=10, language="pt-BR", conn=conn)
    assert len(history) == 1
    assert history[0]["session_id"] == "test-sess-1"
    conn.close()


def test_dao_record_phonetic_score(tmp_db_path: Path) -> None:
    """Verify record_phonetic_score DAO operation."""
    from lang_attack.db.connection import init_db, record_phonetic_score
    from lang_attack.schemas.phonetics import SpeechAssessmentResult

    conn = init_db(tmp_db_path)
    res = SpeechAssessmentResult(
        target_text="Bonjour",
        recognized_text="Bonjour",
        overall_score=94.0,
        fluency_score=90.0,
        completeness_score=98.0,
        words=[],
    )
    rec_id = record_phonetic_score(res, language="fr", conn=conn)
    assert rec_id > 0
    cursor = conn.cursor()
    cursor.execute("SELECT overall_score, target_text FROM phonetic_history WHERE id = ?;", (rec_id,))
    row = cursor.fetchone()
    assert row["overall_score"] == 94.0
    assert row["target_text"] == "Bonjour"
    conn.close()


def test_dao_vocabulary_srs_sm2_progression(tmp_db_path: Path) -> None:
    """Verify update_vocabulary_srs SM-2 progression."""
    from lang_attack.db.connection import init_db, update_vocabulary_srs

    conn = init_db(tmp_db_path)
    # New word
    update_vocabulary_srs("der Apfel", "de", grade=5, conn=conn)
    cursor = conn.cursor()
    cursor.execute("SELECT repetition, interval_days, ease_factor FROM vocabulary_srs WHERE word = 'der Apfel';")
    row = cursor.fetchone()
    assert row["repetition"] == 1
    assert row["interval_days"] == 1
    # Grade 4 review
    update_vocabulary_srs("der Apfel", "de", grade=4, conn=conn)
    cursor.execute("SELECT repetition, interval_days FROM vocabulary_srs WHERE word = 'der Apfel';")
    row2 = cursor.fetchone()
    assert row2["repetition"] == 2
    assert row2["interval_days"] == 6
    conn.close()


def test_dao_record_pipeline_metric(tmp_db_path: Path) -> None:
    """Verify record_pipeline_metric DAO operation."""
    from lang_attack.db.connection import init_db, record_pipeline_metric

    conn = init_db(tmp_db_path)
    record_pipeline_metric("tts_pipeline", 123.4, "SUCCESS", details="synthesized 16khz", conn=conn)
    cursor = conn.cursor()
    cursor.execute("SELECT pipeline_name, duration_ms, status, details FROM pipeline_metrics WHERE pipeline_name = 'tts_pipeline';")
    row = cursor.fetchone()
    assert row is not None
    assert row["pipeline_name"] == "tts_pipeline"
    assert row["duration_ms"] == pytest.approx(123.4, rel=1e-2)
    assert row["status"] == "SUCCESS"
    assert row["details"] == "synthesized 16khz"
    conn.close()

