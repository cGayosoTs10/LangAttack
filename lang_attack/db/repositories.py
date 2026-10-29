"""Repository pattern classes for SQLite persistence operations."""

from __future__ import annotations

import sqlite3
from typing import Any

from lang_attack.db.connection import (
    get_connection,
    get_due_srs_words,
    get_session_history,
    record_phonetic_score,
    record_pipeline_metric,
    record_session,
    update_vocabulary_srs,
)


class SessionRepository:
    """Repository managing study sessions and activity persistence."""

    def __init__(self, conn: sqlite3.Connection | None = None):
        self._conn = conn

    def create_or_update(
        self,
        session_id: str,
        language: str,
        start_time: Any,
        end_time: Any = None,
        duration_seconds: int = 0,
        words_practiced: int = 0,
        average_score: float = 0.0,
    ) -> None:
        record_session(
            session_id=session_id,
            language=language,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration_seconds,
            words_practiced=words_practiced,
            average_score=average_score,
            conn=self._conn,
        )

    def list_recent(self, limit: int = 50, language: str | None = None) -> list[dict[str, Any]]:
        return get_session_history(limit=limit, language=language, conn=self._conn)


class PhoneticRepository:
    """Repository managing longitudinal phoneme assessments and audits."""

    def __init__(self, conn: sqlite3.Connection | None = None):
        self._conn = conn

    def add_evaluation(
        self,
        session_id: str | None,
        language: str,
        word: str,
        expected_phoneme: str,
        detected_phoneme: str,
        gop_score: float,
    ) -> int:
        return record_phonetic_score(
            session_id=session_id,
            language=language,
            word=word,
            expected_phoneme=expected_phoneme,
            detected_phoneme=detected_phoneme,
            gop_score=gop_score,
            conn=self._conn,
        )

    def list_history(
        self,
        language: str | None = None,
        min_gop: float | None = None,
        max_gop: float | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        conn = self._conn or get_connection()
        should_close = self._conn is None
        try:
            query = "SELECT * FROM phonetic_history WHERE 1=1"
            params: list[Any] = []
            if language:
                query += " AND language = ?"
                params.append(language)
            if min_gop is not None:
                query += " AND gop_score >= ?"
                params.append(min_gop)
            if max_gop is not None:
                query += " AND gop_score <= ?"
                params.append(max_gop)
            query += " ORDER BY id DESC LIMIT ?"
            params.append(limit)

            cursor = conn.cursor()
            cursor.execute(query, tuple(params))
            return [dict(row) for row in cursor.fetchall()]
        finally:
            if should_close:
                conn.close()


class SRSRepository:
    """Repository managing spaced repetition system (SM-2) vocabulary."""

    def __init__(self, conn: sqlite3.Connection | None = None):
        self._conn = conn

    def review_term(
        self,
        language: str,
        term: str,
        grade: int,
        translation: str | None = None,
        context_sentence: str | None = None,
    ) -> dict[str, Any]:
        return update_vocabulary_srs(
            language=language,
            term=term,
            grade=grade,
            translation=translation,
            context_sentence=context_sentence,
            conn=self._conn,
        )

    def get_due_items(self, language: str | None = None) -> list[dict[str, Any]]:
        return get_due_srs_words(language=language, conn=self._conn)


class MetricsRepository:
    """Repository managing persisted pipeline performance metrics."""

    def __init__(self, conn: sqlite3.Connection | None = None):
        self._conn = conn

    def record_metric(self, pipeline_name: str, latency_ms: float, status: str) -> None:
        record_pipeline_metric(pipeline_name, latency_ms, status, conn=self._conn)

    def list_metrics(self, pipeline_name: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        conn = self._conn or get_connection()
        should_close = self._conn is None
        try:
            query = "SELECT * FROM pipeline_metrics WHERE 1=1"
            params: list[Any] = []
            if pipeline_name:
                query += " AND pipeline_name = ?"
                params.append(pipeline_name)
            query += " ORDER BY id DESC LIMIT ?"
            params.append(limit)

            cursor = conn.cursor()
            cursor.execute(query, tuple(params))
            return [dict(row) for row in cursor.fetchall()]
        finally:
            if should_close:
                conn.close()


__all__ = [
    "MetricsRepository",
    "PhoneticRepository",
    "SRSRepository",
    "SessionRepository",
]
