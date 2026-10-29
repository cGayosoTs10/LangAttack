"""Database management and repository package."""

from lang_attack.db.connection import (
    get_connection,
    get_db_path,
    get_due_srs_words,
    get_session_history,
    init_db,
    record_phonetic_score,
    record_pipeline_metric,
    record_session,
    update_vocabulary_srs,
)
from lang_attack.db.repositories import (
    MetricsRepository,
    PhoneticRepository,
    SessionRepository,
    SRSRepository,
)

__all__ = [
    "MetricsRepository",
    "PhoneticRepository",
    "SRSRepository",
    "SessionRepository",
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
