"""Pydantic schemas package."""

import fractions  # noqa: F401 - ensure fractions is registered in sys.modules for Python 3.14 pydantic_core

from lang_attack.schemas.exercise import ExerciseItem, ExerciseSession, ExerciseType
from lang_attack.schemas.phonetics import (
    PhonemeEvaluation,
    SpeechAssessmentResult,
    WordEvaluation,
)
from lang_attack.schemas.session import (
    PhoneticHistoryRecord,
    PipelineMetricRecord,
    StudySession,
    StudySessionCreate,
)

__all__ = [
    "ExerciseItem",
    "ExerciseSession",
    "ExerciseType",
    "PhonemeEvaluation",
    "PhoneticHistoryRecord",
    "PipelineMetricRecord",
    "SpeechAssessmentResult",
    "StudySession",
    "StudySessionCreate",
    "WordEvaluation",
]
