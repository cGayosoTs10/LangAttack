"""Pydantic schemas for session telemetry, study metrics, and history."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class StudySessionCreate(BaseModel):
    """Schema for creating a new study session."""

    language: str = Field(..., min_length=1, max_length=100, description="Language or topic vault identifier")
    start_time: datetime = Field(default_factory=datetime.now)


class StudySession(BaseModel):
    """Schema representing a persisted study session."""

    id: str = Field(..., description="Unique session UUID")
    language: str = Field(..., description="Language identifier")
    start_time: datetime = Field(..., description="Session start timestamp")
    end_time: datetime | None = Field(default=None, description="Session completion timestamp")
    duration_seconds: int = Field(default=0, ge=0, description="Total active practice seconds")
    words_practiced: int = Field(default=0, ge=0, description="Total words assessed or practiced")
    average_score: float = Field(default=0.0, ge=0.0, le=100.0, description="Average pronunciation / exercise score")


class PhoneticHistoryRecord(BaseModel):
    """Longitudinal phoneme audit record."""

    id: int | None = Field(default=None, description="Database primary key")
    session_id: str | None = Field(default=None, description="Associated study session UUID")
    language: str = Field(..., description="Language code")
    word: str = Field(..., description="Evaluated word")
    expected_phoneme: str = Field(..., description="Target IPA phoneme")
    detected_phoneme: str = Field(..., description="Detected IPA phoneme")
    gop_score: float = Field(..., ge=0.0, le=100.0, description="Goodness of Pronunciation score")
    timestamp: datetime = Field(default_factory=datetime.now, description="Assessment timestamp")


class PipelineMetricRecord(BaseModel):
    """Performance telemetry record for AI/audio pipelines."""

    id: int | None = Field(default=None, description="Database primary key")
    pipeline_name: str = Field(..., description="Identifier of the monitored pipeline")
    latency_ms: float = Field(..., ge=0.0, description="Elapsed execution time in milliseconds")
    status: str = Field(..., description="'SUCCESS' or 'FAIL'")
    timestamp: datetime = Field(default_factory=datetime.now, description="Event timestamp")


__all__ = [
    "PhoneticHistoryRecord",
    "PipelineMetricRecord",
    "StudySession",
    "StudySessionCreate",
]
