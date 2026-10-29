"""Phonetics assessment Pydantic v2 schemas."""

import math
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator


class PhonemeEvaluation(BaseModel):
    """Evaluation of an individual phoneme."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    phoneme: str = Field(..., min_length=1, description="Target or detected phoneme symbol")
    score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        validation_alias=AliasChoices("score", "gop_score"),
        description="GOP score [0.0, 100.0]",
    )
    status: Literal["correct", "warning", "incorrect"] = Field(
        ..., description="Assessment status category"
    )
    articulation_tip: str | None = Field(
        default=None, description="Actionable articulation placement guidance"
    )
    ipa_symbol: str | None = Field(
        default=None,
        validation_alias=AliasChoices("ipa_symbol", "ipa"),
        description="Standard IPA symbol",
    )
    expected_phoneme: str | None = Field(default=None, description="Canonical phoneme")
    detected_phoneme: str | None = Field(default=None, description="Detected phoneme")
    is_correct: bool | None = Field(default=None, description="Correctness flag")
    start_time: float | None = Field(default=None, ge=0.0, description="Start timestamp (s)")
    end_time: float | None = Field(default=None, ge=0.0, description="End timestamp (s)")

    @property
    def gop_score(self) -> float:
        """Alias property for score."""
        return self.score

    @property
    def ipa(self) -> str | None:
        """Alias property for ipa_symbol."""
        return self.ipa_symbol

    @field_validator("status", mode="before")
    @classmethod
    def normalize_status(cls, val: Any) -> Any:
        if isinstance(val, str):
            return val.strip().lower()
        return val

    @field_validator("score", "start_time", "end_time", mode="after")
    @classmethod
    def check_finite(cls, val: float | None) -> float | None:
        if val is not None and (math.isnan(val) or math.isinf(val)):
            raise ValueError("Numeric values must be finite and not NaN/Inf")
        return val

    @model_validator(mode="after")
    def validate_timestamps(self) -> "PhonemeEvaluation":
        """Verify time bounds are chronological."""
        if (
            self.start_time is not None
            and self.end_time is not None
            and self.end_time < self.start_time
        ):
            raise ValueError("end_time cannot be earlier than start_time")
        return self


class WordEvaluation(BaseModel):
    """Pronunciation evaluation of a word token."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    word: str = Field(..., min_length=1, description="Word token")
    score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        validation_alias=AliasChoices("score", "gop_score"),
        description="Word pronunciation score [0.0, 100.0]",
    )
    phonemes: list[PhonemeEvaluation] = Field(
        default_factory=list, description="Constituent phonemes"
    )
    start_time: float | None = Field(default=None, ge=0.0, description="Start timestamp (s)")
    end_time: float | None = Field(default=None, ge=0.0, description="End timestamp (s)")
    is_correct: bool | None = Field(default=None, description="Correctness flag")

    @property
    def gop_score(self) -> float:
        """Alias property for score."""
        return self.score

    @field_validator("score", "start_time", "end_time", mode="after")
    @classmethod
    def check_finite(cls, val: float | None) -> float | None:
        if val is not None and (math.isnan(val) or math.isinf(val)):
            raise ValueError("Numeric values must be finite and not NaN/Inf")
        return val

    @model_validator(mode="after")
    def validate_timestamps(self) -> "WordEvaluation":
        """Verify time bounds are chronological."""
        if (
            self.start_time is not None
            and self.end_time is not None
            and self.end_time < self.start_time
        ):
            raise ValueError("end_time cannot be earlier than start_time")
        return self


class SpeechAssessmentResult(BaseModel):
    """Holistic speech assessment result across an utterance."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    target_text: str = Field(
        default="",
        validation_alias=AliasChoices("target_text", "reference_text"),
        description="Target reference text",
    )
    recognized_text: str = Field(
        default="",
        validation_alias=AliasChoices("recognized_text", "transcript"),
        description="Recognized transcript hypothesis",
    )
    overall_score: float = Field(
        ..., ge=0.0, le=100.0, description="Overall pronunciation score [0.0, 100.0]"
    )
    words: list[WordEvaluation] = Field(
        default_factory=list, description="Word-level evaluations"
    )
    recorded_audio_path: str | None = Field(default=None, description="Path to audio file")
    audio_duration_s: float | None = Field(default=None, description="Audio length in seconds")
    fluency_score: float | None = Field(
        default=None, ge=0.0, le=100.0, description="Fluency score [0.0, 100.0]"
    )
    completeness_score: float | None = Field(
        default=None, ge=0.0, le=100.0, description="Completeness score [0.0, 100.0]"
    )
    latency_ms: float = Field(
        default=0.0, ge=0.0, description="Inference pipeline latency (ms)"
    )
    feedback_tips: list[str] = Field(
        default_factory=list, description="Actionable articulation feedback"
    )
    articulation_tips: list[str] = Field(
        default_factory=list, description="Targeted articulatory guidance"
    )
    pedagogical_feedback: str | None = Field(
        default=None, description="Primary pedagogical feedback tip"
    )

    @field_validator("overall_score", "fluency_score", "completeness_score", "latency_ms", mode="after")
    @classmethod
    def check_finite(cls, val: float | None) -> float | None:
        if val is not None and (math.isnan(val) or math.isinf(val)):
            raise ValueError("Numeric values must be finite and not NaN/Inf")
        return val

    @model_validator(mode="after")
    def populate_pedagogical_feedback(self) -> "SpeechAssessmentResult":
        if self.pedagogical_feedback is None:
            if self.feedback_tips:
                self.pedagogical_feedback = self.feedback_tips[0]
            elif self.articulation_tips:
                self.pedagogical_feedback = self.articulation_tips[0]
        return self

    @property
    def reference_text(self) -> str:
        """Alias property for target_text."""
        return self.target_text

    @property
    def transcript(self) -> str:
        """Alias property for recognized_text."""
        return self.recognized_text


__all__ = [
    "PhonemeEvaluation",
    "SpeechAssessmentResult",
    "WordEvaluation",
]
