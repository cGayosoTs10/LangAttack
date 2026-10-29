"""Exercise and drill session Pydantic v2 schemas."""

import math
from enum import StrEnum
from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator


class ExerciseType(StrEnum):
    """Enumeration of supported language exercise formats."""

    CLOZE = "cloze"
    MULTIPLE_CHOICE = "multiple_choice"
    SPEECH_REPETITION = "speech_repetition"
    TRANSLATION = "translation"


class ExerciseItem(BaseModel):
    """Specification of an interactive exercise item."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        validation_alias=AliasChoices("id", "item_id"),
        description="Unique exercise item ID",
    )
    type: ExerciseType = Field(..., description="Drill type")
    prompt: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        validation_alias=AliasChoices("prompt", "question"),
        description="Question or exercise prompt text",
    )
    options: list[str] = Field(default_factory=list, max_length=20, description="Available choices")
    correct_answer: str = Field(
        ..., min_length=1, max_length=1000, description="Expected correct answer"
    )
    explanation: str | None = Field(
        default=None,
        max_length=2000,
        validation_alias=AliasChoices("explanation", "context_sentence"),
        description="Grammar or contextual rationale",
    )
    context_sentence: str | None = Field(
        default=None,
        max_length=2000,
        description="Contextual sentence or reading passage",
    )
    acceptable_alternatives: list[str] = Field(
        default_factory=list,
        max_length=50,
        description="List of acceptable alternative answers",
    )
    language: str = Field(
        default="en",
        min_length=1,
        max_length=50,
        description="Target study language code",
    )

    @property
    def question(self) -> str:
        """Alias property for prompt."""
        return self.prompt

    @property
    def item_id(self) -> str:
        """Alias property for item id."""
        return self.id

    @field_validator("prompt", "correct_answer", mode="after")
    @classmethod
    def ensure_non_whitespace(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("Field cannot be empty or purely whitespace")
        return trimmed

    @field_validator("type", mode="before")
    @classmethod
    def normalize_type(cls, value: Any) -> Any:
        """Normalize case-insensitive drill type strings to ExerciseType enum."""
        if isinstance(value, str):
            v_lower = value.strip().lower()
            for member in ExerciseType:
                if member.value == v_lower or member.name.lower() == v_lower:
                    return member
        return value

    @model_validator(mode="after")
    def sync_context_and_validate_choices(self) -> "ExerciseItem":
        """Synchronize explanation/context_sentence and verify multiple-choice constraints."""
        if self.context_sentence is None and self.explanation is not None:
            self.context_sentence = self.explanation
        elif self.explanation is None and self.context_sentence is not None:
            self.explanation = self.context_sentence

        if self.type == ExerciseType.MULTIPLE_CHOICE:
            clean_options = [opt.strip() for opt in self.options if opt.strip()]
            if len(clean_options) < 2:
                raise ValueError(
                    "options must contain at least 2 distinct choices for multiple_choice exercises"
                )
            clean_answer = self.correct_answer.strip()
            if not any(opt.lower() == clean_answer.lower() for opt in clean_options):
                raise ValueError(
                    "correct_answer must be included in options for multiple_choice exercises"
                )
            self.options = clean_options
        return self


class ExerciseSession(BaseModel):
    """Stateful container for an interactive drill session."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        validation_alias=AliasChoices("id", "session_id"),
        description="Session ID",
    )
    title: str = Field(..., min_length=1, max_length=256, description="Session topic or title")
    language: str = Field(default="en", min_length=1, max_length=50, description="Target language identifier")
    items: list[ExerciseItem] = Field(default_factory=list, description="Sequence of drill items")
    current_index: int = Field(default=0, ge=0, description="Active item index")
    score: float = Field(default=0.0, ge=0.0, le=100.0, description="Cumulative score")

    @field_validator("score", mode="after")
    @classmethod
    def check_finite(cls, val: float) -> float:
        if math.isnan(val) or math.isinf(val):
            raise ValueError("score must be finite and not NaN/Inf")
        return val

    @model_validator(mode="before")
    @classmethod
    def normalize_session_id(cls, data: Any) -> Any:
        """Allow initializing id via session_id if id is missing or None."""
        if isinstance(data, dict):
            if not data.get("id") and data.get("session_id"):
                data["id"] = data["session_id"]
            elif not data.get("session_id") and data.get("id"):
                data["session_id"] = data["id"]
        return data

    @property
    def session_id(self) -> str:
        """Alias property for session id."""
        return self.id

    @property
    def is_completed(self) -> bool:
        """Check if all drill items have been navigated."""
        return self.current_index >= len(self.items)


__all__ = [
    "ExerciseItem",
    "ExerciseSession",
    "ExerciseType",
]
