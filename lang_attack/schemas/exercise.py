"""Exercise and drill session Pydantic v2 schemas."""

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
        validation_alias=AliasChoices("id", "item_id"),
        description="Unique exercise item ID",
    )
    type: ExerciseType = Field(..., description="Drill type")
    prompt: str = Field(..., min_length=1, description="Question or exercise prompt text")
    options: list[str] = Field(default_factory=list, description="Available choices")
    correct_answer: str = Field(..., description="Expected correct answer")
    explanation: str | None = Field(default=None, description="Grammar or contextual rationale")
    language: str = Field(default="en", description="Target study language code")

    @property
    def item_id(self) -> str:
        """Alias property for item id."""
        return self.id

    @field_validator("type", mode="before")
    @classmethod
    def normalize_type(cls, value: Any) -> Any:
        """Normalize case-insensitive drill type strings to ExerciseType enum."""
        if isinstance(value, str):
            v_lower = value.lower()
            for member in ExerciseType:
                if member.value == v_lower or member.name.lower() == v_lower:
                    return member
        return value

    @model_validator(mode="after")
    def validate_multiple_choice(self) -> "ExerciseItem":
        """Verify multiple-choice constraints (non-empty choices containing the answer)."""
        if self.type == ExerciseType.MULTIPLE_CHOICE:
            if not self.options:
                raise ValueError("options cannot be empty for multiple_choice exercises")
            if self.correct_answer not in self.options:
                raise ValueError(
                    "correct_answer must be included in options for multiple_choice exercises"
                )
        return self


class ExerciseSession(BaseModel):
    """Stateful container for an interactive drill session."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    id: str = Field(
        ...,
        min_length=1,
        validation_alias=AliasChoices("id", "session_id"),
        description="Session ID",
    )
    title: str = Field(..., min_length=1, description="Session topic or title")
    language: str = Field(default="en", description="Target language identifier")
    items: list[ExerciseItem] = Field(default_factory=list, description="Sequence of drill items")
    current_index: int = Field(default=0, ge=0, description="Active item index")
    score: float = Field(default=0.0, description="Cumulative score")

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
