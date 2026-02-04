"""Pydantic schemas for the detection endpoint."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import AnyHttpUrl, BaseModel, Field, field_validator, model_validator

SUPPORTED_LANG_CODES = {"en", "hi", "ta", "ml", "te"}


class DetectionRequest(BaseModel):
    """Inbound payload definition for /detect."""

    audio_base64: Optional[str] = Field(
        default=None,
        min_length=1,
        description="Base64-encoded MP3 audio clip.",
    )
    audio_url: Optional[AnyHttpUrl] = Field(
        default=None,
        description="Public URL to an MP3 audio clip.",
    )
    language: Optional[str] = Field(
        default=None,
        description="Optional ISO 639-1 language code: en, hi, ta, ml, te.",
    )

    @field_validator("audio_base64")
    @classmethod
    def _strip_audio(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        text = value.strip()
        if not text:
            raise ValueError("audio_base64 cannot be empty")
        return text

    @field_validator("language")
    @classmethod
    def _validate_language(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        normalized = value.lower()
        if normalized not in SUPPORTED_LANG_CODES:
            raise ValueError("Unsupported language code. Use en, hi, ta, ml, or te.")
        return normalized

    @model_validator(mode="after")
    def _ensure_audio_source(self) -> "DetectionRequest":
        if not self.audio_base64 and not self.audio_url:
            raise ValueError("Provide either audio_base64 or audio_url.")
        return self


ClassificationLabel = Literal["AI_GENERATED", "HUMAN"]


class DetectionResponse(BaseModel):
    """Outbound payload for detection results."""

    classification: ClassificationLabel
    confidence: float = Field(..., ge=0.0, le=1.0)
    explanation: list[str] = Field(..., min_length=1)
