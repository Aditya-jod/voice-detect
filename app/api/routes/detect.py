"""Detection endpoint implementation."""
from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.core.config import Settings, get_settings
from app.models import InferenceResult, ModelRegistry
from app.schemas import DetectionRequest, DetectionResponse
from app.services.audio import AudioDecoder, AudioPreprocessor, AudioProcessingError

router = APIRouter(prefix="/detect", tags=["detection"])
logger = logging.getLogger(__name__)


def _bad_request(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail={"error": {"type": "BAD_REQUEST", "detail": detail}},
    )


def _fallback_response() -> DetectionResponse:
    return DetectionResponse(
        classification="HUMAN",
        confidence=0.05,
        explanation=["Model fallback due to inference error"],
    )


@router.post("", response_model=DetectionResponse)
async def detect_voice(
    payload: DetectionRequest,
    request: Request,
    settings: Settings = Depends(get_settings),
) -> DetectionResponse:
    preprocessor: AudioPreprocessor = request.app.state.audio_preprocessor
    model_registry: ModelRegistry = request.app.state.model_registry

    try:
        audio_bytes = AudioDecoder.decode_base64_mp3(
            payload.audio_base64,
            max_bytes=settings.max_base64_audio_bytes,
        )
        waveform = preprocessor(audio_bytes)
    except AudioProcessingError as exc:
        raise _bad_request(str(exc)) from exc

    inference_call = asyncio.to_thread(model_registry.predict, waveform, payload.language)

    try:
        result: InferenceResult = await asyncio.wait_for(
            inference_call,
            timeout=settings.inference_timeout_seconds,
        )
    except asyncio.TimeoutError:
        logger.warning("Inference timed out after %.2fs", settings.inference_timeout_seconds)
        return _fallback_response()
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.exception("Inference pipeline failed: %s", exc)
        return _fallback_response()

    return DetectionResponse(
        classification=result.classification,
        confidence=result.confidence,
        explanation=result.explanation,
    )
