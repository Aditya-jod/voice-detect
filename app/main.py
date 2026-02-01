"""FastAPI application entrypoint."""
from __future__ import annotations

from fastapi import FastAPI

from app.api.routes import detect_router
from app.core.config import get_settings
from app.core.security import register_security
from app.models import ModelRegistry
from app.services.audio import AudioPreprocessor


APP_TITLE = "AI Voice Authenticity API"
APP_VERSION = "0.1.0"


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=APP_TITLE, version=APP_VERSION)

    register_security(app, settings)
    app.include_router(detect_router)

    @app.on_event("startup")
    async def _startup_event() -> None:
        # Instantiate reusable components once during startup to avoid per-request overhead.
        app.state.audio_preprocessor = AudioPreprocessor(
            target_sample_rate=settings.target_sample_rate,
            max_duration_seconds=settings.max_audio_duration_seconds,
        )
        model_registry = ModelRegistry(settings=settings)
        model_registry.load()
        app.state.model_registry = model_registry

    @app.get("/health", tags=["system"])
    async def health_check() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
