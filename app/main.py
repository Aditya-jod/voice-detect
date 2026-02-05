"""FastAPI application entrypoint."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import detect_router
from app.core.config import get_settings
from app.core.logging_config import configure_logging
from app.core.security import register_security
from app.models import ModelRegistry
from app.services.audio import AudioPreprocessor, RemoteAudioFetcher


APP_TITLE = "AI Voice Authenticity API"
APP_VERSION = "0.1.0"
WEB_DIR = Path(__file__).resolve().parent.parent / "web"


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)
    app = FastAPI(title=APP_TITLE, version=APP_VERSION)

    register_security(app, settings)
    app.include_router(detect_router)

    if WEB_DIR.exists():
        # Mount the lightweight testing console at /tester without interfering with the API routes.
        app.mount("/tester", StaticFiles(directory=str(WEB_DIR), html=True), name="tester")

        @app.get("/", include_in_schema=False)
        async def tester_redirect() -> RedirectResponse:
            return RedirectResponse(url="/tester/", status_code=307)

    @app.on_event("startup")
    async def _startup_event() -> None:
        # Instantiate reusable components once during startup to avoid per-request overhead.
        app.state.audio_preprocessor = AudioPreprocessor(
            target_sample_rate=settings.target_sample_rate,
            max_duration_seconds=settings.max_audio_duration_seconds,
        )
        app.state.remote_fetcher = RemoteAudioFetcher(
            max_bytes=settings.max_remote_audio_bytes,
            timeout_seconds=settings.remote_fetch_timeout_seconds,
        )
        model_registry = ModelRegistry(settings=settings)
        model_registry.load()
        app.state.model_registry = model_registry

    @app.get("/health", tags=["system"])
    async def health_check() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
