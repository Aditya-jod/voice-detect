"""API key authentication utilities and middleware."""
from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, status
from fastapi.requests import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from app.core.config import Settings, get_settings


class APIKeyAuthMiddleware(BaseHTTPMiddleware):
    """Reject requests lacking the correct API key header."""

    header_name: str = "X-API-KEY"
    public_paths: tuple[str, ...] = ("/", "/health", "/docs", "/openapi.json", "/redoc")
    public_prefixes: tuple[str, ...] = ("/tester",)

    def __init__(self, app: ASGIApp, settings: Optional[Settings] = None) -> None:
        super().__init__(app)
        self._settings = settings or get_settings()

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        path = request.url.path
        if path in self.public_paths or any(path.startswith(prefix) for prefix in self.public_prefixes):
            return await call_next(request)

        provided_key = request.headers.get(self.header_name)
        if not provided_key:
            return self._unauthorized_response("Missing API key header.")

        if provided_key != self._settings.api_key:
            return self._unauthorized_response("Invalid API key.")

        return await call_next(request)

    @staticmethod
    def _unauthorized_response(detail: str) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={
                "error": {
                    "type": "UNAUTHORIZED",
                    "detail": detail,
                }
            },
            headers={"WWW-Authenticate": "API-Key"},
        )


def register_security(app: FastAPI, settings: Optional[Settings] = None) -> None:
    """Attach authentication middleware to the FastAPI app."""

    app.add_middleware(APIKeyAuthMiddleware, settings=settings or get_settings())