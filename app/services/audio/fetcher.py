"""Remote audio download helper."""
from __future__ import annotations

from dataclasses import dataclass

import httpx

from app.services.audio.ingestion import AudioProcessingError


@dataclass(slots=True)
class RemoteAudioFetcher:
    """Download MP3 payloads from remote URLs with size and timeout guards."""

    max_bytes: int
    timeout_seconds: float

    async def fetch(self, url: str) -> bytes:
        timeout = httpx.Timeout(self.timeout_seconds)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream("GET", url) as response:
                    response.raise_for_status()
                    content_type = response.headers.get("Content-Type", "").lower()
                    if content_type and "audio" not in content_type:
                        raise AudioProcessingError("Remote resource is not an audio stream.")

                    buffer = bytearray()
                    async for chunk in response.aiter_bytes():
                        buffer.extend(chunk)
                        if len(buffer) > self.max_bytes:
                            raise AudioProcessingError("Remote audio exceeds maximum allowed size.")
        except httpx.HTTPError as exc:
            raise AudioProcessingError(f"Unable to download remote audio: {exc}") from exc

        if not buffer:
            raise AudioProcessingError("Remote audio download returned empty payload.")

        return bytes(buffer)
