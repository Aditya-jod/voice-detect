"""Audio ingestion utilities: base64 decoding and waveform preprocessing."""
from __future__ import annotations

import base64
import io
from dataclasses import dataclass

import torch
import torchaudio


class AudioProcessingError(RuntimeError):
    """Raised when audio ingestion fails."""


class AudioDecoder:
    """Decode base64-encoded MP3 payloads into raw bytes."""

    @staticmethod
    def decode_base64_mp3(payload: str, max_bytes: int | None = None) -> bytes:
        normalized_payload = payload.strip()
        try:
            payload_size = len(normalized_payload.encode("ascii"))
        except UnicodeEncodeError as exc:
            raise AudioProcessingError("Base64 audio payload must be ASCII.") from exc

        if max_bytes is not None and payload_size > max_bytes:
            raise AudioProcessingError("Base64 payload exceeds maximum allowed size.")

        try:
            return base64.b64decode(normalized_payload, validate=True)
        except (base64.binascii.Error, ValueError) as exc:  # type: ignore[attr-defined]
            raise AudioProcessingError("Invalid base64 audio payload.") from exc


@dataclass(slots=True)
class AudioPreprocessor:
    """Convert MP3 bytes into a normalized mono waveform tensor."""

    target_sample_rate: int
    max_duration_seconds: float

    def __post_init__(self) -> None:
        if self.target_sample_rate <= 0:
            raise ValueError("target_sample_rate must be positive")
        if self.max_duration_seconds <= 0:
            raise ValueError("max_duration_seconds must be positive")

    def __call__(self, mp3_bytes: bytes) -> torch.Tensor:
        waveform, sample_rate = self._load_waveform(mp3_bytes)
        waveform = self._ensure_mono(waveform)
        waveform = self._resample_if_needed(waveform, sample_rate)
        self._validate_duration(waveform)
        return self._normalize(waveform)

    def _load_waveform(self, mp3_bytes: bytes) -> tuple[torch.Tensor, int]:
        buffer = io.BytesIO(mp3_bytes)
        try:
            waveform, sample_rate = torchaudio.load(buffer)
        except Exception as exc:  # pragma: no cover - torchaudio raises many subclasses
            raise AudioProcessingError("Unable to decode MP3 bytes.") from exc

        if waveform.numel() == 0:
            raise AudioProcessingError("Decoded audio is empty.")
        return waveform, sample_rate

    def _ensure_mono(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.size(0) == 1:
            return waveform
        return waveform.mean(dim=0, keepdim=True)

    def _resample_if_needed(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if sample_rate == self.target_sample_rate:
            return waveform
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)
        return resampler(waveform)

    def _validate_duration(self, waveform: torch.Tensor) -> None:
        max_samples = int(self.target_sample_rate * self.max_duration_seconds)
        if waveform.size(-1) > max_samples:
            duration_seconds = waveform.size(-1) / self.target_sample_rate
            raise AudioProcessingError(
                f"Audio duration {duration_seconds:.2f}s exceeds limit of {self.max_duration_seconds:.2f}s."
            )

    @staticmethod
    def _normalize(waveform: torch.Tensor) -> torch.Tensor:
        peak = waveform.abs().max()
        if peak == 0:
            return waveform
        return waveform / peak