"""Audio ingestion utilities: base64 decoding and waveform preprocessing."""
from __future__ import annotations

import base64
import io
from dataclasses import dataclass

import librosa
import numpy as np


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
    """Convert MP3/streaming bytes into a normalized mono waveform array."""

    target_sample_rate: int
    max_duration_seconds: float

    def __post_init__(self) -> None:
        if self.target_sample_rate <= 0:
            raise ValueError("target_sample_rate must be positive")
        if self.max_duration_seconds <= 0:
            raise ValueError("max_duration_seconds must be positive")

    def __call__(self, audio_bytes: bytes) -> np.ndarray:
        waveform, sample_rate = self._load_waveform(audio_bytes)
        waveform = self._ensure_mono(waveform)
        waveform = self._resample_if_needed(waveform, sample_rate)
        self._validate_duration(waveform)
        return self._normalize(waveform)

    def _load_waveform(self, audio_bytes: bytes) -> tuple[np.ndarray, int]:
        buffer = io.BytesIO(audio_bytes)
        try:
            waveform, sample_rate = librosa.load(buffer, sr=None, mono=False)
        except Exception as exc:  # pragma: no cover - librosa surfaces many decoder errors
            raise AudioProcessingError("Unable to decode audio bytes.") from exc

        if waveform.size == 0:
            raise AudioProcessingError("Decoded audio is empty.")

        if waveform.ndim == 1:
            waveform = waveform[np.newaxis, :]

        return waveform.astype(np.float32, copy=False), int(sample_rate)

    def _ensure_mono(self, waveform: np.ndarray) -> np.ndarray:
        if waveform.shape[0] == 1:
            return waveform
        return waveform.mean(axis=0, keepdims=True)

    def _resample_if_needed(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        if sample_rate == self.target_sample_rate:
            return waveform
        return librosa.resample(
            waveform,
            orig_sr=sample_rate,
            target_sr=self.target_sample_rate,
            axis=-1,
        )

    def _validate_duration(self, waveform: np.ndarray) -> None:
        max_samples = int(self.target_sample_rate * self.max_duration_seconds)
        if waveform.shape[-1] > max_samples:
            duration_seconds = waveform.shape[-1] / self.target_sample_rate
            raise AudioProcessingError(
                f"Audio duration {duration_seconds:.2f}s exceeds limit of {self.max_duration_seconds:.2f}s."
            )

    @staticmethod
    def _normalize(waveform: np.ndarray) -> np.ndarray:
        peak = float(np.abs(waveform).max())
        if peak == 0:
            return waveform
        return waveform / peak