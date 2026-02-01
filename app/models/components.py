"""Reusable model components for embeddings and classification."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torchaudio


class EmbeddingEncoder:
    """Wrap a torchaudio pipeline for self-supervised speech embeddings."""

    def __init__(self, bundle_name: str) -> None:
        try:
            self.bundle = getattr(torchaudio.pipelines, bundle_name)
        except AttributeError as exc:  # pragma: no cover - sanity guard
            raise ValueError(f"Unknown torchaudio bundle '{bundle_name}'.") from exc

        self.model = self.bundle.get_model()
        self.model.eval()
        self.sample_rate = self.bundle.sample_rate
        self._resampler: Optional[torchaudio.transforms.Resample] = None

    def _ensure_resampler(self, source_rate: int) -> Optional[torchaudio.transforms.Resample]:
        if source_rate == self.sample_rate:
            return None
        if self._resampler is None or self._resampler.orig_freq != source_rate:
            self._resampler = torchaudio.transforms.Resample(orig_freq=source_rate, new_freq=self.sample_rate)
        return self._resampler

    @torch.inference_mode()
    def __call__(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        resampler = self._ensure_resampler(sample_rate)
        if resampler is not None:
            waveform = resampler(waveform)

        features, _ = self.model.extract_features(waveform)
        # Use the last hidden layer and mean-pool over time to get a fixed embedding.
        embedding = features[-1].mean(dim=1).squeeze(0)
        return embedding


class ClassificationHead(torch.nn.Module):
    """Lightweight feed-forward classifier trained offline."""

    def __init__(self, input_dim: int, checkpoint_path: str) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.LayerNorm(input_dim),
            torch.nn.Linear(input_dim, 256),
            torch.nn.GELU(),
            torch.nn.Dropout(p=0.15),
            torch.nn.Linear(256, 2),
        )
        self._load_weights(checkpoint_path)
        self.eval()

    def _load_weights(self, checkpoint_path: str) -> None:
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Classifier checkpoint missing: {path}")
        state_dict = torch.load(path, map_location="cpu")
        self.load_state_dict(state_dict)

    @torch.inference_mode()
    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.net(embedding)
