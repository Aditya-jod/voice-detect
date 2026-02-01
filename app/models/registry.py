"""Model registry for embedding encoder and classifier head."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from app.core.config import Settings
from app.models.components import ClassificationHead, EmbeddingEncoder


@dataclass(slots=True)
class InferenceResult:
    """Structured response from the detector pipeline."""

    classification: str
    confidence: float
    explanation: list[str]


class ModelRegistry:
    """Load and serve embedding + classification models."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.cache_dir = Path(settings.model_cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._encoder: Optional[EmbeddingEncoder] = None
        self._classifier: Optional[ClassificationHead] = None

    def load(self) -> None:
        """Instantiate models once during startup."""

        self._encoder = EmbeddingEncoder(self.settings.embedding_bundle_name)
        self._classifier = ClassificationHead(
            input_dim=self.settings.classifier_input_dim,
            checkpoint_path=self.settings.classifier_checkpoint_path,
        )

    def predict(self, waveform: torch.Tensor, language: Optional[str] = None) -> InferenceResult:
        """Run synchronous inference using preloaded models."""

        if self._encoder is None or self._classifier is None:
            raise RuntimeError("Models have not been loaded yet.")

        embedding = self._encoder(waveform, sample_rate=self.settings.target_sample_rate)
        logits = self._classifier(embedding)
        probabilities = torch.softmax(logits, dim=-1)
        ai_prob = probabilities[..., 1].item()
        classification = "AI_GENERATED" if ai_prob >= 0.5 else "HUMAN"
        confidence = ai_prob if classification == "AI_GENERATED" else 1 - ai_prob

        explanation = [
            "Self-supervised speech embedding analyzed via pretrained wav2vec2 backbone.",
            f"Language hint: {language}" if language else "Language hint: unspecified",
        ]

        return InferenceResult(
            classification=classification,
            confidence=max(confidence, 0.01),
            explanation=explanation,
        )
