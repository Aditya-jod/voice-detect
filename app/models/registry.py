"""Model registry that wraps a pretrained Hugging Face detector."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

from app.core.config import Settings
from app.schemas.detection import ClassificationLabel


@dataclass(slots=True)
class InferenceResult:
    """Structured response from the detector pipeline."""

    classification: ClassificationLabel
    confidence: float
    explanation: list[str]


class ModelRegistry:
    """Load and serve a Hugging Face audio classification model."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.cache_dir = Path(settings.hf_cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._feature_extractor: Optional[Any] = None
        self._model: Optional[Any] = None
        self._ai_index: Optional[int] = None
        self._human_index: Optional[int] = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._logger = logging.getLogger(self.__class__.__name__)

    def load(self) -> None:
        """Instantiate transformers components once during startup."""

        self._logger.info("Loading Hugging Face model '%s'", self.settings.hf_model_name)
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            self.settings.hf_model_name,
            cache_dir=self.cache_dir,
        )
        model = AutoModelForAudioClassification.from_pretrained(
            self.settings.hf_model_name,
            cache_dir=self.cache_dir,
        ).to(self._device)
        model.eval()

        self._feature_extractor = feature_extractor
        self._model = model

        id2label = {int(k): v.upper() for k, v in model.config.id2label.items()}
        label2id = {label: idx for idx, label in id2label.items()}

        ai_label = self.settings.hf_ai_label.upper()
        human_label = self.settings.hf_human_label.upper()
        missing = [label for label in (ai_label, human_label) if label not in label2id]
        if missing:
            raise ValueError(f"Labels {missing} not found in model id2label mapping: {id2label}")

        self._ai_index = label2id[ai_label]
        self._human_index = label2id[human_label]
        self._logger.info(
            "Model ready with labels AI=%s (index %s), HUMAN=%s (index %s)",
            ai_label,
            self._ai_index,
            human_label,
            self._human_index,
        )

    def predict(self, waveform: torch.Tensor, language: Optional[str] = None) -> InferenceResult:
        """Run synchronous inference using the pretrained model."""

        feature_extractor = self._feature_extractor
        model = self._model

        if feature_extractor is None or model is None:
            raise RuntimeError("Models have not been loaded yet.")
        if self._ai_index is None or self._human_index is None:
            raise RuntimeError("Model labels have not been configured.")

        audio_array = waveform.squeeze(0).to(torch.float32).cpu().numpy()
        inputs = feature_extractor(
            audio_array,
            sampling_rate=self.settings.target_sample_rate,
            return_tensors="pt",
        )
        inputs = {key: value.to(self._device) for key, value in inputs.items()}

        with torch.inference_mode():
            logits = model(**inputs).logits

        probabilities = torch.softmax(logits, dim=-1).squeeze(0).cpu()
        ai_prob = float(probabilities[self._ai_index])
        human_prob = float(probabilities[self._human_index])

        classification: ClassificationLabel
        if ai_prob >= human_prob:
            classification = "AI_GENERATED"
            confidence = ai_prob
        else:
            classification = "HUMAN"
            confidence = human_prob

        explanation = [
            (
                f"HF model {self.settings.hf_model_name} confidence -> "
                f"{self.settings.hf_ai_label}: {ai_prob:.3f}, {self.settings.hf_human_label}: {human_prob:.3f}"
            ),
            f"Language hint: {language}" if language else "Language hint: unspecified",
        ]

        return InferenceResult(
            classification=classification,
            confidence=max(confidence, 0.01),
            explanation=explanation,
        )
