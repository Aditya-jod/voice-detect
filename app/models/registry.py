"""Model registry that wraps a pretrained Hugging Face detector."""
from __future__ import annotations

import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoConfig, AutoFeatureExtractor

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
        self._feature_extractor: Optional[Callable[..., Any]] = None
        self._session: Optional[ort.InferenceSession] = None
        self._ai_index: Optional[int] = None
        self._human_index: Optional[int] = None
        self._logger = logging.getLogger(self.__class__.__name__)
        self._attempted_auto_export = False

    def load(self) -> None:
        """Instantiate feature extractor and ONNX session once during startup."""

        self._logger.info("Initializing feature extractor for '%s'", self.settings.hf_model_name)
        self._feature_extractor = AutoFeatureExtractor.from_pretrained(
            self.settings.hf_model_name,
            cache_dir=self.cache_dir,
        )

        config = AutoConfig.from_pretrained(
            self.settings.hf_model_name,
            cache_dir=self.cache_dir,
        )

        onnx_path = Path(self.settings.onnx_model_path)
        if not onnx_path.is_file():
            self._logger.warning("ONNX model missing at '%s'. Attempting auto-export via scripts/export_to_onnx.py", onnx_path)
            if not self._attempted_auto_export and self._try_generate_onnx(onnx_path):
                self._logger.info("Successfully generated ONNX graph at '%s'", onnx_path)
            if not onnx_path.is_file():
                raise FileNotFoundError(
                    "ONNX model not found at %s. Run the export/quantize scripts or provide a valid VOICE_DETECT_ONNX_PATH." % onnx_path
                )

        self._logger.info("Loading ONNX runtime session from '%s'", onnx_path)
        session_options = ort.SessionOptions()
        session_options.enable_mem_pattern = False
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self._session = ort.InferenceSession(
            str(onnx_path),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )

        id2label = {int(k): v.upper() for k, v in config.id2label.items()}
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

    def _try_generate_onnx(self, target_path: Path) -> bool:
        self._attempted_auto_export = True
        script_path = Path(__file__).resolve().parents[2] / "scripts" / "export_to_onnx.py"
        if not script_path.is_file():
            self._logger.error("Export script missing at '%s'", script_path)
            return False

        output_dir = target_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(script_path),
            "--model-name-or-path",
            self.settings.hf_model_name,
            "--output",
            str(output_dir),
        ]
        env = os.environ.copy()
        env.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

        try:
            subprocess.run(cmd, check=True, env=env)
        except subprocess.CalledProcessError as exc:
            self._logger.error("Auto-export failed with code %s", exc.returncode)
            return False

        return target_path.is_file()

    def predict(self, waveform: torch.Tensor, language: Optional[str] = None) -> InferenceResult:
        """Run synchronous inference using the pretrained model."""

        feature_extractor = self._feature_extractor
        session = self._session

        if feature_extractor is None or session is None:
            raise RuntimeError("Model artefacts have not been loaded yet.")
        if self._ai_index is None or self._human_index is None:
            raise RuntimeError("Model labels have not been configured.")

        audio_array = waveform.squeeze(0).to(torch.float32).cpu().numpy()
        inputs = feature_extractor(
            audio_array,
            sampling_rate=self.settings.target_sample_rate,
            return_tensors="np",
        )
        ort_inputs = {session.get_inputs()[0].name: inputs["input_values"]}
        logits = session.run(None, ort_inputs)[0]
        logits_tensor = torch.from_numpy(np.asarray(logits))

        probabilities = torch.softmax(logits_tensor, dim=-1).squeeze(0).cpu()
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
