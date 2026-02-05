"""Model registry that wraps a pretrained Hugging Face detector."""
from __future__ import annotations

import contextlib
import gzip
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import onnxruntime as ort
import httpx
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
        self._ai_label_name: Optional[str] = None
        self._human_label_name: Optional[str] = None
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
        archive_path = Path(str(onnx_path) + ".gz")
        if not onnx_path.is_file() and archive_path.is_file():
            self._logger.info(
                "Expanding compressed ONNX artifact from '%s' to '%s'",
                archive_path,
                onnx_path,
            )
            if self._extract_local_archive(archive_path, onnx_path):
                self._logger.info("Decompressed ONNX graph to '%s'", onnx_path)
        if not onnx_path.is_file() and self.settings.onnx_download_url:
            self._logger.info(
                "ONNX model missing at '%s'. Downloading artifact from '%s'",
                onnx_path,
                self.settings.onnx_download_url,
            )
            if self._download_onnx(onnx_path, self.settings.onnx_download_url):
                self._logger.info("Successfully downloaded ONNX graph to '%s'", onnx_path)

        if not onnx_path.is_file() and self.settings.allow_auto_export:
            self._logger.warning(
                "ONNX model missing at '%s'. Attempting auto-export via scripts/export_to_onnx.py",
                onnx_path,
            )
            if not self._attempted_auto_export and self._try_generate_onnx(onnx_path):
                self._logger.info("Successfully generated ONNX graph at '%s'", onnx_path)

        if not onnx_path.is_file():
            raise FileNotFoundError(
                "ONNX model not found at %s. Provide VOICE_DETECT_ONNX_PATH, drop a compressed archive alongside it, set VOICE_DETECT_ONNX_URL, or enable VOICE_DETECT_ALLOW_AUTO_EXPORT." % onnx_path
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

        self._ai_index, self._ai_label_name = self._resolve_label_index(
            requested_label=ai_label,
            label2id=label2id,
            kind="AI",
        )
        self._human_index, self._human_label_name = self._resolve_label_index(
            requested_label=human_label,
            label2id=label2id,
            kind="HUMAN",
        )
        self._logger.info(
            "Model ready with labels AI=%s (index %s), HUMAN=%s (index %s)",
            self._ai_label_name,
            self._ai_index,
            self._human_label_name,
            self._human_index,
        )

    def _resolve_label_index(self, requested_label: str, label2id: dict[str, int], kind: str) -> tuple[int, str]:
        label = requested_label.upper()
        if label in label2id:
            return label2id[label], label

        for candidate in self._label_synonyms(kind):
            if candidate in label2id:
                self._logger.warning(
                    "%s label '%s' missing from model. Using fallback '%s' instead.",
                    kind,
                    requested_label,
                    candidate,
                )
                return label2id[candidate], candidate

        available = ", ".join(sorted(label2id.keys()))
        raise ValueError(
            f"{kind} label '{requested_label}' not found in model outputs. Available labels: [{available}]"
        )

    @staticmethod
    def _label_synonyms(kind: str) -> tuple[str, ...]:
        mapping = {
            "AI": ("AI", "FAKE", "SYNTHETIC", "DEEPFAKE", "GENERATED", "BOT"),
            "HUMAN": ("HUMAN", "REAL", "GENUINE", "AUTHENTIC", "LIVE"),
        }
        return mapping.get(kind.upper(), tuple())

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

    def _download_onnx(self, target_path: Path, url: str) -> bool:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = target_path.with_suffix(".tmp")
        timeout = httpx.Timeout(180.0, connect=30.0)
        try:
            with httpx.stream("GET", url, follow_redirects=True, timeout=timeout) as response:
                response.raise_for_status()
                with tmp_path.open("wb") as sink:
                    for chunk in response.iter_bytes(1 << 20):
                        if chunk:
                            sink.write(chunk)
        except Exception as exc:
            self._logger.error("Failed to download ONNX artifact from '%s': %s", url, exc)
            with contextlib.suppress(FileNotFoundError):
                tmp_path.unlink()
            return False

        tmp_path.replace(target_path)
        return True

    def _extract_local_archive(self, archive_path: Path, target_path: Path) -> bool:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
        try:
            with gzip.open(archive_path, "rb") as source, tmp_path.open("wb") as sink:
                shutil.copyfileobj(source, sink)
        except Exception as exc:
            self._logger.error("Failed to expand compressed ONNX artifact '%s': %s", archive_path, exc)
            with contextlib.suppress(FileNotFoundError):
                tmp_path.unlink()
            return False

        tmp_path.replace(target_path)
        return True

    def predict(self, waveform: np.ndarray, language: Optional[str] = None) -> InferenceResult:
        """Run synchronous inference using the pretrained model."""

        feature_extractor = self._feature_extractor
        session = self._session

        if feature_extractor is None or session is None:
            raise RuntimeError("Model artefacts have not been loaded yet.")
        if self._ai_index is None or self._human_index is None:
            raise RuntimeError("Model labels have not been configured.")

        audio_array = np.asarray(waveform.squeeze(0), dtype=np.float32)
        inputs = feature_extractor(
            audio_array,
            sampling_rate=self.settings.target_sample_rate,
            return_tensors="np",
        )
        ort_inputs = {session.get_inputs()[0].name: inputs["input_values"]}
        logits = session.run(None, ort_inputs)[0]
        probabilities = self._softmax(np.asarray(logits))[0]
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
                f"{self._ai_label_name or self.settings.hf_ai_label}: {ai_prob:.3f}, "
                f"{self._human_label_name or self.settings.hf_human_label}: {human_prob:.3f}"
            ),
            f"Language hint: {language}" if language else "Language hint: unspecified",
        ]

        return InferenceResult(
            classification=classification,
            confidence=max(confidence, 0.01),
            explanation=explanation,
        )

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        logits = np.asarray(logits, dtype=np.float32)
        logits -= np.max(logits, axis=-1, keepdims=True)
        exp = np.exp(logits)
        return exp / np.sum(exp, axis=-1, keepdims=True)
