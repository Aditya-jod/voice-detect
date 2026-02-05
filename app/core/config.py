"""Application configuration utilities."""
from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Centralized runtime settings sourced from environment variables."""

    api_key: str = Field(..., alias="VOICE_DETECT_API_KEY")
    log_level: str = Field("INFO", alias="VOICE_DETECT_LOG_LEVEL")
    target_sample_rate: int = Field(16000, alias="VOICE_DETECT_SAMPLE_RATE")
    max_audio_duration_seconds: float = Field(30.0, alias="VOICE_DETECT_MAX_DURATION")
    max_base64_audio_bytes: int = Field(6 * 1024 * 1024, alias="VOICE_DETECT_MAX_B64_BYTES")
    inference_timeout_seconds: float = Field(8.0, alias="VOICE_DETECT_INFERENCE_TIMEOUT")
    max_remote_audio_bytes: int = Field(8 * 1024 * 1024, alias="VOICE_DETECT_MAX_REMOTE_BYTES")
    remote_fetch_timeout_seconds: float = Field(5.0, alias="VOICE_DETECT_REMOTE_TIMEOUT")

    model_cache_dir: str = Field(".cache/models", alias="VOICE_DETECT_MODEL_CACHE")
    hf_model_name: str = Field("MelodyMachine/Deepfake-audio-detection-V2", alias="VOICE_DETECT_HF_MODEL")
    hf_cache_dir: str = Field(".cache/hf", alias="VOICE_DETECT_HF_CACHE")
    hf_ai_label: str = Field("AI", alias="VOICE_DETECT_HF_AI_LABEL")
    hf_human_label: str = Field("HUMAN", alias="VOICE_DETECT_HF_HUMAN_LABEL")
    onnx_model_path: str = Field("onnx-model/model/model.onnx", alias="VOICE_DETECT_ONNX_PATH")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance to avoid repeated parsing."""
    return Settings()  # pyright: ignore[reportCallIssue]