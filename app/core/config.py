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

    model_cache_dir: str = Field(".cache/models", alias="VOICE_DETECT_MODEL_CACHE")
    embedding_bundle_name: str = Field("WAV2VEC2_ASR_BASE_960H", alias="VOICE_DETECT_EMBEDDING_BUNDLE")
    classifier_checkpoint_path: str = Field(
        "app/models/artifacts/classifier_head.pt", alias="VOICE_DETECT_CLASSIFIER_CKPT"
    )
    classifier_input_dim: int = Field(768, alias="VOICE_DETECT_CLASSIFIER_INPUT_DIM")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance to avoid repeated parsing."""
    return Settings()