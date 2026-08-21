"""Application configuration loaded from environment / .env file."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Type-safe application settings loaded from env / .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    model_name: str = Field(default="facebook/bart-large-cnn")
    max_concurrency: int = Field(default=4, ge=1, le=64)
    chunk_size: int = Field(default=1000, ge=100)
    log_level: str = Field(default="INFO")
    quantize: bool = Field(
        default=False,
        description="Enable dynamic INT8 quantization for faster CPU inference.",
    )

    max_output_tokens: int = Field(
        default=256,
        ge=32,
        le=1024,
        description="Maximum number of tokens in the generated summary.",
    )


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()