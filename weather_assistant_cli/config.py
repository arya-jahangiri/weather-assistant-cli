"""Runtime configuration for weather-assistant-cli."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: SecretStr = Field(alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4.1-mini", alias="OPENAI_MODEL")
    log_format: Literal["text", "json"] = Field(default="text", alias="LOG_FORMAT")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="WARNING",
        alias="LOG_LEVEL",
    )
    max_concurrency: int = Field(default=5, ge=1, le=20, alias="MAX_CONCURRENCY")
    http_timeout_seconds: float = Field(default=10.0, gt=0, alias="HTTP_TIMEOUT_SECONDS")
    openai_timeout_seconds: float = Field(default=45.0, gt=0, alias="OPENAI_TIMEOUT_SECONDS")
    openai_retry_attempts: int = Field(default=2, ge=0, le=6, alias="OPENAI_RETRY_ATTEMPTS")
    openai_retry_base_delay_seconds: float = Field(
        default=0.5,
        gt=0,
        le=5,
        alias="OPENAI_RETRY_BASE_DELAY_SECONDS",
    )
    max_input_chars: int = Field(default=500, ge=1, le=5000, alias="MAX_INPUT_CHARS")
    retry_attempts: int = Field(default=2, ge=0, le=6, alias="RETRY_ATTEMPTS")
    retry_base_delay_seconds: float = Field(
        default=0.4,
        gt=0,
        le=5,
        alias="RETRY_BASE_DELAY_SECONDS",
    )
    tool_iteration_limit: int = Field(default=6, ge=1, le=12, alias="TOOL_ITERATION_LIMIT")

    @field_validator("openai_model")
    @classmethod
    def validate_model_name(cls, value: str) -> str:
        """Ensure the configured model name is non-empty after trimming."""

        normalized = value.strip()
        if not normalized:
            raise ValueError("OPENAI_MODEL cannot be empty")
        return normalized


def load_settings() -> Settings:
    """Load and validate settings from environment variables."""

    return Settings()  # pyright: ignore[reportCallIssue]
