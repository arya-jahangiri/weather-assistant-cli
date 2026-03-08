"""Tests for settings loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from weather_assistant_cli.config import Settings, load_settings


def test_settings_reject_empty_model_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """OPENAI_MODEL must not be blank after trimming."""

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "   ")

    with pytest.raises(ValidationError):
        Settings()


def test_load_settings_reads_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """load_settings should return a populated Settings object from env vars."""

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    settings = load_settings()
    assert settings.openai_model == "gpt-4.1-mini"
    assert settings.log_level == "WARNING"
    assert settings.openai_retry_attempts == 2
    assert settings.openai_retry_base_delay_seconds == 0.5
