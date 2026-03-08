"""Shared test fixtures for weather-assistant-cli subsystems."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

from weather_assistant_cli.config import Settings

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Return validated settings populated from test environment values."""

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4.1-mini")
    monkeypatch.setenv("MAX_CONCURRENCY", "5")
    monkeypatch.setenv("OPENAI_RETRY_ATTEMPTS", "2")
    monkeypatch.setenv("OPENAI_RETRY_BASE_DELAY_SECONDS", "0.001")
    monkeypatch.setenv("RETRY_ATTEMPTS", "2")
    monkeypatch.setenv("RETRY_BASE_DELAY_SECONDS", "0.001")
    return Settings()


@pytest.fixture
def logger() -> logging.Logger:
    """Provide a stable logger for dependency construction in tests."""

    return logging.getLogger("test")
