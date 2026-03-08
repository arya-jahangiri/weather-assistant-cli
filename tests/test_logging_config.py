"""Tests for logging formatters and logger configuration."""

from __future__ import annotations

import json
import logging

import pytest

from weather_assistant_cli.logging_config import (
    JsonFormatter,
    TextFormatter,
    configure_logging,
    with_context,
)


def test_json_formatter_includes_context_fields() -> None:
    """JSON formatter should include arbitrary structured context keys."""

    record = logging.makeLogRecord(
        {
            "name": "weather_assistant_cli",
            "levelno": logging.INFO,
            "levelname": "INFO",
            "msg": "tool_call_finished",
            "args": (),
            "run_id": "run-1",
            "turn_id": "turn-1",
            "query": "London",
            "status": "success",
            "error_code": None,
            "latency_ms": 12.4,
            "attempt": 1,
            "delay_seconds": 0.5,
        }
    )

    payload = json.loads(JsonFormatter().format(record))
    assert payload["event"] == "tool_call_finished"
    assert payload["run_id"] == "run-1"
    assert payload["query"] == "London"
    assert payload["attempt"] == 1
    assert payload["delay_seconds"] == 0.5


def test_text_formatter_emits_context_suffix() -> None:
    """Text formatter should include arbitrary context values when present."""

    record = logging.makeLogRecord(
        {
            "name": "weather_assistant_cli",
            "levelno": logging.INFO,
            "levelname": "INFO",
            "msg": "turn_received",
            "args": (),
            "run_id": "run-1",
            "turn_id": "turn-1",
            "attempt": 0,
            "delay_seconds": 0.25,
        }
    )

    output = TextFormatter().format(record)
    assert "turn_received" in output
    assert "context={" in output
    assert "'attempt': 0" in output
    assert "'delay_seconds': 0.25" in output


def test_retry_log_format_includes_turn_context_and_backoff_fields() -> None:
    """Formatted retry logs should preserve both turn ids and retry metadata."""

    record = logging.makeLogRecord(
        {
            "name": "weather_assistant_cli",
            "levelno": logging.DEBUG,
            "levelname": "DEBUG",
            "msg": "weather_retry_sleep",
            "args": (),
            "run_id": "run-1",
            "turn_id": "turn-1",
            "attempt": 0,
            "delay_seconds": 0.4,
        }
    )

    payload = json.loads(JsonFormatter().format(record))
    assert payload["run_id"] == "run-1"
    assert payload["turn_id"] == "turn-1"
    assert payload["attempt"] == 0
    assert payload["delay_seconds"] == 0.4

    text_output = TextFormatter().format(record)
    assert "'run_id': 'run-1'" in text_output
    assert "'turn_id': 'turn-1'" in text_output
    assert "'attempt': 0" in text_output
    assert "'delay_seconds': 0.4" in text_output


def test_configure_logging_switches_formatter_modes() -> None:
    """configure_logging should attach the expected formatter type."""

    configure_logging("json")
    root = logging.getLogger()
    assert root.handlers
    assert isinstance(root.handlers[0].formatter, JsonFormatter)

    configure_logging("text")
    root = logging.getLogger()
    assert root.handlers
    assert isinstance(root.handlers[0].formatter, TextFormatter)


def test_with_context_returns_logger_adapter() -> None:
    """with_context should preserve context values on adapter extra."""

    adapter = with_context(logging.getLogger("weather_assistant_cli"), {"run_id": "run-1"})
    assert adapter.extra["run_id"] == "run-1"


def test_with_context_merges_call_extra(caplog: pytest.LogCaptureFixture) -> None:
    """Adapter context should merge with per-log extra fields."""

    caplog.set_level(logging.INFO, logger="weather_assistant_cli")
    adapter = with_context(logging.getLogger("weather_assistant_cli"), {"run_id": "run-1"})

    adapter.info("merged_event", extra={"error_code": "E_TEST"})

    record = caplog.records[-1]
    assert record.run_id == "run-1"
    assert record.error_code == "E_TEST"
