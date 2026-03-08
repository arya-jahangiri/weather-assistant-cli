"""Logging setup with text and JSON output modes."""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Mapping, MutableMapping
from datetime import UTC, datetime
from typing import Any, cast

from typing_extensions import override

LoggerLike = logging.Logger | logging.LoggerAdapter[logging.Logger]
_STANDARD_LOG_RECORD_FIELDS = frozenset(logging.makeLogRecord({}).__dict__) | {"message", "asctime"}
_RESERVED_FORMATTER_FIELDS = frozenset({"timestamp", "level", "event", "logger", "exception"})


class ContextLoggerAdapter(logging.LoggerAdapter[logging.Logger]):
    """LoggerAdapter that preserves adapter context and per-call extras."""

    @override
    def process(
        self,
        msg: object,
        kwargs: MutableMapping[str, Any],
    ) -> tuple[object, MutableMapping[str, Any]]:
        """Merge adapter context with any extra fields supplied at log call time."""

        base_extra: Mapping[str, Any]
        if isinstance(self.extra, Mapping):
            base_extra = cast(Mapping[str, Any], self.extra)
        else:
            base_extra = {}
        extra = dict(base_extra)
        call_extra = kwargs.get("extra")
        if isinstance(call_extra, Mapping):
            extra.update(cast(Mapping[str, Any], call_extra))
        kwargs["extra"] = extra
        return msg, kwargs


class JsonFormatter(logging.Formatter):
    """Simple JSON formatter for structured log consumption."""

    @override
    def format(self, record: logging.LogRecord) -> str:
        """Format a log record as a single-line JSON object."""

        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "event": record.getMessage(),
            "logger": record.name,
        }

        payload.update(_extract_record_context(record))

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=True)


class TextFormatter(logging.Formatter):
    """Human-friendly console formatter for local development."""

    @override
    def format(self, record: logging.LogRecord) -> str:
        """Format a log record with optional context fields."""

        base = f"{record.levelname.lower()} {record.name}: {record.getMessage()}"
        context = _extract_record_context(record)
        if context:
            return f"{base} | context={context}"
        return base


def _extract_record_context(record: logging.LogRecord) -> dict[str, Any]:
    """Return user-supplied LogRecord extras while excluding built-in fields."""

    return {
        key: value
        for key, value in record.__dict__.items()
        if key not in _STANDARD_LOG_RECORD_FIELDS and key not in _RESERVED_FORMATTER_FIELDS
    }


def configure_logging(log_format: str, log_level: str = "WARNING") -> None:
    """Configure root logger once for the selected output format."""

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.WARNING))

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setLevel(getattr(logging, log_level.upper(), logging.WARNING))
    if log_format == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(TextFormatter())

    root_logger.addHandler(handler)

    # Suppress noisy third-party transport chatter by default.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


def with_context(logger: logging.Logger, context: Mapping[str, Any]) -> ContextLoggerAdapter:
    """Attach structured context to a logger via LoggerAdapter."""

    return ContextLoggerAdapter(logger, extra=dict(context))
