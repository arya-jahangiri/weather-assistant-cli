"""Tests for OpenAI Responses streaming gateway behavior."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any, cast

import httpx
import pytest
from openai import APIStatusError, APITimeoutError

import weather_assistant_cli.openai_gateway as openai_gateway_module
from weather_assistant_cli.logging_config import with_context
from weather_assistant_cli.openai_gateway import (
    OpenAIAuthFailure,
    OpenAIClientFailure,
    OpenAIProtocolError,
    OpenAIStreamInterrupted,
    OpenAITransientFailure,
    ResponsesGateway,
)


@dataclass
class StreamPlan:
    """One scripted stream invocation outcome."""

    events: list[dict[str, object]]
    final_response: dict[str, object]
    exception_factory: Callable[[], Exception] | None = None
    iteration_exception_factory: Callable[[], Exception] | None = None


class FakeStream:
    """Async stream context manager used by scripted gateway tests."""

    def __init__(self, plan: StreamPlan) -> None:
        """Store scripted events and final response."""

        self._plan = plan

    async def __aenter__(self) -> FakeStream:
        """Return self when entering async context manager."""

        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        """Exit context manager without swallowing errors."""

        return None

    def __aiter__(self) -> AsyncIterator[dict[str, object]]:
        """Yield scripted stream events asynchronously."""

        return self._iter_events()

    async def _iter_events(self) -> AsyncIterator[dict[str, object]]:
        """Yield each scripted event in order."""

        for event in self._plan.events:
            yield event
        if self._plan.iteration_exception_factory is not None:
            raise self._plan.iteration_exception_factory()

    async def get_final_response(self) -> dict[str, object]:
        """Return scripted final response payload."""

        return self._plan.final_response


class ScriptedResponses:
    """Scripted fake for openai.responses surface."""

    def __init__(self, plans: list[StreamPlan]) -> None:
        """Initialize with a queue of stream outcomes."""

        self._plans = plans
        self.calls: list[dict[str, object]] = []
        self.create_calls: list[dict[str, object]] = []

    def stream(self, **kwargs: object) -> FakeStream:
        """Return next scripted stream or raise scripted exception."""

        self.calls.append(kwargs)
        plan = self._plans.pop(0)
        if plan.exception_factory is not None:
            raise plan.exception_factory()
        return FakeStream(plan)

    async def create(self, **kwargs: object) -> object:
        """Return the next scripted non-streaming response."""

        self.create_calls.append(kwargs)
        plan = self._plans.pop(0)
        if plan.exception_factory is not None:
            raise plan.exception_factory()
        return plan.final_response


class FakeOpenAI:
    """Container exposing scripted responses fake."""

    def __init__(self, plans: list[StreamPlan]) -> None:
        """Attach scripted responses facade."""

        self.responses = ScriptedResponses(plans)
        self.close_calls = 0

    async def close(self) -> None:
        """Track async close calls for gateway shutdown tests."""

        self.close_calls += 1


def _timeout_error() -> APITimeoutError:
    """Create APITimeoutError with synthetic request metadata."""

    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    return APITimeoutError(request=request)


def _status_error(status_code: int) -> APIStatusError:
    """Create APIStatusError with the given HTTP status."""

    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    response = httpx.Response(status_code, request=request)
    return APIStatusError(f"HTTP {status_code}", response=response, body={})


@pytest.mark.asyncio
async def test_stream_turn_suppresses_buffered_text_when_first_output_item_is_function_call(
    settings,
    logger,
) -> None:
    """Tool rounds should drop buffered text once the stream declares a function call."""

    fake_openai = FakeOpenAI(
        [
            StreamPlan(
                events=[
                    {"type": "response.output_text.delta", "delta": "Hello "},
                    {"type": "response.output_text.delta", "delta": "world"},
                    {"type": "response.output_item.added", "item": {"type": "function_call"}},
                ],
                final_response={
                    "id": "resp_1",
                    "output": [
                        {
                            "type": "function_call",
                            "call_id": "call_london",
                            "name": "get_weather",
                            "arguments": '{"query": "London"}',
                        }
                    ],
                },
            )
        ]
    )
    gateway = ResponsesGateway(
        settings=settings,
        logger=logger,
        tool_schemas=[{"type": "function", "name": "get_weather"}],
        instructions="test-instructions",
        openai_client=cast(Any, fake_openai),
    )

    chunks: list[str] = []
    response = await gateway.stream_turn("weather in london", None, chunks.append)

    assert chunks == []
    assert response.response_id == "resp_1"
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].call_id == "call_london"
    assert response.tool_calls[0].name == "get_weather"
    assert response.output_text == ""


@pytest.mark.asyncio
async def test_stream_turn_flushes_buffered_text_when_first_output_item_is_message(
    settings,
    logger,
) -> None:
    """Text should stream once the first output item is classified as a message."""

    fake_openai = FakeOpenAI(
        [
            StreamPlan(
                events=[
                    {"type": "response.output_text.delta", "delta": "Hello "},
                    {"type": "response.output_item.added", "item": {"type": "message"}},
                    {"type": "response.output_text.delta", "delta": "world"},
                ],
                final_response={
                    "id": "resp_1",
                    "output": [
                        {
                            "type": "message",
                            "content": [{"type": "output_text", "text": "Hello world"}],
                        }
                    ],
                },
            )
        ]
    )
    gateway = ResponsesGateway(
        settings=settings,
        logger=logger,
        tool_schemas=[{"type": "function", "name": "get_weather"}],
        instructions="test-instructions",
        openai_client=cast(Any, fake_openai),
    )

    chunks: list[str] = []
    response = await gateway.stream_turn("hello", None, chunks.append)

    assert "".join(chunks) == "Hello world"
    assert response.tool_calls == []
    assert response.output_text == "Hello world"


@pytest.mark.asyncio
async def test_stream_turn_extracts_refusal_text_from_final_response(settings, logger) -> None:
    """Refusal-only final responses should surface visible text instead of appearing blank."""

    fake_openai = FakeOpenAI(
        [
            StreamPlan(
                events=[],
                final_response={
                    "id": "resp_1",
                    "output_text": "",
                    "output": [
                        {
                            "type": "message",
                            "content": [
                                {
                                    "type": "refusal",
                                    "refusal": "I can only help with weather requests.",
                                }
                            ],
                        }
                    ],
                },
            )
        ]
    )
    gateway = ResponsesGateway(
        settings=settings,
        logger=logger,
        tool_schemas=[{"type": "function", "name": "get_weather"}],
        instructions="test-instructions",
        openai_client=cast(Any, fake_openai),
    )

    response = await gateway.stream_turn("tell me a joke", None, lambda _chunk: None)

    assert response.output_text == "I can only help with weather requests."


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("final_response", "message_fragment"),
    [
        ({}, "response.id"),
        ({"id": "   ", "output": []}, "response.id"),
        ({"id": "resp_1", "output": {}}, "response.output"),
        (
            {"id": "resp_1", "output": [{"type": "function_call", "name": "get_weather"}]},
            "function_call.call_id",
        ),
        (
            {"id": "resp_1", "output": [{"type": "function_call", "call_id": "call_1"}]},
            "function_call.name",
        ),
        (
            {
                "id": "resp_1",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "get_weather",
                        "arguments": None,
                    }
                ],
            },
            "arguments=None",
        ),
        (
            {
                "id": "resp_1",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "get_weather",
                        "arguments": object(),
                    }
                ],
            },
            "non-serializable",
        ),
    ],
)
async def test_stream_turn_rejects_malformed_final_payload(
    settings,
    logger,
    final_response: dict[str, object],
    message_fragment: str,
) -> None:
    """Malformed final responses should raise OpenAIProtocolError."""

    gateway = ResponsesGateway(
        settings=settings,
        logger=logger,
        tool_schemas=[{"type": "function", "name": "get_weather"}],
        instructions="test-instructions",
        openai_client=cast(
            Any,
            FakeOpenAI([StreamPlan(events=[], final_response=final_response)]),
        ),
    )

    with pytest.raises(OpenAIProtocolError, match=message_fragment):
        await gateway.stream_turn("weather", None, lambda _chunk: None)


@pytest.mark.asyncio
async def test_stream_turn_forwards_previous_response_id_and_parallel_tool_calls(
    settings,
    logger,
) -> None:
    """Gateway should include previous_response_id and enable parallel tool calls."""

    fake_openai = FakeOpenAI([StreamPlan(events=[], final_response={"id": "resp_2", "output": []})])
    gateway = ResponsesGateway(
        settings=settings,
        logger=logger,
        tool_schemas=[{"type": "function", "name": "get_weather"}],
        instructions="test-instructions",
        openai_client=cast(Any, fake_openai),
    )

    await gateway.stream_turn("weather", "resp_prev", lambda _chunk: None)

    assert fake_openai.responses.calls
    assert fake_openai.responses.calls[0]["previous_response_id"] == "resp_prev"
    assert fake_openai.responses.calls[0]["parallel_tool_calls"] is True


@pytest.mark.asyncio
async def test_stream_turn_retries_timeout(settings, logger) -> None:
    """Transient timeout should be retried before succeeding."""

    fake_openai = FakeOpenAI(
        [
            StreamPlan(events=[], final_response={}, exception_factory=_timeout_error),
            StreamPlan(events=[], final_response={"id": "resp_ok", "output": []}),
        ]
    )
    gateway = ResponsesGateway(
        settings=settings,
        logger=logger,
        tool_schemas=[{"type": "function", "name": "get_weather"}],
        instructions="test-instructions",
        openai_client=cast(Any, fake_openai),
    )

    response = await gateway.stream_turn("weather", None, lambda _chunk: None)

    assert response.response_id == "resp_ok"
    assert len(fake_openai.responses.calls) == 2


@pytest.mark.asyncio
async def test_stream_turn_retries_retryable_statuses(settings, logger) -> None:
    """HTTP 5xx and 429 errors should be retried before succeeding."""

    for first_error in (_status_error(500), _status_error(429)):
        fake_openai = FakeOpenAI(
            [
                StreamPlan(
                    events=[], final_response={}, exception_factory=lambda exc=first_error: exc
                ),
                StreamPlan(events=[], final_response={"id": "resp_ok", "output": []}),
            ]
        )
        gateway = ResponsesGateway(
            settings=settings,
            logger=logger,
            tool_schemas=[{"type": "function", "name": "get_weather"}],
            instructions="test-instructions",
            openai_client=cast(Any, fake_openai),
        )

        response = await gateway.stream_turn("weather", None, lambda _chunk: None)

        assert response.response_id == "resp_ok"
        assert len(fake_openai.responses.calls) == 2


@pytest.mark.asyncio
async def test_stream_turn_exhaustion_raises_transient_failure(settings, logger) -> None:
    """Exhausted transient retries should raise OpenAITransientFailure."""

    fake_openai = FakeOpenAI(
        [
            StreamPlan(events=[], final_response={}, exception_factory=_timeout_error),
            StreamPlan(events=[], final_response={}, exception_factory=_timeout_error),
            StreamPlan(events=[], final_response={}, exception_factory=_timeout_error),
        ]
    )
    gateway = ResponsesGateway(
        settings=settings,
        logger=logger,
        tool_schemas=[{"type": "function", "name": "get_weather"}],
        instructions="test-instructions",
        openai_client=cast(Any, fake_openai),
    )

    with pytest.raises(OpenAITransientFailure):
        await gateway.stream_turn("weather", None, lambda _chunk: None)

    assert len(fake_openai.responses.calls) == settings.openai_retry_attempts + 1


@pytest.mark.asyncio
async def test_stream_turn_retries_when_buffered_text_never_became_visible(
    settings, logger
) -> None:
    """Buffered text should not block retries until the stream classifies the output item."""

    fake_openai = FakeOpenAI(
        [
            StreamPlan(
                events=[{"type": "response.output_text.delta", "delta": "Hello "}],
                final_response={},
                iteration_exception_factory=_timeout_error,
            ),
            StreamPlan(events=[], final_response={"id": "resp_ok", "output": []}),
        ]
    )
    gateway = ResponsesGateway(
        settings=settings,
        logger=logger,
        tool_schemas=[{"type": "function", "name": "get_weather"}],
        instructions="test-instructions",
        openai_client=cast(Any, fake_openai),
    )

    chunks: list[str] = []
    response = await gateway.stream_turn("weather", None, chunks.append)

    assert response.response_id == "resp_ok"
    assert chunks == []
    assert len(fake_openai.responses.calls) == 2


@pytest.mark.asyncio
async def test_stream_turn_does_not_retry_after_visible_output(settings, logger) -> None:
    """Visible streamed output should still surface interruption instead of retrying."""

    fake_openai = FakeOpenAI(
        [
            StreamPlan(
                events=[
                    {"type": "response.output_item.added", "item": {"type": "message"}},
                    {"type": "response.output_text.delta", "delta": "Hello "},
                ],
                final_response={},
                iteration_exception_factory=_timeout_error,
            ),
            StreamPlan(events=[], final_response={"id": "resp_unused", "output": []}),
        ]
    )
    gateway = ResponsesGateway(
        settings=settings,
        logger=logger,
        tool_schemas=[{"type": "function", "name": "get_weather"}],
        instructions="test-instructions",
        openai_client=cast(Any, fake_openai),
    )

    chunks: list[str] = []
    with pytest.raises(OpenAIStreamInterrupted):
        await gateway.stream_turn("weather", None, chunks.append)

    assert chunks == ["Hello "]
    assert len(fake_openai.responses.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [401, 403])
async def test_stream_turn_maps_auth_failures(settings, logger, status_code: int) -> None:
    """Auth and permission failures should raise OpenAIAuthFailure."""

    fake_openai = FakeOpenAI(
        [
            StreamPlan(
                events=[], final_response={}, exception_factory=lambda: _status_error(status_code)
            )
        ]
    )
    gateway = ResponsesGateway(
        settings=settings,
        logger=logger,
        tool_schemas=[{"type": "function", "name": "get_weather"}],
        instructions="test-instructions",
        openai_client=cast(Any, fake_openai),
    )

    with pytest.raises(OpenAIAuthFailure):
        await gateway.stream_turn("weather", None, lambda _chunk: None)


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [400, 404])
async def test_stream_turn_maps_client_failures(settings, logger, status_code: int) -> None:
    """Non-retryable request failures should raise OpenAIClientFailure."""

    fake_openai = FakeOpenAI(
        [
            StreamPlan(
                events=[], final_response={}, exception_factory=lambda: _status_error(status_code)
            )
        ]
    )
    gateway = ResponsesGateway(
        settings=settings,
        logger=logger,
        tool_schemas=[{"type": "function", "name": "get_weather"}],
        instructions="test-instructions",
        openai_client=cast(Any, fake_openai),
    )

    with pytest.raises(OpenAIClientFailure):
        await gateway.stream_turn("weather", None, lambda _chunk: None)


@pytest.mark.asyncio
async def test_stream_turn_non_retryable_status_after_partial_output_is_interrupted(
    settings,
    logger,
) -> None:
    """Any failure after partial output should surface as an interrupted stream."""

    fake_openai = FakeOpenAI(
        [
            StreamPlan(
                events=[
                    {"type": "response.output_item.added", "item": {"type": "message"}},
                    {"type": "response.output_text.delta", "delta": "Hello "},
                ],
                final_response={},
                iteration_exception_factory=lambda: _status_error(401),
            )
        ]
    )
    gateway = ResponsesGateway(
        settings=settings,
        logger=logger,
        tool_schemas=[{"type": "function", "name": "get_weather"}],
        instructions="test-instructions",
        openai_client=cast(Any, fake_openai),
    )

    chunks: list[str] = []
    with pytest.raises(OpenAIStreamInterrupted):
        await gateway.stream_turn("weather", None, chunks.append)

    assert chunks == ["Hello "]


@pytest.mark.asyncio
async def test_aclose_closes_owned_client(
    settings,
    logger,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gateway should close only the AsyncOpenAI client it created internally."""

    fake_openai = FakeOpenAI([])
    monkeypatch.setattr(
        openai_gateway_module,
        "AsyncOpenAI",
        lambda **_kwargs: cast(Any, fake_openai),
    )

    gateway = ResponsesGateway(
        settings=settings,
        logger=logger,
        tool_schemas=[{"type": "function", "name": "get_weather"}],
        instructions="test-instructions",
    )

    await gateway.aclose()

    assert fake_openai.close_calls == 1


@pytest.mark.asyncio
async def test_aclose_does_not_close_injected_client(settings, logger) -> None:
    """Injected AsyncOpenAI clients should remain caller-owned."""

    fake_openai = FakeOpenAI([])
    gateway = ResponsesGateway(
        settings=settings,
        logger=logger,
        tool_schemas=[{"type": "function", "name": "get_weather"}],
        instructions="test-instructions",
        openai_client=cast(Any, fake_openai),
    )

    await gateway.aclose()

    assert fake_openai.close_calls == 0


@pytest.mark.asyncio
async def test_stream_turn_logs_turn_context_on_auth_failure(
    settings,
    logger,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Gateway failure logs should include turn context when a LoggerAdapter is passed."""

    caplog.set_level(logging.ERROR, logger="test")
    turn_logger = with_context(logger, {"run_id": "run-1", "turn_id": "turn-1"})
    fake_openai = FakeOpenAI(
        [StreamPlan(events=[], final_response={}, exception_factory=lambda: _status_error(401))]
    )
    gateway = ResponsesGateway(
        settings=settings,
        logger=logger,
        tool_schemas=[{"type": "function", "name": "get_weather"}],
        instructions="test-instructions",
        openai_client=cast(Any, fake_openai),
    )

    with pytest.raises(OpenAIAuthFailure):
        await gateway.stream_turn("weather", None, lambda _chunk: None, logger=turn_logger)

    record = next(
        record for record in caplog.records if record.getMessage() == "openai_auth_failure"
    )
    assert record.run_id == "run-1"
    assert record.turn_id == "turn-1"
    assert record.error_code == "HTTP_401"
