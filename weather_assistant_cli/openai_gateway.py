"""OpenAI Responses streaming gateway with retry logic and typed decoding."""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias, TypedDict, cast

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI

from weather_assistant_cli.config import Settings
from weather_assistant_cli.logging_config import LoggerLike
from weather_assistant_cli.retry import BackoffPolicy, backoff_delay_seconds
from weather_assistant_cli.tools import FunctionCallOutputInputItem, ToolCall


class SystemMessageItem(TypedDict):
    """Typed Responses API system message input item."""

    role: Literal["system"]
    content: str


ResponsesInputItem: TypeAlias = FunctionCallOutputInputItem | SystemMessageItem
ResponsesInput: TypeAlias = str | list[ResponsesInputItem]
ToolChoiceValue: TypeAlias = Literal["none", "auto", "required"] | dict[str, object]


@dataclass(frozen=True, slots=True)
class ModelResponse:
    """Normalized response payload consumed by the turn controller."""

    response_id: str
    tool_calls: list[ToolCall]
    output_text: str


class OpenAITransientFailure(RuntimeError):
    """Raised when OpenAI transient failures exceed retry budget."""


class OpenAIStreamInterrupted(RuntimeError):
    """Raised when an OpenAI failure happens after partial output."""


class OpenAIProtocolError(RuntimeError):
    """Raised when the Responses API returns a malformed final payload."""


class OpenAIAuthFailure(RuntimeError):
    """Raised when OpenAI rejects the request due to auth or access problems."""


class OpenAIClientFailure(RuntimeError):
    """Raised when OpenAI rejects the request due to client-side request issues."""


class ResponsesGateway:
    """Gateway for model streaming and function-call extraction."""

    def __init__(
        self,
        *,
        settings: Settings,
        logger: LoggerLike,
        tool_schemas: list[dict[str, object]],
        instructions: str,
        openai_client: AsyncOpenAI | None = None,
    ) -> None:
        """Create a gateway configured for a single model and tool set."""

        self._model = settings.openai_model
        self._instructions = instructions
        self._tool_schemas = list(tool_schemas)
        self._logger = logger
        self._retry_policy = BackoffPolicy(
            attempts=settings.openai_retry_attempts,
            base_delay_seconds=settings.openai_retry_base_delay_seconds,
            with_jitter=True,
        )
        self._owns_openai_client = openai_client is None
        self._openai = openai_client or AsyncOpenAI(
            api_key=settings.openai_api_key.get_secret_value(),
            timeout=settings.openai_timeout_seconds,
        )

    async def stream_turn(
        self,
        input_payload: ResponsesInput,
        previous_response_id: str | None,
        on_text_chunk: Callable[[str], None],
        *,
        tool_choice: ToolChoiceValue | None = None,
        logger: LoggerLike | None = None,
    ) -> ModelResponse:
        """Stream one model turn and return normalized final response metadata."""

        attempt = 0
        active_logger = logger or self._logger
        while True:
            emitted_text = False

            def tracked_on_text_chunk(chunk: str) -> None:
                nonlocal emitted_text
                emitted_text = True
                on_text_chunk(chunk)

            try:
                return await self._stream_once(
                    input_payload,
                    previous_response_id,
                    tracked_on_text_chunk,
                    tool_choice=tool_choice,
                )
            except OpenAIProtocolError:
                active_logger.error(
                    "openai_protocol_error",
                    extra={"error_code": "PROTOCOL_ERROR", "status": "failure"},
                )
                raise
            except (
                APIConnectionError,
                APITimeoutError,
                httpx.TimeoutException,
                httpx.ConnectError,
            ) as exc:
                if emitted_text:
                    active_logger.error(
                        "openai_stream_interrupted",
                        extra={"error_code": exc.__class__.__name__, "status": "failure"},
                    )
                    raise OpenAIStreamInterrupted from exc
                if self._should_retry(attempt):
                    await self._sleep_before_retry(attempt)
                    attempt += 1
                    continue
                active_logger.error(
                    "openai_transient_failure_exhausted",
                    extra={"error_code": exc.__class__.__name__, "status": "failure"},
                )
                raise OpenAITransientFailure from exc
            except APIStatusError as exc:
                if emitted_text:
                    active_logger.error(
                        "openai_stream_interrupted",
                        extra={"error_code": f"HTTP_{exc.status_code}", "status": "failure"},
                    )
                    raise OpenAIStreamInterrupted from exc
                if exc.status_code == 429 or exc.status_code >= 500:
                    if self._should_retry(attempt):
                        await self._sleep_before_retry(attempt)
                        attempt += 1
                        continue
                    active_logger.error(
                        "openai_retryable_status_exhausted",
                        extra={"error_code": f"HTTP_{exc.status_code}", "status": "failure"},
                    )
                    raise OpenAITransientFailure from exc
                if exc.status_code in {401, 403}:
                    active_logger.error(
                        "openai_auth_failure",
                        extra={"error_code": f"HTTP_{exc.status_code}", "status": "failure"},
                    )
                    raise OpenAIAuthFailure from exc
                active_logger.error(
                    "openai_client_failure",
                    extra={"error_code": f"HTTP_{exc.status_code}", "status": "failure"},
                )
                raise OpenAIClientFailure from exc

    async def aclose(self) -> None:
        """Close the owned OpenAI client, if the caller did not inject one."""

        if not self._owns_openai_client:
            return

        close_method = getattr(self._openai, "close", None)
        if close_method is None:
            close_method = getattr(self._openai, "aclose", None)
        if close_method is None:
            return

        result = close_method()
        if inspect.isawaitable(result):
            await cast(Awaitable[object], result)

    async def _stream_once(
        self,
        input_payload: ResponsesInput,
        previous_response_id: str | None,
        on_text_chunk: Callable[[str], None],
        *,
        tool_choice: ToolChoiceValue | None,
    ) -> ModelResponse:
        """Execute one non-retried Responses API streaming call."""

        responses_api = cast(Any, self._openai.responses)
        request_kwargs = self._build_request_kwargs(
            input_payload,
            previous_response_id,
            tool_choice=tool_choice,
        )

        async with responses_api.stream(**request_kwargs) as stream:
            buffered_chunks: list[str] = []
            emit_text: bool | None = None
            typed_stream = cast(AsyncIterator[object], stream)
            async for event in typed_stream:
                output_item_type = self._extract_output_item_type(event)
                if output_item_type is not None and emit_text is None:
                    emit_text = output_item_type != "function_call"
                    if emit_text:
                        for chunk in buffered_chunks:
                            on_text_chunk(chunk)
                    buffered_chunks.clear()
                delta = self._extract_text_delta(event)
                if not delta:
                    continue
                if emit_text is True:
                    on_text_chunk(delta)
                    continue
                if emit_text is None:
                    buffered_chunks.append(delta)
            final_response = await stream.get_final_response()

        model_response = self._parse_model_response(final_response)
        if emit_text is None and buffered_chunks and not model_response.tool_calls:
            for chunk in buffered_chunks:
                on_text_chunk(chunk)
        return model_response

    def _build_request_kwargs(
        self,
        input_payload: ResponsesInput,
        previous_response_id: str | None,
        *,
        tool_choice: ToolChoiceValue | None,
    ) -> dict[str, object]:
        """Build a shared Responses API request payload."""

        request_kwargs: dict[str, object] = {
            "model": self._model,
            "instructions": self._instructions,
            "tools": self._tool_schemas,
            "input": input_payload,
            "parallel_tool_calls": True,
        }
        if previous_response_id is not None:
            request_kwargs["previous_response_id"] = previous_response_id
        if tool_choice is not None:
            request_kwargs["tool_choice"] = tool_choice
        return request_kwargs

    def _parse_model_response(self, response: object) -> ModelResponse:
        """Extract tool calls and response id from a final Responses payload."""

        raw_response_id = self._read_value(response, "id")
        response_id = self._require_non_empty_str(raw_response_id, "response.id")
        output_items = self._read_value(response, "output")
        if not isinstance(output_items, list):
            raise OpenAIProtocolError("Responses API returned a non-list response.output payload.")

        tool_calls: list[ToolCall] = []
        output_list = cast(list[object], output_items)
        for item in output_list:
            item_type = self._read_value(item, "type")
            if not isinstance(item_type, str) or not item_type:
                raise OpenAIProtocolError(
                    "Responses API returned an output item without a valid type."
                )
            if item_type != "function_call":
                continue

            raw_call_id = self._read_value(item, "call_id")
            raw_name = self._read_value(item, "name")
            raw_arguments = self._read_value(item, "arguments")

            call_id = self._require_non_empty_str(raw_call_id, "function_call.call_id")
            name = self._require_non_empty_str(raw_name, "function_call.name")
            arguments_json = self._serialize_arguments(raw_arguments)
            tool_calls.append(
                ToolCall(
                    call_id=call_id,
                    name=name,
                    arguments_json=arguments_json,
                )
            )

        return ModelResponse(
            response_id=response_id,
            tool_calls=tool_calls,
            output_text=self._extract_output_text(response, output_list),
        )

    def _serialize_arguments(self, arguments: object) -> str:
        """Serialize tool-call arguments into canonical JSON text."""

        if isinstance(arguments, str):
            return arguments
        if arguments is None:
            raise OpenAIProtocolError("Responses API returned function_call arguments=None.")
        try:
            return json.dumps(arguments)
        except TypeError as exc:
            raise OpenAIProtocolError(
                "Responses API returned non-serializable function_call arguments."
            ) from exc

    def _extract_text_delta(self, event: object) -> str:
        """Extract incremental output text from a stream event."""

        event_type = self._read_value(event, "type")
        if event_type not in {"response.output_text.delta", "response.refusal.delta"}:
            return ""
        delta = self._read_value(event, "delta")
        return delta if isinstance(delta, str) else ""

    def _extract_output_item_type(self, event: object) -> str | None:
        """Extract the first streamed output-item type when available."""

        event_type = self._read_value(event, "type")
        if event_type != "response.output_item.added":
            return None
        item = self._read_value(event, "item")
        item_type = self._read_value(item, "type")
        return item_type if isinstance(item_type, str) and item_type else None

    def _extract_output_text(self, response: object, output_list: list[object]) -> str:
        """Aggregate assistant output text from the final response payload."""

        output_text = self._read_value(response, "output_text")
        if isinstance(output_text, str) and output_text:
            return output_text

        text_chunks: list[str] = []
        for item in output_list:
            if self._read_value(item, "type") != "message":
                continue
            content_items = self._read_value(item, "content")
            if not isinstance(content_items, list):
                continue
            for content in cast(list[object], content_items):
                content_type = self._read_value(content, "type")
                if content_type == "output_text":
                    text = self._read_value(content, "text")
                elif content_type == "refusal":
                    text = self._read_value(content, "refusal")
                else:
                    continue
                if isinstance(text, str):
                    text_chunks.append(text)

        return "".join(text_chunks)

    def _should_retry(self, attempt: int) -> bool:
        """Return True when another retry attempt is allowed."""

        return attempt < self._retry_policy.attempts

    async def _sleep_before_retry(self, attempt: int) -> None:
        """Sleep for one backoff interval before retrying."""

        delay_seconds = backoff_delay_seconds(self._retry_policy, attempt)
        await asyncio.sleep(delay_seconds)

    @staticmethod
    def _read_value(source: object, key: str) -> object | None:
        """Read a field from either mapping-like or attribute-like objects."""

        if isinstance(source, Mapping):
            mapping_source = cast(Mapping[str, object], source)
            return mapping_source.get(key)
        return getattr(source, key, None)

    @staticmethod
    def _require_non_empty_str(value: object, field_name: str) -> str:
        """Validate a protocol field as a non-empty string."""

        if not isinstance(value, str):
            raise OpenAIProtocolError(f"Responses API returned no valid {field_name}.")
        if not value.strip():
            raise OpenAIProtocolError(f"Responses API returned no valid {field_name}.")
        return value
