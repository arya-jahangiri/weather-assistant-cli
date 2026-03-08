"""Generic tool contracts and execution helpers."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, Protocol, TypedDict

from pydantic import BaseModel, ConfigDict

from weather_assistant_cli.logging_config import LoggerLike


@dataclass(frozen=True, slots=True)
class ToolStartEvent:
    """User-facing metadata describing a tool invocation."""

    tool_name: str
    preview_text: str | None = None


class ToolExecutionErrorCode(StrEnum):
    """Stable error codes for generic tool execution failures."""

    BAD_TOOL_ARGS = "BAD_TOOL_ARGS"
    TOOL_RUNTIME_ERROR = "TOOL_RUNTIME_ERROR"
    UNKNOWN_TOOL = "UNKNOWN_TOOL"


class FunctionCallOutputInputItem(TypedDict):
    """Typed Responses API function_call_output input item."""

    type: Literal["function_call_output"]
    call_id: str
    output: str


@dataclass(frozen=True, slots=True)
class ToolCall:
    """Normalized function call emitted by the LLM."""

    call_id: str
    name: str
    arguments_json: str


@dataclass(frozen=True, slots=True)
class ToolOutput:
    """Result returned by a tool handler."""

    payload_json: str
    error_code: str | None


class ToolFailurePayload(BaseModel):
    """Serialized payload for generic tool execution failures."""

    model_config = ConfigDict(extra="forbid")

    ok: Literal[False] = False
    tool_name: str
    error_code: ToolExecutionErrorCode
    error_message: str


@dataclass(frozen=True, slots=True)
class FunctionCallOutputItem:
    """Responses API function_call_output item with internal tool metadata."""

    tool_name: str
    call_id: str
    output_json: str
    error_code: str | None

    def as_response_item(self) -> FunctionCallOutputInputItem:
        """Convert to the Responses API input-item shape."""

        return {
            "type": "function_call_output",
            "call_id": self.call_id,
            "output": self.output_json,
        }


class ToolHandler(Protocol):
    """Interface for tool execution adapters."""

    @property
    def name(self) -> str:
        """Return the tool/function name exposed to the LLM."""

        ...

    @property
    def schema(self) -> dict[str, object]:
        """Return the OpenAI tool schema payload."""

        ...

    def preview_invocation(self, arguments_json: str) -> str | None:
        """Return user-facing tool-call context when argument parsing succeeds."""

        ...

    async def run(self, arguments_json: str, *, logger: LoggerLike | None = None) -> ToolOutput:
        """Execute tool logic and return serialized tool payload."""

        ...


class ToolExecutor:
    """Concurrent tool execution engine with bounded parallelism."""

    def __init__(
        self,
        *,
        handlers: Mapping[str, ToolHandler],
        max_concurrency: int,
        logger: LoggerLike,
    ) -> None:
        """Initialize handler lookup, concurrency controls, and logging."""

        self._handlers = dict(handlers)
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._logger = logger

    async def execute(
        self,
        tool_calls: list[ToolCall],
        *,
        on_tool_start: Callable[[ToolStartEvent], None] | None,
        logger: LoggerLike | None = None,
    ) -> list[FunctionCallOutputItem]:
        """Execute tool calls and return Responses API function_call_output items."""

        active_logger = logger or self._logger
        ordered_results: list[FunctionCallOutputItem | asyncio.Task[FunctionCallOutputItem]] = []
        async_tasks: list[asyncio.Task[FunctionCallOutputItem]] = []

        for call in tool_calls:
            handler = self._handlers.get(call.name)
            if handler is None:
                payload = ToolFailurePayload(
                    tool_name=call.name,
                    error_code=ToolExecutionErrorCode.UNKNOWN_TOOL,
                    error_message=f"Unsupported tool name: {call.name}",
                )
                ordered_results.append(
                    FunctionCallOutputItem(
                        tool_name=call.name,
                        call_id=call.call_id,
                        output_json=payload.model_dump_json(),
                        error_code=ToolExecutionErrorCode.UNKNOWN_TOOL.value,
                    )
                )
                continue

            start_event = ToolStartEvent(
                tool_name=call.name,
                preview_text=handler.preview_invocation(call.arguments_json),
            )
            if on_tool_start:
                on_tool_start(start_event)

            task = asyncio.create_task(
                self._execute_single(call.call_id, call.arguments_json, handler, active_logger)
            )
            async_tasks.append(task)
            ordered_results.append(task)

        if async_tasks:
            await asyncio.gather(*async_tasks)

        return [
            entry if isinstance(entry, FunctionCallOutputItem) else entry.result()
            for entry in ordered_results
        ]

    async def _execute_single(
        self,
        call_id: str,
        arguments_json: str,
        handler: ToolHandler,
        logger: LoggerLike,
    ) -> FunctionCallOutputItem:
        """Execute one tool call under semaphore and emit latency logs."""

        start_time = time.perf_counter()
        async with self._semaphore:
            try:
                output = await handler.run(arguments_json, logger=logger)
            except Exception:
                elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
                logger.exception(
                    "tool_call_crashed",
                    extra={
                        "status": "failure",
                        "error_code": ToolExecutionErrorCode.TOOL_RUNTIME_ERROR.value,
                        "latency_ms": elapsed_ms,
                    },
                )
                payload = ToolFailurePayload(
                    tool_name=handler.name,
                    error_code=ToolExecutionErrorCode.TOOL_RUNTIME_ERROR,
                    error_message="Tool execution failed unexpectedly.",
                )
                return FunctionCallOutputItem(
                    tool_name=handler.name,
                    call_id=call_id,
                    output_json=payload.model_dump_json(),
                    error_code=ToolExecutionErrorCode.TOOL_RUNTIME_ERROR.value,
                )

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        logger.debug(
            "tool_call_finished",
            extra={
                "status": "success" if output.error_code is None else "failure",
                "error_code": output.error_code,
                "latency_ms": elapsed_ms,
            },
        )
        return FunctionCallOutputItem(
            tool_name=handler.name,
            call_id=call_id,
            output_json=output.payload_json,
            error_code=output.error_code,
        )
