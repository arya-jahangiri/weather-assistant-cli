"""Tests for concurrent tool execution behavior."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field

import pytest

from weather_assistant_cli.logging_config import with_context
from weather_assistant_cli.tools import (
    ToolCall,
    ToolExecutionErrorCode,
    ToolExecutor,
    ToolOutput,
    ToolStartEvent,
)


@dataclass
class SlowHandler:
    """Tool handler fake tracking max concurrent executions."""

    active_count: int = 0
    max_seen: int = 0
    seen_queries: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        """Return fake tool name."""

        return "slow_tool"

    @property
    def schema(self) -> dict[str, object]:
        """Return minimal tool schema for gateway exposure."""

        return {"type": "function", "name": self.name}

    def preview_invocation(self, arguments_json: str) -> str | None:
        """Return query from JSON args for tool-start callback assertions."""

        payload = json.loads(arguments_json)
        query = payload.get("query")
        return str(query) if isinstance(query, str) else None

    async def run(self, arguments_json: str, *, logger=None) -> ToolOutput:
        """Sleep briefly while tracking concurrent execution depth."""

        del logger
        payload = json.loads(arguments_json)
        query = payload.get("query", "")
        self.seen_queries.append(str(query))

        self.active_count += 1
        self.max_seen = max(self.max_seen, self.active_count)
        await asyncio.sleep(0.01)
        self.active_count -= 1

        return ToolOutput(payload_json='{"ok": true}', error_code=None)


@dataclass
class CrashingHandler:
    """Tool handler fake that raises during execution."""

    @property
    def name(self) -> str:
        """Return fake crashing tool name."""

        return "crash_tool"

    @property
    def schema(self) -> dict[str, object]:
        """Return minimal tool schema for gateway exposure."""

        return {"type": "function", "name": self.name}

    def preview_invocation(self, arguments_json: str) -> str | None:
        """Return stable preview text for progress callbacks."""

        del arguments_json
        return "boom"

    async def run(self, arguments_json: str, *, logger=None) -> ToolOutput:
        """Raise a runtime error to exercise executor recovery."""

        del arguments_json, logger
        await asyncio.sleep(0.001)
        raise RuntimeError("boom")


@pytest.mark.asyncio
async def test_executor_limits_parallelism_and_emits_tool_start_callbacks(logger) -> None:
    """Executor should enforce max_concurrency and preserve tool-start callbacks."""

    handler = SlowHandler()
    executor = ToolExecutor(
        handlers={handler.name: handler},
        max_concurrency=2,
        logger=logger,
    )

    tool_calls = [
        ToolCall(
            call_id=f"call_{index}",
            name="slow_tool",
            arguments_json=json.dumps({"query": f"city-{index}"}),
        )
        for index in range(6)
    ]

    starts: list[ToolStartEvent] = []
    outputs = await executor.execute(tool_calls, on_tool_start=starts.append)

    assert len(outputs) == 6
    assert handler.max_seen <= 2
    assert all(output.tool_name == "slow_tool" for output in outputs)
    assert starts == [
        ToolStartEvent(tool_name="slow_tool", preview_text=f"city-{index}") for index in range(6)
    ]


@pytest.mark.asyncio
async def test_executor_preserves_output_order_with_unknown_tool(logger) -> None:
    """Outputs should stay in model order even when unknown tools are mixed in."""

    handler = SlowHandler()
    executor = ToolExecutor(
        handlers={handler.name: handler},
        max_concurrency=2,
        logger=logger,
    )

    outputs = await executor.execute(
        [
            ToolCall(call_id="call_1", name="slow_tool", arguments_json='{"query": "Rome"}'),
            ToolCall(call_id="call_2", name="missing_tool", arguments_json="{}"),
            ToolCall(call_id="call_3", name="slow_tool", arguments_json='{"query": "Paris"}'),
        ],
        on_tool_start=None,
    )

    assert [output.call_id for output in outputs] == ["call_1", "call_2", "call_3"]
    assert [output.tool_name for output in outputs] == ["slow_tool", "missing_tool", "slow_tool"]


@pytest.mark.asyncio
async def test_executor_unknown_tool_maps_generic_error_payload(logger) -> None:
    """Unknown tools should return generic UNKNOWN_TOOL payloads."""

    executor = ToolExecutor(handlers={}, max_concurrency=1, logger=logger)
    outputs = await executor.execute(
        [ToolCall(call_id="bad", name="not_registered", arguments_json="{}")],
        on_tool_start=None,
    )

    payload = json.loads(outputs[0].output_json)
    assert outputs[0].tool_name == "not_registered"
    assert outputs[0].error_code == ToolExecutionErrorCode.UNKNOWN_TOOL.value
    assert payload["tool_name"] == "not_registered"
    assert payload["error_code"] == ToolExecutionErrorCode.UNKNOWN_TOOL.value


@pytest.mark.asyncio
async def test_executor_serializes_runtime_error_and_preserves_sibling_results(logger) -> None:
    """Unexpected handler crashes should become TOOL_RUNTIME_ERROR outputs."""

    slow_handler = SlowHandler()
    crash_handler = CrashingHandler()
    executor = ToolExecutor(
        handlers={slow_handler.name: slow_handler, crash_handler.name: crash_handler},
        max_concurrency=3,
        logger=logger,
    )

    outputs = await executor.execute(
        [
            ToolCall(call_id="call_1", name="slow_tool", arguments_json='{"query": "Rome"}'),
            ToolCall(call_id="call_2", name="crash_tool", arguments_json="{}"),
            ToolCall(call_id="call_3", name="slow_tool", arguments_json='{"query": "Paris"}'),
        ],
        on_tool_start=None,
    )

    assert [output.call_id for output in outputs] == ["call_1", "call_2", "call_3"]
    assert [output.tool_name for output in outputs] == ["slow_tool", "crash_tool", "slow_tool"]
    assert outputs[0].error_code is None
    assert outputs[2].error_code is None

    payload = json.loads(outputs[1].output_json)
    assert outputs[1].error_code == ToolExecutionErrorCode.TOOL_RUNTIME_ERROR.value
    assert payload["tool_name"] == "crash_tool"
    assert payload["error_code"] == ToolExecutionErrorCode.TOOL_RUNTIME_ERROR.value
    assert payload["error_message"] == "Tool execution failed unexpectedly."


@pytest.mark.asyncio
async def test_executor_logs_with_turn_context(
    logger,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Executor logs should inherit run and turn ids from the passed LoggerAdapter."""

    caplog.set_level(logging.DEBUG, logger="test")
    turn_logger = with_context(logger, {"run_id": "run-1", "turn_id": "turn-1"})
    handler = SlowHandler()
    executor = ToolExecutor(
        handlers={handler.name: handler},
        max_concurrency=1,
        logger=logger,
    )

    await executor.execute(
        [ToolCall(call_id="call_1", name="slow_tool", arguments_json='{"query": "Rome"}')],
        on_tool_start=None,
        logger=turn_logger,
    )

    record = next(
        record for record in caplog.records if record.getMessage() == "tool_call_finished"
    )
    assert record.run_id == "run-1"
    assert record.turn_id == "turn-1"


@pytest.mark.asyncio
async def test_executor_logs_runtime_error_with_turn_context(
    logger,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Crash logs should inherit run and turn ids from the passed LoggerAdapter."""

    caplog.set_level(logging.ERROR, logger="test")
    turn_logger = with_context(logger, {"run_id": "run-1", "turn_id": "turn-1"})
    crash_handler = CrashingHandler()
    executor = ToolExecutor(
        handlers={crash_handler.name: crash_handler},
        max_concurrency=1,
        logger=logger,
    )

    await executor.execute(
        [ToolCall(call_id="call_1", name="crash_tool", arguments_json="{}")],
        on_tool_start=None,
        logger=turn_logger,
    )

    record = next(record for record in caplog.records if record.getMessage() == "tool_call_crashed")
    assert record.run_id == "run-1"
    assert record.turn_id == "turn-1"
    assert record.error_code == ToolExecutionErrorCode.TOOL_RUNTIME_ERROR.value
