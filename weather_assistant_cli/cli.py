"""Interactive terminal interface and application composition root."""

from __future__ import annotations

import asyncio
import logging
import sys
import uuid
from collections.abc import Sequence
from typing import Any, cast

import httpx
from pydantic import ValidationError

from weather_assistant_cli.assistant import TurnController
from weather_assistant_cli.config import Settings, load_settings
from weather_assistant_cli.logging_config import configure_logging, with_context
from weather_assistant_cli.openai_gateway import ResponsesGateway, ResponsesInputItem
from weather_assistant_cli.plugins import (
    FollowUpMessageBuilder,
    PluginLoadError,
    ToolBundle,
    load_tool_bundles,
)
from weather_assistant_cli.tools import (
    FunctionCallOutputItem,
    ToolExecutor,
    ToolHandler,
    ToolStartEvent,
)

TOOL_STREAMING_INSTRUCTIONS = """
If you intend to call a tool, do not emit user-facing assistant text before or alongside the tool call.
Only emit visible assistant text once you are ready to answer without any more tool calls.
If you need another tool after reviewing earlier tool outputs, call it directly without a prefatory sentence.
""".strip()


class TerminalSink:
    """CLI sink implementation used by the turn controller."""

    def __init__(self) -> None:
        """Initialize sink state for one user turn."""

        self._printed_prefix: bool = False
        self._printed_tool_line: bool = False

    def on_tool_start(self, event: ToolStartEvent) -> None:
        """Render a tool invocation progress line."""

        if not self._printed_tool_line:
            print()
            self._printed_tool_line = True
        if event.preview_text is None:
            print(f"[Calling {event.tool_name}...]")
            return
        print(f"[Calling {event.tool_name} for {event.preview_text}...]")

    def on_text_chunk(self, chunk: str) -> None:
        """Render streaming text as it arrives from the model."""

        if not self._printed_prefix:
            print()
            print("Assistant: ", end="", flush=True)
            self._printed_prefix = True
        print(chunk, end="", flush=True)

    def finish_turn(self) -> None:
        """Finalize turn output by writing trailing newline when needed."""

        if self._printed_prefix:
            print()


def is_exit_command(user_text: str) -> bool:
    """Return True when the user requests program termination."""

    return user_text.strip().lower() in {"quit", "exit"}


def is_reset_command(user_text: str) -> bool:
    """Return True when the user requests conversation reset."""

    return user_text.strip().lower() == "reset"


def build_follow_up_messages(
    outputs: Sequence[FunctionCallOutputItem],
    builders: Sequence[tuple[str, FollowUpMessageBuilder]],
) -> list[ResponsesInputItem]:
    """Build aggregated follow-up guidance from tool-local outputs."""

    messages: list[ResponsesInputItem] = []
    for tool_name, builder in builders:
        matching_outputs = [output for output in outputs if output.tool_name == tool_name]
        if not matching_outputs:
            continue
        messages.extend(builder(matching_outputs))
    return messages


def build_tool_handlers(tool_bundles: Sequence[ToolBundle]) -> dict[str, ToolHandler]:
    """Build handler lookup from the registered tool bundles."""

    return {bundle.handler.name: bundle.handler for bundle in tool_bundles}


def compose_system_instructions(tool_bundles: Sequence[ToolBundle]) -> str:
    """Combine shared and tool-specific model instructions."""

    parts = [TOOL_STREAMING_INSTRUCTIONS]
    parts.extend(
        bundle.instructions.strip() for bundle in tool_bundles if bundle.instructions.strip()
    )
    return "\n\n".join(parts)


def compose_follow_up_builder(
    tool_bundles: Sequence[ToolBundle],
) -> FollowUpMessageBuilder | None:
    """Combine tool-specific follow-up builders into one callback."""

    builders = [
        (bundle.handler.name, bundle.build_follow_up_messages)
        for bundle in tool_bundles
        if bundle.build_follow_up_messages is not None
    ]
    if not builders:
        return None

    def build(outputs: Sequence[FunctionCallOutputItem]) -> list[ResponsesInputItem]:
        return build_follow_up_messages(outputs, builders)

    return build


def format_settings_error(exc: ValidationError) -> str:
    """Build a concise CLI-facing configuration error message."""

    details: list[str] = []
    raw_errors = cast(list[dict[str, Any]], exc.errors())
    for error in raw_errors:
        location = ".".join(str(part) for part in error.get("loc", ()))
        message = str(error["msg"])
        if location:
            details.append(f"{location}: {message}")
        else:
            details.append(message)

    lines = ["Configuration error. Check your environment variables or .env file:"]
    lines.extend(f"- {detail}" for detail in details)
    return "\n".join(lines)


async def run_cli(settings: Settings) -> int:
    """Run the interactive assistant loop until exit."""

    configure_logging(settings.log_format, settings.log_level)
    base_logger = logging.getLogger("weather_assistant_cli")
    run_id = str(uuid.uuid4())

    http_timeout = httpx.Timeout(
        connect=settings.http_timeout_seconds,
        read=settings.http_timeout_seconds,
        write=settings.http_timeout_seconds,
        pool=settings.http_timeout_seconds,
    )

    async with httpx.AsyncClient(timeout=http_timeout) as http_client:
        try:
            tool_bundles = load_tool_bundles(
                settings=settings,
                http_client=http_client,
                logger=base_logger,
            )
        except PluginLoadError as exc:
            print(f"Plugin load error: {exc}", file=sys.stderr)
            return 1
        handlers = build_tool_handlers(tool_bundles)
        tool_executor = ToolExecutor(
            handlers=handlers,
            max_concurrency=settings.max_concurrency,
            logger=base_logger,
        )
        responses_gateway = ResponsesGateway(
            settings=settings,
            logger=base_logger,
            tool_schemas=[handler.schema for handler in handlers.values()],
            instructions=compose_system_instructions(tool_bundles),
        )
        turn_controller = TurnController(
            settings=settings,
            responses_gateway=responses_gateway,
            tool_executor=tool_executor,
            build_follow_up_messages=compose_follow_up_builder(tool_bundles),
        )
        try:
            print("Weather Assistant - ask about the weather in any city.")
            print("Type 'quit' to exit.")

            while True:
                print()
                try:
                    user_text = await asyncio.to_thread(input, "You: ")
                except (EOFError, KeyboardInterrupt):
                    print("\nGoodbye!")
                    return 0

                if is_exit_command(user_text):
                    print("Goodbye!")
                    return 0

                if is_reset_command(user_text):
                    turn_controller.reset_context()
                    print("Conversation reset.")
                    continue

                if not user_text.strip():
                    continue

                turn_logger = with_context(
                    base_logger,
                    {
                        "run_id": run_id,
                        "turn_id": str(uuid.uuid4()),
                    },
                )
                turn_logger.debug("turn_received")

                sink = TerminalSink()
                try:
                    await turn_controller.handle_turn(user_text, sink, logger=turn_logger)
                except ValueError as exc:
                    sink.finish_turn()
                    print(f"Assistant: {exc}")
                except Exception:
                    sink.finish_turn()
                    turn_logger.exception("turn_failed", extra={"error_code": "UNEXPECTED_ERROR"})
                    print("Assistant: Something unexpected went wrong. Please try again.")
                else:
                    sink.finish_turn()
        finally:
            await responses_gateway.aclose()


async def async_main() -> int:
    """Load settings and run the async CLI application."""

    try:
        settings = load_settings()
    except ValidationError as exc:
        print(format_settings_error(exc), file=sys.stderr)
        return 1
    return await run_cli(settings)


def main() -> int:
    """Run the CLI entrypoint inside asyncio event loop management."""

    return asyncio.run(async_main())
