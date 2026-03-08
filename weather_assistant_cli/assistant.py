"""Turn orchestration for weather-assistant-cli."""

from __future__ import annotations

from typing import Protocol

from weather_assistant_cli.config import Settings
from weather_assistant_cli.logging_config import LoggerLike
from weather_assistant_cli.openai_gateway import (
    OpenAIAuthFailure,
    OpenAIClientFailure,
    OpenAIProtocolError,
    OpenAIStreamInterrupted,
    OpenAITransientFailure,
    ResponsesGateway,
    ResponsesInput,
    ResponsesInputItem,
)
from weather_assistant_cli.plugins import FollowUpMessageBuilder
from weather_assistant_cli.tools import ToolExecutor, ToolStartEvent


class AssistantSink(Protocol):
    """Terminal-facing callback surface used by the turn controller."""

    def on_text_chunk(self, chunk: str) -> None:
        """Render streamed assistant text incrementally."""

    def on_tool_start(self, event: ToolStartEvent) -> None:
        """Render tool start notifications for user transparency."""


class TurnController:
    """State machine for a single CLI assistant session."""

    def __init__(
        self,
        *,
        settings: Settings,
        responses_gateway: ResponsesGateway,
        tool_executor: ToolExecutor,
        build_follow_up_messages: FollowUpMessageBuilder | None = None,
    ) -> None:
        """Initialize turn orchestration dependencies."""

        self._settings = settings
        self._responses_gateway = responses_gateway
        self._tool_executor = tool_executor
        self._build_follow_up_messages = build_follow_up_messages
        self._previous_response_id: str | None = None

    def reset_context(self) -> None:
        """Reset conversation continuity for a fresh model context."""

        self._previous_response_id = None

    async def handle_turn(
        self,
        user_text: str,
        sink: AssistantSink,
        *,
        logger: LoggerLike | None = None,
    ) -> None:
        """Handle one user turn through model/tool/model round-tripping."""

        normalized_user_text = user_text.strip()
        if not normalized_user_text:
            raise ValueError("Input cannot be empty")
        if len(normalized_user_text) > self._settings.max_input_chars:
            raise ValueError(f"Input exceeds MAX_INPUT_CHARS ({self._settings.max_input_chars}).")

        pending_input: ResponsesInput = normalized_user_text
        previous_response_id = self._previous_response_id
        follow_up_messages_added = False

        for iteration in range(self._settings.tool_iteration_limit):
            round_streamed_text = False

            def on_round_text_chunk(chunk: str) -> None:
                nonlocal round_streamed_text
                round_streamed_text = True
                sink.on_text_chunk(chunk)

            try:
                model_response = await self._responses_gateway.stream_turn(
                    pending_input,
                    previous_response_id,
                    on_round_text_chunk,
                    tool_choice="auto",
                    logger=logger,
                )
            except OpenAIStreamInterrupted:
                sink.on_text_chunk("\n\nSorry, the response was interrupted. Please try again.")
                return
            except OpenAITransientFailure:
                sink.on_text_chunk(
                    "I'm having trouble reaching the language model right now. "
                    "Please try again in a moment."
                )
                return
            except OpenAIAuthFailure:
                sink.on_text_chunk(
                    "I couldn't reach OpenAI with the current API configuration. "
                    "Check OPENAI_API_KEY and model access, then try again."
                )
                return
            except OpenAIClientFailure:
                sink.on_text_chunk(
                    "The OpenAI request was rejected. "
                    "Check the configured model and request settings, then try again."
                )
                return
            except OpenAIProtocolError:
                sink.on_text_chunk(
                    "I ran into an unexpected language-model response issue. Please try again."
                )
                return

            if not model_response.tool_calls:
                if not round_streamed_text and model_response.output_text:
                    sink.on_text_chunk(model_response.output_text)
                self._previous_response_id = model_response.response_id
                return

            outputs = await self._tool_executor.execute(
                model_response.tool_calls,
                on_tool_start=sink.on_tool_start,
                logger=logger,
            )

            roundtrip_input: list[ResponsesInputItem] = [
                output.as_response_item() for output in outputs
            ]

            if not follow_up_messages_added and self._build_follow_up_messages is not None:
                follow_up_messages = self._build_follow_up_messages(outputs)
                roundtrip_input.extend(follow_up_messages)
                follow_up_messages_added = bool(follow_up_messages)

            pending_input = roundtrip_input
            previous_response_id = model_response.response_id

            if iteration == self._settings.tool_iteration_limit - 1:
                self._previous_response_id = previous_response_id
                sink.on_text_chunk(
                    "I couldn't complete that request after multiple tool-call rounds. "
                    "Please try a simpler phrasing."
                )
                return
