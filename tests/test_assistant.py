"""Tests for TurnController orchestration and follow-up guidance."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from weather_assistant_cli.assistant import TurnController
from weather_assistant_cli.logging_config import with_context
from weather_assistant_cli.openai_gateway import (
    ModelResponse,
    OpenAIAuthFailure,
    OpenAIClientFailure,
    OpenAIProtocolError,
    OpenAIStreamInterrupted,
    OpenAITransientFailure,
    ResponsesInputItem,
)
from weather_assistant_cli.tools import FunctionCallOutputItem, ToolCall, ToolStartEvent


@dataclass
class StreamStep:
    """One scripted LLM step for turn-controller tests."""

    response: ModelResponse
    chunks: list[str]


class FakeResponsesGateway:
    """Scripted fake for ResponsesGateway."""

    def __init__(
        self,
        steps: list[StreamStep],
        *,
        fail_with_transient: bool = False,
        fail_with_protocol: bool = False,
        fail_with_auth: bool = False,
        fail_with_client: bool = False,
        interrupt_with_chunks: list[str] | None = None,
    ) -> None:
        """Initialize fake with scripted responses or failure modes."""

        self._steps = steps
        self._fail_with_transient = fail_with_transient
        self._fail_with_protocol = fail_with_protocol
        self._fail_with_auth = fail_with_auth
        self._fail_with_client = fail_with_client
        self._interrupt_with_chunks = interrupt_with_chunks
        self.calls: list[tuple[object, str | None, object | None, object | None]] = []

    async def stream_turn(
        self,
        input_payload,
        previous_response_id,
        on_text_chunk,
        *,
        tool_choice=None,
        logger=None,
    ):
        """Return scripted step or raise configured failure."""

        self.calls.append((input_payload, previous_response_id, logger, tool_choice))
        if self._fail_with_transient:
            raise OpenAITransientFailure()
        if self._fail_with_protocol:
            raise OpenAIProtocolError("malformed final response")
        if self._fail_with_auth:
            raise OpenAIAuthFailure()
        if self._fail_with_client:
            raise OpenAIClientFailure()
        if self._interrupt_with_chunks is not None:
            for chunk in self._interrupt_with_chunks:
                on_text_chunk(chunk)
            raise OpenAIStreamInterrupted()

        step = self._steps.pop(0)
        for chunk in step.chunks:
            on_text_chunk(chunk)
        return step.response


class FakeToolExecutor:
    """Scripted fake for ToolExecutor."""

    def __init__(self, outputs_per_call: list[list[FunctionCallOutputItem]]) -> None:
        """Initialize scripted executor outputs for each invocation."""

        self._outputs_per_call = outputs_per_call
        self.calls: list[tuple[list[ToolCall], object | None]] = []

    async def execute(self, tool_calls, *, on_tool_start, logger=None):
        """Return scripted outputs and emit tool-start callbacks."""

        self.calls.append((tool_calls, logger))
        for call in tool_calls:
            on_tool_start(ToolStartEvent(tool_name=call.name))
        return self._outputs_per_call.pop(0)


@dataclass
class FakeFollowUpBuilder:
    """Scripted follow-up builder used to verify controller delegation."""

    responses_per_call: list[list[ResponsesInputItem]] = field(default_factory=list)
    seen_output_error_codes: list[list[str | None]] = field(default_factory=list)

    def __call__(self, outputs: list[FunctionCallOutputItem]) -> list[ResponsesInputItem]:
        """Return scripted follow-up messages while recording call context."""

        self.seen_output_error_codes.append([output.error_code for output in outputs])
        if not self.responses_per_call:
            return []
        return self.responses_per_call.pop(0)


class RecordingSink:
    """Sink collecting streamed text and tool starts for assertions."""

    def __init__(self) -> None:
        """Initialize empty collection buffers."""

        self.chunks: list[str] = []
        self.tool_starts: list[ToolStartEvent] = []

    def on_text_chunk(self, chunk: str) -> None:
        """Store streamed text chunks."""

        self.chunks.append(chunk)

    def on_tool_start(self, event: ToolStartEvent) -> None:
        """Store tool-start notifications."""

        self.tool_starts.append(event)


@pytest.mark.asyncio
async def test_handle_turn_executes_tool_then_streams_text(settings) -> None:
    """Controller should execute tool calls, stream final text, and preserve tool events."""

    responses_gateway = FakeResponsesGateway(
        [
            StreamStep(
                response=ModelResponse(
                    response_id="resp_1",
                    tool_calls=[
                        ToolCall(call_id="call_1", name="get_weather", arguments_json="{}")
                    ],
                    output_text="Let me check that for you.",
                ),
                chunks=[],
            ),
            StreamStep(
                response=ModelResponse(
                    response_id="resp_2", tool_calls=[], output_text="Hello world"
                ),
                chunks=["Hello", " world"],
            ),
        ]
    )
    tool_executor = FakeToolExecutor(
        [
            [
                FunctionCallOutputItem(
                    tool_name="get_weather",
                    call_id="call_1",
                    output_json='{"ok": true, "name": "London"}',
                    error_code=None,
                )
            ]
        ]
    )
    follow_up_builder = FakeFollowUpBuilder()

    controller = TurnController(
        settings=settings,
        responses_gateway=responses_gateway,
        tool_executor=tool_executor,
        build_follow_up_messages=follow_up_builder,
    )
    sink = RecordingSink()

    await controller.handle_turn("weather in london", sink)

    assert "".join(sink.chunks) == "Hello world"
    assert sink.tool_starts == [ToolStartEvent(tool_name="get_weather")]
    first_call_input, first_call_prev, _, first_tool_choice = responses_gateway.calls[0]
    second_call_input, second_call_prev, _, second_tool_choice = responses_gateway.calls[1]
    assert first_call_input == "weather in london"
    assert first_call_prev is None
    assert first_tool_choice == "auto"
    assert isinstance(second_call_input, list)
    assert second_call_input[0]["type"] == "function_call_output"
    assert second_call_prev == "resp_1"
    assert second_tool_choice == "auto"
    assert follow_up_builder.seen_output_error_codes == [[None]]


@pytest.mark.asyncio
async def test_follow_up_messages_are_appended_once_per_turn(settings) -> None:
    """Controller should append follow-up guidance at most once per turn."""

    responses_gateway = FakeResponsesGateway(
        [
            StreamStep(
                response=ModelResponse(
                    response_id="resp_1",
                    tool_calls=[
                        ToolCall(call_id="call_1", name="get_weather", arguments_json="{}")
                    ],
                    output_text="",
                ),
                chunks=[],
            ),
            StreamStep(
                response=ModelResponse(response_id="resp_2", tool_calls=[], output_text="Done"),
                chunks=["Done"],
            ),
        ]
    )
    tool_executor = FakeToolExecutor(
        [
            [
                FunctionCallOutputItem(
                    tool_name="get_weather",
                    call_id="call_1",
                    output_json='{"ok": false, "error_code": "LOCATION_AMBIGUOUS"}',
                    error_code="LOCATION_AMBIGUOUS",
                )
            ],
        ]
    )
    follow_up_builder = FakeFollowUpBuilder(
        responses_per_call=[
            [
                {
                    "role": "system",
                    "content": "Ask a clarification question without guessing.",
                }
            ]
        ]
    )

    controller = TurnController(
        settings=settings,
        responses_gateway=responses_gateway,
        tool_executor=tool_executor,
        build_follow_up_messages=follow_up_builder,
    )
    sink = RecordingSink()

    await controller.handle_turn("weather in london", sink)

    second_input, _, _, _ = responses_gateway.calls[1]
    _, _, _, second_tool_choice = responses_gateway.calls[1]
    assert isinstance(second_input, list)
    assert second_input[0]["type"] == "function_call_output"
    assert second_input[1]["role"] == "system"
    assert "clarification question" in second_input[1]["content"]
    assert second_tool_choice == "auto"
    assert follow_up_builder.seen_output_error_codes == [["LOCATION_AMBIGUOUS"]]
    assert "".join(sink.chunks) == "Done"


@pytest.mark.asyncio
async def test_handle_turn_keeps_tool_calls_enabled_until_final_answer(settings) -> None:
    """Controller should allow multiple tool/model rounds before the final answer."""

    responses_gateway = FakeResponsesGateway(
        [
            StreamStep(
                response=ModelResponse(
                    response_id="resp_1",
                    tool_calls=[
                        ToolCall(call_id="call_1", name="get_weather", arguments_json="{}")
                    ],
                    output_text="",
                ),
                chunks=[],
            ),
            StreamStep(
                response=ModelResponse(
                    response_id="resp_2",
                    tool_calls=[
                        ToolCall(call_id="call_2", name="get_weather", arguments_json="{}")
                    ],
                    output_text="",
                ),
                chunks=[],
            ),
            StreamStep(
                response=ModelResponse(response_id="resp_3", tool_calls=[], output_text="Done"),
                chunks=["Done"],
            ),
        ]
    )
    tool_executor = FakeToolExecutor(
        [
            [
                FunctionCallOutputItem(
                    tool_name="get_weather",
                    call_id="call_1",
                    output_json="{}",
                    error_code=None,
                )
            ],
            [
                FunctionCallOutputItem(
                    tool_name="get_weather",
                    call_id="call_2",
                    output_json="{}",
                    error_code=None,
                )
            ],
        ]
    )
    controller = TurnController(
        settings=settings,
        responses_gateway=responses_gateway,
        tool_executor=tool_executor,
    )
    sink = RecordingSink()

    await controller.handle_turn("weather in london then paris", sink)

    assert "".join(sink.chunks) == "Done"
    assert [event.tool_name for event in sink.tool_starts] == ["get_weather", "get_weather"]
    assert len(responses_gateway.calls) == 3
    assert [tool_choice for _, _, _, tool_choice in responses_gateway.calls] == [
        "auto",
        "auto",
        "auto",
    ]
    assert responses_gateway.calls[1][1] == "resp_1"
    assert responses_gateway.calls[2][1] == "resp_2"


@pytest.mark.asyncio
async def test_openai_transient_failure_returns_stable_message(settings) -> None:
    """Controller should emit stable fallback message on transient OpenAI failure."""

    controller = TurnController(
        settings=settings,
        responses_gateway=FakeResponsesGateway([], fail_with_transient=True),
        tool_executor=FakeToolExecutor([]),
    )
    sink = RecordingSink()

    await controller.handle_turn("weather in berlin", sink)

    assert "".join(sink.chunks).startswith("I'm having trouble reaching the language model")


@pytest.mark.asyncio
async def test_openai_auth_failure_returns_stable_message(settings) -> None:
    """Controller should emit clear guidance on OpenAI auth/config failures."""

    controller = TurnController(
        settings=settings,
        responses_gateway=FakeResponsesGateway([], fail_with_auth=True),
        tool_executor=FakeToolExecutor([]),
    )
    sink = RecordingSink()

    await controller.handle_turn("weather in berlin", sink)

    assert "".join(sink.chunks) == (
        "I couldn't reach OpenAI with the current API configuration. "
        "Check OPENAI_API_KEY and model access, then try again."
    )


@pytest.mark.asyncio
async def test_openai_client_failure_returns_stable_message(settings) -> None:
    """Controller should emit clear guidance on non-retryable OpenAI request failures."""

    controller = TurnController(
        settings=settings,
        responses_gateway=FakeResponsesGateway([], fail_with_client=True),
        tool_executor=FakeToolExecutor([]),
    )
    sink = RecordingSink()

    await controller.handle_turn("weather in berlin", sink)

    assert "".join(sink.chunks) == (
        "The OpenAI request was rejected. "
        "Check the configured model and request settings, then try again."
    )


@pytest.mark.asyncio
async def test_openai_protocol_error_returns_stable_message(settings) -> None:
    """Controller should emit stable fallback text on malformed final model payloads."""

    controller = TurnController(
        settings=settings,
        responses_gateway=FakeResponsesGateway([], fail_with_protocol=True),
        tool_executor=FakeToolExecutor([]),
    )
    sink = RecordingSink()

    await controller.handle_turn("weather in berlin", sink)

    assert "".join(sink.chunks) == (
        "I ran into an unexpected language-model response issue. Please try again."
    )


@pytest.mark.asyncio
async def test_handle_turn_passes_turn_logger_to_dependencies(settings, logger) -> None:
    """Controller should thread the turn logger through model and tool execution."""

    turn_logger = with_context(logger, {"run_id": "run-1", "turn_id": "turn-1"})
    responses_gateway = FakeResponsesGateway(
        [
            StreamStep(
                response=ModelResponse(
                    response_id="resp_1",
                    tool_calls=[
                        ToolCall(call_id="call_1", name="get_weather", arguments_json="{}")
                    ],
                    output_text="",
                ),
                chunks=[],
            ),
            StreamStep(
                response=ModelResponse(response_id="resp_2", tool_calls=[], output_text="Done"),
                chunks=["Done"],
            ),
        ]
    )
    tool_executor = FakeToolExecutor(
        [
            [
                FunctionCallOutputItem(
                    tool_name="get_weather",
                    call_id="call_1",
                    output_json="{}",
                    error_code=None,
                )
            ]
        ]
    )
    controller = TurnController(
        settings=settings,
        responses_gateway=responses_gateway,
        tool_executor=tool_executor,
    )

    await controller.handle_turn("weather in london", RecordingSink(), logger=turn_logger)

    assert all(call_logger is turn_logger for _, _, call_logger, _ in responses_gateway.calls)
    _, executor_logger = tool_executor.calls[0]
    assert executor_logger is turn_logger


@pytest.mark.asyncio
async def test_stream_interruption_appends_stable_message(settings) -> None:
    """Controller should append a stable message when a streamed round is interrupted."""

    controller = TurnController(
        settings=settings,
        responses_gateway=FakeResponsesGateway(
            [],
            interrupt_with_chunks=["Partial answer"],
        ),
        tool_executor=FakeToolExecutor([]),
    )
    sink = RecordingSink()

    await controller.handle_turn("weather in berlin", sink)

    assert "".join(sink.chunks) == (
        "Partial answer\n\nSorry, the response was interrupted. Please try again."
    )


@pytest.mark.asyncio
async def test_tool_iteration_limit_returns_stable_message(settings) -> None:
    """Controller should return stable guardrail message on tool-loop exhaustion."""

    limited_settings = settings.model_copy(update={"tool_iteration_limit": 1})

    responses_gateway = FakeResponsesGateway(
        [
            StreamStep(
                response=ModelResponse(
                    response_id="resp_1",
                    tool_calls=[
                        ToolCall(call_id="call_1", name="get_weather", arguments_json="{}")
                    ],
                    output_text="",
                ),
                chunks=[],
            ),
        ]
    )
    tool_executor = FakeToolExecutor(
        [
            [
                FunctionCallOutputItem(
                    tool_name="get_weather",
                    call_id="call_1",
                    output_json="{}",
                    error_code=None,
                )
            ]
        ]
    )
    controller = TurnController(
        settings=limited_settings,
        responses_gateway=responses_gateway,
        tool_executor=tool_executor,
    )
    sink = RecordingSink()

    await controller.handle_turn("weather in rome", sink)

    assert "".join(sink.chunks).startswith("I couldn't complete that request")


@pytest.mark.asyncio
async def test_tool_iteration_limit_keeps_last_committed_response_id(settings) -> None:
    """Exhausted tool loops should preserve the last fully completed response id."""

    limited_settings = settings.model_copy(update={"tool_iteration_limit": 1})
    responses_gateway = FakeResponsesGateway(
        [
            StreamStep(
                response=ModelResponse(
                    response_id="resp_committed",
                    tool_calls=[],
                    output_text="First answer",
                ),
                chunks=["First answer"],
            ),
            StreamStep(
                response=ModelResponse(
                    response_id="resp_unfinished",
                    tool_calls=[
                        ToolCall(call_id="call_1", name="get_weather", arguments_json="{}")
                    ],
                    output_text="",
                ),
                chunks=[],
            ),
            StreamStep(
                response=ModelResponse(
                    response_id="resp_after", tool_calls=[], output_text="Second answer"
                ),
                chunks=["Second answer"],
            ),
        ]
    )
    tool_executor = FakeToolExecutor(
        [
            [
                FunctionCallOutputItem(
                    tool_name="get_weather",
                    call_id="call_1",
                    output_json="{}",
                    error_code=None,
                )
            ]
        ]
    )
    controller = TurnController(
        settings=limited_settings,
        responses_gateway=responses_gateway,
        tool_executor=tool_executor,
    )

    await controller.handle_turn("weather in london", RecordingSink())
    await controller.handle_turn("weather in rome", RecordingSink())
    await controller.handle_turn("weather in paris", RecordingSink())

    _, third_turn_prev, _, _ = responses_gateway.calls[2]
    assert third_turn_prev == "resp_unfinished"


@pytest.mark.asyncio
async def test_tool_iteration_limit_without_committed_history_starts_next_turn_fresh(
    settings,
) -> None:
    """Exhaustion before any completed turn should leave the next turn without history."""

    limited_settings = settings.model_copy(update={"tool_iteration_limit": 1})
    responses_gateway = FakeResponsesGateway(
        [
            StreamStep(
                response=ModelResponse(
                    response_id="resp_unfinished",
                    tool_calls=[
                        ToolCall(call_id="call_1", name="get_weather", arguments_json="{}")
                    ],
                    output_text="",
                ),
                chunks=[],
            ),
            StreamStep(
                response=ModelResponse(
                    response_id="resp_after", tool_calls=[], output_text="Recovered"
                ),
                chunks=["Recovered"],
            ),
        ]
    )
    tool_executor = FakeToolExecutor(
        [
            [
                FunctionCallOutputItem(
                    tool_name="get_weather",
                    call_id="call_1",
                    output_json="{}",
                    error_code=None,
                )
            ]
        ]
    )
    controller = TurnController(
        settings=limited_settings,
        responses_gateway=responses_gateway,
        tool_executor=tool_executor,
    )

    await controller.handle_turn("weather in rome", RecordingSink())
    await controller.handle_turn("weather in paris", RecordingSink())

    _, second_turn_prev, _, _ = responses_gateway.calls[1]
    assert second_turn_prev == "resp_unfinished"


@pytest.mark.asyncio
async def test_handle_turn_rejects_overlong_input(settings) -> None:
    """Controller should reject input longer than configured max chars."""

    limited_settings = settings.model_copy(update={"max_input_chars": 5})
    controller = TurnController(
        settings=limited_settings,
        responses_gateway=FakeResponsesGateway([]),
        tool_executor=FakeToolExecutor([]),
    )

    with pytest.raises(ValueError):
        await controller.handle_turn("this is too long", RecordingSink())


@pytest.mark.asyncio
async def test_handle_turn_emits_non_streamed_text_when_no_tool_call_is_needed(settings) -> None:
    """No-tool turns should stream the final assistant text immediately."""

    responses_gateway = FakeResponsesGateway(
        [
            StreamStep(
                response=ModelResponse(
                    response_id="resp_1",
                    tool_calls=[],
                    output_text="Please ask about the weather in a city.",
                ),
                chunks=["Please ask about the weather in a city."],
            )
        ]
    )
    controller = TurnController(
        settings=settings,
        responses_gateway=responses_gateway,
        tool_executor=FakeToolExecutor([]),
    )
    sink = RecordingSink()

    await controller.handle_turn("tell me a joke", sink)

    assert "".join(sink.chunks) == "Please ask about the weather in a city."
    _, _, _, tool_choice = responses_gateway.calls[0]
    assert tool_choice == "auto"


@pytest.mark.asyncio
async def test_handle_turn_falls_back_to_final_text_when_no_stream_chunks_arrive(settings) -> None:
    """No-tool turns should still render final text when the stream emitted no chunks."""

    responses_gateway = FakeResponsesGateway(
        [
            StreamStep(
                response=ModelResponse(
                    response_id="resp_1",
                    tool_calls=[],
                    output_text="Please ask about the weather in a city.",
                ),
                chunks=[],
            )
        ]
    )
    controller = TurnController(
        settings=settings,
        responses_gateway=responses_gateway,
        tool_executor=FakeToolExecutor([]),
    )
    sink = RecordingSink()

    await controller.handle_turn("tell me a joke", sink)

    assert "".join(sink.chunks) == "Please ask about the weather in a city."
