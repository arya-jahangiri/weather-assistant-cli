"""Tests for terminal CLI behavior and entrypoints."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, ClassVar

import pytest
from pydantic import ValidationError

import weather_assistant_cli.cli as cli_module
from weather_assistant_cli.config import Settings
from weather_assistant_cli.plugins import ToolBundle
from weather_assistant_cli.tools import FunctionCallOutputItem, ToolStartEvent


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Build valid settings from environment variables."""

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    return Settings()


def _patch_dependencies(monkeypatch: pytest.MonkeyPatch, *, turn_controller_class: type) -> None:
    """Patch composition-root dependencies with lightweight fakes."""

    class FakeToolHandler:
        """No-op tool handler fake."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            """Accept constructor parameters used by composition root."""

            return None

        @property
        def name(self) -> str:
            """Return tool name expected by executor consumers."""

            return "get_weather"

        @property
        def schema(self) -> dict[str, object]:
            """Return minimal schema required by gateway setup."""

            return {"type": "function", "name": "get_weather"}

        def preview_invocation(self, arguments_json: str) -> str | None:
            """Return static preview value for tests that inspect callbacks."""

            del arguments_json
            return "London"

        async def run(self, arguments_json: str, *, logger=None):
            """Return deterministic payload for executor-level calls."""

            del logger, arguments_json
            return None

    class FakeToolExecutor:
        """No-op tool executor fake."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            """Accept constructor parameters used by composition root."""

            return None

    class FakeResponsesGateway:
        """No-op responses gateway fake."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            """Accept constructor parameters used by composition root."""

            return None

        async def aclose(self) -> None:
            """Support CLI shutdown cleanup."""

            return None

    def fake_load_tool_bundles(
        *, settings: Settings, http_client: Any, logger: Any
    ) -> list[ToolBundle]:
        """Return one deterministic bundle for CLI wiring tests."""

        del settings, http_client, logger
        return [ToolBundle(handler=FakeToolHandler(), instructions="Weather instructions.")]

    monkeypatch.setattr(cli_module, "load_tool_bundles", fake_load_tool_bundles)
    monkeypatch.setattr(cli_module, "ToolExecutor", FakeToolExecutor)
    monkeypatch.setattr(cli_module, "ResponsesGateway", FakeResponsesGateway)
    monkeypatch.setattr(cli_module, "TurnController", turn_controller_class)


def test_is_exit_command() -> None:
    """quit and exit should be recognized case-insensitively."""

    assert cli_module.is_exit_command("quit") is True
    assert cli_module.is_exit_command("EXIT") is True
    assert cli_module.is_exit_command("reset") is False


def test_is_reset_command() -> None:
    """Only reset should trigger context reset."""

    assert cli_module.is_reset_command("reset") is True
    assert cli_module.is_reset_command(" RESET ") is True
    assert cli_module.is_reset_command("restart") is False


def test_terminal_sink_streams_chunks_without_line_buffering(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Terminal sink should print streamed chunks directly as received."""

    sink = cli_module.TerminalSink()
    sink.on_text_chunk("Here")
    sink.on_text_chunk("'s ")
    sink.on_text_chunk("a stream")
    sink.finish_turn()

    stdout = capsys.readouterr().out
    assert "\nAssistant: Here's a stream\n" in stdout


def test_terminal_sink_renders_tool_events(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Terminal sink should render generic tool names with optional preview text."""

    sink = cli_module.TerminalSink()
    sink.on_tool_start(ToolStartEvent(tool_name="get_weather", preview_text="London"))
    sink.on_tool_start(ToolStartEvent(tool_name="summarize"))

    stdout = capsys.readouterr().out
    assert "[Calling get_weather for London...]" in stdout
    assert "[Calling summarize...]" in stdout


def test_tool_bundle_composition_derives_handlers_instructions_and_follow_ups() -> None:
    """CLI composition helpers should derive runtime wiring and tool-local follow-ups."""

    @dataclass
    class FakeHandler:
        name_value: str

        @property
        def name(self) -> str:
            return self.name_value

        @property
        def schema(self) -> dict[str, object]:
            return {"type": "function", "name": self.name}

        def preview_invocation(self, arguments_json: str) -> str | None:
            del arguments_json
            return None

        async def run(self, arguments_json: str, *, logger=None):
            del arguments_json, logger
            return None

    first_calls: list[list[str]] = []

    def build_first_follow_up(outputs: list[FunctionCallOutputItem]) -> list[dict[str, str]]:
        first_calls.append([output.tool_name for output in outputs])
        if any(output.error_code == "LOCATION_AMBIGUOUS" for output in outputs):
            return [{"role": "system", "content": "first"}]
        return []

    second_calls: list[list[str]] = []

    def build_second_follow_up(outputs: list[FunctionCallOutputItem]) -> list[dict[str, str]]:
        second_calls.append([output.tool_name for output in outputs])
        if any(output.error_code == "LOCATION_AMBIGUOUS" for output in outputs):
            return [{"role": "system", "content": "second"}]
        return []

    tool_bundles = [
        cli_module.ToolBundle(
            handler=FakeHandler("alpha"),
            instructions="Alpha instructions.",
            build_follow_up_messages=build_first_follow_up,
        ),
        cli_module.ToolBundle(
            handler=FakeHandler("beta"),
            instructions="Beta instructions.",
            build_follow_up_messages=build_second_follow_up,
        ),
    ]

    handlers = cli_module.build_tool_handlers(tool_bundles)
    instructions = cli_module.compose_system_instructions(tool_bundles)
    follow_up_builder = cli_module.compose_follow_up_builder(tool_bundles)

    assert list(handlers) == ["alpha", "beta"]
    assert instructions == "\n\n".join(
        [
            cli_module.TOOL_STREAMING_INSTRUCTIONS,
            "Alpha instructions.",
            "Beta instructions.",
        ]
    )
    assert follow_up_builder is not None
    assert follow_up_builder([]) == []
    assert follow_up_builder(
        [
            FunctionCallOutputItem(
                tool_name="beta",
                call_id="call_1",
                output_json='{"ok": false}',
                error_code="LOCATION_AMBIGUOUS",
            )
        ]
    ) == [{"role": "system", "content": "second"}]
    assert first_calls == []
    assert second_calls == [["beta"]]


@pytest.mark.asyncio
async def test_run_cli_handles_reset_and_quit(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI should process reset then quit in order."""

    class FakeTurnController:
        """Controller fake tracking reset calls."""

        instances: ClassVar[list[FakeTurnController]] = []

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            """Track constructed controller instance."""

            self.reset_calls = 0
            FakeTurnController.instances.append(self)

        def reset_context(self) -> None:
            """Increment reset counter for assertions."""

            self.reset_calls += 1

        async def handle_turn(
            self,
            user_text: str,
            sink: Any,
            *,
            logger: Any | None = None,
        ) -> None:
            """No-op handle_turn for reset/quit path test."""

            del user_text, sink, logger
            return None

    _patch_dependencies(monkeypatch, turn_controller_class=FakeTurnController)

    inputs: Iterator[str] = iter(["reset", "quit"])

    async def fake_to_thread(_fn: Any, _prompt: str) -> str:
        """Return scripted user input values without blocking."""

        return next(inputs)

    monkeypatch.setattr(cli_module.asyncio, "to_thread", fake_to_thread)

    exit_code = await cli_module.run_cli(settings)

    assert exit_code == 0
    stdout = capsys.readouterr().out
    assert "Weather Assistant - ask about the weather in any city." in stdout
    assert "Type 'quit' to exit." in stdout
    assert "Conversation reset." in stdout
    assert "Goodbye!" in stdout
    assert FakeTurnController.instances[0].reset_calls == 1


@pytest.mark.asyncio
async def test_run_cli_renders_stream_and_unexpected_error(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI should render tool calls, assistant chunks, and fallback error text."""

    class FakeTurnController:
        """Controller fake supporting one success and one failure turn."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            """Initialize turn counter."""

            self.calls = 0

        def reset_context(self) -> None:
            """No-op reset for this test."""

            return None

        async def handle_turn(
            self,
            user_text: str,
            sink: Any,
            *,
            logger: Any | None = None,
        ) -> None:
            """Emit chunks on first call and raise on second."""

            del logger
            self.calls += 1
            if self.calls == 1:
                sink.on_tool_start(ToolStartEvent(tool_name="get_weather", preview_text="London"))
                sink.on_text_chunk("Hello ")
                sink.on_text_chunk("there")
                return
            raise RuntimeError(f"boom: {user_text}")

    _patch_dependencies(monkeypatch, turn_controller_class=FakeTurnController)

    inputs: Iterator[str] = iter(["Weather in London", "second turn", "quit"])

    async def fake_to_thread(_fn: Any, _prompt: str) -> str:
        """Return scripted user input values without blocking."""

        return next(inputs)

    monkeypatch.setattr(cli_module.asyncio, "to_thread", fake_to_thread)

    exit_code = await cli_module.run_cli(settings)

    assert exit_code == 0
    stdout = capsys.readouterr().out
    assert "[Calling get_weather for London...]" in stdout
    assert "Assistant: Hello there" in stdout
    assert "Assistant: Something unexpected went wrong. Please try again." in stdout
    assert "boom: second turn" not in stdout


@pytest.mark.asyncio
async def test_run_cli_finishes_partial_line_before_value_error(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """ValueError fallback text should start on a fresh line after partial output."""

    class FakeTurnController:
        """Controller fake that emits partial output before validation failure."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            """Accept constructor parameters."""

            return None

        def reset_context(self) -> None:
            """No-op reset implementation."""

            return None

        async def handle_turn(
            self,
            user_text: str,
            sink: Any,
            *,
            logger: Any | None = None,
        ) -> None:
            """Emit partial output before raising ValueError."""

            del user_text, logger
            sink.on_text_chunk("Partial answer")
            raise ValueError("Bad input")

    _patch_dependencies(monkeypatch, turn_controller_class=FakeTurnController)

    inputs: Iterator[str] = iter(["weather in london", "quit"])

    async def fake_to_thread(_fn: Any, _prompt: str) -> str:
        """Return scripted user input values without blocking."""

        return next(inputs)

    monkeypatch.setattr(cli_module.asyncio, "to_thread", fake_to_thread)

    assert await cli_module.run_cli(settings) == 0
    stdout = capsys.readouterr().out
    assert "Assistant: Partial answer\nAssistant: Bad input" in stdout


@pytest.mark.asyncio
async def test_run_cli_finishes_partial_line_before_unexpected_error(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Unexpected-error fallback text should start on a fresh line after partial output."""

    class FakeTurnController:
        """Controller fake that emits partial output before crashing."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            """Accept constructor parameters."""

            return None

        def reset_context(self) -> None:
            """No-op reset implementation."""

            return None

        async def handle_turn(
            self,
            user_text: str,
            sink: Any,
            *,
            logger: Any | None = None,
        ) -> None:
            """Emit partial output before raising RuntimeError."""

            del user_text, logger
            sink.on_text_chunk("Partial answer")
            raise RuntimeError("boom")

    _patch_dependencies(monkeypatch, turn_controller_class=FakeTurnController)

    inputs: Iterator[str] = iter(["weather in london", "quit"])

    async def fake_to_thread(_fn: Any, _prompt: str) -> str:
        """Return scripted user input values without blocking."""

        return next(inputs)

    monkeypatch.setattr(cli_module.asyncio, "to_thread", fake_to_thread)

    assert await cli_module.run_cli(settings) == 0
    stdout = capsys.readouterr().out
    assert (
        "Assistant: Partial answer\nAssistant: Something unexpected went wrong. Please try again."
        in stdout
    )


@pytest.mark.asyncio
async def test_run_cli_handles_eof(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    """EOF on stdin should terminate CLI cleanly."""

    class FakeTurnController:
        """Controller fake for EOF test."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            """Accept constructor parameters."""

            return None

        def reset_context(self) -> None:
            """No-op reset implementation."""

            return None

        async def handle_turn(
            self,
            user_text: str,
            sink: Any,
            *,
            logger: Any | None = None,
        ) -> None:
            """No-op handle_turn for EOF test."""

            del user_text, sink, logger
            return None

    _patch_dependencies(monkeypatch, turn_controller_class=FakeTurnController)

    async def fake_to_thread(_fn: Any, _prompt: str) -> str:
        """Raise EOF immediately to trigger graceful exit."""

        raise EOFError

    monkeypatch.setattr(cli_module.asyncio, "to_thread", fake_to_thread)

    assert await cli_module.run_cli(settings) == 0


@pytest.mark.asyncio
async def test_run_cli_uses_to_thread_for_input(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI input should be sourced via asyncio.to_thread for non-blocking behavior."""

    class FakeTurnController:
        """Controller fake that exits after one handled turn and quit command."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            """Initialize call counter."""

            self.calls = 0

        def reset_context(self) -> None:
            """No-op reset implementation."""

            return None

        async def handle_turn(
            self,
            user_text: str,
            sink: Any,
            *,
            logger: Any | None = None,
        ) -> None:
            """Emit one chunk for first call."""

            del logger
            self.calls += 1
            sink.on_text_chunk(f"ack:{user_text}")

    _patch_dependencies(monkeypatch, turn_controller_class=FakeTurnController)

    inputs: Iterator[str] = iter(["weather in paris", "quit"])
    to_thread_calls: list[str] = []

    async def fake_to_thread(_fn: Any, prompt: str) -> str:
        """Capture prompt argument for to_thread call assertions."""

        to_thread_calls.append(prompt)
        return next(inputs)

    monkeypatch.setattr(cli_module.asyncio, "to_thread", fake_to_thread)

    assert await cli_module.run_cli(settings) == 0
    assert to_thread_calls
    assert all(prompt == "You: " for prompt in to_thread_calls)


@pytest.mark.asyncio
async def test_run_cli_closes_responses_gateway_on_quit(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI should close the responses gateway before exiting."""

    class FakeResponsesGateway:
        """Responses gateway fake tracking shutdown calls."""

        instances: ClassVar[list[FakeResponsesGateway]] = []

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            """Track instances created by the composition root."""

            self.closed = 0
            FakeResponsesGateway.instances.append(self)

        async def aclose(self) -> None:
            """Track cleanup execution."""

            self.closed += 1

    class FakeTurnController:
        """Controller fake for quit-path cleanup testing."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            """Accept constructor parameters."""

            return None

        def reset_context(self) -> None:
            """No-op reset implementation."""

            return None

        async def handle_turn(
            self,
            user_text: str,
            sink: Any,
            *,
            logger: Any | None = None,
        ) -> None:
            """No-op handle_turn for this test."""

            del user_text, sink, logger
            return None

    _patch_dependencies(monkeypatch, turn_controller_class=FakeTurnController)
    monkeypatch.setattr(cli_module, "ResponsesGateway", FakeResponsesGateway)

    inputs: Iterator[str] = iter(["quit"])

    async def fake_to_thread(_fn: Any, _prompt: str) -> str:
        """Return scripted quit input without blocking."""

        return next(inputs)

    monkeypatch.setattr(cli_module.asyncio, "to_thread", fake_to_thread)

    assert await cli_module.run_cli(settings) == 0
    assert FakeResponsesGateway.instances[0].closed == 1


@pytest.mark.asyncio
async def test_run_cli_reports_plugin_load_failure(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Plugin loading failures should abort startup with a concise stderr message."""

    def fake_load_tool_bundles(
        *, settings: Settings, http_client: Any, logger: Any
    ) -> list[ToolBundle]:
        """Raise a deterministic plugin load error."""

        del settings, http_client, logger
        raise cli_module.PluginLoadError("bad plugin")

    monkeypatch.setattr(cli_module, "load_tool_bundles", fake_load_tool_bundles)

    assert await cli_module.run_cli(settings) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Plugin load error: bad plugin" in captured.err


@pytest.mark.asyncio
async def test_async_main_passes_loaded_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """async_main should load settings and pass them to run_cli."""

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    expected_settings = Settings()

    async def fake_run_cli(received: Settings) -> int:
        """Assert async_main forwards loaded settings unchanged."""

        assert received == expected_settings
        return 7

    monkeypatch.setattr(cli_module, "load_settings", lambda: expected_settings)
    monkeypatch.setattr(cli_module, "run_cli", fake_run_cli)

    assert await cli_module.async_main() == 7


@pytest.mark.asyncio
async def test_async_main_reports_validation_errors(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """async_main should render a concise settings error instead of a traceback."""

    validation_error = ValidationError.from_exception_data(
        "Settings",
        [{"type": "missing", "loc": ("OPENAI_API_KEY",), "input": None}],
    )

    def fake_load_settings() -> Settings:
        raise validation_error

    monkeypatch.setattr(cli_module, "load_settings", fake_load_settings)

    assert await cli_module.async_main() == 1
    stderr = capsys.readouterr().err
    assert "Configuration error." in stderr
    assert "OPENAI_API_KEY" in stderr


def test_main_uses_asyncio_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """main should delegate loop management to asyncio.run."""

    def fake_asyncio_run(coro: Any) -> int:
        """Assert main passes async_main coroutine to asyncio.run."""

        assert asyncio.iscoroutine(coro)
        coro.close()
        return 5

    monkeypatch.setattr(cli_module.asyncio, "run", fake_asyncio_run)

    assert cli_module.main() == 5
