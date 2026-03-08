# Weather Assistant CLI

`weather-assistant-cli` is an async command-line weather assistant built for an LLM
tool-calling exercise. It uses the OpenAI Responses API for streaming and tool
calling, and Open-Meteo for live weather data.

## What it demonstrates

- Non-blocking terminal input with `asyncio.to_thread`
- OpenAI Responses streaming when the model is answering with best-effort suppression of pre-tool filler text
- Model-driven tool calling with one weather lookup per location
- Concurrent tool execution with bounded parallelism
- Typed and async weather lookups with clear error handling
- A small reusable tool loop that supports repeated tool/model rounds and tool-local
  follow-up guidance without turning the repo into a framework

## Runtime shape

The runtime is intentionally flat:

- `weather_assistant_cli/cli.py`: terminal loop and composition root
- `weather_assistant_cli/assistant.py`: generic turn controller
- `weather_assistant_cli/openai_gateway.py`: OpenAI Responses streaming gateway
- `weather_assistant_cli/plugins/`: lean tool plugin contract, loader, and built-in plugins
  - `plugins/weather/`: prompts, domain models, service, and `get_weather` handler
- `weather_assistant_cli/tools.py`: generic tool contracts and executor
- `weather_assistant_cli/config.py`: pydantic-settings configuration
- `weather_assistant_cli/logging_config.py`: text and JSON log formatters
- `weather_assistant_cli/retry.py`: shared retry/backoff helper

Currently there is one built-in plugin, `weather`, which exposes the `get_weather` tool. The reusable
parts are deliberately thin so the code stays appropriate for the task while still being easy to
extend in a follow-up session.

Tool resolution rounds suppress pre-tool text on a best-effort basis by waiting for the first
streamed output item before printing buffered text. Final answers stream directly to the terminal.
Plugin discovery is deterministic and fail-fast: the app auto-loads non-private direct child
modules under `weather_assistant_cli/plugins/` in alphabetical order and aborts startup with a clear
error if any plugin import/build step is invalid.

## Requirements

- Python 3.11+
- OpenAI API key

## Setup

Recommended:

```bash
make setup
```

This creates `.venv`, installs `.[dev]`, creates `.env` from `.env.example` when needed, and
prompts for `OPENAI_API_KEY`.

Manual fallback:

```bash
python3 -m venv .venv  # or: python -m venv .venv
.venv/bin/python -m pip install -e .[dev]
cp .env.example .env
# then set OPENAI_API_KEY in .env
```

## Run

```bash
make run
```

Commands:

- `quit` or `exit`: stop the app
- `reset`: clear conversation context

## Error handling

- OpenAI retries connection errors, timeouts, `429`, and `5xx` responses
- OpenAI auth/config failures (`401`/`403`) return a clear API-key/access message
- OpenAI client/request failures (`400`/`404`) return a clear configuration message
- Open-Meteo retries timeouts, request errors, and transient `502`/`503`/`504` responses
- Weather lookups return stable typed failures for invalid cities, ambiguous locations, qualifier
  mismatches, network errors, and malformed upstream responses

## Adding another tool later

The extension seam is intentionally small:

1. Add a direct child module under `weather_assistant_cli/plugins/`
2. Export `build_bundle(*, settings, http_client, logger) -> ToolBundle`
3. Return a `ToolBundle` with the new `ToolHandler`, any tool-specific instructions, and optional
   follow-up guidance scoped to outputs from that tool only

That is sufficient for introducing additional tools without changing the turn controller or executor.
Repeated tool/model rounds already work with the existing runtime.

Current limitation: the shipped assistant is still weather-first. The CLI copy and the built-in
system instructions assume a weather assistant, so adding another weather-adjacent tool is
straightforward, while adding a different domain tool would first require relaxing the
weather-specific prompt/branding. This is intentionally not a third-party plugin framework;
discovery is limited to built-in modules under `weather_assistant_cli/plugins/`. Private modules
prefixed with `_` are ignored.

## Developer checks

```bash
make format
make lint
make typecheck
make test
make check
```
