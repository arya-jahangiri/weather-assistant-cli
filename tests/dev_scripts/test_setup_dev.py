"""Tests for the local setup helper script."""

from __future__ import annotations

import sys
from pathlib import Path

from scripts.setup_dev import run_setup, venv_python_path


def test_run_setup_bootstraps_venv_and_env(tmp_path: Path) -> None:
    """Setup should create .env from example and write the prompted API key."""

    (tmp_path / ".env.example").write_text(
        "OPENAI_API_KEY=your_openai_api_key_here\nLOG_LEVEL=WARNING\n",
        encoding="utf-8",
    )

    commands: list[tuple[list[str], Path, bool]] = []

    def fake_run(command: list[str], *, cwd: Path, check: bool) -> object:
        commands.append((command, cwd, check))
        if command[:3] == [sys.executable, "-m", "venv"]:
            python_path = venv_python_path(tmp_path)
            python_path.parent.mkdir(parents=True, exist_ok=True)
            python_path.write_text("", encoding="utf-8")
        return object()

    messages: list[str] = []
    run_setup(
        tmp_path,
        run_command=fake_run,
        prompt_secret=lambda _prompt: "sk-test-123",
        stdout=messages.append,
    )

    assert commands[0] == ([sys.executable, "-m", "venv", str(tmp_path / ".venv")], tmp_path, True)
    assert commands[1] == (
        [str(venv_python_path(tmp_path)), "-m", "pip", "install", "-e", ".[dev]"],
        tmp_path,
        True,
    )
    assert (tmp_path / ".env").read_text(encoding="utf-8") == (
        "OPENAI_API_KEY=sk-test-123\nLOG_LEVEL=WARNING\n"
    )
    assert any("Next commands:" in message for message in messages)


def test_run_setup_reuses_existing_state_and_preserves_other_env_values(tmp_path: Path) -> None:
    """Setup should reuse .venv and only replace OPENAI_API_KEY inside .env."""

    python_path = venv_python_path(tmp_path)
    python_path.parent.mkdir(parents=True, exist_ok=True)
    python_path.write_text("", encoding="utf-8")

    (tmp_path / ".env.example").write_text(
        "OPENAI_API_KEY=your_openai_api_key_here\nLOG_LEVEL=WARNING\n",
        encoding="utf-8",
    )
    (tmp_path / ".env").write_text(
        "OPENAI_API_KEY=sk-old\nLOG_LEVEL=WARNING\nMAX_CONCURRENCY=5\n",
        encoding="utf-8",
    )

    commands: list[tuple[list[str], Path, bool]] = []

    def fake_run(command: list[str], *, cwd: Path, check: bool) -> object:
        commands.append((command, cwd, check))
        return object()

    run_setup(
        tmp_path,
        run_command=fake_run,
        prompt_secret=lambda _prompt: "sk-new",
        stdout=lambda _message: None,
    )

    assert commands == [
        (
            [str(venv_python_path(tmp_path)), "-m", "pip", "install", "-e", ".[dev]"],
            tmp_path,
            True,
        )
    ]
    assert (tmp_path / ".env").read_text(encoding="utf-8") == (
        "OPENAI_API_KEY=sk-new\nLOG_LEVEL=WARNING\nMAX_CONCURRENCY=5\n"
    )
