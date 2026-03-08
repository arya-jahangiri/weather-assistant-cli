"""Tests for the local cleanup helper script."""

from __future__ import annotations

from pathlib import Path

from scripts.clean_reset import run_cleanup


def test_run_cleanup_removes_allowlisted_local_state(tmp_path: Path) -> None:
    """Cleanup should remove dev artifacts without touching source files."""

    source_file = tmp_path / "weather_assistant_cli" / "app.py"
    source_file.parent.mkdir(parents=True, exist_ok=True)
    source_file.write_text("print('ok')\n", encoding="utf-8")

    artifacts = [
        tmp_path / ".env",
        tmp_path / ".coverage",
        tmp_path / ".venv" / "bin" / "python",
        tmp_path / ".pytest_cache" / "state",
        tmp_path / ".ruff_cache" / "state",
        tmp_path / "htmlcov" / "index.html",
        tmp_path / "weather_assistant_cli" / "__pycache__" / "app.cpython-314.pyc",
        tmp_path / "tests" / "__pycache__" / "test.cpython-314.pyc",
        tmp_path / "docs" / ".DS_Store",
        tmp_path / "weather_assistant_cli.egg-info" / "PKG-INFO",
    ]
    for artifact in artifacts:
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text("temp\n", encoding="utf-8")

    exit_code = run_cleanup(tmp_path, assume_yes=True, stdout=lambda _message: None)

    assert exit_code == 0
    assert source_file.exists()
    for artifact in artifacts:
        assert not artifact.exists()
    assert not (tmp_path / ".venv").exists()
    assert not (tmp_path / ".pytest_cache").exists()
    assert not (tmp_path / ".ruff_cache").exists()
    assert not (tmp_path / "htmlcov").exists()
    assert not (tmp_path / "weather_assistant_cli.egg-info").exists()


def test_run_cleanup_is_confirmed_and_idempotent(tmp_path: Path) -> None:
    """Cleanup should prompt by default and stay safe to rerun."""

    env_path = tmp_path / ".env"
    env_path.write_text("OPENAI_API_KEY=sk-test\n", encoding="utf-8")

    messages: list[str] = []
    cancelled = run_cleanup(
        tmp_path,
        assume_yes=False,
        input_reader=lambda _prompt: "n",
        stdout=messages.append,
    )
    assert cancelled == 1
    assert env_path.exists()

    first_pass = run_cleanup(tmp_path, assume_yes=True, stdout=messages.append)
    second_pass = run_cleanup(tmp_path, assume_yes=True, stdout=messages.append)

    assert first_pass == 0
    assert second_pass == 0
    assert not env_path.exists()
