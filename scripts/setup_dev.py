"""Bootstrap a local development environment for weather-assistant-cli."""

from __future__ import annotations

import getpass
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PLACEHOLDER_OPENAI_API_KEY = "your_openai_api_key_here"

CommandRunner = Callable[..., object]
SecretPrompt = Callable[[str], str]
Printer = Callable[[str], object]


def venv_python_path(root: Path) -> Path:
    """Return the interpreter path inside the local virtualenv."""

    scripts_dir = "Scripts" if sys.platform == "win32" else "bin"
    python_name = "python.exe" if sys.platform == "win32" else "python"
    return root / ".venv" / scripts_dir / python_name


def ensure_virtualenv(root: Path, *, run_command: CommandRunner, stdout: Printer) -> Path:
    """Create .venv when it is missing and return its Python path."""

    python_path = venv_python_path(root)
    if python_path.exists():
        stdout("Using existing .venv")
        return python_path

    stdout("Creating .venv")
    run_command([sys.executable, "-m", "venv", str(root / ".venv")], cwd=root, check=True)
    return python_path


def install_dependencies(
    root: Path,
    *,
    python_path: Path,
    run_command: CommandRunner,
    stdout: Printer,
) -> None:
    """Install project and dev dependencies into the virtualenv."""

    stdout("Installing dependencies into .venv")
    run_command(
        [str(python_path), "-m", "pip", "install", "-e", ".[dev]"],
        cwd=root,
        check=True,
    )


def ensure_env_file(root: Path, *, stdout: Printer) -> Path:
    """Create .env from .env.example when needed."""

    env_path = root / ".env"
    if env_path.exists():
        stdout("Using existing .env")
        return env_path

    example_path = root / ".env.example"
    if not example_path.exists():
        raise FileNotFoundError(f"Missing required file: {example_path}")

    shutil.copyfile(example_path, env_path)
    stdout("Created .env from .env.example")
    return env_path


def read_env_value(env_path: Path, key: str) -> str | None:
    """Return the raw value for one env key, if present."""

    if not env_path.exists():
        return None

    prefix = f"{key}="
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if line.startswith(prefix):
            return line[len(prefix) :]
    return None


def upsert_env_value(env_path: Path, key: str, value: str) -> None:
    """Insert or replace one env key while preserving the rest of the file."""

    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    prefix = f"{key}="
    replacement = f"{key}={value}"

    for index, line in enumerate(lines):
        if line.startswith(prefix):
            lines[index] = replacement
            break
    else:
        lines.append(replacement)

    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def prompt_for_openai_api_key(
    existing_value: str | None,
    *,
    prompt_secret: SecretPrompt,
    stdout: Printer,
) -> str:
    """Read an API key from hidden input, keeping an existing real key on blank input."""

    existing_is_real = bool(existing_value and existing_value.strip() != PLACEHOLDER_OPENAI_API_KEY)
    prompt = (
        "Paste OPENAI_API_KEY (press Enter to keep the current key): "
        if existing_is_real
        else "Paste OPENAI_API_KEY: "
    )

    while True:
        entered = prompt_secret(prompt).strip()
        if entered:
            return entered
        if existing_is_real:
            assert existing_value is not None
            return existing_value.strip()
        stdout("OPENAI_API_KEY cannot be empty.")


def run_setup(
    root: Path = ROOT,
    *,
    run_command: CommandRunner = subprocess.run,
    prompt_secret: SecretPrompt = getpass.getpass,
    stdout: Printer = print,
) -> None:
    """Create local tooling state and prompt for OPENAI_API_KEY."""

    python_path = ensure_virtualenv(root, run_command=run_command, stdout=stdout)
    install_dependencies(root, python_path=python_path, run_command=run_command, stdout=stdout)

    env_path = ensure_env_file(root, stdout=stdout)
    existing_key = read_env_value(env_path, "OPENAI_API_KEY")
    api_key = prompt_for_openai_api_key(
        existing_key,
        prompt_secret=prompt_secret,
        stdout=stdout,
    )
    upsert_env_value(env_path, "OPENAI_API_KEY", api_key)

    stdout(f"Saved OPENAI_API_KEY to {env_path}")
    stdout("Next commands:")
    stdout("  make run")
    stdout("  make test")


def main() -> int:
    """Run the setup flow and return a process exit code."""

    try:
        run_setup()
    except KeyboardInterrupt:
        print("\nSetup cancelled.")
        return 130
    except Exception as exc:
        print(f"Setup failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
