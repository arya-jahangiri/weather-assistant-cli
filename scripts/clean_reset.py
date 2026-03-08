"""Remove local development state without touching tracked source files."""

from __future__ import annotations

import argparse
import os
import shutil
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ROOT_DIRS_TO_REMOVE = (".venv", ".pytest_cache", ".ruff_cache", "htmlcov")
ROOT_FILES_TO_REMOVE = (".env", ".coverage")
ROOT_GLOBS_TO_REMOVE = ("*.egg-info",)
RECURSIVE_DIR_NAMES = ("__pycache__",)
RECURSIVE_FILE_NAMES = (".DS_Store",)

InputReader = Callable[[str], str]
Printer = Callable[[str], object]


def collect_cleanup_targets(root: Path) -> list[Path]:
    """Return the fixed allowlist of removable local-state paths."""

    targets: list[Path] = []

    for name in ROOT_DIRS_TO_REMOVE:
        path = root / name
        if path.exists():
            targets.append(path)

    for name in ROOT_FILES_TO_REMOVE:
        path = root / name
        if path.exists():
            targets.append(path)

    for pattern in ROOT_GLOBS_TO_REMOVE:
        targets.extend(sorted(root.glob(pattern)))

    for current_root, dirnames, filenames in os.walk(root, topdown=True):
        current_path = Path(current_root)

        if current_path == root:
            dirnames[:] = [
                name
                for name in dirnames
                if name not in ROOT_DIRS_TO_REMOVE and not name.endswith(".egg-info")
            ]

        removable_dirs = [name for name in dirnames if name in RECURSIVE_DIR_NAMES]
        for name in removable_dirs:
            targets.append(current_path / name)
        dirnames[:] = [name for name in dirnames if name not in RECURSIVE_DIR_NAMES]

        for name in RECURSIVE_FILE_NAMES:
            if name in filenames:
                targets.append(current_path / name)

    unique_targets: dict[Path, None] = {}
    for path in sorted(targets):
        unique_targets[path] = None
    return list(unique_targets)


def confirm_cleanup(
    root: Path,
    targets: list[Path],
    *,
    input_reader: InputReader,
    stdout: Printer,
) -> bool:
    """Ask for confirmation before removing local state."""

    stdout("This will remove local development state:")
    for path in targets:
        stdout(f"  {path.relative_to(root)}")
    response = input_reader("Continue? [y/N]: ").strip().lower()
    return response in {"y", "yes"}


def remove_path(path: Path) -> None:
    """Delete one file or directory if it still exists."""

    if not path.exists():
        return
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
        return
    path.unlink()


def run_cleanup(
    root: Path = ROOT,
    *,
    assume_yes: bool = False,
    input_reader: InputReader = input,
    stdout: Printer = print,
) -> int:
    """Remove allowlisted local-state files and directories."""

    targets = collect_cleanup_targets(root)
    if not targets:
        stdout("Nothing to remove.")
        return 0

    if not assume_yes and not confirm_cleanup(
        root,
        targets,
        input_reader=input_reader,
        stdout=stdout,
    ):
        stdout("Cleanup cancelled.")
        return 1

    for path in targets:
        remove_path(path)
        stdout(f"Removed {path.relative_to(root)}")
    return 0


def parse_args() -> argparse.Namespace:
    """Parse command-line flags for cleanup."""

    parser = argparse.ArgumentParser(description="Remove local development state.")
    parser.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    return parser.parse_args()


def main() -> int:
    """Run cleanup and return a process exit code."""

    args = parse_args()
    return run_cleanup(assume_yes=args.yes)


if __name__ == "__main__":
    raise SystemExit(main())
