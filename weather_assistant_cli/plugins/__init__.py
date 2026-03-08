"""In-repo tool plugin loading and bundle contracts."""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, cast

import httpx

from weather_assistant_cli.config import Settings
from weather_assistant_cli.logging_config import LoggerLike
from weather_assistant_cli.openai_gateway import ResponsesInputItem
from weather_assistant_cli.tools import FunctionCallOutputItem, ToolHandler

FollowUpMessageBuilder = Callable[[Sequence[FunctionCallOutputItem]], list[ResponsesInputItem]]


@dataclass(frozen=True, slots=True)
class ToolBundle:
    """Minimal composition unit for registering one tool."""

    handler: ToolHandler
    instructions: str
    build_follow_up_messages: FollowUpMessageBuilder | None = None


class ToolPlugin(Protocol):
    """Module-level contract for one in-repo tool plugin."""

    def build_bundle(
        self,
        *,
        settings: Settings,
        http_client: httpx.AsyncClient,
        logger: LoggerLike,
    ) -> ToolBundle:
        """Build the tool bundle exposed by this plugin."""

        ...


class PluginLoadError(RuntimeError):
    """Raised when plugin discovery or plugin bundle construction fails."""


def load_tool_bundles(
    *,
    settings: Settings,
    http_client: httpx.AsyncClient,
    logger: LoggerLike,
) -> list[ToolBundle]:
    """Discover, import, and build all in-repo tool bundles."""

    bundles: list[ToolBundle] = []
    seen_tool_names: set[str] = set()

    module_infos = sorted(pkgutil.iter_modules(__path__), key=lambda module_info: module_info.name)
    for module_info in module_infos:
        if module_info.name.startswith("_"):
            continue

        module_name = f"{__name__}.{module_info.name}"
        try:
            module = cast(ToolPlugin, importlib.import_module(module_name))
        except Exception as exc:
            raise PluginLoadError(f"Failed to import plugin '{module_name}': {exc}") from exc

        build_bundle = getattr(module, "build_bundle", None)
        if build_bundle is None or not callable(build_bundle):
            raise PluginLoadError(
                f"Plugin '{module_name}' must export a callable build_bundle(...)."
            )

        try:
            bundle = build_bundle(settings=settings, http_client=http_client, logger=logger)
        except Exception as exc:
            raise PluginLoadError(f"Failed to build plugin '{module_name}': {exc}") from exc

        if not isinstance(bundle, ToolBundle):
            raise PluginLoadError(f"Plugin '{module_name}' returned an invalid ToolBundle.")

        tool_name = bundle.handler.name
        if tool_name in seen_tool_names:
            raise PluginLoadError(f"Duplicate tool name registered: {tool_name}")

        seen_tool_names.add(tool_name)
        bundles.append(bundle)

    if not bundles:
        raise PluginLoadError(f"No tool plugins were discovered under '{__name__}'.")

    return bundles


__all__ = [
    "FollowUpMessageBuilder",
    "PluginLoadError",
    "ToolBundle",
    "ToolPlugin",
    "load_tool_bundles",
]
