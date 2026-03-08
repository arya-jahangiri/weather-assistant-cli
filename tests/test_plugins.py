"""Tests for in-repo plugin discovery and loading."""

from __future__ import annotations

import pkgutil
from dataclasses import dataclass
from types import ModuleType
from typing import Any

import httpx
import pytest

import weather_assistant_cli.plugins as plugins_module
from weather_assistant_cli.plugins import PluginLoadError, ToolBundle, load_tool_bundles
from weather_assistant_cli.plugins.weather import build_bundle as build_weather_bundle
from weather_assistant_cli.plugins.weather.service import build_weather_follow_up_messages


@dataclass
class FakeHandler:
    """Minimal handler implementation for plugin loader tests."""

    name_value: str

    @property
    def name(self) -> str:
        """Return a stable fake tool name."""

        return self.name_value

    @property
    def schema(self) -> dict[str, object]:
        """Return a minimal schema compatible with the runtime."""

        return {"type": "function", "name": self.name}

    def preview_invocation(self, arguments_json: str) -> str | None:
        """Ignore preview arguments in plugin loader tests."""

        del arguments_json
        return None

    async def run(self, arguments_json: str, *, logger=None) -> Any:
        """Ignore tool execution in plugin loader tests."""

        del arguments_json, logger
        return None


def _module_info(name: str, *, ispkg: bool = False) -> pkgutil.ModuleInfo:
    """Build one synthetic module-info entry."""

    return pkgutil.ModuleInfo(module_finder=None, name=name, ispkg=ispkg)


def _plugin_module(
    tool_name: str,
    *,
    instructions: str | None = None,
    build_bundle: Any | None = None,
) -> ModuleType:
    """Create one fake plugin module."""

    module = ModuleType(f"fake_{tool_name}")
    if build_bundle is not None:
        module.build_bundle = build_bundle
        return module

    def default_builder(*, settings, http_client, logger) -> ToolBundle:
        del settings, http_client, logger
        return ToolBundle(
            handler=FakeHandler(tool_name),
            instructions=instructions or f"{tool_name} instructions",
        )

    module.build_bundle = default_builder
    return module


def test_load_tool_bundles_discovers_sorted_modules_and_ignores_private_entries(
    settings: Any,
    logger,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Loader should import public modules and packages in alphabetical order."""

    seen_module_names: list[str] = []
    modules = {
        "weather_assistant_cli.plugins.alpha": _plugin_module("alpha"),
        "weather_assistant_cli.plugins.beta": _plugin_module("beta"),
        "weather_assistant_cli.plugins.gamma": _plugin_module("gamma"),
        "weather_assistant_cli.plugins.nested": _plugin_module("nested"),
    }

    def fake_iter_modules(_paths: Any) -> list[pkgutil.ModuleInfo]:
        return [
            _module_info("gamma"),
            _module_info("_private"),
            _module_info("beta"),
            _module_info("alpha"),
            _module_info("nested", ispkg=True),
        ]

    def fake_import_module(module_name: str) -> ModuleType:
        seen_module_names.append(module_name)
        return modules[module_name]

    monkeypatch.setattr(plugins_module.pkgutil, "iter_modules", fake_iter_modules)
    monkeypatch.setattr(plugins_module.importlib, "import_module", fake_import_module)

    bundles = load_tool_bundles(
        settings=settings,
        http_client=object(),
        logger=logger,
    )

    assert [bundle.handler.name for bundle in bundles] == ["alpha", "beta", "gamma", "nested"]
    assert seen_module_names == [
        "weather_assistant_cli.plugins.alpha",
        "weather_assistant_cli.plugins.beta",
        "weather_assistant_cli.plugins.gamma",
        "weather_assistant_cli.plugins.nested",
    ]


def test_load_tool_bundles_rejects_missing_build_bundle(
    settings: Any,
    logger,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Plugins must export a callable build_bundle."""

    module = ModuleType("missing_builder")

    monkeypatch.setattr(
        plugins_module.pkgutil, "iter_modules", lambda _paths: [_module_info("broken")]
    )
    monkeypatch.setattr(
        plugins_module.importlib,
        "import_module",
        lambda _module_name: module,
    )

    with pytest.raises(PluginLoadError, match="callable build_bundle"):
        load_tool_bundles(settings=settings, http_client=object(), logger=logger)


def test_load_tool_bundles_rejects_invalid_tool_bundle(
    settings: Any,
    logger,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Plugins must return ToolBundle instances."""

    module = _plugin_module("broken", build_bundle=lambda **_kwargs: object())

    monkeypatch.setattr(
        plugins_module.pkgutil, "iter_modules", lambda _paths: [_module_info("broken")]
    )
    monkeypatch.setattr(
        plugins_module.importlib,
        "import_module",
        lambda _module_name: module,
    )

    with pytest.raises(PluginLoadError, match="invalid ToolBundle"):
        load_tool_bundles(settings=settings, http_client=object(), logger=logger)


def test_load_tool_bundles_rejects_duplicate_tool_names(
    settings: Any,
    logger,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two plugins must not expose the same tool name."""

    modules = {
        "weather_assistant_cli.plugins.alpha": _plugin_module("shared"),
        "weather_assistant_cli.plugins.beta": _plugin_module("shared"),
    }

    monkeypatch.setattr(
        plugins_module.pkgutil,
        "iter_modules",
        lambda _paths: [_module_info("alpha"), _module_info("beta")],
    )
    monkeypatch.setattr(
        plugins_module.importlib,
        "import_module",
        lambda module_name: modules[module_name],
    )

    with pytest.raises(PluginLoadError, match="Duplicate tool name registered: shared"):
        load_tool_bundles(settings=settings, http_client=object(), logger=logger)


def test_load_tool_bundles_wraps_import_and_build_failures_with_module_name(
    settings: Any,
    logger,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Import and build failures should surface the plugin module name."""

    def fake_iter_modules(_paths: Any) -> list[pkgutil.ModuleInfo]:
        return [_module_info("broken")]

    def fake_import_module(module_name: str) -> ModuleType:
        if module_name.endswith(".broken"):
            raise RuntimeError("boom")
        return ModuleType("unused")

    monkeypatch.setattr(plugins_module.pkgutil, "iter_modules", fake_iter_modules)
    monkeypatch.setattr(plugins_module.importlib, "import_module", fake_import_module)

    with pytest.raises(PluginLoadError, match=r"weather_assistant_cli\.plugins\.broken"):
        load_tool_bundles(settings=settings, http_client=object(), logger=logger)

    def broken_builder(**_kwargs: Any) -> ToolBundle:
        raise RuntimeError("builder boom")

    module = _plugin_module("broken", build_bundle=broken_builder)
    monkeypatch.setattr(plugins_module.importlib, "import_module", lambda _module_name: module)

    with pytest.raises(PluginLoadError, match=r"weather_assistant_cli\.plugins\.broken"):
        load_tool_bundles(settings=settings, http_client=object(), logger=logger)


def test_load_tool_bundles_rejects_empty_discovery(
    settings: Any,
    logger,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """At least one plugin must be discovered."""

    monkeypatch.setattr(plugins_module.pkgutil, "iter_modules", lambda _paths: [])

    with pytest.raises(PluginLoadError, match="No tool plugins were discovered"):
        load_tool_bundles(settings=settings, http_client=object(), logger=logger)


@pytest.mark.asyncio
async def test_weather_plugin_builds_the_expected_bundle(settings: Any, logger) -> None:
    """The built-in weather plugin should expose the existing weather tool wiring."""

    async with httpx.AsyncClient() as http_client:
        bundle = build_weather_bundle(settings=settings, http_client=http_client, logger=logger)

    assert bundle.handler.name == "get_weather"
    assert "weather-assistant-cli, a friendly CLI weather assistant" in bundle.instructions
    assert bundle.build_follow_up_messages is build_weather_follow_up_messages
