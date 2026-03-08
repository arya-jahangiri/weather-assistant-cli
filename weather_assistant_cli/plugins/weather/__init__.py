"""Built-in weather tool plugin."""

from __future__ import annotations

import httpx

from weather_assistant_cli.config import Settings
from weather_assistant_cli.logging_config import LoggerLike
from weather_assistant_cli.plugins import ToolBundle
from weather_assistant_cli.plugins.weather.prompts import WEATHER_SYSTEM_INSTRUCTIONS
from weather_assistant_cli.plugins.weather.service import (
    WeatherService,
    WeatherToolHandler,
    build_weather_follow_up_messages,
)
from weather_assistant_cli.retry import BackoffPolicy


def build_bundle(
    *,
    settings: Settings,
    http_client: httpx.AsyncClient,
    logger: LoggerLike,
) -> ToolBundle:
    """Build the built-in weather tool bundle."""

    weather_service = WeatherService(
        http_client,
        retry_policy=BackoffPolicy(
            attempts=settings.retry_attempts,
            base_delay_seconds=settings.retry_base_delay_seconds,
            with_jitter=False,
        ),
        logger=logger,
    )
    return ToolBundle(
        handler=WeatherToolHandler(weather_service),
        instructions=WEATHER_SYSTEM_INSTRUCTIONS,
        build_follow_up_messages=build_weather_follow_up_messages,
    )
