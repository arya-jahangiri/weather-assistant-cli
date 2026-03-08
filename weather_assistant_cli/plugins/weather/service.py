"""Weather service, tool handler, and follow-up guidance builder."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping, Sequence
from typing import TypeAlias, cast

import httpx
from pydantic import ValidationError

from weather_assistant_cli.logging_config import LoggerLike
from weather_assistant_cli.openai_gateway import ResponsesInputItem, SystemMessageItem
from weather_assistant_cli.plugins.weather._parsing import (
    DEFAULT_GEOCODE_RESULT_COUNT,
    QUALIFIED_GEOCODE_RESULT_COUNT,
    format_requested_location,
    normalize_optional_value,
    parse_current_weather_payload,
    parse_geocode_payload,
)
from weather_assistant_cli.plugins.weather.models import (
    CurrentWeather,
    GeocodeResult,
    GetWeatherArgs,
    WeatherErrorCode,
    WeatherLookupError,
    WeatherToolFailure,
    WeatherToolResult,
    WeatherToolSuccess,
)
from weather_assistant_cli.plugins.weather.prompts import (
    GET_WEATHER_TOOL_DESCRIPTION,
    LOCATION_RESOLUTION_REPROMPT,
)
from weather_assistant_cli.retry import BackoffPolicy, backoff_delay_seconds
from weather_assistant_cli.tools import (
    FunctionCallOutputItem,
    ToolExecutionErrorCode,
    ToolFailurePayload,
    ToolOutput,
)

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
TRANSIENT_HTTP_STATUSES = {502, 503, 504}
QueryParamValue: TypeAlias = str | int | float

RESOLUTION_ERROR_CODES = frozenset(
    {
        WeatherErrorCode.CITY_NOT_FOUND.value,
        WeatherErrorCode.LOCATION_AMBIGUOUS.value,
        WeatherErrorCode.LOCATION_QUALIFIER_MISMATCH.value,
    }
)
RESOLUTION_FAILURE_CODES = frozenset(
    {
        WeatherErrorCode.CITY_NOT_FOUND,
        WeatherErrorCode.LOCATION_AMBIGUOUS,
        WeatherErrorCode.LOCATION_QUALIFIER_MISMATCH,
    }
)
UPSTREAM_FAILURE_CODES = frozenset(
    {
        WeatherErrorCode.NETWORK_ERROR,
        WeatherErrorCode.GEOCODING_API_ERROR,
        WeatherErrorCode.WEATHER_API_ERROR,
    }
)


def lookup_failure_log_level(error_code: WeatherErrorCode) -> int:
    """Map handled weather lookup failures to an operator-facing log level."""

    if error_code in RESOLUTION_FAILURE_CODES:
        return logging.DEBUG
    if error_code in UPSTREAM_FAILURE_CODES:
        return logging.WARNING
    return logging.ERROR


def build_weather_follow_up_messages(
    outputs: Sequence[FunctionCallOutputItem],
) -> list[ResponsesInputItem]:
    """Return one clarification/repair reprompt when weather resolution fails."""

    if any(output.error_code in RESOLUTION_ERROR_CODES for output in outputs):
        retry_message: SystemMessageItem = {
            "role": "system",
            "content": LOCATION_RESOLUTION_REPROMPT,
        }
        return [retry_message]
    return []


class WeatherService:
    """Fetch and normalize weather data for tool-call consumption."""

    def __init__(
        self,
        http_client: httpx.AsyncClient,
        *,
        retry_policy: BackoffPolicy,
        logger: LoggerLike,
    ) -> None:
        """Initialize weather service dependencies and retry behavior."""

        self._http_client = http_client
        self._retry_policy = retry_policy
        self._logger = logger

    async def get_weather(
        self,
        name: str,
        *,
        admin1: str | None = None,
        country_code: str | None = None,
        logger: LoggerLike | None = None,
    ) -> WeatherToolResult:
        """Resolve a location and return typed weather tool output."""

        normalized_name = name.strip()
        normalized_admin1 = normalize_optional_value(admin1)
        normalized_country_code = normalize_optional_value(country_code)
        active_logger = logger or self._logger
        requested_location = format_requested_location(
            normalized_name,
            normalized_admin1,
            normalized_country_code,
        )

        try:
            geocode = await self._geocode(
                normalized_name,
                admin1=normalized_admin1,
                country_code=normalized_country_code,
                logger=active_logger,
            )
            weather = await self._fetch_current_weather(
                geocode.latitude,
                geocode.longitude,
                logger=active_logger,
            )
            return WeatherToolSuccess(
                name=normalized_name,
                requested_admin1=normalized_admin1,
                requested_country_code=normalized_country_code,
                resolved_name=geocode.resolved_name,
                admin1=geocode.admin1,
                country=geocode.country,
                latitude=geocode.latitude,
                longitude=geocode.longitude,
                temperature_c=weather.temperature_c,
                humidity_percent=weather.humidity_percent,
                wind_kmh=weather.wind_kmh,
                wind_direction_deg=weather.wind_direction_deg,
                weather_description=weather.weather_description,
            )
        except WeatherLookupError as exc:
            active_logger.log(
                lookup_failure_log_level(exc.code),
                "weather_lookup_failed",
                extra={
                    "query": requested_location,
                    "error_code": exc.code.value,
                    "status": "failure",
                },
            )
            return WeatherToolFailure(
                name=normalized_name,
                requested_admin1=normalized_admin1,
                requested_country_code=normalized_country_code,
                error_code=exc.code,
                error_message=exc.message,
                resolved_name=exc.resolved_name,
                admin1=exc.admin1,
                country=exc.country,
                latitude=exc.latitude,
                longitude=exc.longitude,
                candidates=exc.candidates,
            )

    async def _geocode(
        self,
        name: str,
        *,
        admin1: str | None,
        country_code: str | None,
        logger: LoggerLike,
    ) -> GeocodeResult:
        """Resolve location query to a single geocoding result."""

        geocode_count = (
            QUALIFIED_GEOCODE_RESULT_COUNT
            if admin1 is not None or country_code is not None
            else DEFAULT_GEOCODE_RESULT_COUNT
        )
        params: dict[str, str | int] = {
            "name": name,
            "count": geocode_count,
            "language": "en",
            "format": "json",
        }
        if country_code:
            params["countryCode"] = country_code

        payload = await self._get_json(
            GEOCODE_URL,
            params=params,
            upstream_error_code=WeatherErrorCode.GEOCODING_API_ERROR,
            logger=logger,
        )
        return parse_geocode_payload(
            payload,
            name=name,
            admin1=admin1,
            country_code=country_code,
        )

    async def _fetch_current_weather(
        self,
        latitude: float,
        longitude: float,
        *,
        logger: LoggerLike,
    ) -> CurrentWeather:
        """Fetch the current weather observation for resolved coordinates."""

        payload = await self._get_json(
            FORECAST_URL,
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current": (
                    "temperature_2m,relative_humidity_2m,weather_code,"
                    "wind_speed_10m,wind_direction_10m"
                ),
                "timezone": "auto",
            },
            upstream_error_code=WeatherErrorCode.WEATHER_API_ERROR,
            logger=logger,
        )
        return parse_current_weather_payload(payload, latitude=latitude, longitude=longitude)

    async def _get_json(
        self,
        url: str,
        *,
        params: Mapping[str, QueryParamValue],
        upstream_error_code: WeatherErrorCode,
        logger: LoggerLike | None = None,
    ) -> dict[str, object]:
        """Request JSON payload with transient retries and typed errors."""

        attempt = 0
        active_logger = logger or self._logger
        while True:
            try:
                response = await self._http_client.get(url, params=dict(params))
            except (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.ConnectError) as exc:
                if self._should_retry(attempt):
                    await self._sleep_before_retry(attempt, active_logger)
                    attempt += 1
                    continue
                raise WeatherLookupError(
                    WeatherErrorCode.NETWORK_ERROR,
                    "Network timeout while contacting weather APIs.",
                ) from exc
            except httpx.RequestError as exc:
                if self._should_retry(attempt):
                    await self._sleep_before_retry(attempt, active_logger)
                    attempt += 1
                    continue
                raise WeatherLookupError(
                    WeatherErrorCode.NETWORK_ERROR,
                    "Network error while contacting weather APIs.",
                ) from exc

            if response.status_code in TRANSIENT_HTTP_STATUSES and self._should_retry(attempt):
                await self._sleep_before_retry(attempt, active_logger)
                attempt += 1
                continue

            if response.status_code >= 400:
                raise WeatherLookupError(
                    upstream_error_code,
                    f"Upstream API request failed with HTTP {response.status_code}.",
                )

            try:
                payload_object = cast(object, response.json())
            except ValueError as exc:
                raise WeatherLookupError(
                    upstream_error_code,
                    "Upstream API returned invalid JSON.",
                ) from exc

            if not isinstance(payload_object, dict):
                raise WeatherLookupError(
                    upstream_error_code,
                    "Upstream API JSON payload is not an object.",
                )

            return cast(dict[str, object], payload_object)

    def _should_retry(self, attempt: int) -> bool:
        """Return True when another retry attempt is allowed."""

        return attempt < self._retry_policy.attempts

    async def _sleep_before_retry(self, attempt: int, logger: LoggerLike | None = None) -> None:
        """Sleep for one retry interval based on configured policy."""

        delay_seconds = backoff_delay_seconds(self._retry_policy, attempt)
        active_logger = logger or self._logger
        active_logger.debug(
            "weather_retry_sleep",
            extra={"attempt": attempt, "delay_seconds": delay_seconds},
        )
        await asyncio.sleep(delay_seconds)


class WeatherToolHandler:
    """Tool handler for get_weather(name, admin1?, country_code?)."""

    def __init__(self, weather_service: WeatherService) -> None:
        """Initialize handler dependencies."""

        self._weather_service = weather_service

    @property
    def name(self) -> str:
        """Return the tool name exposed to the model."""

        return "get_weather"

    @property
    def schema(self) -> dict[str, object]:
        """Return OpenAI tool schema generated from typed args model."""

        return {
            "type": "function",
            "name": self.name,
            "description": GET_WEATHER_TOOL_DESCRIPTION,
            "parameters": GetWeatherArgs.model_json_schema(),
        }

    def preview_invocation(self, arguments_json: str) -> str | None:
        """Return parsed query for progress rendering, or None if invalid."""

        try:
            args = GetWeatherArgs.model_validate_json(arguments_json)
        except ValidationError:
            return None
        return args.name

    async def run(self, arguments_json: str, *, logger: LoggerLike | None = None) -> ToolOutput:
        """Validate arguments, call weather service, and return tool output JSON."""

        try:
            args = GetWeatherArgs.model_validate_json(arguments_json)
        except ValidationError as exc:
            payload = ToolFailurePayload(
                tool_name=self.name,
                error_code=ToolExecutionErrorCode.BAD_TOOL_ARGS,
                error_message=f"Invalid tool arguments: {exc.errors()}",
            )
            return ToolOutput(
                payload_json=payload.model_dump_json(),
                error_code=ToolExecutionErrorCode.BAD_TOOL_ARGS.value,
            )

        result = await self._weather_service.get_weather(
            args.name,
            admin1=args.admin1,
            country_code=args.country_code,
            logger=logger,
        )
        error_code = result.error_code.value if isinstance(result, WeatherToolFailure) else None
        return ToolOutput(payload_json=result.model_dump_json(), error_code=error_code)
