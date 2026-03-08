"""Tests for weather-domain prompts, tool behavior, and service logic."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx
import pytest
import respx
from pydantic import ValidationError

from weather_assistant_cli.logging_config import with_context
from weather_assistant_cli.plugins.weather._parsing import (
    DEFAULT_GEOCODE_RESULT_COUNT,
    QUALIFIED_GEOCODE_RESULT_COUNT,
    parse_current_weather_payload,
    parse_geocode_payload,
)
from weather_assistant_cli.plugins.weather.models import (
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
    WEATHER_SYSTEM_INSTRUCTIONS,
)
from weather_assistant_cli.plugins.weather.service import (
    FORECAST_URL,
    GEOCODE_URL,
    WeatherService,
    WeatherToolHandler,
    build_weather_follow_up_messages,
)
from weather_assistant_cli.retry import BackoffPolicy
from weather_assistant_cli.tools import (
    FunctionCallOutputItem,
    ToolExecutionErrorCode,
)


@dataclass
class FakeWeatherService:
    """Fake weather service for weather-tool tests."""

    result: WeatherToolResult
    seen_logger: object | None = None
    seen_admin1: str | None = None
    seen_country_code: str | None = None

    async def get_weather(
        self,
        name: str,
        *,
        admin1: str | None = None,
        country_code: str | None = None,
        logger=None,
    ) -> WeatherToolResult:
        """Return preconfigured result with requested args applied."""

        self.seen_logger = logger
        self.seen_admin1 = admin1
        self.seen_country_code = country_code
        return self.result.model_copy(
            update={
                "name": name,
                "requested_admin1": admin1,
                "requested_country_code": country_code,
            }
        )


def _success_result() -> WeatherToolSuccess:
    """Build a valid weather success payload for tests."""

    return WeatherToolSuccess(
        name="",
        resolved_name="London",
        admin1="England",
        country="United Kingdom",
        latitude=51.5,
        longitude=-0.12,
        temperature_c=10.0,
        humidity_percent=70.0,
        wind_kmh=11.0,
        wind_direction_deg=220.0,
        weather_description="overcast",
    )


def _build_service(http_client: httpx.AsyncClient, logger, retries: int = 2) -> WeatherService:
    """Build a weather service with deterministic retry settings for tests."""

    return WeatherService(
        http_client,
        retry_policy=BackoffPolicy(attempts=retries, base_delay_seconds=0.001, with_jitter=False),
        logger=logger,
    )


def _newport_results() -> list[dict[str, object]]:
    """Return a mixed geocoding result set for Newport disambiguation tests."""

    return [
        {
            "name": "Newport",
            "admin1": "Kentucky",
            "country": "United States",
            "country_code": "US",
            "latitude": 39.09145,
            "longitude": -84.49578,
        },
        {
            "name": "Newport",
            "admin1": "Oregon",
            "country": "United States",
            "country_code": "US",
            "latitude": 44.63678,
            "longitude": -124.05345,
        },
        {
            "name": "Brewton",
            "admin1": "Alabama",
            "country": "United States",
            "country_code": "US",
            "latitude": 31.10518,
            "longitude": -87.07219,
        },
        {
            "name": "Newport",
            "admin1": "Oregon",
            "country": "United States",
            "country_code": "US",
            "latitude": 44.63678,
            "longitude": -124.05345,
        },
        {
            "name": "Newport",
            "admin1": "Washington",
            "country": "United States",
            "country_code": "US",
            "latitude": 48.17963,
            "longitude": -117.04326,
        },
    ]


def _springfield_ranked_results() -> list[dict[str, object]]:
    """Return many exact-name matches with the desired qualifier past the top 10."""

    results: list[dict[str, object]] = [
        {
            "name": "Springfield",
            "admin1": f"Region-{index}",
            "country": "United States",
            "latitude": 30.0 + index,
            "longitude": -90.0 - index,
        }
        for index in range(10)
    ]
    results.append(
        {
            "name": "Springfield",
            "admin1": "Oregon",
            "country": "United States",
            "latitude": 44.0462,
            "longitude": -123.0220,
        }
    )
    return results


def test_get_weather_args_trim_structured_fields() -> None:
    """Structured location args should be normalized before use."""

    args = GetWeatherArgs(name="  Newport  ", admin1="  Oregon  ", country_code=" us ")
    assert args.name == "Newport"
    assert args.admin1 == "Oregon"
    assert args.country_code == "US"


def test_get_weather_args_reject_invalid_country_code() -> None:
    """Malformed country codes should fail validation."""

    with pytest.raises(ValidationError):
        GetWeatherArgs(name="Newport", country_code="USA")


def test_weather_tool_success_requires_resolved_weather_fields() -> None:
    """Success payloads should reject impossible partial states."""

    with pytest.raises(ValidationError):
        WeatherToolSuccess(name="London")


def test_weather_tool_failure_requires_error_details() -> None:
    """Failure payloads should reject missing error metadata."""

    with pytest.raises(ValidationError):
        WeatherToolFailure(name="London", error_code=WeatherErrorCode.CITY_NOT_FOUND)


@pytest.mark.asyncio
async def test_weather_tool_returns_service_output() -> None:
    """Valid args should call service and return serialized payload."""

    handler = WeatherToolHandler(FakeWeatherService(_success_result()))

    output = await handler.run('{"name": "London", "admin1": "England", "country_code": "GB"}')

    assert output.error_code is None
    assert '"name":"London"' in output.payload_json
    assert '"requested_admin1":"England"' in output.payload_json
    assert '"requested_country_code":"GB"' in output.payload_json
    assert '"admin1":"England"' in output.payload_json


@pytest.mark.asyncio
async def test_weather_tool_invalid_args_return_generic_bad_tool_args() -> None:
    """Invalid args payload should map to generic BAD_TOOL_ARGS output."""

    handler = WeatherToolHandler(FakeWeatherService(_success_result()))

    output = await handler.run("{}")

    assert output.error_code == ToolExecutionErrorCode.BAD_TOOL_ARGS.value
    assert '"tool_name":"get_weather"' in output.payload_json
    assert '"error_code":"BAD_TOOL_ARGS"' in output.payload_json


def test_weather_tool_preview_invocation_parses_valid_args() -> None:
    """preview_invocation should return parsed query when arguments are valid."""

    handler = WeatherToolHandler(FakeWeatherService(_success_result()))

    assert (
        handler.preview_invocation('{"name": "Berlin", "admin1": "Berlin", "country_code": "DE"}')
        == "Berlin"
    )
    assert handler.preview_invocation('{"name": "London", "country_code": "GB"}') == "London"
    assert handler.preview_invocation("{}") is None


@pytest.mark.asyncio
async def test_weather_tool_surfaces_weather_failure_code() -> None:
    """Weather-domain failures should pass through their weather error code."""

    handler = WeatherToolHandler(
        FakeWeatherService(
            WeatherToolFailure(
                name="Paris",
                error_code=WeatherErrorCode.CITY_NOT_FOUND,
                error_message="No location found.",
            )
        )
    )

    output = await handler.run('{"name": "Paris"}')

    assert output.error_code == WeatherErrorCode.CITY_NOT_FOUND.value
    assert '"error_code":"CITY_NOT_FOUND"' in output.payload_json


@pytest.mark.asyncio
async def test_weather_tool_forwards_logger_to_service(logger) -> None:
    """Handler should pass the turn logger through to the weather service."""

    fake_service = FakeWeatherService(_success_result())
    handler = WeatherToolHandler(fake_service)
    turn_logger = with_context(logger, {"run_id": "run-1", "turn_id": "turn-1"})

    await handler.run(
        '{"name": "Lisbon", "admin1": "Lisbon", "country_code": "PT"}',
        logger=turn_logger,
    )

    assert fake_service.seen_logger is turn_logger
    assert fake_service.seen_admin1 == "Lisbon"
    assert fake_service.seen_country_code == "PT"


def test_weather_tool_schema_description_uses_shared_guidance() -> None:
    """Tool schema should keep the key structured-argument guidance."""

    handler = WeatherToolHandler(FakeWeatherService(_success_result()))
    description = str(handler.schema["description"])

    assert "Get current weather for one location." in description
    assert "Use structured args" in description
    assert "optional admin1, optional country_code" in description
    assert "ISO alpha-2 code" in description
    assert "Do not put state, province, or country text into the name field" in description
    assert "Use admin1 and country_code when the user supplies them" in description
    assert "Pass canonical place names and full region names" in description
    assert "If a shorthand location is genuinely ambiguous" in description
    assert description == GET_WEATHER_TOOL_DESCRIPTION


def test_get_weather_args_schema_prefers_canonical_place_names() -> None:
    """Field descriptions should steer the model toward canonical tool args."""

    schema = GetWeatherArgs.model_json_schema()
    properties = schema["properties"]

    assert "Los Angeles" in str(properties["name"]["description"])
    assert "'LA'" in str(properties["name"]["description"])
    assert "Keep state, province, region, and country text out of this field." in str(
        properties["name"]["description"]
    )
    assert "California" in str(properties["admin1"]["description"])
    assert "District of Columbia" in str(properties["admin1"]["description"])
    assert "needed to disambiguate places with the same name" in str(
        properties["admin1"]["description"]
    )
    assert "needed to disambiguate places with the same name" in str(
        properties["country_code"]["description"]
    )


def test_system_prompt_and_follow_up_guidance_cover_required_behavior() -> None:
    """Prompt guidance should cover scope, tool calling, and disambiguation."""

    assert "Only handle weather and current conditions questions" in WEATHER_SYSTEM_INSTRUCTIONS
    assert "one tool call per location" in WEATHER_SYSTEM_INSTRUCTIONS
    assert "Normalize obvious shorthand or nicknames" in WEATHER_SYSTEM_INSTRUCTIONS
    assert "For standard bare city requests" in WEATHER_SYSTEM_INSTRUCTIONS
    assert "shorthand location is genuinely ambiguous" in WEATHER_SYSTEM_INSTRUCTIONS
    assert "answer directly from that result" in WEATHER_SYSTEM_INSTRUCTIONS
    assert (
        "include temperature, conditions, humidity, wind speed, and wind direction"
        in WEATHER_SYSTEM_INSTRUCTIONS
    )
    assert "Convert wind_direction_deg into a human-readable compass direction" in (
        WEATHER_SYSTEM_INSTRUCTIONS
    )
    assert "show only city names" in WEATHER_SYSTEM_INSTRUCTIONS
    assert "Use admin1 and country_code when the user supplies them" in WEATHER_SYSTEM_INSTRUCTIONS
    assert "do not include state/region names, country names, or country codes" in (
        WEATHER_SYSTEM_INSTRUCTIONS
    )
    assert "location-resolution error" in LOCATION_RESOLUTION_REPROMPT


def test_system_prompt_restores_spec_weather_formatting_details() -> None:
    """Prompt should protect the spec-required answer shape from future simplification."""

    assert "Here's the current weather across the cities:" in WEATHER_SYSTEM_INSTRUCTIONS
    assert "humidity at <x>%" in WEATHER_SYSTEM_INSTRUCTIONS
    assert "wind around <y> km/h from <direction>." in WEATHER_SYSTEM_INSTRUCTIONS
    assert "Use a blank line between city blocks for readability." in WEATHER_SYSTEM_INSTRUCTIONS
    assert "Align all hyphens vertically by padding city names with spaces." in (
        WEATHER_SYSTEM_INSTRUCTIONS
    )
    assert "End with one short comparison sentence" in WEATHER_SYSTEM_INSTRUCTIONS
    assert '"Want to check anywhere else?"' in WEATHER_SYSTEM_INSTRUCTIONS
    assert "not state labels, country codes, country labels, or region labels" in (
        WEATHER_SYSTEM_INSTRUCTIONS
    )


def test_build_weather_follow_up_messages_only_for_resolution_errors() -> None:
    """Follow-up guidance should be added only for location-resolution failures."""

    resolution_outputs = [
        FunctionCallOutputItem(
            tool_name="get_weather",
            call_id="call_1",
            output_json='{"ok": false, "error_code": "LOCATION_AMBIGUOUS"}',
            error_code=WeatherErrorCode.LOCATION_AMBIGUOUS.value,
        )
    ]
    network_outputs = [
        FunctionCallOutputItem(
            tool_name="get_weather",
            call_id="call_1",
            output_json='{"ok": false, "error_code": "NETWORK_ERROR"}',
            error_code=WeatherErrorCode.NETWORK_ERROR.value,
        )
    ]

    messages = build_weather_follow_up_messages(resolution_outputs)
    assert len(messages) == 1
    assert messages[0]["role"] == "system"
    assert "LOCATION_AMBIGUOUS" in messages[0]["content"]
    assert build_weather_follow_up_messages(network_outputs) == []


@pytest.mark.asyncio
@respx.mock
async def test_get_weather_success_with_float_humidity(logger) -> None:
    """Happy path should return normalized weather payload with float humidity."""

    respx.get(GEOCODE_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "name": "London",
                        "admin1": "England",
                        "country": "United Kingdom",
                        "latitude": 51.5072,
                        "longitude": -0.1276,
                    }
                ]
            },
        )
    )
    respx.get(FORECAST_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "current": {
                    "temperature_2m": 12.4,
                    "relative_humidity_2m": 78.5,
                    "weather_code": 63,
                    "wind_speed_10m": 15.0,
                    "wind_direction_10m": 225,
                }
            },
        )
    )

    async with httpx.AsyncClient() as http_client:
        service = _build_service(http_client, logger)
        result = await service.get_weather("London")

    assert result.ok is True
    assert result.name == "London"
    assert result.humidity_percent == pytest.approx(78.5)
    assert result.resolved_name == "London"
    assert result.admin1 == "England"
    assert result.country == "United Kingdom"
    geocode_request = respx.calls[0].request
    assert geocode_request.url.params["count"] == str(DEFAULT_GEOCODE_RESULT_COUNT)


@pytest.mark.asyncio
@respx.mock
async def test_city_not_found_returns_typed_error(logger) -> None:
    """Empty geocoding results should map to CITY_NOT_FOUND."""

    respx.get(GEOCODE_URL).mock(return_value=httpx.Response(200, json={"results": []}))

    async with httpx.AsyncClient() as http_client:
        service = _build_service(http_client, logger)
        result = await service.get_weather("NotARealPlace")

    assert result.ok is False
    assert result.error_code == WeatherErrorCode.CITY_NOT_FOUND


@pytest.mark.asyncio
@respx.mock
async def test_bare_city_uses_first_exact_name_candidate(logger) -> None:
    """Unqualified city lookups should trust the first exact-name geocode result."""

    geocode_route = respx.get(GEOCODE_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "name": "London",
                        "admin1": "England",
                        "country": "United Kingdom",
                        "country_code": "GB",
                        "latitude": 51.5072,
                        "longitude": -0.1276,
                    },
                    {
                        "name": "London",
                        "admin1": "Kentucky",
                        "country": "United States",
                        "country_code": "US",
                        "latitude": 37.12898,
                        "longitude": -84.08326,
                    },
                ]
            },
        )
    )
    respx.get(FORECAST_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "current": {
                    "temperature_2m": 12,
                    "relative_humidity_2m": 79,
                    "weather_code": 3,
                    "wind_speed_10m": 14,
                    "wind_direction_10m": 215,
                }
            },
        )
    )

    async with httpx.AsyncClient() as http_client:
        service = _build_service(http_client, logger)
        result = await service.get_weather("London")

    assert result.ok is True
    assert result.resolved_name == "London"
    assert result.admin1 == "England"
    assert result.country == "United Kingdom"
    request = geocode_route.calls[0].request
    assert request.url.params["count"] == str(DEFAULT_GEOCODE_RESULT_COUNT)
    assert "countryCode" not in request.url.params


@pytest.mark.asyncio
@respx.mock
async def test_country_filtered_city_without_admin1_uses_first_exact_name_candidate(logger) -> None:
    """Country-only lookups should still trust ranking instead of forcing clarification."""

    geocode_route = respx.get(GEOCODE_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "name": "Los Angeles",
                        "admin1": "California",
                        "country": "United States",
                        "country_code": "US",
                        "latitude": 34.0522,
                        "longitude": -118.2437,
                    },
                    {
                        "name": "Los Angeles",
                        "admin1": "Texas",
                        "country": "United States",
                        "country_code": "US",
                        "latitude": 27.7639,
                        "longitude": -97.7422,
                    },
                ]
            },
        )
    )
    respx.get(FORECAST_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "current": {
                    "temperature_2m": 17,
                    "relative_humidity_2m": 22,
                    "weather_code": 0,
                    "wind_speed_10m": 20,
                    "wind_direction_10m": 360,
                }
            },
        )
    )

    async with httpx.AsyncClient() as http_client:
        service = _build_service(http_client, logger)
        result = await service.get_weather("Los Angeles", country_code="US")

    assert result.ok is True
    assert result.admin1 == "California"
    assert result.country == "United States"
    request = geocode_route.calls[0].request
    assert request.url.params["countryCode"] == "US"
    assert request.url.params["count"] == str(QUALIFIED_GEOCODE_RESULT_COUNT)


@pytest.mark.asyncio
@respx.mock
async def test_country_filtered_city_prefers_top_ranked_result_within_country(logger) -> None:
    """Country-only lookups should choose the top-ranked exact-name result within that country."""

    respx.get(GEOCODE_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "name": "Berlin",
                        "admin1": "Berlin",
                        "country": "Germany",
                        "country_code": "DE",
                        "latitude": 52.52,
                        "longitude": 13.405,
                    },
                    {
                        "name": "Berlin",
                        "admin1": "Schleswig-Holstein",
                        "country": "Germany",
                        "country_code": "DE",
                        "latitude": 54.15,
                        "longitude": 10.4,
                    },
                ]
            },
        )
    )
    respx.get(FORECAST_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "current": {
                    "temperature_2m": 8,
                    "relative_humidity_2m": 50,
                    "weather_code": 1,
                    "wind_speed_10m": 13,
                    "wind_direction_10m": 90,
                }
            },
        )
    )

    async with httpx.AsyncClient() as http_client:
        service = _build_service(http_client, logger)
        result = await service.get_weather("Berlin", country_code="DE")

    assert result.ok is True
    assert result.admin1 == "Berlin"
    assert result.country == "Germany"


@pytest.mark.asyncio
@respx.mock
async def test_exact_name_candidates_return_ambiguity_with_deduped_matches(logger) -> None:
    """Admin1-qualified lookups should still surface ambiguity after local filtering."""

    geocode_route = respx.get(GEOCODE_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "name": "Berlin",
                        "admin1": "Berlin",
                        "country": "Germany",
                        "country_code": "DE",
                        "latitude": 52.52,
                        "longitude": 13.405,
                    },
                    {
                        "name": "Berlin",
                        "admin1": "Berlin",
                        "country": "Germany",
                        "country_code": "DE",
                        "latitude": 52.61,
                        "longitude": 13.51,
                    },
                    {
                        "name": "Postdam",
                        "admin1": "Brandenburg",
                        "country": "Germany",
                        "country_code": "DE",
                        "latitude": 52.3906,
                        "longitude": 13.0645,
                    },
                ]
            },
        )
    )

    async with httpx.AsyncClient() as http_client:
        service = _build_service(http_client, logger)
        result = await service.get_weather("Berlin", admin1="Berlin", country_code="DE")

    assert result.ok is False
    assert result.error_code == WeatherErrorCode.LOCATION_AMBIGUOUS
    assert result.requested_admin1 == "Berlin"
    assert result.requested_country_code == "DE"
    assert result.candidates is not None
    assert [(candidate.name, candidate.admin1) for candidate in result.candidates] == [
        ("Berlin", "Berlin"),
        ("Berlin", "Berlin"),
    ]

    request = geocode_route.calls[0].request
    assert request.url.params["name"] == "Berlin"
    assert request.url.params["countryCode"] == "DE"
    assert request.url.params["count"] == str(QUALIFIED_GEOCODE_RESULT_COUNT)
    assert "admin1" not in request.url.params


@pytest.mark.asyncio
@respx.mock
async def test_admin1_filter_resolves_exact_name_candidate(logger) -> None:
    """A supplied admin1 qualifier should filter exact-name candidates locally."""

    geocode_route = respx.get(GEOCODE_URL).mock(
        return_value=httpx.Response(200, json={"results": _newport_results()})
    )
    respx.get(FORECAST_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "current": {
                    "temperature_2m": 11,
                    "relative_humidity_2m": 81,
                    "weather_code": 3,
                    "wind_speed_10m": 18,
                    "wind_direction_10m": 250,
                }
            },
        )
    )

    async with httpx.AsyncClient() as http_client:
        service = _build_service(http_client, logger)
        result = await service.get_weather("Newport", admin1="Oregon", country_code="US")

    assert result.ok is True
    assert result.name == "Newport"
    assert result.requested_admin1 == "Oregon"
    assert result.requested_country_code == "US"
    assert result.resolved_name == "Newport"
    assert result.admin1 == "Oregon"

    request = geocode_route.calls[0].request
    assert request.url.params["name"] == "Newport"
    assert request.url.params["countryCode"] == "US"
    assert request.url.params["count"] == str(QUALIFIED_GEOCODE_RESULT_COUNT)
    assert "admin1" not in request.url.params


@pytest.mark.asyncio
@respx.mock
async def test_admin1_lookup_uses_larger_count_and_finds_result_past_top_ten(logger) -> None:
    """Qualified lookups should use a larger count so local filtering sees the target result."""

    def geocode_response(request: httpx.Request) -> httpx.Response:
        count = int(request.url.params["count"])
        payload = {"results": _springfield_ranked_results()[:count]}
        return httpx.Response(200, json=payload)

    geocode_route = respx.get(GEOCODE_URL).mock(side_effect=geocode_response)
    respx.get(FORECAST_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "current": {
                    "temperature_2m": 9,
                    "relative_humidity_2m": 68,
                    "weather_code": 3,
                    "wind_speed_10m": 12,
                    "wind_direction_10m": 180,
                }
            },
        )
    )

    async with httpx.AsyncClient() as http_client:
        service = _build_service(http_client, logger)
        result = await service.get_weather("Springfield", admin1="Oregon")

    assert result.ok is True
    assert result.admin1 == "Oregon"
    request = geocode_route.calls[0].request
    assert request.url.params["count"] == str(QUALIFIED_GEOCODE_RESULT_COUNT)


@pytest.mark.asyncio
@respx.mock
async def test_admin1_mismatch_returns_typed_alternatives(logger) -> None:
    """A non-matching admin1 should return alternatives instead of guessing."""

    respx.get(GEOCODE_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "name": "Newport",
                        "admin1": "Kentucky",
                        "country": "United States",
                        "country_code": "US",
                        "latitude": 39.09145,
                        "longitude": -84.49578,
                    },
                    {
                        "name": "Newport",
                        "admin1": "Washington",
                        "country": "United States",
                        "country_code": "US",
                        "latitude": 48.17963,
                        "longitude": -117.04326,
                    },
                ]
            },
        )
    )

    async with httpx.AsyncClient() as http_client:
        service = _build_service(http_client, logger)
        result = await service.get_weather("Newport", admin1="Oregon", country_code="US")

    assert result.ok is False
    assert result.error_code == WeatherErrorCode.LOCATION_QUALIFIER_MISMATCH
    assert result.requested_admin1 == "Oregon"
    assert result.candidates is not None
    assert [(candidate.name, candidate.admin1) for candidate in result.candidates] == [
        ("Newport", "Kentucky"),
        ("Newport", "Washington"),
    ]
    assert "admin1='Oregon'" in result.error_message
    assert "country_code='US'" in result.error_message


@pytest.mark.asyncio
@respx.mock
async def test_country_code_filter_resolves_exact_name_candidate_locally(logger) -> None:
    """A supplied country code should filter exact-name candidates locally."""

    respx.get(GEOCODE_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "name": "London",
                        "admin1": "Ontario",
                        "country": "Canada",
                        "country_code": "CA",
                        "latitude": 42.9834,
                        "longitude": -81.233,
                    },
                    {
                        "name": "London",
                        "admin1": "Kentucky",
                        "country": "United States",
                        "country_code": "US",
                        "latitude": 37.12898,
                        "longitude": -84.08326,
                    },
                ]
            },
        )
    )
    respx.get(FORECAST_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "current": {
                    "temperature_2m": 7,
                    "relative_humidity_2m": 70,
                    "weather_code": 3,
                    "wind_speed_10m": 11,
                    "wind_direction_10m": 200,
                }
            },
        )
    )

    async with httpx.AsyncClient() as http_client:
        service = _build_service(http_client, logger)
        result = await service.get_weather("London", country_code="CA")

    assert result.ok is True
    assert result.country == "Canada"
    assert result.requested_country_code == "CA"


@pytest.mark.asyncio
@respx.mock
async def test_country_code_mismatch_returns_typed_alternatives(logger) -> None:
    """A non-matching country code should return alternatives instead of guessing."""

    respx.get(GEOCODE_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "name": "London",
                        "admin1": "England",
                        "country": "United Kingdom",
                        "country_code": "GB",
                        "latitude": 51.5072,
                        "longitude": -0.1276,
                    },
                    {
                        "name": "London",
                        "admin1": "Kentucky",
                        "country": "United States",
                        "country_code": "US",
                        "latitude": 37.12898,
                        "longitude": -84.08326,
                    },
                ]
            },
        )
    )

    async with httpx.AsyncClient() as http_client:
        service = _build_service(http_client, logger)
        result = await service.get_weather("London", country_code="CA")

    assert result.ok is False
    assert result.error_code == WeatherErrorCode.LOCATION_QUALIFIER_MISMATCH
    assert result.requested_country_code == "CA"
    assert result.candidates is not None
    assert [(candidate.name, candidate.country) for candidate in result.candidates] == [
        ("London", "United Kingdom"),
        ("London", "United States"),
    ]
    assert "country_code='CA'" in result.error_message


@pytest.mark.asyncio
@respx.mock
async def test_missing_country_code_during_country_filtered_lookup_maps_to_geocoding_error(
    logger,
) -> None:
    """Country-filtered lookups should reject exact-name rows without country_code."""

    respx.get(GEOCODE_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "name": "London",
                        "admin1": "Ontario",
                        "country": "Canada",
                        "latitude": 42.9834,
                        "longitude": -81.233,
                    }
                ]
            },
        )
    )

    async with httpx.AsyncClient() as http_client:
        service = _build_service(http_client, logger)
        result = await service.get_weather("London", country_code="CA")

    assert result.ok is False
    assert result.error_code == WeatherErrorCode.GEOCODING_API_ERROR


@pytest.mark.asyncio
@respx.mock
async def test_invalid_country_code_during_country_filtered_lookup_maps_to_geocoding_error(
    logger,
) -> None:
    """Country-filtered lookups should reject malformed upstream country_code values."""

    respx.get(GEOCODE_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "name": "London",
                        "admin1": "Ontario",
                        "country": "Canada",
                        "country_code": "CAN",
                        "latitude": 42.9834,
                        "longitude": -81.233,
                    }
                ]
            },
        )
    )

    async with httpx.AsyncClient() as http_client:
        service = _build_service(http_client, logger)
        result = await service.get_weather("London", country_code="CA")

    assert result.ok is False
    assert result.error_code == WeatherErrorCode.GEOCODING_API_ERROR


@pytest.mark.asyncio
@respx.mock
async def test_retries_only_transient_status_codes(logger) -> None:
    """HTTP 502 should retry and then succeed on subsequent attempt."""

    respx.get(GEOCODE_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "name": "Paris",
                        "country": "France",
                        "latitude": 48.8566,
                        "longitude": 2.3522,
                    }
                ]
            },
        )
    )
    forecast_route = respx.get(FORECAST_URL).mock(
        side_effect=[
            httpx.Response(502, json={"reason": "bad gateway"}),
            httpx.Response(
                200,
                json={
                    "current": {
                        "temperature_2m": 14,
                        "relative_humidity_2m": 65,
                        "weather_code": 2,
                        "wind_speed_10m": 10,
                        "wind_direction_10m": 250,
                    }
                },
            ),
        ]
    )

    async with httpx.AsyncClient() as http_client:
        service = _build_service(http_client, logger)
        result = await service.get_weather("Paris")

    assert forecast_route.call_count == 2
    assert result.ok is True


@pytest.mark.asyncio
@respx.mock
async def test_does_not_retry_non_transient_4xx_errors(logger) -> None:
    """HTTP 400 should fail immediately without retries."""

    respx.get(GEOCODE_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "name": "Berlin",
                        "country": "Germany",
                        "latitude": 52.52,
                        "longitude": 13.405,
                    }
                ]
            },
        )
    )
    forecast_route = respx.get(FORECAST_URL).mock(
        return_value=httpx.Response(400, json={"error": "bad request"})
    )

    async with httpx.AsyncClient() as http_client:
        service = _build_service(http_client, logger)
        result = await service.get_weather("Berlin")

    assert forecast_route.call_count == 1
    assert result.ok is False
    assert result.error_code == WeatherErrorCode.WEATHER_API_ERROR


@pytest.mark.asyncio
@respx.mock
async def test_retries_on_timeout_and_recovers(logger) -> None:
    """Read timeout should retry and eventually succeed when upstream recovers."""

    respx.get(GEOCODE_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "name": "Madrid",
                        "country": "Spain",
                        "latitude": 40.4168,
                        "longitude": -3.7038,
                    }
                ]
            },
        )
    )
    forecast_route = respx.get(FORECAST_URL).mock(
        side_effect=[
            httpx.ReadTimeout("timed out"),
            httpx.Response(
                200,
                json={
                    "current": {
                        "temperature_2m": 18,
                        "relative_humidity_2m": 40,
                        "weather_code": 1,
                        "wind_speed_10m": 8,
                        "wind_direction_10m": 140,
                    }
                },
            ),
        ]
    )

    async with httpx.AsyncClient() as http_client:
        service = _build_service(http_client, logger)
        result = await service.get_weather("Madrid")

    assert forecast_route.call_count == 2
    assert result.ok is True


@pytest.mark.asyncio
@respx.mock
async def test_network_error_after_retries_returns_typed_failure(logger) -> None:
    """Repeated network failures should return NETWORK_ERROR after retries are exhausted."""

    request = httpx.Request("GET", GEOCODE_URL)
    respx.get(GEOCODE_URL).mock(side_effect=httpx.ConnectError("dns failure", request=request))

    async with httpx.AsyncClient() as http_client:
        service = _build_service(http_client, logger)
        result = await service.get_weather("Anytown")

    assert result.ok is False
    assert result.error_code == WeatherErrorCode.NETWORK_ERROR


@pytest.mark.asyncio
@respx.mock
async def test_geocode_malformed_payload_maps_to_geocoding_error(logger) -> None:
    """Malformed exact-name results should return GEOCODING_API_ERROR."""

    respx.get(GEOCODE_URL).mock(
        return_value=httpx.Response(200, json={"results": [{"name": "Paris"}]})
    )

    async with httpx.AsyncClient() as http_client:
        service = _build_service(http_client, logger, retries=1)
        result = await service.get_weather("Paris")

    assert result.ok is False
    assert result.error_code == WeatherErrorCode.GEOCODING_API_ERROR


@pytest.mark.asyncio
@respx.mock
async def test_weather_malformed_payload_maps_to_weather_error(logger) -> None:
    """Malformed weather payload should return WEATHER_API_ERROR."""

    respx.get(GEOCODE_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "name": "Rome",
                        "country": "Italy",
                        "latitude": 41.9,
                        "longitude": 12.5,
                    }
                ]
            },
        )
    )
    respx.get(FORECAST_URL).mock(
        return_value=httpx.Response(200, json={"current": {"temperature_2m": "bad"}})
    )

    async with httpx.AsyncClient() as http_client:
        service = _build_service(http_client, logger, retries=1)
        result = await service.get_weather("Rome")

    assert result.ok is False
    assert result.error_code == WeatherErrorCode.WEATHER_API_ERROR


@pytest.mark.asyncio
@respx.mock
async def test_retry_logs_include_turn_context(
    logger,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Transient retry logs should include the active turn context."""

    caplog.set_level(logging.DEBUG, logger="test")
    turn_logger = with_context(logger, {"run_id": "run-1", "turn_id": "turn-1"})
    respx.get(GEOCODE_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "name": "Paris",
                        "country": "France",
                        "latitude": 48.8566,
                        "longitude": 2.3522,
                    }
                ]
            },
        )
    )
    respx.get(FORECAST_URL).mock(
        side_effect=[
            httpx.Response(502, json={"reason": "bad gateway"}),
            httpx.Response(
                200,
                json={
                    "current": {
                        "temperature_2m": 14,
                        "relative_humidity_2m": 65,
                        "weather_code": 2,
                        "wind_speed_10m": 10,
                        "wind_direction_10m": 250,
                    }
                },
            ),
        ]
    )

    async with httpx.AsyncClient() as http_client:
        service = _build_service(http_client, logger)
        result = await service.get_weather("Paris", logger=turn_logger)

    assert result.ok is True
    record = next(
        record for record in caplog.records if record.getMessage() == "weather_retry_sleep"
    )
    assert record.run_id == "run-1"
    assert record.turn_id == "turn-1"
    assert record.attempt == 0
    assert record.delay_seconds == pytest.approx(0.001)


@pytest.mark.asyncio
@respx.mock
async def test_failure_logs_include_turn_context(
    logger,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Resolution failures should log at debug level with the active turn context."""

    caplog.set_level(logging.DEBUG, logger="test")
    turn_logger = with_context(logger, {"run_id": "run-1", "turn_id": "turn-1"})
    respx.get(GEOCODE_URL).mock(return_value=httpx.Response(200, json={"results": []}))

    async with httpx.AsyncClient() as http_client:
        service = _build_service(http_client, logger)
        result = await service.get_weather(
            "NotARealPlace",
            admin1="Nowhere",
            country_code="US",
            logger=turn_logger,
        )

    assert result.ok is False
    record = next(
        record for record in caplog.records if record.getMessage() == "weather_lookup_failed"
    )
    assert record.levelno == logging.DEBUG
    assert record.run_id == "run-1"
    assert record.turn_id == "turn-1"
    assert record.query == "NotARealPlace, Nowhere, US"


@pytest.mark.asyncio
@respx.mock
async def test_upstream_lookup_failure_logs_at_warning(
    logger,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Upstream failures should stay visible as warnings with turn context."""

    caplog.set_level(logging.WARNING, logger="test")
    turn_logger = with_context(logger, {"run_id": "run-1", "turn_id": "turn-1"})
    request = httpx.Request("GET", GEOCODE_URL)
    respx.get(GEOCODE_URL).mock(side_effect=httpx.ConnectError("dns failure", request=request))

    async with httpx.AsyncClient() as http_client:
        service = _build_service(http_client, logger, retries=0)
        result = await service.get_weather("Anytown", logger=turn_logger)

    assert result.ok is False
    record = next(
        record for record in caplog.records if record.getMessage() == "weather_lookup_failed"
    )
    assert record.levelno == logging.WARNING
    assert record.run_id == "run-1"
    assert record.turn_id == "turn-1"
    assert record.query == "Anytown"
    assert record.error_code == WeatherErrorCode.NETWORK_ERROR.value


def test_parser_helpers_reject_bad_types() -> None:
    """Parser helpers should reject invalid payload types through typed errors."""

    with pytest.raises(WeatherLookupError):
        parse_geocode_payload({"results": [True]}, name="London")

    with pytest.raises(WeatherLookupError):
        parse_current_weather_payload(
            {
                "current": {
                    "temperature_2m": True,
                    "relative_humidity_2m": 80,
                    "weather_code": 2,
                    "wind_speed_10m": 10,
                    "wind_direction_10m": 200,
                }
            },
            latitude=0.0,
            longitude=0.0,
        )


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("name", None),
        ("name", True),
        ("name", 42),
        ("name", {"value": "London"}),
        ("name", "   "),
        ("country", None),
        ("country", True),
        ("country", 42),
        ("country", {"value": "United Kingdom"}),
        ("country", "   "),
        ("admin1", True),
        ("admin1", 42),
        ("admin1", {"value": "England"}),
    ],
)
def test_parse_geocode_payload_rejects_malformed_text_fields(
    field_name: str,
    field_value: object,
) -> None:
    """Geocoder text fields should reject non-string and empty values."""

    payload = {
        "results": [
            {
                "name": "London",
                "admin1": "England",
                "country": "United Kingdom",
                "latitude": 51.5072,
                "longitude": -0.1276,
            }
        ]
    }
    payload["results"][0][field_name] = field_value

    with pytest.raises(WeatherLookupError) as exc_info:
        parse_geocode_payload(payload, name="London")

    assert exc_info.value.code == WeatherErrorCode.GEOCODING_API_ERROR


def test_parse_geocode_payload_matches_accented_and_punctuated_names() -> None:
    """Canonicalized exact-name matching should ignore accents and punctuation differences."""

    result = parse_geocode_payload(
        {
            "results": [
                {
                    "name": "A-Coruna",
                    "country": "Spain",
                    "country_code": "ES",
                    "latitude": 43.3623,
                    "longitude": -8.4115,
                }
            ]
        },
        name="A Coruña",
    )

    assert result.resolved_name == "A-Coruna"


def test_parse_geocode_payload_still_rejects_clearly_different_names() -> None:
    """Canonicalization should not broaden matching into different place names."""

    with pytest.raises(WeatherLookupError) as exc_info:
        parse_geocode_payload(
            {
                "results": [
                    {
                        "name": "San Jose",
                        "country": "United States",
                        "country_code": "US",
                        "latitude": 37.3382,
                        "longitude": -121.8863,
                    }
                ]
            },
            name="San Juan",
        )

    assert exc_info.value.code == WeatherErrorCode.CITY_NOT_FOUND


@pytest.mark.parametrize(
    ("weather_code", "should_succeed"),
    [
        (2, True),
        ("2", True),
        ("+2", True),
        ("02", True),
        (2.0, True),
        (2.9, False),
        (float("nan"), False),
        (float("inf"), False),
        (float("-inf"), False),
        ("2.0", False),
        ("2.9", False),
        (True, False),
        (None, False),
    ],
)
def test_parse_current_weather_payload_requires_exact_integer_weather_code(
    weather_code: object,
    should_succeed: bool,
) -> None:
    """Weather code should accept only integer values or integer strings."""

    payload = {
        "current": {
            "temperature_2m": 12.4,
            "relative_humidity_2m": 78.5,
            "weather_code": weather_code,
            "wind_speed_10m": 15.0,
            "wind_direction_10m": 225,
        }
    }

    if should_succeed:
        result = parse_current_weather_payload(payload, latitude=51.5072, longitude=-0.1276)
        assert result.weather_code == 2
        return

    with pytest.raises(WeatherLookupError) as exc_info:
        parse_current_weather_payload(payload, latitude=51.5072, longitude=-0.1276)

    assert exc_info.value.code == WeatherErrorCode.WEATHER_API_ERROR
