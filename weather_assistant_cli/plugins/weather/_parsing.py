"""Geocoding and weather payload parsing with typed coercion helpers."""

from __future__ import annotations

import re
import unicodedata
from typing import cast

from weather_assistant_cli.plugins.weather.models import (
    WEATHER_CODE_MAP,
    CurrentWeather,
    GeocodeCandidate,
    GeocodeResult,
    WeatherErrorCode,
    WeatherLookupError,
)

INTEGER_PATTERN = re.compile(r"[+-]?\d+")
DEFAULT_GEOCODE_RESULT_COUNT = 10
QUALIFIED_GEOCODE_RESULT_COUNT = 100


def parse_geocode_payload(
    payload: dict[str, object],
    *,
    name: str,
    admin1: str | None = None,
    country_code: str | None = None,
) -> GeocodeResult:
    """Parse geocoding payload and resolve one exact-name location candidate."""

    raw_results = payload.get("results")
    if not isinstance(raw_results, list) or not raw_results:
        raise WeatherLookupError(
            WeatherErrorCode.CITY_NOT_FOUND,
            f"No location found for '{format_requested_location(name, admin1, country_code)}'.",
        )
    results_list = cast(list[object], raw_results)

    requested_location = format_requested_location(name, admin1, country_code)
    exact_name_candidates: list[GeocodeResult] = []
    seen_candidates: set[tuple[str, str | None, str, str | None, float, float]] = set()
    requested_name = _canonicalize_match_text(name)
    requested_admin1 = _canonicalize_match_text(admin1) if admin1 is not None else None
    requested_country_code = (
        _coerce_optional_country_code(country_code, "country_code", required=True)
        if country_code is not None
        else None
    )

    for result_obj in results_list:
        if not isinstance(result_obj, dict):
            raise WeatherLookupError(
                WeatherErrorCode.GEOCODING_API_ERROR,
                "Geocoding returned an unexpected payload shape.",
            )
        result = cast(dict[str, object], result_obj)

        try:
            resolved_name = _coerce_str(result.get("name"), "name")
        except ValueError as exc:
            raise WeatherLookupError(
                WeatherErrorCode.GEOCODING_API_ERROR,
                "Geocoding returned an unexpected payload shape.",
            ) from exc

        if _canonicalize_match_text(resolved_name) != requested_name:
            continue

        try:
            candidate_admin1 = _coerce_optional_str(result.get("admin1"), "admin1")
            country = _coerce_str(result.get("country"), "country")
            candidate_country_code = _coerce_optional_country_code(
                result.get("country_code"),
                "country_code",
                required=requested_country_code is not None,
            )
            latitude = _coerce_float(result.get("latitude"), "latitude")
            longitude = _coerce_float(result.get("longitude"), "longitude")
        except ValueError as exc:
            raise WeatherLookupError(
                WeatherErrorCode.GEOCODING_API_ERROR,
                "Geocoding returned an unexpected payload shape.",
            ) from exc

        candidate = GeocodeResult(
            resolved_name=resolved_name,
            admin1=candidate_admin1,
            country=country,
            country_code=candidate_country_code,
            latitude=latitude,
            longitude=longitude,
        )
        dedupe_key = (
            _canonicalize_match_text(candidate.resolved_name),
            _canonicalize_optional_match_text(candidate.admin1),
            _canonicalize_match_text(candidate.country),
            candidate.country_code,
            candidate.latitude,
            candidate.longitude,
        )
        if dedupe_key in seen_candidates:
            continue
        seen_candidates.add(dedupe_key)
        exact_name_candidates.append(candidate)

    if not exact_name_candidates:
        raise WeatherLookupError(
            WeatherErrorCode.CITY_NOT_FOUND,
            f"No location found for '{requested_location}'.",
        )

    filtered_candidates = exact_name_candidates
    if requested_admin1 is not None:
        filtered_candidates = [
            candidate
            for candidate in filtered_candidates
            if candidate.admin1 is not None
            and _canonicalize_match_text(candidate.admin1) == requested_admin1
        ]
    if requested_country_code is not None:
        filtered_candidates = [
            candidate
            for candidate in filtered_candidates
            if candidate.country_code == requested_country_code
        ]
    if (
        requested_admin1 is not None or requested_country_code is not None
    ) and not filtered_candidates:
        raise WeatherLookupError(
            WeatherErrorCode.LOCATION_QUALIFIER_MISMATCH,
            (
                f"Found locations named '{name}', but none matched the requested qualifier(s): "
                f"{_format_requested_qualifiers(admin1, requested_country_code)}."
            ),
            candidates=_as_candidates(exact_name_candidates),
        )

    # When no admin1 is supplied, trust Open-Meteo's ranking after any country filtering.
    # This keeps natural bare-city requests working even if the model inferred a country code.
    if requested_admin1 is None:
        return filtered_candidates[0]

    if len(filtered_candidates) == 1:
        return filtered_candidates[0]

    raise WeatherLookupError(
        WeatherErrorCode.LOCATION_AMBIGUOUS,
        f"Multiple locations matched '{requested_location}'.",
        candidates=_as_candidates(filtered_candidates),
    )


def parse_current_weather_payload(
    payload: dict[str, object],
    *,
    latitude: float,
    longitude: float,
) -> CurrentWeather:
    """Parse forecast payload and extract normalized current weather values."""

    raw_current = payload.get("current")
    if not isinstance(raw_current, dict):
        raise WeatherLookupError(
            WeatherErrorCode.WEATHER_API_ERROR,
            "Weather API returned no current weather data.",
            latitude=latitude,
            longitude=longitude,
        )
    current_payload = cast(dict[str, object], raw_current)

    try:
        weather_code = _coerce_int(current_payload.get("weather_code"), "weather_code")
        return CurrentWeather(
            temperature_c=_coerce_float(current_payload.get("temperature_2m"), "temperature_2m"),
            humidity_percent=_coerce_float(
                current_payload.get("relative_humidity_2m"),
                "relative_humidity_2m",
            ),
            wind_kmh=_coerce_float(current_payload.get("wind_speed_10m"), "wind_speed_10m"),
            wind_direction_deg=_coerce_float(
                current_payload.get("wind_direction_10m"),
                "wind_direction_10m",
            ),
            weather_code=weather_code,
            weather_description=WEATHER_CODE_MAP.get(weather_code, "unknown conditions"),
        )
    except ValueError as exc:
        raise WeatherLookupError(
            WeatherErrorCode.WEATHER_API_ERROR,
            "Weather API returned an unexpected payload shape.",
            latitude=latitude,
            longitude=longitude,
        ) from exc


def _coerce_float(value: object, field_name: str) -> float:
    """Convert raw payload values into floats with explicit validation errors."""

    if isinstance(value, bool) or value is None:
        raise ValueError(f"{field_name} is missing or not numeric")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError as exc:
            raise ValueError(f"{field_name} is not parseable as float") from exc
    raise ValueError(f"{field_name} is not numeric")


def _coerce_int(value: object, field_name: str) -> int:
    """Convert raw payload values into exact integers with explicit validation errors."""

    if isinstance(value, bool) or value is None:
        raise ValueError(f"{field_name} is missing or not integer")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"{field_name} is not an exact integer")
        return int(value)
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            raise ValueError(f"{field_name} is empty")
        if not INTEGER_PATTERN.fullmatch(normalized):
            raise ValueError(f"{field_name} is not parseable as int")
        return int(normalized)
    raise ValueError(f"{field_name} is not integer")


def _coerce_str(value: object, field_name: str) -> str:
    """Convert raw payload values into non-empty strings."""

    if not isinstance(value, str):
        raise ValueError(f"{field_name} is missing or not a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} is empty")
    return normalized


def _coerce_optional_str(value: object, field_name: str) -> str | None:
    """Convert optional payload values into trimmed strings when present."""

    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field_name} is not a string")
    normalized = value.strip()
    return normalized or None


def _coerce_optional_country_code(
    value: object,
    field_name: str,
    *,
    required: bool,
) -> str | None:
    """Convert optional country codes into normalized ISO alpha-2 strings."""

    if value is None:
        if required:
            raise ValueError(f"{field_name} is missing")
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field_name} is not a string")
    normalized = value.strip().upper()
    if len(normalized) != 2 or not normalized.isalpha():
        raise ValueError(f"{field_name} is not a valid ISO alpha-2 code")
    return normalized


def _canonicalize_match_text(value: str) -> str:
    """Canonicalize text for exact matching across city, region, and country labels."""

    normalized_chars: list[str] = []
    for char in unicodedata.normalize("NFKD", value):
        category = unicodedata.category(char)
        if category == "Mn":
            continue
        if char.isalnum():
            normalized_chars.append(char.casefold())
            continue
        if char.isspace() or category.startswith(("P", "S")):
            normalized_chars.append(" ")
            continue
        normalized_chars.append(char.casefold())
    return " ".join("".join(normalized_chars).split())


def _canonicalize_optional_match_text(value: str | None) -> str | None:
    """Normalize optional text for dedupe comparisons."""

    if value is None:
        return None
    return _canonicalize_match_text(value)


def format_requested_location(
    name: str,
    admin1: str | None,
    country_code: str | None,
) -> str:
    """Build a compact requested-location label for logs and error messages."""

    parts = [name]
    if admin1:
        parts.append(admin1)
    if country_code:
        parts.append(country_code)
    return ", ".join(parts)


def _format_requested_qualifiers(admin1: str | None, country_code: str | None) -> str:
    """Build a compact qualifier label for mismatch messages."""

    parts: list[str] = []
    if admin1:
        parts.append(f"admin1={admin1!r}")
    if country_code:
        parts.append(f"country_code={country_code!r}")
    return ", ".join(parts)


def _as_candidates(results: list[GeocodeResult]) -> list[GeocodeCandidate]:
    """Convert resolved results into clarification candidates."""

    return [
        GeocodeCandidate(
            name=result.resolved_name,
            admin1=result.admin1,
            country=result.country,
        )
        for result in results
    ]


def normalize_optional_value(value: str | None) -> str | None:
    """Trim optional values while preserving None for missing qualifiers."""

    if value is None:
        return None
    normalized = value.strip()
    return normalized or None
