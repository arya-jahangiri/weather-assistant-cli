"""Weather domain models, error codes, and Pydantic schemas."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator


class WeatherErrorCode(StrEnum):
    """Stable error codes returned by weather-domain tool outputs."""

    CITY_NOT_FOUND = "CITY_NOT_FOUND"
    LOCATION_AMBIGUOUS = "LOCATION_AMBIGUOUS"
    LOCATION_QUALIFIER_MISMATCH = "LOCATION_QUALIFIER_MISMATCH"
    NETWORK_ERROR = "NETWORK_ERROR"
    GEOCODING_API_ERROR = "GEOCODING_API_ERROR"
    WEATHER_API_ERROR = "WEATHER_API_ERROR"


class GeocodeResult(BaseModel):
    """Resolved location record from Open-Meteo geocoding."""

    model_config = ConfigDict(extra="forbid")

    resolved_name: str
    admin1: str | None = None
    country: str
    country_code: str | None = None
    latitude: float
    longitude: float


class GeocodeCandidate(BaseModel):
    """Minimal candidate record surfaced for location clarification."""

    model_config = ConfigDict(extra="forbid")

    name: str
    admin1: str | None = None
    country: str


class CurrentWeather(BaseModel):
    """Current weather observation returned by Open-Meteo."""

    model_config = ConfigDict(extra="forbid")

    temperature_c: float
    humidity_percent: float
    wind_kmh: float
    wind_direction_deg: float
    weather_code: int
    weather_description: str


class GetWeatherArgs(BaseModel):
    """Arguments for the weather tool function call."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        min_length=1,
        description=(
            "Resolved city or place name for one location. Expand obvious shorthand when the "
            "intended location is clear, for example use 'Los Angeles' rather than 'LA'. "
            "Keep state, province, region, and country text out of this field."
        ),
    )
    admin1: str | None = Field(
        default=None,
        description=(
            "Optional full first-level administrative region name such as 'California' or "
            "'District of Columbia'. Use this when the user supplies a region or when it is "
            "needed to disambiguate places with the same name."
        ),
    )
    country_code: str | None = Field(
        default=None,
        description=(
            "Optional ISO 3166-1 alpha-2 country code, for example 'US' or 'GB'. Use this when "
            "the user supplies a country or when it is needed to disambiguate places with the "
            "same name."
        ),
    )

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        """Normalize and reject blank location names."""

        normalized = value.strip()
        if not normalized:
            raise ValueError("name cannot be empty")
        return normalized

    @field_validator("admin1")
    @classmethod
    def normalize_admin1(cls, value: str | None) -> str | None:
        """Normalize optional region qualifiers."""

        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @field_validator("country_code")
    @classmethod
    def normalize_country_code(cls, value: str | None) -> str | None:
        """Normalize optional ISO alpha-2 country codes."""

        if value is None:
            return None
        normalized = value.strip().upper()
        if not normalized:
            return None
        if len(normalized) != 2 or not normalized.isalpha():
            raise ValueError("country_code must be a 2-letter ISO 3166-1 alpha-2 code")
        return normalized


class WeatherToolBase(BaseModel):
    """Shared fields included in all weather tool payloads."""

    model_config = ConfigDict(extra="forbid")

    name: str
    requested_admin1: str | None = None
    requested_country_code: str | None = None


class WeatherToolSuccess(WeatherToolBase):
    """Successful weather tool output consumed by the LLM."""

    ok: Literal[True] = True
    resolved_name: str
    admin1: str | None = None
    country: str
    latitude: float
    longitude: float
    temperature_c: float
    humidity_percent: float
    wind_kmh: float
    wind_direction_deg: float
    weather_description: str


class WeatherToolFailure(WeatherToolBase):
    """Failed weather tool output consumed by the LLM."""

    ok: Literal[False] = False
    resolved_name: str | None = None
    admin1: str | None = None
    country: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    candidates: list[GeocodeCandidate] | None = None
    error_code: WeatherErrorCode
    error_message: str


WeatherToolResult: TypeAlias = WeatherToolSuccess | WeatherToolFailure


@dataclass(slots=True)
class WeatherLookupError(Exception):
    """Exception carrying stable error-code metadata for tool outputs."""

    code: WeatherErrorCode
    message: str
    resolved_name: str | None = None
    admin1: str | None = None
    country: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    candidates: list[GeocodeCandidate] | None = None


WEATHER_CODE_MAP: dict[int, str] = {
    0: "clear skies",
    1: "mostly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    56: "light freezing drizzle",
    57: "dense freezing drizzle",
    61: "slight rain",
    63: "moderate rain",
    65: "heavy rain",
    66: "light freezing rain",
    67: "heavy freezing rain",
    71: "slight snowfall",
    73: "moderate snowfall",
    75: "heavy snowfall",
    77: "snow grains",
    80: "slight rain showers",
    81: "moderate rain showers",
    82: "violent rain showers",
    85: "slight snow showers",
    86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with slight hail",
    99: "thunderstorm with heavy hail",
}
