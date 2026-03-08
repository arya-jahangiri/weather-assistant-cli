"""Deterministic retry/backoff utilities."""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BackoffPolicy:
    """Configuration for exponential backoff retries."""

    attempts: int
    base_delay_seconds: float
    with_jitter: bool = False


def backoff_delay_seconds(policy: BackoffPolicy, attempt_index: int) -> float:
    """Compute the delay for a retry attempt using exponential backoff."""

    base_delay = policy.base_delay_seconds * (2**attempt_index)
    if not policy.with_jitter:
        return base_delay
    return base_delay + random.uniform(0.0, policy.base_delay_seconds)
