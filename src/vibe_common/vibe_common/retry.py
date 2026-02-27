# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Reusable retry utilities with exponential backoff.

Design notes:
- No external deps. Project avoids tenacity/backoff by convention.
- `compute_backoff` is separated from the decorator so callers that can't
  decorate (e.g. the worker's run_op_with_retry loop) can still use the
  same delay calculation.
- Full jitter (https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/):
  sleep = random(0, min(cap, base * exp^attempt)). Prevents thundering herd
  when many workers retry simultaneously.
"""

import asyncio
import functools
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional, TypeVar, Union, cast

T = TypeVar("T")


def _default_retryable(e: BaseException) -> bool:
    return True


@dataclass
class RetryPolicy:
    """Configuration for retry behaviour.

    Attributes:
        max_attempts: total tries including the first. 1 = no retry.
        base_delay_s: delay after attempt 0 when exponential_base=1 or jitter off.
        max_delay_s: ceiling on computed delay.
        exponential_base: delay multiplier per attempt. 2.0 = double each time.
        jitter: if True, use full jitter (uniform in [0, computed_delay]).
        retryable: predicate — return True if the exception should trigger a retry.
                   Default retries everything.
    """

    max_attempts: int = 3
    base_delay_s: float = 1.0
    max_delay_s: float = 30.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable: Callable[[BaseException], bool] = field(default=_default_retryable)


def compute_backoff(attempt: int, policy: RetryPolicy) -> float:
    """Returns seconds to sleep before the next attempt.

    `attempt` is zero-indexed: attempt=0 → delay after the first failure.
    """
    raw = policy.base_delay_s * (policy.exponential_base ** attempt)
    capped = min(raw, policy.max_delay_s)
    if policy.jitter:
        return random.uniform(0, capped)
    return capped
