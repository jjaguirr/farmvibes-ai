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
import inspect
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional, TypeVar, cast

__all__ = ["RetryPolicy", "compute_backoff", "with_retry"]

T = TypeVar("T")


def _default_retryable(e: Exception) -> bool:
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
    retryable: Callable[[Exception], bool] = field(default=_default_retryable)

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError(f"max_attempts must be >= 1, got {self.max_attempts}")


def compute_backoff(attempt: int, policy: RetryPolicy) -> float:
    """Returns seconds to sleep before the next attempt.

    `attempt` is zero-indexed: attempt=0 → delay after the first failure.
    """
    if attempt < 0:
        attempt = 0
    try:
        raw = policy.base_delay_s * (policy.exponential_base ** attempt)
    except OverflowError:
        raw = policy.max_delay_s
    capped = min(raw, policy.max_delay_s)
    if policy.jitter:
        return random.uniform(0, capped)
    return capped


def with_retry(
    policy: RetryPolicy,
    logger: Optional[logging.Logger] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator that retries the wrapped callable per `policy`.

    Works on sync and async callables. Logs WARNING on each failed attempt
    with the exception class and next delay. Re-raises the last exception
    when attempts are exhausted or the exception is not retryable.
    """
    log = logger or logging.getLogger(__name__)

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        fn_name = getattr(fn, "__name__", repr(fn))
        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> T:
                last_exc: Optional[BaseException] = None
                for attempt in range(policy.max_attempts):
                    try:
                        return await cast(Callable[..., Awaitable[T]], fn)(*args, **kwargs)
                    except Exception as e:
                        last_exc = e
                        if not policy.retryable(e):
                            raise
                        if attempt + 1 >= policy.max_attempts:
                            raise
                        delay = compute_backoff(attempt, policy)
                        log.warning(
                            f"{fn_name}: attempt {attempt + 1}/{policy.max_attempts} "
                            f"failed with {type(e).__name__}: {e}. "
                            f"Retrying in {delay:.2f}s."
                        )
                        await asyncio.sleep(delay)
                assert last_exc is not None
                raise last_exc
            return cast(Callable[..., T], async_wrapper)

        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> T:
            last_exc: Optional[BaseException] = None
            for attempt in range(policy.max_attempts):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    last_exc = e
                    if not policy.retryable(e):
                        raise
                    if attempt + 1 >= policy.max_attempts:
                        raise
                    delay = compute_backoff(attempt, policy)
                    log.warning(
                        f"{fn_name}: attempt {attempt + 1}/{policy.max_attempts} "
                        f"failed with {type(e).__name__}: {e}. "
                        f"Retrying in {delay:.2f}s."
                    )
                    time.sleep(delay)
            assert last_exc is not None
            raise last_exc
        return sync_wrapper

    return decorator
