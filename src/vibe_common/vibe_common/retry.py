import logging
import os
import random
import time
from functools import wraps
from typing import Callable, Optional, Tuple, Type

logger = logging.getLogger(__name__)


def retry_with_backoff(
    max_retries: int = int(os.environ.get("VIBE_RETRY_MAX_RETRIES", "5")),
    base_delay: float = float(os.environ.get("VIBE_RETRY_BASE_DELAY", "1.0")),
    max_delay: float = float(os.environ.get("VIBE_RETRY_MAX_DELAY", "60.0")),
    backoff_factor: float = float(os.environ.get("VIBE_RETRY_BACKOFF_FACTOR", "2.0")),
    retryable_exceptions: Tuple[Type[BaseException], ...] = (Exception,),
    abort: Optional[Callable[[], bool]] = None,
):
    """Decorator that retries a function with exponential backoff and full jitter.

    Args:
        max_retries: Maximum number of attempts (not retries -- 3 means 3 total attempts).
        base_delay: Initial delay in seconds before the first retry.
        max_delay: Maximum delay in seconds between retries.
        backoff_factor: Multiplier applied to the delay after each retry.
        retryable_exceptions: Tuple of exception types that trigger a retry.
        abort: Optional callable that returns True to stop retrying early.
    """

    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            delay = base_delay
            last_exception = None
            for attempt in range(1, max_retries + 1):
                try:
                    return fn(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    if attempt == max_retries:
                        logger.error(
                            f"Attempt {attempt}/{max_retries} failed with "
                            f"{type(e).__name__}: {e}. No retries remaining."
                        )
                        raise
                    if abort is not None and abort():
                        logger.warning(
                            f"Attempt {attempt}/{max_retries} failed with "
                            f"{type(e).__name__}: {e}. Abort requested, stopping retries."
                        )
                        raise
                    jittered_delay = random.uniform(0, min(delay, max_delay))
                    logger.warning(
                        f"Attempt {attempt}/{max_retries} failed with "
                        f"{type(e).__name__}: {e}. "
                        f"Retrying in {jittered_delay:.2f}s."
                    )
                    time.sleep(jittered_delay)
                    delay = min(delay * backoff_factor, max_delay)
            raise last_exception

        return wrapper

    return decorator
