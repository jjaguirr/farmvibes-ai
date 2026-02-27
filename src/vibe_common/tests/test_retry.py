# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest
from vibe_common.retry import RetryPolicy, compute_backoff


def test_backoff_grows_exponentially():
    p = RetryPolicy(base_delay_s=1.0, exponential_base=2.0, max_delay_s=1000.0, jitter=False)
    assert compute_backoff(0, p) == 1.0
    assert compute_backoff(1, p) == 2.0
    assert compute_backoff(2, p) == 4.0
    assert compute_backoff(3, p) == 8.0


def test_backoff_capped_at_max_delay():
    p = RetryPolicy(base_delay_s=1.0, exponential_base=2.0, max_delay_s=5.0, jitter=False)
    assert compute_backoff(10, p) == 5.0


def test_backoff_jitter_within_range():
    p = RetryPolicy(base_delay_s=2.0, exponential_base=2.0, max_delay_s=100.0, jitter=True)
    for _ in range(20):
        delay = compute_backoff(0, p)
        assert 0.0 <= delay <= 2.0


def test_retryable_predicate_default_accepts_all():
    p = RetryPolicy()
    assert p.retryable(ValueError("boom"))
    assert p.retryable(ConnectionError())


def test_retryable_predicate_custom():
    p = RetryPolicy(retryable=lambda e: isinstance(e, ConnectionError))
    assert p.retryable(ConnectionError())
    assert not p.retryable(ValueError())
