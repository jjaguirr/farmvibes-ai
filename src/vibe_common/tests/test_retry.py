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


from unittest.mock import MagicMock
from vibe_common.retry import with_retry


def test_with_retry_returns_on_first_success():
    mock = MagicMock(return_value="ok")
    p = RetryPolicy(max_attempts=3, base_delay_s=0.001, jitter=False)
    wrapped = with_retry(p)(mock)
    assert wrapped() == "ok"
    assert mock.call_count == 1


def test_with_retry_retries_until_success():
    mock = MagicMock(side_effect=[ConnectionError(), ConnectionError(), "ok"])
    p = RetryPolicy(max_attempts=5, base_delay_s=0.001, jitter=False)
    wrapped = with_retry(p)(mock)
    assert wrapped() == "ok"
    assert mock.call_count == 3


def test_with_retry_exhausts_and_raises_last():
    mock = MagicMock(side_effect=ValueError("persistent"))
    p = RetryPolicy(max_attempts=3, base_delay_s=0.001, jitter=False)
    wrapped = with_retry(p)(mock)
    with pytest.raises(ValueError, match="persistent"):
        wrapped()
    assert mock.call_count == 3


def test_with_retry_respects_retryable_predicate():
    mock = MagicMock(side_effect=ValueError("not retryable"))
    p = RetryPolicy(
        max_attempts=5,
        base_delay_s=0.001,
        retryable=lambda e: isinstance(e, ConnectionError),
    )
    wrapped = with_retry(p)(mock)
    with pytest.raises(ValueError):
        wrapped()
    assert mock.call_count == 1


def test_with_retry_logs_each_attempt(caplog):
    import logging
    mock = MagicMock(side_effect=[ConnectionError("flaky"), "ok"])
    p = RetryPolicy(max_attempts=3, base_delay_s=0.001, jitter=False)
    wrapped = with_retry(p, logger=logging.getLogger("test"))(mock)
    with caplog.at_level(logging.WARNING):
        wrapped()
    assert any("attempt 1/3" in r.message and "ConnectionError" in r.message
               for r in caplog.records)
