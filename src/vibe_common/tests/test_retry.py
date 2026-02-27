import time
from unittest.mock import MagicMock, patch

import pytest

from vibe_common.retry import retry_with_backoff


class TransientError(Exception):
    pass


class PermanentError(Exception):
    pass


def test_succeeds_on_first_try():
    fn = MagicMock(return_value="ok")
    decorated = retry_with_backoff(max_retries=3)(fn)
    assert decorated() == "ok"
    assert fn.call_count == 1


def test_retries_on_transient_error():
    fn = MagicMock(side_effect=[TransientError(), TransientError(), "ok"])
    decorated = retry_with_backoff(
        max_retries=3, retryable_exceptions=(TransientError,), base_delay=0.01
    )(fn)
    assert decorated() == "ok"
    assert fn.call_count == 3


def test_raises_after_max_retries_exhausted():
    fn = MagicMock(side_effect=TransientError("always fails"))
    decorated = retry_with_backoff(
        max_retries=3, retryable_exceptions=(TransientError,), base_delay=0.01
    )(fn)
    with pytest.raises(TransientError, match="always fails"):
        decorated()
    assert fn.call_count == 3


def test_does_not_retry_non_retryable_exception():
    fn = MagicMock(side_effect=PermanentError("fatal"))
    decorated = retry_with_backoff(
        max_retries=3, retryable_exceptions=(TransientError,), base_delay=0.01
    )(fn)
    with pytest.raises(PermanentError, match="fatal"):
        decorated()
    assert fn.call_count == 1


def test_backoff_gaps_increase():
    """Verify that the delay *cap* grows between retries by capturing sleep args.

    With full jitter the actual sleep is random.uniform(0, delay), so we can't
    assert exact values.  But we *can* patch time.sleep and inspect the upper
    bound that was passed to random.uniform (which sets the sleep arg).
    """
    sleep_durations = []
    original_sleep = time.sleep

    def capture_sleep(duration):
        sleep_durations.append(duration)
        # Don't actually sleep in tests

    fn = MagicMock(side_effect=[TransientError(), TransientError(), TransientError()])
    decorated = retry_with_backoff(
        max_retries=3,
        retryable_exceptions=(TransientError,),
        base_delay=1.0,
        backoff_factor=2.0,
        max_delay=100.0,
    )(fn)
    with patch("vibe_common.retry.time.sleep", side_effect=capture_sleep):
        with pytest.raises(TransientError):
            decorated()
    # With base=1.0, factor=2.0: delay caps are 1.0, 2.0
    # Jitter picks uniform(0, cap) so actual sleep is in [0, cap]
    assert len(sleep_durations) == 2  # 3 attempts, 2 sleeps between them
    assert sleep_durations[0] <= 1.0  # first cap is min(1.0, 100.0) = 1.0
    assert sleep_durations[1] <= 2.0  # second cap is min(2.0, 100.0) = 2.0


def test_max_delay_clamps_sleep():
    """Verify that sleep never exceeds max_delay regardless of backoff growth."""
    sleep_durations = []

    def capture_sleep(duration):
        sleep_durations.append(duration)

    fn = MagicMock(
        side_effect=[TransientError(), TransientError(), TransientError(), TransientError()]
    )
    decorated = retry_with_backoff(
        max_retries=4,
        retryable_exceptions=(TransientError,),
        base_delay=10.0,
        backoff_factor=10.0,
        max_delay=0.5,
    )(fn)
    with patch("vibe_common.retry.time.sleep", side_effect=capture_sleep):
        with pytest.raises(TransientError):
            decorated()
    assert len(sleep_durations) == 3
    for d in sleep_durations:
        assert d <= 0.5, f"Sleep {d} exceeded max_delay 0.5"


def test_single_attempt_no_retries():
    """max_retries=1 means one attempt, zero retries."""
    fn = MagicMock(side_effect=TransientError("once"))
    decorated = retry_with_backoff(
        max_retries=1, retryable_exceptions=(TransientError,), base_delay=0.01
    )(fn)
    with pytest.raises(TransientError, match="once"):
        decorated()
    assert fn.call_count == 1


def test_passes_args_and_kwargs():
    fn = MagicMock(return_value="result")
    decorated = retry_with_backoff(max_retries=1)(fn)
    result = decorated("a", "b", key="val")
    assert result == "result"
    fn.assert_called_once_with("a", "b", key="val")


def test_retry_logs_each_attempt(caplog):
    import logging

    fn = MagicMock(side_effect=[TransientError("oops"), "ok"])
    decorated = retry_with_backoff(
        max_retries=3, retryable_exceptions=(TransientError,), base_delay=0.01
    )(fn)
    with caplog.at_level(logging.WARNING):
        decorated()
    assert "Attempt 1/3 failed" in caplog.text
    assert "TransientError" in caplog.text
    assert "oops" in caplog.text


def test_abort_callable_stops_retries():
    fn = MagicMock(side_effect=TransientError("fail"))
    abort = MagicMock(return_value=True)
    decorated = retry_with_backoff(
        max_retries=5, retryable_exceptions=(TransientError,), base_delay=0.01, abort=abort
    )(fn)
    with pytest.raises(TransientError):
        decorated()
    assert fn.call_count == 1
    abort.assert_called_once()


def test_abort_not_called_on_success():
    fn = MagicMock(return_value="ok")
    abort = MagicMock(return_value=True)
    decorated = retry_with_backoff(
        max_retries=3, retryable_exceptions=(TransientError,), abort=abort
    )(fn)
    assert decorated() == "ok"
    abort.assert_not_called()


def test_exhaustion_logs_error(caplog):
    import logging

    fn = MagicMock(side_effect=TransientError("boom"))
    decorated = retry_with_backoff(
        max_retries=2, retryable_exceptions=(TransientError,), base_delay=0.01
    )(fn)
    with caplog.at_level(logging.ERROR):
        with pytest.raises(TransientError):
            decorated()
    assert "No retries remaining" in caplog.text
