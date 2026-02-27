# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for vibe_agent.worker hardening (Task 11).

These tests require pebble/dapr/hydra_zen installed — they don't collect on
the minimal local dev env. Full validation runs on the VM (Task 13).
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vibe_agent.worker import WorkerMessenger


# --- WorkerMessenger send cap (Task 6) ---

@pytest.mark.anyio
async def test_messenger_send_caps_retries():
    """send() must not loop forever when pubsub is down."""
    messenger = WorkerMessenger()
    messenger.max_send_attempts = 3
    messenger.retry_interval_s = 0.01

    with patch("vibe_agent.worker.send_async", new=AsyncMock(return_value=False)):
        with pytest.raises(RuntimeError, match="3 attempts"):
            await messenger.send(MagicMock())


@pytest.mark.anyio
async def test_messenger_send_succeeds_after_transient_failure():
    """send() should succeed once send_async returns True, not keep retrying."""
    messenger = WorkerMessenger()
    messenger.max_send_attempts = 5
    messenger.retry_interval_s = 0.01

    call_count = 0
    async def flaky_send(*a, **kw):
        nonlocal call_count
        call_count += 1
        return call_count >= 3  # succeed on 3rd try

    with patch("vibe_agent.worker.send_async", new=flaky_send):
        await messenger.send(MagicMock())  # should not raise
    assert call_count == 3


@pytest.mark.anyio
async def test_messenger_send_respects_deadline():
    """Deadline already past → give up before ever calling send_async."""
    import time

    messenger = WorkerMessenger()
    messenger.max_send_attempts = 100
    messenger.retry_interval_s = 0.01
    messenger.deadline = time.monotonic() - 1.0  # already expired

    mock_send = AsyncMock(return_value=False)
    with patch("vibe_agent.worker.send_async", new=mock_send):
        with pytest.raises(RuntimeError, match="deadline"):
            await messenger.send(MagicMock())
    mock_send.assert_not_called()  # bailed at the gate, no wasted attempt


# --- Drain-and-requeue shutdown (Task 7) ---

import signal
import time

from vibe_agent.worker import Worker, ShuttingDownException


def test_sigterm_sets_deadline_without_cancelling_child(worker: Worker):
    """SIGTERM sets shutdown_deadline; does NOT immediately cancel the child."""
    worker.current_child = MagicMock()
    worker.current_message = MagicMock()

    worker.pre_stop_hook(signal.SIGTERM, None)

    assert worker.shutting_down
    assert worker.shutdown_deadline is not None
    assert worker.shutdown_deadline > time.monotonic()
    worker.current_child.cancel.assert_not_called()


def test_messenger_learns_shutdown_deadline(worker: Worker):
    """Messenger's deadline is set so status sends can't overrun the grace period."""
    assert worker.messenger.deadline is None
    worker.pre_stop_hook(signal.SIGTERM, None)
    assert worker.messenger.deadline == worker.shutdown_deadline


def test_drain_lets_op_finish_before_deadline(worker: Worker):
    """If subprocess finishes before deadline, return its result — don't cancel."""
    worker.current_message = MagicMock()
    worker.is_workflow_complete = MagicMock(return_value=False)
    worker.shutting_down = True
    worker.shutdown_deadline = time.monotonic() + 30

    mock_child = MagicMock()
    mock_child.result.return_value = {"output": "success"}

    result = worker.get_future_result(mock_child, monitoring_period_s=1, timeout_s=60)
    assert result == {"output": "success"}
    mock_child.cancel.assert_not_called()


def test_drain_cancels_after_deadline(worker: Worker):
    """Deadline passed → cancel child, raise ShuttingDownException for redelivery."""
    import concurrent.futures

    worker.current_message = MagicMock()
    worker.is_workflow_complete = MagicMock(return_value=False)
    worker.shutting_down = True
    worker.shutdown_deadline = time.monotonic() - 1  # already expired

    mock_child = MagicMock()
    mock_child.result.side_effect = concurrent.futures.TimeoutError()

    with pytest.raises(ShuttingDownException):
        worker.get_future_result(mock_child, monitoring_period_s=0.01, timeout_s=60)

    mock_child.cancel.assert_called_once()


# --- OOM detection (Task 8) ---

from pebble.common import ProcessExpired


def _mock_content(op_name="test_op"):
    """Minimal stand-in for CacheInfoExecuteRequestContent."""
    content = MagicMock()
    content.operation_spec = MagicMock()
    content.operation_spec.name = op_name
    content.input = {}
    return content


def test_oom_exitcode_produces_factual_error_and_no_retry(worker: Worker):
    """SIGKILL exit → error names the op, signal, exit code, memory. No retry."""
    oom_exc = ProcessExpired("Abnormal termination")
    oom_exc.exitcode = -9
    worker.try_run_op = MagicMock(side_effect=oom_exc)

    with patch("vibe_agent.worker.get_memory_info") as mock_mem:
        mock_mem.return_value = MagicMock(
            __str__=lambda self: "4096MB / 4096MB (100%)"
        )
        with pytest.raises(RuntimeError) as exc_info:
            worker.run_op_with_retry(
                _mock_content("download_s2"), run_id=MagicMock(), timeout_s=30
            )

    msg = str(exc_info.value)
    # Every factual claim is present; no speculation/guidance.
    assert "download_s2" in msg
    assert "SIGKILL" in msg
    assert "exit code -9" in msg
    assert "4096MB" in msg and "100%" in msg
    assert "likely" not in msg.lower()  # regression guard: factual-only
    assert "consider" not in msg.lower()
    assert worker.try_run_op.call_count == 1  # no retry on OOM


def test_non_oom_process_expired_still_retries(worker: Worker):
    """Non-SIGKILL ProcessExpired (e.g. exit 1) retries normally."""
    generic_exc = ProcessExpired("died")
    generic_exc.exitcode = 1
    worker.try_run_op = MagicMock(side_effect=generic_exc)
    worker.max_tries = 3

    with patch.object(worker, "_interruptible_sleep"):  # skip real backoff sleeps
        with pytest.raises(RuntimeError):
            worker.run_op_with_retry(_mock_content(), run_id=MagicMock(), timeout_s=30)

    assert worker.try_run_op.call_count == 3


# --- Retry backoff (Task 9) ---

import traceback as tb


def test_retry_sleeps_with_backoff_between_attempts(worker: Worker):
    """Backoff: compute_backoff called once per gap with correct attempt index and
    the worker's configured policy; _interruptible_sleep receives the exact delays."""
    worker.max_tries = 3
    worker.op_retry_base_delay_s = 0.5
    worker.op_retry_max_delay_s = 10.0

    fake_tb = tb.TracebackException(ValueError, ValueError("transient"), None)
    worker.try_run_op = MagicMock(return_value=fake_tb)

    # Return deterministic delays so we can verify wiring end-to-end.
    # compute_backoff's own math is covered in vibe_common/tests/test_retry.py.
    planned_delays = [0.5, 1.0]  # indexed by attempt
    def fake_backoff(attempt, policy):
        assert policy.base_delay_s == 0.5
        assert policy.max_delay_s == 10.0
        return planned_delays[attempt]

    sleep_calls = []
    with patch("vibe_agent.worker.compute_backoff", side_effect=fake_backoff) as mock_cb, \
         patch.object(worker, "_interruptible_sleep", side_effect=sleep_calls.append):
        with pytest.raises(RuntimeError):
            worker.run_op_with_retry(_mock_content(), run_id=MagicMock(), timeout_s=30)

    # 3 tries → 2 gaps; attempt indices 0, 1
    assert [c.args[0] for c in mock_cb.call_args_list] == [0, 1]
    # Sleep durations are exactly what compute_backoff returned.
    assert sleep_calls == planned_delays


def test_retry_backoff_abandoned_on_shutdown(worker: Worker):
    """Shutdown set before first iteration → bail at top of loop; never attempt the op."""
    worker.max_tries = 3
    worker.op_retry_base_delay_s = 10.0
    worker.op_retry_max_delay_s = 10.0
    worker.shutting_down = True
    worker.try_run_op = MagicMock()

    start = time.monotonic()
    with pytest.raises(ShuttingDownException):
        worker.run_op_with_retry(_mock_content(), run_id=MagicMock(), timeout_s=30)
    elapsed = time.monotonic() - start

    worker.try_run_op.assert_not_called()  # top-of-loop check fires before any attempt
    assert elapsed < 0.1  # no backoff sleep happened


# --- Memory pressure warning + heartbeat (Task 10) ---

import concurrent.futures
import logging


def test_memory_pressure_warning_logged_once(worker: Worker, caplog):
    """Usage > 85% → log WARNING once per crossing with stats; don't spam every poll."""
    worker.current_message = MagicMock()
    worker.is_workflow_complete = MagicMock(return_value=False)

    mock_child = MagicMock()
    # Two timeout polls then success — warning should fire on poll 1, not poll 2.
    mock_child.result.side_effect = [
        concurrent.futures.TimeoutError(),
        concurrent.futures.TimeoutError(),
        {"done": True},
    ]

    high_mem = MagicMock(usage_fraction=0.92, usage_mb=3700.0, limit_mb=4000.0)
    high_mem.__str__ = lambda self: "3700MB / 4000MB (92%)"
    with patch("vibe_agent.worker.get_memory_info", return_value=high_mem), \
         caplog.at_level(logging.WARNING):
        worker.get_future_result(mock_child, monitoring_period_s=0.01, timeout_s=30)

    warnings = [r for r in caplog.records if "high memory" in r.message.lower()]
    assert len(warnings) == 1  # one-shot across 2 polls
    # Factual content: actual numbers in the log line, not just a label.
    assert "3700MB" in warnings[0].message
    assert "92%" in warnings[0].message


def test_heartbeat_sent_during_long_op(worker: Worker):
    """One heartbeat per poll-timeout (interval=0); payload carries op_name, elapsed, memory."""
    worker.current_message = MagicMock()
    worker.current_message.id = "00-0123456789abcdef0123456789abcdef-0123456789abcdef-01"
    worker.current_message.content = MagicMock()
    worker.current_message.content.operation_spec.name = "long_op"
    worker.is_workflow_complete = MagicMock(return_value=False)
    worker.heartbeat_interval_s = 0.0  # fire on every poll

    mock_child = MagicMock()
    mock_child.result.side_effect = [
        concurrent.futures.TimeoutError(),
        concurrent.futures.TimeoutError(),
        {"done": True},
    ]

    sent_messages = []
    async def capture_send(msg, *a, **kw):
        sent_messages.append(msg)
        return True

    low_mem = MagicMock(usage_fraction=0.10, usage_mb=100.0, limit_mb=1000.0)
    with patch("vibe_agent.worker.send_async", new=capture_send), \
         patch("vibe_agent.worker.get_memory_info", return_value=low_mem):
        result = worker.get_future_result(mock_child, monitoring_period_s=0.01, timeout_s=30)

    assert result == {"done": True}

    from vibe_common.messaging import HeartbeatMessage
    heartbeats = [m for m in sent_messages if isinstance(m, HeartbeatMessage)]
    assert len(heartbeats) == 2  # interval=0 → one per timeout-poll, exactly
    for hb in heartbeats:
        assert hb.content.op_name == "long_op"
        assert hb.content.elapsed_s >= 0.0
        assert hb.content.memory_usage_mb == 100.0
        assert hb.content.memory_limit_mb == 1000.0
