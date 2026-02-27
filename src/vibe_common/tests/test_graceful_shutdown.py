import threading
import time
from unittest.mock import MagicMock

import pytest

from vibe_common.graceful_shutdown import ShutdownManager, ShutdownState


# --- State transitions ---


def test_initial_state_is_running():
    mgr = ShutdownManager()
    assert mgr.state == ShutdownState.RUNNING
    assert not mgr.is_draining


def test_shutdown_transitions_to_draining():
    mgr = ShutdownManager()
    mgr.shutdown()
    assert mgr.state == ShutdownState.DRAINING
    assert mgr.is_draining


def test_finalize_transitions_to_shutting_down():
    mgr = ShutdownManager()
    mgr.shutdown()
    mgr.finalize()
    assert mgr.state == ShutdownState.SHUTTING_DOWN


def test_is_draining_true_in_shutting_down_state():
    """is_draining should be True in both DRAINING and SHUTTING_DOWN states."""
    mgr = ShutdownManager()
    mgr.shutdown()
    mgr.finalize()
    assert mgr.state == ShutdownState.SHUTTING_DOWN
    assert mgr.is_draining is True


def test_double_shutdown_is_idempotent():
    mgr = ShutdownManager()
    cb = MagicMock()
    mgr.on_shutdown(cb)
    mgr.shutdown()
    mgr.shutdown()  # Second call should be no-op
    mgr.finalize()
    cb.assert_called_once()


# --- Callbacks ---


def test_callbacks_called_in_lifo_order():
    mgr = ShutdownManager()
    order = []
    mgr.on_shutdown(lambda: order.append("first"))
    mgr.on_shutdown(lambda: order.append("second"))
    mgr.on_shutdown(lambda: order.append("third"))
    mgr.shutdown()
    mgr.finalize()
    assert order == ["third", "second", "first"]


def test_callback_exception_does_not_block_others():
    mgr = ShutdownManager()
    order = []
    mgr.on_shutdown(lambda: order.append("first"))

    def raise_error():
        raise RuntimeError("boom")

    mgr.on_shutdown(raise_error)
    mgr.on_shutdown(lambda: order.append("third"))
    mgr.shutdown()
    mgr.finalize()
    # LIFO: "third" runs first, then raise_error (fails), then "first"
    assert order == ["third", "first"]


def test_callbacks_not_called_before_finalize():
    """Callbacks should only run during finalize(), not during shutdown()."""
    mgr = ShutdownManager()
    cb = MagicMock()
    mgr.on_shutdown(cb)
    mgr.shutdown()
    cb.assert_not_called()
    mgr.finalize()
    cb.assert_called_once()


# --- Work completion ---


def test_wait_for_completion_returns_true_when_event_set():
    mgr = ShutdownManager()
    mgr.mark_work_complete()
    mgr.shutdown()
    assert mgr.wait_for_completion(timeout_s=1.0) is True


def test_wait_for_completion_returns_false_on_timeout():
    mgr = ShutdownManager()
    mgr.shutdown()
    start = time.monotonic()
    result = mgr.wait_for_completion(timeout_s=0.05)
    elapsed = time.monotonic() - start
    assert result is False
    assert elapsed >= 0.04, "Should have waited close to the timeout"


def test_wait_for_completion_unblocks_when_work_completes():
    mgr = ShutdownManager()
    mgr.shutdown()

    def complete_later():
        time.sleep(0.05)
        mgr.mark_work_complete()

    t = threading.Thread(target=complete_later)
    t.start()
    start = time.monotonic()
    result = mgr.wait_for_completion(timeout_s=2.0)
    elapsed = time.monotonic() - start
    t.join()
    assert result is True
    assert elapsed < 1.0, "Should have unblocked well before timeout"


def test_mark_work_complete_multiple_times():
    """Calling mark_work_complete multiple times should not raise or change behavior."""
    mgr = ShutdownManager()
    mgr.mark_work_complete()
    mgr.mark_work_complete()
    assert mgr.wait_for_completion(timeout_s=0.01) is True


# --- Signal registration ---


def test_signal_handler_triggers_shutdown():
    """_signal_handler should transition to DRAINING state."""
    mgr = ShutdownManager()
    # Call the handler directly (we can't reliably send signals in tests)
    mgr._signal_handler(14, None)  # 14 = SIGALRM on most systems
    assert mgr.state == ShutdownState.DRAINING
    assert mgr.is_draining
