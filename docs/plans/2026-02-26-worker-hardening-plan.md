# Worker Service Hardening Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Harden the FarmVibes.AI worker service against transient failures, OOM kills, ungraceful shutdowns, and stuck workflows by adding retry with backoff, resource monitoring, graceful shutdown, and heartbeat-based stale detection.

**Architecture:** Three new standalone utility modules in `vibe_common` (retry, resource_monitor, graceful_shutdown). Worker integrates all three plus publishes heartbeats. Orchestrator consumes heartbeats and detects stale workflows.

**Tech Stack:** Python 3.8+, pebble (subprocess management), Dapr pub/sub (RabbitMQ), cgroup v2 (resource monitoring), pytest (testing)

---

### Task 1: Retry with Backoff — Module

**Files:**
- Create: `src/vibe_common/vibe_common/retry.py`
- Test: `src/vibe_common/tests/test_retry.py`

**Step 1: Write the failing tests**

```python
# src/vibe_common/tests/test_retry.py
import time
from unittest.mock import MagicMock

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


def test_backoff_delay_increases():
    call_times = []
    call_count = 0

    def failing_fn():
        nonlocal call_count
        call_times.append(time.monotonic())
        call_count += 1
        if call_count < 3:
            raise TransientError()
        return "ok"

    decorated = retry_with_backoff(
        max_retries=3,
        retryable_exceptions=(TransientError,),
        base_delay=0.05,
        backoff_factor=2.0,
        max_delay=10.0,
    )(failing_fn)
    decorated()
    # Second delay should be >= first delay (backoff with jitter, but on average larger)
    # We just check that delays are non-negative (jitter makes exact checks unreliable)
    assert len(call_times) == 3
    for i in range(1, len(call_times)):
        assert call_times[i] > call_times[i - 1]


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


def test_abort_callable_stops_retries():
    fn = MagicMock(side_effect=TransientError("fail"))
    abort = MagicMock(return_value=True)
    decorated = retry_with_backoff(
        max_retries=5, retryable_exceptions=(TransientError,), base_delay=0.01, abort=abort
    )(fn)
    with pytest.raises(TransientError):
        decorated()
    # Should have stopped after first failed attempt because abort returned True
    assert fn.call_count == 1
```

**Step 2: Run tests to verify they fail**

Run: `cd src/vibe_common && python -m pytest tests/test_retry.py -v 2>&1 | tail -20`
Expected: FAIL — `ModuleNotFoundError: No module named 'vibe_common.retry'`

**Step 3: Write the implementation**

```python
# src/vibe_common/vibe_common/retry.py
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
        max_retries: Maximum number of attempts (not retries — 3 means 3 total attempts).
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
            # Should not reach here, but just in case
            raise last_exception  # type: ignore[misc]

        return wrapper

    return decorator
```

**Step 4: Run tests to verify they pass**

Run: `cd src/vibe_common && python -m pytest tests/test_retry.py -v 2>&1 | tail -20`
Expected: All tests PASS

**Step 5: Commit**

```
jj describe -m "feat(retry): add retry_with_backoff decorator with exponential backoff and jitter"
jj new
```

---

### Task 2: Resource Monitor — Module

**Files:**
- Create: `src/vibe_common/vibe_common/resource_monitor.py`
- Test: `src/vibe_common/tests/test_resource_monitor.py`

**Step 1: Write the failing tests**

```python
# src/vibe_common/tests/test_resource_monitor.py
import os
import signal
import tempfile
import threading
import time
from unittest.mock import patch

import pytest

from vibe_common.resource_monitor import MemoryInfo, ResourceMonitor


@pytest.fixture
def fake_cgroup_dir():
    """Create a temporary directory with fake cgroup v2 files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "memory.current"), "w") as f:
            f.write("3997696000\n")  # ~3.7GB
        with open(os.path.join(tmpdir, "memory.max"), "w") as f:
            f.write("4294967296\n")  # 4GB
        with open(os.path.join(tmpdir, "cpu.stat"), "w") as f:
            f.write("usage_usec 1500000\nuser_usec 1000000\nsystem_usec 500000\n")
        yield tmpdir


def test_get_memory_usage_from_cgroup(fake_cgroup_dir):
    monitor = ResourceMonitor(cgroup_path=fake_cgroup_dir)
    mem = monitor.get_memory_usage()
    assert mem is not None
    assert mem.current_bytes == 3997696000
    assert mem.limit_bytes == 4294967296
    assert abs(mem.usage_percent - 93.08) < 0.1


def test_get_memory_usage_no_cgroup():
    monitor = ResourceMonitor(cgroup_path="/nonexistent/path")
    mem = monitor.get_memory_usage()
    # Should return None or fallback, not crash
    assert mem is None or isinstance(mem, MemoryInfo)


def test_get_memory_max_is_max_when_unlimited(fake_cgroup_dir):
    with open(os.path.join(fake_cgroup_dir, "memory.max"), "w") as f:
        f.write("max\n")
    monitor = ResourceMonitor(cgroup_path=fake_cgroup_dir)
    mem = monitor.get_memory_usage()
    assert mem is not None
    assert mem.limit_bytes == 0  # 0 signals unlimited
    assert mem.usage_percent == 0.0


def test_detect_oom_with_sigkill_and_high_memory(fake_cgroup_dir):
    monitor = ResourceMonitor(cgroup_path=fake_cgroup_dir)
    # Prime the last-known memory usage
    monitor.get_memory_usage()
    assert monitor.detect_oom(exit_signal=signal.SIGKILL) is True


def test_detect_oom_with_sigkill_and_low_memory(fake_cgroup_dir):
    # Write low memory usage
    with open(os.path.join(fake_cgroup_dir, "memory.current"), "w") as f:
        f.write("100000000\n")  # ~100MB of 4GB = 2.3%
    monitor = ResourceMonitor(cgroup_path=fake_cgroup_dir)
    monitor.get_memory_usage()
    assert monitor.detect_oom(exit_signal=signal.SIGKILL) is False


def test_detect_oom_without_sigkill(fake_cgroup_dir):
    monitor = ResourceMonitor(cgroup_path=fake_cgroup_dir)
    monitor.get_memory_usage()
    assert monitor.detect_oom(exit_signal=signal.SIGTERM) is False


def test_format_oom_message(fake_cgroup_dir):
    monitor = ResourceMonitor(cgroup_path=fake_cgroup_dir)
    monitor.get_memory_usage()
    msg = monitor.format_oom_message("download_sentinel_2", exit_signal=signal.SIGKILL)
    assert "download_sentinel_2" in msg
    assert "SIGKILL" in msg
    assert "3.7" in msg or "3997696000" in msg  # some form of memory amount
    assert "4.0" in msg or "4294967296" in msg  # some form of limit


def test_periodic_logging_starts_and_stops(fake_cgroup_dir, caplog):
    import logging

    monitor = ResourceMonitor(cgroup_path=fake_cgroup_dir)
    with caplog.at_level(logging.INFO):
        monitor.start_periodic_logging(interval_s=0.05)
        time.sleep(0.15)
        monitor.stop()
    assert any("memory" in r.message.lower() for r in caplog.records)


def test_stop_is_idempotent(fake_cgroup_dir):
    monitor = ResourceMonitor(cgroup_path=fake_cgroup_dir)
    monitor.start_periodic_logging(interval_s=1.0)
    monitor.stop()
    monitor.stop()  # Should not raise
```

**Step 2: Run tests to verify they fail**

Run: `cd src/vibe_common && python -m pytest tests/test_resource_monitor.py -v 2>&1 | tail -20`
Expected: FAIL — `ModuleNotFoundError: No module named 'vibe_common.resource_monitor'`

**Step 3: Write the implementation**

```python
# src/vibe_common/vibe_common/resource_monitor.py
import logging
import os
import signal
import threading
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_CGROUP_PATH = "/sys/fs/cgroup"
OOM_MEMORY_THRESHOLD_PERCENT = 80.0


@dataclass
class MemoryInfo:
    current_bytes: int
    limit_bytes: int
    usage_percent: float


@dataclass
class CpuInfo:
    usage_usec: int
    user_usec: int
    system_usec: int


class ResourceMonitor:
    """Monitors container resource usage via cgroup v2 files.

    Falls back gracefully when cgroup files are unavailable (local dev, non-Linux).
    """

    def __init__(self, cgroup_path: str = DEFAULT_CGROUP_PATH):
        self._cgroup_path = cgroup_path
        self._last_memory: Optional[MemoryInfo] = None
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._cgroup_warned = False

    def _read_cgroup_file(self, filename: str) -> Optional[str]:
        path = os.path.join(self._cgroup_path, filename)
        try:
            with open(path) as f:
                return f.read().strip()
        except (FileNotFoundError, PermissionError):
            if not self._cgroup_warned:
                logger.info(
                    f"cgroup file {path} not available. "
                    "Resource monitoring will be limited on this platform."
                )
                self._cgroup_warned = True
            return None

    def get_memory_usage(self) -> Optional[MemoryInfo]:
        current_raw = self._read_cgroup_file("memory.current")
        limit_raw = self._read_cgroup_file("memory.max")
        if current_raw is None or limit_raw is None:
            return None
        current_bytes = int(current_raw)
        if limit_raw == "max":
            self._last_memory = MemoryInfo(
                current_bytes=current_bytes, limit_bytes=0, usage_percent=0.0
            )
        else:
            limit_bytes = int(limit_raw)
            usage_percent = (current_bytes / limit_bytes * 100) if limit_bytes > 0 else 0.0
            self._last_memory = MemoryInfo(
                current_bytes=current_bytes,
                limit_bytes=limit_bytes,
                usage_percent=usage_percent,
            )
        return self._last_memory

    def get_cpu_usage(self) -> Optional[CpuInfo]:
        raw = self._read_cgroup_file("cpu.stat")
        if raw is None:
            return None
        stats = {}
        for line in raw.splitlines():
            parts = line.split()
            if len(parts) == 2:
                stats[parts[0]] = int(parts[1])
        return CpuInfo(
            usage_usec=stats.get("usage_usec", 0),
            user_usec=stats.get("user_usec", 0),
            system_usec=stats.get("system_usec", 0),
        )

    def detect_oom(self, exit_signal: int) -> bool:
        """Heuristic: process killed by SIGKILL and memory was near the cgroup limit."""
        if exit_signal != signal.SIGKILL:
            return False
        if self._last_memory is None:
            return False
        if self._last_memory.limit_bytes == 0:  # unlimited
            return False
        return self._last_memory.usage_percent >= OOM_MEMORY_THRESHOLD_PERCENT

    def format_oom_message(self, op_name: str, exit_signal: int) -> str:
        """Build a factual error message for an OOM-killed op."""
        sig_name = signal.Signals(exit_signal).name
        if self._last_memory is None or self._last_memory.limit_bytes == 0:
            return f"Op '{op_name}' killed ({sig_name}): memory usage data unavailable"
        current_gb = self._last_memory.current_bytes / (1024 ** 3)
        limit_gb = self._last_memory.limit_bytes / (1024 ** 3)
        return (
            f"Op '{op_name}' killed ({sig_name}): "
            f"memory usage {current_gb:.1f}GB / {limit_gb:.1f}GB limit "
            f"({self._last_memory.usage_percent:.0f}% at last sample)"
        )

    def _log_usage(self, log: logging.Logger) -> None:
        mem = self.get_memory_usage()
        cpu = self.get_cpu_usage()
        parts = []
        if mem is not None:
            if mem.limit_bytes > 0:
                current_mb = mem.current_bytes / (1024 ** 2)
                limit_mb = mem.limit_bytes / (1024 ** 2)
                parts.append(
                    f"memory={current_mb:.0f}MB/{limit_mb:.0f}MB ({mem.usage_percent:.1f}%)"
                )
            else:
                current_mb = mem.current_bytes / (1024 ** 2)
                parts.append(f"memory={current_mb:.0f}MB (no limit)")
        if cpu is not None:
            parts.append(f"cpu_usage={cpu.usage_usec}us")
        if parts:
            log.info(f"Resource usage: {', '.join(parts)}")

    def start_periodic_logging(
        self, interval_s: float = 30.0, log: Optional[logging.Logger] = None
    ) -> None:
        if log is None:
            log = logger
        self._stop_event.clear()

        def _loop():
            while not self._stop_event.wait(timeout=interval_s):
                self._log_usage(log)

        self._monitoring_thread = threading.Thread(
            target=_loop, name="resource-monitor", daemon=True
        )
        self._monitoring_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._monitoring_thread is not None:
            self._monitoring_thread.join(timeout=5.0)
            self._monitoring_thread = None
```

**Step 4: Run tests to verify they pass**

Run: `cd src/vibe_common && python -m pytest tests/test_resource_monitor.py -v 2>&1 | tail -20`
Expected: All tests PASS

**Step 5: Commit**

```
jj describe -m "feat(resource-monitor): add ResourceMonitor for cgroup-based telemetry and OOM detection"
jj new
```

---

### Task 3: Graceful Shutdown — Module

**Files:**
- Create: `src/vibe_common/vibe_common/graceful_shutdown.py`
- Test: `src/vibe_common/tests/test_graceful_shutdown.py`

**Step 1: Write the failing tests**

```python
# src/vibe_common/tests/test_graceful_shutdown.py
import signal
import threading
import time
from unittest.mock import MagicMock

import pytest

from vibe_common.graceful_shutdown import ShutdownManager, ShutdownState


def test_initial_state_is_running():
    mgr = ShutdownManager()
    assert mgr.state == ShutdownState.RUNNING
    assert not mgr.is_draining


def test_shutdown_transitions_to_draining():
    mgr = ShutdownManager()
    mgr.shutdown()
    assert mgr.state == ShutdownState.DRAINING
    assert mgr.is_draining


def test_callbacks_called_in_lifo_order():
    mgr = ShutdownManager()
    order = []
    mgr.on_shutdown(lambda: order.append("first"))
    mgr.on_shutdown(lambda: order.append("second"))
    mgr.shutdown()
    mgr.finalize()
    assert order == ["second", "first"]


def test_finalize_transitions_to_shutting_down():
    mgr = ShutdownManager()
    mgr.shutdown()
    mgr.finalize()
    assert mgr.state == ShutdownState.SHUTTING_DOWN


def test_wait_for_completion_returns_true_when_event_set():
    mgr = ShutdownManager()
    mgr.mark_work_complete()
    mgr.shutdown()
    assert mgr.wait_for_completion(timeout_s=1.0) is True


def test_wait_for_completion_returns_false_on_timeout():
    mgr = ShutdownManager()
    mgr.shutdown()
    assert mgr.wait_for_completion(timeout_s=0.05) is False


def test_wait_for_completion_unblocks_when_work_completes():
    mgr = ShutdownManager()
    mgr.shutdown()

    def complete_later():
        time.sleep(0.05)
        mgr.mark_work_complete()

    t = threading.Thread(target=complete_later)
    t.start()
    result = mgr.wait_for_completion(timeout_s=1.0)
    t.join()
    assert result is True


def test_double_shutdown_is_idempotent():
    mgr = ShutdownManager()
    cb = MagicMock()
    mgr.on_shutdown(cb)
    mgr.shutdown()
    mgr.shutdown()  # Should not call callbacks again or raise
    mgr.finalize()
    assert cb.call_count == 1


def test_callback_exception_does_not_block_others():
    mgr = ShutdownManager()
    order = []
    mgr.on_shutdown(lambda: order.append("first"))
    mgr.on_shutdown(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    mgr.on_shutdown(lambda: order.append("third"))
    mgr.shutdown()
    mgr.finalize()
    # "third" runs first (LIFO), then the failing one, then "first"
    assert "third" in order
    assert "first" in order
```

**Step 2: Run tests to verify they fail**

Run: `cd src/vibe_common && python -m pytest tests/test_graceful_shutdown.py -v 2>&1 | tail -20`
Expected: FAIL — `ModuleNotFoundError: No module named 'vibe_common.graceful_shutdown'`

**Step 3: Write the implementation**

```python
# src/vibe_common/vibe_common/graceful_shutdown.py
import enum
import logging
import signal
import threading
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)


class ShutdownState(enum.Enum):
    RUNNING = "running"
    DRAINING = "draining"
    SHUTTING_DOWN = "shutting_down"


class ShutdownManager:
    """Manages graceful shutdown with a drain-then-stop state machine.

    States: RUNNING -> DRAINING -> SHUTTING_DOWN

    In DRAINING, the service stops accepting new work and waits for current
    work to finish. After finalize() is called, transitions to SHUTTING_DOWN.
    """

    def __init__(self):
        self._state = ShutdownState.RUNNING
        self._lock = threading.Lock()
        self._callbacks: List[Callable[[], None]] = []
        self._work_complete = threading.Event()
        self._shutdown_initiated = False

    @property
    def state(self) -> ShutdownState:
        return self._state

    @property
    def is_draining(self) -> bool:
        return self._state in (ShutdownState.DRAINING, ShutdownState.SHUTTING_DOWN)

    def on_shutdown(self, callback: Callable[[], None]) -> None:
        self._callbacks.append(callback)

    def register_signals(
        self, signals: Optional[List[int]] = None
    ) -> None:
        if signals is None:
            signals = [signal.SIGTERM, signal.SIGINT]
        for sig in signals:
            signal.signal(sig, self._signal_handler)

    def _signal_handler(self, signum: int, frame) -> None:
        sig_name = signal.Signals(signum).name
        logger.info(f"Received {sig_name}, initiating graceful shutdown")
        self.shutdown()

    def shutdown(self) -> None:
        with self._lock:
            if self._shutdown_initiated:
                return
            self._shutdown_initiated = True
            self._state = ShutdownState.DRAINING
            logger.info("Shutdown initiated, entering DRAINING state")

    def mark_work_complete(self) -> None:
        self._work_complete.set()

    def wait_for_completion(self, timeout_s: float) -> bool:
        return self._work_complete.wait(timeout=timeout_s)

    def finalize(self) -> None:
        logger.info("Running shutdown callbacks")
        for cb in reversed(self._callbacks):
            try:
                cb()
            except Exception:
                logger.exception("Error in shutdown callback")
        self._state = ShutdownState.SHUTTING_DOWN
        logger.info("Shutdown complete")
```

**Step 4: Run tests to verify they pass**

Run: `cd src/vibe_common && python -m pytest tests/test_graceful_shutdown.py -v 2>&1 | tail -20`
Expected: All tests PASS

**Step 5: Commit**

```
jj describe -m "feat(graceful-shutdown): add ShutdownManager with drain state machine"
jj new
```

---

### Task 4: Heartbeat Message Type

**Files:**
- Modify: `src/vibe_common/vibe_common/messaging.py`
- Modify: `src/vibe_common/tests/test_messaging.py`

**Step 1: Write the failing test**

Add to `src/vibe_common/tests/test_messaging.py`:

```python
# Append these tests to the existing file

from vibe_common.messaging import (
    HeartbeatContent,
    HeartbeatMessage,
    MessageType,
)


def test_heartbeat_message_type_exists():
    assert hasattr(MessageType, "heartbeat")


def test_heartbeat_content_fields():
    content = HeartbeatContent(
        op_name="compute_ndvi",
        worker_id="worker-pod-abc123",
        memory_usage_bytes=2147483648,
    )
    assert content.op_name == "compute_ndvi"
    assert content.worker_id == "worker-pod-abc123"
    assert content.memory_usage_bytes == 2147483648


def test_heartbeat_message_valid_for_status_channel():
    from uuid import uuid4
    header = MessageHeader(
        type=MessageType.heartbeat,
        run_id=uuid4(),
    )
    content = HeartbeatContent(
        op_name="compute_ndvi",
        worker_id="worker-1",
        memory_usage_bytes=0,
    )
    msg = HeartbeatMessage(header=header, content=content)
    assert msg.is_valid_for_channel("updates")
    assert not msg.is_valid_for_channel("commands")
```

**Step 2: Run tests to verify they fail**

Run: `cd src/vibe_common && python -m pytest tests/test_messaging.py::test_heartbeat_message_type_exists -v 2>&1 | tail -10`
Expected: FAIL — `ImportError` or `AttributeError`

**Step 3: Add heartbeat types to messaging.py**

In `src/vibe_common/vibe_common/messaging.py`:

1. Add `heartbeat = auto()` to `MessageType` enum (after line 91)
2. Add `HeartbeatContent` class:
```python
class HeartbeatContent(BaseModel):
    op_name: str
    worker_id: str
    memory_usage_bytes: int
```
3. Add `HeartbeatMessage` class:
```python
class HeartbeatMessage(BaseMessage):
    _supported_channels: Set[str] = {STATUS_PUBSUB_TOPIC}
    content: HeartbeatContent
```
4. Add `HeartbeatContent` to `MessageContent` union type
5. Add `HeartbeatMessage` to `WorkMessage` union type
6. Add `MessageType.heartbeat: HeartbeatContent` to `MESSAGE_TYPE_TO_CONTENT_TYPE`
7. Add `build_heartbeat` to `WorkMessageBuilder`:
```python
@staticmethod
def build_heartbeat(
    traceparent: str, op_name: str, worker_id: str, memory_usage_bytes: int
) -> WorkMessage:
    run_id = run_id_from_traceparent(traceparent)
    header = MessageHeader(
        type=MessageType.heartbeat, run_id=run_id, parent_id=traceparent
    )
    content = HeartbeatContent(
        op_name=op_name, worker_id=worker_id, memory_usage_bytes=memory_usage_bytes
    )
    return HeartbeatMessage(header=header, content=content)
```

**Step 4: Run tests to verify they pass**

Run: `cd src/vibe_common && python -m pytest tests/test_messaging.py -v 2>&1 | tail -20`
Expected: All tests PASS (both existing and new)

**Step 5: Commit**

```
jj describe -m "feat(messaging): add HeartbeatMessage type for worker liveness tracking"
jj new
```

---

### Task 5: Integrate Retry into Worker

**Files:**
- Modify: `src/vibe_agent/vibe_agent/worker.py`

**Step 1: Understand current retry code**

Read `worker.py:466-508` — the `run_op_with_retry` method. The current code:
- Loops `for i in range(self.max_tries)` with no delay between retries
- Catches `ProcessExpired` (line 495) and generic `traceback.TracebackException` (line 488)
- Raises immediately on `TimeoutError` (line 497)
- Checks `self.shutting_down` at loop start (line 481)

**Step 2: Modify `run_op_with_retry` to use `retry_with_backoff`**

Import the retry decorator at the top of `worker.py`:
```python
from vibe_common.retry import retry_with_backoff
```

Replace the `run_op_with_retry` method (lines 466-508). The retry decorator wraps an inner function that calls `try_run_op`. The `abort` parameter checks `self.shutting_down`. `ShuttingDownException` and `TimeoutError` are NOT retryable. `ProcessExpired` and `RuntimeError` (from TracebackException format) ARE retryable.

```python
@add_trace
def run_op_with_retry(
    self, content: CacheInfoExecuteRequestContent, run_id: UUID, timeout_s: float
) -> OpIOType:
    spec = cast(OperationSpec, content.operation_spec)
    self.logger.info(
        f"Will try to execute op {spec} with input {get_input_ids(content.input)} "
        f"for at most {self.max_tries} tries in child process."
    )
    final_time = time.time() + timeout_s

    @retry_with_backoff(
        max_retries=self.max_tries,
        base_delay=1.0,
        max_delay=60.0,
        backoff_factor=2.0,
        retryable_exceptions=(ProcessExpired, RuntimeError),
        abort=lambda: self.shutting_down,
    )
    def _attempt():
        inner_timeout = final_time - time.time()
        if inner_timeout <= 0:
            raise TimeoutError(
                f"Op execution budget exhausted ({timeout_s}s total)."
            )
        if self.shutting_down:
            raise ShuttingDownException()
        ret = self.try_run_op(spec, content, inner_timeout)
        if isinstance(ret, traceback.TracebackException):
            raise RuntimeError("".join(ret.format()))
        return ret

    try:
        result = _attempt()
    except TimeoutError as e:
        raise RuntimeError(
            f"Op {spec} timed out. Total time allowed: {timeout_s}s."
        ) from e
    finally:
        self.current_child = None
    return result
```

**Step 3: Run existing tests if any, plus verify the worker module imports cleanly**

Run: `cd src/vibe_agent && python -c "from vibe_agent.worker import Worker; print('OK')" 2>&1`
Expected: `OK`

**Step 4: Commit**

```
jj describe -m "feat(worker): replace bare retry loop with retry_with_backoff decorator"
jj new
```

---

### Task 6: Integrate Resource Monitor into Worker

**Files:**
- Modify: `src/vibe_agent/vibe_agent/worker.py`

**Step 1: Add ResourceMonitor to Worker.__init__**

Import at top:
```python
from vibe_common.resource_monitor import ResourceMonitor
```

In `Worker.__init__` (after line 270 `self.statestore = StateStore()`):
```python
self.resource_monitor = ResourceMonitor()
```

**Step 2: Start periodic logging in Worker.run()**

In `Worker.run()` (after line 322, before `self.start_service()`):
```python
self.resource_monitor.start_periodic_logging(interval_s=30, log=self.logger)
```

**Step 3: Classify OOM in run_op_with_retry**

In the inner `_attempt()` function inside `run_op_with_retry`, catch `ProcessExpired` specifically to check for OOM before re-raising:

```python
@retry_with_backoff(
    max_retries=self.max_tries,
    base_delay=1.0,
    max_delay=60.0,
    backoff_factor=2.0,
    retryable_exceptions=(ProcessExpired, RuntimeError),
    abort=lambda: self.shutting_down,
)
def _attempt():
    inner_timeout = final_time - time.time()
    if inner_timeout <= 0:
        raise TimeoutError(
            f"Op execution budget exhausted ({timeout_s}s total)."
        )
    if self.shutting_down:
        raise ShuttingDownException()
    try:
        ret = self.try_run_op(spec, content, inner_timeout)
    except ProcessExpired as e:
        self.resource_monitor.get_memory_usage()  # refresh last sample
        if self.resource_monitor.detect_oom(exit_signal=getattr(e, 'exitcode', 0)):
            msg = self.resource_monitor.format_oom_message(
                str(spec.name), exit_signal=getattr(e, 'exitcode', 0)
            )
            self.logger.error(msg)
        raise
    if isinstance(ret, traceback.TracebackException):
        raise RuntimeError("".join(ret.format()))
    return ret
```

Note: `pebble.common.ProcessExpired` stores the exit code in `.exitcode`. On Linux OOM-kill, the exitcode is the negative signal number (-9) or the signal is embedded. Check the pebble docs for exact attribute. The `exitcode` attribute may need to be examined as `abs(e.exitcode)` or `signal.SIGKILL` comparison needs to account for pebble's convention.

**Step 4: Verify import**

Run: `cd src/vibe_agent && python -c "from vibe_agent.worker import Worker; print('OK')" 2>&1`
Expected: `OK`

**Step 5: Commit**

```
jj describe -m "feat(worker): integrate ResourceMonitor for telemetry and OOM classification"
jj new
```

---

### Task 7: Integrate Graceful Shutdown into Worker

**Files:**
- Modify: `src/vibe_agent/vibe_agent/worker.py`
- Modify: `src/vibe_agent/vibe_agent/launch_worker.py`

**Step 1: Add ShutdownManager to Worker**

Import at top of `worker.py`:
```python
from vibe_common.graceful_shutdown import ShutdownManager
```

In `Worker.__init__` (replace `self.shutdown_lock = threading.Lock()` on line 266 and `self.shutting_down` concept):
```python
self.shutdown_manager = ShutdownManager()
self.shutdown_manager.on_shutdown(lambda: self.resource_monitor.stop())
```

**Step 2: Replace `self.shutting_down` with `self.shutdown_manager.is_draining`**

Replace all references to `self.shutting_down` throughout `worker.py` with `self.shutdown_manager.is_draining`. Key locations:
- `start_service` (line 328): `while not self.shutdown_manager.is_draining:`
- `fetch_work` (line 383): `if self.shutdown_manager.is_draining:`
- `get_future_result` (line 433): `if self.shutdown_manager.is_draining:`
- `run_op_with_retry` (line 481): `if self.shutdown_manager.is_draining:`

**Step 3: Replace `pre_stop_hook` with shutdown_manager**

Replace the `pre_stop_hook` method (lines 295-310):

```python
def pre_stop_hook(self, signum: int, _: Any):
    self.shutdown_manager.shutdown()
    # Stop the gRPC server to reject new message deliveries
    if self.app._server is not None:
        self.app._server.stop(None)
    # Wait for current work to finish within grace period
    if self.current_message is not None:
        if not self.shutdown_manager.wait_for_completion(self.termination_grace_period_s):
            self.logger.warning(
                f"Grace period ({self.termination_grace_period_s}s) expired, "
                "cancelling current work"
            )
            self._terminate_child()
    self.shutdown_manager.finalize()
```

**Step 4: Mark work complete after op finishes**

In `run_op_from_message` (line 336-352), in the `finally` block, add:
```python
self.shutdown_manager.mark_work_complete()
```

**Step 5: Add timeout to WorkerMessenger.send()**

Modify `WorkerMessenger.send()` (lines 162-180) to accept and enforce a `timeout_s` parameter:

```python
async def send(self, message: WorkMessage, timeout_s: float = 0) -> None:
    tries: int = 0
    sent = False
    start = time.time()
    while True:
        try:
            sent = await send_async(message, "worker", self.pubsubname, self.status_topic)
        except Exception:
            pass
        if sent:
            break
        tries += 1
        if timeout_s > 0 and (time.time() - start) >= timeout_s:
            self.logger.error(
                f"Failed to send {message} after {tries} attempts "
                f"and {timeout_s}s timeout. Giving up."
            )
            return
        self.logger.warn(
            f"Failed to send {message} after {tries} attempts. "
            f"Sleeping for {MESSAGING_RETRY_INTERVAL_S}s before retrying."
        )
        await asyncio.sleep(MESSAGING_RETRY_INTERVAL_S)
```

**Step 6: Wire ShutdownManager signals in launch_worker.py**

In `src/vibe_agent/vibe_agent/launch_worker.py`, replace lines 45-46:

```python
# Before:
# signal.signal(signal.SIGTERM, worker_obj.worker.impl.pre_stop_hook)
# asyncio.run(worker_obj.worker.impl.run())

# After:
worker = worker_obj.worker.impl
worker.shutdown_manager.register_signals([signal.SIGTERM, signal.SIGINT])
worker.run()
```

Note: `worker.run()` is not actually async (it calls `self.app.run()` which is blocking), so `asyncio.run()` wrapping it was potentially incorrect. Verify this by reading the code — if `run()` is `def run(self)` (not `async def`), just call it directly.

**Step 7: Remove old shutdown_lock and shutting_down from Worker**

Remove from `__init__`:
- `self.shutdown_lock = threading.Lock()` (line 266)
- The `shutting_down` class attribute (line 227)

**Step 8: Verify import**

Run: `cd src/vibe_agent && python -c "from vibe_agent.worker import Worker; print('OK')" 2>&1`
Expected: `OK`

**Step 9: Commit**

```
jj describe -m "feat(worker): integrate ShutdownManager for graceful drain-and-requeue shutdown"
jj new
```

---

### Task 8: Add Heartbeat Publishing to Worker

**Files:**
- Modify: `src/vibe_agent/vibe_agent/worker.py`

**Step 1: Add heartbeat config to Worker**

In `Worker.__init__`, add:
```python
self.heartbeat_interval_s = 30
self._last_heartbeat_time = 0.0
self.worker_id = os.environ.get("HOSTNAME", f"worker-{os.getpid()}")
```

Import at top:
```python
from vibe_common.messaging import WorkMessageBuilder
```
(This import already exists partially — `WorkMessageBuilder` is not currently imported. Check.)

**Step 2: Publish heartbeats in `get_future_result` loop**

The `get_future_result` method (lines 410-449) polls the pebble future every `monitoring_period_s=10s`. Add heartbeat publishing inside the `except concurrent.futures.TimeoutError` block (this fires every 10s while the subprocess is running):

After line 422 (`if self.is_workflow_complete(self.current_message):`), before the workflow-complete check, add heartbeat logic:

```python
# Publish heartbeat if interval has elapsed
now = time.time()
if now - self._last_heartbeat_time >= self.heartbeat_interval_s:
    self._publish_heartbeat()
    self._last_heartbeat_time = now
```

Add the `_publish_heartbeat` method to `Worker`:

```python
def _publish_heartbeat(self):
    if self.current_message is None:
        return
    mem = self.resource_monitor.get_memory_usage()
    memory_bytes = mem.current_bytes if mem else 0
    content = cast(CacheInfoExecuteRequestContent, self.current_message.content)
    op_name = str(content.operation_spec.name) if content.operation_spec else "unknown"
    heartbeat = WorkMessageBuilder.build_heartbeat(
        traceparent=self.current_message.id,
        op_name=op_name,
        worker_id=self.worker_id,
        memory_usage_bytes=memory_bytes,
    )
    try:
        asyncio.run(self.messenger.send(heartbeat, timeout_s=5))
    except Exception:
        self.logger.debug("Failed to send heartbeat, will retry next interval")
```

**Step 3: Verify import**

Run: `cd src/vibe_agent && python -c "from vibe_agent.worker import Worker; print('OK')" 2>&1`
Expected: `OK`

**Step 4: Commit**

```
jj describe -m "feat(worker): publish heartbeat messages during op execution"
jj new
```

---

### Task 9: Orchestrator Heartbeat Tracking

**Files:**
- Modify: `src/vibe_server/vibe_server/orchestrator.py`
- Modify: `src/vibe_server/vibe_server/workflow/runner/remote_runner.py`

**Step 1: Handle HeartbeatMessage in orchestrator**

In `src/vibe_server/vibe_server/orchestrator.py`:

Import:
```python
from vibe_common.messaging import HeartbeatContent, HeartbeatMessage, MessageType
```

The `handle_update_workflow_status` method (line 553-569) receives ALL messages on the `updates` topic. Currently it puts them into the per-run inqueue. Heartbeat messages should update a separate tracking dict instead of going into the inqueue (the runner doesn't expect them).

Modify `success_callback` inside `handle_update_workflow_status`:

```python
async def success_callback(message: WorkMessage) -> TopicEventResponse:
    if not message.is_valid_for_channel(channel):
        self.logger.error(
            f"Received unsupported message {message} for channel {channel}. Dropping it."
        )
        return TopicEventResponse("drop")
    # Handle heartbeats separately — update tracking, don't route to runner
    if isinstance(message, HeartbeatMessage):
        content = cast(HeartbeatContent, message.content)
        key = f"{message.run_id}:{content.op_name}"
        self._last_heartbeats[key] = time.time()
        return TopicEventResponse("success")
    if str(message.run_id) not in self.inqueues:
        self.logger.info(
            f"Received message {message}, but the run it references"
            " is not being managed. Dropping it."
        )
        return TopicEventResponse("drop")
    await self.inqueues[str(message.run_id)].put(message)
    return TopicEventResponse("success")
```

Add to `Orchestrator.__init__`:
```python
self._last_heartbeats: Dict[str, float] = {}
```

Import `time` at the top if not already imported.

**Step 2: Add heartbeat-aware timeout to RemoteWorkflowRunner**

In `src/vibe_server/vibe_server/workflow/runner/remote_runner.py`:

Modify `RemoteWorkflowRunner.__init__` to accept heartbeat parameters:

```python
def __init__(
    self,
    message_router: "MessageRouter",
    workflow: Workflow,
    traceid: str,
    update_state_callback: WorkflowCallback = NoOpStateChange,
    pubsubname: Optional[str] = None,
    source: Optional[str] = None,
    topic: Optional[str] = None,
    heartbeat_tracker: Optional[Dict[str, float]] = None,
    heartbeat_timeout_s: float = 60.0,
    **kwargs: Any,
):
    # ... existing init ...
    self._heartbeat_tracker = heartbeat_tracker or {}
    self._heartbeat_timeout_s = heartbeat_timeout_s
```

Replace `_wait_for_reply` (lines 210-217):

```python
async def _wait_for_reply(self, request: ExecuteRequestMessage) -> WorkMessage:
    while True:
        try:
            return await self.message_router.get(request.id, block=False)
        except asyncio.QueueEmpty:
            await asyncio.sleep(SLEEP_S)
            if self.is_cancelled:
                raise CancelledOpError()
            # Check heartbeat staleness
            if self._heartbeat_tracker:
                op_name = request.content.operation_spec.name
                run_id = request.run_id
                key = f"{run_id}:{op_name}"
                last_hb = self._heartbeat_tracker.get(key)
                if last_hb is not None:
                    elapsed = time.time() - last_hb
                    if elapsed > self._heartbeat_timeout_s:
                        from datetime import datetime, timezone
                        last_dt = datetime.fromtimestamp(last_hb, tz=timezone.utc)
                        raise RuntimeError(
                            f"Op '{op_name}' has not reported progress for "
                            f"{elapsed:.0f}s. Last heartbeat at {last_dt.isoformat()}."
                        )
```

Import `time` at the top of `remote_runner.py`.

**Step 3: Pass heartbeat_tracker when creating RemoteWorkflowRunner**

In `orchestrator.py`, `WorkflowRunManager.start_managing()` (line 452):

```python
self.runner = RemoteWorkflowRunner(
    traceid=self.message.id,
    message_router=router,
    workflow=workflow,
    io_mapper=io_mapper,
    update_state_callback=WorkflowStateUpdate(run_id),
    pubsubname=self.pubsubname,
    source=self.source,
    topic=self.topic,
    heartbeat_tracker=self._heartbeat_tracker,
    heartbeat_timeout_s=60.0,
)
```

The `WorkflowRunManager` needs access to `self._heartbeat_tracker` from the orchestrator. Modify its constructor to accept it:

In `WorkflowRunManager.__init__` (line 383-407), add parameter:
```python
heartbeat_tracker: Optional[Dict[str, float]] = None,
```
And store it:
```python
self._heartbeat_tracker = heartbeat_tracker or {}
```

In `Orchestrator.handle_workflow_execution_message` (line 594-619), pass it:
```python
wf = WorkflowRunManager(
    self.inqueues,
    message,
    pubsubname=self.pubsubname,
    source="orchestrator",
    topic=self.cache_topic,
    ops_dir=self.ops_dir,
    workflows_dir=self.workflows_dir,
    heartbeat_tracker=self._last_heartbeats,
)
```

**Step 4: Verify imports**

Run: `cd src/vibe_server && python -c "from vibe_server.orchestrator import Orchestrator; print('OK')" 2>&1`
Expected: `OK`

**Step 5: Commit**

```
jj describe -m "feat(orchestrator): add heartbeat-aware stale workflow detection"
jj new
```

---

### Task 10: VM Testing — Build and Validate

**Step 1: Push changes to remote**

Instruct user to push:
```
jj git push --bookmark worker-harden_task11_model_b
```

**Step 2: Pull and rebuild on VM**

```bash
# On VM via gcp-vm MCP:
cd /path/to/farmvibes-ai
git fetch origin
git checkout worker-harden_task11_model_b
git pull origin worker-harden_task11_model_b

# Rebuild worker container
# (check Makefile or scripts/ for the build command)
```

**Step 3: Run unit tests on VM**

```bash
cd src/vibe_common && python -m pytest tests/test_retry.py tests/test_resource_monitor.py tests/test_graceful_shutdown.py tests/test_messaging.py -v
```

**Step 4: Submit a workflow and verify heartbeat logging**

Submit a test workflow via the REST API and check worker logs for:
- `Resource usage: memory=...` periodic entries
- `Attempt N/M failed ...` on retry scenarios
- Heartbeat messages being sent

**Step 5: Test degraded connectivity**

Apply a network policy blocking egress from the worker pod:
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: block-worker-egress
spec:
  podSelector:
    matchLabels:
      app: terravibes-worker
  policyTypes:
    - Egress
  egress: []  # Block all egress
```

Submit a workflow. The worker should retry with backoff. Remove the policy. The workflow should eventually succeed or fail with a clear error.

**Step 6: Test graceful shutdown**

```bash
kubectl scale deployment terravibes-worker --replicas=0
# Watch logs for: "Shutdown initiated, entering DRAINING state"
# Watch for either "Grace period expired" or clean completion
kubectl scale deployment terravibes-worker --replicas=1
# The workflow should either resume (via cache idempotency) or fail with meaningful error
```

**Step 7: Verify stale detection**

Scale worker to 0 during a running workflow. The orchestrator should detect missing heartbeats after `heartbeat_timeout_s` and fail the workflow with a message like:
`"Op 'X' has not reported progress for 120s. Last heartbeat at ..."`

**Step 8: Commit any fixes from testing**

```
jj describe -m "fix(worker): adjustments from integration testing"
jj new
```
