# Worker Service Hardening — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the worker resilient to transient failures, OOM, and shutdown signals; emit heartbeats so the orchestrator can eventually detect stuck work.

**Architecture:** Three new standalone utility modules in `vibe_common` (retry, resources, heartbeat message type). The worker (`vibe_agent/worker.py`) consumes all three. Shutdown switches from "kill immediately" to "drain until deadline, then kill." Terraform sets the K8s grace period so the pod isn't SIGKILLed before the Python deadline fires.

**Tech Stack:** Python 3.8+, pebble (subprocess pool), Dapr gRPC, pytest + unittest.mock, Hydra/hydra-zen config, jj for VCS. No new dependencies.

**Reference docs:**
- Design: `docs/plans/2026-02-26-worker-hardening-design.md`
- Worker source: `src/vibe_agent/vibe_agent/worker.py`
- Messaging: `src/vibe_common/vibe_common/messaging.py`
- Dapr resiliency config: `src/vibe_core/vibe_core/terraform/local/modules/kubernetes/dapr.tf:97-123`
- Existing OOM troubleshooting entry: `docs/source/docfiles/markdown/TROUBLESHOOTING.md:153-173`

**Running tests:** `pytest src/vibe_common/tests/test_<file>.py -v` (pytest.ini sets `pythonpath=src`). Tests will fail locally without `pebble`/`dapr` installed — run on the VM inside the dev container, or install deps locally: `pip install -e src/vibe_common -e src/vibe_agent pebble dapr`.

---

## Phase 1: Utility modules (no worker dependency)

### Task 1: `vibe_common/retry.py` — backoff computation

**Files:**
- Create: `src/vibe_common/vibe_common/retry.py`
- Create: `src/vibe_common/tests/test_retry.py`

**Step 1: Write failing tests**

Create `src/vibe_common/tests/test_retry.py`:

```python
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
    # attempt 0 → base=2.0, jitter picks uniformly in [0, 2.0]
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
```

**Step 2: Run tests — expect FAIL**

```bash
pytest src/vibe_common/tests/test_retry.py -v
```
Expected: `ModuleNotFoundError: No module named 'vibe_common.retry'`

**Step 3: Implement**

Create `src/vibe_common/vibe_common/retry.py`:

```python
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
```

**Step 4: Run tests — expect PASS**

```bash
pytest src/vibe_common/tests/test_retry.py -v
```

**Step 5: Commit**

```bash
jj commit -m "feat(common): add RetryPolicy and compute_backoff with exponential + jitter"
```

---

### Task 2: `vibe_common/retry.py` — `@with_retry` decorator (sync)

**Files:**
- Modify: `src/vibe_common/vibe_common/retry.py`
- Modify: `src/vibe_common/tests/test_retry.py`

**Step 1: Write failing tests**

Append to `src/vibe_common/tests/test_retry.py`:

```python
from unittest.mock import MagicMock, call
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
    assert mock.call_count == 1  # no retry, predicate rejected


def test_with_retry_logs_each_attempt(caplog):
    import logging
    mock = MagicMock(side_effect=[ConnectionError("flaky"), "ok"])
    p = RetryPolicy(max_attempts=3, base_delay_s=0.001, jitter=False)
    wrapped = with_retry(p, logger=logging.getLogger("test"))(mock)
    with caplog.at_level(logging.WARNING):
        wrapped()
    assert any("attempt 1/3" in r.message and "ConnectionError" in r.message
               for r in caplog.records)
```

**Step 2: Run — expect FAIL**

```bash
pytest src/vibe_common/tests/test_retry.py -v -k with_retry
```
Expected: `ImportError: cannot import name 'with_retry'`

**Step 3: Implement**

Append to `src/vibe_common/vibe_common/retry.py`:

```python
def with_retry(
    policy: RetryPolicy,
    logger: Optional[logging.Logger] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator that retries the wrapped callable per `policy`.

    Works on sync and async callables. Logs WARNING on each failed attempt
    with the exception class and next delay. Re-raises the last exception
    when attempts are exhausted or the exception is not retryable.

    Example:
        @with_retry(RetryPolicy(max_attempts=5), logger)
        def fetch(url: str) -> bytes: ...
    """
    log = logger or logging.getLogger(__name__)

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        if asyncio.iscoroutinefunction(fn):
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
                            f"{fn.__name__}: attempt {attempt + 1}/{policy.max_attempts} "
                            f"failed with {type(e).__name__}: {e}. "
                            f"Retrying in {delay:.2f}s."
                        )
                        await asyncio.sleep(delay)
                assert last_exc is not None  # unreachable
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
                        f"{fn.__name__}: attempt {attempt + 1}/{policy.max_attempts} "
                        f"failed with {type(e).__name__}: {e}. "
                        f"Retrying in {delay:.2f}s."
                    )
                    time.sleep(delay)
            assert last_exc is not None  # unreachable
            raise last_exc
        return sync_wrapper

    return decorator
```

**Step 4: Run — expect PASS**

```bash
pytest src/vibe_common/tests/test_retry.py -v
```

**Step 5: Commit**

```bash
jj commit -m "feat(common): add @with_retry decorator (sync + async)"
```

---

### Task 3: `vibe_common/retry.py` — async path test

**Files:**
- Modify: `src/vibe_common/tests/test_retry.py`

**Step 1: Write failing test**

Append to `src/vibe_common/tests/test_retry.py`:

```python
import asyncio


@pytest.mark.anyio
async def test_with_retry_async_path():
    calls = []

    async def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise ConnectionError("transient")
        return "ok"

    p = RetryPolicy(max_attempts=5, base_delay_s=0.001, jitter=False)
    wrapped = with_retry(p)(flaky)
    result = await wrapped()
    assert result == "ok"
    assert len(calls) == 3
```

**Step 2: Run — should PASS** (implementation already handles async)

```bash
pytest src/vibe_common/tests/test_retry.py::test_with_retry_async_path -v
```

**Step 3: Commit**

```bash
jj commit -m "test(common): cover async path of @with_retry"
```

---

### Task 4: `vibe_common/resources.py` — cgroup memory reading

**Files:**
- Create: `src/vibe_common/vibe_common/resources.py`
- Create: `src/vibe_common/tests/test_resources.py`

**Step 1: Write failing tests**

Create `src/vibe_common/tests/test_resources.py`:

```python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from unittest.mock import mock_open, patch

import pytest
from vibe_common.resources import MemoryInfo, get_memory_info, is_oom_exitcode


def test_memory_info_computed_properties():
    info = MemoryInfo(limit_bytes=1024 * 1024 * 1024, usage_bytes=512 * 1024 * 1024)
    assert info.usage_mb == 512.0
    assert info.limit_mb == 1024.0
    assert info.usage_fraction == pytest.approx(0.5)


def test_memory_info_no_limit():
    info = MemoryInfo(limit_bytes=None, usage_bytes=100 * 1024 * 1024)
    assert info.limit_mb is None
    assert info.usage_fraction is None


def test_is_oom_exitcode_negative_sigkill():
    # multiprocessing/pebble report signal death as -signum
    assert is_oom_exitcode(-9)


def test_is_oom_exitcode_shell_convention():
    # shell exit codes: 128 + signal number
    assert is_oom_exitcode(137)


def test_is_oom_exitcode_normal_exit():
    assert not is_oom_exitcode(0)
    assert not is_oom_exitcode(1)
    assert not is_oom_exitcode(-2)  # SIGINT, not OOM


def test_get_memory_info_cgroup_v2():
    # cgroup v2: /sys/fs/cgroup/memory.max and memory.current
    def fake_exists(path):
        return path == "/sys/fs/cgroup/memory.max" or path == "/sys/fs/cgroup/memory.current"

    file_contents = {
        "/sys/fs/cgroup/memory.max": "2147483648\n",
        "/sys/fs/cgroup/memory.current": "536870912\n",
    }

    def fake_open(path, *a, **kw):
        return mock_open(read_data=file_contents[path])()

    with patch("os.path.exists", side_effect=fake_exists), \
         patch("builtins.open", side_effect=fake_open):
        info = get_memory_info()
        assert info.limit_bytes == 2147483648
        assert info.usage_bytes == 536870912


def test_get_memory_info_cgroup_v2_unlimited():
    # cgroup v2 uses "max" literal for no limit
    def fake_exists(path):
        return path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current")

    file_contents = {
        "/sys/fs/cgroup/memory.max": "max\n",
        "/sys/fs/cgroup/memory.current": "100000\n",
    }

    def fake_open(path, *a, **kw):
        return mock_open(read_data=file_contents[path])()

    with patch("os.path.exists", side_effect=fake_exists), \
         patch("builtins.open", side_effect=fake_open):
        info = get_memory_info()
        assert info.limit_bytes is None
        assert info.usage_bytes == 100000


def test_get_memory_info_cgroup_v1():
    # cgroup v1: /sys/fs/cgroup/memory/memory.limit_in_bytes and .usage_in_bytes
    def fake_exists(path):
        return path in (
            "/sys/fs/cgroup/memory/memory.limit_in_bytes",
            "/sys/fs/cgroup/memory/memory.usage_in_bytes",
        )

    file_contents = {
        "/sys/fs/cgroup/memory/memory.limit_in_bytes": "1073741824\n",
        "/sys/fs/cgroup/memory/memory.usage_in_bytes": "268435456\n",
    }

    def fake_open(path, *a, **kw):
        return mock_open(read_data=file_contents[path])()

    with patch("os.path.exists", side_effect=fake_exists), \
         patch("builtins.open", side_effect=fake_open):
        info = get_memory_info()
        assert info.limit_bytes == 1073741824
        assert info.usage_bytes == 268435456


def test_get_memory_info_cgroup_v1_huge_limit_treated_as_unlimited():
    # cgroup v1 uses a giant sentinel (~2^63) when no limit is set
    def fake_exists(path):
        return path in (
            "/sys/fs/cgroup/memory/memory.limit_in_bytes",
            "/sys/fs/cgroup/memory/memory.usage_in_bytes",
        )

    file_contents = {
        "/sys/fs/cgroup/memory/memory.limit_in_bytes": "9223372036854771712\n",
        "/sys/fs/cgroup/memory/memory.usage_in_bytes": "1000000\n",
    }

    def fake_open(path, *a, **kw):
        return mock_open(read_data=file_contents[path])()

    with patch("os.path.exists", side_effect=fake_exists), \
         patch("builtins.open", side_effect=fake_open):
        info = get_memory_info()
        assert info.limit_bytes is None
```

**Step 2: Run — expect FAIL**

```bash
pytest src/vibe_common/tests/test_resources.py -v
```
Expected: `ModuleNotFoundError: No module named 'vibe_common.resources'`

**Step 3: Implement**

Create `src/vibe_common/vibe_common/resources.py`:

```python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Container resource introspection.

Reads cgroup memory limit/usage directly. No psutil dependency — cgroup files
are plain text integers, and psutil is not a vibe_common dep.

cgroup v2 (current Kubernetes):
    /sys/fs/cgroup/memory.max      — limit in bytes, or the literal "max"
    /sys/fs/cgroup/memory.current  — current usage in bytes

cgroup v1 (older clusters):
    /sys/fs/cgroup/memory/memory.limit_in_bytes  — giant sentinel if unlimited
    /sys/fs/cgroup/memory/memory.usage_in_bytes

Fallback (not containerized): /proc/self/status VmRSS line.
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# cgroup v1 reports a near-2^63 value when unlimited. Anything bigger than
# this threshold is treated as "no limit." 2^62 = 4 exabytes — if a real
# limit ever approaches this we have bigger problems.
_UNLIMITED_THRESHOLD = 1 << 62

_CGROUP_V2_LIMIT = "/sys/fs/cgroup/memory.max"
_CGROUP_V2_USAGE = "/sys/fs/cgroup/memory.current"
_CGROUP_V1_LIMIT = "/sys/fs/cgroup/memory/memory.limit_in_bytes"
_CGROUP_V1_USAGE = "/sys/fs/cgroup/memory/memory.usage_in_bytes"
_PROC_STATUS = "/proc/self/status"


@dataclass
class MemoryInfo:
    limit_bytes: Optional[int]
    usage_bytes: int

    @property
    def usage_mb(self) -> float:
        return self.usage_bytes / (1024 * 1024)

    @property
    def limit_mb(self) -> Optional[float]:
        return self.limit_bytes / (1024 * 1024) if self.limit_bytes else None

    @property
    def usage_fraction(self) -> Optional[float]:
        if not self.limit_bytes:
            return None
        return self.usage_bytes / self.limit_bytes

    def __str__(self) -> str:
        if self.limit_mb is not None:
            return f"{self.usage_mb:.0f}MB / {self.limit_mb:.0f}MB ({self.usage_fraction:.0%})"
        return f"{self.usage_mb:.0f}MB (no limit)"


def _read_int(path: str) -> Optional[int]:
    try:
        with open(path) as f:
            raw = f.read().strip()
        if raw == "max":  # cgroup v2 unlimited sentinel
            return None
        value = int(raw)
        if value >= _UNLIMITED_THRESHOLD:  # cgroup v1 unlimited sentinel
            return None
        return value
    except (OSError, ValueError) as e:
        logger.debug(f"Couldn't read {path}: {e}")
        return None


def _read_proc_rss() -> int:
    """Parse VmRSS from /proc/self/status. Returns 0 if unavailable."""
    try:
        with open(_PROC_STATUS) as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    # Format: "VmRSS:\t  12345 kB"
                    parts = line.split()
                    return int(parts[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return 0


def get_memory_info() -> MemoryInfo:
    """Returns current memory usage and limit for this container/process.

    Prefers cgroup v2, falls back to v1, falls back to /proc/self/status RSS
    with no limit. Never raises — returns MemoryInfo(None, 0) in the worst case.
    """
    # cgroup v2
    if os.path.exists(_CGROUP_V2_LIMIT) and os.path.exists(_CGROUP_V2_USAGE):
        limit = _read_int(_CGROUP_V2_LIMIT)
        usage = _read_int(_CGROUP_V2_USAGE) or 0
        return MemoryInfo(limit_bytes=limit, usage_bytes=usage)

    # cgroup v1
    if os.path.exists(_CGROUP_V1_LIMIT) and os.path.exists(_CGROUP_V1_USAGE):
        limit = _read_int(_CGROUP_V1_LIMIT)
        usage = _read_int(_CGROUP_V1_USAGE) or 0
        return MemoryInfo(limit_bytes=limit, usage_bytes=usage)

    # not containerized or cgroup fs not mounted — best effort
    return MemoryInfo(limit_bytes=None, usage_bytes=_read_proc_rss())


def is_oom_exitcode(exitcode: int) -> bool:
    """True if the exit code indicates the process was SIGKILLed.

    In a container, SIGKILL from the kernel almost always means the OOM killer.
    Two conventions exist:
      - multiprocessing/pebble: negative signal number (-9)
      - shell: 128 + signal number (137)

    This can also mean someone ran `kill -9` manually, but that's rare in
    production and the error message is appropriately hedged ("likely").
    """
    return exitcode == -9 or exitcode == 137
```

**Step 4: Run — expect PASS**

```bash
pytest src/vibe_common/tests/test_resources.py -v
```

**Step 5: Commit**

```bash
jj commit -m "feat(common): add resources module for cgroup memory introspection and OOM detection"
```

---

### Task 5: Heartbeat message type in `vibe_common/messaging.py`

**Files:**
- Modify: `src/vibe_common/vibe_common/messaging.py`
- Modify: `src/vibe_common/tests/test_messaging.py`

**Step 1: Write failing test**

Append to `src/vibe_common/tests/test_messaging.py`:

```python
from vibe_common.messaging import (
    HeartbeatContent,
    HeartbeatMessage,
    MessageType,
    WorkMessageBuilder,
)


def test_heartbeat_message_construction(traceparent: str):
    msg = WorkMessageBuilder.build_heartbeat(
        traceparent,
        op_name="download_sentinel2",
        elapsed_s=42.5,
        memory_usage_mb=1024.0,
        memory_limit_mb=4096.0,
    )
    assert isinstance(msg, HeartbeatMessage)
    assert msg.header.type == MessageType.heartbeat
    assert msg.content.op_name == "download_sentinel2"
    assert msg.content.elapsed_s == 42.5
    assert msg.content.memory_usage_mb == 1024.0


def test_heartbeat_valid_on_status_channel(traceparent: str):
    from vibe_common.constants import STATUS_PUBSUB_TOPIC
    msg = WorkMessageBuilder.build_heartbeat(traceparent, "op", 1.0, None, None)
    assert msg.is_valid_for_channel(STATUS_PUBSUB_TOPIC)
```

**Step 2: Run — expect FAIL**

```bash
pytest src/vibe_common/tests/test_messaging.py -v -k heartbeat
```
Expected: `ImportError: cannot import name 'HeartbeatContent'`

**Step 3: Implement**

Edit `src/vibe_common/vibe_common/messaging.py`:

1. Add to `MessageType` enum (around line 92):
```python
    heartbeat = auto()
```

2. Add content class after `AckContent` (around line 148):
```python
class HeartbeatContent(BaseModel):
    """Periodic liveness signal from worker while an op runs.

    The orchestrator consumer is future work — for now this is fire-and-forget
    so operators can grep logs and so we don't need to break protocol later.
    """

    op_name: str
    elapsed_s: float
    memory_usage_mb: Optional[float] = None
    memory_limit_mb: Optional[float] = None
```

3. Add message class after `AckMessage` (around line 295):
```python
class HeartbeatMessage(BaseMessage):
    _supported_channels: Set[str] = {STATUS_PUBSUB_TOPIC}
    content: HeartbeatContent
```

4. Add to `MessageContent` union (around line 63):
```python
MessageContent = Union[
    "AckContent",
    "CacheInfoExecuteRequestContent",
    ...  # existing
    "HeartbeatContent",  # NEW
]
```

5. Add to `WorkMessage` union (around line 296):
```python
WorkMessage = Union[
    AckMessage,
    ...  # existing
    HeartbeatMessage,  # NEW
]
```

6. Add to `MESSAGE_TYPE_TO_CONTENT_TYPE` dict (around line 396):
```python
    MessageType.heartbeat: HeartbeatContent,
```

7. Add builder method in `WorkMessageBuilder` (around line 393):
```python
    @staticmethod
    def build_heartbeat(
        traceparent: str,
        op_name: str,
        elapsed_s: float,
        memory_usage_mb: Optional[float],
        memory_limit_mb: Optional[float],
    ) -> WorkMessage:
        run_id = run_id_from_traceparent(traceparent)
        header = MessageHeader(
            type=MessageType.heartbeat, run_id=run_id, parent_id=traceparent
        )
        content = HeartbeatContent(
            op_name=op_name,
            elapsed_s=elapsed_s,
            memory_usage_mb=memory_usage_mb,
            memory_limit_mb=memory_limit_mb,
        )
        return HeartbeatMessage(header=header, content=content)
```

**Step 4: Run — expect PASS**

```bash
pytest src/vibe_common/tests/test_messaging.py -v -k heartbeat
```

Also verify existing messaging tests still pass:

```bash
pytest src/vibe_common/tests/test_messaging.py -v
```

**Step 5: Commit**

```bash
jj commit -m "feat(common): add HeartbeatMessage type on STATUS_PUBSUB_TOPIC"
```

---

## Phase 2: Worker modifications

> **Important for all tasks in this phase:** The `Worker.__init__` creates
> `App()` (Dapr gRPC app) and `StateStore()`, both of which need a live Dapr
> sidecar. Tests must patch these. Create a `worker_fixture` in
> `src/vibe_agent/tests/conftest.py` that patches both and returns a Worker
> instance.

### Task 6: Worker test scaffolding — fixture + `WorkerMessenger.send` cap

This task bundles the fixture creation with the first testable worker change
(messenger cap) so the fixture is immediately exercised.

**Files:**
- Modify: `src/vibe_agent/tests/conftest.py` (add fixture)
- Create: `src/vibe_agent/tests/test_worker.py`
- Modify: `src/vibe_agent/vibe_agent/worker.py:162-180` (`WorkerMessenger.send`)

**Step 1: Add fixture to conftest**

Append to `src/vibe_agent/tests/conftest.py`:

```python
from unittest.mock import MagicMock, patch

@pytest.fixture
def worker(tmp_path):
    """Worker instance with Dapr/StateStore patched out."""
    from vibe_agent.worker import Worker

    with patch("vibe_agent.worker.App"), \
         patch("vibe_agent.worker.StateStore"):
        w = Worker(
            termination_grace_period_s=90,
            control_topic="commands",
            max_tries=3,
            factory_spec=MagicMock(),
            port=3000,
        )
        yield w
```

**Step 2: Write failing test**

Create `src/vibe_agent/tests/test_worker.py`:

```python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from vibe_agent.worker import WorkerMessenger


@pytest.mark.anyio
async def test_messenger_send_caps_retries():
    """Messenger must not loop forever when the pubsub is down."""
    messenger = WorkerMessenger()
    messenger.max_send_attempts = 3
    messenger.retry_interval_s = 0.01

    with patch("vibe_agent.worker.send_async", new=AsyncMock(return_value=False)):
        with pytest.raises(RuntimeError, match="3 attempts"):
            await messenger.send(MagicMock())


@pytest.mark.anyio
async def test_messenger_send_succeeds_after_transient_failure():
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
```

**Step 3: Run — expect FAIL**

```bash
pytest src/vibe_agent/tests/test_worker.py -v -k messenger
```
Expected: `AttributeError: 'WorkerMessenger' object has no attribute 'max_send_attempts'`
(or loops until pytest timeout if patch paths are wrong — adjust if so)

**Step 4: Implement**

Edit `src/vibe_agent/vibe_agent/worker.py`:

Replace `MESSAGING_RETRY_INTERVAL_S = 1` (line 54) with:
```python
MESSAGING_RETRY_INTERVAL_S = 1
MESSAGING_MAX_SEND_ATTEMPTS = 60  # ~1 minute cap; worker doesn't hang forever
```

Rewrite `WorkerMessenger.__init__` and `send` (lines 155-180):
```python
    def __init__(
        self,
        pubsubname: str = CONTROL_STATUS_PUBSUB,
        status_topic: str = STATUS_PUBSUB_TOPIC,
        max_send_attempts: int = MESSAGING_MAX_SEND_ATTEMPTS,
        retry_interval_s: float = MESSAGING_RETRY_INTERVAL_S,
    ):
        self.pubsubname = pubsubname
        self.status_topic = status_topic
        self.max_send_attempts = max_send_attempts
        self.retry_interval_s = retry_interval_s
        # Deadline is set by the Worker during shutdown so send() doesn't
        # overrun the termination grace period. None = no deadline.
        self.deadline: Optional[float] = None
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def _attempts_remaining(self, tries_so_far: int) -> bool:
        if tries_so_far >= self.max_send_attempts:
            return False
        if self.deadline is not None and time.monotonic() >= self.deadline:
            return False
        return True

    async def send(self, message: WorkMessage) -> None:
        tries: int = 0
        sent = False
        while not sent:
            if not self._attempts_remaining(tries):
                msg = (
                    f"Giving up sending {message.header.type} after {tries} attempts"
                    f"{' (shutdown deadline reached)' if self.deadline else ''}."
                )
                self.logger.error(msg)
                raise RuntimeError(msg)
            try:
                sent = await send_async(message, "worker", self.pubsubname, self.status_topic)
            except Exception:
                sent = False
            if sent:
                break
            tries += 1
            self.logger.warning(
                f"Failed to send {message.header.type} (attempt {tries}/"
                f"{self.max_send_attempts}). Retrying in {self.retry_interval_s}s."
            )
            await asyncio.sleep(self.retry_interval_s)
```

Also update `Worker.__init__` to pass config through (around line 264):
```python
self.messenger = WorkerMessenger(pubsubname, status_topic)
```
stays the same — defaults are fine. The deadline gets set in `pre_stop_hook`
(Task 7).

**Step 5: Run — expect PASS**

```bash
pytest src/vibe_agent/tests/test_worker.py -v -k messenger
```

**Step 6: Commit**

```bash
jj commit -m "fix(worker): cap WorkerMessenger.send retries instead of looping forever"
```

---

### Task 7: Drain-and-requeue shutdown

**Files:**
- Modify: `src/vibe_agent/vibe_agent/worker.py:295-311` (`pre_stop_hook`)
- Modify: `src/vibe_agent/vibe_agent/worker.py:410-449` (`get_future_result`)
- Modify: `src/vibe_agent/vibe_agent/worker.py:55` (bump `TERMINATION_GRACE_PERIOD_S`)
- Modify: `src/vibe_agent/tests/test_worker.py`

**Step 1: Write failing tests**

Append to `src/vibe_agent/tests/test_worker.py`:

```python
import signal
import time
from vibe_agent.worker import Worker, ShuttingDownException


def test_sigterm_sets_deadline_without_cancelling_child(worker: Worker):
    """SIGTERM should set shutdown_deadline, NOT immediately cancel the child."""
    worker.current_child = MagicMock()
    worker.current_message = MagicMock()

    worker.pre_stop_hook(signal.SIGTERM, None)

    assert worker.shutting_down
    assert worker.shutdown_deadline is not None
    assert worker.shutdown_deadline > time.monotonic()
    # Critical: child must NOT be cancelled yet — we're draining
    worker.current_child.cancel.assert_not_called()


def test_drain_lets_op_finish_before_deadline(worker: Worker):
    """If the subprocess finishes before the deadline, return its result normally."""
    worker.current_message = MagicMock()
    worker.statestore = MagicMock()
    # Simulate: shutdown started, 30s left on the clock
    worker.shutting_down = True
    worker.shutdown_deadline = time.monotonic() + 30

    # Mock a child that finishes quickly
    mock_child = MagicMock()
    mock_child.result.return_value = {"output": "success"}

    result = worker.get_future_result(mock_child, monitoring_period_s=1, timeout_s=60)
    assert result == {"output": "success"}
    mock_child.cancel.assert_not_called()


def test_drain_cancels_after_deadline(worker: Worker):
    """If the deadline passes, cancel the child and raise ShuttingDownException."""
    import concurrent.futures

    worker.current_message = MagicMock()
    worker.statestore = MagicMock()
    # is_workflow_complete should return False so we don't take that branch
    worker.is_workflow_complete = MagicMock(return_value=False)

    worker.shutting_down = True
    # Deadline already in the past — next poll should cancel
    worker.shutdown_deadline = time.monotonic() - 1

    mock_child = MagicMock()
    # child.result() keeps timing out — simulates a long-running op
    mock_child.result.side_effect = concurrent.futures.TimeoutError()

    with pytest.raises(ShuttingDownException):
        worker.get_future_result(mock_child, monitoring_period_s=0.1, timeout_s=60)

    mock_child.cancel.assert_called_once()


def test_messenger_deadline_set_on_shutdown(worker: Worker):
    """Messenger should learn the shutdown deadline so it can cap send retries."""
    assert worker.messenger.deadline is None
    worker.pre_stop_hook(signal.SIGTERM, None)
    assert worker.messenger.deadline is not None
    assert worker.messenger.deadline == worker.shutdown_deadline
```

**Step 2: Run — expect FAIL**

```bash
pytest src/vibe_agent/tests/test_worker.py -v -k "sigterm or drain"
```
Expected: `AttributeError: 'Worker' object has no attribute 'shutdown_deadline'`

**Step 3: Implement**

Edit `src/vibe_agent/vibe_agent/worker.py`:

**3a. Bump constant** (line 55):
```python
TERMINATION_GRACE_PERIOD_S = 90
```
Comment above it:
```python
# Time we give an in-progress op to finish naturally after SIGTERM before
# force-cancelling. Must be less than the pod's terminationGracePeriodSeconds
# (set in worker.tf) so we still have time to send a failure status and let
# Dapr flush before Kubernetes SIGKILLs us.
```

**3b. Add `shutdown_deadline` field** (around line 227, with other class attrs):
```python
    shutdown_deadline: Optional[float] = None
```

**3c. Rewrite `pre_stop_hook`** (lines 295-311):
```python
    def pre_stop_hook(self, signum: int, _: Any):
        """Initiate drain: stop accepting work, let the current op finish.

        We do NOT cancel the child here. get_future_result() polls
        shutdown_deadline and cancels only when the grace period expires.
        If the op finishes before then, the result is delivered normally.
        """
        with self.shutdown_lock:
            if self.shutting_down:
                self.logger.warning(
                    f"Shutdown already in progress. Ignoring signal {signum}."
                )
                return
            self.shutting_down = True
            self.shutdown_deadline = time.monotonic() + self.termination_grace_period_s
            # Propagate to messenger so status sends don't overrun the deadline.
            self.messenger.deadline = self.shutdown_deadline
            self.logger.info(
                f"SIGTERM received. Draining: current op has "
                f"{self.termination_grace_period_s}s to finish naturally "
                f"before being cancelled."
            )
```

**3d. Update `get_future_result` shutdown check** (lines 433-436):

Replace:
```python
                if self.shutting_down:
                    self.logger.info("Shutdown process initiated. Terminating child process.")
                    child.cancel()
                    raise ShuttingDownException()
```
with:
```python
                if self.shutting_down:
                    if self.shutdown_deadline is None or time.monotonic() >= self.shutdown_deadline:
                        self.logger.warning(
                            "Shutdown grace period expired. Cancelling child process "
                            "and returning message to queue for redelivery."
                        )
                        child.cancel()
                        raise ShuttingDownException()
                    remaining = self.shutdown_deadline - time.monotonic()
                    self.logger.info(
                        f"Draining: {remaining:.0f}s remaining for op to finish naturally."
                    )
                    # Fall through — keep waiting
```

**3e. Move server stop** — Remove `self.app._server.stop(None)` from
`pre_stop_hook`. The server must stay up so `fetch_work` can return its
`TopicEventResponse`. Add a `_finalize_shutdown` method and call it from
`start_service` after `app.run` returns:

Around line 326-334 (`start_service`), change to:
```python
    @dapr_ready
    def start_service(self):
        self.logger.info(f"Starting worker listening on port {self.port}")
        while not self.shutting_down:
            self.app.run(self.port)
            time.sleep(1)
        # Drain complete (or we were never busy) — tear down gRPC server.
        self._finalize_shutdown()

    def _finalize_shutdown(self):
        self.logger.info("Drain complete. Stopping gRPC server.")
        if self.app._server is not None:
            self.app._server.stop(None)
```

**However**, note: `app.run()` is blocking and won't return while the server
is alive. The existing loop exists because *something* was spuriously
stopping it. With drain, we need `app.run()` to return when
`shutting_down=True`. Check: does Dapr's `App.run()` respect `stop()`? If
yes, we need a thread that calls `stop()` after the last `fetch_work`
completes. Simpler approach: set a threading.Event when `fetch_work` finishes
while shutting down, and have a monitor thread call `app._server.stop(None)`
then.

**Pragmatic alternative** (recommended): keep server stop in `pre_stop_hook`
but delayed. Spawn a daemon thread that waits for `work_lock` to be free
(meaning `fetch_work` returned), then stops the server:

```python
    def pre_stop_hook(self, signum: int, _: Any):
        with self.shutdown_lock:
            if self.shutting_down:
                self.logger.warning(f"Shutdown already in progress. Ignoring signal {signum}.")
                return
            self.shutting_down = True
            self.shutdown_deadline = time.monotonic() + self.termination_grace_period_s
            self.messenger.deadline = self.shutdown_deadline
            self.logger.info(
                f"SIGTERM received. Draining: current op has "
                f"{self.termination_grace_period_s}s to finish naturally."
            )
            # Stop the server only after the current fetch_work returns.
            threading.Thread(target=self._wait_and_stop_server, daemon=True).start()

    def _wait_and_stop_server(self):
        """Wait for the work_lock to be free (op done), then stop the gRPC server."""
        self.work_lock.acquire()
        self.work_lock.release()
        self.logger.info("Drain complete. Stopping gRPC server.")
        if self.app._server is not None:
            self.app._server.stop(None)
```

Use this version — simpler and doesn't change the `start_service` loop.

**Step 4: Run — expect PASS**

```bash
pytest src/vibe_agent/tests/test_worker.py -v -k "sigterm or drain"
```

Also run all worker tests to catch regressions:
```bash
pytest src/vibe_agent/tests/test_worker.py -v
```

**Step 5: Commit**

```bash
jj commit -m "feat(worker): drain-and-requeue shutdown — let op finish before grace deadline, then cancel"
```

---

### Task 8: OOM detection via `ProcessExpired.exitcode`

**Files:**
- Modify: `src/vibe_agent/vibe_agent/worker.py:495-496` (`run_op_with_retry`)
- Modify: `src/vibe_agent/tests/test_worker.py`

**Step 1: Write failing tests**

Append to `src/vibe_agent/tests/test_worker.py`:

```python
from pebble.common import ProcessExpired
from vibe_agent.worker import Worker


def _mock_content(op_name="test_op"):
    """Fake CacheInfoExecuteRequestContent for run_op_with_retry."""
    content = MagicMock()
    content.operation_spec = MagicMock()
    content.operation_spec.name = op_name
    content.input = {}
    return content


def test_oom_exitcode_produces_clear_error_and_no_retry(worker: Worker):
    """SIGKILL exit should produce a 'likely out of memory' error and stop retrying."""
    oom_exc = ProcessExpired("Abnormal termination")
    oom_exc.exitcode = -9
    worker.try_run_op = MagicMock(side_effect=oom_exc)

    with patch("vibe_agent.worker.get_memory_info") as mock_mem:
        mock_mem.return_value = MagicMock(usage_mb=4096.0, limit_mb=4096.0, __str__=lambda s: "4096MB / 4096MB (100%)")
        with pytest.raises(RuntimeError, match="out of memory"):
            worker.run_op_with_retry(_mock_content(), run_id=MagicMock(), timeout_s=30)

    # Should NOT retry — OOM will just recur
    assert worker.try_run_op.call_count == 1


def test_non_oom_process_expired_still_retries(worker: Worker):
    """Non-SIGKILL ProcessExpired (e.g. exitcode=1) should retry normally."""
    generic_exc = ProcessExpired("died")
    generic_exc.exitcode = 1
    worker.try_run_op = MagicMock(side_effect=generic_exc)
    worker.max_tries = 3

    with pytest.raises(RuntimeError):  # final failure after all retries
        worker.run_op_with_retry(_mock_content(), run_id=MagicMock(), timeout_s=30)

    assert worker.try_run_op.call_count == 3
```

**Step 2: Run — expect FAIL**

```bash
pytest src/vibe_agent/tests/test_worker.py -v -k oom
```
Expected: `AssertionError: assert 3 == 1` (currently all ProcessExpired retries)
and `DID NOT RAISE RuntimeError match='out of memory'`

**Step 3: Implement**

Edit `src/vibe_agent/vibe_agent/worker.py`:

Add imports at top:
```python
from vibe_common.resources import get_memory_info, is_oom_exitcode
```

Replace `ProcessExpired` handler (lines 495-496) inside `run_op_with_retry`:
```python
            except ProcessExpired as e:
                exitcode = getattr(e, "exitcode", None)
                if exitcode is not None and is_oom_exitcode(exitcode):
                    # SIGKILL in a container almost always = OOM killer.
                    # Don't retry — the same input will OOM again. Build a
                    # user-facing error that explains what happened instead
                    # of the opaque "ProcessExpired: Abnormal termination"
                    # that users currently see (see TROUBLESHOOTING.md).
                    mem = get_memory_info()
                    msg = (
                        f"Op {spec.name} subprocess was killed by SIGKILL "
                        f"(exitcode={exitcode}). This is likely out of memory. "
                        f"Container memory: {mem}. "
                        f"Consider reducing input size or increasing worker memory."
                    )
                    self.logger.error(msg)
                    raise RuntimeError(msg) from e
                self.logger.exception(
                    f"Child process died unexpectedly on try {i+1}/{self.max_tries} "
                    f"(exitcode={exitcode}). Retrying."
                )
```

**Step 4: Run — expect PASS**

```bash
pytest src/vibe_agent/tests/test_worker.py -v -k oom
```

**Step 5: Commit**

```bash
jj commit -m "feat(worker): detect OOM via ProcessExpired.exitcode and produce clear error"
```

---

### Task 9: Retry with backoff in `run_op_with_retry`

**Files:**
- Modify: `src/vibe_agent/vibe_agent/worker.py:466-508` (`run_op_with_retry`)
- Modify: `src/vibe_agent/vibe_agent/worker.py:511-527` (`WorkerConfig`)
- Modify: `src/vibe_agent/tests/test_worker.py`

**Step 1: Write failing tests**

Append to `src/vibe_agent/tests/test_worker.py`:

```python
import traceback as tb


def test_retry_sleeps_with_backoff_between_attempts(worker: Worker):
    """Failed attempts should sleep with increasing delays."""
    worker.max_tries = 3
    worker.op_retry_base_delay_s = 0.01
    worker.op_retry_max_delay_s = 1.0

    # Return a TracebackException (simulates op raising inside subprocess)
    fake_tb = tb.TracebackException(ValueError, ValueError("transient"), None)
    worker.try_run_op = MagicMock(return_value=fake_tb)

    with patch("vibe_agent.worker.time.sleep") as mock_sleep:
        with pytest.raises(RuntimeError):
            worker.run_op_with_retry(_mock_content(), run_id=MagicMock(), timeout_s=30)

    # max_tries=3 → 2 gaps between attempts → 2 sleep calls
    # (compute_backoff might be called via a sleep loop; count >= 2)
    sleep_delays = [c.args[0] for c in mock_sleep.call_args_list if c.args]
    # With base=0.01, exp=2: delays would be ~0.01, ~0.02 (jitter may reduce)
    assert len(sleep_delays) >= 2  # at least two backoff sleeps happened


def test_retry_backoff_abandoned_on_shutdown(worker: Worker):
    """Backoff sleep must check shutting_down so we don't block drain."""
    worker.max_tries = 3
    worker.op_retry_base_delay_s = 10.0  # long enough to matter
    worker.op_retry_max_delay_s = 10.0
    worker.shutting_down = True  # already shutting down
    worker.shutdown_deadline = time.monotonic() + 100  # but deadline not expired

    fake_tb = tb.TracebackException(ValueError, ValueError("x"), None)
    worker.try_run_op = MagicMock(return_value=fake_tb)

    start = time.monotonic()
    with pytest.raises(ShuttingDownException):
        worker.run_op_with_retry(_mock_content(), run_id=MagicMock(), timeout_s=30)
    elapsed = time.monotonic() - start

    # Should bail fast (within ~1s), not sleep 10s
    assert elapsed < 3.0
```

**Step 2: Run — expect FAIL**

```bash
pytest src/vibe_agent/tests/test_worker.py -v -k "retry_sleeps or retry_backoff"
```
Expected: no sleeps happen currently, first assertion fails

**Step 3: Implement**

Edit `src/vibe_agent/vibe_agent/worker.py`:

Add import:
```python
from vibe_common.retry import RetryPolicy, compute_backoff
```

Add constants after `TERMINATION_GRACE_PERIOD_S`:
```python
OP_RETRY_BASE_DELAY_S = 2.0
OP_RETRY_MAX_DELAY_S = 60.0
```

Add fields to `Worker.__init__` (accept as kwargs, around line 249):
```python
        op_retry_base_delay_s: float = OP_RETRY_BASE_DELAY_S,
        op_retry_max_delay_s: float = OP_RETRY_MAX_DELAY_S,
```
Store them:
```python
        self.op_retry_base_delay_s = op_retry_base_delay_s
        self.op_retry_max_delay_s = op_retry_max_delay_s
```

Add helper method on `Worker`:
```python
    def _interruptible_sleep(self, total_s: float):
        """Sleep in ~1s slices so shutdown can interrupt a long backoff."""
        end = time.monotonic() + total_s
        while time.monotonic() < end:
            if self.shutting_down:
                raise ShuttingDownException()
            time.sleep(min(1.0, end - time.monotonic()))
```

Modify `run_op_with_retry` (around line 479) — after a failed attempt
but before the next loop iteration, insert backoff. The structure becomes:

```python
        retry_policy = RetryPolicy(
            max_attempts=self.max_tries,
            base_delay_s=self.op_retry_base_delay_s,
            max_delay_s=self.op_retry_max_delay_s,
            jitter=True,
        )
        for i in range(self.max_tries):
            inner_timeout = final_time - time.time()
            if self.shutting_down:
                self.logger.info("Stopping op retry loop — shutdown in progress.")
                raise ShuttingDownException()
            try:
                ret = self.try_run_op(spec, content, inner_timeout)
                if not isinstance(ret, traceback.TracebackException):
                    self.logger.debug(f"Op {spec} succeeded on try {i+1} (run {run_id})")
                    break
                self.logger.error(
                    f"Op {spec.name} failed in subprocess "
                    f"(try {i+1}/{self.max_tries}): {''.join(ret.format())}"
                )
            except ProcessExpired as e:
                # ... OOM handling from Task 8 ...
                # (non-OOM falls through to backoff below)
            except TimeoutError as e:
                # ... existing timeout handling — re-raise, don't retry ...
                raise RuntimeError(msg) from e

            # Attempt failed (TracebackException or non-OOM ProcessExpired).
            # Backoff before the next try — but only if there IS a next try.
            if i + 1 < self.max_tries:
                delay = compute_backoff(i, retry_policy)
                self.logger.warning(
                    f"Op {spec.name}: backing off {delay:.1f}s before "
                    f"try {i+2}/{self.max_tries}."
                )
                self._interruptible_sleep(delay)
```

Add new fields to `WorkerConfig` builds() call (around line 511):
```python
WorkerConfig = builds(
    Worker,
    ...  # existing
    op_retry_base_delay_s=OP_RETRY_BASE_DELAY_S,
    op_retry_max_delay_s=OP_RETRY_MAX_DELAY_S,
    ...
)
```

**Step 4: Run — expect PASS**

```bash
pytest src/vibe_agent/tests/test_worker.py -v
```

**Step 5: Commit**

```bash
jj commit -m "feat(worker): exponential backoff between op retries, interruptible on shutdown"
```

---

### Task 10: Memory pressure warning + heartbeat

These are both added to the same poll loop in `get_future_result` — combine
them.

**Files:**
- Modify: `src/vibe_agent/vibe_agent/worker.py:410-449` (`get_future_result`)
- Modify: `src/vibe_agent/tests/test_worker.py`

**Step 1: Write failing tests**

Append to `src/vibe_agent/tests/test_worker.py`:

```python
import concurrent.futures


def test_memory_pressure_warning_logged_once(worker: Worker, caplog):
    """When usage > 85%, log WARNING once — don't spam every poll."""
    import logging

    worker.current_message = MagicMock()
    worker.is_workflow_complete = MagicMock(return_value=False)

    mock_child = MagicMock()
    # Timeout twice (two poll iterations), then return
    mock_child.result.side_effect = [
        concurrent.futures.TimeoutError(),
        concurrent.futures.TimeoutError(),
        {"done": True},
    ]

    high_mem = MagicMock(usage_fraction=0.92, usage_mb=3700.0, limit_mb=4000.0)
    with patch("vibe_agent.worker.get_memory_info", return_value=high_mem), \
         caplog.at_level(logging.WARNING):
        worker.get_future_result(mock_child, monitoring_period_s=0.01, timeout_s=30)

    warnings = [r for r in caplog.records if "memory pressure" in r.message.lower()]
    assert len(warnings) == 1  # one-shot, not three


def test_heartbeat_sent_during_long_op(worker: Worker):
    """Heartbeat published periodically while subprocess runs."""
    worker.current_message = MagicMock()
    worker.current_message.id = "00-abc-def-01"  # fake traceparent
    worker.is_workflow_complete = MagicMock(return_value=False)
    worker.heartbeat_interval_s = 0.01  # fast for test

    mock_child = MagicMock()
    mock_child.result.side_effect = [
        concurrent.futures.TimeoutError(),
        concurrent.futures.TimeoutError(),
        concurrent.futures.TimeoutError(),
        {"done": True},
    ]

    sent_messages = []
    async def capture_send(msg):
        sent_messages.append(msg)

    worker.messenger.send = capture_send

    with patch("vibe_agent.worker.get_memory_info") as mock_mem:
        mock_mem.return_value = MagicMock(usage_fraction=0.1, usage_mb=100.0, limit_mb=1000.0)
        # Also need time.monotonic to advance past heartbeat_interval_s between polls
        worker.get_future_result(mock_child, monitoring_period_s=0.02, timeout_s=30)

    from vibe_common.messaging import HeartbeatMessage
    heartbeats = [m for m in sent_messages if isinstance(m, HeartbeatMessage)]
    assert len(heartbeats) >= 1
```

**Step 2: Run — expect FAIL**

```bash
pytest src/vibe_agent/tests/test_worker.py -v -k "memory_pressure or heartbeat"
```

**Step 3: Implement**

Edit `src/vibe_agent/vibe_agent/worker.py`:

Add constants after `OP_RETRY_MAX_DELAY_S`:
```python
MEMORY_WARNING_THRESHOLD = 0.85
HEARTBEAT_INTERVAL_S = 30
```

Add `__init__` kwarg + store:
```python
        heartbeat_interval_s: int = HEARTBEAT_INTERVAL_S,
        ...
        self.heartbeat_interval_s = heartbeat_interval_s
```

Add import:
```python
from vibe_common.messaging import (
    ...,
    WorkMessageBuilder,  # if not already there — it is, via existing imports? check
)
```
(`WorkMessageBuilder` is already imported at line 34.)

Rewrite `get_future_result` to embed the two checks. Key: keep the structure
linear — we're in the `except TimeoutError` branch each poll iteration.

```python
    def get_future_result(
        self, child: ProcessFuture, monitoring_period_s: int, timeout_s: float
    ) -> Any:
        start_time = time.time()
        start_mono = time.monotonic()
        last_heartbeat_mono = start_mono
        mem_warning_fired = False

        while time.time() - start_time < timeout_s:
            try:
                return child.result(monitoring_period_s)
            except concurrent.futures.TimeoutError:
                assert self.current_message is not None, (
                    "Correctness issue: current_message should not be None here."
                )
                if self.is_workflow_complete(self.current_message):
                    self.logger.info(
                        f"Workflow {self.current_message.run_id} is complete. "
                        "Terminating child process."
                    )
                    child.cancel()
                    raise RuntimeError(
                        "Workflow was completed/failed/cancelled while running op."
                    )
                if self.shutting_down:
                    if self.shutdown_deadline is None or time.monotonic() >= self.shutdown_deadline:
                        self.logger.warning(
                            "Shutdown grace period expired. Cancelling child process "
                            "and returning message to queue for redelivery."
                        )
                        child.cancel()
                        raise ShuttingDownException()
                    remaining = self.shutdown_deadline - time.monotonic()
                    self.logger.info(f"Draining: {remaining:.0f}s remaining for op to finish.")

                # Memory pressure check. One-shot per threshold crossing —
                # don't spam every 10s while a big op runs.
                mem = get_memory_info()
                if mem.usage_fraction is not None:
                    if mem.usage_fraction > MEMORY_WARNING_THRESHOLD:
                        if not mem_warning_fired:
                            self.logger.warning(
                                f"Memory pressure: {mem}. Op may be at risk of OOM."
                            )
                            mem_warning_fired = True
                    elif mem.usage_fraction < MEMORY_WARNING_THRESHOLD - 0.05:
                        # Reset if usage drops back down (hysteresis)
                        mem_warning_fired = False

                # Heartbeat — best effort, never blocks the loop. The
                # orchestrator consumer is future work; for now this shows
                # up in logs/traces so operators can see liveness.
                if time.monotonic() - last_heartbeat_mono >= self.heartbeat_interval_s:
                    last_heartbeat_mono = time.monotonic()
                    self._send_heartbeat(start_mono, mem)

                continue
            except concurrent.futures.CancelledError:
                if self.shutting_down:
                    raise ShuttingDownException()
                self.logger.warning(
                    f"Child cancelled while running {self.current_message} "
                    "but we're not shutting down. Unexpected."
                )
                raise
            except Exception as e:
                self.logger.exception(f"Child process failed: {e}")
                return traceback.TracebackException.from_exception(e)
        raise TimeoutError(f"Op execution exceeded {timeout_s} seconds.")

    def _send_heartbeat(self, start_mono: float, mem: "MemoryInfo"):
        """Fire-and-forget heartbeat to the status topic. Failure is logged at DEBUG."""
        if self.current_message is None:
            return
        try:
            content = cast(CacheInfoExecuteRequestContent, self.current_message.content)
            op_name = str(content.operation_spec.name)
        except Exception:
            op_name = "unknown"
        elapsed = time.monotonic() - start_mono
        hb = WorkMessageBuilder.build_heartbeat(
            self.current_message.id,
            op_name=op_name,
            elapsed_s=elapsed,
            memory_usage_mb=mem.usage_mb,
            memory_limit_mb=mem.limit_mb,
        )
        try:
            # Single attempt, no retry — heartbeats are cheap & frequent.
            asyncio.run(send_async(hb, "worker", self.pubsubname, self.status_topic))
        except Exception as e:
            self.logger.debug(f"Heartbeat send failed (ignored): {e}")
```

Add to `WorkerConfig` builds():
```python
    heartbeat_interval_s=HEARTBEAT_INTERVAL_S,
```

**Step 4: Run — expect PASS**

```bash
pytest src/vibe_agent/tests/test_worker.py -v
```

**Step 5: Commit**

```bash
jj commit -m "feat(worker): memory pressure warning + periodic heartbeat during op execution"
```

---

## Phase 3: Terraform

### Task 11: K8s termination grace period

**Files:**
- Modify: `src/vibe_core/vibe_core/terraform/services/worker.tf:72`
- Modify: `src/vibe_core/vibe_core/terraform/services/variables.tf`

**Step 1: Add variable**

Append to `src/vibe_core/vibe_core/terraform/services/variables.tf`:

```hcl
variable "worker_termination_grace_period_s" {
  description = "Pod terminationGracePeriodSeconds. Must exceed worker's internal termination_grace_period_s (default 90) by at least 30s for status flush + Dapr sidecar shutdown."
  default     = 120
}
```

**Step 2: Add to pod spec in worker.tf**

In `src/vibe_core/vibe_core/terraform/services/worker.tf`, inside the `spec`
block at line 72 (right after the opening `spec {`):

```hcl
      spec {
        termination_grace_period_seconds = var.worker_termination_grace_period_s
        node_selector = {
        ...
```

**Step 3: Pass internal grace period via args**

In `local.worker_extra_args` (around line 18), add:

```hcl
  worker_extra_args = concat(
    [
      "worker.impl.logdir=${var.log_dir}",
      "worker.impl.loglevel=${var.farmvibes_log_level}",
      # Keep internal grace < pod grace by 30s buffer for status flush + Dapr drain
      "worker.impl.termination_grace_period_s=${var.worker_termination_grace_period_s - 30}",
    ],
    ...
```

**Step 4: Validate syntax** (no unit test for terraform here)

```bash
cd src/vibe_core/vibe_core/terraform/services && terraform fmt -check && terraform validate 2>&1 | head -20
```
If `terraform` not installed locally, skip validation and rely on VM test.

**Step 5: Commit**

```bash
jj commit -m "feat(terraform): set worker terminationGracePeriodSeconds for drain shutdown"
```

---

## Phase 4: Integration check & docs

### Task 12: Update TROUBLESHOOTING.md

**Files:**
- Modify: `docs/source/docfiles/markdown/TROUBLESHOOTING.md:153-173`

**Step 1: Edit**

The "Abnormal Termination" section currently tells users to guess at OOM.
Update it to reflect the new error message:

Replace the section body (keep the `<details><summary>` wrapper) with:
```markdown
Some workflows (SpaceEye, SAM) can use large amounts of memory. When the
kernel OOM-killer terminates the offending subprocess, the run error will
now say:

    Op <name> subprocess was killed by SIGKILL (exitcode=-9). This is
    likely out of memory. Container memory: <X>MB / <Y>MB (<Z>%).

If you see this, reduce the input region/time range, scale down worker
replicas (`kubectl scale deployment terravibes-worker --replicas=1`), or
increase `worker_memory_request` in your terraform variables.

Older versions showed `ProcessExpired: Abnormal termination` — same cause.
```

**Step 2: Commit**

```bash
jj commit -m "docs: update TROUBLESHOOTING for new OOM error format"
```

---

### Task 13: Full test suite pass

**Step 1: Run all modified test files**

```bash
pytest src/vibe_common/tests/test_retry.py \
       src/vibe_common/tests/test_resources.py \
       src/vibe_common/tests/test_messaging.py \
       src/vibe_agent/tests/test_worker.py -v
```

Expected: all PASS

**Step 2: Run broader suite to catch regressions**

```bash
pytest src/vibe_common/tests/ src/vibe_agent/tests/ -v
```

Expected: all PASS (or same baseline failures as `main`)

**Step 3: Commit if any fixups needed**

```bash
jj commit -m "test: fixups from full suite run"
```

---

### Task 14: Squash/organize commits & prepare push

**Step 1: Review history**

```bash
jj log -r 'main..@'
```

**Step 2: Move bookmark to tip**

```bash
jj bookmark set worker-harden_task11_model_a -r @-
```
(or `@` if working copy is non-empty)

**Step 3: Push** (user action — do not execute)

Instruct user:
```
jj git push --bookmark worker-harden_task11_model_a
```

---

## Manual validation on VM (post-push)

Not scripted — do after code lands. Checklist:

1. `cd ~/farmvibes-ai && git fetch && git checkout worker-harden_task11_model_a`
2. Rebuild worker image: `make local-worker` (or whatever the make target is — check Makefile)
3. Restart cluster, submit a short workflow → baseline success
4. `kubectl scale deployment terravibes-worker --replicas=0` mid-run → watch logs for "Draining: Ns remaining" → scale back up → workflow completes or message redelivered
5. Set tiny memory limit on worker deployment, submit SpaceEye → error contains "likely out of memory"
6. Grep worker logs for `HeartbeatMessage` / `heartbeat` entries
7. Network-policy test: block egress from worker pod, verify retry logging, restore, verify success
