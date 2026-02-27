# Worker Service Hardening — Design

**Date:** 2026-02-26
**Task:** 11 / model_a
**Branch:** `worker-harden_task11_model_a`

## Problem

The worker (`src/vibe_agent/vibe_agent/worker.py`) runs workflow steps in
pebble subprocesses and reports results to the orchestrator via Dapr pubsub.
Failure handling today is minimal:

| Symptom | Cause | Location |
|---|---|---|
| Failing download hammered 5× with no delay | `run_op_with_retry` has no backoff | worker.py:479-504 |
| OOM shows as "ProcessExpired: Abnormal termination" | `ProcessExpired.exitcode` never inspected | worker.py:495 |
| SIGTERM kills in-progress work immediately | `pre_stop_hook` cancels child synchronously | worker.py:295-311 |
| `termination_grace_period_s` is dead config | Stored but never read | worker.py:252 |
| Worker can hang forever sending status | `WorkerMessenger.send` retries with no cap | worker.py:162-180 |
| Orchestrator waits forever for dead worker | No liveness signal from worker | orchestrator.py (no heartbeat consumer) |

Dapr already handles message-level redelivery (`dapr.tf:97-123` — exponential
retry, infinite retries, `requeueInFailure: true`). The gaps above are all at
the Python application layer.

## Decisions

| Concern | Decision |
|---|---|
| Retry utility location | Hand-rolled in `vibe_common/retry.py` (no new deps) |
| Resource monitor location | `vibe_common/resources.py` (cgroup direct reads, no psutil) |
| Heartbeat channel | Reuse `STATUS_PUBSUB_TOPIC` (orchestrator already subscribes) |
| Shutdown strategy | **Drain-and-requeue**: on SIGTERM let subprocess finish, cancel only if grace deadline expires |
| Stale detection | Worker-side heartbeat only — orchestrator consumer is follow-up work |

## Components

### 1. `vibe_common/retry.py` (new)

```python
@dataclass
class RetryPolicy:
    max_attempts: int = 3
    base_delay_s: float = 1.0
    max_delay_s: float = 30.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable: Callable[[BaseException], bool] = lambda e: True
```

API: `compute_backoff(attempt, policy) -> float`, `@with_retry(policy, logger)`
decorator (sync+async), `RetryLoop` iterator for imperative use. Each retry
logs WARNING with attempt number, exception class, next delay. Exhaustion
re-raises last exception.

### 2. `vibe_common/resources.py` (new)

```python
@dataclass
class MemoryInfo:
    limit_bytes: int | None   # None = unbounded
    usage_bytes: int
    # computed: usage_fraction, usage_mb, limit_mb
```

`get_memory_info()` — reads cgroup v2 (`/sys/fs/cgroup/memory.max`,
`memory.current`), falls back to cgroup v1 (`memory/memory.limit_in_bytes`,
`memory.usage_in_bytes`), falls back to `/proc/self/status` VmRSS.

`is_oom_exitcode(exitcode)` — true for `-9` or `137` (128+SIGKILL).

### 3. `vibe_common/messaging.py` (extended)

New: `MessageType.heartbeat`, `HeartbeatContent(op_name: str, elapsed_s: float,
memory_usage_mb: float | None, memory_limit_mb: float | None)`,
`HeartbeatMessage` (channel: `STATUS_PUBSUB_TOPIC`),
`WorkMessageBuilder.build_heartbeat(traceparent, content)`.

### 4. `vibe_agent/worker.py` (modified)

#### Retry backoff — `run_op_with_retry`

- Between failed attempts, sleep `compute_backoff(i, policy)`
- Poll `shutting_down` during sleep (1s granularity) to stay responsive
- Skip backoff on `ProcessExpired` (subprocess crash ≠ transient)
- New hydra-configurable fields: `op_retry_base_delay_s=2.0`,
  `op_retry_max_delay_s=60.0`

#### OOM detection — `ProcessExpired` handler

Inspect `e.exitcode`. If `is_oom_exitcode()`: build error message with op name
+ current `MemoryInfo`, log ERROR, raise `RuntimeError(msg)` (no retry — OOM
will recur). Otherwise keep current retry behavior. User-facing error becomes
"likely out of memory, usage X/Y MB" instead of "Abnormal termination."

#### Memory pressure warning — `get_future_result` poll loop

Each iteration (~10s): check `get_memory_info()`. If `usage_fraction > 0.85`
and not already warned, log WARNING once. Reset flag if usage drops back below
0.80.

#### Drain shutdown — `pre_stop_hook` + `get_future_result`

`pre_stop_hook` no longer cancels child or stops server. It sets
`shutting_down = True` and `shutdown_deadline = now + termination_grace_period_s`
and returns. New work is already rejected (existing check at worker.py:383).

`get_future_result` changes `if shutting_down` from immediate cancel to
deadline check: continue waiting until `monotonic() >= shutdown_deadline`, then
cancel and raise `ShuttingDownException` → Dapr gets `retry` → message
redelivered to another worker.

If op finishes before deadline, normal success path executes.

Server stop moves to `start_service` loop exit (after `fetch_work` returns and
`work_lock` releases).

#### `WorkerMessenger.send` cap

Replace `while True` with bounded retry. During shutdown, cap = time remaining
to `shutdown_deadline`. Otherwise cap ≈ 60 attempts. On exhaustion: ERROR log +
raise. Messenger needs a reference to worker's shutdown state — either pass a
callback or make Messenger aware of a deadline.

#### Heartbeat — `get_future_result` poll loop

Every `heartbeat_interval_s` (default 30s, configurable) while subprocess runs:
build and send `HeartbeatMessage`. Best-effort — single send, no retry, log
DEBUG on failure. Never blocks the monitoring loop.

### 5. Terraform

`services/worker.tf`: add `termination_grace_period_seconds =
var.worker_termination_grace_period_s` to pod spec. Pass internal grace
(`worker.impl.termination_grace_period_s`) via args, = K8s value − 30s buffer.

`services/variables.tf`: `worker_termination_grace_period_s` (default 120).

Bump `TERMINATION_GRACE_PERIOD_S` in worker.py from 5 → 90.

### 6. Tests

New `src/vibe_agent/tests/test_worker.py`:
- Retry backoff timing
- OOM exitcode → clear error, no retry
- Non-OOM ProcessExpired → still retries
- SIGTERM sets deadline, does not cancel immediately
- Drain success path (op finishes before deadline)
- Drain timeout path (deadline expires → cancel + ShuttingDownException)
- Messenger retry cap enforced
- Heartbeat published during long op

New `src/vibe_common/tests/test_retry.py`, `test_resources.py` for utility
modules.

## Validation (manual, on VM)

1. Submit short workflow → success (baseline)
2. `kubectl scale deployment terravibes-worker --replicas=0` mid-run →
   logs show "Draining, Ns remaining" → workflow completes or message
   redelivered after scale-up
3. Force OOM (large SpaceEye region or tiny memory limit) → run status error
   contains "out of memory" with MB figures
4. Grep worker logs for heartbeat messages with elapsed/memory fields
5. Simulate Dapr sidecar down during shutdown → worker exits within grace
   period (doesn't hang on status send)

## Out of scope

- Orchestrator-side heartbeat consumer / stale detection (follow-up)
- Per-op retry classification (transient vs permanent) — all exceptions treated
  equally for now except OOM
- Checkpoint/resume for long ops
