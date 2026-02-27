# Worker Service Hardening Design

## Context

The FarmVibes.AI worker service executes workflow steps in isolated subprocesses via pebble, consuming tasks from RabbitMQ through Dapr's pub/sub sidecar. Each worker pod processes one task at a time (lock + prefetch=1). The orchestrator manages workflow state in Dapr statestore (Redis/CosmosDB).

Several failure modes exist in the current implementation that cause workflows to hang, produce unhelpful error messages, or fail silently.

## Confirmed Failure Modes

1. **No backoff between op retries** (`worker.py:479-508`): All 5 retry attempts fire back-to-back with zero delay. Transient failures (e.g., Planetary Computer downtime) cause rapid subprocess churn.
2. **OOM kills retried identically** (`worker.py:487-508`): `ProcessExpired` (SIGKILL/OOM) is caught and retried 5 times with no mitigation or specific error message.
3. **Termination grace period stored but never enforced** (`worker.py:295-310, 517`): `termination_grace_period_s=5` is set in config but never referenced. Shutdown immediately cancels the subprocess.
4. **Infinite messaging retry blocks shutdown** (`worker.py:163-180`): `WorkerMessenger.send()` loops forever if Dapr sidecar is unreachable.
5. **No proactive resource monitoring** (`worker.py:77-84`): `parse_resources_usage()` only fires reactively on SIGTERM/SIGCHLD. No periodic reporting.
6. **No stale workflow detection**: `RemoteWorkflowRunner._wait_for_reply()` blocks on `asyncio.Queue.get()` indefinitely. A stuck worker means a stuck workflow with no timeout.

## Decisions

| Area | Decision |
|---|---|
| Module scope | Generic utilities in `vibe_common`, pragmatic about worker concerns |
| Retry | Exponential backoff with jitter between op retries, configurable, standalone decorator |
| Shutdown | Drain + requeue: stop accepting work, let subprocess finish within grace period, requeue on timeout |
| Stale detection | Worker heartbeats on updates topic + orchestrator-side timeout monitoring |
| Resource monitoring | Pure telemetry: periodic cgroup usage logging, OOM error classification with factual messages |
| Error messages | Factual: op name, signal, memory consumed vs limit. No actionable guidance. |

## Design

### 1. Retry with Backoff

**New file:** `vibe_common/vibe_common/retry.py`

API:
- `@retry_with_backoff(max_retries=5, base_delay=1.0, max_delay=60.0, backoff_factor=2.0, retryable_exceptions=(Exception,))` — decorator for sync callables
- `retry_with_backoff_async(...)` — async variant
- Configurable via kwargs or environment variables (`VIBE_RETRY_MAX_RETRIES`, etc.)
- Full jitter: `random.uniform(0, calculated_delay)` to prevent thundering herd
- Each retry logs: attempt number, exception type, delay before next attempt

**Worker integration:** Replace the bare `for i in range(self.max_tries)` loop in `run_op_with_retry` with the retry decorator wrapping `try_run_op`. The worker controls which exceptions are retryable: `ProcessExpired` and general op failures are retryable; `TimeoutError` and `ShuttingDownException` are not.

### 2. Resource Monitoring

**New file:** `vibe_common/vibe_common/resource_monitor.py`

API:
- `ResourceMonitor` class reading cgroup v2 files (`/sys/fs/cgroup/memory.current`, `memory.max`, `cpu.stat`)
- `get_memory_usage() -> MemoryInfo(current_bytes, limit_bytes, usage_percent)`
- `get_cpu_usage() -> CpuInfo(usage_usec, system_usec, nr_periods, nr_throttled)`
- `start_periodic_logging(interval_s=30, logger=None)` — background thread logging usage
- `stop()` — clean shutdown of monitoring thread
- `detect_oom(exit_code, exit_signal) -> bool` — heuristic: SIGKILL + memory near cgroup limit at last sample

Fallback: If cgroup files don't exist (local dev, non-Linux), fall back to `resource` module. Log a warning once.

**Worker integration:** Start at boot, stop during shutdown. On `ProcessExpired` in `try_run_op`, call `detect_oom()` to classify the failure. OOM error messages are factual:

```
Op 'download_sentinel_2' killed (SIGKILL): memory usage 3.8GB / 4.0GB limit (95% at last sample)
```

### 3. Graceful Shutdown

**New file:** `vibe_common/vibe_common/graceful_shutdown.py`

API:
- `ShutdownManager` class with states: `RUNNING`, `DRAINING`, `SHUTTING_DOWN`
- `register_signals(signals=[SIGTERM, SIGINT])` — install signal handlers
- `on_shutdown(callback)` — register cleanup callbacks (LIFO order)
- `is_draining -> bool` — check if shutdown requested
- `wait_for_completion(timeout_s) -> bool` — block until current work finishes or timeout
- `shutdown()` — set DRAINING, wait grace period, run callbacks, set SHUTTING_DOWN

**Worker integration:**
1. SIGTERM arrives → `ShutdownManager` transitions to DRAINING
2. `fetch_work` returns `retry` for new messages; gRPC server stops
3. Wait for current subprocess up to `termination_grace_period_s` (enforced via `pebble_future.result(timeout=grace_period)`)
4. If subprocess finishes: send result reply, shut down cleanly
5. If timeout expires: cancel subprocess, return `retry` for requeue, shut down
6. Cleanup callbacks: stop ResourceMonitor, flush logs

**Messaging fix:** `WorkerMessenger.send()` gets a `timeout` parameter. During shutdown, send attempts capped at a few seconds instead of looping forever.

### 4. Stale Workflow Detection

**Worker side — heartbeat publishing:**

New message type `HeartbeatMessage` in `vibe_common/messaging.py` containing: `run_id`, `op_name`, `worker_id`, `timestamp`, `memory_usage_bytes`.

Published on the existing `updates` topic every `heartbeat_interval_s` (configurable, default 30s). Piggybacks on the existing `get_future_result` monitoring loop (`child_monitoring_period_s=10s`) — no new thread needed.

**Orchestrator side — timeout tracking:**

- `MessageRouter` handles `HeartbeatMessage`, updates `last_heartbeat: dict[str, float]` mapping (`run_id:op_name` → timestamp)
- `RemoteWorkflowRunner._wait_for_reply()` changes from `await self.queue.get()` to `await asyncio.wait_for(self.queue.get(), timeout=heartbeat_timeout_s)`
- On timeout, check heartbeat recency:
  - Recent heartbeat: worker alive, reset wait
  - Stale (no heartbeat within `2 * heartbeat_interval_s`): fail the step with `"Op 'compute_ndvi' on worker 'worker-3' has not reported progress for 120s. Last heartbeat at 2026-02-26T14:32:01Z."`
- Configurable `heartbeat_timeout_s` (default: `2 * heartbeat_interval_s` = 60s)
- Stale steps are marked failed in statestore. No automatic retry of stale steps.

## File Changes

### New files

| File | Purpose |
|---|---|
| `vibe_common/vibe_common/retry.py` | Retry decorator with exponential backoff and jitter |
| `vibe_common/vibe_common/resource_monitor.py` | Cgroup reader, periodic logging, OOM detection |
| `vibe_common/vibe_common/graceful_shutdown.py` | Signal handling, drain state machine, timed grace period |

### Modified files

| File | Changes |
|---|---|
| `vibe_common/vibe_common/messaging.py` | Add `HeartbeatMessage` type |
| `vibe_agent/vibe_agent/worker.py` | Replace bare retry loop with decorator; use `ShutdownManager`; start `ResourceMonitor`; publish heartbeats; classify OOM errors; add timeout to `WorkerMessenger.send()` |
| `vibe_agent/vibe_agent/launch_worker.py` | Wire `ShutdownManager.register_signals()` instead of raw `signal.signal()` |
| `vibe_server/vibe_server/orchestrator.py` | Handle `HeartbeatMessage` in message routing; track last heartbeat per step |
| `vibe_server/vibe_server/workflow/runner/remote_runner.py` | Add `asyncio.wait_for` with heartbeat-aware timeout in reply waiting |

### Not changed

- Op implementations (no checkpoint interface)
- Dapr/Terraform configs (existing resiliency policy, prefetch, consumer timeout are adequate)
- Cache service
- REST API health endpoint

## Test Strategy

- Unit tests for each new module (retry with mock callables, resource monitor with mock cgroup files, shutdown manager state transitions)
- Integration: submit workflow with simulated degraded connectivity, verify retry + eventual success or clean failure
- Integration: scale worker to zero mid-workflow, verify workflow fails with meaningful error via stale heartbeat detection
- Verify structured, factual log output during all failure scenarios
