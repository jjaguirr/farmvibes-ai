# Integration Test Suite Design

## Problem

Scattered test files, no coherent integration test suite exercising the full pipeline.
Unit tests mock heavily; real problems (misconfigured services, broken workflows, bad error handling) only surface against the live cluster.

## Decisions

- **Two-tier workflow testing:** fast schema regression (all workflows via API describe) + slow execution regression (parametrized, starting with helloworld)
- **Helpers in `vibe_dev/testing/integration.py`:** reusable by other projects, no new packages
- **Approach B:** test modules by concern, shared helpers in `vibe_dev/testing/`
- **No new dependencies:** requests (from vibe_core), pytest (from vibe_dev), subprocess for CLI
- **Existing tests untouched:** `test_cluster_integration.py` keeps its inline fixtures

## Helpers Module: `vibe_dev/testing/integration.py`

### ClusterClient
Wraps requests against a configurable base URL. Methods: `get`, `post`, `health`, `list_workflows`, `describe_workflow`, `submit_run`, `get_run`, `list_runs`, `cancel_run`, `delete_run`.

### WorkflowPoller
Polls a run until terminal state or timeout. Raises `TimeoutError` with last known status. Defaults: 120s timeout, 5s interval.

### WorkflowSpec
Dataclass for parametrized execution tests: `name`, `geometry`, `time_range`, `expected_sinks`, `timeout_s`.

### URL Resolution Order
`FARMVIBES_AI_BASE_URL` env var > `~/.config/farmvibes-ai/remote_service_url` > `~/.config/farmvibes-ai/service_url` > `http://127.0.0.1:31108`

## Fixtures: `src/tests_local_cluster/conftest.py`

- `cluster_url` — session-scoped, resolved once
- `cluster_client` — session-scoped ClusterClient
- `workflow_poller` — session-scoped WorkflowPoller
- `vibe_client` — session-scoped FarmvibesAiClient
- `all_workflows` — session-scoped, fetched from API once

Markers: `fast` (seconds), `slow` (minutes).

## Test Modules

### `test_health.py` [fast]
- Root endpoint (`/v0/`)
- System metrics (`/v0/system-metrics`)
- Liveness probe (`/healthz/live`)
- Readiness probe (`/healthz/ready`)
- Detailed health (`/v0/health`)
- Workflows list (`/v0/workflows`)

### `test_schema_regression.py` [fast]
- `test_workflow_describe` — parametrized over all workflows, validates response fields
- `test_workflow_yaml` — parametrized, validates YAML representation parses correctly

### `test_workflow_execution.py` [slow]
- `test_workflow_run` — parametrized via WorkflowSpec (starts with helloworld)
- `test_workflow_caching` — run helloworld twice, verify cache hit

### `test_cli.py` [fast]
- `test_cli_status` — exit code 0, output contains URL
- `test_cli_health` — exit code reflects cluster state

### `test_failures.py` [slow]
- Invalid workflow name → error response
- Invalid geometry → error response
- Invalid date range → error response
- Cancel running workflow → cancelled status
- Get nonexistent run → 404

## Constraints

- Tests assert what the code says should work; failures indicate real problems
- No xfail, no skip for deployed endpoints
- Schema regression is dynamic (workflow list from live API)
- Timeouts prevent silent hangs
- `pytest -m fast` < 30s, full suite < 10min
