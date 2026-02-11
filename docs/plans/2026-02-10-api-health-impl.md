# API Health & Monitoring Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add health/readiness/liveness endpoints, Kubernetes probes, and structured request logging to the FarmVibes REST API.

**Architecture:** Health checks probe Dapr sidecar + state store concurrently with 1s timeouts. Request context (ID, path, duration) is injected into every log record via a `contextvars.ContextVar` populated by Starlette middleware. All new routes use `@version(0)` on `TerravibesAPI`.

**Tech Stack:** FastAPI, Starlette middleware, aiohttp (already used in `VibeDaprClient`), Python stdlib `contextvars`, pytest + anyio for async tests.

---

## Task 1: Extend `JSON_FORMAT` and add `RequestContextFilter` in `logconfig.py`

**Files:**
- Modify: `src/vibe_core/vibe_core/logconfig.py`
- Test: `src/vibe_core/tests/test_logconfig.py` (create)

### Step 1: Write the failing tests

Create `src/vibe_core/tests/test_logconfig.py`:

```python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import logging
from contextvars import ContextVar

import pytest

from vibe_core.logconfig import (
    JSON_FORMAT,
    RequestContext,
    RequestContextFilter,
    configure_logging,
)


def _capture_log_record(message: str, level: int = logging.INFO) -> logging.LogRecord:
    """Emit one log record through a NullHandler-backed logger and return it."""
    records = []

    class Capture(logging.Handler):
        def emit(self, record: logging.LogRecord):
            records.append(record)

    logger = logging.getLogger("test_logconfig_capture")
    logger.setLevel(logging.DEBUG)
    h = Capture()
    h.addFilter(RequestContextFilter())
    logger.addHandler(h)
    try:
        logger.log(level, message)
    finally:
        logger.removeHandler(h)
    return records[0]


def test_json_format_contains_request_fields():
    assert "request_id" in JSON_FORMAT
    assert "request_path" in JSON_FORMAT
    assert "duration_ms" in JSON_FORMAT


def test_request_context_filter_defaults_to_empty():
    record = _capture_log_record("hello")
    assert record.request_id == ""
    assert record.request_path == ""
    assert record.duration_ms == ""


def test_request_context_filter_injects_values():
    RequestContext.set("req-123", "/v0/health", "42")
    try:
        record = _capture_log_record("hello")
        assert record.request_id == "req-123"
        assert record.request_path == "/v0/health"
        assert record.duration_ms == "42"
    finally:
        RequestContext.clear()


def test_request_context_filter_clears():
    RequestContext.set("req-999", "/v0/liveness", "5")
    RequestContext.clear()
    record = _capture_log_record("hello")
    assert record.request_id == ""
```

### Step 2: Run tests to confirm they fail

```bash
cd /Users/jose/Documents/Work/01-PROMETHEUS/tasks/08/model_a
python -m pytest src/vibe_core/tests/test_logconfig.py -v 2>&1 | tail -20
```

Expected: `ImportError` or `FAILED` — `RequestContext`, `RequestContextFilter` don't exist yet.

### Step 3: Implement in `logconfig.py`

**3a.** Add imports at top of file (after existing imports):
```python
from contextvars import ContextVar
from typing import Tuple
```

**3b.** After the `JSON_FORMAT` constant, add:
```python
JSON_FORMAT = (
    '{"app_id": "%(app)s", "instance": "%(hostname)s", "level": "%(levelname)s", '
    '"msg": %(json_message)s, "scope": "%(name)s", "time": "%(asctime)s", "type": "log", '
    '"ver": "dev", "request_id": "%(request_id)s", "path": "%(request_path)s", '
    '"duration_ms": "%(duration_ms)s"}'
)
```

Wait — `JSON_FORMAT` is already defined. Replace the existing constant entirely.

**3c.** Add the `RequestContext` helper and `RequestContextFilter` class after `JsonMessageFilter`:

```python
# ContextVar holding (request_id, request_path, duration_ms); empty strings when unset.
_REQUEST_CONTEXT: ContextVar[Tuple[str, str, str]] = ContextVar(
    "_REQUEST_CONTEXT", default=("", "", "")
)


class RequestContext:
    """Namespace for setting/clearing per-request log context."""

    @staticmethod
    def set(request_id: str, path: str, duration_ms: str) -> None:
        _REQUEST_CONTEXT.set((request_id, path, duration_ms))

    @staticmethod
    def clear() -> None:
        _REQUEST_CONTEXT.set(("", "", ""))


class RequestContextFilter(Filter):
    """Injects request_id, request_path, duration_ms from ContextVar into log records."""

    def filter(self, record: LogRecord) -> bool:
        request_id, path, duration_ms = _REQUEST_CONTEXT.get()
        record.request_id = request_id
        record.request_path = path
        record.duration_ms = duration_ms
        return True
```

**3d.** In `configure_logging()`, add `RequestContextFilter` to every handler alongside the existing filters:

```python
        handler.addFilter(RequestContextFilter())
```

Add it after `handler.addFilter(JsonMessageFilter())`.

### Step 4: Run tests to confirm they pass

```bash
python -m pytest src/vibe_core/tests/test_logconfig.py -v 2>&1 | tail -20
```

Expected: all 4 tests `PASSED`.

### Step 5: Commit

```bash
jj describe -m "feat(logging): add RequestContextFilter and request fields to JSON_FORMAT"
jj new
```

---

## Task 2: Add `health_check()` method to `TerravibesProvider` in `server.py`

**Files:**
- Modify: `src/vibe_server/vibe_server/server.py`
- Test: `src/vibe_server/tests/test_health.py` (create)

The health check probes three Dapr endpoints via `aiohttp`. All run concurrently under `asyncio.gather` with individual 1s timeouts.

### Step 1: Write the failing tests

Create `src/vibe_server/tests/test_health.py`:

```python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vibe_dev.testing import anyio_backend  # noqa: F401


async def _make_mock_response(status: int) -> MagicMock:
    mock = MagicMock()
    mock.status = status
    mock.__aenter__ = AsyncMock(return_value=mock)
    mock.__aexit__ = AsyncMock(return_value=False)
    return mock


@pytest.mark.anyio
@patch("vibe_server.server.aiohttp.ClientSession")
async def test_health_all_ok(mock_session_cls):
    from vibe_server.server import TerravibesProvider
    from vibe_server.href_handler import LocalHrefHandler

    ok_resp = await _make_mock_response(200)
    not_found_resp = await _make_mock_response(404)

    session_instance = MagicMock()
    session_instance.__aenter__ = AsyncMock(return_value=session_instance)
    session_instance.__aexit__ = AsyncMock(return_value=False)
    # dapr=200, state_store=404 (key absent = ok), pubsub=200
    session_instance.get = AsyncMock(side_effect=[ok_resp, not_found_resp, ok_resp])
    mock_session_cls.return_value = session_instance

    provider = TerravibesProvider(LocalHrefHandler("/tmp"))
    result = await provider.health_check()

    assert result["status"] == "healthy"
    assert result["checks"]["dapr"] == "ok"
    assert result["checks"]["state_store"] == "ok"
    assert result["checks"]["pubsub"] == "ok"


@pytest.mark.anyio
@patch("vibe_server.server.aiohttp.ClientSession")
async def test_health_redis_down(mock_session_cls):
    from vibe_server.server import TerravibesProvider
    from vibe_server.href_handler import LocalHrefHandler

    ok_resp = await _make_mock_response(200)
    error_resp = await _make_mock_response(500)

    session_instance = MagicMock()
    session_instance.__aenter__ = AsyncMock(return_value=session_instance)
    session_instance.__aexit__ = AsyncMock(return_value=False)
    # dapr=200, state_store=500 (Redis down), pubsub=200
    session_instance.get = AsyncMock(side_effect=[ok_resp, error_resp, ok_resp])
    mock_session_cls.return_value = session_instance

    provider = TerravibesProvider(LocalHrefHandler("/tmp"))
    result = await provider.health_check()

    assert result["status"] == "degraded"
    assert result["checks"]["dapr"] == "ok"
    assert "error" in result["checks"]["state_store"]
    assert result["checks"]["pubsub"] == "ok"


@pytest.mark.anyio
@patch("vibe_server.server.aiohttp.ClientSession")
async def test_health_dapr_connection_error(mock_session_cls):
    from vibe_server.server import TerravibesProvider
    from vibe_server.href_handler import LocalHrefHandler

    session_instance = MagicMock()
    session_instance.__aenter__ = AsyncMock(return_value=session_instance)
    session_instance.__aexit__ = AsyncMock(return_value=False)
    session_instance.get = AsyncMock(side_effect=Exception("connection refused"))
    mock_session_cls.return_value = session_instance

    provider = TerravibesProvider(LocalHrefHandler("/tmp"))
    result = await provider.health_check()

    assert result["status"] == "degraded"
    for check in result["checks"].values():
        assert "error" in check
```

### Step 2: Run tests to confirm they fail

```bash
python -m pytest src/vibe_server/tests/test_health.py -v 2>&1 | tail -20
```

Expected: `ImportError` — `health_check` method doesn't exist yet.

### Step 3: Implement `health_check()` in `server.py`

**3a.** Add `import aiohttp` at the top of `server.py` (after existing imports):
```python
import aiohttp
```

**3b.** Add `HEALTH_CHECK_TIMEOUT_S: Final[float] = 1.0` after the existing constants near the top of the file.

**3c.** Add to `TerravibesProvider`, after `system_metrics()`:

```python
async def health_check(self) -> Dict[str, Any]:
    """Probe Dapr sidecar, state store, and pubsub. Returns status dict."""
    from dapr.conf import settings as dapr_settings

    host = dapr_settings.DAPR_RUNTIME_HOST
    port = dapr_settings.DAPR_HTTP_PORT
    base = f"http://{host}:{port}"

    checks: Dict[str, str] = {}

    async def _probe(name: str, url: str, ok_statuses: tuple) -> None:
        try:
            timeout = aiohttp.ClientTimeout(total=HEALTH_CHECK_TIMEOUT_S)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as resp:
                    if resp.status in ok_statuses:
                        checks[name] = "ok"
                    else:
                        checks[name] = f"error: HTTP {resp.status}"
        except Exception as exc:
            checks[name] = f"error: {exc}"

    await asyncio.gather(
        _probe("dapr", f"{base}/v1.0/healthz", (200,)),
        _probe("state_store", f"{base}/v1.0/state/statestore/health-probe", (200, 404)),
        _probe("pubsub", f"{base}/v1.0/healthz/outbound", (200,)),
    )

    overall = "healthy" if all(v == "ok" for v in checks.values()) else "degraded"
    return {"status": overall, "checks": checks}
```

### Step 4: Run tests to confirm they pass

```bash
python -m pytest src/vibe_server/tests/test_health.py -v 2>&1 | tail -20
```

Expected: all 3 tests `PASSED`.

### Step 5: Commit

```bash
jj describe -m "feat(health): add health_check() method to TerravibesProvider"
jj new
```

---

## Task 3: Add health/liveness/readiness/status routes to `TerravibesAPI`

**Files:**
- Modify: `src/vibe_server/vibe_server/server.py`
- Test: `src/vibe_server/tests/test_health.py` (extend)

Routes to add inside `TerravibesAPI.__init__`, with `@version(0)`:

- `GET /liveness` → HTTP 200, `{"status": "alive"}`
- `GET /readiness` → calls `health_check()`, HTTP 200 or 503
- `GET /health` → alias for readiness
- `GET /status` → alias for readiness (restores CLI compat)

### Step 1: Write the failing tests

Append to `src/vibe_server/tests/test_health.py`:

```python
@pytest.mark.anyio
@patch("vibe_server.server.TerravibesProvider.health_check")
async def test_liveness_endpoint(mock_health):
    """Liveness always returns 200 regardless of dependency state."""
    from httpx import AsyncClient, ASGITransport
    from vibe_server.server import TerravibesAPI
    from vibe_server.href_handler import LocalHrefHandler

    api = TerravibesAPI(href_handler=LocalHrefHandler("/tmp"))
    async with AsyncClient(
        transport=ASGITransport(app=api.versioned_wrapper), base_url="http://test"
    ) as client:
        resp = await client.get("/v0/liveness")
    assert resp.status_code == 200
    assert resp.json()["status"] == "alive"
    mock_health.assert_not_called()


@pytest.mark.anyio
@patch("vibe_server.server.TerravibesProvider.health_check")
async def test_readiness_healthy(mock_health):
    mock_health.return_value = {
        "status": "healthy",
        "checks": {"dapr": "ok", "state_store": "ok", "pubsub": "ok"},
    }
    from httpx import AsyncClient, ASGITransport
    from vibe_server.server import TerravibesAPI
    from vibe_server.href_handler import LocalHrefHandler

    api = TerravibesAPI(href_handler=LocalHrefHandler("/tmp"))
    async with AsyncClient(
        transport=ASGITransport(app=api.versioned_wrapper), base_url="http://test"
    ) as client:
        resp = await client.get("/v0/readiness")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


@pytest.mark.anyio
@patch("vibe_server.server.TerravibesProvider.health_check")
async def test_readiness_degraded_returns_503(mock_health):
    mock_health.return_value = {
        "status": "degraded",
        "checks": {"dapr": "ok", "state_store": "error: HTTP 500", "pubsub": "ok"},
    }
    from httpx import AsyncClient, ASGITransport
    from vibe_server.server import TerravibesAPI
    from vibe_server.href_handler import LocalHrefHandler

    api = TerravibesAPI(href_handler=LocalHrefHandler("/tmp"))
    async with AsyncClient(
        transport=ASGITransport(app=api.versioned_wrapper), base_url="http://test"
    ) as client:
        resp = await client.get("/v0/readiness")
    assert resp.status_code == 503


@pytest.mark.anyio
@patch("vibe_server.server.TerravibesProvider.health_check")
async def test_status_alias(mock_health):
    """/status returns the same payload as /readiness."""
    mock_health.return_value = {
        "status": "healthy",
        "checks": {"dapr": "ok", "state_store": "ok", "pubsub": "ok"},
    }
    from httpx import AsyncClient, ASGITransport
    from vibe_server.server import TerravibesAPI
    from vibe_server.href_handler import LocalHrefHandler

    api = TerravibesAPI(href_handler=LocalHrefHandler("/tmp"))
    async with AsyncClient(
        transport=ASGITransport(app=api.versioned_wrapper), base_url="http://test"
    ) as client:
        resp = await client.get("/v0/status")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"
```

### Step 2: Run tests to confirm they fail

```bash
python -m pytest src/vibe_server/tests/test_health.py -v -k "endpoint or healthy or degraded or alias" 2>&1 | tail -20
```

Expected: `FAILED` — routes don't exist yet.

### Step 3: Add routes to `TerravibesAPI.__init__`

Add after the existing `@self.get("/system-metrics")` block (around line 638 in `server.py`), still inside `__init__`:

```python
        @self.get("/liveness")
        @version(0)
        async def terravibes_liveness() -> Dict[str, str]:
            """Kubernetes liveness probe. Always returns 200 if the process is alive."""
            return {"status": "alive"}

        async def _readiness_response() -> JSONResponse:
            result = await self.terravibes.health_check()
            http_status = status.HTTP_200_OK if result["status"] == "healthy" else status.HTTP_503_SERVICE_UNAVAILABLE
            return JSONResponse(status_code=http_status, content=result)

        @self.get("/readiness")
        @version(0)
        async def terravibes_readiness() -> JSONResponse:
            """Kubernetes readiness probe. Returns 503 if any dependency is down."""
            return await _readiness_response()

        @self.get("/health")
        @version(0)
        async def terravibes_health() -> JSONResponse:
            """Full dependency health report."""
            return await _readiness_response()

        @self.get("/status")
        @version(0)
        async def terravibes_status() -> JSONResponse:
            """Health summary (CLI-compatible alias for /health)."""
            return await _readiness_response()
```

### Step 4: Run all health tests

```bash
python -m pytest src/vibe_server/tests/test_health.py -v 2>&1 | tail -25
```

Expected: all tests `PASSED`.

### Step 5: Commit

```bash
jj describe -m "feat(health): add liveness/readiness/health/status endpoints"
jj new
```

---

## Task 4: Add `RequestLoggingMiddleware` to `server.py`

**Files:**
- Modify: `src/vibe_server/vibe_server/server.py`
- Test: `src/vibe_server/tests/test_health.py` (extend)

The middleware generates a UUID request ID, sets `RequestContext`, calls the next handler, then clears context and logs one structured line.

### Step 1: Write the failing test

Append to `src/vibe_server/tests/test_health.py`:

```python
@pytest.mark.anyio
@patch("vibe_server.server.TerravibesProvider.health_check")
async def test_request_logging_middleware_sets_context(mock_health, caplog):
    """Middleware injects request_id into log output."""
    import logging
    mock_health.return_value = {
        "status": "healthy",
        "checks": {"dapr": "ok", "state_store": "ok", "pubsub": "ok"},
    }
    from httpx import AsyncClient, ASGITransport
    from vibe_server.server import TerravibesAPI
    from vibe_server.href_handler import LocalHrefHandler

    api = TerravibesAPI(href_handler=LocalHrefHandler("/tmp"))
    with caplog.at_level(logging.INFO):
        async with AsyncClient(
            transport=ASGITransport(app=api.versioned_wrapper), base_url="http://test"
        ) as client:
            resp = await client.get("/v0/health")
    assert resp.status_code == 200
    # At least one log line should mention the path
    assert any("/v0/health" in r.message for r in caplog.records)
```

### Step 2: Run test to confirm it fails

```bash
python -m pytest src/vibe_server/tests/test_health.py::test_request_logging_middleware_sets_context -v 2>&1 | tail -15
```

Expected: `FAILED` — middleware not implemented yet.

### Step 3: Implement `RequestLoggingMiddleware`

**3a.** Add imports to `server.py`:
```python
import uuid
from contextvars import copy_context
from starlette.middleware.base import BaseHTTPMiddleware
from vibe_core.logconfig import RequestContext
```

**3b.** Add the middleware class just before `class TerravibesAPI`:

```python
class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Generates a request ID, injects it into log context, logs one line per request."""

    _logger = logging.getLogger(__name__ + ".RequestLoggingMiddleware")

    async def dispatch(self, request: Any, call_next: Any) -> Any:
        request_id = str(uuid4())
        path = request.url.path
        start = datetime.utcnow()
        RequestContext.set(request_id, path, "")
        try:
            response = await call_next(request)
        finally:
            duration_ms = str(int((datetime.utcnow() - start).total_seconds() * 1000))
            RequestContext.set(request_id, path, duration_ms)
            self._logger.info(
                f"method={request.method} path={path} "
                f"status={getattr(response, 'status_code', '?')} "
                f"duration_ms={duration_ms} request_id={request_id}"
            )
            RequestContext.clear()
        return response
```

**3c.** In `TerravibesAPI.__init__`, add the middleware to `self.versioned_wrapper` after it is created (after the `CORSMiddleware` block):

```python
        self.versioned_wrapper.add_middleware(RequestLoggingMiddleware)
```

### Step 4: Run tests

```bash
python -m pytest src/vibe_server/tests/test_health.py -v 2>&1 | tail -25
```

Expected: all tests `PASSED`.

### Step 5: Commit

```bash
jj describe -m "feat(logging): add RequestLoggingMiddleware with per-request context"
jj new
```

---

## Task 5: Add Kubernetes probes to `restapi.tf`

**Files:**
- Modify: `src/vibe_core/vibe_core/terraform/services/restapi.tf`

No tests for Terraform HCL — validate with `terraform validate` instead.

### Step 1: Add probe blocks

In the `container` block of `kubernetes_deployment.restapi`, add after the existing `port { container_port = 3000 }` block:

```hcl
          liveness_probe {
            http_get {
              path = "/v0/liveness"
              port = 3000
            }
            initial_delay_seconds = 15
            failure_threshold     = 3
          }

          readiness_probe {
            http_get {
              path = "/v0/readiness"
              port = 3000
            }
            initial_delay_seconds = 20
            failure_threshold     = 3
          }
```

### Step 2: Validate Terraform syntax

```bash
cd src/vibe_core/vibe_core/terraform/services
terraform init -backend=false 2>&1 | tail -5
terraform validate 2>&1
```

Expected: `Success! The configuration is valid.`

### Step 3: Commit

```bash
cd /Users/jose/Documents/Work/01-PROMETHEUS/tasks/08/model_a
jj describe -m "feat(k8s): add liveness and readiness probes to restapi deployment"
jj new
```

---

## Task 6: Run full test suite and verify

### Step 1: Run all new tests together

```bash
cd /Users/jose/Documents/Work/01-PROMETHEUS/tasks/08/model_a
python -m pytest src/vibe_core/tests/test_logconfig.py src/vibe_server/tests/test_health.py -v 2>&1 | tail -30
```

Expected: all tests `PASSED`, no regressions.

### Step 2: Run existing server tests to check for regressions

```bash
python -m pytest src/vibe_server/tests/ -v --ignore=src/vibe_server/tests/test_health.py 2>&1 | tail -20
```

Expected: same pass/fail ratio as before this work.

### Step 3: Final commit (squash description if needed)

```bash
jj describe -m "chore: verify test suite passes after health monitoring implementation"
```

---

## VM Verification (after pushing)

Once the user has pushed the branch, on the GCP VM:

```bash
source ~/farmvibes-env_08_a.sh
cd ~/farmvibes-ai-work/08_a && git pull
pip install -e src/vibe_core/ -q

# Redeploy
farmvibes-ai local setup --port $_FARMVIBES_PORT

# Test endpoints
curl -s http://localhost:$_FARMVIBES_PORT/v0/liveness | python3 -m json.tool
curl -s http://localhost:$_FARMVIBES_PORT/v0/health | python3 -m json.tool
curl -s http://localhost:$_FARMVIBES_PORT/v0/status | python3 -m json.tool

# Kill Redis and confirm health reports degraded
kubectl scale deployment redis --replicas=0 -n farmvibes
sleep 5
curl -s http://localhost:$_FARMVIBES_PORT/v0/health | python3 -m json.tool
# Expect: status=degraded, state_store=error

# Restore Redis
kubectl scale deployment redis --replicas=1 -n farmvibes
```
