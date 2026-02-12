# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from prometheus_client import REGISTRY

from vibe_server.metrics import (
    REQUEST_COUNT,
    REQUEST_DURATION,
    WORKFLOW_RUNS_ACTIVE,
    WORKFLOW_RUNS_CREATED,
    PrometheusMiddleware,
    metrics_endpoint,
    normalize_path,
    record_workflow_created,
    update_active_runs_gauge,
)


def test_normalize_path_replaces_uuids():
    """UUID path segments should be normalized to {id}."""
    path = "/v0/runs/a1b2c3d4-e5f6-7890-abcd-ef1234567890"
    assert normalize_path(path) == "/v0/runs/{id}"


def test_normalize_path_replaces_multiple_uuids():
    """Multiple UUIDs in a path should all be replaced."""
    path = "/v0/runs/a1b2c3d4-e5f6-7890-abcd-ef1234567890/tasks/b2c3d4e5-f6a7-8901-bcde-f12345678901"
    assert normalize_path(path) == "/v0/runs/{id}/tasks/{id}"


def test_normalize_path_no_uuid():
    """Paths without UUIDs should be unchanged."""
    path = "/v0/workflows/helloworld"
    assert normalize_path(path) == "/v0/workflows/helloworld"


def test_record_workflow_created():
    """record_workflow_created should increment the counter."""
    before = REGISTRY.get_sample_value(
        "workflow_runs_total", {"status": "created"}
    ) or 0
    record_workflow_created()
    after = REGISTRY.get_sample_value(
        "workflow_runs_total", {"status": "created"}
    )
    assert after == before + 1


@pytest.mark.anyio
async def test_update_active_runs_gauge_with_runs():
    """Active runs gauge should reflect the number of run IDs."""
    import vibe_server.metrics as m

    m._active_gauge_last_refresh = 0  # force refresh

    mock_store = AsyncMock()
    mock_store.retrieve = AsyncMock(return_value=["run-1", "run-2", "run-3"])

    await update_active_runs_gauge(mock_store)
    assert REGISTRY.get_sample_value("workflow_runs_active") == 3


@pytest.mark.anyio
async def test_update_active_runs_gauge_no_runs():
    """Active runs gauge should be 0 when no runs exist (KeyError)."""
    import vibe_server.metrics as m

    m._active_gauge_last_refresh = 0

    mock_store = AsyncMock()
    mock_store.retrieve = AsyncMock(side_effect=KeyError("Key runs not found"))

    await update_active_runs_gauge(mock_store)
    assert REGISTRY.get_sample_value("workflow_runs_active") == 0


@pytest.mark.anyio
async def test_update_active_runs_gauge_cached():
    """Gauge should not query state store when cache is fresh."""
    import time
    import vibe_server.metrics as m

    m._active_gauge_last_refresh = time.time()  # just refreshed

    mock_store = AsyncMock()
    mock_store.retrieve = AsyncMock(return_value=["run-1"])

    await update_active_runs_gauge(mock_store)
    mock_store.retrieve.assert_not_called()


@pytest.mark.anyio
async def test_update_active_runs_gauge_error_silent():
    """Gauge refresh errors should be silently caught."""
    import vibe_server.metrics as m

    m._active_gauge_last_refresh = 0

    mock_store = AsyncMock()
    mock_store.retrieve = AsyncMock(side_effect=RuntimeError("connection failed"))

    # Should not raise
    await update_active_runs_gauge(mock_store)


# --- PrometheusMiddleware tests ---

def _make_app():
    """Create a minimal FastAPI app with PrometheusMiddleware for testing."""
    app = FastAPI()
    app.add_middleware(PrometheusMiddleware)

    @app.get("/v0/workflows")
    async def workflows():
        return JSONResponse({"workflows": []})

    @app.get("/v0/runs/{run_id}")
    async def get_run(run_id: str):
        return JSONResponse({"id": run_id})

    @app.get("/metrics")
    async def metrics():
        return JSONResponse({"skip": True})

    @app.get("/healthz/live")
    async def liveness():
        return JSONResponse({"status": "alive"})

    @app.get("/healthz/ready")
    async def readiness():
        return JSONResponse({"status": "ready"})

    return app


@pytest.mark.anyio
async def test_prometheus_middleware_records_request():
    """Middleware should increment counter and observe duration for normal requests."""
    app = _make_app()
    before = REGISTRY.get_sample_value(
        "http_requests_total", {"method": "GET", "endpoint": "/v0/workflows", "status_code": "200"}
    ) or 0

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/v0/workflows")

    assert resp.status_code == 200
    after = REGISTRY.get_sample_value(
        "http_requests_total", {"method": "GET", "endpoint": "/v0/workflows", "status_code": "200"}
    )
    assert after == before + 1

    # Duration should have been observed
    count = REGISTRY.get_sample_value(
        "http_request_duration_seconds_count", {"method": "GET", "endpoint": "/v0/workflows"}
    )
    assert count is not None and count >= 1


@pytest.mark.anyio
async def test_prometheus_middleware_normalizes_uuid_paths():
    """Middleware should replace UUIDs in path labels with {id}."""
    app = _make_app()
    run_id = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(f"/v0/runs/{run_id}")

    assert resp.status_code == 200
    count = REGISTRY.get_sample_value(
        "http_requests_total",
        {"method": "GET", "endpoint": "/v0/runs/{id}", "status_code": "200"},
    )
    assert count is not None and count >= 1


@pytest.mark.anyio
async def test_prometheus_middleware_skips_metrics_path():
    """Middleware should not record metrics for /metrics requests."""
    app = _make_app()
    before = REGISTRY.get_sample_value(
        "http_requests_total", {"method": "GET", "endpoint": "/metrics", "status_code": "200"}
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/metrics")

    assert resp.status_code == 200
    after = REGISTRY.get_sample_value(
        "http_requests_total", {"method": "GET", "endpoint": "/metrics", "status_code": "200"}
    )
    # Should not have changed — /metrics is in _SKIP_PATHS
    assert after == before


@pytest.mark.anyio
async def test_prometheus_middleware_skips_healthz_paths():
    """Middleware should not record metrics for /healthz/* requests."""
    app = _make_app()
    before_live = REGISTRY.get_sample_value(
        "http_requests_total", {"method": "GET", "endpoint": "/healthz/live", "status_code": "200"}
    )
    before_ready = REGISTRY.get_sample_value(
        "http_requests_total", {"method": "GET", "endpoint": "/healthz/ready", "status_code": "200"}
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        await client.get("/healthz/live")
        await client.get("/healthz/ready")

    after_live = REGISTRY.get_sample_value(
        "http_requests_total", {"method": "GET", "endpoint": "/healthz/live", "status_code": "200"}
    )
    after_ready = REGISTRY.get_sample_value(
        "http_requests_total", {"method": "GET", "endpoint": "/healthz/ready", "status_code": "200"}
    )
    assert after_live == before_live
    assert after_ready == before_ready


@pytest.mark.anyio
async def test_metrics_endpoint_returns_prometheus_format():
    """metrics_endpoint should return Prometheus exposition format."""
    resp = await metrics_endpoint()
    assert resp.status_code == 200
    assert b"http_requests_total" in resp.body
    assert "text/plain" in resp.media_type or "openmetrics" in resp.media_type
