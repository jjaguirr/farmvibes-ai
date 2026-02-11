# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import logging
import uuid
from unittest.mock import MagicMock, patch

import httpx
import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from vibe_core.logconfig import (
    JSON_FORMAT,
    RequestContextFilter,
    request_duration_var,
    request_endpoint_var,
    request_id_var,
)
from vibe_server.logging_middleware import RequestContextMiddleware


def test_request_context_filter_defaults():
    """Filter should set empty strings when no context is active."""
    f = RequestContextFilter()
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=0, msg="hello", args=(), exc_info=None
    )
    f.filter(record)
    assert record.request_id == ""
    assert record.endpoint == ""
    assert record.duration_ms == ""


def test_request_context_filter_with_context():
    """Filter should inject values from contextvars."""
    token_rid = request_id_var.set("abc-123")
    token_ep = request_endpoint_var.set("/v0/runs")
    token_dur = request_duration_var.set("42.5")

    try:
        f = RequestContextFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="hello",
            args=(),
            exc_info=None,
        )
        f.filter(record)
        assert record.request_id == "abc-123"
        assert record.endpoint == "/v0/runs"
        assert record.duration_ms == "42.5"
    finally:
        request_id_var.reset(token_rid)
        request_endpoint_var.reset(token_ep)
        request_duration_var.reset(token_dur)


def test_json_format_includes_request_fields():
    """JSON_FORMAT string should contain the new context fields."""
    assert "%(request_id)s" in JSON_FORMAT
    assert "%(endpoint)s" in JSON_FORMAT
    assert "%(duration_ms)s" in JSON_FORMAT


# --- RequestContextMiddleware tests ---

def _make_app():
    """Create a minimal FastAPI app with RequestContextMiddleware."""
    app = FastAPI()
    app.add_middleware(RequestContextMiddleware)

    @app.get("/v0/test")
    async def test_endpoint():
        return JSONResponse({"ok": True})

    return app


@pytest.mark.anyio
async def test_middleware_sets_x_request_id_header():
    """Middleware should add X-Request-ID to every response."""
    app = _make_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/v0/test")

    assert resp.status_code == 200
    rid = resp.headers.get("x-request-id")
    assert rid is not None
    assert len(rid) > 0


@pytest.mark.anyio
async def test_middleware_generates_uuid_without_otel():
    """Without an active OTel span, middleware should generate a valid UUID."""
    app = _make_app()

    # Patch trace to return a span with no trace_id (trace_id=0 means invalid)
    mock_span_ctx = MagicMock()
    mock_span_ctx.trace_id = 0
    mock_span = MagicMock()
    mock_span.get_span_context.return_value = mock_span_ctx

    with patch("vibe_server.logging_middleware.trace.get_current_span", return_value=mock_span):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/v0/test")

    rid = resp.headers["x-request-id"]
    # Should be a valid UUID
    parsed = uuid.UUID(rid)
    assert str(parsed) == rid


@pytest.mark.anyio
async def test_middleware_uses_otel_trace_id():
    """With an active OTel span, middleware should use its trace_id."""
    app = _make_app()

    trace_id = 0xABCDEF1234567890ABCDEF1234567890
    mock_span_ctx = MagicMock()
    mock_span_ctx.trace_id = trace_id
    mock_span = MagicMock()
    mock_span.get_span_context.return_value = mock_span_ctx

    with patch("vibe_server.logging_middleware.trace.get_current_span", return_value=mock_span):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/v0/test")

    rid = resp.headers["x-request-id"]
    assert rid == format(trace_id, "032x")


@pytest.mark.anyio
async def test_middleware_sets_endpoint_contextvar():
    """Middleware should populate request_endpoint_var with the request path."""
    app = FastAPI()
    app.add_middleware(RequestContextMiddleware)

    captured_endpoint = None

    @app.get("/v0/workflows")
    async def workflows():
        nonlocal captured_endpoint
        captured_endpoint = request_endpoint_var.get()
        return JSONResponse({"ok": True})

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        await client.get("/v0/workflows")

    assert captured_endpoint == "/v0/workflows"


@pytest.mark.anyio
async def test_middleware_sets_request_id_contextvar():
    """Middleware should populate request_id_var before the handler runs."""
    app = FastAPI()
    app.add_middleware(RequestContextMiddleware)

    captured_rid = None

    @app.get("/v0/test")
    async def test_ep():
        nonlocal captured_rid
        captured_rid = request_id_var.get()
        return JSONResponse({"ok": True})

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/v0/test")

    # The contextvar should match the response header
    assert captured_rid is not None
    assert captured_rid == resp.headers["x-request-id"]


@pytest.mark.anyio
async def test_middleware_sets_duration_contextvar():
    """Middleware should populate request_duration_var after the handler runs."""
    app = FastAPI()
    app.add_middleware(RequestContextMiddleware)

    @app.get("/v0/test")
    async def test_ep():
        return JSONResponse({"ok": True})

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        await client.get("/v0/test")

    # duration_ms should have been set to a numeric string
    duration = request_duration_var.get()
    assert duration != ""
    assert float(duration) >= 0
