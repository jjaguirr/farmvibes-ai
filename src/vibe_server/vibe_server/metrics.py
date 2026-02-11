# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Prometheus metrics definitions, middleware, and endpoint for the FarmVibes REST API."""

import asyncio
import logging
import re
import time

from fastapi import Request
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from vibe_common.constants import RUNS_KEY
from vibe_common.statestore import StateStore

LOGGER = logging.getLogger(__name__)

_UUID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
    re.IGNORECASE,
)

# Paths excluded from request metrics to avoid self-referential noise
_SKIP_PATHS = frozenset(("/metrics", "/healthz/live", "/healthz/ready"))

# --- Metric definitions ---

REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"],
)

REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

WORKFLOW_RUNS_CREATED = Counter(
    "workflow_runs_total",
    "Total workflow runs created",
    ["status"],
)

WORKFLOW_RUNS_ACTIVE = Gauge(
    "workflow_runs_active",
    "Number of active (non-finished) workflow runs",
)

# --- Gauge refresh cache ---
_active_gauge_last_refresh: float = 0.0
_GAUGE_REFRESH_INTERVAL_S: float = 30.0
_gauge_refresh_lock = asyncio.Lock()


def normalize_path(path: str) -> str:
    """Replace UUID path segments with {id} to prevent label cardinality explosion."""
    return _UUID_RE.sub("{id}", path)


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Middleware that records request count and duration as Prometheus metrics."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if request.url.path in _SKIP_PATHS:
            return await call_next(request)

        method = request.method
        path = normalize_path(request.url.path)

        start = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start

        REQUEST_COUNT.labels(
            method=method, endpoint=path, status_code=response.status_code
        ).inc()
        REQUEST_DURATION.labels(method=method, endpoint=path).observe(duration)

        return response


def record_workflow_created():
    """Record that a new workflow run was successfully created."""
    WORKFLOW_RUNS_CREATED.labels(status="created").inc()


async def update_active_runs_gauge(state_store: StateStore) -> None:
    """Refresh the active-runs gauge from the state store, with caching.

    Only queries the state store if at least _GAUGE_REFRESH_INTERVAL_S have
    elapsed since the last refresh. Uses an asyncio.Lock to prevent concurrent
    scrapes from stampeding the state store. Errors are logged and silently
    ignored so the /metrics endpoint always succeeds.
    """
    global _active_gauge_last_refresh

    async with _gauge_refresh_lock:
        now = time.time()
        if now - _active_gauge_last_refresh < _GAUGE_REFRESH_INTERVAL_S:
            return

        try:
            run_ids = await state_store.retrieve(RUNS_KEY)
            if isinstance(run_ids, list):
                WORKFLOW_RUNS_ACTIVE.set(len(run_ids))
            _active_gauge_last_refresh = now
        except KeyError:
            # No runs exist yet
            WORKFLOW_RUNS_ACTIVE.set(0)
            _active_gauge_last_refresh = now
        except Exception:
            LOGGER.warning("Failed to refresh active runs gauge", exc_info=True)


async def metrics_endpoint() -> Response:
    """Handler for the /metrics endpoint — returns Prometheus exposition format."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
