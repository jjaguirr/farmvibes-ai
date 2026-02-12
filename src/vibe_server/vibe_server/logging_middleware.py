# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""FastAPI middleware that populates per-request logging context variables."""

import time
import uuid

from fastapi import Request
from opentelemetry import trace
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from vibe_core.logconfig import request_duration_var, request_endpoint_var, request_id_var


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Sets request_id, endpoint, and duration_ms contextvars for each request.

    The request_id is derived from the active OpenTelemetry trace context when
    available, otherwise a new UUID is generated. All values are picked up by
    :class:`vibe_core.logconfig.RequestContextFilter` and injected into log
    records automatically.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        span = trace.get_current_span()
        span_ctx = span.get_span_context()
        if span_ctx and span_ctx.trace_id:
            rid = format(span_ctx.trace_id, "032x")
        else:
            rid = str(uuid.uuid4())

        request_id_var.set(rid)
        request_endpoint_var.set(request.url.path)

        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        request_duration_var.set(str(duration_ms))

        response.headers["X-Request-ID"] = rid
        return response
