# API Health & Monitoring Design

**Date:** 2026-02-10
**Branch:** `api-health_task08_model_a`
**Scope:** Add health endpoints, liveness/readiness probes, and structured request logging to the FarmVibes REST API.

---

## What Already Exists

- `GET /system-metrics` — CPU, memory, disk via `MetricsDict`. Kept as-is.
- `configure_logging()` in `logconfig.py` — JSON output via `JSON_FORMAT`, `AppFilter`, `HostnameFilter`, `JsonMessageFilter`. Extended, not replaced.
- `setup_telemetry()` + `@add_trace` — OTel tracing on all endpoints. Untouched.
- Dapr sidecar Prometheus scraping at port 9090. Untouched.
- CLI `_check_api_health()` calls `GET /v0/status`. Currently returns 404 — fixed by this work.

---

## Health Endpoints

All new routes use `@version(0)` and are added in `TerravibesAPI.__init__`.

### `GET /v0/liveness`
Pure in-process check. No I/O, no Dapr calls. Always returns HTTP 200.

```json
{"status": "alive"}
```

Used as the Kubernetes **liveness probe** target.

### `GET /v0/readiness` (and `/v0/health`, `/v0/status` — same handler)
Probes each dependency concurrently with a 1-second timeout. Returns HTTP 200 if all checks pass, HTTP 503 if any fail. The endpoint always responds regardless of dependency state.

```json
{
  "status": "healthy" | "degraded" | "down",
  "checks": {
    "dapr":        "ok" | "error: <reason>",
    "state_store": "ok" | "error: <reason>",
    "pubsub":      "ok" | "error: <reason>"
  }
}
```

**Dependency checks (all async, 1s timeout each, run concurrently):**

| Check | Mechanism | Healthy signal |
|---|---|---|
| `dapr` | `GET http://localhost:{DAPR_HTTP_PORT}/v1.0/healthz` | HTTP 200 |
| `state_store` | `GET /v1.0/state/statestore/health-probe` | HTTP 404 (key absent = Redis up) |
| `pubsub` | `GET http://localhost:{DAPR_HTTP_PORT}/v1.0/healthz/outbound` | HTTP 200 |

`/health` and `/status` are aliases for `/readiness` (same handler function).
`/status` restores compatibility with `_check_api_health()` in the CLI.

**Status logic:**
- All checks pass → `"healthy"`, HTTP 200
- Any check fails → `"degraded"`, HTTP 503 (liveness intact, readiness false)

---

## Structured Logging

Extends `logconfig.py` — no new dependencies (stdlib `contextvars` only).

### New `JSON_FORMAT` fields (optional, default to `""`)

```
"request_id":   "%(request_id)s"
"path":         "%(request_path)s"
"duration_ms":  "%(duration_ms)s"
```

Non-request log lines (startup, background) emit empty strings for these fields — format stays valid.

### New `RequestContextFilter` (in `logconfig.py`)

Reads from a `contextvars.ContextVar` and injects `request_id`, `request_path`, `duration_ms` into every log record emitted during a request.

### New `RequestLoggingMiddleware` (in `server.py`)

Starlette middleware that:
1. Generates a UUID `request_id` per request
2. Sets the context var before calling `await call_next(request)`
3. Calculates `duration_ms` after the response
4. Emits one structured INFO log line per request with method, path, status, duration

Added to `self.versioned_wrapper` (the top-level ASGI app) alongside the existing `CORSMiddleware`.

---

## Kubernetes Manifests (`restapi.tf`)

Add probes to the `container` block inside `kubernetes_deployment.restapi`:

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

No changes to `period_seconds` (K8s defaults: 10s).
No changes to Prometheus scraping annotations (Dapr sidecar stays at port 9090).

---

## Files Changed

| File | Change |
|---|---|
| `src/vibe_core/vibe_core/logconfig.py` | Add `RequestContextFilter`; extend `JSON_FORMAT` with optional request fields |
| `src/vibe_server/vibe_server/server.py` | Add health handler methods to `TerravibesProvider`; add routes to `TerravibesAPI`; add `RequestLoggingMiddleware` |
| `src/vibe_core/vibe_core/terraform/services/restapi.tf` | Add `liveness_probe` and `readiness_probe` blocks |

No new Python dependencies required.
