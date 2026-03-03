# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Service health tests.

These hit every endpoint the REST API is documented to expose and check
that responses have the expected shape. They do not submit workflows, so
they're cheap — the whole file should finish in a few seconds against a
healthy cluster.

The /v0/health tests are marked ``newapi`` and will skip on older cluster
images. This is intentional: a cluster running pre-health-endpoint images
is still a valid deployment target, and the test suite shouldn't flunk it
for code that simply isn't in that image.
"""

from __future__ import annotations

import pytest

from .helpers import raw_get

pytestmark = pytest.mark.fast


# ---------------------------------------------------------------------------
# Root + liveness
# ---------------------------------------------------------------------------


def test_root_responds(service_url: str) -> None:
    """GET /v0/ — the one endpoint every image has. If this fails, nothing
    else in the suite is going to work."""
    r = raw_get(service_url, "v0/")
    assert r.status_code == 200, f"root returned {r.status_code}: {r.text}"
    body = r.json()
    assert "message" in body
    assert "running" in body["message"].lower()


def test_openapi_schema_available(service_url: str) -> None:
    """The OpenAPI schema is how we know what the server *actually* exposes,
    as opposed to what the source tree says it should. Failing here means
    FastAPI itself is broken, which is catastrophic."""
    r = raw_get(service_url, "v0/openapi.json")
    assert r.status_code == 200
    schema = r.json()
    assert "paths" in schema
    # These paths are the contract. If any disappears, clients break.
    for required in ("/runs", "/workflows", "/system-metrics"):
        assert required in schema["paths"], (
            f"Required path {required} missing from OpenAPI schema. "
            f"Available: {sorted(schema['paths'].keys())}"
        )


# ---------------------------------------------------------------------------
# Workflow catalog
# ---------------------------------------------------------------------------


def test_workflow_list_nonempty(workflow_catalog: list[str]) -> None:
    # A fresh cluster ships ~90 workflows. Zero means the workflow registry
    # failed to load at startup — usually a mounted-volume problem.
    assert len(workflow_catalog) > 0, "workflow catalog is empty"


def test_helloworld_in_catalog(workflow_catalog: list[str]) -> None:
    # helloworld is the canary. It has no external dependencies, no secrets,
    # no imagery downloads. If it's missing, smoke tests can't run.
    assert "helloworld" in workflow_catalog


def test_workflow_describe(service_url: str) -> None:
    """Describe returns the workflow's input/output contract."""
    r = raw_get(service_url, "v0/workflows/helloworld?return_format=description")
    assert r.status_code == 200
    desc = r.json()
    assert desc["name"] == "helloworld"
    assert "inputs" in desc and "user_input" in desc["inputs"]
    assert "outputs" in desc and "raster" in desc["outputs"]


def test_workflow_yaml_format(service_url: str) -> None:
    """return_format=yaml gives the raw workflow dict (sources/sinks/tasks),
    as opposed to return_format=description which gives the processed
    inputs/outputs view. Despite the name, it's served as JSON — "yaml"
    refers to the source format on disk, not the wire format."""
    r = raw_get(service_url, "v0/workflows/helloworld?return_format=yaml")
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body, dict)
    # These are the YAML-level keys (workflow DAG structure). If they
    # vanish, the server is returning the description format regardless
    # of the query param.
    assert "sources" in body
    assert "sinks" in body
    assert body.get("name") == "helloworld"


# ---------------------------------------------------------------------------
# System metrics
# ---------------------------------------------------------------------------


def test_system_metrics_shape(service_url: str) -> None:
    r = raw_get(service_url, "v0/system-metrics")
    assert r.status_code == 200
    m = r.json()
    # These fields are what the CLI status command consumes. If the shape
    # changes without updating the CLI, status will crash.
    for field in ("cpu_usage", "free_mem", "total_mem", "disk_free"):
        assert field in m, f"metrics missing '{field}': {m}"
        assert isinstance(m[field], (int, float)), f"{field} is not numeric: {m[field]!r}"
    assert "load_avg" in m
    assert isinstance(m["load_avg"], list) and len(m["load_avg"]) == 3


def test_system_metrics_sane(service_url: str) -> None:
    """Sanity-check that metrics aren't garbage. Zero total memory or
    negative free disk means the metrics collector inside the container
    is reading the wrong cgroup path."""
    m = raw_get(service_url, "v0/system-metrics").json()
    assert m["total_mem"] > 0
    assert m["disk_free"] >= 0
    assert 0 <= m["cpu_usage"] <= 100 * 64  # per-core percentages can exceed 100


# ---------------------------------------------------------------------------
# Run listing
# ---------------------------------------------------------------------------


def test_list_runs_returns_list(service_url: str) -> None:
    """GET /v0/runs with no filters. Should always return a list, even if
    empty on a fresh cluster."""
    r = raw_get(service_url, "v0/runs")
    assert r.status_code == 200
    runs = r.json()
    assert isinstance(runs, list)


# ---------------------------------------------------------------------------
# Detailed health (newapi — skips on old images)
# ---------------------------------------------------------------------------

# The dependency names here are what the orchestrator service-invokes via
# Dapr. If this list changes in terraform without updating the health
# checker, the test catches the drift.
_EXPECTED_HEALTH_DEPS = {
    "dapr",
    "statestore",
    "terravibes-orchestrator",
    "terravibes-cache",
    "terravibes-data-ops",
}


@pytest.mark.newapi
def test_health_endpoint_reports_all_deps(service_url: str) -> None:
    r = raw_get(service_url, "v0/health", timeout_s=20.0)
    if r.status_code == 404:
        pytest.skip("/v0/health not in this cluster image (pre-dates health endpoint)")
    assert r.status_code in (200, 503), f"unexpected health status code {r.status_code}"
    body = r.json()
    assert body["status"] in ("healthy", "degraded", "unhealthy")
    reported = {d["name"] for d in body.get("dependencies", [])}
    missing = _EXPECTED_HEALTH_DEPS - reported
    assert not missing, (
        f"Health endpoint didn't report on: {missing}. "
        f"Either a service was removed from the cluster or the health "
        f"checker config drifted from terraform. Reported: {reported}"
    )


@pytest.mark.newapi
def test_health_endpoint_all_healthy(service_url: str) -> None:
    """Assert that every dependency is actually healthy.

    This is separate from the structural test above so that a degraded-but-
    reporting cluster produces one failure (this test) rather than zero
    (if we only checked structure) or two (if we conflated them).
    """
    r = raw_get(service_url, "v0/health", timeout_s=20.0)
    if r.status_code == 404:
        pytest.skip("/v0/health not in this cluster image")
    body = r.json()
    unhealthy = [
        d for d in body.get("dependencies", []) if d.get("status") != "healthy"
    ]
    assert not unhealthy, (
        f"Unhealthy dependencies: "
        f"{[(d['name'], d.get('status'), d.get('detail')) for d in unhealthy]}"
    )
    assert body["status"] == "healthy"


@pytest.mark.newapi
def test_liveness_probe(service_url: str) -> None:
    """K8s liveness probe endpoint. Outside the /v0/ version prefix."""
    r = raw_get(service_url, "healthz/live")
    if r.status_code == 404:
        pytest.skip("/healthz/live not in this cluster image")
    assert r.status_code == 200


@pytest.mark.newapi
def test_readiness_probe(service_url: str) -> None:
    r = raw_get(service_url, "healthz/ready")
    if r.status_code == 404:
        pytest.skip("/healthz/ready not in this cluster image")
    # 503 here means Dapr or state store is down — that's a real failure
    # that should surface.
    assert r.status_code == 200, f"readiness probe returned {r.status_code}: {r.text}"
