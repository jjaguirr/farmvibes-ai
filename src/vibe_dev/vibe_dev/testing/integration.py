# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Helpers for REST-level integration tests against a live FarmVibes.AI cluster.

Provides:
- :func:`resolve_cluster_url` -- locate the cluster base URL
- :class:`ClusterClient`      -- thin HTTP client wrapping ``requests``
- :class:`WorkflowPoller`     -- poll a run until terminal state or timeout
- :class:`WorkflowSpec`       -- dataclass for parametrized workflow tests
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import yaml
from shapely.geometry import Polygon, mapping

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Terminal run statuses (mirrors vibe_core.datamodel.RunStatus)
# ---------------------------------------------------------------------------
_TERMINAL_STATUSES = frozenset({"done", "failed", "cancelled", "deleted"})

# ---------------------------------------------------------------------------
# Config-file locations (same convention as vibe_core.client)
# ---------------------------------------------------------------------------
_XDG_CONFIG_HOME = os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config"))
_REMOTE_URL_PATH = os.path.join(_XDG_CONFIG_HOME, "farmvibes-ai", "remote_service_url")
_LOCAL_URL_PATH = os.path.join(_XDG_CONFIG_HOME, "farmvibes-ai", "service_url")
_FALLBACK_URL = "http://127.0.0.1:31108"


# ---- resolve_cluster_url --------------------------------------------------


def resolve_cluster_url() -> str:
    """Resolve the cluster base URL using the standard FarmVibes.AI lookup chain.

    Resolution order:

    1. ``FARMVIBES_AI_BASE_URL`` environment variable
    2. ``~/.config/farmvibes-ai/remote_service_url`` file
    3. ``~/.config/farmvibes-ai/service_url`` file
    4. ``http://127.0.0.1:31108`` (fallback)

    Returns:
        The resolved base URL (trailing slash stripped).
    """
    env_url = os.environ.get("FARMVIBES_AI_BASE_URL")
    if env_url:
        return env_url.rstrip("/")

    for path in (_REMOTE_URL_PATH, _LOCAL_URL_PATH):
        try:
            url = Path(path).read_text().strip()
            if url:
                return url.rstrip("/")
        except (FileNotFoundError, OSError):
            continue

    return _FALLBACK_URL


# ---- ClusterClient --------------------------------------------------------


class ClusterClient:
    """Thin HTTP client that talks to the FarmVibes.AI REST API.

    All convenience methods return parsed JSON (``dict`` / ``list``) or raw
    :class:`requests.Response` objects, as documented per method.

    Args:
        base_url: Cluster base URL.  Resolved via :func:`resolve_cluster_url`
            when *None*.
        timeout: Default request timeout in seconds.
    """

    def __init__(self, base_url: Optional[str] = None, timeout: float = 30.0) -> None:
        self.base_url: str = (base_url or resolve_cluster_url()).rstrip("/")
        self.timeout: float = timeout
        self._session: requests.Session = requests.Session()

    # -- raw HTTP helpers ---------------------------------------------------

    def get(self, path: str, **kwargs: Any) -> requests.Response:
        """Send a GET request to *path* (relative to ``base_url``)."""
        kwargs.setdefault("timeout", self.timeout)
        return self._session.get(f"{self.base_url}{path}", **kwargs)

    def post(self, path: str, **kwargs: Any) -> requests.Response:
        """Send a POST request to *path* (relative to ``base_url``)."""
        kwargs.setdefault("timeout", self.timeout)
        return self._session.post(f"{self.base_url}{path}", **kwargs)

    def delete(self, path: str, **kwargs: Any) -> requests.Response:
        """Send a DELETE request to *path* (relative to ``base_url``)."""
        kwargs.setdefault("timeout", self.timeout)
        return self._session.delete(f"{self.base_url}{path}", **kwargs)

    # -- convenience endpoints ----------------------------------------------

    def root(self) -> dict:
        """``GET /v0/`` -- root message."""
        resp = self.get("/v0/")
        resp.raise_for_status()
        return resp.json()

    def system_metrics(self) -> dict:
        """``GET /v0/system-metrics`` -- CPU, memory, disk stats."""
        resp = self.get("/v0/system-metrics")
        resp.raise_for_status()
        return resp.json()

    def liveness(self) -> requests.Response:
        """``GET /healthz/live`` -- Kubernetes liveness probe."""
        return self.get("/healthz/live")

    def readiness(self) -> requests.Response:
        """``GET /healthz/ready`` -- Kubernetes readiness probe."""
        return self.get("/healthz/ready")

    def health(self) -> requests.Response:
        """``GET /v0/health`` -- detailed health check."""
        return self.get("/v0/health")

    # -- workflow endpoints -------------------------------------------------

    def list_workflows(self) -> List[str]:
        """``GET /v0/workflows`` -- list available workflow names."""
        resp = self.get("/v0/workflows")
        resp.raise_for_status()
        return resp.json()

    def describe_workflow(self, name: str) -> dict:
        """``GET /v0/workflows/{name}?return_format=description``."""
        resp = self.get(f"/v0/workflows/{name}", params={"return_format": "description"})
        resp.raise_for_status()
        return resp.json()

    def get_workflow_yaml(self, name: str) -> str:
        """``GET /v0/workflows/{name}?return_format=yaml`` -- returns YAML string."""
        resp = self.get(f"/v0/workflows/{name}", params={"return_format": "yaml"})
        resp.raise_for_status()
        return yaml.dump(resp.json(), default_flow_style=False, sort_keys=False)

    # -- run endpoints ------------------------------------------------------

    def submit_run(
        self,
        workflow: str,
        name: str,
        geometry: Polygon,
        time_range: Tuple[datetime, datetime],
        parameters: Optional[Dict[str, Any]] = None,
    ) -> dict:
        """``POST /v0/runs`` -- submit a new workflow run.

        Args:
            workflow: Workflow name (e.g. ``"helloworld"``).
            name: Human-readable run name.
            geometry: A :class:`shapely.geometry.Polygon` for the AOI.
            time_range: ``(start, end)`` datetime pair.
            parameters: Optional workflow parameters.

        Returns:
            Response JSON (contains ``id`` and ``location``).
        """
        start_date, end_date = time_range
        payload: Dict[str, Any] = {
            "name": name,
            "workflow": workflow,
            "parameters": parameters,
            "user_input": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "geojson": {
                    "type": "FeatureCollection",
                    "features": [
                        {
                            "type": "Feature",
                            "geometry": mapping(geometry),
                        }
                    ],
                },
            },
        }
        resp = self.post("/v0/runs", json=payload)
        resp.raise_for_status()
        return resp.json()

    def get_run(self, run_id: str) -> dict:
        """``GET /v0/runs/{run_id}`` -- full run details."""
        resp = self.get(f"/v0/runs/{run_id}")
        resp.raise_for_status()
        return resp.json()

    def list_runs(
        self,
        ids: Optional[List[str]] = None,
        fields: Optional[List[str]] = None,
    ) -> List[dict]:
        """``GET /v0/runs`` -- list runs, optionally filtering by *ids* and *fields*.

        Args:
            ids: Run UUIDs to retrieve.  When *None*, all runs are returned.
            fields: Summary fields to include (e.g. ``["id", "details.status"]``).

        Returns:
            A list of run summaries or run-id strings, depending on *fields*.
        """
        params: Dict[str, Any] = {}
        if ids is not None:
            params["ids"] = ids
        if fields is not None:
            params["fields"] = fields
        resp = self.get("/v0/runs", params=params)
        resp.raise_for_status()
        return resp.json()

    def cancel_run(self, run_id: str) -> requests.Response:
        """``POST /v0/runs/{run_id}/cancel`` -- request cancellation."""
        return self.post(f"/v0/runs/{run_id}/cancel")

    def delete_run(self, run_id: str) -> requests.Response:
        """``DELETE /v0/runs/{run_id}`` -- delete run data."""
        return self.delete(f"/v0/runs/{run_id}")


# ---- WorkflowPoller -------------------------------------------------------


class WorkflowPoller:
    """Poll a workflow run until it reaches a terminal state or times out.

    Terminal statuses: ``done``, ``failed``, ``cancelled``, ``deleted``.

    Args:
        client: A :class:`ClusterClient` instance.
        timeout_s: Default polling timeout in seconds.
        interval_s: Seconds between polls.
    """

    def __init__(
        self,
        client: ClusterClient,
        timeout_s: float = 120.0,
        interval_s: float = 5.0,
    ) -> None:
        self.client = client
        self.timeout_s = timeout_s
        self.interval_s = interval_s

    def poll(self, run_id: str, timeout_s: Optional[float] = None) -> dict:
        """Block until *run_id* reaches a terminal status.

        Queries ``list_runs`` with ``fields=["id", "details.status"]`` to
        minimise payload size, then fetches the full run on completion.

        Args:
            run_id: UUID of the run to watch.
            timeout_s: Override the default timeout for this call.

        Returns:
            Full run dict from ``get_run``.

        Raises:
            TimeoutError: If the run does not finish within the timeout.
        """
        effective_timeout = timeout_s if timeout_s is not None else self.timeout_s
        deadline = time.monotonic() + effective_timeout
        last_status = "unknown"

        while time.monotonic() < deadline:
            try:
                summaries = self.client.list_runs(
                    ids=[run_id],
                    fields=["id", "details.status"],
                )
                if summaries:
                    last_status = summaries[0].get("details.status", "unknown")
                    if last_status in _TERMINAL_STATUSES:
                        return self.client.get_run(run_id)
            except requests.RequestException as exc:
                logger.warning("Poll request failed for run %s: %s", run_id, exc)

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(self.interval_s, remaining))

        raise TimeoutError(
            f"Run {run_id} did not reach terminal state within {effective_timeout}s "
            f"(last status: {last_status})"
        )


# ---- WorkflowSpec ---------------------------------------------------------


@dataclass
class WorkflowSpec:
    """Specification for a parametrized workflow integration test.

    Attributes:
        name: Workflow name (e.g. ``"helloworld"``).
        geometry: Area-of-interest polygon.
        time_range: ``(start_datetime, end_datetime)`` pair.
        expected_sinks: Sink names that the completed run must contain.
        timeout_s: Per-run polling timeout in seconds.
        parameters: Optional workflow parameters dict.
        run_name: Human-readable run name.  Auto-generated from *name* when
            left empty.
    """

    name: str
    geometry: Polygon
    time_range: Tuple[datetime, datetime]
    expected_sinks: List[str]
    timeout_s: float = 120.0
    parameters: Optional[Dict[str, Any]] = None
    run_name: str = ""

    def __post_init__(self) -> None:
        if not self.run_name:
            self.run_name = f"integ-{self.name}-{uuid.uuid4().hex[:8]}"
