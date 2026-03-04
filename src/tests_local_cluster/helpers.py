# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Reusable helpers for integration tests against a live FarmVibes.AI cluster.

These are deliberately independent of any specific test. Projects that consume
the FarmVibes.AI API can import from here to build their own test suites.

Key design choices:
  - URL resolution is explicit: env var > service_url file > fallback.
    No magic. No implicit dependency on `get_default_vibe_client`, because
    that function silently falls back when the cluster is down, which is
    exactly the wrong behaviour for a test suite.
  - Timeouts are bounded everywhere. Every poll loop has a hard cap.
  - Runs are tracked and cleaned up by default, so repeated test runs don't
    accumulate junk in the cluster.
"""

from __future__ import annotations

import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, TypeVar

import httpx
from shapely.geometry import Polygon

from vibe_core.client import FarmvibesAiClient, VibeWorkflowRun
from vibe_core.datamodel import RunStatus

T = TypeVar("T")

# ---------------------------------------------------------------------------
# URL resolution
# ---------------------------------------------------------------------------

ENV_VAR = "FARMVIBES_AI_SERVICE_URL"
_XDG = os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config"))
_SERVICE_URL_FILE = os.path.join(_XDG, "farmvibes-ai", "service_url")
_FALLBACK = "http://127.0.0.1:31108/"


def resolve_service_url() -> str:
    """Resolve the FarmVibes.AI REST API base URL.

    Precedence: ``FARMVIBES_AI_SERVICE_URL`` env var, then
    ``~/.config/farmvibes-ai/service_url``, then the hardcoded fallback.

    The returned URL always ends with ``/`` so it can be used directly
    with ``urljoin`` or simple string concatenation.
    """
    url = os.environ.get(ENV_VAR, "").strip()
    if not url:
        try:
            with open(_SERVICE_URL_FILE) as f:
                url = f.read().strip()
        except FileNotFoundError:
            url = _FALLBACK
    if not url.endswith("/"):
        url += "/"
    return url


def probe_reachable(base_url: str, timeout_s: float = 5.0) -> Tuple[bool, str]:
    """Check if the REST API is reachable at all.

    Returns ``(reachable, detail)``. Hits ``GET /v0/`` which every
    FarmVibes.AI server image exposes. Does not use the FarmvibesAiClient
    because that wraps errors in ways that hide the transport failure.
    """
    try:
        r = httpx.get(f"{base_url}v0/", timeout=timeout_s)
    except httpx.HTTPError as e:
        return False, f"{type(e).__name__}: {e}"
    if r.status_code != 200:
        return False, f"HTTP {r.status_code}: {r.text[:200]}"
    return True, "ok"


# ---------------------------------------------------------------------------
# Polling
# ---------------------------------------------------------------------------


class PollTimeout(RuntimeError):
    """Raised when :func:`poll_until` exceeds its timeout."""


def poll_until(
    predicate: Callable[[], T],
    timeout_s: float,
    interval_s: float = 2.0,
    what: str = "condition",
) -> T:
    """Poll ``predicate`` until it returns a truthy value or timeout expires.

    Returns the truthy value. Raises :class:`PollTimeout` on timeout.

    The predicate is called at least once regardless of ``timeout_s``.
    ``interval_s`` is the sleep between calls, not the total period, so
    the actual wall time is roughly ``timeout_s + (one predicate call)``.
    """
    deadline = time.monotonic() + timeout_s
    last: Any = None
    while True:
        last = predicate()
        if last:
            return last
        if time.monotonic() >= deadline:
            raise PollTimeout(
                f"Timed out after {timeout_s:.0f}s waiting for {what} (last={last!r})"
            )
        time.sleep(interval_s)


# ---------------------------------------------------------------------------
# Fixed test inputs
# ---------------------------------------------------------------------------
# The polygon is a small patch in western Kentucky, USA. Same coordinates the
# existing helloworld fixture uses, so results are comparable. Kept small so
# workflows that actually fetch imagery don't pull gigabytes.

KENTUCKY_POLYGON = Polygon(
    [
        (-88.062073, 37.081398),
        (-88.026349, 37.085464),
        (-88.012445, 37.069230),
        (-88.035932, 37.048441),
        (-88.068120, 37.058834),
        (-88.062073, 37.081398),
    ]
)

FIXED_TIME_RANGE = (
    datetime(2021, 2, 1, tzinfo=timezone.utc),
    datetime(2021, 2, 11, tzinfo=timezone.utc),
)


# ---------------------------------------------------------------------------
# Run submission + cleanup
# ---------------------------------------------------------------------------


@dataclass
class RunRecord:
    """Bookkeeping for a submitted run so it can be cleaned up later."""

    run: VibeWorkflowRun
    workflow: str
    submitted_at: float = field(default_factory=time.monotonic)

    @property
    def elapsed_s(self) -> float:
        return time.monotonic() - self.submitted_at


class WorkflowSubmitter:
    """Thin wrapper around :class:`FarmvibesAiClient` that tracks submitted
    runs so tests don't leave garbage behind.

    Use :meth:`submit_and_wait` for the common case, or :meth:`submit` +
    :meth:`wait` when you need to do something between submission and
    completion (e.g. cancel).

    The :meth:`cleanup` method is idempotent and safe to call even if the
    cluster died mid-test.
    """

    def __init__(self, client: FarmvibesAiClient):
        self.client = client
        self._records: List[RunRecord] = []

    def submit(
        self,
        workflow: str,
        *,
        name_prefix: str = "itest",
        geometry: Optional[Polygon] = None,
        time_range: Optional[Tuple[datetime, datetime]] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> RunRecord:
        """Submit a workflow run. Does not wait. Records for cleanup."""
        # Unique suffix so concurrent test runs (e.g. two pytest workers,
        # or two model lanes) don't collide in run-name lookups.
        name = f"{name_prefix}-{workflow.replace('/', '-')}-{uuid.uuid4().hex[:8]}"
        run = self.client.run(
            workflow,
            name,
            geometry=geometry or KENTUCKY_POLYGON,
            time_range=time_range or FIXED_TIME_RANGE,
            parameters=parameters,
        )
        rec = RunRecord(run=run, workflow=workflow)
        self._records.append(rec)
        return rec

    def wait(self, rec: RunRecord, timeout_s: float) -> RunStatus:
        """Wait for a run to reach a terminal state.

        Unlike ``VibeWorkflowRun.block_until_complete``, this treats
        ``failed`` as a valid terminal state rather than letting it pass
        silently — callers get the status back and assert on it
        themselves. This means a workflow that fails fast doesn't sit
        there polling until timeout.
        """

        def _terminal() -> Optional[RunStatus]:
            s = rec.run.status
            return s if RunStatus.finished(s) else None

        status = poll_until(
            _terminal, timeout_s, interval_s=3.0, what=f"workflow '{rec.workflow}' to finish"
        )
        return status

    def submit_and_wait(
        self, workflow: str, *, timeout_s: float, **kwargs: Any
    ) -> Tuple[RunRecord, RunStatus]:
        rec = self.submit(workflow, **kwargs)
        status = self.wait(rec, timeout_s)
        return rec, status

    def cleanup(self) -> None:
        """Best-effort cleanup of every submitted run.

        Runs that are still in-flight are cancelled first, because the
        server rejects DELETE for unfinished runs (400). Exceptions are
        swallowed because cleanup runs in teardown and we'd rather see
        the real test failure than a teardown cascade.
        """
        for rec in self._records:
            try:
                s = rec.run.status
                if not RunStatus.finished(s) and s != RunStatus.deleted:
                    rec.run.cancel()
                    poll_until(
                        lambda: RunStatus.finished(rec.run.status),
                        timeout_s=30,
                        interval_s=2.0,
                        what="cancel to land",
                    )
            except Exception:
                pass
            try:
                if rec.run.status != RunStatus.deleted:
                    rec.run.delete()
            except Exception:
                pass
        self._records.clear()


@contextmanager
def tracked_runs(client: FarmvibesAiClient) -> Iterator[WorkflowSubmitter]:
    """Context manager flavour of :class:`WorkflowSubmitter` for ad-hoc use
    outside of pytest fixtures. Guarantees cleanup even on exception.
    """
    sub = WorkflowSubmitter(client)
    try:
        yield sub
    finally:
        sub.cleanup()


# ---------------------------------------------------------------------------
# Raw HTTP helpers — for tests that need to see actual status codes
# ---------------------------------------------------------------------------


def raw_get(base_url: str, path: str, timeout_s: float = 10.0) -> httpx.Response:
    """GET a path under the base URL. Returns the raw response.

    ``path`` should not start with ``/`` — it's joined to ``base_url``
    which already ends with ``/``.
    """
    return httpx.get(f"{base_url}{path.lstrip('/')}", timeout=timeout_s)


def raw_post(
    base_url: str, path: str, json: Dict[str, Any], timeout_s: float = 10.0
) -> httpx.Response:
    """POST JSON to a path under the base URL. Returns the raw response.

    Same path-joining rules as :func:`raw_get`. Used by failure-path tests
    that need to submit deliberately-broken bodies and inspect the exact
    status code the server returns.
    """
    return httpx.post(f"{base_url}{path.lstrip('/')}", json=json, timeout=timeout_s)
