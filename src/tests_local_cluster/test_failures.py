# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Failure-path tests.

The goal here is to verify that bad input produces a useful, specific error
— not a 500, not a hang, not a zombie run. These are fast because they
never actually execute a workflow; the server rejects them before anything
hits the orchestrator.

A note on status codes: the codes asserted here were probed against a live
cluster in March 2026. They do NOT all match what the source on main says
(e.g. bad-workflow-name is 400 on the cluster but 404 in server.py on main).
We test against reality, not the source tree.
"""

from __future__ import annotations

import uuid

import pytest
from requests.exceptions import HTTPError

from vibe_core.client import FarmvibesAiClient

from .helpers import FIXED_TIME_RANGE, KENTUCKY_POLYGON, raw_get, raw_post

pytestmark = pytest.mark.fast


# ---------------------------------------------------------------------------
# Non-existent resources
# ---------------------------------------------------------------------------


def test_describe_nonexistent_workflow_returns_404(service_url: str) -> None:
    r = raw_get(service_url, "v0/workflows/this-workflow-does-not-exist-xyz")
    assert r.status_code == 404
    body = r.json()
    # The error message should name the thing that wasn't found. "Not found"
    # alone is useless when you're debugging a typo in a 50-char workflow path.
    assert "this-workflow-does-not-exist-xyz" in body.get("message", "")


def test_get_nonexistent_run_returns_404(service_url: str) -> None:
    # Valid UUID format, but doesn't exist. If this returned 400 it would
    # mean the server is rejecting the format, not the lookup.
    fake_id = "00000000-0000-0000-0000-000000000000"
    r = raw_get(service_url, f"v0/runs/{fake_id}")
    assert r.status_code == 404
    assert fake_id in r.json().get("message", "")


def test_cancel_nonexistent_run_returns_404(service_url: str) -> None:
    fake_id = str(uuid.uuid4())
    r = raw_post(service_url, f"v0/runs/{fake_id}/cancel", json={})
    assert r.status_code == 404


def test_resubmit_nonexistent_run_returns_404(service_url: str) -> None:
    fake_id = str(uuid.uuid4())
    r = raw_post(service_url, f"v0/runs/{fake_id}/resubmit", json={})
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Bad workflow submission
# ---------------------------------------------------------------------------


def _minimal_run_body(workflow: str) -> dict:
    """Smallest valid-looking run submission body.

    Built by hand rather than via FarmvibesAiClient so we can break it
    in controlled ways and see the server's raw response.
    """
    import json as _json

    from shapely.geometry import mapping

    return {
        "name": f"failtest-{uuid.uuid4().hex[:8]}",
        "workflow": workflow,
        "parameters": {},
        "user_input": {
            "start_date": FIXED_TIME_RANGE[0].isoformat(),
            "end_date": FIXED_TIME_RANGE[1].isoformat(),
            "geojson": {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": _json.loads(
                            _json.dumps(mapping(KENTUCKY_POLYGON))
                        ),
                        "properties": {},
                    }
                ],
            },
        },
    }


def test_submit_nonexistent_workflow_returns_4xx_with_message(service_url: str) -> None:
    """Submitting a run for a workflow that doesn't exist.

    Status code: live cluster returns 400, main branch source says 404.
    We accept either — what matters is that it's a 4xx (client error, not
    server error) and that the error message names the unknown workflow
    so the user can fix their typo.
    """
    body = _minimal_run_body("definitely-not-a-real-workflow-qwerty")
    r = raw_post(service_url, "v0/runs", json=body)
    assert 400 <= r.status_code < 500, (
        f"Expected 4xx for unknown workflow, got {r.status_code}. "
        f"A 5xx here means the server crashed trying to look up the workflow "
        f"instead of handling the miss gracefully."
    )
    msg = r.json().get("message", r.text)
    assert "definitely-not-a-real-workflow-qwerty" in msg, (
        f"Error message doesn't name the unknown workflow. Got: {msg!r}"
    )


def test_submit_malformed_body_returns_422(service_url: str) -> None:
    """Completely bogus request body.

    422 is FastAPI/Pydantic's validation-failure code. If this returns 500,
    the validation layer was bypassed somehow and the server choked on
    garbage downstream.
    """
    r = raw_post(service_url, "v0/runs", json={"garbage": True})
    assert r.status_code == 422
    # FastAPI's validation errors come back as {"detail": [{...}, ...]}.
    # We don't care about the exact text (it's Pydantic-version-dependent)
    # but it should at least tell us required fields are missing.
    detail = r.json().get("detail", [])
    assert isinstance(detail, list) and len(detail) > 0


def test_submit_missing_user_input_returns_4xx(service_url: str) -> None:
    """Workflow exists, but user_input is missing. Should be rejected
    before anything hits the orchestrator."""
    r = raw_post(
        service_url,
        "v0/runs",
        json={"name": "bad", "workflow": "helloworld", "parameters": {}},
    )
    assert 400 <= r.status_code < 500


# ---------------------------------------------------------------------------
# Client-level failure surface
# ---------------------------------------------------------------------------
# The raw HTTP tests above verify the server. These verify that the Python
# client surfaces server errors as exceptions rather than swallowing them.


def test_client_raises_httperror_for_bad_workflow(vibe_client: FarmvibesAiClient) -> None:
    """When the server 4xx's a submission, the client must raise.

    If this test fails because no exception was raised, it means the client
    returned a fake VibeWorkflowRun for a run that was never actually
    created — the worst possible failure mode because the caller thinks
    they have a handle to something real.
    """
    with pytest.raises(HTTPError) as exc_info:
        vibe_client.run(
            "client-failure-test-workflow-does-not-exist",
            "failtest",
            geometry=KENTUCKY_POLYGON,
            time_range=FIXED_TIME_RANGE,
        )
    # The exception message should carry the server's error detail, not
    # just "400 Bad Request".
    assert "client-failure-test-workflow-does-not-exist" in str(exc_info.value)


def test_client_raises_for_nonexistent_run_lookup(vibe_client: FarmvibesAiClient) -> None:
    with pytest.raises(HTTPError):
        vibe_client.describe_run("00000000-0000-0000-0000-000000000000")


# ---------------------------------------------------------------------------
# No zombie runs
# ---------------------------------------------------------------------------


def test_failed_submission_leaves_no_zombie_run(
    service_url: str, vibe_client: FarmvibesAiClient
) -> None:
    """A rejected submission should not create a run record.

    This catches the case where the server writes to the state store
    *before* validating the workflow, leaving a dangling run that will
    never execute and never be cleaned up.
    """
    before = set(vibe_client.list_runs())
    body = _minimal_run_body("zombie-check-nonexistent-workflow")
    r = raw_post(service_url, "v0/runs", json=body)
    assert 400 <= r.status_code < 500, "precondition: submission should be rejected"

    after = set(vibe_client.list_runs())
    new_runs = after - before
    assert not new_runs, (
        f"Rejected submission created {len(new_runs)} zombie run(s): {new_runs}. "
        f"The server is writing to the state store before validating input."
    )
