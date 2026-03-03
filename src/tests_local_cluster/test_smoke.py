# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""End-to-end smoke test.

One test. One workflow. One job: prove the whole pipeline works.

This submits helloworld — the only workflow guaranteed to have zero external
dependencies (no Planetary Computer key, no ADMAG creds, no imagery
download) — and waits for it to produce a raster. If this fails, nothing
else matters.
"""

from __future__ import annotations

import pytest

from vibe_core.datamodel import RunStatus

from .helpers import WorkflowSubmitter

pytestmark = pytest.mark.slow

# Probed at ~40s on a cold cluster, ~15s warm. 10 minutes is the contract
# from the task spec — if helloworld takes longer than that, something is
# pathologically wrong (stuck orchestrator, dead worker, rabbit backed up).
SMOKE_TIMEOUT_S = 600


def test_helloworld_end_to_end(submitter: WorkflowSubmitter) -> None:
    """The smoke test.

    What "pass" means here, in order:
      1. The REST API accepted the submission (we get a run ID back).
      2. The orchestrator picked it up and dispatched to a worker.
      3. The worker executed the op and published a result.
      4. The orchestrator collected the result and marked the run done.
      5. The result is retrievable and has the shape we expect.

    Each of those is a different service. This test doesn't tell you
    *which* one broke, but it tells you *that* something broke — which
    is exactly what a smoke test is for.
    """
    rec, status = submitter.submit_and_wait("helloworld", timeout_s=SMOKE_TIMEOUT_S)

    # --- Status assertion --------------------------------------------------
    # RunStatus.failed is also terminal, so `wait` returns for it too.
    # We want to surface the failure reason in the assertion message,
    # because "AssertionError: failed != done" tells you nothing.
    if status == RunStatus.failed:
        tasks = rec.run.task_details
        reason = rec.run.reason
        pytest.fail(
            f"helloworld failed after {rec.elapsed_s:.1f}s.\n"
            f"  reason: {reason}\n"
            f"  tasks:  {tasks}\n"
            f"This means the pipeline is broken somewhere between the "
            f"orchestrator and the worker. Check worker logs."
        )
    assert status == RunStatus.done, f"unexpected terminal status: {status}"

    # --- Output shape assertion --------------------------------------------
    out = rec.run.output
    assert out is not None, (
        "Run status is 'done' but output is None. This is an orchestrator "
        "bug: it marked the run complete without attaching results."
    )
    assert "raster" in out, f"expected 'raster' in output keys, got: {list(out.keys())}"
    rasters = out["raster"]
    assert isinstance(rasters, list) and len(rasters) > 0, (
        f"'raster' output is empty: {rasters!r}"
    )

    # --- Asset assertion ---------------------------------------------------
    # The raster should have at least one asset with a path. We don't
    # check the pixels here (that's what the legacy test_cluster_integration
    # test with expected.tif does) — this is a smoke test, not a
    # correctness test.
    first = rasters[0]
    assert hasattr(first, "assets"), f"raster has no assets attribute: {type(first)}"
    assert len(first.assets) > 0, "raster has no assets"


def test_helloworld_run_is_listable(submitter: WorkflowSubmitter) -> None:
    """A submitted run should appear in GET /v0/runs.

    This is separate from the main smoke test because it exercises a
    different path: the state store query layer. The main smoke test
    could pass even if list_runs was broken, as long as describe_run
    worked.
    """
    rec = submitter.submit("helloworld", name_prefix="listable")
    # Don't wait for completion — just check it shows up.
    all_runs = submitter.client.list_runs()
    assert rec.run.id in all_runs, (
        f"Submitted run {rec.run.id} not found in list_runs(). "
        f"Either the state store write is lagging behind the HTTP 201 "
        f"response, or list_runs has a filtering bug."
    )
    # Now let it finish so cleanup works cleanly.
    submitter.wait(rec, timeout_s=SMOKE_TIMEOUT_S)


def test_helloworld_run_is_deletable(submitter: WorkflowSubmitter) -> None:
    """Full lifecycle: submit → complete → delete.

    The submitter fixture already deletes on teardown, but that's best-
    effort and swallows exceptions. This test asserts that delete actually
    works, because if it doesn't, the cluster's state store fills up with
    test runs over time.
    """
    from .helpers import poll_until

    rec, status = submitter.submit_and_wait("helloworld", timeout_s=SMOKE_TIMEOUT_S)
    assert status == RunStatus.done

    rec.run.delete()
    final = poll_until(
        lambda: rec.run.status if rec.run.status == RunStatus.deleted else None,
        timeout_s=60,
        what="run deletion",
    )
    assert final == RunStatus.deleted
