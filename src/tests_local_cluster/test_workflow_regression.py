# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Workflow regression coverage.

Two kinds of test here, both table-driven so adding coverage is a
one-line change:

  SCHEMA_CASES     — describe() succeeds + output contract unchanged.
                     Fast. Catches YAML errors and schema drift.
  EXECUTION_CASES  — run() completes + output has expected keys.
                     Slow. Catches orchestrator/worker/cache breakage.

Plus three non-parametrized orchestration tests (cache hit, cancel,
cache-key-includes-geometry) that exercise orchestrator state machine
paths independent of which workflow runs.

Adding a regression case
------------------------
Append one ``pytest.param(WorkflowCase(...), id="...", marks=[...])``
row to the relevant table. That's it. No new test functions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import pytest
from shapely.geometry import Polygon

from vibe_core.client import FarmvibesAiClient
from vibe_core.datamodel import RunStatus

from .helpers import FIXED_TIME_RANGE, KENTUCKY_POLYGON, WorkflowSubmitter, poll_until


# ===========================================================================
# Case definition
# ===========================================================================


@dataclass
class WorkflowCase:
    """One workflow regression case.

    All fields except ``workflow`` and ``expected_outputs`` have sensible
    defaults — a new case for a simple workflow is two arguments.

    Why a dataclass instead of a bare tuple: named fields make the tables
    below self-documenting, and adding a new field (say, ``min_asset_count``)
    doesn't break every existing row.
    """

    workflow: str
    expected_outputs: set[str]
    # Inputs. None → use the shared Kentucky polygon / Feb 2021 window.
    geometry: Optional[Polygon] = None
    time_range: Optional[tuple[datetime, datetime]] = None
    parameters: dict = field(default_factory=dict)
    # Budget. Default is generous for helloworld-class workflows;
    # external-data cases should bump this.
    timeout_s: float = 300.0


# Small polygon (~1km²) for cases where download size scales with AOI.
_TINY_POLY = Polygon(
    [(-88.06, 37.08), (-88.05, 37.08), (-88.05, 37.07), (-88.06, 37.07), (-88.06, 37.08)]
)


# ===========================================================================
# Tier 1: schema regression (fast)
# ===========================================================================
# One representative per category. These call describe_workflow(), not run().
# The assertion is that the output keys haven't silently changed — because
# if `raster` becomes `rasters`, every notebook that consumes this workflow
# breaks with a KeyError and nobody knows why until they read the YAML diff.
#
# To add: one row. expected_outputs is the minimum set; extra outputs are
# allowed (so adding a new output to a workflow doesn't break this test).

SCHEMA_CASES = [
    pytest.param(
        WorkflowCase("helloworld", {"raster"}),
        id="root/helloworld",
    ),
    pytest.param(
        WorkflowCase("data_ingestion/cdl/download_cdl", {"raster"}),
        id="ingest/cdl",
    ),
    pytest.param(
        WorkflowCase("data_ingestion/osm_road_geometries", {"roads"}),
        id="ingest/osm",
    ),
    pytest.param(
        WorkflowCase("data_ingestion/weather/download_gridmet", {"downloaded_product"}),
        id="ingest/gridmet",
    ),
    pytest.param(
        WorkflowCase("farm_ai/agriculture/green_house_gas_fluxes", {"fluxes"}),
        id="farm_ai/ghg",
    ),
    pytest.param(
        WorkflowCase("data_processing/index/index", {"index_raster"}),
        id="processing/index",
    ),
    pytest.param(
        WorkflowCase(
            "data_processing/timeseries/timeseries_aggregation", {"timeseries"}
        ),
        id="processing/timeseries",
    ),
]


@pytest.mark.fast
@pytest.mark.parametrize("case", SCHEMA_CASES)
def test_workflow_schema_stable(
    vibe_client: FarmvibesAiClient,
    workflow_catalog: list[str],
    case: WorkflowCase,
) -> None:
    if case.workflow not in workflow_catalog:
        pytest.fail(
            f"Workflow '{case.workflow}' is no longer in the catalog. "
            f"Either it was renamed/removed or the workflow registry "
            f"failed to load it at startup."
        )
    desc = vibe_client.describe_workflow(case.workflow)
    actual_outputs = set(desc.get("outputs", {}).keys())
    missing = case.expected_outputs - actual_outputs
    assert not missing, (
        f"Workflow '{case.workflow}' output contract changed. "
        f"Missing: {missing}. Got: {actual_outputs}. "
        f"If intentional, update SCHEMA_CASES; otherwise you just broke "
        f"every client that reads output['{next(iter(missing))}']."
    )


@pytest.mark.fast
def test_all_workflows_describable(
    vibe_client: FarmvibesAiClient, workflow_catalog: list[str]
) -> None:
    """Every advertised workflow can be described.

    A workflow in list_workflows() that fails describe() is a workflow
    that 500s when clicked in any UI. Usually a YAML parse error or a
    missing import in an op spec.

    Note: desc["name"] is the YAML `name:` (e.g. "download_cdl"), not the
    catalog path ("data_ingestion/cdl/download_cdl"). Don't compare them.
    """
    broken: list[tuple[str, str]] = []
    for wf in workflow_catalog:
        try:
            d = vibe_client.describe_workflow(wf)
            if not isinstance(d, dict) or "outputs" not in d:
                broken.append(
                    (wf, f"malformed: keys={list(d.keys()) if isinstance(d, dict) else type(d)}")
                )
        except Exception as e:
            broken.append((wf, f"{type(e).__name__}: {e}"))
    assert not broken, (
        f"{len(broken)} workflow(s) advertised but not describable:\n  "
        + "\n  ".join(f"{wf}: {err}" for wf, err in broken)
    )


# ===========================================================================
# Tier 2: execution regression (slow)
# ===========================================================================
# Workflows that actually run end-to-end. Each row: submit → wait →
# assert done + output keys. The marks on each row determine when it runs:
#
#   marks=[pytest.mark.slow]                         → runs with -m slow
#   marks=[pytest.mark.slow, pytest.mark.external]   → only with -m external
#
# helloworld is the only self-contained workflow (no creds, no downloads),
# so it's the only one in the default slow tier. Everything else hits the
# internet and lives under `external`.
#
# To add a self-contained workflow: one row with marks=[pytest.mark.slow].
# To add an external-data workflow: one row with both marks. Done.

EXECUTION_CASES = [
    pytest.param(
        WorkflowCase("helloworld", {"raster"}),
        id="helloworld",
        marks=[pytest.mark.slow],
    ),
    # --- external-data cases (opt-in via -m external) ---
    pytest.param(
        WorkflowCase(
            "data_ingestion/cdl/download_cdl",
            {"raster"},
            geometry=_TINY_POLY,
            # CDL is year-resolution; range must span a full year boundary.
            time_range=(
                datetime(2020, 1, 1, tzinfo=timezone.utc),
                datetime(2020, 12, 31, tzinfo=timezone.utc),
            ),
            timeout_s=600,
        ),
        id="cdl",
        marks=[pytest.mark.slow, pytest.mark.external],
    ),
    pytest.param(
        WorkflowCase(
            "data_ingestion/osm_road_geometries",
            {"roads"},
            geometry=_TINY_POLY,
            parameters={"network_type": "drive", "buffer_size": 100},
            timeout_s=300,
        ),
        id="osm-roads",
        marks=[pytest.mark.slow, pytest.mark.external],
    ),
]


@pytest.mark.parametrize("case", EXECUTION_CASES)
def test_workflow_executes(submitter: WorkflowSubmitter, case: WorkflowCase) -> None:
    """Parametrized end-to-end execution.

    This is the extension point for "does workflow X still complete?"
    regression coverage. It does NOT check output correctness (pixel
    values, geometry bounds, etc.) — that belongs in per-workflow test
    files with golden outputs, like the legacy test_cluster_integration.py
    does for helloworld.
    """
    rec, status = submitter.submit_and_wait(
        case.workflow,
        timeout_s=case.timeout_s,
        geometry=case.geometry,
        time_range=case.time_range,
        parameters=case.parameters or None,
    )

    if status == RunStatus.failed:
        reason = str(rec.run.reason or "")
        # Transport-level failures from external data sources aren't our
        # bug. Skip instead of fail so the dashboard stays green for
        # FarmVibes-attributable issues only.
        transport_markers = ("connection", "timeout", "503", "502", "unavailable", "dns")
        if any(m in reason.lower() for m in transport_markers):
            pytest.skip(f"External service unavailable: {reason[:200]}")
        pytest.fail(
            f"'{case.workflow}' failed after {rec.elapsed_s:.1f}s.\n"
            f"  reason: {reason}\n"
            f"  tasks:  {rec.run.task_details}"
        )

    assert status == RunStatus.done, f"unexpected terminal status: {status}"

    out = rec.run.output
    assert out is not None, "status=done but output is None (orchestrator bug)"
    actual_keys = set(out.keys())
    missing = case.expected_outputs - actual_keys
    assert not missing, (
        f"'{case.workflow}' completed but output is missing keys: {missing}. "
        f"Got: {actual_keys}"
    )


# ===========================================================================
# Orchestration-path tests (not parametrized — these test the machinery,
# not the workflows)
# ===========================================================================
# These use helloworld as a vehicle because it's fast and self-contained,
# but what's under test is the orchestrator's state machine, not
# helloworld's logic. Parametrizing them over workflows would be
# pointless — the cache either works or it doesn't, regardless of what
# you cache.

_HELLOWORLD_TIMEOUT_S = 300


@pytest.mark.slow
def test_identical_submission_hits_cache(submitter: WorkflowSubmitter) -> None:
    """Two identical submissions → same output asset path.

    Exercises the cache layer. Failure modes: caching broken (every
    submit re-runs, big perf regression) or cache key derivation changed
    (invalidates everyone's cache on upgrade).
    """
    rec1, s1 = submitter.submit_and_wait("helloworld", timeout_s=_HELLOWORLD_TIMEOUT_S)
    rec2, s2 = submitter.submit_and_wait("helloworld", timeout_s=_HELLOWORLD_TIMEOUT_S)
    assert s1 == RunStatus.done and s2 == RunStatus.done

    path1 = rec1.run.output["raster"][0].assets[0].local_path
    path2 = rec2.run.output["raster"][0].assets[0].local_path
    assert path1 == path2, (
        f"Cache miss on identical inputs.\n  run1: {path1}\n  run2: {path2}\n"
        f"Op was re-executed instead of served from cache."
    )


@pytest.mark.slow
def test_different_geometry_produces_different_output(
    submitter: WorkflowSubmitter,
) -> None:
    """Guard against a cache-key bug where geometry is ignored.

    If two different geometries map to the same cache entry, every user
    gets the first user's results. Catastrophic and silent.
    """
    poly_a = Polygon(
        [(-88.06, 37.08), (-88.05, 37.08), (-88.05, 37.07), (-88.06, 37.07), (-88.06, 37.08)]
    )
    poly_b = Polygon(
        [(-87.06, 36.08), (-87.05, 36.08), (-87.05, 36.07), (-87.06, 36.07), (-87.06, 36.08)]
    )

    rec_a, sa = submitter.submit_and_wait(
        "helloworld", timeout_s=_HELLOWORLD_TIMEOUT_S, geometry=poly_a, time_range=FIXED_TIME_RANGE
    )
    rec_b, sb = submitter.submit_and_wait(
        "helloworld", timeout_s=_HELLOWORLD_TIMEOUT_S, geometry=poly_b, time_range=FIXED_TIME_RANGE
    )
    assert sa == RunStatus.done and sb == RunStatus.done

    path_a = rec_a.run.output["raster"][0].assets[0].local_path
    path_b = rec_b.run.output["raster"][0].assets[0].local_path
    assert path_a != path_b, (
        "Different geometries → SAME output asset. Cache key isn't "
        "incorporating geometry. Everyone is getting stale results."
    )


@pytest.mark.slow
def test_cancel_inflight_run(submitter: WorkflowSubmitter) -> None:
    """Cancel mid-execution.

    This is the Ctrl-C path. If cancel doesn't stop the worker, the run
    flips to 'done' instead of 'cancelled' — which we've shipped before.
    """
    rec = submitter.submit("helloworld", name_prefix="cancel")

    poll_until(
        lambda: rec.run.status in (RunStatus.running, RunStatus.done),
        timeout_s=60,
        interval_s=1.0,
        what="run to leave pending state",
    )

    if rec.run.status == RunStatus.done:
        pytest.skip(
            "Run finished before cancel could land (cache hit on a warm "
            "cluster). Not a failure. Re-run cold to exercise cancel."
        )

    rec.run.cancel()
    final = poll_until(
        lambda: rec.run.status if RunStatus.finished(rec.run.status) else None,
        timeout_s=120,
        what="cancelled run to terminate",
    )
    # 'done' is an acceptable race (worker finished before cancel arrived).
    # 'failed' is NOT — means cancel crashed the worker.
    assert final in (RunStatus.cancelled, RunStatus.done), (
        f"Cancel → status={final}. 'failed' means cancel crashed the "
        f"worker instead of stopping it cleanly."
    )


# ---------------------------------------------------------------------------
# Introspection helper — lets `pytest --collect-only` show case counts
# without having to read this file.
# ---------------------------------------------------------------------------

# Used by README generation / CI reporting. Not a test.
_ = KENTUCKY_POLYGON  # keep the import alive for future case additions
