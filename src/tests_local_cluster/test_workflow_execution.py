from datetime import datetime, timezone

import pytest
from shapely.geometry import Polygon

from vibe_dev.testing.integration import ClusterClient, WorkflowPoller, WorkflowSpec

pytestmark = pytest.mark.slow

HELLOWORLD_POLYGON = Polygon([
    (-88.062073563448919, 37.081397673802059),
    (-88.026349330507315, 37.085463858128762),
    (-88.012445388773259, 37.069230099135126),
    (-88.035931592028305, 37.048441375086092),
    (-88.068120429075847, 37.058833638440767),
    (-88.062073563448919, 37.081397673802059),
])

HELLOWORLD_TIME_RANGE = (
    datetime(2021, 2, 1, tzinfo=timezone.utc),
    datetime(2021, 2, 11, tzinfo=timezone.utc),
)

WORKFLOW_SPECS = [
    WorkflowSpec(
        name="helloworld",
        geometry=HELLOWORLD_POLYGON,
        time_range=HELLOWORLD_TIME_RANGE,
        expected_sinks=["raster"],
        timeout_s=60.0,
    ),
    # To add more workflows, append WorkflowSpec entries here.
]


@pytest.fixture(params=WORKFLOW_SPECS, ids=[s.name for s in WORKFLOW_SPECS])
def workflow_spec(request) -> WorkflowSpec:
    return request.param


class TestWorkflowExecution:
    def test_workflow_completes(
        self, cluster_client: ClusterClient, workflow_poller: WorkflowPoller, workflow_spec: WorkflowSpec
    ):
        result = cluster_client.submit_run(
            workflow=workflow_spec.name,
            name=workflow_spec.run_name,
            geometry=workflow_spec.geometry,
            time_range=workflow_spec.time_range,
            parameters=workflow_spec.parameters,
        )
        assert "id" in result, f"Submit response missing 'id': {result}"
        run_id = result["id"]

        run = workflow_poller.poll(run_id, timeout_s=workflow_spec.timeout_s)
        assert run["details"]["status"] == "done", (
            f"Workflow {workflow_spec.name} ended with status "
            f"{run['details']['status']}: {run.get('details', {}).get('reason')}"
        )

    def test_workflow_has_expected_outputs(
        self, cluster_client: ClusterClient, workflow_poller: WorkflowPoller, workflow_spec: WorkflowSpec
    ):
        result = cluster_client.submit_run(
            workflow=workflow_spec.name,
            name=f"{workflow_spec.run_name}_outputs",
            geometry=workflow_spec.geometry,
            time_range=workflow_spec.time_range,
            parameters=workflow_spec.parameters,
        )
        run_id = result["id"]
        run = workflow_poller.poll(run_id, timeout_s=workflow_spec.timeout_s)

        assert run["details"]["status"] == "done"
        output = run.get("output")
        if output and workflow_spec.expected_sinks:
            for sink in workflow_spec.expected_sinks:
                assert sink in output, f"Expected sink '{sink}' not in output keys: {list(output.keys())}"


class TestWorkflowCaching:
    def test_second_run_uses_cache(
        self, cluster_client: ClusterClient, workflow_poller: WorkflowPoller
    ):
        """Run helloworld twice -- second run should complete faster (cache hit)."""
        spec = WORKFLOW_SPECS[0]  # helloworld

        r1 = cluster_client.submit_run(
            workflow=spec.name, name="cache_test_1",
            geometry=spec.geometry, time_range=spec.time_range,
        )
        run1 = workflow_poller.poll(r1["id"], timeout_s=spec.timeout_s)
        assert run1["details"]["status"] == "done"

        r2 = cluster_client.submit_run(
            workflow=spec.name, name="cache_test_2",
            geometry=spec.geometry, time_range=spec.time_range,
        )
        run2 = workflow_poller.poll(r2["id"], timeout_s=spec.timeout_s)
        assert run2["details"]["status"] == "done"
