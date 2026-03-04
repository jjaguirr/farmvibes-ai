import uuid
from datetime import datetime, timezone

import pytest
from shapely.geometry import Polygon

from vibe_dev.testing.integration import ClusterClient, WorkflowPoller

pytestmark = pytest.mark.slow

VALID_POLYGON = Polygon([
    (-88.062073563448919, 37.081397673802059),
    (-88.026349330507315, 37.085463858128762),
    (-88.012445388773259, 37.069230099135126),
    (-88.035931592028305, 37.048441375086092),
    (-88.068120429075847, 37.058833638440767),
    (-88.062073563448919, 37.081397673802059),
])

VALID_TIME_RANGE = (
    datetime(2021, 2, 1, tzinfo=timezone.utc),
    datetime(2021, 2, 11, tzinfo=timezone.utc),
)


class TestInvalidWorkflow:
    def test_nonexistent_workflow_returns_error(self, cluster_client: ClusterClient):
        r = cluster_client.post("/v0/runs", json={
            "name": "bad_workflow_test",
            "workflow": "this_workflow_does_not_exist_xyz",
            "parameters": None,
            "user_input": {
                "start_date": VALID_TIME_RANGE[0].isoformat(),
                "end_date": VALID_TIME_RANGE[1].isoformat(),
                "geojson": {
                    "type": "FeatureCollection",
                    "features": [{"type": "Feature", "geometry": {
                        "type": "Polygon",
                        "coordinates": [list(VALID_POLYGON.exterior.coords)],
                    }}],
                },
            },
        })
        assert r.status_code >= 400, f"Expected 4xx, got {r.status_code}: {r.text}"


class TestMalformedPayload:
    def test_missing_required_fields(self, cluster_client: ClusterClient):
        """POST /v0/runs with missing fields must return 422."""
        r = cluster_client.post("/v0/runs", json={"garbage": True})
        assert r.status_code == 422, f"Expected 422, got {r.status_code}: {r.text}"

    def test_invalid_geojson_type(self, cluster_client: ClusterClient):
        """POST /v0/runs with geojson as a string instead of dict must return 422."""
        r = cluster_client.post("/v0/runs", json={
            "name": "bad_geojson_test",
            "workflow": "helloworld",
            "parameters": None,
            "user_input": {
                "start_date": "2021-02-01T00:00:00+00:00",
                "end_date": "2021-02-11T00:00:00+00:00",
                "geojson": "not a geojson",
            },
        })
        assert r.status_code == 422, f"Expected 422, got {r.status_code}: {r.text}"

    def test_missing_user_input(self, cluster_client: ClusterClient):
        """POST /v0/runs without user_input must return 422."""
        r = cluster_client.post("/v0/runs", json={
            "name": "no_input_test",
            "workflow": "helloworld",
            "parameters": None,
        })
        assert r.status_code == 422, f"Expected 422, got {r.status_code}: {r.text}"


class TestNonexistentRun:
    def test_get_unknown_run_returns_error(self, cluster_client: ClusterClient):
        fake_id = str(uuid.uuid4())
        r = cluster_client.get(f"/v0/runs/{fake_id}")
        assert r.status_code >= 400, f"Expected error for nonexistent run, got {r.status_code}"


class TestCancelWorkflow:
    def test_cancel_running_workflow(
        self, cluster_client: ClusterClient, workflow_poller: WorkflowPoller
    ):
        result = cluster_client.submit_run(
            workflow="helloworld",
            name="cancel_test",
            geometry=VALID_POLYGON,
            time_range=VALID_TIME_RANGE,
        )
        run_id = result["id"]

        cancel_r = cluster_client.cancel_run(run_id)
        # Accept either success (200) or already-done (various codes)
        assert cancel_r.status_code < 500, f"Server error on cancel: {cancel_r.text}"

        # Poll to terminal state
        run = workflow_poller.poll(run_id, timeout_s=60.0)
        assert run["details"]["status"] in ("cancelled", "done", "failed"), (
            f"Run ended in unexpected state: {run['details']['status']}"
        )
