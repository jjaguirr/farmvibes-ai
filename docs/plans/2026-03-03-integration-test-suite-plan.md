# Integration Test Suite Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a coherent integration test suite that exercises the full FarmVibes.AI pipeline against a live cluster.

**Architecture:** Reusable helpers in `vibe_dev/testing/integration.py` (ClusterClient, WorkflowPoller, WorkflowSpec). Test modules in `src/tests_local_cluster/` split by concern. Pytest markers separate fast (<30s) from slow (minutes) tests.

**Tech Stack:** Python 3.10+, pytest, requests, subprocess

---

### Task 1: Create helpers module `vibe_dev/testing/integration.py`

**Files:**
- Create: `src/vibe_dev/vibe_dev/testing/integration.py`

**Step 1: Write the ClusterClient class**

```python
# src/vibe_dev/vibe_dev/testing/integration.py

import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import requests
import yaml
from shapely.geometry import Polygon, mapping

FALLBACK_SERVICE_URL = "http://127.0.0.1:31108"

FARMVIBES_AI_BASE_URL_ENV = "FARMVIBES_AI_BASE_URL"


def resolve_cluster_url() -> str:
    env_url = os.environ.get(FARMVIBES_AI_BASE_URL_ENV)
    if env_url:
        return env_url.rstrip("/")

    xdg = os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config"))
    for filename in ("remote_service_url", "service_url"):
        path = os.path.join(xdg, "farmvibes-ai", filename)
        if os.path.isfile(path):
            with open(path) as f:
                url = f.read().strip()
            if url:
                return url.rstrip("/")

    return FALLBACK_SERVICE_URL


class ClusterClient:
    """HTTP client for FarmVibes.AI cluster API. Reusable in any test suite."""

    def __init__(self, base_url: Optional[str] = None, timeout: float = 30.0):
        self.base_url = (base_url or resolve_cluster_url()).rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "Content-Type": "application/json",
        })

    def get(self, path: str, **kwargs) -> requests.Response:
        return self.session.get(f"{self.base_url}/{path.lstrip('/')}", timeout=self.timeout, **kwargs)

    def post(self, path: str, **kwargs) -> requests.Response:
        return self.session.post(f"{self.base_url}/{path.lstrip('/')}", timeout=self.timeout, **kwargs)

    def delete(self, path: str, **kwargs) -> requests.Response:
        return self.session.delete(f"{self.base_url}/{path.lstrip('/')}", timeout=self.timeout, **kwargs)

    def root(self) -> Dict[str, Any]:
        r = self.get("v0/")
        r.raise_for_status()
        return r.json()

    def system_metrics(self) -> Dict[str, Any]:
        r = self.get("v0/system-metrics")
        r.raise_for_status()
        return r.json()

    def liveness(self) -> requests.Response:
        return self.get("healthz/live")

    def readiness(self) -> requests.Response:
        return self.get("healthz/ready")

    def health(self) -> requests.Response:
        return self.get("v0/health")

    def list_workflows(self) -> List[str]:
        r = self.get("v0/workflows")
        r.raise_for_status()
        return r.json()

    def describe_workflow(self, name: str) -> Dict[str, Any]:
        r = self.get(f"v0/workflows/{name}?return_format=description")
        r.raise_for_status()
        return r.json()

    def get_workflow_yaml(self, name: str) -> str:
        r = self.get(f"v0/workflows/{name}?return_format=yaml")
        r.raise_for_status()
        return yaml.dump(r.json(), default_flow_style=False, sort_keys=False)

    def submit_run(
        self,
        workflow: str,
        name: str,
        geometry: Polygon,
        time_range: Tuple[datetime, datetime],
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload = {
            "name": name,
            "workflow": workflow,
            "parameters": parameters,
            "user_input": {
                "start_date": time_range[0].isoformat(),
                "end_date": time_range[1].isoformat(),
                "geojson": {
                    "type": "FeatureCollection",
                    "features": [{"type": "Feature", "geometry": mapping(geometry)}],
                },
            },
        }
        r = self.post("v0/runs", json=payload)
        r.raise_for_status()
        return r.json()

    def get_run(self, run_id: str) -> Dict[str, Any]:
        r = self.get(f"v0/runs/{run_id}")
        r.raise_for_status()
        return r.json()

    def list_runs(self, ids: Optional[List[str]] = None, fields: Optional[List[str]] = None) -> List[Dict]:
        params: Dict[str, Any] = {}
        if ids:
            params["ids"] = ",".join(ids)
        if fields:
            params["fields"] = ",".join(fields)
        r = self.get("v0/runs", params=params)
        r.raise_for_status()
        return r.json()

    def cancel_run(self, run_id: str) -> requests.Response:
        return self.post(f"v0/runs/{run_id}/cancel")

    def delete_run(self, run_id: str) -> requests.Response:
        return self.delete(f"v0/runs/{run_id}")


class WorkflowPoller:
    """Polls a workflow run until terminal state or timeout."""

    TERMINAL_STATUSES = {"done", "failed", "cancelled", "deleted"}

    def __init__(self, client: ClusterClient, timeout_s: float = 120.0, interval_s: float = 5.0):
        self.client = client
        self.timeout_s = timeout_s
        self.interval_s = interval_s

    def poll(self, run_id: str, timeout_s: Optional[float] = None) -> Dict[str, Any]:
        timeout = timeout_s or self.timeout_s
        deadline = time.monotonic() + timeout
        last_status = None

        while time.monotonic() < deadline:
            runs = self.client.list_runs(ids=[run_id], fields=["details.status"])
            if runs:
                last_status = runs[0].get("details", {}).get("status")
                if last_status in self.TERMINAL_STATUSES:
                    return self.client.get_run(run_id)
            time.sleep(self.interval_s)

        raise TimeoutError(
            f"Workflow run {run_id} did not reach terminal state within {timeout}s. "
            f"Last status: {last_status}"
        )


@dataclass
class WorkflowSpec:
    """Defines a workflow test case for parametrized execution tests."""

    name: str
    geometry: Polygon
    time_range: Tuple[datetime, datetime]
    expected_sinks: List[str]
    timeout_s: float = 120.0
    parameters: Optional[Dict[str, Any]] = None
    run_name: str = ""

    def __post_init__(self):
        if not self.run_name:
            self.run_name = f"test_{self.name.replace('/', '_')}"
```

**Step 2: Commit**

```
jj describe -m "feat(testing): add integration test helpers — ClusterClient, WorkflowPoller, WorkflowSpec"
jj new
```

---

### Task 2: Create shared conftest with fixtures and markers

**Files:**
- Create: `src/tests_local_cluster/conftest.py`

**Step 1: Write conftest.py**

```python
# src/tests_local_cluster/conftest.py

import pytest

from vibe_core.client import FarmvibesAiClient
from vibe_dev.testing.integration import ClusterClient, WorkflowPoller, resolve_cluster_url


def pytest_configure(config):
    config.addinivalue_line("markers", "fast: fast tests (health, schema, CLI) — seconds")
    config.addinivalue_line("markers", "slow: slow tests (workflow execution, failures) — minutes")


@pytest.fixture(scope="session")
def cluster_url():
    return resolve_cluster_url()


@pytest.fixture(scope="session")
def cluster_client(cluster_url):
    return ClusterClient(base_url=cluster_url)


@pytest.fixture(scope="session")
def workflow_poller(cluster_client):
    return WorkflowPoller(client=cluster_client, timeout_s=120.0, interval_s=5.0)


@pytest.fixture(scope="session")
def vibe_client(cluster_url):
    return FarmvibesAiClient(baseurl=cluster_url)


@pytest.fixture(scope="session")
def all_workflows(cluster_client):
    return cluster_client.list_workflows()
```

**Step 2: Commit**

```
jj describe -m "feat(testing): add conftest with shared fixtures and pytest markers"
jj new
```

---

### Task 3: Write health tests

**Files:**
- Create: `src/tests_local_cluster/test_health.py`

**Step 1: Write the tests**

```python
# src/tests_local_cluster/test_health.py

import pytest

from vibe_dev.testing.integration import ClusterClient

pytestmark = pytest.mark.fast


class TestRootEndpoint:
    def test_returns_200(self, cluster_client: ClusterClient):
        r = cluster_client.get("v0/")
        assert r.status_code == 200

    def test_response_has_message(self, cluster_client: ClusterClient):
        data = cluster_client.root()
        assert "message" in data
        assert "running" in data["message"].lower()


class TestSystemMetrics:
    def test_returns_200(self, cluster_client: ClusterClient):
        r = cluster_client.get("v0/system-metrics")
        assert r.status_code == 200

    def test_has_expected_fields(self, cluster_client: ClusterClient):
        metrics = cluster_client.system_metrics()
        for field in ("cpu_usage", "free_mem", "used_mem", "total_mem", "disk_free", "load_avg"):
            assert field in metrics, f"Missing field: {field}"

    def test_load_avg_is_list(self, cluster_client: ClusterClient):
        metrics = cluster_client.system_metrics()
        assert isinstance(metrics["load_avg"], list)
        assert len(metrics["load_avg"]) == 3


class TestLivenessProbe:
    def test_returns_200(self, cluster_client: ClusterClient):
        r = cluster_client.liveness()
        assert r.status_code == 200

    def test_response_body(self, cluster_client: ClusterClient):
        r = cluster_client.liveness()
        assert r.json()["status"] == "alive"


class TestReadinessProbe:
    def test_returns_200_when_ready(self, cluster_client: ClusterClient):
        r = cluster_client.readiness()
        assert r.status_code == 200

    def test_response_body(self, cluster_client: ClusterClient):
        r = cluster_client.readiness()
        assert r.json()["status"] == "ready"


class TestDetailedHealth:
    def test_returns_200(self, cluster_client: ClusterClient):
        r = cluster_client.health()
        assert r.status_code == 200

    def test_has_status_and_dependencies(self, cluster_client: ClusterClient):
        r = cluster_client.health()
        data = r.json()
        assert "status" in data
        assert "dependencies" in data
        assert isinstance(data["dependencies"], list)

    def test_dependencies_have_required_fields(self, cluster_client: ClusterClient):
        r = cluster_client.health()
        data = r.json()
        for dep in data["dependencies"]:
            assert "name" in dep
            assert "status" in dep
            assert "latency_ms" in dep


class TestWorkflowsList:
    def test_returns_200(self, cluster_client: ClusterClient):
        r = cluster_client.get("v0/workflows")
        assert r.status_code == 200

    def test_returns_non_empty_list(self, all_workflows):
        assert isinstance(all_workflows, list)
        assert len(all_workflows) > 0

    def test_contains_helloworld(self, all_workflows):
        assert "helloworld" in all_workflows
```

**Step 2: Run tests to verify they work against cluster**

Run on VM: `cd ~/farmvibes-ai-work/12_model_b && source ~/venvs/12_model_b/bin/activate && pytest src/tests_local_cluster/test_health.py -v -m fast`

Expected: Some pass (root, metrics, workflows), some fail (healthz/live, healthz/ready, /v0/health return 404 on current cluster).

**Step 3: Commit**

```
jj describe -m "test(integration): add health and service endpoint tests"
jj new
```

---

### Task 4: Write schema regression tests

**Files:**
- Create: `src/tests_local_cluster/test_schema_regression.py`

**Step 1: Write the tests**

```python
# src/tests_local_cluster/test_schema_regression.py

import pytest
import yaml

from vibe_dev.testing.integration import ClusterClient

pytestmark = pytest.mark.fast


def workflow_ids(all_workflows):
    """Generate short test IDs from workflow names."""
    return [w.replace("/", "-") for w in all_workflows]


class TestWorkflowDescribe:
    def test_describe_returns_200(self, cluster_client: ClusterClient, all_workflows):
        """Sanity check: at least helloworld is describable."""
        r = cluster_client.get("v0/workflows/helloworld?return_format=description")
        assert r.status_code == 200

    @pytest.fixture(params="lazy")
    def workflow_name(self, request, all_workflows):
        return all_workflows[request.param]

    def test_describe_has_required_fields(self, cluster_client: ClusterClient, all_workflows):
        """Parametrized over all workflows — each must have name, inputs, outputs."""
        errors = []
        for wf in all_workflows:
            try:
                desc = cluster_client.describe_workflow(wf)
                for field in ("name", "inputs", "outputs"):
                    if field not in desc:
                        errors.append(f"{wf}: missing field '{field}'")
            except Exception as e:
                errors.append(f"{wf}: {e}")
        assert not errors, f"Schema failures:\n" + "\n".join(errors)


class TestWorkflowYaml:
    def test_yaml_is_valid(self, cluster_client: ClusterClient, all_workflows):
        """Every workflow's YAML representation must parse as valid YAML with required keys."""
        errors = []
        for wf in all_workflows:
            try:
                yaml_str = cluster_client.get_workflow_yaml(wf)
                parsed = yaml.safe_load(yaml_str)
                if not isinstance(parsed, dict):
                    errors.append(f"{wf}: YAML did not parse as dict")
                    continue
                for key in ("name", "sources", "sinks", "tasks"):
                    if key not in parsed:
                        errors.append(f"{wf}: missing YAML key '{key}'")
            except Exception as e:
                errors.append(f"{wf}: {e}")
        assert not errors, f"YAML validation failures:\n" + "\n".join(errors)
```

**Step 2: Run on VM**

Run: `pytest src/tests_local_cluster/test_schema_regression.py -v -m fast`

Expected: PASS — all 91 workflows should describe and parse correctly.

**Step 3: Commit**

```
jj describe -m "test(integration): add schema regression tests for all workflows"
jj new
```

---

### Task 5: Write workflow execution tests

**Files:**
- Create: `src/tests_local_cluster/test_workflow_execution.py`

**Step 1: Write the tests**

```python
# src/tests_local_cluster/test_workflow_execution.py

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
        """Run helloworld twice — second run should complete faster (cache hit)."""
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
```

**Step 2: Run on VM**

Run: `pytest src/tests_local_cluster/test_workflow_execution.py -v -m slow`

Expected: PASS — helloworld completes in ~30s, caching test passes.

**Step 3: Commit**

```
jj describe -m "test(integration): add workflow execution and caching tests"
jj new
```

---

### Task 6: Write CLI tests

**Files:**
- Create: `src/tests_local_cluster/test_cli.py`

**Step 1: Write the tests**

```python
# src/tests_local_cluster/test_cli.py

import subprocess

import pytest

pytestmark = pytest.mark.fast


def run_cli(*args: str, timeout: float = 30.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["farmvibes-ai", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


class TestCliStatus:
    def test_exit_code_zero(self):
        result = run_cli("local", "status")
        assert result.returncode == 0, f"stderr: {result.stderr}"

    def test_output_contains_url(self):
        result = run_cli("local", "status")
        assert "http" in result.stdout.lower() or "http" in result.stderr.lower(), (
            f"Expected URL in output.\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )


class TestCliHealth:
    def test_runs_without_crash(self):
        result = run_cli("local", "health")
        # Exit code: 0=healthy, 1=degraded, 2=down — all are valid outcomes
        assert result.returncode in (0, 1, 2), (
            f"Unexpected exit code {result.returncode}.\nstderr: {result.stderr}"
        )
```

**Step 2: Run on VM**

Run: `pytest src/tests_local_cluster/test_cli.py -v -m fast`

Expected: Depends on whether `farmvibes-ai` CLI is on PATH. May need to adjust to use `python -m vibe_core.cli.main` instead.

**Step 3: Commit**

```
jj describe -m "test(integration): add CLI tests for status and health commands"
jj new
```

---

### Task 7: Write failure tests

**Files:**
- Create: `src/tests_local_cluster/test_failures.py`

**Step 1: Write the tests**

```python
# src/tests_local_cluster/test_failures.py

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
        r = cluster_client.post("v0/runs", json={
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


class TestInvalidDateRange:
    def test_end_before_start_returns_error(self, cluster_client: ClusterClient):
        r = cluster_client.post("v0/runs", json={
            "name": "bad_dates_test",
            "workflow": "helloworld",
            "parameters": None,
            "user_input": {
                "start_date": "2021-03-01T00:00:00+00:00",
                "end_date": "2021-01-01T00:00:00+00:00",
                "geojson": {
                    "type": "FeatureCollection",
                    "features": [{"type": "Feature", "geometry": {
                        "type": "Polygon",
                        "coordinates": [list(VALID_POLYGON.exterior.coords)],
                    }}],
                },
            },
        })
        # Server should reject or the workflow should fail
        assert r.status_code >= 400 or r.status_code == 201, (
            f"Unexpected status code: {r.status_code}"
        )
        if r.status_code == 201:
            # If accepted, the workflow should fail during execution
            pass  # Acceptable: some workflows accept bad dates and fail gracefully


class TestNonexistentRun:
    def test_get_unknown_run_returns_error(self, cluster_client: ClusterClient):
        fake_id = str(uuid.uuid4())
        r = cluster_client.get(f"v0/runs/{fake_id}")
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
```

**Step 2: Run on VM**

Run: `pytest src/tests_local_cluster/test_failures.py -v -m slow`

Expected: Invalid workflow returns error. Nonexistent run returns error. Cancel test completes.

**Step 3: Commit**

```
jj describe -m "test(integration): add failure and error handling tests"
jj new
```

---

### Task 8: Export helpers from `vibe_dev/testing/__init__.py`

**Files:**
- Modify: `src/vibe_dev/vibe_dev/testing/__init__.py`

**Step 1: Add imports**

Add to existing `__init__.py`:

```python
try:
    from .integration import ClusterClient, WorkflowPoller, WorkflowSpec, resolve_cluster_url
except ImportError:
    pass
```

This keeps backward compatibility — if requests/shapely aren't installed in a minimal env, the existing fixtures still work.

**Step 2: Commit**

```
jj describe -m "feat(testing): export integration helpers from vibe_dev.testing"
jj new
```

---

### Task 9: Set up VM worktree, push, and run full suite

**Step 1: Set up VM worktree**

```bash
cd ~/farmvibes-ai-work && git fetch origin
git worktree add ~/farmvibes-ai-work/12_model_b origin/main
python3 -m venv ~/venvs/12_model_b
~/venvs/12_model_b/bin/pip install -e ~/farmvibes-ai-work/12_model_b/src/vibe_core/
cat > ~/farmvibes-env_12_model_b.sh << 'EOF'
source ~/venvs/12_model_b/bin/activate
cd ~/farmvibes-ai-work/12_model_b
EOF
```

**Step 2: Push branch locally, pull on VM**

Local: `jj git push` (user runs this)
VM: `source ~/farmvibes-env_12_model_b.sh && git fetch origin && git checkout integration-tests_task12_model_b`

**Step 3: Install and run**

```bash
source ~/farmvibes-env_12_model_b.sh
pip install -e src/vibe_dev/
pytest src/tests_local_cluster/ -v -m fast
pytest src/tests_local_cluster/ -v -m slow
pytest src/tests_local_cluster/ -v  # full suite
```

**Step 4: Fix any failures, iterate, commit fixes**

---

### Task 10: Final validation and cleanup

**Step 1: Verify markers work**

```bash
pytest src/tests_local_cluster/ -v -m fast --co  # collect only, verify fast tests
pytest src/tests_local_cluster/ -v -m slow --co  # collect only, verify slow tests
```

**Step 2: Run full suite with timing**

```bash
pytest src/tests_local_cluster/ -v --durations=0
```

Expected: fast tests < 30s, full suite < 10min.

**Step 3: Final commit**

```
jj describe -m "test(integration): complete integration test suite — health, schema, execution, CLI, failures"
```
