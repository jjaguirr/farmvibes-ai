import pytest
from vibe_dev.testing.integration import ClusterClient

pytestmark = pytest.mark.fast


class TestRootEndpoint:
    def test_returns_200(self, cluster_client: ClusterClient):
        r = cluster_client.get("/v0/")
        assert r.status_code == 200

    def test_response_has_message(self, cluster_client: ClusterClient):
        data = cluster_client.root()
        assert "message" in data
        assert "running" in data["message"].lower()


class TestSystemMetrics:
    def test_returns_200(self, cluster_client: ClusterClient):
        r = cluster_client.get("/v0/system-metrics")
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
        r = cluster_client.get("/v0/workflows")
        assert r.status_code == 200

    def test_returns_non_empty_list(self, all_workflows):
        assert isinstance(all_workflows, list)
        assert len(all_workflows) > 0

    def test_contains_helloworld(self, all_workflows):
        assert "helloworld" in all_workflows
