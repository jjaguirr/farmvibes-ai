import requests
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
    client = ClusterClient(base_url=cluster_url)
    # Fail fast if cluster is unreachable — don't let every test hang individually
    try:
        r = client.get("/v0/", timeout=5)
        r.raise_for_status()
    except (requests.ConnectionError, requests.Timeout) as exc:
        pytest.exit(
            f"Cluster unreachable at {cluster_url} — aborting test session.\n{exc}",
            returncode=1,
        )
    return client


@pytest.fixture(scope="session")
def workflow_poller(cluster_client):
    return WorkflowPoller(client=cluster_client, timeout_s=120.0, interval_s=5.0)


@pytest.fixture(scope="session")
def vibe_client(cluster_url):
    return FarmvibesAiClient(baseurl=cluster_url)


@pytest.fixture(scope="session")
def all_workflows(cluster_client):
    return cluster_client.list_workflows()
