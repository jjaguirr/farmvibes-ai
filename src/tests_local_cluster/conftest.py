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
