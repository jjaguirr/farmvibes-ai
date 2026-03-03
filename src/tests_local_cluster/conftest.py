# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Pytest wiring for the live-cluster integration suite.

Running the suite
-----------------
These tests talk to a real FarmVibes.AI cluster. Point them at one with::

    export FARMVIBES_AI_SERVICE_URL=http://127.0.0.1:31109/
    pytest src/tests_local_cluster -m fast          # health + failure, seconds
    pytest src/tests_local_cluster -m slow          # workflow execution, minutes
    pytest src/tests_local_cluster                  # everything

If ``FARMVIBES_AI_SERVICE_URL`` is unset, the URL is read from
``~/.config/farmvibes-ai/service_url`` (the same file the CLI writes).

If the cluster is unreachable, the whole suite skips with a single clear
message instead of forty individual connection-refused failures.
"""

from __future__ import annotations

from typing import Iterator

import pytest

from vibe_core.client import FarmvibesAiClient

from .helpers import WorkflowSubmitter, probe_reachable, resolve_service_url


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "fast: quick tests (health checks, input validation) — safe to run on every commit",
    )
    config.addinivalue_line(
        "markers",
        "slow: tests that submit real workflows and wait — budget minutes, not seconds",
    )
    config.addinivalue_line(
        "markers",
        "cli: tests that shell out to the `farmvibes-ai` CLI — require kubeconfig access",
    )
    config.addinivalue_line(
        "markers",
        "newapi: tests that exercise endpoints added after the 2025 image baseline "
        "(/v0/health, /healthz/*) — will skip on older cluster images",
    )
    config.addinivalue_line(
        "markers",
        "external: tests that download real data from third-party services "
        "(USDA, OSM, Planetary Computer) — opt-in only, flaky by nature",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Auto-skip `external` tests unless explicitly selected.

    Without this, `pytest src/tests_local_cluster` would include the
    external tier by default, which means CI runs hit USDA servers on
    every commit. We invert that: you have to ask for `-m external` to
    run them, but `-m slow` won't accidentally pull them in.
    """
    markexpr = config.getoption("-m", default="")
    if "external" in markexpr:
        return  # user asked for them explicitly
    skip_ext = pytest.mark.skip(reason="external marker not selected (use -m external)")
    for item in items:
        if "external" in item.keywords:
            item.add_marker(skip_ext)


# ---------------------------------------------------------------------------
# Session fixtures
# ---------------------------------------------------------------------------
# `service_url` and `vibe_client` are session-scoped because connecting is
# cheap but the reachability probe is not something we want to repeat per-test.
# The probe skips the entire session if the cluster is down — this is
# deliberate. An integration suite that runs green against a dead cluster is
# worse than useless.


@pytest.fixture(scope="session")
def service_url() -> str:
    url = resolve_service_url()
    ok, detail = probe_reachable(url)
    if not ok:
        pytest.skip(
            f"FarmVibes.AI cluster unreachable at {url}: {detail}. "
            f"Set FARMVIBES_AI_SERVICE_URL or start the cluster "
            f"(`farmvibes-ai local start`).",
            allow_module_level=True,
        )
    return url


@pytest.fixture(scope="session")
def vibe_client(service_url: str) -> FarmvibesAiClient:
    return FarmvibesAiClient(service_url)


@pytest.fixture(scope="session")
def workflow_catalog(vibe_client: FarmvibesAiClient) -> list[str]:
    """The full workflow list from the live cluster.

    Session-scoped because it's a non-trivial HTTP call and the catalog
    doesn't change mid-session. Tests that need to assert "workflow X is
    available" should use this rather than calling list_workflows again.
    """
    return vibe_client.list_workflows()


@pytest.fixture
def submitter(vibe_client: FarmvibesAiClient) -> Iterator[WorkflowSubmitter]:
    """Per-test workflow submitter with automatic cleanup.

    Function-scoped on purpose: each test gets a clean slate and its runs
    are torn down even if the test body raises. This is the fixture to use
    in any test that submits a workflow.
    """
    sub = WorkflowSubmitter(vibe_client)
    yield sub
    sub.cleanup()
