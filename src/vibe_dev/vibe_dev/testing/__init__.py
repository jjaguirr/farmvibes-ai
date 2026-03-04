# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest


@pytest.fixture
def anyio_backend():
    return "asyncio"


try:
    from .integration import ClusterClient, WorkflowPoller, WorkflowSpec, resolve_cluster_url
except ImportError:
    pass
