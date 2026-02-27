# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# vibe_dev is an optional test-infrastructure package. When it's installed
# (CI, full local dev), pull in the shared fixtures. When it isn't, fall
# back to a local anyio_backend so pure-stdlib tests (retry, resources)
# can still be collected and run.
try:
    from vibe_dev.testing import anyio_backend
    from vibe_dev.testing.fake_workflows_fixtures import fake_ops_dir, fake_workflows_dir
    from vibe_dev.testing.workflow_fixtures import (
        SimpleStrData,
        SimpleStrDataType,
        simple_op_spec,
        workflow_execution_message,
    )

    __all__ = [
        "SimpleStrDataType",
        "SimpleStrData",
        "workflow_execution_message",
        "simple_op_spec",
        "fake_ops_dir",
        "fake_workflows_dir",
        "anyio_backend",
    ]
except ImportError:
    import pytest

    @pytest.fixture
    def anyio_backend():
        return "asyncio"

    __all__ = ["anyio_backend"]
