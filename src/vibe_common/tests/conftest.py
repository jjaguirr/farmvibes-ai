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
except ModuleNotFoundError as e:
    # Tolerate only the expected "vibe_dev infrastructure unavailable" case —
    # either vibe_dev itself is missing, or one of its transitive deps is missing
    # while vibe_dev was mid-import. Any other missing module is a real failure.
    import traceback

    _in_vibe_dev_chain = any(
        "vibe_dev" in frame.filename for frame in traceback.extract_tb(e.__traceback__)
    )
    if not (e.name and e.name.startswith("vibe_dev")) and not _in_vibe_dev_chain:
        raise  # real missing dep, not the expected "vibe_dev not installed"
    import pytest

    @pytest.fixture
    def anyio_backend():
        return "asyncio"

    __all__ = ["anyio_backend"]
