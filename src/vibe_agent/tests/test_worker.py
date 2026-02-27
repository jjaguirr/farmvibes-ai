# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for vibe_agent.worker hardening (Task 11).

These tests require pebble/dapr/hydra_zen installed — they don't collect on
the minimal local dev env. Full validation runs on the VM (Task 13).
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vibe_agent.worker import WorkerMessenger


# --- WorkerMessenger send cap (Task 6) ---

@pytest.mark.anyio
async def test_messenger_send_caps_retries():
    """send() must not loop forever when pubsub is down."""
    messenger = WorkerMessenger()
    messenger.max_send_attempts = 3
    messenger.retry_interval_s = 0.01

    with patch("vibe_agent.worker.send_async", new=AsyncMock(return_value=False)):
        with pytest.raises(RuntimeError, match="3 attempts"):
            await messenger.send(MagicMock())


@pytest.mark.anyio
async def test_messenger_send_succeeds_after_transient_failure():
    """send() should succeed once send_async returns True, not keep retrying."""
    messenger = WorkerMessenger()
    messenger.max_send_attempts = 5
    messenger.retry_interval_s = 0.01

    call_count = 0
    async def flaky_send(*a, **kw):
        nonlocal call_count
        call_count += 1
        return call_count >= 3  # succeed on 3rd try

    with patch("vibe_agent.worker.send_async", new=flaky_send):
        await messenger.send(MagicMock())  # should not raise
    assert call_count == 3


@pytest.mark.anyio
async def test_messenger_send_respects_deadline():
    """When deadline is in the past, send() gives up immediately."""
    import time

    messenger = WorkerMessenger()
    messenger.max_send_attempts = 100
    messenger.retry_interval_s = 0.01
    messenger.deadline = time.monotonic() - 1.0  # already expired

    with patch("vibe_agent.worker.send_async", new=AsyncMock(return_value=False)):
        with pytest.raises(RuntimeError, match="deadline"):
            await messenger.send(MagicMock())
