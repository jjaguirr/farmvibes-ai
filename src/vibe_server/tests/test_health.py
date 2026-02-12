# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vibe_server.health import (
    DependencyStatus,
    check_dapr_sidecar,
    check_service,
    check_statestore,
    get_detailed_health,
    get_readiness,
)


def _mock_aiohttp_session(resp_status=200, get_side_effect=None):
    """Create a mock aiohttp.ClientSession with proper async context manager support."""
    mock_resp = MagicMock()
    mock_resp.status = resp_status

    @asynccontextmanager
    async def _get(*args, **kwargs):
        if get_side_effect:
            raise get_side_effect
        yield mock_resp

    mock_session = MagicMock()
    mock_session.get = _get

    @asynccontextmanager
    async def _session_ctx(*args, **kwargs):
        yield mock_session

    return _session_ctx


@pytest.mark.anyio
async def test_check_dapr_sidecar_healthy():
    """Dapr sidecar returning 204 should report healthy."""
    mock_cls = _mock_aiohttp_session(resp_status=204)

    with patch("vibe_server.health.aiohttp.ClientSession", mock_cls):
        result = await check_dapr_sidecar()
    assert result.status == DependencyStatus.healthy
    assert result.name == "dapr"
    assert result.latency_ms > 0


@pytest.mark.anyio
async def test_check_dapr_sidecar_unhealthy_status():
    """Dapr sidecar returning non-2xx should report unhealthy."""
    mock_cls = _mock_aiohttp_session(resp_status=503)

    with patch("vibe_server.health.aiohttp.ClientSession", mock_cls):
        result = await check_dapr_sidecar()
    assert result.status == DependencyStatus.unhealthy
    assert "503" in result.detail


@pytest.mark.anyio
async def test_check_dapr_sidecar_connection_error():
    """Connection failure to Dapr sidecar should report unhealthy."""
    mock_cls = _mock_aiohttp_session(get_side_effect=ConnectionError("Connection refused"))

    with patch("vibe_server.health.aiohttp.ClientSession", mock_cls):
        result = await check_dapr_sidecar()
    assert result.status == DependencyStatus.unhealthy
    assert "Connection refused" in result.detail


@pytest.mark.anyio
async def test_check_statestore_healthy():
    """State store returning data should report healthy."""
    mock_store = AsyncMock()
    mock_store.retrieve = AsyncMock(return_value=["run-1", "run-2"])

    result = await check_statestore(mock_store)
    assert result.status == DependencyStatus.healthy
    assert result.name == "statestore"


@pytest.mark.anyio
async def test_check_statestore_key_not_found_is_healthy():
    """State store raising KeyError (no runs exist) should still report healthy."""
    mock_store = AsyncMock()
    mock_store.retrieve = AsyncMock(side_effect=KeyError("Key runs not found"))

    result = await check_statestore(mock_store)
    assert result.status == DependencyStatus.healthy


@pytest.mark.anyio
async def test_check_statestore_unhealthy():
    """State store raising RuntimeError should report unhealthy."""
    mock_store = AsyncMock()
    mock_store.retrieve = AsyncMock(side_effect=RuntimeError("State store is not configured"))

    result = await check_statestore(mock_store)
    assert result.status == DependencyStatus.unhealthy
    assert "not configured" in result.detail


@pytest.mark.anyio
async def test_check_statestore_timeout():
    """State store exceeding timeout should report unhealthy."""
    mock_store = AsyncMock()
    mock_store.retrieve = AsyncMock(side_effect=asyncio.TimeoutError())

    result = await check_statestore(mock_store)
    assert result.status == DependencyStatus.unhealthy
    assert result.latency_ms >= 0


@pytest.mark.anyio
async def test_check_service_healthy():
    """Service invocation returning 200 should report healthy."""
    mock_cls = _mock_aiohttp_session(resp_status=200)

    with patch("vibe_server.health.aiohttp.ClientSession", mock_cls):
        result = await check_service("terravibes-orchestrator")
    assert result.status == DependencyStatus.healthy
    assert result.name == "terravibes-orchestrator"


@pytest.mark.anyio
async def test_check_service_healthy_501():
    """Service returning 501 (no root handler) should still report healthy."""
    mock_cls = _mock_aiohttp_session(resp_status=501)

    with patch("vibe_server.health.aiohttp.ClientSession", mock_cls):
        result = await check_service("terravibes-cache")
    assert result.status == DependencyStatus.healthy


@pytest.mark.anyio
async def test_check_service_unhealthy_502():
    """Service returning 502 (gateway error) should report unhealthy."""
    mock_cls = _mock_aiohttp_session(resp_status=502)

    with patch("vibe_server.health.aiohttp.ClientSession", mock_cls):
        result = await check_service("terravibes-cache")
    assert result.status == DependencyStatus.unhealthy


@pytest.mark.anyio
async def test_check_service_connection_refused():
    """Service unreachable should report unhealthy."""
    mock_cls = _mock_aiohttp_session(get_side_effect=ConnectionError("Connection refused"))

    with patch("vibe_server.health.aiohttp.ClientSession", mock_cls):
        result = await check_service("terravibes-data-ops")
    assert result.status == DependencyStatus.unhealthy


@pytest.mark.anyio
async def test_check_dapr_sidecar_timeout():
    """Dapr sidecar timeout should report unhealthy."""
    mock_cls = _mock_aiohttp_session(get_side_effect=asyncio.TimeoutError())

    with patch("vibe_server.health.aiohttp.ClientSession", mock_cls):
        result = await check_dapr_sidecar()
    assert result.status == DependencyStatus.unhealthy
    assert result.latency_ms >= 0


@pytest.mark.anyio
async def test_check_service_timeout():
    """Service invocation timeout should report unhealthy."""
    mock_cls = _mock_aiohttp_session(get_side_effect=asyncio.TimeoutError())

    with patch("vibe_server.health.aiohttp.ClientSession", mock_cls):
        result = await check_service("terravibes-orchestrator")
    assert result.status == DependencyStatus.unhealthy
    assert result.latency_ms >= 0


@pytest.mark.anyio
async def test_get_detailed_health_all_healthy():
    """Detailed health returns 'healthy' when all dependencies pass."""
    healthy_dep = MagicMock()
    healthy_dep.status = DependencyStatus.healthy
    healthy_dep.name = "mock"
    healthy_dep.latency_ms = 1.0
    healthy_dep.detail = ""

    with patch("vibe_server.health.check_dapr_sidecar", return_value=healthy_dep), \
         patch("vibe_server.health.check_statestore", return_value=healthy_dep), \
         patch("vibe_server.health.check_service", return_value=healthy_dep):
        mock_store = AsyncMock()
        report = await get_detailed_health(mock_store)

    assert report.status == "healthy"
    assert len(report.dependencies) == 5  # dapr + statestore + 3 services


@pytest.mark.anyio
async def test_get_detailed_health_degraded():
    """Detailed health returns 'degraded' when a service (not Dapr) is unhealthy."""
    healthy_dep = MagicMock()
    healthy_dep.status = DependencyStatus.healthy
    healthy_dep.name = "mock"
    healthy_dep.latency_ms = 1.0
    healthy_dep.detail = ""

    unhealthy_dep = MagicMock()
    unhealthy_dep.status = DependencyStatus.unhealthy
    unhealthy_dep.name = "terravibes-cache"
    unhealthy_dep.latency_ms = 2000.0
    unhealthy_dep.detail = "timeout"

    async def mock_check_service(name):
        if name == "terravibes-cache":
            return unhealthy_dep
        return healthy_dep

    with patch("vibe_server.health.check_dapr_sidecar", return_value=healthy_dep), \
         patch("vibe_server.health.check_statestore", return_value=healthy_dep), \
         patch("vibe_server.health.check_service", side_effect=mock_check_service):
        mock_store = AsyncMock()
        report = await get_detailed_health(mock_store)

    assert report.status == "degraded"


@pytest.mark.anyio
async def test_get_detailed_health_unhealthy_dapr_down():
    """Detailed health returns 'unhealthy' when Dapr sidecar is down."""
    unhealthy_dapr = MagicMock()
    unhealthy_dapr.status = DependencyStatus.unhealthy
    unhealthy_dapr.name = "dapr"
    unhealthy_dapr.latency_ms = 0
    unhealthy_dapr.detail = "Connection refused"

    healthy_dep = MagicMock()
    healthy_dep.status = DependencyStatus.healthy
    healthy_dep.name = "mock"
    healthy_dep.latency_ms = 1.0
    healthy_dep.detail = ""

    with patch("vibe_server.health.check_dapr_sidecar", return_value=unhealthy_dapr), \
         patch("vibe_server.health.check_statestore", return_value=healthy_dep), \
         patch("vibe_server.health.check_service", return_value=healthy_dep):
        mock_store = AsyncMock()
        report = await get_detailed_health(mock_store)

    assert report.status == "unhealthy"


@pytest.mark.anyio
async def test_get_readiness_healthy():
    """Readiness returns True when Dapr and state store are healthy."""
    healthy_dep = MagicMock()
    healthy_dep.status = DependencyStatus.healthy

    with patch("vibe_server.health.check_dapr_sidecar", return_value=healthy_dep), \
         patch("vibe_server.health.check_statestore", return_value=healthy_dep):
        mock_store = AsyncMock()
        result = await get_readiness(mock_store)

    assert result is True


@pytest.mark.anyio
async def test_get_readiness_unhealthy():
    """Readiness returns False when state store is unhealthy."""
    healthy_dep = MagicMock()
    healthy_dep.status = DependencyStatus.healthy

    unhealthy_dep = MagicMock()
    unhealthy_dep.status = DependencyStatus.unhealthy

    with patch("vibe_server.health.check_dapr_sidecar", return_value=healthy_dep), \
         patch("vibe_server.health.check_statestore", return_value=unhealthy_dep):
        mock_store = AsyncMock()
        result = await get_readiness(mock_store)

    assert result is False


@pytest.mark.anyio
async def test_get_detailed_health_gather_exception():
    """Detailed health handles unexpected exceptions from gather defensively."""
    healthy_dep = MagicMock()
    healthy_dep.status = DependencyStatus.healthy
    healthy_dep.name = "mock"
    healthy_dep.latency_ms = 1.0
    healthy_dep.detail = ""

    async def exploding_dapr():
        raise RuntimeError("unexpected kaboom")

    with patch("vibe_server.health.check_dapr_sidecar", side_effect=exploding_dapr), \
         patch("vibe_server.health.check_statestore", return_value=healthy_dep), \
         patch("vibe_server.health.check_service", return_value=healthy_dep):
        mock_store = AsyncMock()
        report = await get_detailed_health(mock_store)

    assert report.status == "unhealthy"  # dapr is down
    dapr_result = report.dependencies[0]
    assert dapr_result.name == "dapr"
    assert dapr_result.status == DependencyStatus.unhealthy
    assert "kaboom" in dapr_result.detail


@pytest.mark.anyio
async def test_get_readiness_exception_returns_false():
    """Readiness returns False when an unexpected exception occurs."""
    with patch("vibe_server.health.check_dapr_sidecar", side_effect=RuntimeError("crash")), \
         patch("vibe_server.health.check_statestore", side_effect=RuntimeError("crash")):
        mock_store = AsyncMock()
        result = await get_readiness(mock_store)

    assert result is False
