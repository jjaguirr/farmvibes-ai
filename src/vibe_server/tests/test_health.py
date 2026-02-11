# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vibe_dev.testing import anyio_backend  # noqa: F401


def _mock_response(status: int) -> MagicMock:
    resp = MagicMock()
    resp.status = status
    resp.__aenter__ = AsyncMock(return_value=resp)
    resp.__aexit__ = AsyncMock(return_value=False)
    return resp


def _mock_session(side_effects):
    session = MagicMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    session.get = MagicMock(side_effect=side_effects)
    return session


# ---------------------------------------------------------------------------
# health_check() unit tests
# ---------------------------------------------------------------------------


@pytest.mark.anyio
@patch("vibe_server.server.aiohttp.ClientSession")
async def test_health_all_ok(mock_session_cls):
    from vibe_server.href_handler import LocalHrefHandler
    from vibe_server.server import TerravibesProvider

    mock_session_cls.return_value = _mock_session(
        [_mock_response(200), _mock_response(404), _mock_response(200)]
    )

    provider = TerravibesProvider(LocalHrefHandler("/tmp"))
    result = await provider.health_check()

    assert result["status"] == "healthy"
    assert result["checks"]["dapr"] == "ok"
    assert result["checks"]["state_store"] == "ok"
    assert result["checks"]["pubsub"] == "ok"


@pytest.mark.anyio
@patch("vibe_server.server.aiohttp.ClientSession")
async def test_health_redis_down(mock_session_cls):
    from vibe_server.href_handler import LocalHrefHandler
    from vibe_server.server import TerravibesProvider

    mock_session_cls.return_value = _mock_session(
        [_mock_response(200), _mock_response(500), _mock_response(200)]
    )

    provider = TerravibesProvider(LocalHrefHandler("/tmp"))
    result = await provider.health_check()

    assert result["status"] == "degraded"
    assert result["checks"]["dapr"] == "ok"
    assert result["checks"]["state_store"].startswith("error")
    assert result["checks"]["pubsub"] == "ok"


@pytest.mark.anyio
@patch("vibe_server.server.aiohttp.ClientSession")
async def test_health_connection_error(mock_session_cls):
    from vibe_server.href_handler import LocalHrefHandler
    from vibe_server.server import TerravibesProvider

    session = MagicMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    session.get = MagicMock(side_effect=Exception("connection refused"))
    mock_session_cls.return_value = session

    provider = TerravibesProvider(LocalHrefHandler("/tmp"))
    result = await provider.health_check()

    assert result["status"] == "degraded"
    for check in result["checks"].values():
        assert check.startswith("error")


# ---------------------------------------------------------------------------
# Endpoint tests
# ---------------------------------------------------------------------------


@pytest.mark.anyio
@patch("vibe_server.server.TerravibesProvider.health_check")
async def test_liveness_always_200(mock_health):
    """Liveness never calls health_check and always returns 200."""
    from httpx import ASGITransport, AsyncClient

    from vibe_server.href_handler import LocalHrefHandler
    from vibe_server.server import TerravibesAPI

    api = TerravibesAPI(href_handler=LocalHrefHandler("/tmp"))
    async with AsyncClient(
        transport=ASGITransport(app=api.versioned_wrapper), base_url="http://test"
    ) as client:
        resp = await client.get("/v0/liveness")

    assert resp.status_code == 200
    assert resp.json()["status"] == "alive"
    mock_health.assert_not_called()


@pytest.mark.anyio
@patch("vibe_server.server.TerravibesProvider.health_check")
async def test_readiness_healthy_returns_200(mock_health):
    mock_health.return_value = {
        "status": "healthy",
        "checks": {"dapr": "ok", "state_store": "ok", "pubsub": "ok"},
    }
    from httpx import ASGITransport, AsyncClient

    from vibe_server.href_handler import LocalHrefHandler
    from vibe_server.server import TerravibesAPI

    api = TerravibesAPI(href_handler=LocalHrefHandler("/tmp"))
    async with AsyncClient(
        transport=ASGITransport(app=api.versioned_wrapper), base_url="http://test"
    ) as client:
        resp = await client.get("/v0/readiness")

    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


@pytest.mark.anyio
@patch("vibe_server.server.TerravibesProvider.health_check")
async def test_readiness_degraded_returns_503(mock_health):
    mock_health.return_value = {
        "status": "degraded",
        "checks": {"dapr": "ok", "state_store": "error: HTTP 500", "pubsub": "ok"},
    }
    from httpx import ASGITransport, AsyncClient

    from vibe_server.href_handler import LocalHrefHandler
    from vibe_server.server import TerravibesAPI

    api = TerravibesAPI(href_handler=LocalHrefHandler("/tmp"))
    async with AsyncClient(
        transport=ASGITransport(app=api.versioned_wrapper), base_url="http://test"
    ) as client:
        resp = await client.get("/v0/readiness")

    assert resp.status_code == 503


@pytest.mark.anyio
@patch("vibe_server.server.TerravibesProvider.health_check")
async def test_health_and_status_are_aliases(mock_health):
    """/health and /status return the same payload."""
    mock_health.return_value = {
        "status": "healthy",
        "checks": {"dapr": "ok", "state_store": "ok", "pubsub": "ok"},
    }
    from httpx import ASGITransport, AsyncClient

    from vibe_server.href_handler import LocalHrefHandler
    from vibe_server.server import TerravibesAPI

    api = TerravibesAPI(href_handler=LocalHrefHandler("/tmp"))
    async with AsyncClient(
        transport=ASGITransport(app=api.versioned_wrapper), base_url="http://test"
    ) as client:
        health_resp = await client.get("/v0/health")
        status_resp = await client.get("/v0/status")

    assert health_resp.status_code == 200
    assert status_resp.status_code == 200
    assert health_resp.json() == status_resp.json()
