# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Health check logic for the FarmVibes REST API.

Provides liveness, readiness, and detailed dependency health checks.
Dependency probes use raw aiohttp with short timeouts instead of
VibeDaprClient (which has aggressive retry logic unsuitable for probes).
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List

import aiohttp
from dapr.conf import settings

from vibe_common.constants import RUNS_KEY
from vibe_common.statestore import StateStore

LOGGER = logging.getLogger(__name__)
HEALTH_CHECK_TIMEOUT_S = 2.0

DAPR_HEALTHZ_URL = (
    f"http://{settings.DAPR_RUNTIME_HOST}:{settings.DAPR_HTTP_PORT}/v1.0/healthz"
)
DAPR_INVOKE_URL = (
    f"http://{settings.DAPR_RUNTIME_HOST}:{settings.DAPR_HTTP_PORT}"
    "/v1.0/invoke/{service}/method/"
)

CHECKED_SERVICES = [
    "terravibes-orchestrator",
    "terravibes-cache",
    "terravibes-data-ops",
]


class DependencyStatus(str, Enum):
    healthy = "healthy"
    unhealthy = "unhealthy"


@dataclass
class DependencyHealth:
    name: str
    status: DependencyStatus
    latency_ms: float
    detail: str = ""


@dataclass
class HealthReport:
    status: str  # "healthy", "degraded", or "unhealthy"
    dependencies: List[DependencyHealth] = field(default_factory=list)


async def check_dapr_sidecar() -> DependencyHealth:
    """Check Dapr sidecar health via its /v1.0/healthz endpoint."""
    start = time.perf_counter()
    try:
        timeout = aiohttp.ClientTimeout(total=HEALTH_CHECK_TIMEOUT_S)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(DAPR_HEALTHZ_URL) as resp:
                latency = (time.perf_counter() - start) * 1000
                if 200 <= resp.status < 300:
                    return DependencyHealth(
                        name="dapr", status=DependencyStatus.healthy, latency_ms=latency
                    )
                return DependencyHealth(
                    name="dapr",
                    status=DependencyStatus.unhealthy,
                    latency_ms=latency,
                    detail=f"HTTP {resp.status}",
                )
    except Exception as exc:
        latency = (time.perf_counter() - start) * 1000
        return DependencyHealth(
            name="dapr",
            status=DependencyStatus.unhealthy,
            latency_ms=latency,
            detail=str(exc),
        )


async def check_statestore(state_store: StateStore) -> DependencyHealth:
    """Check state store (Redis) reachability via Dapr state API.

    A KeyError (no runs key) is expected and counts as healthy — it proves
    the Dapr-to-Redis path works. Any other exception means unhealthy.
    """
    start = time.perf_counter()
    try:
        await asyncio.wait_for(state_store.retrieve(RUNS_KEY), timeout=HEALTH_CHECK_TIMEOUT_S)
        latency = (time.perf_counter() - start) * 1000
        return DependencyHealth(
            name="statestore", status=DependencyStatus.healthy, latency_ms=latency
        )
    except KeyError:
        latency = (time.perf_counter() - start) * 1000
        return DependencyHealth(
            name="statestore", status=DependencyStatus.healthy, latency_ms=latency
        )
    except Exception as exc:
        latency = (time.perf_counter() - start) * 1000
        return DependencyHealth(
            name="statestore",
            status=DependencyStatus.unhealthy,
            latency_ms=latency,
            detail=str(exc),
        )


async def check_service(service_name: str) -> DependencyHealth:
    """Check a Dapr service via service invocation (GET root method)."""
    start = time.perf_counter()
    url = DAPR_INVOKE_URL.format(service=service_name)
    try:
        timeout = aiohttp.ClientTimeout(total=HEALTH_CHECK_TIMEOUT_S)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as resp:
                latency = (time.perf_counter() - start) * 1000
                if resp.status not in (502, 503, 504):
                    return DependencyHealth(
                        name=service_name,
                        status=DependencyStatus.healthy,
                        latency_ms=latency,
                    )
                return DependencyHealth(
                    name=service_name,
                    status=DependencyStatus.unhealthy,
                    latency_ms=latency,
                    detail=f"HTTP {resp.status}",
                )
    except Exception as exc:
        latency = (time.perf_counter() - start) * 1000
        return DependencyHealth(
            name=service_name,
            status=DependencyStatus.unhealthy,
            latency_ms=latency,
            detail=str(exc),
        )


async def get_detailed_health(state_store: StateStore) -> HealthReport:
    """Run all dependency checks concurrently and return a full health report."""
    check_names = ["dapr", "statestore"] + CHECKED_SERVICES
    checks = [
        check_dapr_sidecar(),
        check_statestore(state_store),
    ] + [check_service(svc) for svc in CHECKED_SERVICES]

    results: List[DependencyHealth] = []
    raw = await asyncio.gather(*checks, return_exceptions=True)
    for name, result in zip(check_names, raw):
        if isinstance(result, Exception):
            # Should not happen since each check catches exceptions internally,
            # but handle defensively.
            results.append(
                DependencyHealth(
                    name=name,
                    status=DependencyStatus.unhealthy,
                    latency_ms=0,
                    detail=str(result),
                )
            )
        else:
            results.append(result)

    unhealthy = [r for r in results if r.status == DependencyStatus.unhealthy]
    dapr_down = any(r.name == "dapr" and r.status == DependencyStatus.unhealthy for r in results)

    if not unhealthy:
        overall = "healthy"
    elif dapr_down:
        overall = "unhealthy"
    else:
        overall = "degraded"

    return HealthReport(status=overall, dependencies=results)


async def get_readiness(state_store: StateStore) -> bool:
    """Check if the service is ready to accept requests.

    Requires both Dapr sidecar and state store to be healthy.
    """
    try:
        dapr_check, store_check = await asyncio.gather(
            check_dapr_sidecar(),
            check_statestore(state_store),
        )
        return (
            dapr_check.status == DependencyStatus.healthy
            and store_check.status == DependencyStatus.healthy
        )
    except Exception:
        return False
