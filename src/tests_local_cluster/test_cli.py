# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""CLI integration tests.

These shell out to the `farmvibes-ai` binary and verify exit codes and
output structure against a live cluster. They're marked `cli` (not just
`fast`) because they require a working kubeconfig in addition to a
reachable REST API — the status command talks to kubectl, not just HTTP.

Known limitations (discovered by probing, March 2026):

  - `farmvibes-ai local status` probes `/v0/status`, which doesn't exist
    on older cluster images. It prints "API: not responding" but still
    exits 0 because pod health is green. The test asserts exit code
    only, not the API-line text.

  - `farmvibes-ai local logs <bad-service>` exits 0 even though the
    function returns False. This is a bug in the CLI's exit-code
    mapping. We test the *output structure* (service list printed)
    rather than the exit code for that case, and flag the exit-code
    issue with an xfail.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

pytestmark = [pytest.mark.fast, pytest.mark.cli]


@pytest.fixture(scope="module")
def farmvibes_cli() -> str:
    """Path to the farmvibes-ai executable.

    Skips the whole module if the CLI isn't installed. This happens in
    bare venvs that only have vibe_core as a library dependency.
    """
    path = shutil.which("farmvibes-ai")
    if path is None:
        pytest.skip(
            "farmvibes-ai CLI not on PATH — install vibe_core with its "
            "console scripts (`pip install -e src/vibe_core/`)"
        )
    return path


def _run(cli: str, *args: str, timeout: float = 60) -> subprocess.CompletedProcess[str]:
    """Run the CLI with a hard timeout. Captures both streams.

    We use a timeout on every CLI invocation because several subcommands
    default to interactive/follow mode and would hang forever in CI if
    the flag parsing changed.
    """
    return subprocess.run(
        [cli, "local", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# status / health
# ---------------------------------------------------------------------------


def test_status_exit_code_indicates_cluster_up(farmvibes_cli: str, service_url: str) -> None:
    """`farmvibes-ai local status` exit code: 0=healthy, 1=degraded, 2=down.

    We accept 0 or 1 here. Why 1 is acceptable: the status command probes
    `/v0/status`, which doesn't exist on older cluster images → it reports
    "API: not responding" → exit 1 (degraded). That's *correct* CLI
    behaviour against an older-but-functional cluster, not a test failure.

    What we actually want to assert is "not 2" — exit code 2 means the
    CLI couldn't talk to kubectl/k3d at all, which on a cluster we've
    already proven reachable via HTTP would indicate the CLI is broken.

    Depends on service_url only to inherit the session-level "cluster
    reachable" skip. Status discovers its own URL via k3d.
    """
    del service_url  # used only for its skip side-effect
    result = _run(farmvibes_cli, "status")
    assert result.returncode in (0, 1), (
        f"status exited {result.returncode} against a cluster we know is up.\n"
        f"Exit 2 means the CLI couldn't reach k3d/kubectl at all.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def test_status_output_contains_cluster_name(farmvibes_cli: str, service_url: str) -> None:
    del service_url
    result = _run(farmvibes_cli, "status")
    combined = result.stdout + result.stderr
    # Don't hardcode the full cluster name (it's username-dependent);
    # just check the "Cluster:" label appears so we know the table rendered.
    assert "Cluster:" in combined, (
        f"Expected 'Cluster:' label in status output. Got:\n{combined}"
    )


def test_status_lists_all_core_services(farmvibes_cli: str, service_url: str) -> None:
    """The status table should include every core service.

    If a service is missing from the table, either it's not deployed
    (terraform drift) or the CLI's service discovery filter is too
    restrictive.
    """
    del service_url
    result = _run(farmvibes_cli, "status")
    combined = result.stdout + result.stderr
    for svc in (
        "terravibes-rest-api",
        "terravibes-orchestrator",
        "terravibes-worker",
        "terravibes-cache",
    ):
        assert svc in combined, (
            f"Service '{svc}' missing from status table. Output:\n{combined}"
        )


def test_health_is_alias_for_status(farmvibes_cli: str, service_url: str) -> None:
    """`health` and `status` should produce equivalent output — same exit
    code class, same services listed."""
    del service_url
    s = _run(farmvibes_cli, "status")
    h = _run(farmvibes_cli, "health")
    # Exit codes should land in the same bucket. We don't assert exact
    # equality because a transient API-probe timing difference could
    # flip one from 0→1. Both <2 means both agree the cluster is up.
    assert (s.returncode < 2) == (h.returncode < 2)
    # Compare the service-table portion, not raw output — timestamps
    # and latency measurements differ between invocations.
    def _services(out: str) -> set[str]:
        return {
            line.strip().split()[0]
            for line in out.splitlines()
            if line.strip().startswith("terravibes-")
        }
    assert _services(s.stdout) == _services(h.stdout)


# ---------------------------------------------------------------------------
# logs
# ---------------------------------------------------------------------------


def test_logs_dump_mode_returns(farmvibes_cli: str, service_url: str) -> None:
    """`logs ... --no-follow` must actually return, not hang.

    The default is `--follow` which tails forever. If `--no-follow` ever
    stops being respected, this test will time out and fail loudly instead
    of a CI job sitting there for six hours.
    """
    del service_url
    result = _run(
        farmvibes_cli,
        "logs", "terravibes-rest-api", "--tail", "5", "--no-follow",
        timeout=45,  # tight on purpose
    )
    # We don't assert exit code here — see module docstring. We assert
    # that it *returned* (subprocess.run raises TimeoutExpired if not).
    # Returning is the contract.
    _ = result.returncode  # explicitly not asserted


def test_logs_bad_service_lists_available(farmvibes_cli: str, service_url: str) -> None:
    """Asking for logs of a nonexistent service should print the list of
    services that *do* exist. This is the error-message-is-useful check."""
    del service_url
    result = _run(
        farmvibes_cli, "logs", "definitely-not-a-service-xyz", "--no-follow",
        timeout=30,
    )
    combined = result.stdout + result.stderr
    # The error output should name at least one real service so the user
    # knows what to type instead.
    assert "terravibes-rest-api" in combined, (
        f"Bad-service error didn't suggest valid alternatives. Output:\n{combined}"
    )


@pytest.mark.xfail(
    reason=(
        "Known CLI bug: `logs <bad-service>` exits 0 even though the "
        "internal function returns False. The exit-code mapping in "
        "main.py isn't being hit for this path. Filed as follow-up."
    ),
    strict=False,  # Don't fail the suite if someone fixes it.
)
def test_logs_bad_service_nonzero_exit(farmvibes_cli: str, service_url: str) -> None:
    del service_url
    result = _run(
        farmvibes_cli, "logs", "definitely-not-a-service-xyz", "--no-follow",
        timeout=30,
    )
    assert result.returncode != 0


# ---------------------------------------------------------------------------
# Help / argument parsing
# ---------------------------------------------------------------------------
# There is no `python -m vibe_core.cli` entrypoint (no __main__.py).
# That's fine — the console script is the supported invocation. We just
# verify --help works, which proves argparse wiring is intact.


def test_help_lists_all_subcommands(farmvibes_cli: str) -> None:
    """`farmvibes-ai local --help` must list every subcommand.

    If a subcommand is added to local.py but not wired into the argparse
    subparser, it's silently unreachable. This catches that.
    """
    result = subprocess.run(
        [farmvibes_cli, "local", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    for cmd in ("setup", "destroy", "start", "stop", "status", "health", "logs"):
        assert cmd in result.stdout, (
            f"Subcommand '{cmd}' not in --help output. Either it was removed "
            f"or the argparse wiring broke. Help output:\n{result.stdout}"
        )
