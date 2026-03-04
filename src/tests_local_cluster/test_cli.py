import subprocess

import pytest

pytestmark = pytest.mark.fast


def run_cli(*args: str, timeout: float = 30.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["farmvibes-ai", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


class TestCliStatus:
    def test_runs_successfully(self):
        result = run_cli("local", "status")
        # Exit code 0 = all good, 1 = degraded/issues detected — both are valid
        assert result.returncode in (0, 1), (
            f"Unexpected exit code {result.returncode}.\nstderr: {result.stderr}"
        )

    def test_output_contains_url(self):
        result = run_cli("local", "status")
        assert "http" in result.stdout.lower() or "http" in result.stderr.lower(), (
            f"Expected URL in output.\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )

    def test_output_contains_cluster_info(self):
        result = run_cli("local", "status")
        # Should show cluster name and service information
        assert "cluster" in result.stdout.lower() or "service" in result.stdout.lower(), (
            f"Expected cluster info in output.\nstdout: {result.stdout}"
        )


class TestCliHealth:
    def test_runs_without_crash(self):
        result = run_cli("local", "health")
        # Exit code: 0=healthy, 1=degraded, 2=down — all are valid outcomes
        assert result.returncode in (0, 1, 2), (
            f"Unexpected exit code {result.returncode}.\nstderr: {result.stderr}"
        )
