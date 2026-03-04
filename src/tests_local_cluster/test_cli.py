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
    def test_exit_code_zero(self):
        result = run_cli("local", "status")
        assert result.returncode == 0, f"stderr: {result.stderr}"

    def test_output_contains_url(self):
        result = run_cli("local", "status")
        assert "http" in result.stdout.lower() or "http" in result.stderr.lower(), (
            f"Expected URL in output.\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )


class TestCliHealth:
    def test_runs_without_crash(self):
        result = run_cli("local", "health")
        # Exit code: 0=healthy, 1=degraded, 2=down — all are valid outcomes
        assert result.returncode in (0, 1, 2), (
            f"Unexpected exit code {result.returncode}.\nstderr: {result.stderr}"
        )
