import os
import signal
import tempfile
import time

import pytest

from vibe_common.resource_monitor import CpuInfo, MemoryInfo, ResourceMonitor


@pytest.fixture
def fake_cgroup_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "memory.current"), "w") as f:
            f.write("3997696000\n")  # ~3.7GB
        with open(os.path.join(tmpdir, "memory.max"), "w") as f:
            f.write("4294967296\n")  # 4GB
        with open(os.path.join(tmpdir, "cpu.stat"), "w") as f:
            f.write(
                "usage_usec 1500000\nuser_usec 1000000\nsystem_usec 500000\n"
                "nr_periods 100\nnr_throttled 5\n"
            )
        yield tmpdir


# --- Memory reading ---


def test_get_memory_usage_from_cgroup(fake_cgroup_dir):
    monitor = ResourceMonitor(cgroup_path=fake_cgroup_dir)
    mem = monitor.get_memory_usage()
    assert mem is not None
    assert mem.current_bytes == 3997696000
    assert mem.limit_bytes == 4294967296
    assert abs(mem.usage_percent - 93.08) < 0.01


def test_get_memory_usage_no_cgroup_returns_none():
    monitor = ResourceMonitor(cgroup_path="/nonexistent/path")
    mem = monitor.get_memory_usage()
    assert mem is None


def test_get_memory_max_is_max_when_unlimited(fake_cgroup_dir):
    with open(os.path.join(fake_cgroup_dir, "memory.max"), "w") as f:
        f.write("max\n")
    monitor = ResourceMonitor(cgroup_path=fake_cgroup_dir)
    mem = monitor.get_memory_usage()
    assert mem is not None
    assert mem.limit_bytes == 0
    assert mem.usage_percent == 0.0


# --- OOM detection ---


def test_detect_oom_with_sigkill_and_high_memory(fake_cgroup_dir):
    monitor = ResourceMonitor(cgroup_path=fake_cgroup_dir)
    monitor.get_memory_usage()  # 93% usage
    assert monitor.detect_oom(exit_signal=signal.SIGKILL) is True


def test_detect_oom_with_sigkill_and_low_memory(fake_cgroup_dir):
    with open(os.path.join(fake_cgroup_dir, "memory.current"), "w") as f:
        f.write("100000000\n")  # ~2.3% of 4GB
    monitor = ResourceMonitor(cgroup_path=fake_cgroup_dir)
    monitor.get_memory_usage()
    assert monitor.detect_oom(exit_signal=signal.SIGKILL) is False


def test_detect_oom_without_sigkill(fake_cgroup_dir):
    monitor = ResourceMonitor(cgroup_path=fake_cgroup_dir)
    monitor.get_memory_usage()
    assert monitor.detect_oom(exit_signal=signal.SIGTERM) is False


def test_detect_oom_without_prior_memory_reading():
    """detect_oom returns False when get_memory_usage was never called."""
    monitor = ResourceMonitor(cgroup_path="/nonexistent")
    assert monitor.detect_oom(exit_signal=signal.SIGKILL) is False


def test_detect_oom_at_threshold_boundary(fake_cgroup_dir):
    """Memory at exactly 80% of limit should trigger OOM detection."""
    # Use round numbers to avoid float truncation: 800/1000 = 80.0% exactly
    with open(os.path.join(fake_cgroup_dir, "memory.current"), "w") as f:
        f.write("800000\n")
    with open(os.path.join(fake_cgroup_dir, "memory.max"), "w") as f:
        f.write("1000000\n")
    monitor = ResourceMonitor(cgroup_path=fake_cgroup_dir)
    mem = monitor.get_memory_usage()
    assert mem.usage_percent == 80.0
    assert monitor.detect_oom(exit_signal=signal.SIGKILL) is True


def test_detect_oom_just_below_threshold(fake_cgroup_dir):
    """Memory at 79.99% should NOT trigger OOM detection."""
    # 799900/1000000 = 79.99%
    with open(os.path.join(fake_cgroup_dir, "memory.max"), "w") as f:
        f.write("1000000\n")
    current = 799900
    with open(os.path.join(fake_cgroup_dir, "memory.current"), "w") as f:
        f.write(f"{current}\n")
    monitor = ResourceMonitor(cgroup_path=fake_cgroup_dir)
    monitor.get_memory_usage()
    assert monitor.detect_oom(exit_signal=signal.SIGKILL) is False


# --- OOM message formatting ---


def test_format_oom_message_includes_all_fields(fake_cgroup_dir):
    monitor = ResourceMonitor(cgroup_path=fake_cgroup_dir)
    monitor.get_memory_usage()
    msg = monitor.format_oom_message("download_sentinel_2", exit_signal=signal.SIGKILL)
    # Exact format: "Op 'X' killed (SIGKILL): memory usage 3.7GB / 4.0GB limit (93% at last sample)"
    assert msg.startswith("Op 'download_sentinel_2' killed (SIGKILL):")
    assert "3.7GB" in msg
    assert "4.0GB" in msg
    assert "93%" in msg


def test_format_oom_message_without_prior_reading():
    """When no memory data is available, message says so."""
    monitor = ResourceMonitor(cgroup_path="/nonexistent")
    msg = monitor.format_oom_message("some_op", exit_signal=signal.SIGKILL)
    assert "some_op" in msg
    assert "SIGKILL" in msg
    assert "unavailable" in msg


def test_format_oom_message_unlimited_memory(fake_cgroup_dir):
    """When memory.max is 'max' (unlimited), message says unavailable."""
    with open(os.path.join(fake_cgroup_dir, "memory.max"), "w") as f:
        f.write("max\n")
    monitor = ResourceMonitor(cgroup_path=fake_cgroup_dir)
    monitor.get_memory_usage()
    msg = monitor.format_oom_message("my_op", exit_signal=signal.SIGKILL)
    assert "unavailable" in msg


# --- CPU reading ---


def test_get_cpu_usage(fake_cgroup_dir):
    monitor = ResourceMonitor(cgroup_path=fake_cgroup_dir)
    cpu = monitor.get_cpu_usage()
    assert cpu is not None
    assert isinstance(cpu, CpuInfo)
    assert cpu.usage_usec == 1500000
    assert cpu.system_usec == 500000
    assert cpu.nr_periods == 100
    assert cpu.nr_throttled == 5


def test_get_cpu_usage_no_cgroup():
    monitor = ResourceMonitor(cgroup_path="/nonexistent/path")
    cpu = monitor.get_cpu_usage()
    assert cpu is None


# --- Periodic logging ---


def test_periodic_logging_produces_formatted_entries(fake_cgroup_dir, caplog):
    import logging

    monitor = ResourceMonitor(cgroup_path=fake_cgroup_dir)
    with caplog.at_level(logging.INFO):
        monitor.start_periodic_logging(interval_s=0.05)
        time.sleep(0.15)
        monitor.stop()
    resource_msgs = [r.message for r in caplog.records if "Resource usage" in r.message]
    assert len(resource_msgs) >= 1, "Expected at least one resource usage log entry"
    # Verify the log format: should contain memory=XXXMB/XXXMB and cpu_usage=XXXus
    sample = resource_msgs[0]
    assert "memory=" in sample
    assert "MB" in sample
    assert "cpu_usage=" in sample


def test_stop_is_idempotent(fake_cgroup_dir):
    monitor = ResourceMonitor(cgroup_path=fake_cgroup_dir)
    monitor.start_periodic_logging(interval_s=1.0)
    monitor.stop()
    monitor.stop()  # Should not raise


def test_stop_actually_stops_logging(fake_cgroup_dir, caplog):
    import logging

    monitor = ResourceMonitor(cgroup_path=fake_cgroup_dir)
    with caplog.at_level(logging.INFO):
        monitor.start_periodic_logging(interval_s=0.05)
        time.sleep(0.12)
        monitor.stop()
        count_at_stop = len([r for r in caplog.records if "Resource usage" in r.message])
        time.sleep(0.12)
        count_after_wait = len([r for r in caplog.records if "Resource usage" in r.message])
    assert count_at_stop == count_after_wait, "Logging continued after stop()"


# --- Cgroup file edge cases ---


def test_cgroup_warning_logged_once(caplog):
    import logging

    monitor = ResourceMonitor(cgroup_path="/nonexistent")
    with caplog.at_level(logging.INFO):
        monitor.get_memory_usage()
        monitor.get_memory_usage()
        monitor.get_cpu_usage()
    warning_msgs = [r for r in caplog.records if "not available" in r.message]
    assert len(warning_msgs) == 1, "Expected cgroup warning to be logged exactly once"
