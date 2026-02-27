# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from unittest.mock import mock_open, patch

import pytest
from vibe_common.resources import MemoryInfo, get_memory_info, is_oom_exitcode


def test_memory_info_computed_properties():
    info = MemoryInfo(limit_bytes=1024 * 1024 * 1024, usage_bytes=512 * 1024 * 1024)
    assert info.usage_mb == 512.0
    assert info.limit_mb == 1024.0
    assert info.usage_fraction == pytest.approx(0.5)


def test_memory_info_no_limit():
    info = MemoryInfo(limit_bytes=None, usage_bytes=100 * 1024 * 1024)
    assert info.limit_mb is None
    assert info.usage_fraction is None


def test_is_oom_exitcode_negative_sigkill():
    assert is_oom_exitcode(-9)


def test_is_oom_exitcode_shell_convention():
    assert is_oom_exitcode(137)


def test_is_oom_exitcode_normal_exit():
    assert not is_oom_exitcode(0)
    assert not is_oom_exitcode(1)
    assert not is_oom_exitcode(-2)


def test_get_memory_info_cgroup_v2():
    def fake_exists(path):
        return path == "/sys/fs/cgroup/memory.max" or path == "/sys/fs/cgroup/memory.current"

    file_contents = {
        "/sys/fs/cgroup/memory.max": "2147483648\n",
        "/sys/fs/cgroup/memory.current": "536870912\n",
    }

    def fake_open(path, *a, **kw):
        return mock_open(read_data=file_contents[path])()

    with patch("os.path.exists", side_effect=fake_exists), \
         patch("builtins.open", side_effect=fake_open):
        info = get_memory_info()
        assert info.limit_bytes == 2147483648
        assert info.usage_bytes == 536870912


def test_get_memory_info_cgroup_v2_unlimited():
    def fake_exists(path):
        return path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current")

    file_contents = {
        "/sys/fs/cgroup/memory.max": "max\n",
        "/sys/fs/cgroup/memory.current": "100000\n",
    }

    def fake_open(path, *a, **kw):
        return mock_open(read_data=file_contents[path])()

    with patch("os.path.exists", side_effect=fake_exists), \
         patch("builtins.open", side_effect=fake_open):
        info = get_memory_info()
        assert info.limit_bytes is None
        assert info.usage_bytes == 100000


def test_get_memory_info_cgroup_v1():
    def fake_exists(path):
        return path in (
            "/sys/fs/cgroup/memory/memory.limit_in_bytes",
            "/sys/fs/cgroup/memory/memory.usage_in_bytes",
        )

    file_contents = {
        "/sys/fs/cgroup/memory/memory.limit_in_bytes": "1073741824\n",
        "/sys/fs/cgroup/memory/memory.usage_in_bytes": "268435456\n",
    }

    def fake_open(path, *a, **kw):
        return mock_open(read_data=file_contents[path])()

    with patch("os.path.exists", side_effect=fake_exists), \
         patch("builtins.open", side_effect=fake_open):
        info = get_memory_info()
        assert info.limit_bytes == 1073741824
        assert info.usage_bytes == 268435456


def test_get_memory_info_cgroup_v1_huge_limit_treated_as_unlimited():
    def fake_exists(path):
        return path in (
            "/sys/fs/cgroup/memory/memory.limit_in_bytes",
            "/sys/fs/cgroup/memory/memory.usage_in_bytes",
        )

    file_contents = {
        "/sys/fs/cgroup/memory/memory.limit_in_bytes": "9223372036854771712\n",
        "/sys/fs/cgroup/memory/memory.usage_in_bytes": "1000000\n",
    }

    def fake_open(path, *a, **kw):
        return mock_open(read_data=file_contents[path])()

    with patch("os.path.exists", side_effect=fake_exists), \
         patch("builtins.open", side_effect=fake_open):
        info = get_memory_info()
        assert info.limit_bytes is None


def test_get_memory_info_proc_fallback():
    proc_status = (
        "Name:\tworker\n"
        "VmPeak:\t  200000 kB\n"
        "VmRSS:\t  150000 kB\n"
        "VmSwap:\t       0 kB\n"
    )
    with patch("os.path.exists", return_value=False), \
         patch("builtins.open", mock_open(read_data=proc_status)):
        info = get_memory_info()
        assert info.limit_bytes is None
        assert info.usage_bytes == 150000 * 1024


def test_get_memory_info_never_raises_on_filesystem_error():
    # os.path.exists can raise PermissionError during container teardown
    with patch("os.path.exists", side_effect=PermissionError("cgroup fs gone")):
        info = get_memory_info()
        assert info.limit_bytes is None
        assert info.usage_bytes == 0
