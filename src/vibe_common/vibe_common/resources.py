# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Container resource introspection.

Reads cgroup memory limit/usage directly. No psutil dependency — cgroup files
are plain text integers, and psutil is not a vibe_common dep.

cgroup v2 (current Kubernetes):
    /sys/fs/cgroup/memory.max      — limit in bytes, or the literal "max"
    /sys/fs/cgroup/memory.current  — current usage in bytes

cgroup v1 (older clusters):
    /sys/fs/cgroup/memory/memory.limit_in_bytes  — giant sentinel if unlimited
    /sys/fs/cgroup/memory/memory.usage_in_bytes

Fallback (not containerized): /proc/self/status VmRSS line.
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional

__all__ = ["MemoryInfo", "get_memory_info", "is_oom_exitcode"]

logger = logging.getLogger(__name__)

# cgroup v1 reports a near-2^63 value when unlimited. Anything bigger than
# this threshold is treated as "no limit." 2^62 = 4 exabytes — if a real
# limit ever approaches this we have bigger problems.
_UNLIMITED_THRESHOLD = 1 << 62

_CGROUP_V2_LIMIT = "/sys/fs/cgroup/memory.max"
_CGROUP_V2_USAGE = "/sys/fs/cgroup/memory.current"
_CGROUP_V1_LIMIT = "/sys/fs/cgroup/memory/memory.limit_in_bytes"
_CGROUP_V1_USAGE = "/sys/fs/cgroup/memory/memory.usage_in_bytes"
_PROC_STATUS = "/proc/self/status"


@dataclass
class MemoryInfo:
    limit_bytes: Optional[int]
    usage_bytes: int

    @property
    def usage_mb(self) -> float:
        return self.usage_bytes / (1024 * 1024)

    @property
    def limit_mb(self) -> Optional[float]:
        if self.limit_bytes is None:
            return None
        return self.limit_bytes / (1024 * 1024)

    @property
    def usage_fraction(self) -> Optional[float]:
        if self.limit_bytes is None or self.limit_bytes == 0:
            return None
        return self.usage_bytes / self.limit_bytes

    def __str__(self) -> str:
        if self.limit_mb is not None:
            return f"{self.usage_mb:.0f}MB / {self.limit_mb:.0f}MB ({self.usage_fraction:.0%})"
        return f"{self.usage_mb:.0f}MB (no limit)"


def _read_int(path: str) -> Optional[int]:
    try:
        with open(path) as f:
            raw = f.read().strip()
        if raw == "max":
            return None
        value = int(raw)
        if value >= _UNLIMITED_THRESHOLD:
            return None
        return value
    except (OSError, ValueError) as e:
        logger.debug(f"Couldn't read {path}: {e}")
        return None


def _read_proc_rss() -> int:
    """Parse VmRSS from /proc/self/status. Returns 0 if unavailable."""
    try:
        with open(_PROC_STATUS) as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    return int(parts[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return 0


def get_memory_info() -> MemoryInfo:
    """Returns current memory usage and limit for this container/process.

    Prefers cgroup v2, falls back to v1, falls back to /proc/self/status RSS
    with no limit. Never raises — returns MemoryInfo(None, 0) in the worst case.
    """
    try:
        if os.path.exists(_CGROUP_V2_LIMIT) and os.path.exists(_CGROUP_V2_USAGE):
            limit = _read_int(_CGROUP_V2_LIMIT)
            usage = _read_int(_CGROUP_V2_USAGE) or 0
            return MemoryInfo(limit_bytes=limit, usage_bytes=usage)

        if os.path.exists(_CGROUP_V1_LIMIT) and os.path.exists(_CGROUP_V1_USAGE):
            limit = _read_int(_CGROUP_V1_LIMIT)
            usage = _read_int(_CGROUP_V1_USAGE) or 0
            return MemoryInfo(limit_bytes=limit, usage_bytes=usage)

        return MemoryInfo(limit_bytes=None, usage_bytes=_read_proc_rss())
    except Exception as e:
        logger.debug(f"get_memory_info failed: {e}")
        return MemoryInfo(limit_bytes=None, usage_bytes=0)


def is_oom_exitcode(exitcode: int) -> bool:
    """True if the exit code indicates the process was SIGKILLed.

    In a container, SIGKILL from the kernel almost always means the OOM killer.
    Two conventions exist:
      - multiprocessing/pebble: negative signal number (-9)
      - shell: 128 + signal number (137)
    """
    return exitcode == -9 or exitcode == 137
