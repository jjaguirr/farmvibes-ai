import logging
import os
import signal
import threading
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_CGROUP_PATH = "/sys/fs/cgroup"
OOM_MEMORY_THRESHOLD_PERCENT = 80.0


@dataclass
class MemoryInfo:
    current_bytes: int
    limit_bytes: int
    usage_percent: float


@dataclass
class CpuInfo:
    usage_usec: int
    system_usec: int
    nr_periods: int
    nr_throttled: int


class ResourceMonitor:
    def __init__(self, cgroup_path: str = DEFAULT_CGROUP_PATH):
        self._cgroup_path = cgroup_path
        self._last_memory: Optional[MemoryInfo] = None
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._cgroup_warned = False

    def _read_cgroup_file(self, filename: str) -> Optional[str]:
        path = os.path.join(self._cgroup_path, filename)
        try:
            with open(path) as f:
                return f.read().strip()
        except (FileNotFoundError, PermissionError):
            if not self._cgroup_warned:
                logger.info(
                    f"cgroup file {path} not available. "
                    "Resource monitoring will be limited on this platform."
                )
                self._cgroup_warned = True
            return None

    def get_memory_usage(self) -> Optional[MemoryInfo]:
        current_raw = self._read_cgroup_file("memory.current")
        limit_raw = self._read_cgroup_file("memory.max")
        if current_raw is None or limit_raw is None:
            return None
        current_bytes = int(current_raw)
        if limit_raw == "max":
            self._last_memory = MemoryInfo(
                current_bytes=current_bytes, limit_bytes=0, usage_percent=0.0
            )
        else:
            limit_bytes = int(limit_raw)
            usage_percent = (current_bytes / limit_bytes * 100) if limit_bytes > 0 else 0.0
            self._last_memory = MemoryInfo(
                current_bytes=current_bytes,
                limit_bytes=limit_bytes,
                usage_percent=usage_percent,
            )
        return self._last_memory

    def get_cpu_usage(self) -> Optional[CpuInfo]:
        raw = self._read_cgroup_file("cpu.stat")
        if raw is None:
            return None
        stats = {}
        for line in raw.splitlines():
            parts = line.split()
            if len(parts) == 2:
                stats[parts[0]] = int(parts[1])
        return CpuInfo(
            usage_usec=stats.get("usage_usec", 0),
            system_usec=stats.get("system_usec", 0),
            nr_periods=stats.get("nr_periods", 0),
            nr_throttled=stats.get("nr_throttled", 0),
        )

    def detect_oom(self, exit_signal: int) -> bool:
        if exit_signal != signal.SIGKILL:
            return False
        if self._last_memory is None:
            return False
        if self._last_memory.limit_bytes == 0:
            return False
        return self._last_memory.usage_percent >= OOM_MEMORY_THRESHOLD_PERCENT

    def format_oom_message(self, op_name: str, exit_signal: int) -> str:
        sig_name = signal.Signals(exit_signal).name
        if self._last_memory is None or self._last_memory.limit_bytes == 0:
            return f"Op '{op_name}' killed ({sig_name}): memory usage data unavailable"
        current_gb = self._last_memory.current_bytes / (1024**3)
        limit_gb = self._last_memory.limit_bytes / (1024**3)
        return (
            f"Op '{op_name}' killed ({sig_name}): "
            f"memory usage {current_gb:.1f}GB / {limit_gb:.1f}GB limit "
            f"({self._last_memory.usage_percent:.0f}% at last sample)"
        )

    def _log_usage(self, target_logger: logging.Logger) -> None:
        mem = self.get_memory_usage()
        cpu = self.get_cpu_usage()
        parts = []
        if mem is not None:
            if mem.limit_bytes > 0:
                current_mb = mem.current_bytes / (1024**2)
                limit_mb = mem.limit_bytes / (1024**2)
                parts.append(
                    f"memory={current_mb:.0f}MB/{limit_mb:.0f}MB ({mem.usage_percent:.1f}%)"
                )
            else:
                current_mb = mem.current_bytes / (1024**2)
                parts.append(f"memory={current_mb:.0f}MB (no limit)")
        if cpu is not None:
            parts.append(f"cpu_usage={cpu.usage_usec}us")
        if parts:
            target_logger.info(f"Resource usage: {', '.join(parts)}")

    def start_periodic_logging(
        self, interval_s: float = 30.0, logger: Optional[logging.Logger] = None
    ) -> None:
        if logger is None:
            logger = globals()["logger"]
        self._stop_event.clear()
        target_logger = logger

        def _loop():
            while not self._stop_event.wait(timeout=interval_s):
                self._log_usage(target_logger)

        self._monitoring_thread = threading.Thread(
            target=_loop, name="resource-monitor", daemon=True
        )
        self._monitoring_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._monitoring_thread is not None:
            self._monitoring_thread.join(timeout=5.0)
            self._monitoring_thread = None
