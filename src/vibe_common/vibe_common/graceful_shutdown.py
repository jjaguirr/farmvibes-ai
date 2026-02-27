import enum
import logging
import signal
import threading
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)


class ShutdownState(enum.Enum):
    RUNNING = "running"
    DRAINING = "draining"
    SHUTTING_DOWN = "shutting_down"


class ShutdownManager:
    """Manages graceful shutdown with a drain-then-stop state machine.

    States: RUNNING -> DRAINING -> SHUTTING_DOWN

    In DRAINING, the service stops accepting new work and waits for current
    work to finish. After finalize() is called, transitions to SHUTTING_DOWN.
    """

    def __init__(self):
        self._state = ShutdownState.RUNNING
        self._lock = threading.Lock()
        self._callbacks: List[Callable[[], None]] = []
        self._work_complete = threading.Event()
        self._shutdown_initiated = False

    @property
    def state(self) -> ShutdownState:
        return self._state

    @property
    def is_draining(self) -> bool:
        return self._state in (ShutdownState.DRAINING, ShutdownState.SHUTTING_DOWN)

    def on_shutdown(self, callback: Callable[[], None]) -> None:
        self._callbacks.append(callback)

    def register_signals(self, signals: Optional[List[int]] = None) -> None:
        if signals is None:
            signals = [signal.SIGTERM, signal.SIGINT]
        for sig in signals:
            signal.signal(sig, self._signal_handler)

    def _signal_handler(self, signum: int, frame) -> None:
        sig_name = signal.Signals(signum).name
        logger.info(f"Received {sig_name}, initiating graceful shutdown")
        self.shutdown()

    def shutdown(self) -> None:
        with self._lock:
            if self._shutdown_initiated:
                return
            self._shutdown_initiated = True
            self._state = ShutdownState.DRAINING
            logger.info("Shutdown initiated, entering DRAINING state")

    def mark_work_complete(self) -> None:
        self._work_complete.set()

    def wait_for_completion(self, timeout_s: float) -> bool:
        return self._work_complete.wait(timeout=timeout_s)

    def finalize(self) -> None:
        logger.info("Running shutdown callbacks")
        for cb in reversed(self._callbacks):
            try:
                cb()
            except Exception:
                logger.exception("Error in shutdown callback")
        self._state = ShutdownState.SHUTTING_DOWN
        logger.info("Shutdown complete")
