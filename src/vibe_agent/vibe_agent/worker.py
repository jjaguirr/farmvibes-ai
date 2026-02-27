# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import asyncio
import concurrent.futures
import json
import logging
import os
import resource
import signal
import sys
import threading
import time
import traceback
from multiprocessing.context import ForkServerContext
from typing import Any, Dict, List, Optional, Tuple, Union, cast
from uuid import UUID

import pebble.concurrent
from cloudevents.sdk.event import v1
from dapr.conf import settings
from dapr.ext.grpc import App, TopicEventResponse
from hydra_zen import MISSING, builds, instantiate
from opentelemetry import trace
from pebble import ProcessFuture
from pebble.common import ProcessExpired

from vibe_common.constants import CONTROL_STATUS_PUBSUB, STATUS_PUBSUB_TOPIC
from vibe_common.dapr import dapr_ready
from vibe_common.graceful_shutdown import ShutdownManager
from vibe_common.messaging import (
    CacheInfoExecuteRequestContent,
    CacheInfoExecuteRequestMessage,
    WorkMessage,
    WorkMessageBuilder,
    accept_or_fail_event,
    extract_message_header_from_event,
    send_async,
)
from vibe_common.resource_monitor import ResourceMonitor
from vibe_common.retry import retry_with_backoff
from vibe_common.schemas import CacheInfo
from vibe_common.statestore import StateStore
from vibe_common.telemetry import (
    add_span_attributes,
    add_trace,
    setup_telemetry,
    update_telemetry_context,
)
from vibe_core.data.core_types import OpIOType
from vibe_core.datamodel import RunConfig, RunStatus
from vibe_core.logconfig import LOG_BACKUP_COUNT, MAX_LOG_FILE_BYTES, configure_logging
from vibe_core.utils import get_input_ids

from .ops import OperationFactoryConfig, OperationSpec

MESSAGING_RETRY_INTERVAL_S = 1
TERMINATION_GRACE_PERIOD_S = 5
MAX_OP_EXECUTION_TIME_S = 60 * 60 * 3
SHUTDOWN_SEND_TIMEOUT_S = 10


class ShuttingDownException(Exception):
    pass


class OpSignalHandler:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.resource_description = {
            "ru_utime": "User time",
            "ru_stime": "System time",
            "ru_maxrss": "Max. Resident Set Size",
            "ru_ixrss": "Shared Memory Size",
            "ru_idrss": "Unshared Memory Size",
            "ru_isrss": "Stack Size",
            "ru_inblock": "Block inputs",
            "ru_oublock": "Block outputs",
        }

    def parse_resources_usage(self, rusages: List[resource.struct_rusage]):
        return {
            resource: {
                "description": description,
                "value": sum([getattr(rusage, resource) for rusage in rusages]),
            }
            for resource, description in self.resource_description.items()
        }

    def build_log_message(self, signum: int, child_pid: Optional[Tuple[int, int]]) -> str:
        resource_usages = [resource.getrusage(resource.RUSAGE_SELF)]

        if signum == signal.SIGTERM:
            msgs_list = ["Terminating op gracefully with SIGTERM."]
        else:
            msgs_list = [
                f"Received signal when executing op (signal {signal.Signals(signum).name}).",
            ]

        if child_pid:
            pid, exit_code = child_pid
            msgs_list.append(f" Child pid = {pid} exit code = {exit_code >> 8},")
            resource_usages.append(resource.getrusage(resource.RUSAGE_CHILDREN))

        msgs_list.append(f"Op resources = {self.parse_resources_usage(resource_usages)}")

        return " ".join(msgs_list)

    def get_log_function(self, child_pid: Optional[Tuple[int, int]]):
        if child_pid:
            _, exit_code = child_pid
            if not os.WIFEXITED(exit_code):
                return self.logger.error

        return self.logger.info

    def log(self, signum: int, _: Any):
        child_pid = None
        try:
            child_pid = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            # That's OK. There is no child process
            pass

        message = self.build_log_message(signum, child_pid)
        log_function = self.get_log_function(child_pid)
        log_function(message)


@pebble.concurrent.process(daemon=False, context=ForkServerContext())
# This must not be a daemonic process. Otherwise, we won't be able to run ops
# that start children.
def run_op(
    factory_spec: OperationFactoryConfig,  # type: ignore
    spec: OperationSpec,
    input: OpIOType,
    cache_info: CacheInfo,
) -> Union[OpIOType, traceback.TracebackException]:
    logger = logging.getLogger(f"{__name__}.run_op")
    logger.info(f"Building op {spec.name} to process input {get_input_ids(input)}")

    op_signal_handler = OpSignalHandler(logger)

    for sign in (signal.SIGINT, signal.SIGTERM, signal.SIGCHLD):
        signal.signal(sign, op_signal_handler.log)

    try:
        factory = instantiate(factory_spec)
        return factory.build(spec).run(input, cache_info)
    except Exception as e:
        return traceback.TracebackException.from_exception(e)


class WorkerMessenger:
    pubsubname: str
    status_topic: str
    logger: logging.Logger

    def __init__(
        self,
        pubsubname: str = CONTROL_STATUS_PUBSUB,
        status_topic: str = STATUS_PUBSUB_TOPIC,
        shutdown_manager: Optional[ShutdownManager] = None,
    ):
        self.pubsubname = pubsubname
        self.status_topic = status_topic
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._shutdown_manager = shutdown_manager

    async def send(self, message: WorkMessage, timeout_s: float = 0) -> None:
        tries: int = 0
        sent = False
        start = time.time()
        # During shutdown, apply a bounded timeout so we don't block the drain
        # sequence waiting for a sidecar that may already be gone.
        effective_timeout = timeout_s
        if effective_timeout <= 0 and self._shutdown_manager is not None:
            if self._shutdown_manager.is_draining:
                effective_timeout = SHUTDOWN_SEND_TIMEOUT_S
                self.logger.info(
                    f"Shutdown in progress, applying {effective_timeout}s send deadline"
                )
        while True:
            try:
                sent = await send_async(message, "worker", self.pubsubname, self.status_topic)
            except Exception:
                pass
            if sent:
                break
            tries += 1
            if effective_timeout > 0 and (time.time() - start) >= effective_timeout:
                self.logger.error(
                    f"Failed to send {message} after {tries} attempts "
                    f"and {effective_timeout}s timeout. Giving up."
                )
                return
            self.logger.warn(
                f"Failed to send {message} after {tries} attempts. "
                f"Sleeping for {MESSAGING_RETRY_INTERVAL_S}s before retrying."
            )
            await asyncio.sleep(MESSAGING_RETRY_INTERVAL_S)

    async def send_ack_reply(self, origin: WorkMessage) -> None:
        await self.send(WorkMessageBuilder.build_ack_reply(origin.id))
        self.logger.debug(msg=f"Sent ACK for {origin.id}")

    @add_trace
    async def send_success_reply(
        self,
        origin: WorkMessage,
        out: OpIOType,
        cache_info: Optional[CacheInfo] = None,
    ) -> None:
        if cache_info is None and not isinstance(origin, CacheInfoExecuteRequestMessage):
            raise ValueError(
                "cache_info must be provided if origin is not a CacheInfoExecuteRequestMessage"
            )
        if not cache_info:
            content = cast(CacheInfoExecuteRequestContent, origin.content)
            cache_info = CacheInfo(
                name=content.cache_info.name,
                version=content.cache_info.version,
                ids=content.cache_info.ids,
                parameters=content.cache_info.parameters,
            )
        await self.send(WorkMessageBuilder.build_execute_reply(origin.id, cache_info, out))
        self.logger.debug(msg=f"Sent success response for {origin.id}")

    async def send_failure_reply(self, traceparent: str, e: Exception, tb: List[str]) -> None:
        assert type(e) is not None, "`send_failure_reply` called without an exception to handle"
        reply = WorkMessageBuilder.build_error(
            traceparent,
            str(type(e)),
            str(e),
            tb,
        )
        await self.send(reply)
        self.logger.debug(f"Sent failure response for {traceparent}")


class Worker:
    app: App
    max_tries: int
    pubsubname: str
    status_topic: str
    control_topic: str
    current_message: Optional[WorkMessage] = None
    child_monitoring_period_s: int = 10
    termination_grace_period_s: int = 2
    state_store: StateStore
    current_child: Optional[ProcessFuture] = None
    factory_spec: OperationFactoryConfig  # type: ignore
    otel_service_name: str

    def __init__(
        self,
        termination_grace_period_s: int,
        control_topic: str,
        max_tries: int,
        factory_spec: OperationFactoryConfig,  # type: ignore
        port: int = settings.HTTP_APP_PORT,
        pubsubname: str = CONTROL_STATUS_PUBSUB,
        status_topic: str = STATUS_PUBSUB_TOPIC,
        logdir: Optional[str] = None,
        max_log_file_bytes: int = MAX_LOG_FILE_BYTES,
        log_backup_count: int = LOG_BACKUP_COUNT,
        loglevel: Optional[str] = None,
        otel_service_name: str = "",
        **kwargs: Dict[str, Any],
    ):
        self.pubsubname = pubsubname
        self.termination_grace_period_s = termination_grace_period_s
        self.control_topic = control_topic
        self.status_topic = status_topic
        self.port = port
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logdir: Optional[str] = logdir
        self.loglevel = loglevel
        self.max_log_file_bytes = max_log_file_bytes
        self.log_backup_count = log_backup_count
        self.otel_service_name = otel_service_name

        self.app = App()
        self.shutdown_manager = ShutdownManager()
        self.messenger = WorkerMessenger(pubsubname, status_topic, self.shutdown_manager)
        self.current_message = None
        self.work_lock = threading.Lock()
        self.max_tries = max_tries
        self.factory_spec = factory_spec
        self.statestore = StateStore()
        self.resource_monitor = ResourceMonitor()
        self.shutdown_manager.on_shutdown(lambda: self.resource_monitor.stop())
        self.heartbeat_interval_s = 30
        self._last_heartbeat_time = 0.0
        self.worker_id = os.environ.get("HOSTNAME", f"worker-{os.getpid()}")
        self.name = self.__class__.__name__
        self._setup_routes_and_events()

    def _terminate_child(self):
        if self.current_child is not None:
            try:
                self.current_child.cancel()
            except Exception:
                self.logger.info(
                    f"Failed to terminate child {self.current_child}, "
                    "probably because it terminated already"
                )

    def _publish_heartbeat(self):
        if self.current_message is None:
            return
        mem = self.resource_monitor.get_memory_usage()
        memory_bytes = mem.current_bytes if mem else 0
        content = cast(CacheInfoExecuteRequestContent, self.current_message.content)
        op_name = str(content.operation_spec.name) if content.operation_spec else "unknown"
        heartbeat = WorkMessageBuilder.build_heartbeat(
            traceparent=self.current_message.id,
            op_name=op_name,
            worker_id=self.worker_id,
            memory_usage_bytes=memory_bytes,
        )
        logger = self.logger

        def _send():
            try:
                asyncio.run(self.messenger.send(heartbeat, timeout_s=5))
            except Exception:
                logger.debug("Failed to send heartbeat, will retry next interval")

        threading.Thread(target=_send, name="heartbeat-send", daemon=True).start()

    def _setup_routes_and_events(self):
        @self.app.subscribe(self.pubsubname, self.control_topic)
        def fetch_work(event: v1.Event) -> TopicEventResponse:
            return self.fetch_work(self.control_topic, event)

        @self.app.method(name="shutdown")
        def shutdown() -> TopicEventResponse:
            self.logger.info("Initiating shutdown sequence")
            self.pre_stop_hook(signal.SIGTERM, None)
            return TopicEventResponse("retry")

    def pre_stop_hook(self, signum: int, _: Any):
        self.shutdown_manager.shutdown()
        # Stop the gRPC server to reject new message deliveries
        if self.app._server is not None:
            self.app._server.stop(None)
        # Wait for current work to finish within grace period
        if self.current_message is not None:
            if not self.shutdown_manager.wait_for_completion(self.termination_grace_period_s):
                self.logger.warning(
                    f"Grace period ({self.termination_grace_period_s}s) expired, "
                    "cancelling current work"
                )
                self._terminate_child()
        self.shutdown_manager.finalize()

    def run(self):
        appname = "terravibes-worker"
        configure_logging(
            default_level=self.loglevel,
            appname=appname,
            logdir=self.logdir,
            max_log_file_bytes=self.max_log_file_bytes,
            log_backup_count=self.log_backup_count,
        )
        if self.otel_service_name:
            setup_telemetry(appname, self.otel_service_name)
        self.resource_monitor.start_periodic_logging(interval_s=30, logger=self.logger)
        self.start_service()

    @dapr_ready
    def start_service(self):
        self.logger.info(f"Starting worker listening on port {self.port}")
        while not self.shutdown_manager.is_draining:
            # For some reason, the FastAPI lifecycle shutdown action is
            # executing without us intending for it to run. We add this loop
            # here to bring the server up if we haven't explicitly initiated the
            # shutdown routine.
            self.app.run(self.port)
            time.sleep(1)

    @add_trace
    def run_op_from_message(self, message: WorkMessage, timeout_s: float):
        try:
            self.current_message = message
            content = cast(CacheInfoExecuteRequestContent, message.content)
            out = self.run_op_with_retry(content, message.run_id, timeout_s)
            asyncio.run(self.messenger.send_success_reply(message, out))
        except ShuttingDownException:
            # We are shutting down. Don't send a reply. Another worker will pick
            # this up.
            raise
        except Exception as e:
            _, _, tb = sys.exc_info()
            asyncio.run(self.messenger.send_failure_reply(message.id, e, traceback.format_tb(tb)))
            raise
        finally:
            self.current_message = None
            self.shutdown_manager.mark_work_complete()

    def is_workflow_complete(self, message: WorkMessage) -> bool:
        try:
            run = asyncio.run(self.statestore.retrieve(str(message.run_id)))
        except KeyError:
            self.logger.warn(
                f"Run {message.run_id} not found in statestore. Assuming it's not complete."
            )
            return False
        if not isinstance(run, dict):
            run = json.loads(run)
        runconfig = RunConfig(**run)
        return RunStatus.finished(runconfig.details.status)

    def fetch_work(self, channel: str, event: v1.Event) -> TopicEventResponse:
        @add_trace
        def success_callback(message: WorkMessage) -> TopicEventResponse:
            add_span_attributes({"run_id": str(message.run_id)})
            if not message.is_valid_for_channel(channel):
                self.logger.warning(
                    f"Received invalid message {message} for channel {channel}. Dropping it."
                )
                return TopicEventResponse("drop")
            if self.is_workflow_complete(message):
                self.logger.warning(
                    f"Rejecting event with id {event.id} for completed/failed/cancelled "
                    f"workflow {message.run_id}."
                )
                return TopicEventResponse("drop")

            if self.shutdown_manager.is_draining:
                self.logger.info(f"Shutdown in progress. Rejecting event {event.id}")
                return TopicEventResponse("retry")

            if not self.work_lock.acquire(blocking=False):
                self.logger.info(f"Worker busy. Rejecting new work event {event.id}")
                return TopicEventResponse("retry")
            try:
                asyncio.run(self.messenger.send_ack_reply(message))
                self.run_op_from_message(message, MAX_OP_EXECUTION_TIME_S)
                return TopicEventResponse("success")
            except ShuttingDownException:
                self.shutdown_manager.mark_work_complete()
                return TopicEventResponse("retry")
            except Exception:
                self.shutdown_manager.mark_work_complete()
                self.logger.exception(f"Failed to run op for event {event.id}")
                raise
            finally:
                self.work_lock.release()

        @add_trace
        def failure_callback(event: v1.Event, e: Exception, tb: List[str]) -> TopicEventResponse:
            asyncio.run(self.messenger.send_failure_reply(event.id, e, tb))
            return TopicEventResponse("drop")

        update_telemetry_context(extract_message_header_from_event(event).current_trace_parent)
        return accept_or_fail_event(event, success_callback, failure_callback)  # type: ignore

    def get_future_result(
        self, child: ProcessFuture, monitoring_period_s: int, timeout_s: float
    ) -> Any:
        start_time = time.time()
        while time.time() - start_time < timeout_s:
            try:
                ret = child.result(monitoring_period_s)
                return ret
            except concurrent.futures.TimeoutError:
                # Publish heartbeat if interval has elapsed
                now = time.time()
                if now - self._last_heartbeat_time >= self.heartbeat_interval_s:
                    self._publish_heartbeat()
                    self._last_heartbeat_time = now
                assert self.current_message is not None, (
                    "There's a correctness issue in the worker code. "
                    "`current_message` should not be `None`."
                )
                if self.is_workflow_complete(self.current_message):
                    self.logger.info(
                        f"Workflow {self.current_message.run_id} is complete. "
                        "Terminating child process."
                    )
                    child.cancel()
                    raise RuntimeError(
                        "Workflow was completed/failed/cancelled while running op. "
                        "Terminating child process."
                    )
                if self.shutdown_manager.is_draining:
                    self.logger.info("Shutdown process initiated. Terminating child process.")
                    child.cancel()
                    raise ShuttingDownException()
                continue
            except concurrent.futures.CancelledError:
                if self.shutdown_manager.is_draining:
                    raise ShuttingDownException()
                self.logger.warn(
                    f"Child process was cancelled while running op {self.current_message}. "
                    "But we're not shutting down. This is unexpected."
                )
                raise
            except Exception as e:
                self.logger.exception(f"Child process failed with exception {e}")
                return traceback.TracebackException.from_exception(e)
        raise TimeoutError(f"Op execution took longer than the allowed {timeout_s} seconds.")

    @add_trace
    def try_run_op(
        self, spec: OperationSpec, content: CacheInfoExecuteRequestContent, inner_timeout: float
    ) -> Union[OpIOType, traceback.TracebackException]:
        trace.get_current_span().set_attribute("op_name", str(spec.name))
        self.current_child = cast(
            ProcessFuture,
            run_op(self.factory_spec, spec, content.input, content.cache_info),  # type: ignore
        )
        ret = self.get_future_result(
            self.current_child, self.child_monitoring_period_s, inner_timeout
        )

        return ret

    @add_trace
    def run_op_with_retry(
        self, content: CacheInfoExecuteRequestContent, run_id: UUID, timeout_s: float
    ) -> OpIOType:
        spec = cast(OperationSpec, content.operation_spec)
        self.logger.info(
            f"Will try to execute op {spec} with input {get_input_ids(content.input)} "
            f"for at most {self.max_tries} tries in child process."
        )
        final_time = time.time() + timeout_s

        @retry_with_backoff(
            max_retries=self.max_tries,
            base_delay=1.0,
            max_delay=60.0,
            backoff_factor=2.0,
            retryable_exceptions=(ProcessExpired, RuntimeError),
            abort=lambda: self.shutdown_manager.is_draining,
        )
        def _attempt():
            inner_timeout = final_time - time.time()
            if inner_timeout <= 0:
                raise TimeoutError(
                    f"Op execution budget exhausted ({timeout_s}s total)."
                )
            if self.shutdown_manager.is_draining:
                raise ShuttingDownException()
            try:
                ret = self.try_run_op(spec, content, inner_timeout)
            except ProcessExpired as e:
                self.resource_monitor.get_memory_usage()  # refresh last sample
                exit_signal = abs(e.exitcode) if hasattr(e, 'exitcode') and e.exitcode else 0
                if self.resource_monitor.detect_oom(exit_signal=exit_signal):
                    msg = self.resource_monitor.format_oom_message(
                        str(spec.name), exit_signal=exit_signal
                    )
                    self.logger.error(msg)
                raise
            if isinstance(ret, traceback.TracebackException):
                raise RuntimeError("".join(ret.format()))
            return ret

        try:
            result = _attempt()
        except TimeoutError as e:
            raise RuntimeError(
                f"Op {spec} timed out. Total time allowed: {timeout_s}s."
            ) from e
        finally:
            self.current_child = None
        return result


WorkerConfig = builds(
    Worker,
    port=settings.GRPC_APP_PORT,
    pubsubname=CONTROL_STATUS_PUBSUB,
    control_topic=MISSING,
    status_topic=STATUS_PUBSUB_TOPIC,
    max_tries=5,
    termination_grace_period_s=TERMINATION_GRACE_PERIOD_S,
    factory_spec=OperationFactoryConfig,
    zen_partial=False,
    hydra_recursive=False,
    logdir=None,
    max_log_file_bytes=MAX_LOG_FILE_BYTES,
    log_backup_count=LOG_BACKUP_COUNT,
    loglevel=None,
    otel_service_name="",
)
