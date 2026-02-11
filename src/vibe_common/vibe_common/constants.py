# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from typing import Dict, Final, List, Tuple

HeaderDict = Dict[str, str]
WorkReply = Tuple[str, int, HeaderDict]

# Defined here directly to remove the import from vibe_core.cli.local.
DATA_SUFFIX: Final[str] = "data"

DEFAULT_STORE_PATH: Final[str] = os.environ.get(
    "DEFAULT_STORE_PATH", os.path.join("/mnt", DATA_SUFFIX)
)
DEFAULT_CATALOG_PATH: Final[str] = os.environ.get(
    "DEFAULT_CATALOG_PATH", os.path.join(DEFAULT_STORE_PATH, "stac")
)
DEFAULT_ASSET_PATH: Final[str] = os.environ.get(
    "DEFAULT_ASSET_PATH", os.path.join(DEFAULT_STORE_PATH, "assets")
)
DEFAULT_BLOB_ASSET_MANAGER_CONTAINER: Final[str] = "assets"
DEFAULT_COSMOS_DATABASE_NAME: Final[str] = "prod-catalog"
DEFAULT_STAC_COSMOS_CONTAINER: Final[str] = "prod-stac"
DEFAULT_COSMOS_KEY_VAULT_KEY_NAME: Final[str] = "stac-cosmos-write-key"
DEFAULT_COSMOS_URI: Final[str] = ""
DEFAULT_SECRET_STORE_NAME: Final[str] = "azurekeyvault"

CONTROL_STATUS_PUBSUB: Final[str] = "control-pubsub"
CONTROL_PUBSUB_TOPIC: Final[str] = "commands"
CACHE_PUBSUB_TOPIC: Final[str] = "cache-commands"
STATUS_PUBSUB_TOPIC: Final[str] = "updates"

TRACEPARENT_VERSION: Final[str] = "00"
TRACEPARENT_FLAGS: Final[int] = 1

TRACE_FORMAT: Final[str] = "032x"
SPAN_FORMAT: Final[str] = "016x"
FLAGS_FORMAT: Final[str] = "02x"

TRACEPARENT_STRING = (
    f"{TRACEPARENT_VERSION}-{{trace_id:{TRACE_FORMAT}}}"
    f"-{{parent_id:{SPAN_FORMAT}}}-{{trace_flags:{FLAGS_FORMAT}}}"
)
TRACEPARENT_HEADER_KEY: Final[str] = "Traceparent"

WORKFLOW_ARTIFACTS_PUBSUB_TOPIC: Final[str] = "workflow-artifacts-commands"
WORKFLOW_REQUEST_PUBSUB_TOPIC: Final[str] = "workflow_execution_request"
STATE_URL_PATH = "/v1.0/state"

PUBSUB_URL_TEMPLATE: Final[str] = "http://{}:{}/v1.0/publish/{}/{}"
SERVICE_INVOCACATION_URL_PATH = "/v1.0/invoke"

RUNS_KEY: Final[str] = "runs"
ALLOWED_ORIGINS: Final[List[str]] = [
    o
    for o in os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:3000,"
        "http://localhost,"
        "http://127.0.0.1:8080,"
        "http://127.0.0.1:3000,",
    ).split(",")
    if o
]

MAX_PARALLEL_REQUESTS: Final[int] = 8

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OPS_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "..", "ops"))
if not os.path.exists(DEFAULT_OPS_DIR):
    DEFAULT_OPS_DIR = os.path.join("/", "app", "ops")

# ---------------------------------------------------------------------------
# Dapr-dependent URL templates — computed lazily on first access so that
# importing vibe_common.constants does not require a running Dapr sidecar.
# All three are cached in the module dict after the first access.
# ---------------------------------------------------------------------------

_DAPR_LAZY: Final[frozenset] = frozenset(
    {"STATE_URL_TEMPLATE", "PUBSUB_WORKFLOW_URL", "DATA_OPS_INVOKE_URL_TEMPLATE"}
)


def __getattr__(name: str) -> str:
    if name in _DAPR_LAZY:
        from dapr.conf import settings  # type: ignore[import]

        host = str(settings.DAPR_RUNTIME_HOST)
        port = str(settings.DAPR_HTTP_PORT)
        vals = {
            "STATE_URL_TEMPLATE": (
                f"http://{host}:{port}{STATE_URL_PATH}" + "/{}/{}"
            ),
            "PUBSUB_WORKFLOW_URL": PUBSUB_URL_TEMPLATE.format(
                host, port, CONTROL_STATUS_PUBSUB, WORKFLOW_REQUEST_PUBSUB_TOPIC
            ),
            "DATA_OPS_INVOKE_URL_TEMPLATE": (
                f"http://{host}:{port}{SERVICE_INVOCACATION_URL_PATH}"
                "/terravibes-data-ops/method/{}/{}"
            ),
        }
        # Cache all three so future accesses hit __dict__ directly.
        globals().update(vals)
        return vals[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
