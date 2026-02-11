# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .config import load_config as _load_config

_cfg = _load_config()

DEFAULT_IMAGE_PREFIX = _cfg.image_prefix
DEFAULT_IMAGE_TAG = _cfg.image_tag
DEFAULT_REGISTRY_PATH = _cfg.image_registry

LOCAL_SERVICE_URL_PATH_FILE = "service_url"
REMOTE_SERVICE_URL_PATH_FILE = "remote_service_url"
MAX_WORKER_NODES = _cfg.max_worker_nodes

AZURE_CR_DOMAIN = "azurecr.io"

# Local constants
ONNX_SUBDIR = "onnx_resources"
FARMVIBES_AI_LOG_LEVEL = _cfg.log_level
REDIS_IMAGE_REPOSITORY = _cfg.redis_image_repository
REDIS_IMAGE_TAG = _cfg.redis_image_tag
RABBITMQ_IMAGE_REPOSITORY = _cfg.rabbitmq_image_repository
RABBITMQ_IMAGE_TAG = _cfg.rabbitmq_image_tag
