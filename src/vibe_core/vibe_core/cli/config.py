# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Central FarmVibes.AI configuration.

All values have sensible defaults and can be overridden via environment variables
using the ``FARMVIBES_AI_`` prefix:

  FARMVIBES_AI_REGISTRY                  Container registry
                                          (default: mcr.microsoft.com)
  FARMVIBES_AI_IMAGE_PREFIX              Image path prefix inside the registry
                                          (default: farmai/terravibes/)
  FARMVIBES_AI_IMAGE_TAG                 Image tag for FarmVibes services
                                          (default: 12088305617)
  FARMVIBES_AI_REDIS_IMAGE_REPOSITORY    Redis image repository
                                          (default: bitnamilegacy/redis)
  FARMVIBES_AI_REDIS_IMAGE_TAG           Redis image tag
                                          (default: 7.4.1-debian-12-r2)
  FARMVIBES_AI_RABBITMQ_IMAGE_REPOSITORY RabbitMQ image repository
                                          (default: bitnamilegacy/rabbitmq)
  FARMVIBES_AI_RABBITMQ_IMAGE_TAG        RabbitMQ image tag
                                          (default: 4.0.4-debian-12-r1)
  FARMVIBES_AI_PORT                      REST API host port (default: 31108)
  FARMVIBES_AI_REGISTRY_PORT             Local registry host port (default: 5000)
  FARMVIBES_AI_WORKER_REPLICAS           Worker replica count
                                          (default: max(1, cpu_count()//2 - 1))
  FARMVIBES_AI_MAX_WORKER_NODES          Max worker VM nodes for AKS (default: 3)
  FARMVIBES_AI_LOG_LEVEL                 Log level (default: DEBUG)
  FARMVIBES_AI_MAX_LOG_FILE_BYTES        Max bytes per log file (optional)
  FARMVIBES_AI_LOG_BACKUP_COUNT          Log rotation backup count (optional)

Related env vars not managed by this module (documented here for completeness):
  FARMVIBES_AI_STORAGE_PATH   Data storage root (handled in local.py)
  FARMVIBES_AI_CONFIG_DIR     Config directory override (handled in osartifacts.py)
  FARMVIBES_AI_CLUSTER_NAME   Cluster name override (handled in parsers.py)
"""

from __future__ import annotations

import re
from multiprocessing import cpu_count
from typing import Optional

from pydantic import BaseSettings, validator

_VALID_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "WARN", "ERROR", "CRITICAL"})
# Basic sanity check — allows alphanumerics plus common image-ref delimiters.
# Docker performs complete reference validation when pulling images.
_IMAGE_COMPONENT_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._\-/:@]*$")


class FarmVibesConfig(BaseSettings):
    """Central FarmVibes.AI configuration.

    Instantiate with ``FarmVibesConfig()`` to read defaults and any
    ``FARMVIBES_AI_*`` environment variable overrides automatically.
    """

    # ── Image configuration ────────────────────────────────────────────────
    registry: str = "mcr.microsoft.com"
    image_prefix: str = "farmai/terravibes/"
    image_tag: str = "12088305617"
    redis_image_repository: str = "bitnamilegacy/redis"
    redis_image_tag: str = "7.4.1-debian-12-r2"
    rabbitmq_image_repository: str = "bitnamilegacy/rabbitmq"
    rabbitmq_image_tag: str = "4.0.4-debian-12-r1"

    # ── Port configuration ─────────────────────────────────────────────────
    port: int = 31108
    registry_port: int = 5000

    # ── Resource configuration ─────────────────────────────────────────────
    worker_replicas: int = max(1, cpu_count() // 2 - 1)
    max_worker_nodes: int = 3

    # ── Logging configuration ──────────────────────────────────────────────
    log_level: str = "DEBUG"
    max_log_file_bytes: Optional[int] = None
    log_backup_count: Optional[int] = None

    class Config:
        env_prefix = "FARMVIBES_AI_"

    # ── Validators ─────────────────────────────────────────────────────────

    @validator("registry", "image_tag", "redis_image_repository", "redis_image_tag",
               "rabbitmq_image_repository", "rabbitmq_image_tag")
    @classmethod
    def _valid_image_component(cls, v: str, field) -> str:
        # Explicit empty check gives a clearer error than the regex would.
        if not v:
            raise ValueError(f"{field.name} must not be empty")
        if not _IMAGE_COMPONENT_RE.match(v):
            raise ValueError(
                f"{field.name}={v!r} is not a valid image reference component. "
                "Allowed characters: alphanumerics, '.', '-', '_', '/', ':', '@'."
            )
        return v

    @validator("image_prefix")
    @classmethod
    def _valid_image_prefix(cls, v: str) -> str:
        # Empty prefix is allowed (no prefix path needed in some deployments)
        if v and not _IMAGE_COMPONENT_RE.match(v):
            raise ValueError(
                f"image_prefix={v!r} is not a valid image reference component."
            )
        return v

    @validator("port", "registry_port")
    @classmethod
    def _valid_port(cls, v: int, field) -> int:
        if not (1 <= v <= 65535):
            raise ValueError(
                f"{field.name}={v} is outside the valid port range (1-65535)"
            )
        return v

    @validator("log_level")
    @classmethod
    def _valid_log_level(cls, v: str) -> str:
        if v.upper() not in _VALID_LOG_LEVELS:
            raise ValueError(
                f"log_level={v!r} is not a valid log level. "
                f"Valid values: {', '.join(sorted(_VALID_LOG_LEVELS))}"
            )
        return v.upper()

    @validator("worker_replicas")
    @classmethod
    def _valid_worker_replicas(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"worker_replicas={v} must be >= 1")
        return v

    @validator("max_worker_nodes")
    @classmethod
    def _valid_max_worker_nodes(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"max_worker_nodes={v} must be >= 1")
        return v


# ── Module-level singleton ─────────────────────────────────────────────────────

_config: Optional[FarmVibesConfig] = None


def get_config() -> FarmVibesConfig:
    """Return the process-wide config singleton.

    Creates a fresh ``FarmVibesConfig`` on first call (reading env vars).
    Subsequent calls return the cached instance.

    Note: not thread-safe. In multi-threaded contexts, call
    ``load_and_validate_config()`` once during startup before spawning threads.
    """
    global _config
    if _config is None:
        _config = FarmVibesConfig()
    return _config


def load_and_validate_config() -> FarmVibesConfig:
    """Build, validate, cache and return the config.

    Raises ``pydantic.ValidationError`` with a clear message if any value is
    invalid. Call this once at CLI startup to surface config errors before any
    Terraform or Kubernetes work begins.
    """
    global _config
    cfg = FarmVibesConfig()
    _config = cfg
    return cfg
