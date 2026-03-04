# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import re
from multiprocessing import cpu_count
from typing import Any, Dict, Optional

from pydantic import BaseSettings, Field, validator

# Module-level storage for the active profile overrides. Set by load_config()
# before FarmVibesConfig is instantiated; read by the profile_settings source.
# This is module state rather than a class attribute because Pydantic v1
# customise_sources is a classmethod with no access to instance init kwargs.
_profile_overrides: Dict[str, Any] = {}


def _profile_settings(settings: BaseSettings) -> Dict[str, Any]:
    """Pydantic settings source: returns the active profile's overrides.

    Pydantic drops keys that map to None (treats them as 'not set'), so a
    profile value of `null` correctly means 'no override for this key'.
    """
    return {k: v for k, v in _profile_overrides.items() if v is not None}


class FarmVibesConfig(BaseSettings):
    """Central configuration for FarmVibes.AI.

    Every field is overridable via FARMVIBES_<FIELD_NAME> environment variable
    or by selecting a deployment profile.
    Priority: defaults < profile < env vars < CLI args.
    """

    class Config:
        env_prefix = "FARMVIBES_"

        @classmethod
        def customise_sources(cls, init_settings, env_settings, file_secret_settings):
            # Pydantic v1 resolves sources left-to-right: first non-missing wins.
            # Profile sits BELOW env (env beats profile) and ABOVE field defaults.
            return (init_settings, env_settings, _profile_settings, file_secret_settings)

    # --- Images ---
    image_registry: str = "mcr.microsoft.com"
    image_prefix: str = "farmai/terravibes/"
    image_tag: str = "12088305617"
    redis_image_repository: str = "bitnamilegacy/redis"
    redis_image_tag: str = "7.4.1-debian-12-r2"
    rabbitmq_image_repository: str = "bitnamilegacy/rabbitmq"
    rabbitmq_image_tag: str = "4.0.4-debian-12-r1"

    # --- Cluster ---
    worker_replicas: int = Field(default_factory=lambda: max(1, cpu_count() // 2 - 1))
    worker_memory_request: str = "100Mi"
    worker_memory_limit: Optional[str] = None
    worker_cpu_request: Optional[str] = None
    worker_cpu_limit: Optional[str] = None
    max_worker_nodes: int = 3
    log_level: str = "DEBUG"
    max_log_file_bytes: Optional[int] = None
    log_backup_count: Optional[int] = None
    enable_telemetry: bool = False
    storage_path: str = Field(
        default_factory=lambda: os.path.join(os.path.expanduser("~"), ".cache", "farmvibes-ai")
    )

    # --- Network ---
    host: str = "127.0.0.1"
    port: int = 31108
    registry_port: int = 5000

    # --- Validators ---
    @validator("image_tag", "redis_image_tag", "rabbitmq_image_tag", allow_reuse=True)
    def tag_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Image tag must not be empty")
        return v

    @validator("redis_image_repository", "rabbitmq_image_repository", allow_reuse=True)
    def repo_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Image repository must not be empty")
        return v

    @validator("image_registry", allow_reuse=True)
    def registry_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Image registry must not be empty")
        return v

    @validator("port", "registry_port", allow_reuse=True)
    def port_in_range(cls, v):
        if not (1 <= v <= 65535):
            raise ValueError(f"Port {v} is not in valid range 1-65535")
        return v

    @validator("worker_replicas", allow_reuse=True)
    def replicas_non_negative(cls, v):
        if v < 0:
            raise ValueError(f"Worker replicas must be >= 0, got {v}")
        return v

    @validator("worker_memory_request", "worker_memory_limit", allow_reuse=True)
    def valid_k8s_memory(cls, v):
        if v is None:
            return v
        if not re.match(r"^[1-9]\d*([EPTGMK]i?)?$", v):
            raise ValueError(
                f"Invalid Kubernetes memory format: '{v}'. "
                "Expected format like '100Mi', '8Gi', '512'"
            )
        return v

    @validator("worker_cpu_request", "worker_cpu_limit", allow_reuse=True)
    def valid_k8s_cpu(cls, v):
        if v is None:
            return v
        # K8s CPU: whole/decimal cores ("2", "0.5") or millicores ("500m").
        if not re.match(r"^(\d+m|\d+(\.\d+)?)$", v) or v in ("0", "0m"):
            raise ValueError(
                f"Invalid Kubernetes CPU format: '{v}'. "
                "Expected format like '500m', '1', '0.5', '2'"
            )
        return v

    @validator("max_log_file_bytes", "log_backup_count", allow_reuse=True)
    def optional_non_negative_int(cls, v):
        if v is not None and v < 0:
            raise ValueError(f"Value must be >= 0, got {v}")
        return v

    @validator("log_level", allow_reuse=True)
    def valid_log_level(cls, v):
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "debug", "info", "warning", "error"}
        if v not in allowed:
            raise ValueError(f"Invalid log level: '{v}'. Must be one of: {sorted(allowed)}")
        return v


def load_config(profile_overrides: Optional[Dict[str, Any]] = None) -> FarmVibesConfig:
    """Load and validate configuration.

    Layering (lowest to highest precedence):
      1. Hardcoded field defaults
      2. profile_overrides — values from the active deployment profile YAML
      3. FARMVIBES_* environment variables
      4. (CLI args — applied downstream by argparse, not here)

    Raises pydantic.ValidationError if any value is invalid.
    """
    global _profile_overrides
    _profile_overrides = profile_overrides or {}
    try:
        return FarmVibesConfig()
    finally:
        # Don't leave stale state for the next call. Matters in tests.
        _profile_overrides = {}
