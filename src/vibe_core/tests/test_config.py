# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from vibe_core.cli.config import FarmVibesConfig, load_config


class TestEnvOverrides:
    """The core contract: FARMVIBES_* env vars override config values."""

    @pytest.mark.parametrize(
        "env_var, field, value, expected",
        [
            ("FARMVIBES_IMAGE_TAG", "image_tag", "custom-tag", "custom-tag"),
            ("FARMVIBES_REDIS_IMAGE_TAG", "redis_image_tag", "7.5.0-r1", "7.5.0-r1"),
            ("FARMVIBES_REDIS_IMAGE_REPOSITORY", "redis_image_repository", "docker.io/redis", "docker.io/redis"),
            ("FARMVIBES_RABBITMQ_IMAGE_TAG", "rabbitmq_image_tag", "4.1.0-r1", "4.1.0-r1"),
            ("FARMVIBES_RABBITMQ_IMAGE_REPOSITORY", "rabbitmq_image_repository", "docker.io/rabbitmq", "docker.io/rabbitmq"),
            ("FARMVIBES_IMAGE_REGISTRY", "image_registry", "ghcr.io", "ghcr.io"),
            ("FARMVIBES_IMAGE_PREFIX", "image_prefix", "custom/", "custom/"),
            ("FARMVIBES_WORKER_REPLICAS", "worker_replicas", "5", 5),
            ("FARMVIBES_WORKER_MEMORY_REQUEST", "worker_memory_request", "8Gi", "8Gi"),
            ("FARMVIBES_PORT", "port", "31200", 31200),
            ("FARMVIBES_HOST", "host", "0.0.0.0", "0.0.0.0"),
            ("FARMVIBES_REGISTRY_PORT", "registry_port", "5001", 5001),
            ("FARMVIBES_LOG_LEVEL", "log_level", "INFO", "INFO"),
            ("FARMVIBES_MAX_WORKER_NODES", "max_worker_nodes", "5", 5),
            ("FARMVIBES_ENABLE_TELEMETRY", "enable_telemetry", "true", True),
            ("FARMVIBES_MAX_LOG_FILE_BYTES", "max_log_file_bytes", "1048576", 1048576),
            ("FARMVIBES_LOG_BACKUP_COUNT", "log_backup_count", "3", 3),
        ],
    )
    def test_env_var_overrides_field(self, env_var, field, value, expected):
        with patch.dict(os.environ, {env_var: value}):
            cfg = FarmVibesConfig()
            assert getattr(cfg, field) == expected


class TestValidation:
    """Bad config must fail early, not mid-Terraform."""

    @pytest.mark.parametrize(
        "env_var, value, error_match",
        [
            ("FARMVIBES_IMAGE_TAG", "", "Image tag must not be empty"),
            ("FARMVIBES_IMAGE_TAG", "   ", "Image tag must not be empty"),
            ("FARMVIBES_REDIS_IMAGE_TAG", "", "Image tag must not be empty"),
            ("FARMVIBES_REDIS_IMAGE_REPOSITORY", "", "Image repository must not be empty"),
            ("FARMVIBES_RABBITMQ_IMAGE_REPOSITORY", "", "Image repository must not be empty"),
            ("FARMVIBES_IMAGE_REGISTRY", "", "Image registry must not be empty"),
            ("FARMVIBES_PORT", "0", "not in valid range"),
            ("FARMVIBES_PORT", "99999", "not in valid range"),
            ("FARMVIBES_REGISTRY_PORT", "-1", "not in valid range"),
            ("FARMVIBES_WORKER_REPLICAS", "-1", "must be >= 0"),
            ("FARMVIBES_WORKER_MEMORY_REQUEST", "lots", "Invalid Kubernetes memory format"),
            ("FARMVIBES_WORKER_MEMORY_REQUEST", "Mi", "Invalid Kubernetes memory format"),
            ("FARMVIBES_WORKER_MEMORY_REQUEST", "0", "Invalid Kubernetes memory format"),
            ("FARMVIBES_WORKER_MEMORY_REQUEST", "0Mi", "Invalid Kubernetes memory format"),
            ("FARMVIBES_LOG_LEVEL", "TRACE", "Invalid log level"),
            ("FARMVIBES_LOG_LEVEL", "verbose", "Invalid log level"),
            ("FARMVIBES_PORT", "65536", "not in valid range"),
            ("FARMVIBES_REGISTRY_PORT", "65536", "not in valid range"),
            ("FARMVIBES_MAX_LOG_FILE_BYTES", "-1", "must be >= 0"),
            ("FARMVIBES_LOG_BACKUP_COUNT", "-1", "must be >= 0"),
        ],
    )
    def test_bad_value_rejected(self, env_var, value, error_match):
        with patch.dict(os.environ, {env_var: value}):
            with pytest.raises(ValidationError, match=error_match):
                FarmVibesConfig()

    @pytest.mark.parametrize("mem", ["100Mi", "8Gi", "512", "1Ti", "256Ki", "100M", "2G"])
    def test_valid_k8s_memory_accepted(self, mem):
        with patch.dict(os.environ, {"FARMVIBES_WORKER_MEMORY_REQUEST": mem}):
            cfg = FarmVibesConfig()
            assert cfg.worker_memory_request == mem

    def test_zero_replicas_allowed(self):
        """Zero replicas is valid (scale-to-zero)."""
        with patch.dict(os.environ, {"FARMVIBES_WORKER_REPLICAS": "0"}):
            cfg = FarmVibesConfig()
            assert cfg.worker_replicas == 0

    @pytest.mark.parametrize(
        "field, value, expected",
        [
            ("port", "1", 1),
            ("port", "65535", 65535),
            ("registry_port", "1", 1),
            ("registry_port", "65535", 65535),
        ],
    )
    def test_port_boundaries_accepted(self, field, value, expected):
        env_var = f"FARMVIBES_{field.upper()}"
        with patch.dict(os.environ, {env_var: value}):
            cfg = FarmVibesConfig()
            assert getattr(cfg, field) == expected

    def test_max_log_file_bytes_zero_allowed(self):
        with patch.dict(os.environ, {"FARMVIBES_MAX_LOG_FILE_BYTES": "0"}):
            cfg = FarmVibesConfig()
            assert cfg.max_log_file_bytes == 0


class TestConfigFlowsToConstants:
    """Env overrides must propagate through constants.py to downstream consumers."""

    def test_env_override_reaches_constants(self):
        """Setting FARMVIBES_REDIS_IMAGE_TAG must change what constants.py exports."""
        import importlib
        import vibe_core.cli.config as config_mod
        import vibe_core.cli.constants as constants_mod

        try:
            with patch.dict(os.environ, {"FARMVIBES_REDIS_IMAGE_TAG": "override-test-999"}):
                importlib.reload(config_mod)
                importlib.reload(constants_mod)
                assert constants_mod.REDIS_IMAGE_TAG == "override-test-999"
        finally:
            importlib.reload(config_mod)
            importlib.reload(constants_mod)


class TestConfigFlowsToParsers:
    """Env overrides must become CLI arg defaults."""

    def test_env_override_becomes_parser_default(self):
        import importlib
        import vibe_core.cli.config as config_mod
        import vibe_core.cli.parsers as parsers_mod

        try:
            with patch.dict(os.environ, {"FARMVIBES_PORT": "32000"}):
                importlib.reload(config_mod)
                importlib.reload(parsers_mod)

                parser = parsers_mod.LocalCliParser("local")
                args = parser.parse(["setup", "--cluster-name", "test"])
                assert args.port == 32000
        finally:
            importlib.reload(config_mod)
            importlib.reload(parsers_mod)

    def test_cli_arg_overrides_env(self):
        """CLI --port flag must beat FARMVIBES_PORT env var."""
        import importlib
        import vibe_core.cli.config as config_mod
        import vibe_core.cli.parsers as parsers_mod

        try:
            with patch.dict(os.environ, {"FARMVIBES_PORT": "32000"}):
                importlib.reload(config_mod)
                importlib.reload(parsers_mod)

                parser = parsers_mod.LocalCliParser("local")
                args = parser.parse(["setup", "--cluster-name", "test", "--port", "33000"])
                assert args.port == 33000
        finally:
            importlib.reload(config_mod)
            importlib.reload(parsers_mod)


class TestDispatchValidation:
    """dispatch() must reject bad config before touching infrastructure."""

    def test_dispatch_fails_on_bad_config(self):
        import argparse
        import importlib
        import vibe_core.cli.config as config_mod
        from vibe_core.cli.local import dispatch

        try:
            with patch.dict(os.environ, {"FARMVIBES_IMAGE_TAG": ""}):
                importlib.reload(config_mod)

                args = argparse.Namespace(
                    action="setup", cluster_name="test", auto_confirm=False
                )
                result = dispatch(args)
                assert result is False
        finally:
            importlib.reload(config_mod)
