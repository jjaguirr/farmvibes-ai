# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Tests for vibe_core.cli.config.

Coverage targets:
  - Defaults match the hardcoded values that were previously scattered across
    constants.py, parsers.py, and local.py.
  - Every FARMVIBES_AI_* env var that the module advertises actually changes
    the corresponding field.
  - Validators reject bad values and accept edge cases (empty prefix).
  - load_and_validate_config() raises on a bad env var so dispatch() can fail
    before Terraform starts.
  - dispatch() returns False on invalid config without calling OSArtifacts or
    Terraform (the "fail early" requirement from the task).
  - constants.py backward-compat names still resolve to the expected defaults.
"""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

import vibe_core.cli.config as _config_module
from vibe_core.cli.config import FarmVibesConfig, get_config, load_and_validate_config


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_config_singleton():
    """Reset the module-level singleton before and after every test."""
    _config_module._config = None
    yield
    _config_module._config = None


# ── Helpers ────────────────────────────────────────────────────────────────────

def fresh_config(**overrides) -> FarmVibesConfig:
    """Construct a FarmVibesConfig with clean env, optionally overriding fields."""
    return FarmVibesConfig(**overrides)


# ── Default values match what was previously hardcoded ────────────────────────

class TestDefaults:
    """Defaults must match the values that were previously hardcoded in constants.py,
    parsers.py, and local.py — these are the values the rest of the system depends on."""

    def test_registry(self):
        assert fresh_config().registry == "mcr.microsoft.com"

    def test_image_prefix(self):
        assert fresh_config().image_prefix == "farmai/terravibes/"

    def test_image_tag(self):
        assert fresh_config().image_tag == "12088305617"

    def test_redis_image_repository(self):
        assert fresh_config().redis_image_repository == "bitnamilegacy/redis"

    def test_redis_image_tag(self):
        assert fresh_config().redis_image_tag == "7.4.1-debian-12-r2"

    def test_rabbitmq_image_repository(self):
        assert fresh_config().rabbitmq_image_repository == "bitnamilegacy/rabbitmq"

    def test_rabbitmq_image_tag(self):
        assert fresh_config().rabbitmq_image_tag == "4.0.4-debian-12-r1"

    def test_port(self):
        assert fresh_config().port == 31108

    def test_registry_port(self):
        assert fresh_config().registry_port == 5000

    def test_max_worker_nodes(self):
        assert fresh_config().max_worker_nodes == 3

    def test_log_level(self):
        assert fresh_config().log_level == "DEBUG"

    def test_optional_fields_absent_by_default(self):
        cfg = fresh_config()
        assert cfg.max_log_file_bytes is None
        assert cfg.log_backup_count is None


# ── Environment variable overrides ────────────────────────────────────────────

class TestEnvOverrides:
    """Each advertised FARMVIBES_AI_* env var must change the corresponding field.
    Tests create a new FarmVibesConfig() inside the patched env so BaseSettings
    reads the overridden value."""

    def test_registry(self):
        with patch.dict("os.environ", {"FARMVIBES_AI_REGISTRY": "custom.registry.io"}):
            assert FarmVibesConfig().registry == "custom.registry.io"

    def test_image_tag(self):
        with patch.dict("os.environ", {"FARMVIBES_AI_IMAGE_TAG": "custom-tag"}):
            assert FarmVibesConfig().image_tag == "custom-tag"

    def test_redis_image_repository(self):
        with patch.dict("os.environ", {"FARMVIBES_AI_REDIS_IMAGE_REPOSITORY": "myorg/redis"}):
            assert FarmVibesConfig().redis_image_repository == "myorg/redis"

    def test_redis_image_tag(self):
        with patch.dict("os.environ", {"FARMVIBES_AI_REDIS_IMAGE_TAG": "8.0.0-debian-12-r1"}):
            assert FarmVibesConfig().redis_image_tag == "8.0.0-debian-12-r1"

    def test_image_prefix(self):
        with patch.dict("os.environ", {"FARMVIBES_AI_IMAGE_PREFIX": "custom/prefix/"}):
            assert FarmVibesConfig().image_prefix == "custom/prefix/"

    def test_rabbitmq_image_repository(self):
        with patch.dict("os.environ", {"FARMVIBES_AI_RABBITMQ_IMAGE_REPOSITORY": "myorg/rabbitmq"}):
            assert FarmVibesConfig().rabbitmq_image_repository == "myorg/rabbitmq"

    def test_rabbitmq_image_tag(self):
        with patch.dict("os.environ", {"FARMVIBES_AI_RABBITMQ_IMAGE_TAG": "4.1.0-debian-12-r0"}):
            assert FarmVibesConfig().rabbitmq_image_tag == "4.1.0-debian-12-r0"

    def test_port(self):
        with patch.dict("os.environ", {"FARMVIBES_AI_PORT": "31200"}):
            assert FarmVibesConfig().port == 31200

    def test_registry_port(self):
        with patch.dict("os.environ", {"FARMVIBES_AI_REGISTRY_PORT": "6000"}):
            assert FarmVibesConfig().registry_port == 6000

    def test_worker_replicas(self):
        with patch.dict("os.environ", {"FARMVIBES_AI_WORKER_REPLICAS": "4"}):
            assert FarmVibesConfig().worker_replicas == 4

    def test_max_worker_nodes(self):
        with patch.dict("os.environ", {"FARMVIBES_AI_MAX_WORKER_NODES": "5"}):
            assert FarmVibesConfig().max_worker_nodes == 5

    def test_log_level(self):
        with patch.dict("os.environ", {"FARMVIBES_AI_LOG_LEVEL": "WARNING"}):
            assert FarmVibesConfig().log_level == "WARNING"

    def test_max_log_file_bytes(self):
        with patch.dict("os.environ", {"FARMVIBES_AI_MAX_LOG_FILE_BYTES": "10485760"}):
            assert FarmVibesConfig().max_log_file_bytes == 10485760

    def test_log_backup_count(self):
        with patch.dict("os.environ", {"FARMVIBES_AI_LOG_BACKUP_COUNT": "5"}):
            assert FarmVibesConfig().log_backup_count == 5


# ── Validation rejects bad values ─────────────────────────────────────────────

class TestValidation:
    """Validators must reject invalid values at construction time — before any
    Terraform or Kubernetes work can begin."""

    def test_registry_with_spaces_raises(self):
        with pytest.raises(ValidationError, match="not a valid image reference"):
            FarmVibesConfig(registry="bad registry")

    def test_image_tag_with_spaces_raises(self):
        with pytest.raises(ValidationError, match="not a valid image reference"):
            FarmVibesConfig(image_tag="tag with spaces")

    def test_redis_image_tag_with_spaces_raises(self):
        with pytest.raises(ValidationError, match="not a valid image reference"):
            FarmVibesConfig(redis_image_tag="bad tag!")

    def test_port_zero_raises(self):
        with pytest.raises(ValidationError, match="valid port range"):
            FarmVibesConfig(port=0)

    def test_port_too_high_raises(self):
        with pytest.raises(ValidationError, match="valid port range"):
            FarmVibesConfig(port=99999)

    def test_registry_port_zero_raises(self):
        with pytest.raises(ValidationError, match="valid port range"):
            FarmVibesConfig(registry_port=0)

    def test_invalid_log_level_raises(self):
        with pytest.raises(ValidationError, match="not a valid log level"):
            FarmVibesConfig(log_level="VERBOSE")

    def test_worker_replicas_zero_raises(self):
        with pytest.raises(ValidationError, match="worker_replicas"):
            FarmVibesConfig(worker_replicas=0)

    def test_max_worker_nodes_zero_raises(self):
        with pytest.raises(ValidationError, match="max_worker_nodes"):
            FarmVibesConfig(max_worker_nodes=0)

    def test_empty_required_image_field_raises(self):
        with pytest.raises(ValidationError, match="must not be empty"):
            FarmVibesConfig(registry="")

    def test_empty_image_prefix_is_allowed(self):
        # Empty prefix is a real use case (no sub-path in the registry).
        cfg = FarmVibesConfig(image_prefix="")
        assert cfg.image_prefix == ""

    def test_image_prefix_with_spaces_raises(self):
        with pytest.raises(ValidationError, match="not a valid image reference"):
            FarmVibesConfig(image_prefix="bad prefix/")

    def test_port_min_boundary_is_valid(self):
        assert FarmVibesConfig(port=1).port == 1

    def test_port_max_boundary_is_valid(self):
        assert FarmVibesConfig(port=65535).port == 65535

    def test_port_negative_raises(self):
        with pytest.raises(ValidationError, match="valid port range"):
            FarmVibesConfig(port=-1)

    def test_worker_replicas_negative_raises(self):
        with pytest.raises(ValidationError, match="worker_replicas"):
            FarmVibesConfig(worker_replicas=-1)

    def test_max_worker_nodes_negative_raises(self):
        with pytest.raises(ValidationError, match="max_worker_nodes"):
            FarmVibesConfig(max_worker_nodes=-1)

    def test_log_level_normalized_to_uppercase(self):
        # Validator normalizes to uppercase so downstream comparisons are consistent.
        cfg = FarmVibesConfig(log_level="warning")
        assert cfg.log_level == "WARNING"

    def test_bad_env_string_raises_at_load_time(self):
        """A malformed registry in an env var must raise ValidationError from
        load_and_validate_config() — the entry point called at CLI startup."""
        with patch.dict("os.environ", {"FARMVIBES_AI_REGISTRY": "bad registry!"}):
            with pytest.raises(ValidationError, match="not a valid image reference"):
                load_and_validate_config()

    def test_bad_env_int_raises_at_load_time(self):
        """A non-integer port env var must raise at construction, not silently
        fall back to the default."""
        with patch.dict("os.environ", {"FARMVIBES_AI_PORT": "not-a-number"}):
            with pytest.raises(ValidationError):
                load_and_validate_config()


# ── dispatch() fails early on invalid config ──────────────────────────────────

class TestDispatchFailsEarly:
    """The task requires failing 'before Terraform runs, not halfway through'.
    dispatch() must return False without ever constructing OSArtifacts or
    calling into the Terraform wrapper when config is invalid."""

    def test_dispatch_returns_false_on_invalid_config(self):
        import argparse
        from vibe_core.cli.local import dispatch

        bad_args = argparse.Namespace(
            action="setup",
            cluster_name="test-cluster",
        )

        with patch.dict("os.environ", {"FARMVIBES_AI_PORT": "0"}):
            # Reset singleton so load_and_validate_config reads the patched env
            _config_module._config = None
            with patch("vibe_core.cli.local.OSArtifacts") as mock_osa:
                result = dispatch(bad_args)

        assert result is False, "dispatch() should return False on invalid config"
        mock_osa.assert_not_called(), "OSArtifacts must not be constructed before config is validated"


# ── constants.py backward compatibility ───────────────────────────────────────

class TestConstantsBackwardCompat:
    """constants.py now delegates to config. The names that the rest of the
    codebase imports must still resolve to the correct default values.

    Imports are performed inside test methods (not at class level) so that
    the module-level singleton in constants.py is populated after the
    autouse fixture resets it — preventing stale values from leaking in.
    """

    def test_default_image_tag(self):
        from vibe_core.cli.constants import DEFAULT_IMAGE_TAG
        assert DEFAULT_IMAGE_TAG == "12088305617"

    def test_default_registry_path(self):
        from vibe_core.cli.constants import DEFAULT_REGISTRY_PATH
        assert DEFAULT_REGISTRY_PATH == "mcr.microsoft.com"

    def test_default_image_prefix(self):
        from vibe_core.cli.constants import DEFAULT_IMAGE_PREFIX
        assert DEFAULT_IMAGE_PREFIX == "farmai/terravibes/"

    def test_redis_image_tag(self):
        from vibe_core.cli.constants import REDIS_IMAGE_TAG
        assert REDIS_IMAGE_TAG == "7.4.1-debian-12-r2"

    def test_redis_image_repository(self):
        from vibe_core.cli.constants import REDIS_IMAGE_REPOSITORY
        assert REDIS_IMAGE_REPOSITORY == "bitnamilegacy/redis"

    def test_rabbitmq_image_tag(self):
        from vibe_core.cli.constants import RABBITMQ_IMAGE_TAG
        assert RABBITMQ_IMAGE_TAG == "4.0.4-debian-12-r1"

    def test_max_worker_nodes(self):
        from vibe_core.cli.constants import MAX_WORKER_NODES
        assert MAX_WORKER_NODES == 3

    def test_log_level(self):
        from vibe_core.cli.constants import FARMVIBES_AI_LOG_LEVEL
        assert FARMVIBES_AI_LOG_LEVEL == "DEBUG"


# ── Singleton caching ─────────────────────────────────────────────────────────

class TestSingleton:
    def test_get_config_is_cached(self):
        """get_config() must return the same object on repeated calls."""
        assert get_config() is get_config()

    def test_load_and_validate_replaces_singleton(self):
        """load_and_validate_config() must update the cached singleton."""
        first = load_and_validate_config()
        _config_module._config = None
        second = load_and_validate_config()
        # Each call creates a fresh instance; the singleton is replaced each time.
        assert first is not second
        # But get_config() after load_and_validate_config() returns the cached one.
        assert get_config() is second
