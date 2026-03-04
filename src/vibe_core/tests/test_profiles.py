# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from multiprocessing import cpu_count
from pathlib import Path
from unittest.mock import patch

import pytest

from vibe_core.cli.profiles import ProfileValidationError, validate_profile
from vibe_core.cli.profiles import load_profile, PROFILE_SCHEMA
from vibe_core.cli.profiles import resolve_profile_args

# A minimal schema for testing — generic, not FarmVibes-specific
SCHEMA = {
    "count": {"type": int, "min": 1, "max": 10},
    "memory": {"type": str, "pattern": r"^[1-9]\d*Mi$"},
    "level": {"type": str, "choices": ["low", "high"]},
    "enabled": {"type": bool},
    "limit": {"type": int, "min": 0, "optional": True},
}


class TestValidateProfile:
    def test_valid_profile_passes(self):
        profile = {"count": 3, "memory": "512Mi", "level": "high", "enabled": True}
        result = validate_profile(profile, SCHEMA)
        assert result == profile

    def test_empty_profile_passes(self):
        result = validate_profile({}, SCHEMA)
        assert result == {}

    def test_unknown_key_raises(self):
        with pytest.raises(ProfileValidationError, match="Unknown profile key 'countz'"):
            validate_profile({"countz": 3}, SCHEMA)

    def test_unknown_key_lists_valid_keys(self):
        with pytest.raises(ProfileValidationError, match="count"):
            validate_profile({"bad_key": 1}, SCHEMA)

    def test_wrong_type_raises(self):
        with pytest.raises(ProfileValidationError, match="expects int"):
            validate_profile({"count": "three"}, SCHEMA)

    def test_bool_not_accepted_as_int(self):
        with pytest.raises(ProfileValidationError, match="expects int"):
            validate_profile({"count": True}, SCHEMA)

    def test_below_min_raises(self):
        with pytest.raises(ProfileValidationError, match="must be >= 1"):
            validate_profile({"count": 0}, SCHEMA)

    def test_above_max_raises(self):
        with pytest.raises(ProfileValidationError, match="must be <= 10"):
            validate_profile({"count": 11}, SCHEMA)

    def test_pattern_mismatch_raises(self):
        with pytest.raises(ProfileValidationError, match="invalid format"):
            validate_profile({"memory": "abc"}, SCHEMA)

    def test_pattern_match_passes(self):
        result = validate_profile({"memory": "256Mi"}, SCHEMA)
        assert result["memory"] == "256Mi"

    def test_invalid_choice_raises(self):
        with pytest.raises(ProfileValidationError, match="must be one of"):
            validate_profile({"level": "medium"}, SCHEMA)

    def test_valid_choice_passes(self):
        result = validate_profile({"level": "low"}, SCHEMA)
        assert result["level"] == "low"

    def test_bool_field_rejects_string(self):
        with pytest.raises(ProfileValidationError, match="expects bool"):
            validate_profile({"enabled": "yes"}, SCHEMA)

    def test_optional_field_accepts_value(self):
        result = validate_profile({"limit": 5}, SCHEMA)
        assert result["limit"] == 5

    def test_auto_sentinel_passes_for_int(self):
        result = validate_profile({"count": "auto"}, SCHEMA)
        assert result["count"] == "auto"


class TestLoadProfile:
    def test_loads_builtin_minimal(self):
        profile = load_profile("minimal")
        assert profile["worker_replicas"] == 1
        assert profile["worker_memory_request"] == "64Mi"

    def test_loads_builtin_default(self):
        profile = load_profile("default")
        assert profile["worker_replicas"] == "auto"

    def test_loads_builtin_production(self):
        profile = load_profile("production")
        assert profile["worker_memory_request"] == "512Mi"

    def test_missing_profile_raises(self):
        with pytest.raises(FileNotFoundError, match="No profile found"):
            load_profile("nonexistent")

    def test_user_dir_takes_precedence(self, tmp_path):
        user_profiles = tmp_path / "profiles"
        user_profiles.mkdir()
        custom = user_profiles / "minimal.yaml"
        custom.write_text("worker_replicas: 42\n")

        profile = load_profile("minimal", user_profile_dir=user_profiles)
        assert profile["worker_replicas"] == 42

    def test_custom_profile_from_user_dir(self, tmp_path):
        user_profiles = tmp_path / "profiles"
        user_profiles.mkdir()
        custom = user_profiles / "staging.yaml"
        custom.write_text("worker_replicas: 2\nworker_memory_request: 256Mi\n")

        profile = load_profile("staging", user_profile_dir=user_profiles)
        assert profile["worker_replicas"] == 2
        assert profile["worker_memory_request"] == "256Mi"

    def test_invalid_yaml_raises(self, tmp_path):
        user_profiles = tmp_path / "profiles"
        user_profiles.mkdir()
        bad = user_profiles / "bad.yaml"
        bad.write_text("{{not: valid: yaml::")

        with pytest.raises(ProfileValidationError, match="Invalid YAML"):
            load_profile("bad", user_profile_dir=user_profiles)

    def test_profile_with_unknown_key_raises(self, tmp_path):
        user_profiles = tmp_path / "profiles"
        user_profiles.mkdir()
        bad = user_profiles / "typo.yaml"
        bad.write_text("workerz_count: 3\n")

        with pytest.raises(ProfileValidationError, match="Unknown profile key 'workerz_count'"):
            load_profile("typo", user_profile_dir=user_profiles)


class TestResolveProfileArgs:
    """Profile values override defaults; CLI args override profiles."""

    def test_profile_overrides_defaults(self):
        defaults = {"worker_replicas": 4, "log_level": "DEBUG"}
        profile = {"worker_replicas": 1, "log_level": "INFO"}
        cli_explicit = {}

        result = resolve_profile_args(defaults, profile, cli_explicit)
        assert result["worker_replicas"] == 1
        assert result["log_level"] == "INFO"

    def test_cli_overrides_profile(self):
        defaults = {"worker_replicas": 4}
        profile = {"worker_replicas": 1}
        cli_explicit = {"worker_replicas": 8}

        result = resolve_profile_args(defaults, profile, cli_explicit)
        assert result["worker_replicas"] == 8

    def test_default_used_when_no_profile_or_cli(self):
        defaults = {"worker_replicas": 4, "log_level": "DEBUG"}
        profile = {}
        cli_explicit = {}

        result = resolve_profile_args(defaults, profile, cli_explicit)
        assert result["worker_replicas"] == 4
        assert result["log_level"] == "DEBUG"

    def test_auto_sentinel_resolves_to_cpu_calculation(self):
        defaults = {"worker_replicas": 4}
        profile = {"worker_replicas": "auto"}
        cli_explicit = {}

        result = resolve_profile_args(defaults, profile, cli_explicit)
        expected = max(1, cpu_count() // 2 - 1)
        assert result["worker_replicas"] == expected

    def test_cli_overrides_auto_sentinel(self):
        defaults = {"worker_replicas": 4}
        profile = {"worker_replicas": "auto"}
        cli_explicit = {"worker_replicas": 2}

        result = resolve_profile_args(defaults, profile, cli_explicit)
        assert result["worker_replicas"] == 2

    def test_mixed_sources(self):
        defaults = {"worker_replicas": 4, "log_level": "DEBUG", "servers": 1}
        profile = {"worker_replicas": 1}
        cli_explicit = {"log_level": "ERROR"}

        result = resolve_profile_args(defaults, profile, cli_explicit)
        assert result["worker_replicas"] == 1  # from profile
        assert result["log_level"] == "ERROR"  # from CLI
        assert result["servers"] == 1          # from default
