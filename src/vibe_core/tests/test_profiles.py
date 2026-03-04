# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for deployment profile loading, validation, and layering."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest


# =============================================================================
# validate_profile_keys — standalone validator, decoupled from FarmVibes keys
# =============================================================================

class TestValidateProfileKeys:
    def test_all_known_keys_pass(self):
        from vibe_core.cli.profiles import validate_profile_keys
        validate_profile_keys(
            profile={"worker_replicas": 3, "log_level": "INFO"},
            schema={"worker_replicas", "log_level", "port"},
            source="test.yaml",
        )  # should not raise

    def test_unknown_key_raises_with_key_name_in_message(self):
        from vibe_core.cli.profiles import ProfileError, validate_profile_keys
        with pytest.raises(ProfileError, match="workerz_count"):
            validate_profile_keys(
                profile={"workerz_count": 3},
                schema={"worker_replicas"},
                source="custom.yaml",
            )

    def test_unknown_key_error_includes_source_path(self):
        from vibe_core.cli.profiles import ProfileError, validate_profile_keys
        with pytest.raises(ProfileError, match="custom.yaml"):
            validate_profile_keys(
                profile={"bad_key": 1},
                schema={"good_key"},
                source="custom.yaml",
            )

    def test_multiple_unknown_keys_all_reported(self):
        from vibe_core.cli.profiles import ProfileError, validate_profile_keys
        with pytest.raises(ProfileError) as exc_info:
            validate_profile_keys(
                profile={"typo_one": 1, "typo_two": 2, "valid": 3},
                schema={"valid"},
                source="x.yaml",
            )
        msg = str(exc_info.value)
        assert "typo_one" in msg
        assert "typo_two" in msg

    def test_empty_profile_is_valid(self):
        from vibe_core.cli.profiles import validate_profile_keys
        validate_profile_keys(profile={}, schema={"anything"}, source="empty.yaml")

    def test_schema_is_not_farmvibes_coupled(self):
        """Validator is generic: schema is just a set of strings, works with ANY schema."""
        from vibe_core.cli.profiles import validate_profile_keys
        validate_profile_keys(
            profile={"foo": 1, "bar": 2},
            schema={"foo", "bar", "baz"},  # arbitrary schema, no FarmVibes keys
            source="generic.yaml",
        )


# =============================================================================
# resolve_profile_name — pre-scan argv for --profile, fall back to env, then default
# =============================================================================

class TestResolveProfileName:
    def test_flag_long_form(self):
        from vibe_core.cli.profiles import resolve_profile_name
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FARMVIBES_PROFILE", None)
            assert resolve_profile_name(["local", "setup", "--profile", "minimal"]) == "minimal"

    def test_flag_equals_form(self):
        from vibe_core.cli.profiles import resolve_profile_name
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FARMVIBES_PROFILE", None)
            assert resolve_profile_name(["setup", "--profile=production"]) == "production"

    def test_env_var_used_when_no_flag(self):
        from vibe_core.cli.profiles import resolve_profile_name
        with patch.dict(os.environ, {"FARMVIBES_PROFILE": "production"}):
            assert resolve_profile_name(["local", "setup"]) == "production"

    def test_flag_beats_env(self):
        """Spec: flag wins if both are set."""
        from vibe_core.cli.profiles import resolve_profile_name
        with patch.dict(os.environ, {"FARMVIBES_PROFILE": "production"}):
            assert resolve_profile_name(["--profile", "minimal"]) == "minimal"

    def test_defaults_to_default_when_nothing_set(self):
        from vibe_core.cli.profiles import resolve_profile_name
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FARMVIBES_PROFILE", None)
            assert resolve_profile_name(["local", "setup"]) == "default"

    def test_flag_at_end_of_argv_with_no_value_ignored(self):
        """Don't crash on malformed argv — argparse will catch it later."""
        from vibe_core.cli.profiles import resolve_profile_name
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FARMVIBES_PROFILE", None)
            assert resolve_profile_name(["setup", "--profile"]) == "default"


# =============================================================================
# load_profile — file discovery and YAML parsing
# =============================================================================

class TestLoadProfile:
    def test_loads_yaml_from_user_dir(self, tmp_path):
        from vibe_core.cli.profiles import load_profile
        user_dir = tmp_path / "profiles"
        user_dir.mkdir()
        (user_dir / "custom.yaml").write_text("worker_replicas: 7\nlog_level: INFO\n")

        profile, source = load_profile("custom", search_dirs=[user_dir])
        assert profile == {"worker_replicas": 7, "log_level": "INFO"}
        assert "custom.yaml" in source

    def test_user_dir_shadows_builtin(self, tmp_path):
        """Earlier search dirs win."""
        from vibe_core.cli.profiles import load_profile
        user_dir = tmp_path / "user"
        builtin_dir = tmp_path / "builtin"
        user_dir.mkdir()
        builtin_dir.mkdir()
        (user_dir / "prod.yaml").write_text("worker_replicas: 99\n")
        (builtin_dir / "prod.yaml").write_text("worker_replicas: 1\n")

        profile, _ = load_profile("prod", search_dirs=[user_dir, builtin_dir])
        assert profile["worker_replicas"] == 99

    def test_not_found_raises_with_name_and_searched_dirs(self, tmp_path):
        from vibe_core.cli.profiles import ProfileError, load_profile
        with pytest.raises(ProfileError) as exc_info:
            load_profile("ghost", search_dirs=[tmp_path])
        assert "ghost" in str(exc_info.value)
        assert str(tmp_path) in str(exc_info.value)

    def test_malformed_yaml_raises_profile_error(self, tmp_path):
        from vibe_core.cli.profiles import ProfileError, load_profile
        (tmp_path / "broken.yaml").write_text("worker_replicas: [unclosed\n")
        with pytest.raises(ProfileError, match="broken.yaml"):
            load_profile("broken", search_dirs=[tmp_path])

    def test_empty_yaml_returns_empty_dict(self, tmp_path):
        """An empty profile means 'no overrides' — valid."""
        from vibe_core.cli.profiles import load_profile
        (tmp_path / "empty.yaml").write_text("# just comments\n")
        profile, _ = load_profile("empty", search_dirs=[tmp_path])
        assert profile == {}

    def test_profile_with_null_values_preserved(self, tmp_path):
        """null in YAML means 'unset this' — must pass through as None."""
        from vibe_core.cli.profiles import load_profile
        (tmp_path / "nulls.yaml").write_text("worker_memory_limit: null\n")
        profile, _ = load_profile("nulls", search_dirs=[tmp_path])
        assert profile == {"worker_memory_limit": None}

    def test_non_dict_toplevel_rejected(self, tmp_path):
        """A profile that's a YAML list or scalar is malformed."""
        from vibe_core.cli.profiles import ProfileError, load_profile
        (tmp_path / "list.yaml").write_text("- not\n- a\n- dict\n")
        with pytest.raises(ProfileError, match="must be a YAML mapping"):
            load_profile("list", search_dirs=[tmp_path])


# =============================================================================
# Built-in profiles — shipped YAMLs must be valid
# =============================================================================

class TestBuiltinProfiles:
    @pytest.mark.parametrize("name", ["minimal", "default", "production"])
    def test_builtin_profile_exists_and_loads(self, name):
        from vibe_core.cli.profiles import BUILTIN_PROFILES_DIR, load_profile
        profile, source = load_profile(name, search_dirs=[BUILTIN_PROFILES_DIR])
        assert isinstance(profile, dict)
        assert name in source

    @pytest.mark.parametrize("name", ["minimal", "default", "production"])
    def test_builtin_profile_has_no_unknown_keys(self, name):
        from vibe_core.cli.config import FarmVibesConfig
        from vibe_core.cli.profiles import (
            BUILTIN_PROFILES_DIR,
            load_profile,
            validate_profile_keys,
        )
        profile, source = load_profile(name, search_dirs=[BUILTIN_PROFILES_DIR])
        schema = set(FarmVibesConfig.__fields__.keys())
        validate_profile_keys(profile, schema, source)

    @pytest.mark.parametrize("name", ["minimal", "default", "production"])
    def test_builtin_profile_values_pass_pydantic_validators(self, name):
        """Every built-in profile must produce a valid FarmVibesConfig when applied."""
        from vibe_core.cli.config import FarmVibesConfig
        from vibe_core.cli.profiles import BUILTIN_PROFILES_DIR, load_profile
        profile, _ = load_profile(name, search_dirs=[BUILTIN_PROFILES_DIR])
        # Strip Nones — Pydantic v1 treats explicit None as a set value
        non_null = {k: v for k, v in profile.items() if v is not None}
        with patch.dict(os.environ, {}, clear=False):
            for k in list(os.environ):
                if k.startswith("FARMVIBES_"):
                    del os.environ[k]
            FarmVibesConfig(**non_null)  # raises ValidationError if bad


class TestDefaultProfileDrift:
    """default.yaml must mirror FarmVibesConfig defaults. Catches silent drift."""

    def test_default_yaml_values_match_pydantic_defaults(self):
        from vibe_core.cli.config import FarmVibesConfig
        from vibe_core.cli.profiles import BUILTIN_PROFILES_DIR, load_profile

        profile, _ = load_profile("default", search_dirs=[BUILTIN_PROFILES_DIR])

        with patch.dict(os.environ, {}, clear=False):
            for k in list(os.environ):
                if k.startswith("FARMVIBES_"):
                    del os.environ[k]
            cfg = FarmVibesConfig()

        mismatches = []
        for key, yaml_val in profile.items():
            py_val = getattr(cfg, key)
            # worker_replicas is machine-dependent (cpu_count); skip exact value check
            if key == "worker_replicas":
                continue
            if py_val != yaml_val:
                mismatches.append(f"  {key}: yaml={yaml_val!r} != python={py_val!r}")

        assert not mismatches, (
            "default.yaml has drifted from FarmVibesConfig defaults:\n"
            + "\n".join(mismatches)
        )


# =============================================================================
# Config layering — profile sits between defaults and env vars
# =============================================================================

class TestConfigLayering:
    """Precedence: defaults < profile < env < CLI args."""

    def test_profile_beats_default(self):
        from vibe_core.cli.config import load_config
        with patch.dict(os.environ, {}, clear=False):
            for k in list(os.environ):
                if k.startswith("FARMVIBES_"):
                    del os.environ[k]
            cfg = load_config(profile_overrides={"log_level": "WARNING"})
        assert cfg.log_level == "WARNING"  # default is DEBUG

    def test_env_beats_profile(self):
        from vibe_core.cli.config import load_config
        with patch.dict(os.environ, {"FARMVIBES_LOG_LEVEL": "ERROR"}):
            cfg = load_config(profile_overrides={"log_level": "WARNING"})
        assert cfg.log_level == "ERROR"

    def test_profile_values_validated_by_pydantic(self):
        """Profile values go through the same validators as defaults/env."""
        from pydantic import ValidationError
        from vibe_core.cli.config import load_config
        with pytest.raises(ValidationError, match="Invalid log level"):
            load_config(profile_overrides={"log_level": "TRACE"})

    def test_no_profile_overrides_is_same_as_before(self):
        """Backwards compat: load_config() with no profile == old behavior."""
        from vibe_core.cli.config import load_config
        with patch.dict(os.environ, {}, clear=False):
            for k in list(os.environ):
                if k.startswith("FARMVIBES_"):
                    del os.environ[k]
            cfg_none = load_config()
            cfg_empty = load_config(profile_overrides={})
        assert cfg_none.log_level == cfg_empty.log_level == "DEBUG"

    def test_profile_sets_new_resource_fields(self):
        from vibe_core.cli.config import load_config
        with patch.dict(os.environ, {}, clear=False):
            for k in list(os.environ):
                if k.startswith("FARMVIBES_"):
                    del os.environ[k]
            cfg = load_config(profile_overrides={
                "worker_memory_limit": "4Gi",
                "worker_cpu_request": "500m",
                "worker_cpu_limit": "2",
            })
        assert cfg.worker_memory_limit == "4Gi"
        assert cfg.worker_cpu_request == "500m"
        assert cfg.worker_cpu_limit == "2"

    def test_new_resource_fields_default_none(self):
        """No limit by default — matches current behavior."""
        from vibe_core.cli.config import load_config
        with patch.dict(os.environ, {}, clear=False):
            for k in list(os.environ):
                if k.startswith("FARMVIBES_"):
                    del os.environ[k]
            cfg = load_config()
        assert cfg.worker_memory_limit is None
        assert cfg.worker_cpu_request is None
        assert cfg.worker_cpu_limit is None


# =============================================================================
# End-to-end: profile -> argparse defaults
# =============================================================================

class TestProfileFlowsToArgparse:
    """Profile values must become argparse defaults, and CLI flags must still win."""

    def _apply_and_parse(self, profile_name, cli_argv):
        """Helper: apply profile, reload parser modules, parse argv."""
        import importlib
        import sys
        import vibe_core.cli.config as config_mod
        import vibe_core.cli.parsers as parsers_mod
        from vibe_core.cli.profiles import BUILTIN_PROFILES_DIR, load_profile

        overrides, _ = load_profile(profile_name, search_dirs=[BUILTIN_PROFILES_DIR])
        orig_load = config_mod.load_config
        try:
            config_mod.load_config = lambda: orig_load(profile_overrides=overrides)
            importlib.reload(parsers_mod)
            parser = parsers_mod.LocalCliParser("local")
            return parser.parse(cli_argv)
        finally:
            config_mod.load_config = orig_load
            importlib.reload(parsers_mod)

    def test_production_profile_sets_argparse_defaults(self):
        with patch.dict(os.environ, {}, clear=False):
            for k in list(os.environ):
                if k.startswith("FARMVIBES_"):
                    del os.environ[k]
            args = self._apply_and_parse(
                "production", ["setup", "--cluster-name", "t"]
            )
        assert args.worker_replicas == 8
        assert args.worker_memory_request == "2Gi"
        assert args.worker_memory_limit == "8Gi"
        assert args.worker_cpu_limit == "4"
        assert args.log_level == "INFO"

    def test_minimal_profile_sets_single_worker(self):
        with patch.dict(os.environ, {}, clear=False):
            for k in list(os.environ):
                if k.startswith("FARMVIBES_"):
                    del os.environ[k]
            args = self._apply_and_parse(
                "minimal", ["setup", "--cluster-name", "t"]
            )
        assert args.worker_replicas == 1
        assert args.worker_memory_limit is None

    def test_cli_flag_beats_profile_end_to_end(self):
        """Full chain: production profile says 8 replicas, CLI says 3, CLI wins."""
        with patch.dict(os.environ, {}, clear=False):
            for k in list(os.environ):
                if k.startswith("FARMVIBES_"):
                    del os.environ[k]
            args = self._apply_and_parse(
                "production",
                ["setup", "--cluster-name", "t", "--worker-replicas", "3"],
            )
        assert args.worker_replicas == 3

    def test_env_beats_profile_end_to_end(self):
        """production profile says 8, env says 5, env wins."""
        with patch.dict(os.environ, {"FARMVIBES_WORKER_REPLICAS": "5"}):
            args = self._apply_and_parse(
                "production", ["setup", "--cluster-name", "t"]
            )
        assert args.worker_replicas == 5


# =============================================================================
# New field validators
# =============================================================================

class TestCpuValidator:
    @pytest.mark.parametrize("v", ["500m", "1", "2", "0.5", "100m", "4000m"])
    def test_valid_k8s_cpu_accepted(self, v):
        from vibe_core.cli.config import FarmVibesConfig
        with patch.dict(os.environ, {"FARMVIBES_WORKER_CPU_LIMIT": v}):
            cfg = FarmVibesConfig()
            assert cfg.worker_cpu_limit == v

    @pytest.mark.parametrize("v", ["lots", "500x", "-1", "1.5m", ""])
    def test_invalid_k8s_cpu_rejected(self, v):
        from pydantic import ValidationError
        from vibe_core.cli.config import FarmVibesConfig
        with patch.dict(os.environ, {"FARMVIBES_WORKER_CPU_LIMIT": v}):
            with pytest.raises(ValidationError, match="Invalid Kubernetes CPU"):
                FarmVibesConfig()
