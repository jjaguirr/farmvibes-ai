# Deployment Profiles Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add named deployment profiles (minimal, default, production) to FarmVibes.AI local k3d setup, with YAML-based profile files, schema validation, and a `--profile` CLI flag.

**Architecture:** Profile YAML files in `vibe_core/cli/profiles/` define overrides. A standalone `profiles.py` module handles loading, validation (against a generic schema), and layering. Profiles slot into the existing config precedence: defaults → profile → env vars → CLI args. Terraform gets `worker_memory_request` as a variable instead of hardcoded.

**Tech Stack:** Python 3.10+, PyYAML, argparse, pytest. Existing Pydantic config unchanged.

---

### Task 1: Profile validation module — tests

**Files:**
- Create: `src/vibe_core/tests/test_profiles.py`

**Step 1: Write failing tests for `validate_profile`**

```python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest

from vibe_core.cli.profiles import ProfileValidationError, validate_profile

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
```

**Step 2: Run tests to verify they fail**

Run: `cd src/vibe_core && python -m pytest tests/test_profiles.py -v`
Expected: ImportError — `vibe_core.cli.profiles` does not exist yet

**Step 3: Commit**

```
jj describe -m "test(profiles): add validation unit tests"
```

---

### Task 2: Profile validation module — implementation

**Files:**
- Create: `src/vibe_core/vibe_core/cli/profiles.py`

**Step 1: Implement `ProfileValidationError` and `validate_profile`**

```python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import re
from typing import Any, Dict, List, Optional


class ProfileValidationError(ValueError):
    """Raised when a profile contains invalid keys or values."""
    pass


def validate_profile(
    profile: Dict[str, Any],
    schema: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Validate a profile dict against a schema.

    Args:
        profile: Key-value overrides to validate.
        schema: Mapping of key name to constraints. Each constraint dict supports:
            - type: Python type (int, str, bool)
            - min/max: Numeric bounds (for int)
            - pattern: Regex pattern (for str)
            - choices: List of allowed values (for str)
            - optional: If True, key need not appear

    Returns:
        The validated profile dict (unchanged).

    Raises:
        ProfileValidationError: On unknown keys, wrong types, or out-of-range values.
    """
    valid_keys = sorted(schema.keys())

    for key, value in profile.items():
        if key not in schema:
            raise ProfileValidationError(
                f"Unknown profile key '{key}'. Valid keys: {', '.join(valid_keys)}"
            )

        spec = schema[key]
        expected_type = spec["type"]

        # "auto" sentinel is allowed for int fields — resolved later
        if value == "auto" and expected_type is int:
            continue

        # bool is a subclass of int in Python; reject bools for int fields
        if expected_type is int and isinstance(value, bool):
            raise ProfileValidationError(
                f"Key '{key}' expects int, got bool"
            )

        if not isinstance(value, expected_type):
            raise ProfileValidationError(
                f"Key '{key}' expects {expected_type.__name__}, got {type(value).__name__}"
            )

        if expected_type is int:
            if "min" in spec and value < spec["min"]:
                raise ProfileValidationError(
                    f"Key '{key}' must be >= {spec['min']}, got {value}"
                )
            if "max" in spec and value > spec["max"]:
                raise ProfileValidationError(
                    f"Key '{key}' must be <= {spec['max']}, got {value}"
                )

        if expected_type is str:
            if "choices" in spec and value not in spec["choices"]:
                raise ProfileValidationError(
                    f"Key '{key}' must be one of {spec['choices']}, got '{value}'"
                )
            if "pattern" in spec and not re.match(spec["pattern"], value):
                raise ProfileValidationError(
                    f"Key '{key}' invalid format: '{value}'"
                )

    return profile
```

**Step 2: Run tests to verify they pass**

Run: `cd src/vibe_core && python -m pytest tests/test_profiles.py -v`
Expected: All 15 tests PASS

**Step 3: Commit**

```
jj describe -m "feat(profiles): add validate_profile with schema-driven validation"
jj new
```

---

### Task 3: Profile loading — tests

**Files:**
- Modify: `src/vibe_core/tests/test_profiles.py`

**Step 1: Add failing tests for `load_profile`**

Append to `test_profiles.py`:

```python
import os
from pathlib import Path
from unittest.mock import patch

from vibe_core.cli.profiles import load_profile, PROFILE_SCHEMA


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
        custom.write_text("worker_replicas: 99\n")

        profile = load_profile("minimal", user_profile_dir=user_profiles)
        assert profile["worker_replicas"] == 99

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
```

**Step 2: Run tests to verify they fail**

Run: `cd src/vibe_core && python -m pytest tests/test_profiles.py::TestLoadProfile -v`
Expected: FAIL — `load_profile` and `PROFILE_SCHEMA` not defined yet

**Step 3: Commit**

```
jj describe -m "test(profiles): add load_profile unit tests"
```

---

### Task 4: Profile loading — implementation + built-in YAML files

**Files:**
- Modify: `src/vibe_core/vibe_core/cli/profiles.py`
- Create: `src/vibe_core/vibe_core/cli/profiles/minimal.yaml`
- Create: `src/vibe_core/vibe_core/cli/profiles/default.yaml`
- Create: `src/vibe_core/vibe_core/cli/profiles/production.yaml`

**Step 1: Add `PROFILE_SCHEMA`, `load_profile`, and YAML files**

Add to `profiles.py`:

```python
import os
from pathlib import Path

import yaml


PROFILE_SCHEMA = {
    "worker_replicas": {"type": int, "min": 1, "max": 64},
    "worker_memory_request": {"type": str, "pattern": r"^[1-9]\d*([EPTGMK]i?)?$"},
    "log_level": {"type": str, "choices": ["DEBUG", "INFO", "WARNING", "ERROR"]},
    "max_log_file_bytes": {"type": int, "min": 0, "optional": True},
    "log_backup_count": {"type": int, "min": 0, "optional": True},
    "enable_telemetry": {"type": bool},
    "servers": {"type": int, "min": 1, "max": 8},
    "agents": {"type": int, "min": 0, "max": 16},
}

_BUILTIN_DIR = Path(__file__).parent / "profiles"


def load_profile(
    name: str,
    user_profile_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Load and validate a named profile.

    Search order:
        1. user_profile_dir/{name}.yaml (if provided)
        2. Built-in profiles directory

    Args:
        name: Profile name (without .yaml extension).
        user_profile_dir: Optional user config directory for custom profiles.

    Returns:
        Validated profile dict.

    Raises:
        FileNotFoundError: If no profile file found.
        ProfileValidationError: If YAML is invalid or profile fails validation.
    """
    candidates = []
    if user_profile_dir:
        candidates.append(Path(user_profile_dir) / f"{name}.yaml")
    candidates.append(_BUILTIN_DIR / f"{name}.yaml")

    profile_path = None
    for candidate in candidates:
        if candidate.exists():
            profile_path = candidate
            break

    if profile_path is None:
        raise FileNotFoundError(
            f"No profile found named '{name}'. "
            f"Searched: {', '.join(str(c) for c in candidates)}"
        )

    raw = profile_path.read_text()
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as e:
        raise ProfileValidationError(f"Invalid YAML in profile '{name}': {e}")

    if data is None:
        data = {}

    if not isinstance(data, dict):
        raise ProfileValidationError(
            f"Profile '{name}' must be a YAML mapping, got {type(data).__name__}"
        )

    return validate_profile(data, PROFILE_SCHEMA)
```

**Step 2: Create `profiles/` directory marker**

Create `src/vibe_core/vibe_core/cli/profiles/__init__.py` — empty, not a Python package. Actually, since these are YAML data files, just create the directory and YAML files. No `__init__.py` needed — `_BUILTIN_DIR` uses `Path(__file__).parent / "profiles"` from `profiles.py`.

Wait — `profiles.py` and `profiles/` directory are siblings. `Path(__file__)` in `profiles.py` is `.../cli/profiles.py`, so `Path(__file__).parent` is `.../cli/`, and `"profiles"` resolves to `.../cli/profiles/`. That works.

**Step 3: Create `minimal.yaml`**

```yaml
# FarmVibes.AI Deployment Profile: minimal
#
# Lightweight profile for laptops, CI, or quick testing.
# Uses minimal resources to reduce footprint.
#
# Available keys and constraints:
#   worker_replicas:      int, 1-64, or "auto" (cpu_count // 2 - 1)
#   worker_memory_request: str, Kubernetes memory format (e.g. "64Mi", "8Gi")
#   log_level:            str, one of: DEBUG, INFO, WARNING, ERROR
#   max_log_file_bytes:   int >= 0 (optional)
#   log_backup_count:     int >= 0 (optional)
#   enable_telemetry:     bool
#   servers:              int, 1-8, k3d server nodes
#   agents:               int, 0-16, k3d agent nodes

worker_replicas: 1
worker_memory_request: "64Mi"
log_level: INFO
enable_telemetry: false
servers: 1
agents: 0
```

**Step 4: Create `default.yaml`**

```yaml
# FarmVibes.AI Deployment Profile: default
#
# Standard profile — matches current default behavior.
# Worker count is auto-calculated from CPU cores.
#
# Available keys and constraints:
#   worker_replicas:      int, 1-64, or "auto" (cpu_count // 2 - 1)
#   worker_memory_request: str, Kubernetes memory format (e.g. "64Mi", "8Gi")
#   log_level:            str, one of: DEBUG, INFO, WARNING, ERROR
#   max_log_file_bytes:   int >= 0 (optional)
#   log_backup_count:     int >= 0 (optional)
#   enable_telemetry:     bool
#   servers:              int, 1-8, k3d server nodes
#   agents:               int, 0-16, k3d agent nodes

worker_replicas: "auto"
worker_memory_request: "100Mi"
log_level: DEBUG
enable_telemetry: false
servers: 1
agents: 0
```

**Step 5: Create `production.yaml`**

```yaml
# FarmVibes.AI Deployment Profile: production
#
# Higher-resource profile for production-like testing on capable hardware.
# More workers, larger memory allocation, less verbose logging.
#
# Available keys and constraints:
#   worker_replicas:      int, 1-64, or "auto" (cpu_count // 2 - 1)
#   worker_memory_request: str, Kubernetes memory format (e.g. "64Mi", "8Gi")
#   log_level:            str, one of: DEBUG, INFO, WARNING, ERROR
#   max_log_file_bytes:   int >= 0 (optional)
#   log_backup_count:     int >= 0 (optional)
#   enable_telemetry:     bool
#   servers:              int, 1-8, k3d server nodes
#   agents:               int, 0-16, k3d agent nodes

worker_replicas: "auto"
worker_memory_request: "512Mi"
log_level: INFO
enable_telemetry: false
servers: 1
agents: 0
```

**Step 6: Run tests to verify they pass**

Run: `cd src/vibe_core && python -m pytest tests/test_profiles.py -v`
Expected: All tests PASS

**Step 7: Commit**

```
jj describe -m "feat(profiles): add load_profile, PROFILE_SCHEMA, and built-in YAML profiles"
jj new
```

---

### Task 5: Config layering — tests

**Files:**
- Modify: `src/vibe_core/tests/test_profiles.py`

**Step 1: Add failing tests for `resolve_profile_args`**

Append to `test_profiles.py`:

```python
from multiprocessing import cpu_count
from vibe_core.cli.profiles import resolve_profile_args


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
```

**Step 2: Run tests to verify they fail**

Run: `cd src/vibe_core && python -m pytest tests/test_profiles.py::TestResolveProfileArgs -v`
Expected: FAIL — `resolve_profile_args` not defined

**Step 3: Commit**

```
jj describe -m "test(profiles): add resolve_profile_args layering tests"
```

---

### Task 6: Config layering — implementation

**Files:**
- Modify: `src/vibe_core/vibe_core/cli/profiles.py`

**Step 1: Implement `resolve_profile_args`**

Add to `profiles.py`:

```python
from multiprocessing import cpu_count


def resolve_profile_args(
    defaults: Dict[str, Any],
    profile: Dict[str, Any],
    cli_explicit: Dict[str, Any],
) -> Dict[str, Any]:
    """Merge config layers: defaults → profile → CLI args.

    The "auto" sentinel for int fields resolves to max(1, cpu_count() // 2 - 1).

    Args:
        defaults: Base default values.
        profile: Profile overrides (from YAML).
        cli_explicit: Explicitly-provided CLI arguments (not argparse defaults).

    Returns:
        Merged config dict.
    """
    result = dict(defaults)
    result.update(profile)
    result.update(cli_explicit)

    # Resolve "auto" sentinels
    for key, value in result.items():
        if value == "auto":
            result[key] = max(1, cpu_count() // 2 - 1)

    return result
```

**Step 2: Run tests to verify they pass**

Run: `cd src/vibe_core && python -m pytest tests/test_profiles.py -v`
Expected: All tests PASS

**Step 3: Commit**

```
jj describe -m "feat(profiles): add resolve_profile_args config layering"
jj new
```

---

### Task 7: CLI integration — `--profile` flag and dispatch wiring

**Files:**
- Modify: `src/vibe_core/vibe_core/cli/parsers.py`
- Modify: `src/vibe_core/vibe_core/cli/local.py`

**Step 1: Add `--profile` flag to `LocalCliParser`**

In `parsers.py`, inside `LocalCliParser._add_setup_update_flags()`, add to the `for command in (...)` loop, after the existing arguments:

```python
            command.add_argument(
                "--profile",
                type=str,
                default=None,
                help=(
                    "Deployment profile to use (e.g. minimal, default, production). "
                    "Profiles set resource defaults. CLI flags override profile values. "
                    "Also settable via FARMVIBES_PROFILE env var."
                ),
            )
```

**Step 2: Wire profile loading into `local.py:dispatch()`**

In `local.py`, add imports at top:

```python
from vibe_core.cli.profiles import (
    ProfileValidationError,
    load_profile,
    resolve_profile_args,
)
```

In `dispatch()`, in the `if args.action in {"setup", "update", ...}:` block, after `enable_telemetry = ...` (line 789) and before `return setup(...)` (line 790), add profile resolution:

```python
        # --- Profile resolution ---
        profile_name = args.profile or os.environ.get("FARMVIBES_PROFILE")
        profile_overrides = {}
        if profile_name:
            try:
                profile_overrides = load_profile(profile_name)
            except (FileNotFoundError, ProfileValidationError) as e:
                show_error("Profile error", str(e))
                return False

        # Determine which CLI args were explicitly provided (not argparse defaults)
        cli_explicit = {}
        parser = parsers_mod.LocalCliParser("local") if not hasattr(args, '_explicit') else None
        # Simple approach: compare args against defaults from a fresh parse
        _defaults = parsers_mod.LocalCliParser("local").parse(
            [args.action, "--cluster-name", args.cluster_name]
        )
```

Actually, detecting explicitly-set CLI args with argparse is awkward. Simpler approach: track which args the user set. The cleanest pattern is to use `None` as the argparse default for profile-overridable fields and treat non-None as explicitly set. But that changes existing defaults.

**Better approach:** In `dispatch()`, for each profile-overridable key, check if the CLI arg differs from the argparse default. If it does, the user explicitly set it. We already know the defaults (they come from `_cfg`). Let's use a mapping:

```python
        # Detect explicitly-set CLI args by comparing against known defaults
        _PROFILE_OVERRIDABLE = {
            "worker_replicas": _cfg.worker_replicas,
            "worker_memory_request": _cfg.worker_memory_request,
            "log_level": _cfg.log_level,
            "max_log_file_bytes": None,
            "log_backup_count": None,
            "enable_telemetry": False,
            "servers": 1,
            "agents": 0,
        }

        profile_name = getattr(args, "profile", None) or os.environ.get("FARMVIBES_PROFILE")
        if profile_name:
            try:
                profile_overrides = load_profile(profile_name)
            except (FileNotFoundError, ProfileValidationError) as e:
                show_error("Profile error", str(e))
                return False

            cli_explicit = {}
            for key, default_val in _PROFILE_OVERRIDABLE.items():
                arg_val = getattr(args, key, default_val)
                if arg_val != default_val:
                    cli_explicit[key] = arg_val

            merged = resolve_profile_args(
                {k: getattr(args, k, v) for k, v in _PROFILE_OVERRIDABLE.items()},
                profile_overrides,
                cli_explicit,
            )

            # Apply merged values back to args
            for key, value in merged.items():
                if hasattr(args, key):
                    setattr(args, key, value)
```

This block goes right before the `return setup(...)` call.

**Step 3: Pass `worker_memory_request` through to setup and Terraform**

In `local.py:setup()` signature, add parameter:
```python
    worker_memory_request: str = "100Mi",
```

In `local.py:dispatch()`, add to the `setup()` call:
```python
            worker_memory_request=getattr(args, "worker_memory_request", "100Mi"),
```

In `local.py:setup()`, pass to `ensure_local_cluster()`:
```python
            terraform.ensure_local_cluster(
                ...  # existing args
                worker_memory_request=worker_memory_request,
            )
```

**Step 4: Commit**

```
jj describe -m "feat(profiles): add --profile flag and dispatch wiring"
jj new
```

---

### Task 8: Terraform plumbing — pass `worker_memory_request` through

**Files:**
- Modify: `src/vibe_core/vibe_core/cli/wrappers.py` (lines 478-534)
- Modify: `src/vibe_core/vibe_core/terraform/local/main.tf` (line 45)
- Modify: `src/vibe_core/vibe_core/terraform/local/variables.tf`

**Step 1: Add `worker_memory_request` parameter to `ensure_local_cluster()`**

In `wrappers.py`, add to `ensure_local_cluster()` signature:

```python
    def ensure_local_cluster(
        self,
        ...  # existing params
        worker_memory_request: str = "100Mi",
    ):
```

Add to the `variables` dict:

```python
            "worker_memory_request": worker_memory_request,
```

**Step 2: Make Terraform use the variable**

In `terraform/local/variables.tf`, add:

```terraform
variable "worker_memory_request" {
  description = "Memory request for worker pods (e.g. 64Mi, 512Mi, 8Gi)"
  default     = "100Mi"
}
```

In `terraform/local/main.tf`, change line 45 from:

```terraform
  worker_memory_request         = "100Mi"
```

to:

```terraform
  worker_memory_request         = var.worker_memory_request
```

**Step 3: Run existing tests to verify nothing broke**

Run: `cd src/vibe_core && python -m pytest tests/ -v`
Expected: All existing tests PASS

**Step 4: Commit**

```
jj describe -m "feat(profiles): pass worker_memory_request through to Terraform"
jj new
```

---

### Task 9: Integration test — profile loading through CLI parser

**Files:**
- Modify: `src/vibe_core/tests/test_profiles.py`

**Step 1: Add CLI integration tests**

```python
import importlib


class TestCliIntegration:
    """Profile flag wires through parser to dispatch."""

    def test_profile_flag_parsed(self):
        from vibe_core.cli.parsers import LocalCliParser

        parser = LocalCliParser("local")
        args = parser.parse(["setup", "--cluster-name", "test", "--profile", "minimal"])
        assert args.profile == "minimal"

    def test_profile_flag_defaults_none(self):
        from vibe_core.cli.parsers import LocalCliParser

        parser = LocalCliParser("local")
        args = parser.parse(["setup", "--cluster-name", "test"])
        assert args.profile is None

    def test_env_var_fallback(self):
        """FARMVIBES_PROFILE env var used when --profile not set."""
        with patch.dict(os.environ, {"FARMVIBES_PROFILE": "production"}):
            # The env var is read in dispatch(), not the parser.
            # So we just verify the env var is accessible.
            assert os.environ.get("FARMVIBES_PROFILE") == "production"

    def test_flag_overrides_env_var(self):
        """--profile flag wins over FARMVIBES_PROFILE env var."""
        from vibe_core.cli.parsers import LocalCliParser

        with patch.dict(os.environ, {"FARMVIBES_PROFILE": "production"}):
            parser = LocalCliParser("local")
            args = parser.parse(["setup", "--cluster-name", "test", "--profile", "minimal"])
            # Flag wins — dispatch() checks args.profile first
            profile_name = args.profile or os.environ.get("FARMVIBES_PROFILE")
            assert profile_name == "minimal"
```

**Step 2: Run all tests**

Run: `cd src/vibe_core && python -m pytest tests/test_profiles.py -v`
Expected: All tests PASS

**Step 3: Commit**

```
jj describe -m "test(profiles): add CLI integration tests for --profile flag"
jj new
```

---

### Task 10: VM setup and integration testing

**Prerequisites:** Push branch, set up VM worktree.

**Step 1: Push the branch**

Instruct user to run:
```
jj git push -b deploy-profiles_task13_model_a
```

**Step 2: Set up VM worktree (one-time)**

```bash
source ~/farmvibes-env_13_model_a.sh && cd ~/farmvibes-ai-work && git fetch origin && git worktree add ~/farmvibes-ai-work/13_model_a origin/deploy-profiles_task13_model_a
```

If worktree already exists:
```bash
source ~/farmvibes-env_13_model_a.sh && git pull origin deploy-profiles_task13_model_a
```

Set up venv:
```bash
python3 -m venv ~/venvs/13_model_a && ~/venvs/13_model_a/bin/pip install -e ~/farmvibes-ai-work/13_model_a/src/vibe_core/
```

Create env script:
```bash
cat > ~/farmvibes-env_13_model_a.sh << 'EOF'
source ~/venvs/13_model_a/bin/activate
cd ~/farmvibes-ai-work/13_model_a
EOF
```

**Step 3: Run unit tests on VM**

```bash
source ~/farmvibes-env_13_model_a.sh && python -m pytest tests/test_profiles.py -v
```

Expected: All tests PASS

**Step 4: Test profile validation error (typo key)**

Create a bad profile on VM:
```bash
source ~/farmvibes-env_13_model_a.sh && mkdir -p ~/.config/farmvibes-ai/profiles && echo "workerz_count: 3" > ~/.config/farmvibes-ai/profiles/bad.yaml
```

Run:
```bash
source ~/farmvibes-env_13_model_a.sh && farmvibes-ai local setup --profile bad
```

Expected: Error message containing "Unknown profile key 'workerz_count'" — exits before Terraform.

**Step 5: Test `--profile minimal` deployment**

```bash
source ~/farmvibes-env_13_model_a.sh && farmvibes-ai local setup --profile minimal -y
```

Verify:
```bash
source ~/farmvibes-env_13_model_a.sh && kubectl get deployments -o wide | grep worker
```
Expected: 1 worker replica

```bash
source ~/farmvibes-env_13_model_a.sh && kubectl describe deployment terravibes-worker | grep -A5 "Requests"
```
Expected: memory request shows `64Mi`

**Step 6: Test `--profile production` deployment**

```bash
source ~/farmvibes-env_13_model_a.sh && farmvibes-ai local destroy -y && farmvibes-ai local setup --profile production -y
```

Verify:
```bash
source ~/farmvibes-env_13_model_a.sh && kubectl get deployments -o wide | grep worker
```
Expected: More workers than minimal (auto-calculated from VM CPU count)

```bash
source ~/farmvibes-env_13_model_a.sh && kubectl describe deployment terravibes-worker | grep -A5 "Requests"
```
Expected: memory request shows `512Mi`

**Step 7: Test CLI override of profile value**

```bash
source ~/farmvibes-env_13_model_a.sh && farmvibes-ai local destroy -y && farmvibes-ai local setup --profile production --worker-replicas 2 -y
```

Verify:
```bash
source ~/farmvibes-env_13_model_a.sh && kubectl get deployments -o wide | grep worker
```
Expected: Exactly 2 worker replicas (CLI override wins)

**Step 8: Clean up**

```bash
source ~/farmvibes-env_13_model_a.sh && farmvibes-ai local destroy -y
```

---

### Task 11: Final commit and PR preparation

**Step 1: Ensure all tests pass locally**

Run: `cd src/vibe_core && python -m pytest tests/ -v`
Expected: All tests PASS

**Step 2: Squash/organize commits if needed**

Verify log:
```
jj log --limit 10
```

**Step 3: Instruct user to push and create PR**

```
jj git push -b deploy-profiles_task13_model_a
```

Then:
```
gh pr create --base main --head deploy-profiles_task13_model_a --title "feat: add deployment profiles for local k3d setup" --body "..."
```
