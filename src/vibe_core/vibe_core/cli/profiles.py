# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import re
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


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


def resolve_profile_args(
    defaults: Dict[str, Any],
    profile: Dict[str, Any],
    cli_explicit: Dict[str, Any],
) -> Dict[str, Any]:
    """Merge config layers: defaults -> profile -> CLI args.

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
