# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import re
from typing import Any, Dict, Optional


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
