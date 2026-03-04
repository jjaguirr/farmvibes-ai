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
