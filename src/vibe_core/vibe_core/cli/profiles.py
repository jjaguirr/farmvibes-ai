# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Deployment profile loading and validation.

A profile is a named YAML file containing overrides for FarmVibesConfig fields.
Profiles sit between hardcoded defaults and environment variables in precedence:

    hardcoded default  <  profile  <  FARMVIBES_* env var  <  CLI flag

See cli/profiles/*.yaml for the shipped profiles.
"""

import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import yaml

BUILTIN_PROFILES_DIR = Path(__file__).parent / "profiles"
USER_PROFILES_DIR = Path.home() / ".config" / "farmvibes-ai" / "profiles"
DEFAULT_PROFILE_NAME = "default"


class ProfileError(Exception):
    """Raised when a profile cannot be found, parsed, or validated."""


def resolve_profile_name(argv: List[str]) -> str:
    """Determine the active profile name before argparse runs.

    Precedence: --profile flag > FARMVIBES_PROFILE env var > "default".

    We pre-scan argv because the profile must be resolved before the argparse
    parser is constructed (profile values feed argparse defaults). This is the
    same trick main.py uses for --verbose.
    """
    # Scan for --profile <name> or --profile=<name>
    for i, tok in enumerate(argv):
        if tok == "--profile":
            if i + 1 < len(argv):
                return argv[i + 1]
        elif tok.startswith("--profile="):
            return tok.split("=", 1)[1]

    env = os.environ.get("FARMVIBES_PROFILE")
    if env:
        return env

    return DEFAULT_PROFILE_NAME


def default_search_dirs() -> List[Path]:
    """Profile search path. User dir shadows built-in so custom profiles win."""
    return [USER_PROFILES_DIR, BUILTIN_PROFILES_DIR]


def load_profile(
    name: str,
    search_dirs: Optional[Iterable[Path]] = None,
) -> Tuple[Dict[str, Any], str]:
    """Load a profile YAML by name.

    Searches each directory in order; first match wins. Returns (profile_dict,
    source_path) so callers can report where a value came from.

    Raises ProfileError if the profile is not found, is malformed YAML, or is
    not a top-level mapping.
    """
    dirs = list(search_dirs) if search_dirs is not None else default_search_dirs()

    for d in dirs:
        path = Path(d) / f"{name}.yaml"
        if path.is_file():
            return _read_profile_file(path), str(path)

    searched = "\n  ".join(str(d) for d in dirs)
    raise ProfileError(
        f"Profile '{name}' not found. Searched:\n  {searched}\n"
        f"Create {name}.yaml in one of these directories, or use a built-in "
        f"profile: minimal, default, production."
    )


def _read_profile_file(path: Path) -> Dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ProfileError(f"Failed to parse profile {path}: {e}") from e

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ProfileError(
            f"Profile {path} must be a YAML mapping (key: value pairs), "
            f"got {type(data).__name__}."
        )
    return data


def validate_profile_keys(
    profile: Dict[str, Any],
    schema: Set[str],
    source: str,
) -> None:
    """Reject unknown keys in a profile dict.

    This is a standalone validator — it knows nothing about FarmVibes config.
    Callers pass in the set of valid keys (the schema). Adding a new profile key
    is a schema change at the call site, not a code change here.

    Raises ProfileError listing ALL unknown keys (not just the first).
    """
    unknown = set(profile) - schema
    if unknown:
        raise ProfileError(
            f"Unknown profile key(s) in {source}: {', '.join(sorted(unknown))}\n"
            f"Valid keys: {', '.join(sorted(schema))}"
        )
